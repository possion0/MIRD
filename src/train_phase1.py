import os
import sys
import transformers
print(f"DEBUG: transformers version: {transformers.__version__}")
print(f"DEBUG: transformers file: {transformers.__file__}")
import argparse
import random
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn import metrics
import wandb
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_set import MyDataset
from model_agentic import MIRDModel
from transformers import CLIPProcessor

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def set_args():
    parser = argparse.ArgumentParser(description="Phase I: Perception Pre-training")
    
    # Device settings
    parser.add_argument('--device', default='0', type=str, help='GPU device number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model settings
    parser.add_argument('--model', default='MIRDModel', type=str)
    parser.add_argument('--text_name', default='text_json_final', type=str)
    parser.add_argument('--simple_linear', default=False, type=bool)
    parser.add_argument('--text_size', default=512, type=int)
    parser.add_argument('--image_size', default=768, type=int)
    parser.add_argument('--label_number', default=2, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--max_len', default=77, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    
    # Training settings
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--dev_batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_proportion', default=0.2, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max_grad_norm', default=5.0, type=float)
    
    # Output settings
    parser.add_argument('--output_dir', default='../output_dir/phase1', type=str)
    parser.add_argument('--limit', default=None, type=int, help='Limit training samples')
    
    # Backbone settings
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--masks', action='store_true')
    
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_phase1(args, model, device, train_data, dev_data, test_data, processor):
    """Phase I training loop"""
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.train_batch_size,
        collate_fn=MyDataset.collate_func,
        shuffle=True
    )
    
    total_steps = int(len(train_loader) * args.num_train_epochs)
    
    # Freeze BERT and ResNet backbones
    logger.info("Freezing BERT and ResNet backbones...")
    for param in model.bert_model.parameters():
        param.requires_grad = False
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Setup optimizer with different learning rates
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    
    clip_params = list(map(id, model.model.parameters()))
    resnet_params = list(map(id, model.backbone.parameters()))
    bert_params = list(map(id, model.bert_model.parameters()))
    
    base_params = filter(
        lambda p: id(p) not in clip_params and id(p) not in bert_params and id(p) not in resnet_params,
        model.parameters()
    )
    
    optimizer = AdamW([
        {"params": base_params},
        {"params": model.model.parameters(), "lr": args.clip_learning_rate},
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps
    )
    
    max_acc = 0.0
    best_epoch = 0
    
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch"):
        sum_loss = 0.0
        sum_step = 0
        
        model.train()
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)")
        
        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list, samples = batch
            
            inputs = processor(
                text=text_list, images=image_list,
                padding='max_length', truncation=True,
                max_length=args.max_len, return_tensors="pt"
            ).to(device)
            labels = torch.tensor(label_list).to(device)
            
            loss, score = model(inputs, batch, labels=labels)
            
            sum_loss += loss.item()
            sum_step += 1
            
            iter_bar.set_description(f"Iter (loss={loss.item():.3f})")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Log training loss
        train_loss = sum_loss / sum_step
        wandb.log({'train_loss': train_loss, 'epoch': i_epoch})
        logger.info(f"Epoch {i_epoch}: train_loss = {train_loss:.4f}")
        
        # Evaluate on dev set
        dev_acc, dev_f1, dev_precision, dev_recall = evaluate(
            args, model, device, dev_data, processor, mode='dev'
        )
        wandb.log({
            'dev_acc': dev_acc, 'dev_f1': dev_f1,
            'dev_precision': dev_precision, 'dev_recall': dev_recall
        })
        logger.info(f"Epoch {i_epoch}: dev_acc={dev_acc:.4f}, dev_f1={dev_f1:.4f}")
        
        # Save best model
        if dev_acc > max_acc:
            max_acc = dev_acc
            best_epoch = i_epoch
            
            save_path = os.path.join(args.output_dir, 'perception_best.pth')
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")
            
            # Evaluate on test set
            test_acc, test_f1, test_precision, test_recall = evaluate(
                args, model, device, test_data, processor, mode='test'
            )
            wandb.log({
                'test_acc': test_acc, 'test_f1': test_f1,
                'test_precision': test_precision, 'test_recall': test_recall
            })
            logger.info(f"Epoch {i_epoch}: test_acc={test_acc:.4f}, test_f1={test_f1:.4f}")
        
        torch.cuda.empty_cache()
    
    logger.info(f"Phase I training complete. Best epoch: {best_epoch}, Best dev acc: {max_acc:.4f}")


def evaluate(args, model, device, data, processor, mode='dev'):
    """Evaluate model on dev/test set"""
    
    data_loader = DataLoader(
        data, batch_size=args.dev_batch_size,
        collate_fn=MyDataset.collate_func, shuffle=False
    )
    
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    
    model.eval()
    
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            text_list, image_list, label_list, id_list, samples = batch
            
            inputs = processor(
                text=text_list, images=image_list,
                padding='max_length', truncation=True,
                max_length=args.max_len, return_tensors="pt"
            ).to(device)
            labels = torch.tensor(label_list).to(device)
            
            loss, t_outputs = model(inputs, batch, labels=labels)
            outputs = torch.argmax(t_outputs, -1)
            
            n_correct += (outputs == labels).sum().item()
            n_total += len(outputs)
            
            if t_targets_all is None:
                t_targets_all = labels
                t_outputs_all = outputs
            else:
                t_targets_all = torch.cat((t_targets_all, labels), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
    
    acc = n_correct / n_total
    f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro')
    precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro')
    recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro')
    
    return acc, f1, precision, recall


def main():
    args = set_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    
    # Initialize wandb
    wandb.init(
        project="MMSD-Agentic",
        name="phase1-perception",
        config=vars(args),
        mode="offline"
    )
    
    # Load data
    train_data = MyDataset(mode='train', text_name=args.text_name, limit=args.limit)
    dev_data = MyDataset(mode='valid', text_name=args.text_name)
    test_data = MyDataset(mode='test', text_name=args.text_name)
    
    logger.info(f"Train size: {len(train_data)}, Dev size: {len(dev_data)}, Test size: {len(test_data)}")
    
    # Initialize model in perception mode
    processor = CLIPProcessor.from_pretrained("./MMSD2.0-main/openai/clip-vit-base-patch32")
    model = MIRDModel(args, mode='perception')
    model.to(device)
    
    logger.info("Starting Phase I: Perception Pre-training...")
    train_phase1(args, model, device, train_data, dev_data, test_data, processor)
    
    wandb.finish()


if __name__ == '__main__':
    main()
