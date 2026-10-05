import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn import metrics
import wandb
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_set_reasoning import ReasoningDataset
from model_agentic import MIRDModel
from config import AgentConfig, QwenConfig, LoRAConfig, ProjectorConfig
from transformers import CLIPProcessor

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def set_args():
    parser = argparse.ArgumentParser(description="Phase III: Agent Instruction Tuning (Multi-Task)")
    
    # Device settings
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model settings
    parser.add_argument('--text_name', default='text_json_final', type=str)
    parser.add_argument('--simple_linear', default=False, type=bool)
    parser.add_argument('--text_size', default=512, type=int)
    parser.add_argument('--image_size', default=768, type=int)
    parser.add_argument('--label_number', default=2, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--max_len', default=77, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    
    # Qwen settings
    parser.add_argument('--qwen_path', default='', type=str)
    parser.add_argument('--qwen_max_length', default=512, type=int)
    
    # LoRA settings
    parser.add_argument('--lora_r', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    
    # Training settings
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--dev_batch_size', default=8, type=int)
    parser.add_argument('--projector_lr', default=1e-4, type=float)
    parser.add_argument('--lora_lr', default=2e-4, type=float)
    parser.add_argument('--cls_head_lr', default=2e-4, type=float,
                        help='Learning rate for the classification head')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    
    # Multi-task settings
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Weight for the generation (LM) loss inside total loss.')
    
    # Checkpoint settings
    parser.add_argument('--perception_checkpoint', default='../output_dir/phase1/perception_best.pth', type=str)
    parser.add_argument('--projector_checkpoint', default='../output_dir/phase2/projector_aligned.pth', type=str)
    parser.add_argument('--output_dir', default='../output_dir/phase3_gen', type=str)
    parser.add_argument('--limit', default=None, type=int)
    
    # Resume training settings
    parser.add_argument('--resume_dir', default=None, type=str,
                        help='Path to a previous Phase 3 output dir to resume training from')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Epoch to resume from (0-indexed).')
    
    # Reasoning data settings
    parser.add_argument('--reasoning_file', default=None, type=str,
                        help='Path to JSON file with reasoning data.')
    
    # Backbone settings
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--masks', action='store_true')
    
    # Checkpoint name settings
    parser.add_argument('--checkpoint_name', default='agent_tuned_gen.pth', type=str)
    parser.add_argument('--lora_adapter_name', default='lora_adapter_gen', type=str)
    
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class BinaryClassificationHead(nn.Module):

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, 1)  # Single logit for BCE
    
    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        logit = self.classifier(x)
        return logit


def prepare_prompt_only_inputs(model, prompt_list, device):

    tokenizer = model.qwen_agent.tokenizer
    
    # Tokenize prompts only
    encoded = tokenizer(
        prompt_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    return encoded.input_ids, encoded.attention_mask


def prepare_gen_inputs(model, prompt_list, label_list, reasoning_list, device):

    tokenizer = model.qwen_agent.tokenizer
    
    full_prompts = []
    texts = []
    
    for prompt, label, reasoning in zip(prompt_list, label_list, reasoning_list):
        label_str = "Sarcastic" if label == 1 else "Not Sarcastic"
        
        # Conditioned Prompt
        conditioned_prompt = f"{prompt}\n\nPrediction: {label_str}\nReasoning:"
        
        # Full text for training (Prompt + Reasoning)
        full_text = f"{conditioned_prompt} {reasoning}"
        
        full_prompts.append(conditioned_prompt)
        texts.append(full_text)
        
    # Tokenize full texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    labels = input_ids.clone()
    
    encoded_prompts = tokenizer(
        full_prompts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    for i, prompt_len in enumerate(encoded_prompts.attention_mask.sum(dim=1)):
        labels[i, :prompt_len] = -100
        labels[i][attention_mask[i] == 0] = -100
        
    return input_ids, attention_mask, labels


def forward_cls(model, cls_head, inputs, batch, qwen_input_ids, qwen_attention_mask):
    fusion_feature = model._perception_forward(inputs, batch)  # (B, 768)

    feature_embeddings = model.projector(fusion_feature)  # (B, hidden_size)
    
    qwen_agent = model.qwen_agent
    inputs_embeds, attention_mask = qwen_agent._prepare_inputs_with_features(
        qwen_input_ids, qwen_attention_mask, feature_embeddings
    )
    outputs = qwen_agent.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    last_hidden_state = outputs.hidden_states[-1]
    batch_size = attention_mask.size(0)
    seq_lengths = attention_mask.sum(dim=1) - 1 
    
    pooled_hidden = torch.stack([
        last_hidden_state[i, seq_lengths[i], :] for i in range(batch_size)
    ])  # (batch_size, hidden_size)

    logits = cls_head(pooled_hidden)  # (batch_size, 1)
    
    return logits


def forward_gen(model, inputs, batch, qwen_input_ids, qwen_attention_mask, qwen_labels):

    model.set_mode('agent')
    outputs = model(
        inputs=inputs,
        batch=batch,
        qwen_input_ids=qwen_input_ids,
        qwen_attention_mask=qwen_attention_mask,
        qwen_labels=qwen_labels
    )
    # outputs is (loss,) in training mode
    return outputs[0]


def train_phase3(args, model, cls_head, device, train_data, dev_data, processor,
                 start_epoch=0, best_acc=0.0, resume_optimizer_state=None, resume_scheduler_state=None):
    """Phase III training loop - Multi-Task"""
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.train_batch_size,
        collate_fn=ReasoningDataset.collate_func,
        shuffle=True
    )
    
    total_steps = int(len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    
    # Freeze perception, train projector and LoRA
    logger.info("Freezing System 1 (perception), training Projector, LoRA, and Classification Head...")
    model.freeze_perception()
    model.unfreeze_projector()
    
    # Setup optimizer
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    
    projector_params = list(model.projector.parameters())
    lora_params = [p for n, p in model.qwen_agent.named_parameters() if 'lora' in n.lower() and p.requires_grad]
    cls_params = list(cls_head.parameters())
    
    optimizer = AdamW([
        {"params": projector_params, "lr": args.projector_lr},
        {"params": lora_params, "lr": args.lora_lr},
        {"params": cls_params, "lr": args.cls_head_lr},
    ], weight_decay=args.weight_decay)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    # Restore states if resuming
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    if resume_scheduler_state is not None:
        scheduler.load_state_dict(resume_scheduler_state)
    
    if start_epoch > 0:
        logger.info(f"Resuming training from epoch {start_epoch}, best_acc={best_acc:.4f}")
    
    for i_epoch in trange(start_epoch, int(args.num_train_epochs), desc="Epoch"):
        sum_loss = 0.0
        sum_loss_cls = 0.0
        sum_loss_gen = 0.0
        sum_step = 0
        
        model.train()
        cls_head.train()
        optimizer.zero_grad()
        
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)")
        
        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list, samples, reasoning_list, prompt_list = batch
            
            # Prepare CLIP inputs
            inputs = processor(
                text=text_list, images=image_list,
                padding='max_length', truncation=True,
                max_length=args.max_len, return_tensors="pt"
            ).to(device)
            
            # --- Task 1: Classification ---
            # Prepare Qwen inputs (PROMPT ONLY)
            qwen_input_ids_cls, qwen_attention_mask_cls = prepare_prompt_only_inputs(
                model, prompt_list, device
            )
            
            logits = forward_cls(
                model, cls_head, inputs, batch,
                qwen_input_ids_cls, qwen_attention_mask_cls
            )
            
            binary_labels = torch.tensor(
                label_list, dtype=torch.float32, device=device
            ).unsqueeze(1)
            
            loss_cls = bce_loss_fn(logits.float(), binary_labels)
            
            # --- Task 2: Generation (Conditioned) ---
            # Prepare Qwen inputs (Prompt + Label + Reasoning)
            qwen_input_ids_gen, qwen_attention_mask_gen, qwen_labels_gen = prepare_gen_inputs(
                model, prompt_list, label_list, reasoning_list, device
            )
            
            loss_gen = forward_gen(
                model, inputs, batch,
                qwen_input_ids_gen, qwen_attention_mask_gen, qwen_labels_gen
            )
            
            # --- Combined Loss ---
            loss = loss_cls + args.alpha * loss_gen
            
            # Gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            sum_loss += loss.item() * args.gradient_accumulation_steps
            sum_loss_cls += loss_cls.item()
            sum_loss_gen += loss_gen.item()
            sum_step += 1
            
            iter_bar.set_description(f"Loss: {loss.item()*args.gradient_accumulation_steps:.3f} (Cls: {loss_cls.item():.3f}, Gen: {loss_gen.item():.3f})")
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.projector.parameters()) + lora_params + cls_params, 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # Epoch summary
        epoch_loss = sum_loss / sum_step
        epoch_loss_cls = sum_loss_cls / sum_step
        epoch_loss_gen = sum_loss_gen / sum_step
        
        wandb.log({
            'train_loss': epoch_loss,
            'train_loss_cls': epoch_loss_cls,
            'train_loss_gen': epoch_loss_gen,
            'epoch': i_epoch
        })
        logger.info(f"Epoch {i_epoch}: loss={epoch_loss:.4f}, cls={epoch_loss_cls:.4f}, gen={epoch_loss_gen:.4f}")
        
        # Evaluate (Using Classification Head)
        # We prioritize classification accuracy for saving "best model"
        dev_acc, dev_f1, dev_gen_sample = evaluate_agent_multitask(args, model, cls_head, device, dev_data, processor)
        wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1})
        logger.info(f"Epoch {i_epoch}: dev_acc={dev_acc:.4f}, dev_f1={dev_f1:.4f}")
        logger.info(f"Gen Sample: {dev_gen_sample}")
        
        # Save best model
        if dev_acc > best_acc:
            best_acc = dev_acc
            
            # Save projector
            projector_path = os.path.join(args.output_dir, 'projector_tuned.pth')
            torch.save(model.projector.state_dict(), projector_path)
            
            # Save LoRA adapter
            lora_path = os.path.join(args.output_dir, 'lora_adapter')
            model.qwen_agent.save_lora_adapter(lora_path)
            
            # Save classification head
            cls_head_path = os.path.join(args.output_dir, 'cls_head.pth')
            torch.save(cls_head.state_dict(), cls_head_path)
            
            logger.info(f"Saved best model (acc={best_acc:.4f}) to {args.output_dir}")
        
        # Save per-epoch checkpoint
        epoch_ckpt = {
            'epoch': i_epoch,
            'best_acc': best_acc,
            'projector_state_dict': model.projector.state_dict(),
            'cls_head_state_dict': cls_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch_loss': epoch_loss,
            'dev_acc': dev_acc,
        }
        epoch_ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{i_epoch}.pt')
        torch.save(epoch_ckpt, epoch_ckpt_path)
        
        # Also save LoRA for this epoch
        epoch_lora_path = os.path.join(args.output_dir, f'lora_epoch_{i_epoch}')
        model.qwen_agent.save_lora_adapter(epoch_lora_path)
        
        torch.cuda.empty_cache()
    
    # Save final model
    final_save_path = os.path.join(args.output_dir, 'agent_final.pth')
    torch.save(model.state_dict(), final_save_path)
    final_adapter_path = os.path.join(args.output_dir, 'lora_final')
    if model.qwen_agent and model.qwen_agent.model:
        model.qwen_agent.model.save_pretrained(final_adapter_path)
    final_cls_path = os.path.join(args.output_dir, 'cls_head_final.pth')
    torch.save(cls_head.state_dict(), final_cls_path)
    
    logger.info(f"Phase III training complete. Best dev acc: {best_acc:.4f}")


def evaluate_agent_multitask(args, model, cls_head, device, data, processor):
    """
    Evaluate Classification Accuracy AND Sample a Generation.
    """
    
    data_loader = DataLoader(
        data, batch_size=args.dev_batch_size,
        collate_fn=ReasoningDataset.collate_func, shuffle=False
    )
    
    n_correct, n_total = 0, 0
    all_labels = []
    all_preds = []
    
    model.eval()
    cls_head.eval()
    
    generated_sample = ""
    
    with torch.no_grad():
        for i_batch, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            text_list, image_list, label_list, id_list, samples, reasoning_list, prompt_list = batch
            
            inputs = processor(
                text=text_list, images=image_list,
                padding='max_length', truncation=True,
                max_length=args.max_len, return_tensors="pt"
            ).to(device)
            
            # --- Task 1: Classification ---
            qwen_input_ids, qwen_attention_mask = prepare_prompt_only_inputs(
                model, prompt_list, device
            )
            
            logits = forward_cls(
                model, cls_head, inputs, batch,
                qwen_input_ids, qwen_attention_mask
            )
            
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long().cpu().tolist()
            
            for pred, label in zip(preds, label_list):
                all_preds.append(pred)
                all_labels.append(label)
                if pred == label:
                    n_correct += 1
                n_total += 1
            
            # --- Task 2: Sample Generation (Consistency Check) ---
            # Only do this for the first batch to check qualitative performance
            if i_batch == 0:
                # 1. Prediction (already have `preds`)
                pred_label = preds[0]
                label_str = "Sarcastic" if pred_label == 1 else "Not Sarcastic"
                
                # 2. Condition
                prompt = prompt_list[0]
                conditioned_prompt = f"{prompt}\n\nPrediction: {label_str}\nReasoning:"
                
                # 3. Generate
                model.set_mode('agent')
                # Manual feature extraction to avoid re-computing too much
                fusion_feature = model._perception_forward(inputs, batch)
                feature_embeddings = model.projector(fusion_feature)
                
                # Generate
                # Note: We need to pass the conditioned prompt text
                generated = model.qwen_agent.generate(feature_embeddings[0:1], [conditioned_prompt])
                generated_sample = f"Pred: {label_str} | GT: {label_list[0]} | Gen: {generated[0]}"
    
    acc = n_correct / n_total if n_total > 0 else 0
    f1 = metrics.f1_score(all_labels, all_preds, average='macro') if all_labels else 0
    
    return acc, f1, generated_sample


def main():
    args = set_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    
    # Initialize wandb
    wandb.init(
        project="MMSD-Agentic",
        name="phase3-multitask-gen",
        config=vars(args),
        mode="offline"
    )
    
    # Create agent config with LoRA
    agent_config = AgentConfig(
        qwen=QwenConfig(model_path=args.qwen_path),
        lora=LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        ),
        projector=ProjectorConfig(),
        mode='agent',
        device=str(device)
    )
    
    # Load data
    train_data = ReasoningDataset(
        mode='train', 
        text_name=args.text_name, 
        reasoning_file=args.reasoning_file,
        limit=args.limit
    )
    dev_data = ReasoningDataset(
        mode='valid', 
        text_name=args.text_name
    )
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Dev size: {len(dev_data)}")
    
    # Initialize model
    processor = CLIPProcessor.from_pretrained("./MMSD2.0-main/openai/clip-vit-base-patch32")
    model = MIRDModel(args, mode='agent', agent_config=agent_config)
    
    # Initialize binary classification head
    qwen_hidden_size = model.qwen_agent.hidden_size
    cls_head = BinaryClassificationHead(hidden_size=qwen_hidden_size, dropout=0.1)
    logger.info(f"Classification head initialized with hidden_size={qwen_hidden_size}")
    
    # Resume training state logic
    start_epoch = args.start_epoch
    best_acc = 0.0
    resume_optimizer_state = None
    resume_scheduler_state = None
    
    if args.resume_dir is not None:
        logger.info(f"Resume mode: loading from {args.resume_dir}")
        
        # Load perception base
        if os.path.exists(args.perception_checkpoint):
            state_dict = torch.load(args.perception_checkpoint, map_location='cpu')
            filtered_state = {k: v for k, v in state_dict.items() 
                             if 'projector' not in k and 'qwen_agent' not in k}
            model.load_state_dict(filtered_state, strict=False)
        
        # Detect checkpoint
        if start_epoch > 0:
            resume_epoch = start_epoch - 1
        else:
            resume_epoch = -1
            for f in os.listdir(args.resume_dir):
                if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
                    ep = int(f.replace('checkpoint_epoch_', '').replace('.pt', ''))
                    resume_epoch = max(resume_epoch, ep)
            if resume_epoch >= 0:
                start_epoch = resume_epoch + 1
        
        if resume_epoch >= 0:
            ckpt_path = os.path.join(args.resume_dir, f'checkpoint_epoch_{resume_epoch}.pt')
            if os.path.exists(ckpt_path):
                logger.info(f"Loading checkpoint from epoch {resume_epoch}: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                
                model.projector.load_state_dict(ckpt['projector_state_dict'])
                if 'cls_head_state_dict' in ckpt:
                    cls_head.load_state_dict(ckpt['cls_head_state_dict'])
                
                lora_dir = os.path.join(args.resume_dir, f'lora_epoch_{resume_epoch}')
                if os.path.exists(lora_dir):
                    from peft import PeftModel
                    model.qwen_agent.model = PeftModel.from_pretrained(
                        model.qwen_agent.model.base_model.model,
                        lora_dir, is_trainable=True
                    )
                
                best_acc = ckpt.get('best_acc', 0.0)
                resume_optimizer_state = ckpt.get('optimizer_state_dict', None)
                resume_scheduler_state = ckpt.get('scheduler_state_dict', None)
            else:
                start_epoch = 0
        else:
            start_epoch = 0
    else:
        # Normal Load
        if os.path.exists(args.perception_checkpoint):
            logger.info(f"Loading perception checkpoint from {args.perception_checkpoint}")
            state_dict = torch.load(args.perception_checkpoint, map_location='cpu')
            filtered_state = {k: v for k, v in state_dict.items() 
                             if 'projector' not in k and 'qwen_agent' not in k}
            model.load_state_dict(filtered_state, strict=False)
        
        if os.path.exists(args.projector_checkpoint):
            logger.info(f"Loading projector checkpoint from {args.projector_checkpoint}")
            projector_state = torch.load(args.projector_checkpoint, map_location='cpu')
            model.projector.load_state_dict(projector_state)
    
    model.to(device)
    cls_head.to(device, dtype=torch.bfloat16)  # Ensure matching dtype with Qwen
    
    logger.info(f"Starting Phase III: Multi-Task Training (Cls + Gen, alpha={args.alpha})...")
    train_phase3(args, model, cls_head, device, train_data, dev_data, processor,
                 start_epoch=start_epoch, best_acc=best_acc,
                 resume_optimizer_state=resume_optimizer_state,
                 resume_scheduler_state=resume_scheduler_state)
    
    wandb.finish()


if __name__ == '__main__':
    main()
