
import os
import sys
import argparse
import random
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
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
    parser = argparse.ArgumentParser(description="Phase II: Projector Alignment")
    
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
    
    # Training settings
    parser.add_argument('--num_train_epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    
    # Checkpoint settings
    parser.add_argument('--perception_checkpoint', default='../output_dir/phase1/perception_best.pth', type=str)
    parser.add_argument('--output_dir', default='../output_dir/phase2', type=str)
    parser.add_argument('--limit', default=None, type=int)
    
    # Reasoning data settings
    parser.add_argument('--reasoning_file', default=None, type=str,
                        help='Path to JSON file with reasoning data. If not specified, uses {text_name}/{mode}_reasoning.json')
    
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


def prepare_qwen_inputs(model, prompt_list, reasoning_list, device):

    tokenizer = model.qwen_agent.tokenizer
    
    # Build full sequences: prompt + reasoning
    full_sequences = []
    for prompt, reasoning in zip(prompt_list, reasoning_list):
        # Ensure reasoning starts with space for proper tokenization
        if not reasoning.startswith(' '):
            reasoning = ' ' + reasoning
        full_sequences.append(prompt + reasoning)
    
    # Tokenize prompts only (for masking)
    encoded_prompts = tokenizer(
        prompt_list,
        padding=True,
        truncation=True,
        max_length=400,
        return_tensors="pt"
    ).to(device)
    
    # Tokenize full sequences (prompt + reasoning)
    encoded_full = tokenizer(
        full_sequences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Create labels: -100 for prompt tokens (only compute loss on reasoning)
    labels = encoded_full.input_ids.clone()
    prompt_length = encoded_prompts.input_ids.shape[1]
    labels[:, :prompt_length] = -100  # Mask prompt tokens
    
    return encoded_full.input_ids, encoded_full.attention_mask, labels


def train_phase2(args, model, device, train_data, processor):
    """Phase II training loop - Projector alignment"""
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.train_batch_size,
        collate_fn=ReasoningDataset.collate_func,
        shuffle=True
    )
    
    total_steps = int(len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    
    # Freeze all except projector
    logger.info("Freezing System 1 and Qwen backbone...")
    model.freeze_perception()
    model.freeze_qwen_backbone()
    model.unfreeze_projector()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer for projector only
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    
    optimizer = AdamW(
        model.projector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps
    )
    
    best_loss = float('inf')
    
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch"):
        sum_loss = 0.0
        sum_step = 0
        
        model.train()
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
            
            # Prepare Qwen inputs using reasoning from dataset
            qwen_input_ids, qwen_attention_mask, qwen_labels = prepare_qwen_inputs(
                model, prompt_list, reasoning_list, device
            )
            
            # Forward pass
            loss_tuple = model(
                inputs, batch,
                qwen_input_ids=qwen_input_ids,
                qwen_attention_mask=qwen_attention_mask,
                qwen_labels=qwen_labels
            )
            loss = loss_tuple[0]
            
            # Gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            sum_loss += loss.item() * args.gradient_accumulation_steps
            sum_step += 1
            
            iter_bar.set_description(f"Iter (loss={loss.item() * args.gradient_accumulation_steps:.3f})")
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # Epoch summary
        epoch_loss = sum_loss / sum_step
        wandb.log({'train_loss': epoch_loss, 'epoch': i_epoch})
        logger.info(f"Epoch {i_epoch}: loss = {epoch_loss:.4f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
            # Save projector weights
            save_path = os.path.join(args.output_dir, 'projector_aligned.pth')
            torch.save(model.projector.state_dict(), save_path)
            logger.info(f"Saved projector to {save_path}")
            
            # Also save full model state
            full_save_path = os.path.join(args.output_dir, 'model_phase2.pth')
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_checkpoint(full_save_path)


        
        torch.cuda.empty_cache()
    
    # Save final model
    final_save_path = os.path.join(args.output_dir, 'projector_final.pth')
    torch.save(model.projector.state_dict(), final_save_path)
    logger.info(f"Saved final projector to {final_save_path}")

    logger.info(f"Phase II training complete. Best loss: {best_loss:.4f}")


def main():
    args = set_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    
    # Initialize wandb
    wandb.init(
        project="MMSD-Agentic",
        name="phase2-alignment",
        config=vars(args),
        mode="offline"
    )
    
    # Create agent config
    agent_config = AgentConfig(
        qwen=QwenConfig(model_path=args.qwen_path),
        lora=LoRAConfig(),  # LoRA not used in phase 2
        projector=ProjectorConfig(),
        mode='alignment',
        device=str(device)
    )
    
    # Load data with reasoning
    train_data = ReasoningDataset(
        mode='train', 
        text_name=args.text_name, 
        reasoning_file=args.reasoning_file,
        limit=args.limit
    )
    logger.info(f"Train size: {len(train_data)}, has_reasoning: {train_data.has_reasoning}")
    
    # Initialize model in alignment mode
    processor = CLIPProcessor.from_pretrained("./MMSD2.0-main/openai/clip-vit-base-patch32")
    model = MIRDModel(args, mode='alignment', agent_config=agent_config)
    
    # Load Phase I checkpoint
    if os.path.exists(args.perception_checkpoint):
        logger.info(f"Loading perception checkpoint from {args.perception_checkpoint}")
        state_dict = torch.load(args.perception_checkpoint, map_location='cpu')
        
        # Filter out projector and qwen_agent keys
        filtered_state = {k: v for k, v in state_dict.items() 
                         if 'projector' not in k and 'qwen_agent' not in k}
        model.load_state_dict(filtered_state, strict=False)
    else:
        logger.warning(f"Perception checkpoint not found: {args.perception_checkpoint}")
    
    model.to(device)
    
    logger.info("Starting Phase II: Projector Alignment...")
    train_phase2(args, model, device, train_data, processor)
    
    wandb.finish()


if __name__ == '__main__':
    main()
