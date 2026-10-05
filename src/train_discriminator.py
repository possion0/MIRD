

import argparse
import logging
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

from data_set_discriminator import DiscriminatorDataset, build_discriminator_messages

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def set_args():
    parser = argparse.ArgumentParser(description="Sarcasm Reasoning Discriminator Training")
    
    # Model
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to Qwen2.5-VL-7B-Instruct model')
    
    # Data
    parser.add_argument('--train_file', type=str,
                        default='./MMSD2.0dataset/data/text_json_final/train_f.json',
                        help='Path to training JSON file')
    parser.add_argument('--valid_file', type=str,
                        default='./MMSD2.0dataset/data/text_json_final/valid_f.json',
                        help='Path to validation JSON file')
    parser.add_argument('--image_dir', type=str,
                        default='./MMSD2.0dataset/data/dataset_image',
                        help='Directory containing images')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples for debugging')
    
    # Training
    parser.add_argument('--output_dir', type=str, default='../output_dir/discriminator',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate for LoRA parameters')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio of total training steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # Qwen2.5-VL specific
    parser.add_argument('--min_pixels', type=int, default=256 * 28 * 28,
                        help='Minimum number of pixels for image processing')
    parser.add_argument('--max_pixels', type=int, default=512 * 28 * 28,
                        help='Maximum number of pixels for image processing')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    
    return parser.parse_args()


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_processor(args):
    """
    Load Qwen2.5-VL-7B with 4-bit quantization and apply LoRA.
    """
    logger.info(f"Loading model from {args.model_path}...")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    
    # Ensure pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    logger.info("Model and processor loaded successfully.")
    return model, processor


def prepare_batch(batch_samples, processor, max_seq_length):

    all_messages = []
    
    for sample in batch_samples:
        messages = build_discriminator_messages(
            text=sample['text'],
            reasoning=sample['reasoning'],
            image_path=sample['image_path'],
            label_str=sample['disc_label'],  # "1" or "0"
        )
        all_messages.append(messages)
    
    # Apply chat template to get text prompts
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in all_messages
    ]
    
    # Collect image inputs - use the Qwen format for image processing
    image_inputs = []
    for msg in all_messages:
        # Extract image from user message content
        user_content = msg[1]["content"]  # user message
        for part in user_content:
            if part["type"] == "image":
                from qwen_vl_utils import process_vision_info
                break
        break
    
    # Use process_vision_info from qwen_vl_utils for proper image processing
    from qwen_vl_utils import process_vision_info
    
    all_image_inputs = []
    all_video_inputs = []
    for msg in all_messages:
        image_inputs, video_inputs = process_vision_info(msg)
        all_image_inputs.extend(image_inputs if image_inputs else [])
        all_video_inputs.extend(video_inputs if video_inputs else [])
    
    # Process through the processor
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    # Create labels: mask everything except the assistant response
    # Labels = input_ids, but with -100 on all positions before/including the assistant prompt tokens
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    
    # For each sequence, find the assistant response and mask everything before it
    for i, text in enumerate(texts):
        # Tokenize the full text and the portion before assistant's answer
        # The assistant response is at the end after the last assistant tag
        # Find the position of the answer in the tokenized sequence
        
        # Build the prompt-only version (without assistant answer)
        prompt_messages = all_messages[i][:-1]  # Remove assistant message
        prompt_messages_with_gen = prompt_messages  # Without add_generation_prompt
        prompt_text = processor.apply_chat_template(
            prompt_messages_with_gen, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize just the prompt part
        prompt_tokens = processor.tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]
        
        # Mask everything before the assistant response (set to -100)
        labels[i, :prompt_length] = -100
    
    # Also mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    inputs["labels"] = labels
    
    return inputs


def evaluate(args, model, processor, eval_dataset, device):
    """
    Evaluate the model on validation set.
    Returns accuracy, precision, recall, f1, and per-class metrics.
    """
    model.eval()
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DiscriminatorDataset.collate_fn,
    )
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_samples in tqdm(eval_loader, desc="Evaluating"):
            # For evaluation, we build messages WITHOUT the assistant answer
            for sample in batch_samples:
                messages = build_discriminator_messages(
                    text=sample['text'],
                    reasoning=sample['reasoning'],
                    image_path=sample['image_path'],
                    label_str=None,  # No answer for inference
                )
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    truncation=True,
                    max_length=args.max_seq_length,
                    return_tensors="pt",
                )
                
                # Move to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}
                
                # Generate
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0,
                )
                
                # Decode only the generated part
                generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                
                # Parse prediction
                pred = parse_prediction(output_text)
                all_preds.append(pred)
                
                # Ground truth
                gt = 1 if sample['disc_label'] == '1' else 0
                all_labels.append(gt)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    report = classification_report(
        all_labels, all_preds,
        target_names=['Incorrect (0)', 'Correct (1)'],
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
    }


def parse_prediction(text):
    """Parse model output to binary prediction. 1=Correct, 0=Incorrect."""
    text = text.strip()
    # Check for exact digits first
    if text == '1':
        return 1
    elif text == '0':
        return 0
    # Fallback heuristic
    if '1' in text:
        return 1
    return 0


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a linear learning rate schedule with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def train(args):
    """Main training loop."""
    seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # Load model
    model, processor = load_model_and_processor(args)
    device = next(model.parameters()).device
    logger.info(f"Model device: {device}")
    
    # Load datasets
    logger.info("Loading training dataset...")
    train_dataset = DiscriminatorDataset(
        json_path=args.train_file,
        image_dir=args.image_dir,
        limit=args.limit,
    )
    
    logger.info("Loading validation dataset...")
    valid_dataset = DiscriminatorDataset(
        json_path=args.valid_file,
        image_dir=args.image_dir,
        limit=args.limit,
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(valid_dataset)}")
    
    # Print a sample for verification
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        sample_messages = build_discriminator_messages(
            text=sample['text'],
            reasoning=sample['reasoning'],
            image_path=sample['image_path'],
            label_str=sample['disc_label'],
        )
        logger.info(f"Sample training instance:")
        logger.info(f"  Image ID: {sample['image_id']}")
        logger.info(f"  Text: {sample['text'][:100]}...")
        logger.info(f"  Reasoning: {sample['reasoning'][:100]}...")
        logger.info(f"  Label: {sample['disc_label']}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DiscriminatorDataset.collate_fn,
    )
    
    # Optimizer - only LoRA params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Training loop
    best_accuracy = 0.0
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            total=len(train_loader),
        )
        
        for step, batch_samples in enumerate(progress_bar):
            try:
                # Prepare inputs
                inputs = prepare_batch(batch_samples, processor, args.max_seq_length)
                
                # Move to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}
                
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                num_batches += 1
                
                # Gradient accumulation step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Update progress bar
                avg_loss = epoch_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': global_step,
                })
                
            except Exception as e:
                logger.error(f"Error processing batch at step {step}: {e}")
                optimizer.zero_grad()
                continue
        
        # Handle remaining gradients
        if (step + 1) % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Evaluation
        logger.info("Running evaluation...")
        eval_results = evaluate(args, model, processor, valid_dataset, device)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy:  {eval_results['accuracy']:.4f}")
        logger.info(f"  Precision: {eval_results['precision']:.4f}")
        logger.info(f"  Recall:    {eval_results['recall']:.4f}")
        logger.info(f"  F1:        {eval_results['f1']:.4f}")
        logger.info(f"\n{eval_results['report']}")
        
        # Save best model
        if eval_results['accuracy'] > best_accuracy:
            best_accuracy = eval_results['accuracy']
            best_adapter_path = os.path.join(args.output_dir, 'best_adapter')
            model.save_pretrained(best_adapter_path)
            processor.save_pretrained(best_adapter_path)
            logger.info(f"*** New best accuracy: {best_accuracy:.4f} - Saved to {best_adapter_path}")
            
            # Save metrics
            with open(os.path.join(best_adapter_path, 'eval_results.json'), 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'accuracy': eval_results['accuracy'],
                    'precision': eval_results['precision'],
                    'recall': eval_results['recall'],
                    'f1': eval_results['f1'],
                }, f, indent=2)
        
        # Save latest checkpoint
        latest_adapter_path = os.path.join(args.output_dir, 'latest_adapter')
        model.save_pretrained(latest_adapter_path)
        processor.save_pretrained(latest_adapter_path)
        
        # Save training state
        training_state = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'best_accuracy': best_accuracy,
            'avg_epoch_loss': avg_epoch_loss,
            'eval_accuracy': eval_results['accuracy'],
        }
        with open(os.path.join(args.output_dir, 'training_state.json'), 'w') as f:
            json.dump(training_state, f, indent=2)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logger.info(f"Best model saved to: {os.path.join(args.output_dir, 'best_adapter')}")


if __name__ == '__main__':
    args = set_args()
    train(args)
