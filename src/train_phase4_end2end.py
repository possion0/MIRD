
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

os.environ["WANDB_MODE"] = "offline"

from data_set_dpo import DPOReasoningDataset
from model_agentic import MIRDModel
from config import AgentConfig, QwenConfig, LoRAConfig, ProjectorConfig
from transformers import CLIPProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ======================= Argument Parsing =======================

def set_args():
    parser = argparse.ArgumentParser(
        description="Phase IV: Implicit Reasoning-Driven End-to-End Fine-Tuning"
    )

    # Device settings
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--seed', type=int, default=42)

    # Model settings (System 1 - perception)
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

    # QLoRA settings
    parser.add_argument('--lora_r', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)

    # Training settings
    parser.add_argument('--num_train_epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--dev_batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--class_head_lr', default=2e-4, type=float)
    parser.add_argument('--lora_class_lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)

    # Phase 4 specific settings
    parser.add_argument('--align_alpha', default=1.0, type=float,
                        help='Weight for the implicit representation alignment loss.')

    # Checkpoint settings
    parser.add_argument('--perception_checkpoint',
                        default='../output_dir/phase1/perception_best.pth',
                        type=str)
    parser.add_argument('--projector_checkpoint',
                        default='../output_dir/phase2/projector_aligned.pth',
                        type=str)
    parser.add_argument('--lora_shared_path',
                        default='../output_dir/phase3_gen/lora_adapter',
                        type=str,
                        help='Path to the pre-trained LoRA_shared adapter.')
    parser.add_argument('--lora_reason_path',
                        default='../output_dir/phase3_dpo/lora_reason_best',
                        type=str,
                        help='Path to the pre-trained LoRA_reason adapter (from DPO).')
    parser.add_argument('--output_dir',
                        default='../output_dir/phase4_e2e', type=str)
    parser.add_argument('--limit', default=None, type=int)

    # Data settings
    parser.add_argument('--train_reasoning_file',
                        default='./MMSD2.0dataset/data/text_json_final/'
                                'train_reasoning_with_negatives.json',
                        type=str)
    parser.add_argument('--valid_reasoning_file',
                        default='./MMSD2.0dataset/data/text_json_final/'
                                'valid_reasoning_with_negatives.json',
                        type=str)

    # Backbone settings (for System 1 compatibility)
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


# ======================= Classification Head =======================

class ClassificationHead(nn.Module):

    def __init__(self, hidden_size, num_classes=2, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):

        hidden_states = hidden_states.to(dtype=self.dense.weight.dtype)
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)


# ======================= Forward Logic =======================

def compute_perception_features(model, inputs, batch_tuple):

    fusion_feature = model._perception_forward(inputs, batch_tuple)  # (B, 768)
    feature_embeddings = model.projector(fusion_feature)  # (B, hidden_size)
    return feature_embeddings


def get_eos_hidden_states(model, feature_embeddings, texts, device, qwen_max_length):

    tokenizer = model.qwen_agent.tokenizer
    qwen_agent = model.qwen_agent

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=qwen_max_length,
        return_tensors="pt"
    ).to(device)

    inputs_embeds, attention_mask = qwen_agent._prepare_inputs_with_features(
        encoded.input_ids, encoded.attention_mask, feature_embeddings
    )

    outputs = qwen_agent.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )

    last_hidden_state = outputs.hidden_states[-1]
    
    # Calculate the exact sequence length (attention_mask includes feature tokens)
    seq_lengths = attention_mask.sum(dim=1) - 1
    
    # Extract the hidden states corresponding to the last valid token
    batch_size = attention_mask.size(0)
    pooled_hidden = torch.stack([
        last_hidden_state[i, seq_lengths[i], :] for i in range(batch_size)
    ])  # (batch_size, hidden_size)

    return pooled_hidden


# ======================= Training Loop =======================

def train_phase4(args, model, class_head, device, train_data, val_data, processor):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_loader = DataLoader(
        dataset=train_data, num_workers=args.num_workers, pin_memory=True,
        batch_size=args.train_batch_size,
        collate_fn=DPOReasoningDataset.collate_func,
        shuffle=True
    )

    total_steps = int(
        len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps
    )

    # ---- Freeze Perception & Projector ----
    model.freeze_perception()
    if model.projector is not None:
        for param in model.projector.parameters():
            param.requires_grad = False

    model.qwen_agent.model.set_adapter("lora_class")
    
    lora_class_params = [
        p for n, p in model.qwen_agent.named_parameters() if p.requires_grad
    ]
    class_head_params = list(class_head.parameters())

    trainable_count = sum(p.numel() for p in lora_class_params + class_head_params)
    logger.info(f"Trainable parameters (LoRA_class + Head): {trainable_count:,}")

    # ---- Optimizer ----
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup

    optimizer = AdamW([
        {"params": lora_class_params, "lr": args.lora_class_lr},
        {"params": class_head_params, "lr": args.class_head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps
    )

    # Loss Functions
    ce_loss_fn = nn.CrossEntropyLoss()
    cos_emb_loss_fn = nn.CosineEmbeddingLoss()

    best_f1 = 0.0

    for i_epoch in trange(args.num_train_epochs, desc="Epoch"):
        sum_loss = 0.0
        sum_acc = 0.0
        sum_step = 0

        model.train()
        class_head.train()
        optimizer.zero_grad()

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)")

        for step, batch in enumerate(iter_bar):
            # DPO data yields positive + 3 negatives, we only need positive + prompt
            (text_list, image_list, label_list, id_list, samples,
             prompt_list, reasoning_list, neg1_list, neg2_list, neg3_list) = batch

            labels = torch.tensor(label_list).to(device)

            clip_inputs = processor(
                text=text_list, images=image_list,
                padding='max_length', truncation=True,
                max_length=args.max_len, return_tensors="pt"
            ).to(device)
            batch_tuple = (text_list, image_list, label_list, id_list, samples)

            # Compute perception features ONCE
            with torch.no_grad():
                feat_emb = compute_perception_features(model, clip_inputs, batch_tuple)

            # --- Teacher Pathway (Frozen) ---
            # Switch to 'lora_reason'
            model.qwen_agent.model.set_adapter("lora_reason")
            with torch.no_grad():
                # Teacher uses prompt + actual optimal reasoning narrative
                teacher_texts = [f"{p} {r}" for p, r in zip(prompt_list, reasoning_list)]
                h_r_teacher = get_eos_hidden_states(
                    model, feat_emb, teacher_texts, device, args.qwen_max_length
                )

            # --- Student Pathway (Trainable) ---
            # Switch to 'lora_class'
            model.qwen_agent.model.set_adapter("lora_class")
            student_texts = prompt_list # Student only sees the prompt, generates empty reasoning representation

            h_r_student = get_eos_hidden_states(
                model, feat_emb, student_texts, device, args.qwen_max_length
            )

            # Convert to float for stable calculations
            h_r_teacher = h_r_teacher.to(device, dtype=torch.float32)
            h_r_student = h_r_student.to(device, dtype=torch.float32)

            # 1. Classification Logits
            logits = class_head(h_r_student)
            
            # Loss Computations
            # L_cls = Cross Entropy Loss
            loss_cls = ce_loss_fn(logits, labels)
            
            # L_align = Cosine Embedding Loss
            # Target is 1 (we want to maximize cosine similarity)
            target_sign = torch.ones(h_r_student.size(0)).to(device)
            loss_align = cos_emb_loss_fn(h_r_student, h_r_teacher, target_sign)

            # Total Loss
            total_loss = loss_cls + args.align_alpha * loss_align

            # Backprop
            total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()

            sum_loss += total_loss.item() * args.gradient_accumulation_steps
            
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            sum_acc += acc.item()
            sum_step += 1

            iter_bar.set_description(
                f"L: {total_loss.item() * args.gradient_accumulation_steps:.3f} "
                f"(Cls:{loss_cls.item():.3f}, Ali:{loss_align.item():.3f})"
            )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    lora_class_params + class_head_params, 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # ---- Epoch Summary ----
        epoch_loss = sum_loss / max(sum_step, 1)
        epoch_acc = sum_acc / max(sum_step, 1)
        
        logger.info(f"Epoch {i_epoch}: Train Loss = {epoch_loss:.4f}, Train Acc = {epoch_acc:.4f}")

        # ---- Validation ----
        val_acc, val_f1, val_prec, val_rec = evaluate_phase4(
            model, class_head, device, val_data, processor, args
        )
        
        logger.info(
            f"Epoch {i_epoch} Validation: "
            f"Acc={val_acc:.4f}, F1={val_f1:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}"
        )
        
        wandb.log({
            'train_loss': epoch_loss, 'train_acc': epoch_acc,
            'val_acc': val_acc, 'val_f1': val_f1, 'val_prec': val_prec, 'val_rec': val_rec,
            'epoch': i_epoch
        })

        # Save Best Model Based on F1
        if val_f1 > best_f1:
            best_f1 = val_f1

            # Save LoRA_class
            lora_class_path = os.path.join(args.output_dir, 'lora_class_best')
            # Safe manual save for just the active adapter lora_class 
            if not os.path.exists(lora_class_path):
                os.makedirs(lora_class_path)
            from peft import get_peft_model_state_dict
            lora_state_dict = get_peft_model_state_dict(model.qwen_agent.model, adapter_name="lora_class")
            torch.save(lora_state_dict, os.path.join(lora_class_path, "adapter_model.bin"))
            model.qwen_agent.model.peft_config["lora_class"].save_pretrained(lora_class_path)
            
            model.qwen_agent.tokenizer.save_pretrained(lora_class_path)

            # Save Classification Head
            class_head_path = os.path.join(args.output_dir, 'class_head_best.pth')
            torch.save(class_head.state_dict(), class_head_path)

            logger.info(f"Saved best model (F1={best_f1:.4f}) to {args.output_dir}")

        torch.cuda.empty_cache()

    logger.info(f"Phase 4 Training Complete. Best Val F1: {best_f1:.4f}")


# ======================= Evaluation =======================

@torch.no_grad()
def evaluate_phase4(model, class_head, device, val_data, processor, args):

    model.eval()
    class_head.eval()
    
    # Must ensure we are using the student adapter for inference
    model.qwen_agent.model.set_adapter("lora_class")

    val_loader = DataLoader(
        val_data, batch_size=args.dev_batch_size, num_workers=args.num_workers,
        collate_fn=DPOReasoningDataset.collate_func, shuffle=False
    )

    all_preds = []
    all_labels = []

    for batch in tqdm(val_loader, desc="Validating"):
        (text_list, image_list, label_list, id_list, samples,
         prompt_list, reasoning_list, neg1_list, neg2_list, neg3_list) = batch

        clip_inputs = processor(
            text=text_list, images=image_list,
            padding='max_length', truncation=True,
            max_length=args.max_len, return_tensors="pt"
        ).to(device)
        batch_tuple = (text_list, image_list, label_list, id_list, samples)

        feat_emb = compute_perception_features(model, clip_inputs, batch_tuple)

        # Student strictly uses prompts without teacher reasoning text
        h_r_student = get_eos_hidden_states(
            model, feat_emb, prompt_list, device, args.qwen_max_length
        )

        h_r_student = h_r_student.to(device, dtype=torch.float32)
        logits = class_head(h_r_student)
        
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label_list)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return acc, f1, prec, rec


# ======================= Model Setup with Dual Adapters =======================

def build_model_phase4(args, device):

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import (
        LoraConfig, get_peft_model, PeftModel,
        TaskType, prepare_model_for_kbit_training
    )

    logger.info("=" * 60)
    logger.info("Building model for Phase IV: Dual Adapters via QLoRA")
    logger.info("=" * 60)

    agent_config = AgentConfig(
        qwen=QwenConfig(model_path=args.qwen_path),
        lora=LoRAConfig(
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
        ),
        projector=ProjectorConfig(),
        mode='agent',
        device=str(device)
    )

    model = MIRDModel(args, mode='perception', agent_config=None)

    from projector import FeatureProjector
    model.projector = FeatureProjector(
        input_dim=agent_config.projector.input_dim,
        hidden_dim=agent_config.projector.hidden_dim,
        output_dim=agent_config.projector.output_dim,
        dropout=agent_config.projector.dropout
    )

    logger.info(f"Loading Qwen from {args.qwen_path} in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_path, trust_remote_code=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    qwen_model = prepare_model_for_kbit_training(qwen_model)

    # 1. Merge LoRA_shared into Base
    if os.path.exists(args.lora_shared_path):
        logger.info(f"Loading and Merging LoRA_shared from {args.lora_shared_path} ...")
        qwen_model = PeftModel.from_pretrained(
            qwen_model,
            args.lora_shared_path,
            adapter_name="lora_shared_temp",
            is_trainable=False
        )
        qwen_model = qwen_model.merge_and_unload()
        logger.info("LoRA_shared merged seamlessly.")
        qwen_model = prepare_model_for_kbit_training(qwen_model)
    else:
        logger.warning(f"LoRA_shared NOT FOUND at {args.lora_shared_path}. Base model will be used without it.")

    # 2. Extract and Load LoRA_reason as an adapter
    if os.path.exists(args.lora_reason_path):
        logger.info(f"Loading LoRA_reason (Teacher) from {args.lora_reason_path}...")
        qwen_model = PeftModel.from_pretrained(
            qwen_model,
            args.lora_reason_path,
            adapter_name="lora_reason",
            is_trainable=False
        )
    else:
        raise ValueError(f"CRITICAL: Teacher adapter LoRA_reason NOT FOUND at {args.lora_reason_path}.")

    # 3. Create NEW LoRA_class adapter
    logger.info("Initializing NEW LoRA_class adapter (Student)...")
    lora_class_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # Add a new adapter and make it the active one
    qwen_model.add_adapter("lora_class", lora_class_config)
    qwen_model.set_adapter("lora_class")

    qwen_model.gradient_checkpointing_enable()

    # Freeze everything except active adapter (lora_class)
    for name, param in qwen_model.named_parameters():
        if "lora_class" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    class QwenAgentWrapper:
        def __init__(self, model, tokenizer, hidden_size):
            self.model = model
            self.tokenizer = tokenizer
            self.hidden_size = hidden_size

        def _prepare_inputs_with_features(self, input_ids, attention_mask, feature_embeddings, feature_positions=None):
            batch_size = input_ids.size(0)
            embed_layer = self.model.get_input_embeddings()
            token_embeddings = embed_layer(input_ids)

            feature_embeddings = feature_embeddings.to(
                device=token_embeddings.device,
                dtype=token_embeddings.dtype
            )
            if feature_embeddings.dim() == 2:
                feature_embeddings = feature_embeddings.unsqueeze(1)

            num_feature_tokens = feature_embeddings.size(1)

            if feature_positions is None:
                inputs_embeds = torch.cat([feature_embeddings, token_embeddings], dim=1)
                feature_mask = torch.ones(
                    batch_size, num_feature_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([feature_mask, attention_mask], dim=1)
            return inputs_embeds, attention_mask

        def named_parameters(self):
            return self.model.named_parameters()

    hidden_size = qwen_model.config.hidden_size
    model.qwen_agent = QwenAgentWrapper(qwen_model, tokenizer, hidden_size)
    model.mode = 'agent'

    # Load Perception and Projector Checkpoints
    if os.path.exists(args.perception_checkpoint):
        logger.info(f"Loading perception checkpoint: {args.perception_checkpoint}")
        state_dict = torch.load(args.perception_checkpoint, map_location='cpu')
        filtered_state = {k: v for k, v in state_dict.items() if 'projector' not in k and 'qwen_agent' not in k}
        model.load_state_dict(filtered_state, strict=False)

    if os.path.exists(args.projector_checkpoint):
        logger.info(f"Loading projector checkpoint: {args.projector_checkpoint}")
        projector_state = torch.load(args.projector_checkpoint, map_location='cpu')
        model.projector.load_state_dict(projector_state)

    logger.info("Model loaded successfully. PEFT Adapters status:")
    logger.info(f"Available Adapters: {list(qwen_model.peft_config.keys())}")
    
    qwen_agent_backup = model.qwen_agent
    model.qwen_agent = None
    model.to(device)
    model.qwen_agent = qwen_agent_backup

    return model, ClassificationHead(hidden_size, num_classes=args.label_number).to(device)


# ======================= Main Function =======================

def main():
    args = set_args()
    
    # Isolate GPUs so device_map="auto" only sees the chosen device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    seed_everything(args.seed)

    wandb.init(
        project="RSAM_Phase4",
        config=vars(args),
        name=f"E2E_Class_Alpha_{args.align_alpha}"
    )

    # Since we isolated the GPU, PyTorch now sees it as "cuda:0" internally
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device (isolated): {device} (Physical GPU {args.device})")

    # Build DUAL-ADAPTER QLoRA model + Classification Head
    model, class_head = build_model_phase4(args, device)

    processor = CLIPProcessor.from_pretrained("./MMSD2.0-main/openai/clip-vit-base-patch32")

    logger.info("Loading training data (with positive reasoning for Teacher)...")
    train_data = DPOReasoningDataset(
        mode='train', text_name=args.text_name,
        reasoning_file=args.train_reasoning_file, limit=args.limit
    )

    logger.info("Loading validation data...")
    val_data = DPOReasoningDataset(
        mode='valid', text_name=args.text_name,
        reasoning_file=args.valid_reasoning_file, limit=args.limit
    )

    train_phase4(args, model, class_head, device, train_data, val_data, processor)


if __name__ == '__main__':
    main()
