
import os
import sys
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

from data_set_dpo import DPOReasoningDataset
from model_agentic import MIRDModel
from config import AgentConfig, QwenConfig, LoRAConfig, ProjectorConfig
from transformers import CLIPProcessor

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ======================= Argument Parsing =======================

def set_args():
    parser = argparse.ArgumentParser(
        description="Phase III-DPO: Multi-View Preference Distillation"
    )

    # Device settings
    parser.add_argument('--device', default='0', type=str,
                        help='GPU ids, e.g. "0" or "0,1,2,3" for multi-GPU.')
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
    parser.add_argument('--num_workers', default=4, type=int,
                        help='DataLoader num_workers for I/O parallelism.')
    parser.add_argument('--reward_head_lr', default=2e-4, type=float)
    parser.add_argument('--lora_reason_lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)

    # DPO settings
    parser.add_argument('--beta', default=0.1, type=float,
                        help='Temperature coefficient for DPO loss.')
    parser.add_argument('--lambda_init', default=1.0, type=float,
                        help='Initial weight for each negative type.')

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
    parser.add_argument('--output_dir',
                        default='../output_dir/phase3_dpo', type=str)
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


# ======================= Reward Head =======================

class RewardHead(nn.Module):

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, hidden_size) - pooled EOS hidden state
        Returns:
            scores: (batch_size, 1) - scalar reward scores
        """
        # Align dtype with RewardHead weights (e.g. bfloat16)
        hidden_states = hidden_states.to(dtype=self.dense.weight.dtype)
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)


# ======================= Core Forward =======================

def compute_perception_features(model, inputs, batch_tuple):

    fusion_feature = model._perception_forward(inputs, batch_tuple)  # (B,768)
    feature_embeddings = model.projector(fusion_feature)  # (B, hidden_size)
    return feature_embeddings


def compute_all_rewards_batched(
    model, reward_head, feature_embeddings,
    prompt_list, pos_list, neg1_list, neg2_list, neg3_list,
    device, qwen_max_length=512
):

    B = len(prompt_list)
    tokenizer = model.qwen_agent.tokenizer
    qwen_agent = model.qwen_agent

    # ---- Build mega-batch of 4B texts ----
    all_reasoning = pos_list + neg1_list + neg2_list + neg3_list  # 4B items
    all_prompts = prompt_list * 4  # repeat prompts 4 times
    full_texts = [
        f"{p} {r}" for p, r in zip(all_prompts, all_reasoning)
    ]

    # ---- Tokenize in one shot ----
    encoded = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=qwen_max_length,
        return_tensors="pt"
    ).to(device)

    # ---- Replicate feature embeddings 4 times ----
    # (B, hidden) -> (4B, hidden)
    feat_4b = feature_embeddings.repeat(4, 1)

    # ---- Prepare inputs with injected features (handles device transfer) ----
    inputs_embeds, attention_mask = qwen_agent._prepare_inputs_with_features(
        encoded.input_ids, encoded.attention_mask, feat_4b
    )

    # ---- ONE Qwen forward pass with batch=4B ----
    outputs = qwen_agent.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )

    # ---- Pool EOS hidden states ----
    last_hidden_state = outputs.hidden_states[-1]
    total_batch = attention_mask.size(0)  # 4B
    seq_lengths = attention_mask.sum(dim=1) - 1

    pooled_hidden = torch.stack([
        last_hidden_state[i, seq_lengths[i], :]
        for i in range(total_batch)
    ])  # (4B, hidden_size)

    # ---- Reward scores ----
    pooled_hidden = pooled_hidden.to(device)
    all_scores = reward_head(pooled_hidden)  # (4B, 1)

    # ---- Split back into 4 groups of B ----
    score_pos  = all_scores[0*B : 1*B]
    score_neg1 = all_scores[1*B : 2*B]
    score_neg2 = all_scores[2*B : 3*B]
    score_neg3 = all_scores[3*B : 4*B]

    return score_pos, score_neg1, score_neg2, score_neg3


# ======================= DPO Loss =======================

def dpo_loss_fn(score_pos, score_neg, beta=0.1):

    diff = beta * (score_pos - score_neg)
    loss = -F.logsigmoid(diff).mean()
    return loss


# ======================= Dynamic Lambda =======================

def compute_dynamic_lambdas(
    model, reward_head, device, val_data, processor, args, beta
):

    val_loader = DataLoader(
        val_data, batch_size=args.dev_batch_size, num_workers=args.num_workers,
        collate_fn=DPOReasoningDataset.collate_func, shuffle=False
    )

    correct = [0, 0, 0]
    total = 0

    model.eval()
    reward_head.eval()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing lambdas"):
            (text_list, image_list, label_list, id_list, samples,
             prompt_list, reasoning_list,
             neg1_list, neg2_list, neg3_list) = batch

            clip_inputs = processor(
                text=text_list, images=image_list,
                padding='max_length', truncation=True,
                max_length=args.max_len, return_tensors="pt"
            ).to(device)

            batch_tuple = (
                text_list, image_list, label_list, id_list, samples
            )

            # Compute perception features ONCE
            feat_emb = compute_perception_features(
                model, clip_inputs, batch_tuple
            )

            # ALL 4 reward scores in ONE batched Qwen forward
            score_pos, s_neg1, s_neg2, s_neg3 = (
                compute_all_rewards_batched(
                    model, reward_head, feat_emb,
                    prompt_list, reasoning_list,
                    neg1_list, neg2_list, neg3_list,
                    device, args.qwen_max_length
                )
            )
            for k, score_neg in enumerate([s_neg1, s_neg2, s_neg3]):
                correct[k] += (score_pos > score_neg).sum().item()

            total += len(text_list)

    accs = [c / max(total, 1) for c in correct]
    # Weight inversely proportional to accuracy (harder -> higher weight)
    errors = [1.0 - a for a in accs]
    total_err = sum(errors) + 1e-8
    lambdas = [3.0 * e / total_err for e in errors]  # Normalize so sum = 3

    logger.info(
        f"Dynamic lambdas: "
        f"λ_missing={lambdas[0]:.3f} (acc={accs[0]:.3f}), "
        f"λ_distorted={lambdas[1]:.3f} (acc={accs[1]:.3f}), "
        f"λ_broken={lambdas[2]:.3f} (acc={accs[2]:.3f})"
    )

    return lambdas, accs


# ======================= Training Loop =======================

def train_dpo(args, model, reward_head, device, train_data, val_data,
              processor):
    """Main DPO training loop."""

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_loader = DataLoader(
        dataset=train_data, num_workers=args.num_workers, pin_memory=True,
        batch_size=args.train_batch_size,
        collate_fn=DPOReasoningDataset.collate_func,
        shuffle=True
    )

    total_steps = int(
        len(train_loader) * args.num_train_epochs
        / args.gradient_accumulation_steps
    )

    # ---- Freeze everything except LoRA_reason & RewardHead ----
    model.freeze_perception()
    # Freeze projector too (only LoRA_reason is trainable)
    if model.projector is not None:
        for param in model.projector.parameters():
            param.requires_grad = False

    # Collect trainable parameters
    lora_reason_params = [
        p for n, p in model.qwen_agent.named_parameters()
        if p.requires_grad
    ]
    reward_params = list(reward_head.parameters())

    trainable_count = sum(
        p.numel() for p in lora_reason_params + reward_params
    )
    logger.info(f"Trainable parameters: {trainable_count:,}")

    # ---- Optimizer ----
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup

    optimizer = AdamW([
        {"params": lora_reason_params, "lr": args.lora_reason_lr},
        {"params": reward_params, "lr": args.reward_head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps
    )

    # ---- Dynamic lambdas (initialized uniformly) ----
    lambdas = [args.lambda_init, args.lambda_init, args.lambda_init]

    best_avg_acc = 0.0

    for i_epoch in trange(args.num_train_epochs, desc="Epoch"):
        # ---- Compute dynamic lambdas on val set (except epoch 0) ----
        if i_epoch > 0:
            lambdas, val_accs = compute_dynamic_lambdas(
                model, reward_head, device, val_data, processor, args,
                args.beta
            )
            wandb.log({
                'lambda_missing': lambdas[0],
                'lambda_distorted': lambdas[1],
                'lambda_broken': lambdas[2],
                'val_acc_missing': val_accs[0],
                'val_acc_distorted': val_accs[1],
                'val_acc_broken': val_accs[2],
                'epoch': i_epoch
            })

        sum_loss = 0.0
        sum_step = 0

        model.train()
        reward_head.train()
        optimizer.zero_grad()

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)")

        for step, batch in enumerate(iter_bar):
            (text_list, image_list, label_list, id_list, samples,
             prompt_list, reasoning_list,
             neg1_list, neg2_list, neg3_list) = batch

            # Prepare CLIP inputs
            clip_inputs = processor(
                text=text_list, images=image_list,
                padding='max_length', truncation=True,
                max_length=args.max_len, return_tensors="pt"
            ).to(device)

            batch_tuple = (
                text_list, image_list, label_list, id_list, samples
            )

            # ---- Compute perception features ONCE ----
            feat_emb = compute_perception_features(
                model, clip_inputs, batch_tuple
            )

            # ---- ALL 4 reward scores in ONE batched Qwen forward ----
            score_pos, s_neg1, s_neg2, s_neg3 = (
                compute_all_rewards_batched(
                    model, reward_head, feat_emb,
                    prompt_list, reasoning_list,
                    neg1_list, neg2_list, neg3_list,
                    device, args.qwen_max_length
                )
            )

            # DPO loss for each negative type with dynamic weights
            total_loss = torch.tensor(0.0, device=device)
            for k, score_neg in enumerate([s_neg1, s_neg2, s_neg3]):
                loss_k = dpo_loss_fn(score_pos, score_neg, beta=args.beta)
                total_loss = total_loss + lambdas[k] * loss_k

            # Gradient accumulation
            total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()

            sum_loss += total_loss.item() * args.gradient_accumulation_steps
            sum_step += 1

            iter_bar.set_description(
                f"Loss: {total_loss.item() * args.gradient_accumulation_steps:.4f}"
            )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    lora_reason_params + reward_params, 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # ---- Epoch summary ----
        epoch_loss = sum_loss / max(sum_step, 1)
        wandb.log({'train_dpo_loss': epoch_loss, 'epoch': i_epoch})
        logger.info(f"Epoch {i_epoch}: DPO loss = {epoch_loss:.4f}")

        # ---- Validation ----
        val_lambdas, val_accs = compute_dynamic_lambdas(
            model, reward_head, device, val_data, processor, args, args.beta
        )
        avg_acc = sum(val_accs) / 3.0
        wandb.log({
            'val_avg_acc': avg_acc,
            'val_acc_missing': val_accs[0],
            'val_acc_distorted': val_accs[1],
            'val_acc_broken': val_accs[2],
            'epoch': i_epoch
        })
        logger.info(
            f"Epoch {i_epoch}: val avg_acc={avg_acc:.4f}, "
            f"accs=[{val_accs[0]:.4f}, {val_accs[1]:.4f}, {val_accs[2]:.4f}]"
        )

        # ---- Save best model ----
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc

            # Save LoRA_reason adapter
            lora_reason_path = os.path.join(
                args.output_dir, 'lora_reason_best'
            )
            model.qwen_agent.model.save_pretrained(lora_reason_path)
            model.qwen_agent.tokenizer.save_pretrained(lora_reason_path)

            # Save RewardHead
            reward_head_path = os.path.join(
                args.output_dir, 'reward_head_best.pth'
            )
            torch.save(reward_head.state_dict(), reward_head_path)

            # Save projector (frozen but kept for completeness)
            projector_path = os.path.join(
                args.output_dir, 'projector_dpo.pth'
            )
            torch.save(model.projector.state_dict(), projector_path)

            logger.info(
                f"Saved best model (avg_acc={best_avg_acc:.4f}) "
                f"to {args.output_dir}"
            )

        # ---- Per-epoch checkpoint ----
        epoch_ckpt = {
            'epoch': i_epoch,
            'best_avg_acc': best_avg_acc,
            'reward_head_state_dict': reward_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'lambdas': lambdas,
            'val_accs': val_accs,
            'epoch_loss': epoch_loss,
        }
        epoch_ckpt_path = os.path.join(
            args.output_dir, f'checkpoint_epoch_{i_epoch}.pt'
        )
        torch.save(epoch_ckpt, epoch_ckpt_path)

        # Also save LoRA_reason for this epoch
        epoch_lora_path = os.path.join(
            args.output_dir, f'lora_reason_epoch_{i_epoch}'
        )
        model.qwen_agent.model.save_pretrained(epoch_lora_path)

        torch.cuda.empty_cache()

    logger.info(
        f"DPO Training complete. Best val avg_acc: {best_avg_acc:.4f}"
    )


# ======================= Model Setup with QLoRA =======================

def build_model_with_qlora(args, device):

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import (
        LoraConfig, get_peft_model, PeftModel,
        TaskType, prepare_model_for_kbit_training
    )

    logger.info("=" * 60)
    logger.info("Building model with QLoRA (4-bit) + dual LoRA adapters")
    logger.info("=" * 60)

    # ---- Step 1: Build System 1 (perception) WITHOUT Qwen ----
    # We will manually initialize Qwen with QLoRA below.
    # Create a minimal agent_config that skips Qwen initialization.
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

    # Build model in 'perception' mode first to avoid loading Qwen twice
    model = MIRDModel(args, mode='perception', agent_config=None)

    # ---- Step 2: Manually initialize projector ----
    from projector import FeatureProjector
    model.projector = FeatureProjector(
        input_dim=agent_config.projector.input_dim,
        hidden_dim=agent_config.projector.hidden_dim,
        output_dim=agent_config.projector.output_dim,
        dropout=agent_config.projector.dropout
    )

    # ---- Step 3: Load Qwen in 4-bit with BitsAndBytesConfig ----
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
        device_map="auto",  # Multi-GPU: spread Qwen across all visible GPUs
    )


    qwen_model = prepare_model_for_kbit_training(qwen_model)

    if os.path.exists(args.lora_shared_path):
        logger.info(
            f"Loading LoRA_shared from {args.lora_shared_path} ..."
        )
        qwen_model = PeftModel.from_pretrained(
            qwen_model,
            args.lora_shared_path,
            adapter_name="lora_shared",
            is_trainable=False
        )
        # Merge LoRA_shared weights into the base model and unload adapter
        logger.info("Merging LoRA_shared into base model weights...")
        qwen_model = qwen_model.merge_and_unload()
        logger.info("LoRA_shared merged successfully. Effect is now permanent.")

        # Re-prepare for k-bit training after merge
        qwen_model = prepare_model_for_kbit_training(qwen_model)
    else:
        logger.warning(
            f"LoRA_shared not found at {args.lora_shared_path}. "
            f"Proceeding without it."
        )

    # ---- Step 5: Add LoRA_reason as the ONLY trainable adapter ----
    logger.info("Adding LoRA_reason adapter (trainable) ...")
    lora_reason_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    qwen_model = get_peft_model(qwen_model, lora_reason_config)

    # Enable gradient checkpointing to save VRAM
    qwen_model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled for Qwen.")

    # Only LoRA_reason parameters are trainable; base model is frozen
    for name, param in qwen_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Print trainable params
    trainable = sum(
        p.numel() for p in qwen_model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in qwen_model.parameters())
    logger.info(
        f"Qwen trainable (LoRA_reason only): {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # ---- Step 6: Attach Qwen to model via a lightweight wrapper ----
    # We create a simple namespace object mimicking QwenAgent interface
    class QwenAgentWrapper:
        """Lightweight wrapper to provide QwenAgent-compatible interface."""
        def __init__(self, model, tokenizer, hidden_size):
            self.model = model
            self.tokenizer = tokenizer
            self.hidden_size = hidden_size

        def _prepare_inputs_with_features(
            self, input_ids, attention_mask, feature_embeddings,
            feature_positions=None
        ):
            batch_size = input_ids.size(0)
            embed_layer = self.model.get_input_embeddings()
            token_embeddings = embed_layer(input_ids)

            # Align device AND dtype: token_embeddings may be on a
            # different GPU when using device_map="auto"
            feature_embeddings = feature_embeddings.to(
                device=token_embeddings.device,
                dtype=token_embeddings.dtype
            )
            if feature_embeddings.dim() == 2:
                feature_embeddings = feature_embeddings.unsqueeze(1)

            num_feature_tokens = feature_embeddings.size(1)

            if feature_positions is None:
                inputs_embeds = torch.cat(
                    [feature_embeddings, token_embeddings], dim=1
                )
                feature_mask = torch.ones(
                    batch_size, num_feature_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat(
                    [feature_mask, attention_mask], dim=1
                )
            return inputs_embeds, attention_mask

        def named_parameters(self):
            return self.model.named_parameters()

        def save_lora_adapter(self, save_path):
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    hidden_size = qwen_model.config.hidden_size
    model.qwen_agent = QwenAgentWrapper(qwen_model, tokenizer, hidden_size)
    model.mode = 'agent'

    # ---- Step 7: Load perception and projector weights ----
    if os.path.exists(args.perception_checkpoint):
        logger.info(
            f"Loading perception checkpoint: {args.perception_checkpoint}"
        )
        state_dict = torch.load(
            args.perception_checkpoint, map_location='cpu'
        )
        filtered_state = {
            k: v for k, v in state_dict.items()
            if 'projector' not in k and 'qwen_agent' not in k
        }
        model.load_state_dict(filtered_state, strict=False)

    if os.path.exists(args.projector_checkpoint):
        logger.info(
            f"Loading projector checkpoint: {args.projector_checkpoint}"
        )
        projector_state = torch.load(
            args.projector_checkpoint, map_location='cpu'
        )
        model.projector.load_state_dict(projector_state)


    qwen_agent_backup = model.qwen_agent
    model.qwen_agent = None
    model.to(device)
    model.qwen_agent = qwen_agent_backup

    # ---- Step 8: Create RewardHead ----
    reward_head = RewardHead(
        hidden_size=hidden_size, dropout=0.1
    ).to(device, dtype=torch.bfloat16)

    logger.info(f"RewardHead initialized (hidden_size={hidden_size})")

    return model, reward_head


# ======================= Main =======================

def main():
    args = set_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    gpu_ids = args.device.split(',')
    logger.info(f"Using GPU(s): {gpu_ids} (Qwen will use device_map='auto')")

    # Primary device for perception modules and RewardHead
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # Initialize wandb
    wandb.init(
        project="MMSD-Agentic",
        name="phase3-dpo-reasoning",
        config=vars(args),
        mode="offline"
    )

    # Load data
    train_data = DPOReasoningDataset(
        mode='train',
        text_name=args.text_name,
        reasoning_file=args.train_reasoning_file,
        limit=args.limit
    )
    val_data = DPOReasoningDataset(
        mode='valid',
        text_name=args.text_name,
        reasoning_file=args.valid_reasoning_file,
        limit=args.limit
    )
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Val size:   {len(val_data)}")

    # Build model with QLoRA
    model, reward_head = build_model_with_qlora(args, device)

    # CLIP processor
    processor = CLIPProcessor.from_pretrained(
        "./MMSD2.0-main/openai/clip-vit-base-patch32"
    )

    # Train
    logger.info("Starting Phase III-DPO: Multi-View Preference Distillation...")
    train_dpo(args, model, reward_head, device, train_data, val_data,
              processor)

    wandb.finish()


if __name__ == '__main__':
    main()
