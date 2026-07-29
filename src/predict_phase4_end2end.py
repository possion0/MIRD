import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import logging
from PIL import Image

from transformers import CLIPProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model_agentic import AgenticRCLMuFN
from config import AgentConfig, QwenConfig, LoRAConfig, ProjectorConfig
from misc import nested_tensor_from_tensor_list
from torchvision import transforms

from train_phase4_end2end import ClassificationHead, compute_perception_features, get_eos_hidden_states


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


WORKING_PATH = "./MMSD2.0dataset/data"

# ======================= Test Dataset =======================

class TestDataset(Dataset):
    """Dataset loader for evaluating on test.json"""
    def __init__(self, json_path, limit=None):
        self.json_path = json_path
        self.data = self._load_data(limit)
        self.image_ids = list(self.data.keys())
        
        for img_id in self.data.keys():
            self.data[img_id]["image_path"] = os.path.join(
                WORKING_PATH, "dataset_image", str(img_id) + ".jpg"
            )

    def _load_data(self, limit):
        data_set = dict()
        cnt = 0
        with open(self.json_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
            
        for data in datas:
            if limit is not None and cnt >= limit:
                break
            
            image_id = data['image_id']
            text = data['text']
            label = data['label']
            
            image_path = os.path.join(WORKING_PATH, "dataset_image", str(image_id) + ".jpg")
            if not os.path.isfile(image_path):
                continue
                
            # Create a simple prompt mimicking the training format
            prompt = (
                f"Analyze this multimodal content for sarcasm detection.\n\n"
                f"Text: {text}\n\n"
                f"Based on the visual and textual features, provide your analysis:\n"
                f"1. Text sentiment analysis\n"
                f"2. Visual content interpretation\n"
                f"3. Text-image relationship\n"
                f"4. Sarcasm judgment\n\n"
                f"Analysis:"
            )
            
            data_set[str(image_id)] = {
                "text": text,
                "label": label,
                "prompt": prompt
            }
            cnt += 1
            
        return data_set

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        d = self.data[img_id]
        
        text = d["text"]
        image = Image.open(d["image_path"]).convert("RGB")
        label = d["label"]
        prompt = d["prompt"]
        
        return text, image, label, img_id, prompt

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        prompt_list = []
        batches = []

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
            prompt_list.append(instance[4])

            samples = transform(instance[1])
            batches.append(samples)

        batch = tuple(batches)
        samples = nested_tensor_from_tensor_list(batch)

        return text_list, image_list, label_list, id_list, samples, prompt_list


# ======================= Argument Parsing =======================

def set_args():
    parser = argparse.ArgumentParser(description="Phase IV: Prediction Script")

    parser.add_argument('--device', default='0', type=str)
    
    # Model parameters (need to match what was trained)
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

    # Backbone settings
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--masks', action='store_true')

    # Path to checkponts
    parser.add_argument('--perception_checkpoint', default='../output_dir/phase1/perception_best.pth', type=str)
    parser.add_argument('--projector_checkpoint', default='../output_dir/phase2/projector_aligned.pth', type=str)
    parser.add_argument('--lora_shared_path', default='../output_dir/phase3_gen/lora_adapter', type=str)
    
    # We only care about lora_class and classification head for inference
    parser.add_argument('--lora_class_path', default='../output_dir/phase4_e2e/lora_class_best', type=str)
    parser.add_argument('--class_head_path', default='../output_dir/phase4_e2e/class_head_best.pth', type=str)
    
    parser.add_argument('--test_file', default='./MMSD2.0dataset/data/text_json_final/test.json', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--limit', default=None, type=int)

    return parser.parse_args()


# ======================= Model Loading Function =======================

def load_inference_model(args, device):
    """
    Build the Qwen model incorporating only what's needed for inference:
    1. Base Model + LoRA_shared (Merged into base via PEFT)
    2. LoRA_class (Loaded as an adapter)
    """
    logger.info("=" * 60)
    logger.info("Building Inference Model for Phase IV")
    logger.info("=" * 60)

    agent_config = AgentConfig(
        qwen=QwenConfig(model_path=args.qwen_path),
        lora=LoRAConfig(),  # Dummy config
        projector=ProjectorConfig(),
        mode='agent',
        device=str(device)
    )

    model = AgenticRCLMuFN(args, mode='perception', agent_config=None)

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

    # 2. Extract and Load LoRA_class as the active adapter
    if os.path.exists(args.lora_class_path):
        logger.info(f"Loading Student LoRA_class from {args.lora_class_path}...")
        qwen_model = PeftModel.from_pretrained(
            qwen_model,
            args.lora_class_path,
            adapter_name="lora_class",
            is_trainable=False
        )
    else:
        raise ValueError(f"CRITICAL: LoRA_class NOT FOUND at {args.lora_class_path}.")

    # Make sure we are using lora_class
    qwen_model.set_adapter("lora_class")
    
    # Freeze everything
    for param in qwen_model.parameters():
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
        
    # Move ALL perception + projector modules to device in one pass.
    qwen_agent_backup = model.qwen_agent
    model.qwen_agent = None
    model.to(device)
    model.qwen_agent = qwen_agent_backup

    # Load Classification Head
    class_head = ClassificationHead(hidden_size, num_classes=args.label_number).to(device)
    if os.path.exists(args.class_head_path):
        logger.info(f"Loading Classification Head from {args.class_head_path}...")
        class_head.load_state_dict(torch.load(args.class_head_path, map_location=device))
    else:
        raise ValueError(f"CRITICAL: Classification Head NOT FOUND at {args.class_head_path}.")

    model.eval()
    class_head.eval()
    
    return model, class_head

# ======================= Prediction Routine =======================

@torch.no_grad()
def predict(args, model, class_head, device, test_data, processor):
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=TestDataset.collate_func, shuffle=False
    )

    all_preds = []
    all_labels = []

    logger.info("Starting prediction...")
    for batch in tqdm(test_loader, desc="Predicting"):
        text_list, image_list, label_list, id_list, samples, prompt_list = batch

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

    logger.info("====================================")
    logger.info("Prediction Results on Test Dataset:")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("====================================")


def main():
    args = set_args()
    
    # Isolate GPUs so device_map="auto" only sees the chosen device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Since we isolated the GPU, PyTorch now sees it as "cuda:0" internally
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on device (isolated): {device} (Physical GPU {args.device})")

    model, class_head = load_inference_model(args, device)
    processor = CLIPProcessor.from_pretrained("./MMSD2.0-main/openai/clip-vit-base-patch32")

    logger.info(f"Loading test dataset from {args.test_file}...")
    test_data = TestDataset(json_path=args.test_file, limit=args.limit)
    
    predict(args, model, class_head, device, test_data, processor)


if __name__ == '__main__':
    main()
