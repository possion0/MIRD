from torch.utils.data import Dataset
import logging
import os
from PIL import Image
import json
from misc import nested_tensor_from_tensor_list
from torchvision import transforms

logger = logging.getLogger(__name__)

WORKING_PATH = "./MMSD2.0dataset/data"


class ReasoningDataset(Dataset):
    """
    Dataset for Phase II/III training with reasoning data.
    
    Expected JSON format:
    [
        {
            "image_id": 123456,
            "text": "What a great day to have a flat tire!",
            "label": 1,
            "reasoning": "Analysis: The text uses 'great day' (Positive). Image shows flat tire (Negative). Contradiction detected. Conclusion: Yes, this is sarcastic."
        },
        ...
    ]
    
    If 'reasoning' field is missing, template-based reasoning will be generated.
    """
    
    def __init__(self, mode, text_name, reasoning_file=None, limit=None):
        """
        Args:
            mode: 'train', 'valid', or 'test'
            text_name: Folder name containing JSON files
            reasoning_file: Optional path to JSON file with reasoning data.
                           If None, uses default path: {WORKING_PATH}/{text_name}/{mode}_reasoning.json
                           Falls back to {WORKING_PATH}/{text_name}/{mode}.json if reasoning file not found.
            limit: Limit number of samples (for debugging)
        """
        self.text_name = text_name
        self.mode = mode
        self.reasoning_file = reasoning_file
        self.has_reasoning = False
        
        self.data = self.load_data(mode, reasoning_file, limit)
        self.image_ids = list(self.data.keys())
        
        # Set image paths
        for id in self.data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH, "dataset_image", str(id) + ".jpg")
    
    def load_data(self, mode, reasoning_file, limit):
        """Load data with optional reasoning"""
        data_set = dict()
        cnt = 0
        
        # Try to load reasoning file first
        if reasoning_file and os.path.exists(reasoning_file):
            json_path = reasoning_file
            logger.info(f"Loading reasoning data from: {json_path}")
            self.has_reasoning = True
        else:
            # Try default reasoning file path
            default_reasoning_path = os.path.join(WORKING_PATH, self.text_name, f"{mode}_reasoning.json")
            if os.path.exists(default_reasoning_path):
                json_path = default_reasoning_path
                logger.info(f"Loading reasoning data from: {json_path}")
                self.has_reasoning = True
            else:
                # Fall back to standard file
                json_path = os.path.join(WORKING_PATH, self.text_name, f"{mode}.json")
                logger.info(f"Reasoning file not found, using standard data: {json_path}")
                self.has_reasoning = False
        
        with open(json_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        
        for data in datas:
            if limit is not None and cnt >= limit:
                break
            
            image_id = data['image_id']
            text = data['text']
            label = data['label']
            
            # Check if image exists
            image_path = os.path.join(WORKING_PATH, "dataset_image", str(image_id) + ".jpg")
            if not os.path.isfile(image_path):
                continue
            
            entry = {
                "text": text,
                "label": label
            }
            
            # Load reasoning if available
            if 'reasoning' in data:
                entry['reasoning'] = data['reasoning']
            elif 'qwen_target_str' in data:
                # Alternative field name from new.md spec
                entry['reasoning'] = data['qwen_target_str']
            
            # Load prompt if available
            if 'prompt' in data:
                entry['prompt'] = data['prompt']
            elif 'qwen_input_ids' in data:
                entry['prompt'] = data['qwen_input_ids']
            
            data_set[int(image_id)] = entry
            cnt += 1
        
        logger.info(f"Loaded {len(data_set)} samples, has_reasoning={self.has_reasoning}")
        return data_set
    
    def image_loader(self, id):
        return Image.open(self.data[id]["image_path"])
    
    def text_loader(self, id):
        return self.data[id]["text"]
    
    def reasoning_loader(self, id):
        """Load reasoning, generate template if not available"""
        if 'reasoning' in self.data[id]:
            return self.data[id]['reasoning']
        else:
            # Generate template reasoning based on label
            label = self.data[id]['label']
            text = self.data[id]['text']
            return self._generate_template_reasoning(text, label)
    
    def prompt_loader(self, id):
        """Load prompt, generate if not available"""
        if 'prompt' in self.data[id]:
            return self.data[id]['prompt']
        else:
            text = self.data[id]['text']
            return self._generate_prompt(text)
    
    def _generate_prompt(self, text):
        """Generate standard analysis prompt"""
        return f"""Analyze the provided multimodal content (Image + Text) for sarcasm detection.
The image features are provided above.

Text: {text}

Based on the visual and textual features, provide your analysis:
1. Text sentiment analysis
2. Visual content interpretation
3. Text-image relationship
4. Sarcasm judgment

Analysis:"""
    
    def _generate_template_reasoning(self, text, label):
        """Generate template reasoning when pre-generated data is not available"""
        if label == 1:
            return """The text expresses a sentiment that appears to contrast with typical visual expectations. Upon analyzing the multimodal features, there is an apparent contradiction between the textual tone and the visual content. The text uses language that seems positive or neutral on the surface, but the visual context suggests an opposing reality. This incongruity is a classic indicator of sarcasm. Conclusion: Yes, this is sarcastic."""
        else:
            return """The text conveys a straightforward message that aligns well with the visual content. Both modalities express consistent sentiment and meaning. There is no apparent contradiction or ironic undertone detected between the text and image. The message appears genuine and direct. Conclusion: No, this is not sarcastic."""
    
    def __getitem__(self, index):
        id = self.image_ids[index]
        text = self.text_loader(id)
        image = self.image_loader(id)
        label = self.data[id]["label"]
        reasoning = self.reasoning_loader(id)
        prompt = self.prompt_loader(id)
        
        return text, image, label, id, reasoning, prompt
    
    def __len__(self):
        return len(self.image_ids)
    
    @staticmethod
    def collate_func(batch_data):
        """Collate function for DataLoader"""
        batch_size = len(batch_data)
        
        if batch_size == 0:
            return {}
        
        text_list = []
        image_list = []
        label_list = []
        id_list = []
        reasoning_list = []
        prompt_list = []
        batches = []
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
            reasoning_list.append(instance[4])
            prompt_list.append(instance[5])
            
            samples = transform(instance[1])
            batches.append(samples)
        
        batch = tuple(batches)
        samples = nested_tensor_from_tensor_list(batch)
        
        return text_list, image_list, label_list, id_list, samples, reasoning_list, prompt_list


# Example of expected JSON format
EXAMPLE_REASONING_JSON = """
[
    {
        "image_id": 123456,
        "text": "What a great day to have a flat tire!",
        "label": 1,
        "prompt": "Analyze this multimodal content for sarcasm detection.\\n\\nText: What a great day to have a flat tire!\\n\\nBased on the visual and textual features, provide your analysis:\\n1. Text sentiment analysis\\n2. Visual content interpretation\\n3. Text-image relationship\\n4. Sarcasm judgment\\n\\nAnalysis:",
        "reasoning": "The text uses the phrase 'great day' which expresses positive sentiment. However, 'flat tire' describes an inconvenient situation. The image shows a deflated tire on a car, confirming the negative context. There is a stark contradiction between the sarcastic positivity and the negative visual content. Conclusion: Yes, this is sarcastic."
    },
    {
        "image_id": 789012,
        "text": "Beautiful sunset at the beach",
        "label": 0,
        "prompt": "Analyze this multimodal content...",
        "reasoning": "The text describes a 'beautiful sunset' expressing genuine appreciation. The image shows a scenic beach sunset matching the description. Both text and image convey positive, consistent meaning. No contradiction detected. Conclusion: No, this is not sarcastic."
    }
]
"""
