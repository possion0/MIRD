from torch.utils.data import Dataset
import logging
import os
from PIL import Image
import json
from misc import nested_tensor_from_tensor_list
from torchvision import transforms

logger = logging.getLogger(__name__)

WORKING_PATH = "./MMSD2.0dataset/data"


class DPOReasoningDataset(Dataset):


    def __init__(self, mode, text_name, reasoning_file=None, limit=None):

        self.text_name = text_name
        self.mode = mode
        self.reasoning_file = reasoning_file

        self.data = self._load_data(mode, reasoning_file, limit)
        self.image_ids = list(self.data.keys())

        # Set image paths
        for img_id in self.data.keys():
            self.data[img_id]["image_path"] = os.path.join(
                WORKING_PATH, "dataset_image", str(img_id) + ".jpg"
            )

    def _load_data(self, mode, reasoning_file, limit):
        """Load data with positive and negative reasoning."""
        data_set = dict()
        cnt = 0

        # Determine JSON path
        if reasoning_file and os.path.exists(reasoning_file):
            json_path = reasoning_file
        else:
            json_path = os.path.join(
                WORKING_PATH, self.text_name,
                f"{mode}_reasoning_with_negatives.json"
            )

        logger.info(f"[DPO Dataset] Loading from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)

        for data in datas:
            if limit is not None and cnt >= limit:
                break

            image_id = data['image_id']
            text = data['text']
            label = data['label']

            # Check image exists
            image_path = os.path.join(
                WORKING_PATH, "dataset_image", str(image_id) + ".jpg"
            )
            if not os.path.isfile(image_path):
                continue

            # Must have all 4 reasonings for DPO
            reasoning = data.get('reasoning', None)
            neg1 = data.get('negative_reasoning_1', None)
            neg2 = data.get('negative_reasoning_2', None)
            neg3 = data.get('negative_reasoning_3', None)

            if reasoning is None or neg1 is None or neg2 is None or neg3 is None:
                logger.warning(
                    f"Skipping {image_id}: missing reasoning fields."
                )
                continue

            entry = {
                "text": text,
                "label": label,
                "prompt": data.get('prompt', self._generate_prompt(text)),
                "reasoning": reasoning,
                "negative_reasoning_1": neg1,
                "negative_reasoning_2": neg2,
                "negative_reasoning_3": neg3,
            }

            data_set[str(image_id)] = entry
            cnt += 1

        logger.info(
            f"[DPO Dataset] Loaded {len(data_set)} samples with "
            f"positive + 3 negative reasonings."
        )
        return data_set

    def _generate_prompt(self, text):
        """Generate standard analysis prompt (fallback)."""
        return (
            f"Analyze this multimodal content for sarcasm detection.\n\n"
            f"Text: {text}\n\n"
            f"Based on the visual and textual features, provide your analysis:\n"
            f"1. Text sentiment analysis\n"
            f"2. Visual content interpretation\n"
            f"3. Text-image relationship\n"
            f"4. Sarcasm judgment\n\n"
            f"Analysis:"
        )

    def image_loader(self, img_id):
        return Image.open(self.data[img_id]["image_path"])

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        d = self.data[img_id]

        text = d["text"]
        image = self.image_loader(img_id)
        label = d["label"]
        prompt = d["prompt"]
        reasoning = d["reasoning"]
        neg1 = d["negative_reasoning_1"]
        neg2 = d["negative_reasoning_2"]
        neg3 = d["negative_reasoning_3"]

        return text, image, label, img_id, prompt, reasoning, neg1, neg2, neg3

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def collate_func(batch_data):
        """Collate function for DataLoader."""
        batch_size = len(batch_data)
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        prompt_list = []
        reasoning_list = []
        neg1_list = []
        neg2_list = []
        neg3_list = []
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
            reasoning_list.append(instance[5])
            neg1_list.append(instance[6])
            neg2_list.append(instance[7])
            neg3_list.append(instance[8])

            samples = transform(instance[1])
            batches.append(samples)

        batch = tuple(batches)
        samples = nested_tensor_from_tensor_list(batch)

        return (
            text_list, image_list, label_list, id_list, samples,
            prompt_list, reasoning_list, neg1_list, neg2_list, neg3_list
        )
