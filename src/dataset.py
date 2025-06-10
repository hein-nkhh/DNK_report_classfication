import json
import re
import torch
from torch.utils.data import Dataset
import nlpaug.augmenter.word as naw
import random
random.seed(42)

def create_german_augmenter():
    return naw.ContextualWordEmbsAug(
        model_path='deepset/gbert-base',
        action='substitute',
        aug_p=0.3,
        aug_min=1,
        aug_max=3,
        top_k=50,
        stopwords=['.', ',', ';', ':', '?', '!', '-', '„', '“'],
        stopwords_regex=r'^[^\wäöüßÄÖÜ]+$',  # bỏ dấu không phải từ
        device='cuda'  # hoặc 'cpu' nếu không có GPU
    )

def augment_german_sentence(sentence, augmenter, max_attempts=5):
    for _ in range(max_attempts):
        augmented = augmenter.augment(sentence)
        # Nếu không có thay đổi hoặc có dấu thay từ, bỏ
        if (augmented != sentence and
            len(set(augmented.split()) ^ set(sentence.split())) >= 1 and
            all(c.isalpha() or c.isspace() for c in augmented.replace('äöüßÄÖÜ', ''))):
            return augmented
    return None  # không có kết quả tốt

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, mode, n_aug=1, use_augmenter=False):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.augmenter = create_german_augmenter() if (mode == 'train' and use_augmenter) else None
        
        if self.mode == 'train' and self.augmenter:
            augmented_data = []
            for sample in data:
                context = " ".join(sample["context"])
                target = sample["target"]
                text = re.sub(r"\s+", " ", context + target).strip()

                for _ in range(n_aug):
                    aug_text = augment_german_sentence(text, self.augmenter)
                    if aug_text:
                        # Tạo sample mới giống sample gốc nhưng thay 'context' và 'target'
                        new_sample = sample.copy()
                        # Bạn có thể chọn cách cắt ngược lại context + target nếu muốn, ở đây đơn giản hóa:
                        new_sample["context"] = [aug_text]
                        new_sample["target"] = ""
                        augmented_data.append(new_sample)
            
            # Gộp dữ liệu gốc và augmented
            self.data.extend(augmented_data)
            random.shuffle(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            context = " ".join(self.data[idx]["context"])
            target = self.data[idx]["target"]
            text = context + target
            text = re.sub(r"\s+", " ", text).strip()
            
        label = int(self.data[idx]["task_a_label"]) - 1
        
        encodings = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]