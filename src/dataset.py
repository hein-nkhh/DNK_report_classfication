import json
import re
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, mode):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        
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