import json
import re
import torch
from torch.utils.data import Dataset
from deep_translator import GoogleTranslator
import nlpaug.augmenter.word as naw
import random
random.seed(42)

def back_translate_german(paragraph, intermediate_lang='en', max_attempts=3):
    try:
        for _ in range(max_attempts):
            # Translate from German to intermediate language (e.g., English)
            translator_to_intermediate = GoogleTranslator(source='de', target=intermediate_lang)
            intermediate_text = translator_to_intermediate.translate(paragraph)
            
            if not intermediate_text:
                continue
                
            # Translate back to German
            translator_to_german = GoogleTranslator(source=intermediate_lang, target='de')
            back_translated = translator_to_german.translate(intermediate_text)
            
            return back_translated
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, mode, n_aug=1, use_augmenter=False):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.use_augmenter = use_augmenter and mode == 'train'
        
        if self.mode == 'train' and self.use_augmenter:
            augmented_data = []
            for sample in data:
                context = " ".join(sample["context"])
                target = sample["target"]
                text = re.sub(r"\s+", " ", context + target).strip()

                for _ in range(n_aug):
                    # Use back-translation
                    aug_text = back_translate_german(text, intermediate_lang='en')
                    if aug_text and aug_text != text:
                        new_sample = sample.copy()
                        new_sample["context"] = [aug_text]
                        new_sample["target"] = ""
                        augmented_data.append(new_sample)
            
            # Gộp dữ liệu gốc và augmented
            print(f"Added {len(augmented_data)} augmented samples")
            self.data.extend(augmented_data)
            random.shuffle(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val', 'test']:
            context = " ".join(self.data[idx]["context"])
            target = self.data[idx]["target"]
            text = context + target
            text = re.sub(r"\s+", " ", text).strip()
            if self.mode in ['train', 'val']:
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
            else:  # 'val'
                encodings = self.tokenizer(
                    text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
        
                return {
                    'input_ids': encodings['input_ids'].flatten(),
                    'attention_mask': encodings['attention_mask'].flatten()
                }

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]