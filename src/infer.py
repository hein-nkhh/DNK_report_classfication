import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from dataset import MyDataset, load_jsonl
from model import OpinionBERTWithContrastive

# Các hằng số
MODEL_NAME = "FacebookAI/xlm-roberta-large"
BATCH_SIZE = 16
MAX_LEN = 512
NUM_CLASSES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
TEST_DATA_PATH = "data/test_data.jsonl"
OUTPUT_PATH = "predictions.jsonl"

def predict(model, test_loader, tokenizer):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            _, logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Chuyển nhãn từ 0-based về 1-based (như dữ liệu gốc)
            preds = preds + 1
            predictions.extend(preds.tolist())
    
    return predictions

def save_predictions(data, predictions, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item, pred in zip(data, predictions):
            output = {
                "id": item.get("id", ""),
                "predicted_task_a_label": pred
            }
            f.write(json.dumps(output) + "\n")

def main():
    # Load dữ liệu test
    test_data = load_jsonl(TEST_DATA_PATH)
    
    # Khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Khởi tạo dataloader
    test_loader = DataLoader(
        MyDataset(test_data, tokenizer, mode='val'),  # Sử dụng mode='val' vì giống định dạng
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Khởi tạo mô hình
    model = OpinionBERTWithContrastive(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    # Tải trọng số mô hình
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded model from {MODEL_PATH}")
    
    # Dự đoán
    predictions = predict(model, test_loader, tokenizer)
    
    # Lưu kết quả
    save_predictions(test_data, predictions, OUTPUT_PATH)
    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()