import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import random
import os
from dataset import MyDataset, load_jsonl
from model import OpinionBERTWithContrastive

# Thiết lập seed
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)

# Các hằng số
MODEL_NAME = "FacebookAI/xlm-roberta-large"
FREEZE_LAYERS = 8
BATCH_SIZE = 16
EPOCHS = 20
BASE_LR = 2e-5
HEAD_LR = 1e-4
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 20
LAMBDA_CONTRASTIVE = 0.1
PATIENCE = 4

def train_model(model, train_loader, val_loader, criterion_ce, criterion_contrastive, optimizer, scheduler, epochs, patience):
    best_val_f1 = 0
    patience_counter = 0
    best_model_path = "best_model.pth"

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            embeddings, logits = model(input_ids, attention_mask)

            loss_ce = criterion_ce(logits, labels)
            loss_contrastive = criterion_contrastive(embeddings, labels)
            loss = loss_ce + LAMBDA_CONTRASTIVE * loss_contrastive

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                embeddings, logits = model(input_ids, attention_mask)
                loss_ce = criterion_ce(logits, labels)
                loss_contrastive = criterion_contrastive(embeddings, labels)
                loss = loss_ce + LAMBDA_CONTRASTIVE * loss_contrastive
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")

        scheduler.step(avg_val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training completed. Best Val F1: {best_val_f1:.4f}")
    return best_model_path

def main():
    # Load dữ liệu
    train_data = load_jsonl('data/training_data.jsonl')
    val_data = load_jsonl('data/development_data.jsonl')

    # Khởi tạo tokenizer và dataloader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        MyDataset(train_data, tokenizer, mode='train'),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )
    val_loader = DataLoader(
        MyDataset(val_data, tokenizer, mode='val'),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Khởi tạo model, loss, optimizer, scheduler
    model = OpinionBERTWithContrastive(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        freeze_layers=FREEZE_LAYERS
    ).to(DEVICE)

    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_contrastive = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # Huấn luyện
    best_model_path = train_model(
        model, train_loader, val_loader, criterion_ce, criterion_contrastive,
        optimizer, scheduler, EPOCHS, PATIENCE
    )

    print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()