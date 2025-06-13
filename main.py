import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import random
import os
import argparse
from src.dataset import MyDataset, load_jsonl
from src.model import OpinionBERTWithContrastive
import argparse

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

def evaluate(model, val_loader, criterion_ce, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            embeddings, logits = model(input_ids, attention_mask)
            loss = criterion_ce(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Số epoch chờ validation loss không cải thiện để dừng training
            min_delta (float): Giá trị tối thiểu để coi là cải thiện
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

def train_model(model, train_loader, val_loader, criterion_ce, criterion_contrastive, optimizer, scheduler, epochs, patience):
    
    best_val_loss = float('inf')
    best_acc = 0
    early_stop_counter = 0
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
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
        val_loss, val_acc = evaluate(model, val_loader, criterion_ce, DEVICE)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        scheduler.step(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("✅ Saved best model")

        if early_stopping(val_loss):
            print("⏹️ Early stopping triggered.")
            break

    print(f"Training completed. Best Val loss: {best_val_loss:.4f}")
    return best_model_path

def main():
    parser = argparse.ArgumentParser(description="Train OpinionBERT with contrastive loss")
    parser.add_argument('--use_augmenter', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default=MODEL_NAME)
    parser.add_argument('--freeze_layers', type=int, default=FREEZE_LAYERS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--base_lr', type=float, default=BASE_LR)
    parser.add_argument('--head_lr', type=float, default=HEAD_LR)
    parser.add_argument('--max_len', type=int, default=MAX_LEN)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--lambda_contrastive', type=float, default=LAMBDA_CONTRASTIVE)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    args = parser.parse_args()

    # Set seed
    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    # Use args directly
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(
        MyDataset(load_jsonl('data/training_data.jsonl'), tokenizer, mode='train', use_augmenter=args.use_augmenter),
        batch_size=args.batch_size, shuffle=True, generator=g)
    
    val_loader = DataLoader(
        MyDataset(load_jsonl('data/development_data.jsonl'), tokenizer, mode='val'),
        batch_size=args.batch_size, shuffle=False)

    model = OpinionBERTWithContrastive(
        model_name=args.model_name,
        num_classes=args.num_classes,
        freeze_layers=args.freeze_layers
    ).to(DEVICE)

    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_contrastive = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_model_path = train_model(model, train_loader, val_loader, criterion_ce, criterion_contrastive,
                                   optimizer, scheduler, args.epochs, args.patience)

    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()