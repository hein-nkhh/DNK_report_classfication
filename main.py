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
    parser.add_argument('--use_augmenter', action='store_true', help="Enable data augmentation")
    parser.add_argument('--seed', type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument('--freeze_layers', type=int, default=FREEZE_LAYERS, help="Number of layers to freeze in the BERT model")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument('--base_lr', type=float, default=BASE_LR, help="Base learning rate for the optimizer")
    parser.add_argument('--head_lr', type=float, default=HEAD_LR, help="Learning rate for the classification head")
    parser.add_argument('--max_len', type=int, default=MAX_LEN, help="Maximum sequence length for tokenization")
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help="Number of classes for classification")
    parser.add_argument('--lambda_contrastive', type=float, default=LAMBDA_CONTRASTIVE, help="Weight for contrastive loss")
    parser.add_argument('--patience', type=int, default=PATIENCE, help="Patience for early stopping")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing factor for CrossEntropyLoss")

    args = parser.parse_args()
    
    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    MODEL_NAME = args.model_name if hasattr(args, 'model_name') else MODEL_NAME
    FREEZE_LAYERS = args.freeze_layers if hasattr(args, 'freeze_layers') else FREEZE_LAYERS
    BATCH_SIZE = args.batch_size if hasattr(args, 'batch_size') else BATCH_SIZE
    EPOCHS = args.epochs if hasattr(args, 'epochs') else EPOCHS
    BASE_LR = args.base_lr if hasattr(args, 'base_lr') else BASE_LR
    HEAD_LR = args.head_lr if hasattr(args, 'head_lr') else HEAD_LR
    MAX_LEN = args.max_len if hasattr(args, 'max_len') else MAX_LEN
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = args.num_classes if hasattr(args, 'num_classes') else NUM_CLASSES
    LAMBDA_CONTRASTIVE = args.lambda_contrastive if hasattr(args, 'lambda_contrastive') else LAMBDA_CONTRASTIVE
    PATIENCE = args.patience if hasattr(args, 'patience') else PATIENCE

    # Load dữ liệu
    train_data = load_jsonl('data/training_data.jsonl')
    val_data = load_jsonl('data/development_data.jsonl')

    # Khởi tạo tokenizer và dataloader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        MyDataset(train_data, tokenizer, mode='train', use_augmenter=args.use_augmenter),
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

    criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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