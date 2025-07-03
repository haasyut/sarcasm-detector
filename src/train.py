import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_scheduler
)
from sklearn.metrics import accuracy_score, f1_score
from dataset import SarcasmDataset
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np


# Set file paths and hyperparameters
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
SAVE_DIR = "outputs/sarcasm_model"
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 64

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Load datasets and create DataLoaders
train_dataset = SarcasmDataset(TRAIN_PATH, tokenizer, max_length=MAX_LEN)
val_dataset = SarcasmDataset(VAL_PATH, tokenizer, max_length=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Set optimizer and linear learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Training loop for one epoch
def train_epoch():
    model.train()
    all_preds, all_labels = [], []
    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Collect predictions
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Train Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1

# Evaluation loop on validation set
def eval_epoch():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    print(f"Val Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1, precision, recall, all_labels, all_preds

os.makedirs("outputs/eval", exist_ok=True)

history = {
    "epoch": [],
    "train_acc": [],
    "train_f1": [],
    "val_acc": [],
    "val_f1": [],
    "val_precision": [],
    "val_recall": []
}

# Training process with best model saving
best_f1 = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_acc, train_f1 = train_epoch()
    val_acc, val_f1, val_precision, val_recall, y_true, y_pred = eval_epoch()

    print(f"Train Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

    history["epoch"].append(epoch + 1)
    history["train_acc"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)

    if val_f1 > best_f1:
        best_f1 = val_f1
        print("Saving best model...")
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)


print(f"\nBest Validation F1: {best_f1:.4f}")
print(f"Best model saved to: {SAVE_DIR}")

# Save metrics table
metrics_df = pd.DataFrame(history)
metrics_df.to_csv("outputs/eval/metrics_table.csv", index=False)

# Plot Accuracy
plt.plot(history["epoch"], history["train_acc"], label="Train")
plt.plot(history["epoch"], history["val_acc"], label="Validation")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/eval/accuracy_vs_epoch.png")
plt.close()

# Plot F1
plt.plot(history["epoch"], history["train_f1"], label="Train")
plt.plot(history["epoch"], history["val_f1"], label="Validation")
plt.title("F1 Score vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig("outputs/eval/f1_vs_epoch.png")
plt.close()

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Sarcastic", "Sarcastic"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig("outputs/eval/confusion_matrix.png")
plt.close()

# Plot ROC Curve
model.eval()
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_logits = np.array(all_logits)
all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
y_score = all_probs[:, 1]  # score for class 1 (sarcastic)

fpr, tpr, _ = roc_curve(all_labels, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("outputs/eval/roc_curve.png")
plt.close()

print("\n Training complete. Metrics and plots saved to outputs/eval/")