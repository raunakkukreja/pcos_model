"""
PCOSense - Improved CNN Training Script (Anti-Overfitting)
Enhanced regularization techniques to prevent overfitting
"""

import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
DATA_DIR     = r"c:\coding\my projects\PCOS\models\ultrasound_alisha\PCOS Data"
OUTPUT_DIR   = r"c:\coding\my projects\PCOS\models\artifacts_v2_improved"
MODEL_SAVE   = os.path.join(OUTPUT_DIR, "ultrasound_model_v2_improved.pth")
HISTORY_SAVE = os.path.join(OUTPUT_DIR, "training_history_v2_improved.json")

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 5e-5  # Lower learning rate for better generalization
PATIENCE   = 8     # More patience for early stopping

# ─── ENHANCED REGULARIZATION SETTINGS ─────────────────────────────────────────
USE_AGGRESSIVE_AUGMENTATION = True  # Stronger data augmentation
DROPOUT_RATE = 0.5                  # Increased dropout (was 0.4)
WEIGHT_DECAY = 1e-3                 # L2 regularization (was 1e-4)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── DEVICE SETUP ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if device.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── DATA TRANSFORMS - ENHANCED AUGMENTATION ──────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

if USE_AGGRESSIVE_AUGMENTATION:
    print("\n[INFO] Using AGGRESSIVE augmentation to reduce overfitting")
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),      # Increased from 0.2
        transforms.RandomRotation(degrees=30),     # Increased from 15
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Enhanced
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # New: blur augmentation
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # New: random erasing
    ])
else:
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─── DATASET & DATALOADER ────────────────────────────────────────────────────
print(f"\n[INFO] Loading dataset from: {DATA_DIR}")
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names  = full_dataset.classes
print(f"[INFO] Classes found: {class_names}")

# 80/20 train-val split with different seed for variety
n_total = len(full_dataset)
n_val   = int(0.2 * n_total)
n_train = n_total - n_val
train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
print(f"[INFO] Train: {n_train} | Val: {n_val} | Total: {n_total}")

# TransformDataset with error handling
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                x, y = self.subset[idx]
                return self.transform(x), y
            except (OSError, Image.UnidentifiedImageError, Exception) as e:
                if attempt < max_retries - 1:
                    idx = (idx + 1) % len(self.subset)
                else:
                    raise RuntimeError(f"Failed to load valid image after {max_retries} attempts")

train_dataset = TransformDataset(train_dataset_raw, train_transforms)
val_dataset   = TransformDataset(val_dataset_raw,   val_transforms)

# Weighted sampler
train_labels  = [full_dataset.targets[i] for i in train_dataset_raw.indices]
class_counts  = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=0, pin_memory=True)

print(f"[INFO] Class counts → infected: {class_counts[0]}, noninfected: {class_counts[1]}")
print(f"[INFO] Class weights applied → infected: {class_weights[0]:.4f}, noninfected: {class_weights[1]:.4f}")

# ─── MODEL DEFINITION - ENHANCED REGULARIZATION ────────────────────────────────
print("\n[INFO] Building ResNet-18 with Enhanced Regularization...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last two blocks + BatchNorm layers (helps with generalization)
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace FC layer with STRONGER regularization
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE),           # Increased dropout
    nn.Linear(num_features, 256),         # Additional intermediate layer
    nn.BatchNorm1d(256),                  # Batch normalization
    nn.ReLU(),
    nn.Dropout(p=DROPOUT_RATE),           # Extra dropout
    nn.Linear(256, 2)                     # Final output
)
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
print(f"[INFO] Dropout Rate: {DROPOUT_RATE}")
print(f"[INFO] Weight Decay (L2): {WEIGHT_DECAY}")
if USE_AGGRESSIVE_AUGMENTATION:
    print(f"[INFO] Augmentation: AGGRESSIVE (blur, random erasing, increased transforms)")

# ─── LOSS & OPTIMIZER ────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, 
    weight_decay=WEIGHT_DECAY  # Increased L2 regularization
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # ── TRAIN PHASE ──
        model.train()
        running_loss, running_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_loss    += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
        train_loss = running_loss / len(train_dataset)
        train_acc  = running_correct / len(train_dataset)

        # ── VAL PHASE ──
        model.eval()
        val_loss_sum, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs  = model(inputs)
                val_loss_sum += criterion(outputs, labels).item() * inputs.size(0)
                val_correct  += (outputs.argmax(1) == labels).sum().item()
        val_loss = val_loss_sum / len(val_dataset)
        val_acc  = val_correct  / len(val_dataset)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # ── EARLY STOPPING & BEST MODEL ──
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, MODEL_SAVE)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs.")
                break

    model.load_state_dict(best_model_wts)
    return model, history


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" Starting IMPROVED Training (v2)")
    print(" Enhanced regularization to prevent overfitting")
    print("="*60 + "\n")

    model, history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        epochs=EPOCHS, patience=PATIENCE
    )

    # ─── EVALUATION ──────────────────────────────────────────────────────────────
    print("\n[INFO] Running final evaluation on validation set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Improved Model (v2)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_v2_improved.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train", marker="o")
    ax1.plot(history["val_loss"],   label="Val",   marker="o")
    ax1.set_title("Loss (v2 Improved)")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train", marker="o")
    ax2.plot(history["val_acc"],   label="Val",   marker="o")
    ax2.set_title("Accuracy (v2 Improved)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    curves_path = os.path.join(OUTPUT_DIR, "training_curves_v2_improved.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Training curves saved to {curves_path}")
    plt.close()

    # Save history
    with open(HISTORY_SAVE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Training history saved to {HISTORY_SAVE}")
    
    print("\n✅ Improved training complete! Model saved to:", MODEL_SAVE)
    print("\nNext Steps:")
    print("  1. Run: python test_on_ultrasound.py")
    print("  2. Compare cross-dataset accuracy with original model")
    print("  3. If accuracy improved on ultrasound_ dataset → less overfitting ✓")
