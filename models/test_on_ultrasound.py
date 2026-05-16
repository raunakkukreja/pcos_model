"""
Cross-Dataset Inference Script
Tests the trained model on the ultrasound_ dataset to detect overfitting
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
TRAINED_MODEL_PATH = r"c:\coding\my projects\PCOS\models\artifacts\ultrasound_model.pth"
TEST_DATASET_PATH  = r"c:\coding\my projects\PCOS\models\ultrasound_"
OUTPUT_DIR         = r"c:\coding\my projects\PCOS\models\artifacts"

IMG_SIZE   = 224
BATCH_SIZE = 32

# ─── DEVICE SETUP ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Using device: {device}")

# ─── DATA TRANSFORMS ─────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─── MODEL DEFINITION ────────────────────────────────────────────────────────
print("[INFO] Building ResNet-18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last two blocks
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace FC layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 2)
)
model = model.to(device)

# Load trained weights
print(f"[INFO] Loading trained model from: {TRAINED_MODEL_PATH}")
model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
model.eval()

# ─── LOAD TEST DATASETS ──────────────────────────────────────────────────────
def load_dataset_split(split_path, transform):
    """Load test or train split from ultrasound_ dataset"""
    if not os.path.exists(split_path):
        return None
    
    full_dataset = datasets.ImageFolder(split_path)
    
    # Normalize class names to match original model
    class_mapping = {
        'notinfected': 'noninfected',
        'infected': 'infected'
    }
    class_names = [class_mapping.get(c, c) for c in full_dataset.classes]
    
    # Pre-validate and filter corrupted images
    print(f"  [INFO] Validating images in {split_path}...")
    valid_indices = []
    corrupted_count = 0
    
    for idx, (path, _) in enumerate(full_dataset.imgs):
        try:
            img = Image.open(path)
            img.verify()
            valid_indices.append(idx)
        except Exception as e:
            corrupted_count += 1
    
    if corrupted_count > 0:
        print(f"  [WARNING] Skipping {corrupted_count} corrupted images")
    
    # Create subset with only valid images
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
    
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return self.transform(x), y
    
    wrapped_dataset = TransformDataset(valid_dataset, transform)
    loader = DataLoader(wrapped_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    return loader, class_names

# ─── RUN INFERENCE ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" Cross-Dataset Inference Results")
print("="*70 + "\n")

results = {}

for split in ['test', 'train']:
    split_path = os.path.join(TEST_DATASET_PATH, 'data', split)
    print(f"\n[INFO] Evaluating on {split.upper()} split...")
    
    loader, class_names = load_dataset_split(split_path, test_transforms)
    if loader is None:
        print(f"  ⚠️  {split.upper()} split not found")
        continue
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    results[split] = {
        'accuracy': accuracy,
        'preds': all_preds,
        'labels': all_labels,
        'class_names': class_names
    }
    
    print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    print(f"  Total samples: {len(all_labels)}")

# ─── DETAILED ANALYSIS ───────────────────────────────────────────────────────
print("\n" + "="*70)
print(" Classification Reports by Split")
print("="*70)

for split in ['test', 'train']:
    if split not in results:
        continue
    
    result = results[split]
    print(f"\n{split.upper()} Split Classification Report:")
    print(classification_report(
        result['labels'], 
        result['preds'], 
        target_names=result['class_names']
    ))

# ─── COMPARISON WITH TRAINING RESULTS ────────────────────────────────────────
print("\n" + "="*70)
print(" Overfitting Analysis")
print("="*70 + "\n")

print("[INFO] Original Training Validation Accuracy: 99.1%")
print("[INFO] Model Validation Performance Details:")
print("  - Infected: Precision=0.99, Recall=0.99")
print("  - Noninfected: Precision=0.99, Recall=0.98")

if 'test' in results:
    test_acc = results['test']['accuracy']
    print(f"\n[INFO] Cross-Dataset TEST Accuracy: {test_acc:.4f} ({100*test_acc:.2f}%)")
    
    accuracy_drop = (0.991 - test_acc)
    
    print(f"\n📊 Analysis:")
    print(f"  Accuracy Drop: {100*accuracy_drop:.2f}%")
    
    if accuracy_drop < 0.05:
        print(f"  ✅ EXCELLENT: Drop < 5% → Model GENERALIZES VERY WELL")
        print(f"     No significant overfitting detected")
    elif accuracy_drop < 0.15:
        print(f"  ✅ GOOD: Drop < 15% → Model is REASONABLY ROBUST")
        print(f"     Minimal overfitting signs")
    elif accuracy_drop < 0.25:
        print(f"  ⚠️  CAUTION: Drop 15-25% → POTENTIAL OVERFITTING")
        print(f"     Consider retraining with more regularization")
    else:
        print(f"  ❌ WARNING: Drop > 25% → LIKELY OVERFITTING")
        print(f"     Retraining recommended with:")
        print(f"       • Stronger data augmentation")
        print(f"       • Increased dropout")
        print(f"       • L2 regularization")
        print(f"       • Combined training on both datasets")

# ─── VISUALIZATIONS ──────────────────────────────────────────────────────────
if 'test' in results:
    result = results['test']
    cm = confusion_matrix(result['labels'], result['preds'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=result['class_names'], 
                yticklabels=result['class_names'])
    plt.title("Confusion Matrix - Cross-Dataset Test (ultrasound_)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_path = os.path.join(OUTPUT_DIR, "cross_dataset_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Confusion matrix saved to {cm_path}")
    plt.close()

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" Summary")
print("="*70 + "\n")

if 'test' in results and 'train' in results:
    print(f"Test Accuracy:  {results['test']['accuracy']:.4f} ({100*results['test']['accuracy']:.2f}%)")
    print(f"Train Accuracy: {results['train']['accuracy']:.4f} ({100*results['train']['accuracy']:.2f}%)")
    
    # These are accuracies on ultrasound_ dataset, not training curves
    diff = abs(results['test']['accuracy'] - results['train']['accuracy'])
    print(f"Difference:     {diff:.4f} ({100*diff:.2f}%)")
    
    if diff < 0.03:
        print("\n✅ Test and Train accuracies are CONSISTENT")
    else:
        print("\n⚠️  Difference detected between test/train splits")

print("\n" + "="*70 + "\n")
