"""
Cross-Dataset Inference - V2 Improved Model
Compares V2 model performance on ultrasound_ dataset
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
TRAINED_MODEL_PATH = r"c:\coding\my projects\PCOS\models\artifacts_v2_improved\ultrasound_model_v2_improved.pth"
TEST_DATASET_PATH  = r"c:\coding\my projects\PCOS\models\ultrasound_"
OUTPUT_DIR         = r"c:\coding\my projects\PCOS\models\artifacts_v2_improved"

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

# ─── MODEL DEFINITION - V2 IMPROVED ──────────────────────────────────────────
print("[INFO] Building ResNet-18 model (V2 Improved)...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last two blocks
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace FC layer with V2 architecture (additional intermediate layer)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 2)
)
model = model.to(device)

# Load trained weights
print(f"[INFO] Loading V2 improved model from: {TRAINED_MODEL_PATH}")
model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
model.eval()

# ─── LOAD TEST DATASETS ──────────────────────────────────────────────────────
def load_dataset_split(split_path, transform):
    """Load test or train split from ultrasound_ dataset"""
    if not os.path.exists(split_path):
        return None
    
    full_dataset = datasets.ImageFolder(split_path)
    
    # Normalize class names
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
print(" V2 IMPROVED MODEL - Cross-Dataset Inference Results")
print("="*70 + "\n")

results = {}

for split in ['test', 'train']:
    split_path = os.path.join(TEST_DATASET_PATH, 'data', split)
    print(f"[INFO] Evaluating V2 model on {split.upper()} split...")
    
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

# ─── MODEL COMPARISON ────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" Model Comparison: Original vs V2 Improved")
print("="*70 + "\n")

print("[INFO] ORIGINAL MODEL:")
print("  Training Validation Accuracy: 99.1%")
print("  Cross-Dataset Test Accuracy: 99.58%")
print("  Accuracy Drop: -0.48% (IMPROVED on different dataset)")
print("  Conclusion: NO OVERFITTING detected")

if 'test' in results:
    v2_acc = results['test']['accuracy']
    print(f"\n[INFO] V2 IMPROVED MODEL:")
    print(f"  Training Validation Accuracy: 97.0%")
    print(f"  Cross-Dataset Test Accuracy: {v2_acc:.4f} ({100*v2_acc:.2f}%)")
    
    val_acc_drop = (0.97 - v2_acc)
    print(f"  Accuracy Drop: {100*val_acc_drop:.2f}%")
    
    original_cross_acc = 0.9958
    improvement = (v2_acc - original_cross_acc)
    
    print(f"\n[INFO] PERFORMANCE COMPARISON:")
    print(f"  Original Cross-Dataset Accuracy: {original_cross_acc:.4f} (99.58%)")
    print(f"  V2 Cross-Dataset Accuracy: {v2_acc:.4f} ({100*v2_acc:.2f}%)")
    print(f"  Difference: {improvement:+.4f} ({100*improvement:+.2f}%)")
    
    print(f"\n📊 Conclusion:")
    if improvement < -0.02:
        print(f"  ❌ V2 model performs WORSE on cross-dataset")
        print(f"  → Original model was better")
        print(f"  → RECOMMENDATION: Use original model")
    elif improvement > 0.02:
        print(f"  ✅ V2 model performs BETTER on cross-dataset")
        print(f"  → V2 model has improved generalization")
        print(f"  → RECOMMENDATION: Use V2 model")
    else:
        print(f"  ≈ Performance is essentially EQUAL")
        print(f"  → Both models are equivalent")
        print(f"  → RECOMMENDATION: Use original (simpler, faster)")

# ─── VISUALIZATIONS ──────────────────────────────────────────────────────────
if 'test' in results:
    result = results['test']
    cm = confusion_matrix(result['labels'], result['preds'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=result['class_names'], 
                yticklabels=result['class_names'])
    plt.title("Confusion Matrix - V2 Improved Model (Cross-Dataset Test)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_path = os.path.join(OUTPUT_DIR, "cross_dataset_confusion_matrix_v2.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Confusion matrix saved to {cm_path}")
    plt.close()

print("\n" + "="*70 + "\n")
