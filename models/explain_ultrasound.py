"""
PCOSense - Grad-CAM Explainability Script
After training, pass any single ultrasound image through this script to get a heatmap
showing which regions the model looked at to make its PCOS prediction.
Usage: python explain_ultrasound.py --image path/to/ultrasound.jpg
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torch.nn as nn

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH  = r"c:\coding\my projects\PCOS\models\artifacts\ultrasound_model.pth"
CLASS_NAMES = ["infected (PCOS)", "noninfected (Healthy)"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 2)
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"[INFO] Model loaded from {MODEL_PATH}")
    return model

# ─── INFERENCE + GRAD-CAM ────────────────────────────────────────────────────
def explain_image(image_path):
    # Load and preprocess
    raw_img = Image.open(image_path).convert("RGB")
    raw_img_resized = raw_img.resize((224, 224))
    img_array = np.array(raw_img_resized) / 255.0  # for visualization overlay

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    model = load_model()

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = np.argmax(probs)

    print(f"\n[PREDICTION] → {CLASS_NAMES[pred]}")
    print(f"  Confidence: {probs[pred]*100:.1f}%")
    print(f"  Infected (PCOS):  {probs[0]*100:.1f}%")
    print(f"  Noninfected:      {probs[1]*100:.1f}%")

    # Grad-CAM on layer4 (final conv layer of ResNet - highest-level features)
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    cam_image = show_cam_on_image(img_array.astype(np.float32), grayscale_cam, use_rgb=True)

    # ── PLOT ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"PCOSense - Ultrasound Analysis\n"
        f"Prediction: {CLASS_NAMES[pred]}  |  Confidence: {probs[pred]*100:.1f}%",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(raw_img_resized)
    axes[0].set_title("Original Ultrasound")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap\n(red = high importance)")
    axes[1].axis("off")

    axes[2].imshow(cam_image)
    axes[2].set_title("Overlay\n(AI attention regions)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=150, bbox_inches="tight")
    print("\n[INFO] Grad-CAM visualization saved to gradcam_output.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCOSense Grad-CAM Explainability")
    parser.add_argument("--image", type=str, required=True, help="Path to ultrasound image")
    args = parser.parse_args()
    explain_image(args.image)
