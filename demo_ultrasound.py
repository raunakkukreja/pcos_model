"""
PCOSense - Quick Demo Script
Tests the ultrasound model inference + Grad-CAM and saves visual output.
Run in py311: python demo_ultrasound.py
"""

import os
import sys
import io
import base64

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "website", "backend"))

from app.services.inference_cv import predict_ultrasound

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "models", "ultrasound_alisha", "PCOS Data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "artifacts")

def decode_and_save(b64_string, output_path):
    """Decode a data:image/jpeg;base64,... string and save to file."""
    b64_data = b64_string.split(",")[1]
    img_data = base64.b64decode(b64_data)
    with open(output_path, 'wb') as f:
        f.write(img_data)

def run_demo():
    print("=" * 60)
    print("  PCOSense - Ultrasound Model Demo")
    print("=" * 60)
    
    # Find test images from both classes
    test_cases = []
    for class_name in ["infected", "noninfected"]:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= 2:
                test_cases.append((class_name, os.path.join(class_dir, images[0])))
                test_cases.append((class_name, os.path.join(class_dir, images[-1])))
    
    if not test_cases:
        print("❌ No test images found.")
        return
    
    print(f"\n[INFO] Testing {len(test_cases)} images...\n")
    
    for i, (true_label, img_path) in enumerate(test_cases):
        print(f"--- Image {i+1}/{len(test_cases)} -------------------------------")
        print(f"  File: {os.path.basename(img_path)}")
        print(f"  True Label: {true_label}")
        
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        
        try:
            result = predict_ultrasound(img_bytes)
        except Exception as e:
            print(f"  SKIPPED - Error: {e}")
            print()
            continue
        
        pred = result['prediction']
        conf = result['confidence']
        probs = result['probabilities']
        level = result['explanation']['confidence_level']
        
        status = "CORRECT" if pred == true_label else "WRONG"
        
        print(f"  Prediction: {pred.upper()}")
        print(f"  Confidence: {conf:.2%} ({level})")
        print(f"  Probabilities:")
        print(f"    Infected:     {probs['infected']:.2%} {'<--' if pred == 'infected' else ''}")
        print(f"    Non-infected: {probs['noninfected']:.2%} {'<--' if pred == 'noninfected' else ''}")
        print(f"  Result: {status}")
        
        # Save heatmap images
        heatmap_path = os.path.join(OUTPUT_DIR, f"demo_heatmap_{i+1}_{true_label}.jpg")
        decode_and_save(result['heatmap_base64'], heatmap_path)
        print(f"  Heatmap saved: {heatmap_path}")
        
        original_path = os.path.join(OUTPUT_DIR, f"demo_original_{i+1}_{true_label}.jpg")
        decode_and_save(result['original_image_base64'], original_path)
        print(f"  Original saved: {original_path}")
        print()
    
    print("=" * 60)
    print(f"  Demo complete!")
    print(f"  Check saved images in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
