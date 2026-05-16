"""
Dataset Comparison Script
Compares ultrasound_alisha and ultrasound_ datasets to assess compatibility for cross-validation
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

# Dataset paths
DATASET_1 = r"c:\coding\my projects\PCOS\models\ultrasound_alisha\PCOS Data"
DATASET_2_TEST = r"c:\coding\my projects\PCOS\models\ultrasound_\data\test"
DATASET_2_TRAIN = r"c:\coding\my projects\PCOS\models\ultrasound_\data\train"

def analyze_dataset(dataset_path, dataset_name):
    """Analyze a dataset's structure, size, and image properties"""
    print(f"\n{'='*70}")
    print(f"  {dataset_name} Analysis")
    print(f"{'='*70}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    # Count images per class
    stats = {}
    total_images = 0
    all_sizes = []
    all_formats = []
    
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        count = len(images)
        total_images += count
        stats[class_folder] = count
        
        # Sample image properties from first 5 images
        print(f"\n  📁 Class: {class_folder} ({count} images)")
        for i, img_file in enumerate(images[:5]):
            img_path = os.path.join(class_path, img_file)
            try:
                with Image.open(img_path) as img:
                    size = img.size  # (width, height)
                    fmt = img.format
                    all_sizes.append(size)
                    all_formats.append(fmt)
                    if i == 0:
                        print(f"    Sample image: {img_file}")
                        print(f"      Size: {size[0]}×{size[1]} pixels | Format: {fmt}")
            except Exception as e:
                print(f"    ⚠️  Error reading {img_file}: {str(e)[:50]}")
    
    print(f"\n  📊 Summary:")
    for cls, cnt in stats.items():
        print(f"    {cls}: {cnt} images")
    print(f"    Total: {total_images} images")
    
    if all_sizes:
        # Check size consistency
        unique_sizes = set(all_sizes)
        print(f"\n  🖼️  Image Sizes:")
        print(f"    Unique sizes found: {len(unique_sizes)}")
        if len(unique_sizes) <= 5:
            for size in unique_sizes:
                print(f"      - {size[0]}×{size[1]} pixels")
        else:
            # Show range
            widths = [s[0] for s in all_sizes]
            heights = [s[1] for s in all_sizes]
            print(f"      Width range: {min(widths)} - {max(widths)} pixels")
            print(f"      Height range: {min(heights)} - {max(heights)} pixels")
    
    if all_formats:
        unique_formats = set(all_formats)
        print(f"\n  📄 Image Formats:")
        for fmt in unique_formats:
            count = all_formats.count(fmt)
            print(f"    {fmt}: {count} images")
    
    return {
        'path': dataset_path,
        'total_images': total_images,
        'classes': stats,
        'sample_sizes': all_sizes[:20],
        'formats': all_formats[:20]
    }

def compare_datasets(ds1_info, ds2_info):
    """Compare two datasets and provide compatibility assessment"""
    print(f"\n{'='*70}")
    print(f"  Compatibility Assessment")
    print(f"{'='*70}\n")
    
    if not ds1_info or not ds2_info:
        print("❌ Cannot compare - missing dataset information")
        return
    
    # Class comparison
    print("1️⃣  CLASS STRUCTURE:")
    ds1_classes = set(ds1_info['classes'].keys())
    ds2_classes = set(ds2_info['classes'].keys())
    
    # Normalize class names (handle "notinfected" vs "noninfected")
    ds2_classes_normalized = {c.replace('notinfected', 'noninfected') for c in ds2_classes}
    ds1_classes_normalized = {c.replace('notinfected', 'noninfected') for c in ds1_classes}
    
    if ds1_classes_normalized == ds2_classes_normalized:
        print(f"  ✅ Same class labels (after normalization)")
        print(f"     Dataset 1: {ds1_classes}")
        print(f"     Dataset 2: {ds2_classes}")
    else:
        print(f"  ⚠️  Different class labels!")
        print(f"     Dataset 1: {ds1_classes}")
        print(f"     Dataset 2: {ds2_classes}")
    
    # Size comparison
    print("\n2️⃣  IMAGE SIZE CONSISTENCY:")
    ds1_sizes = set(ds1_info['sample_sizes'])
    ds2_sizes = set(ds2_info['sample_sizes'])
    
    if len(ds1_sizes) <= 2 and len(ds2_sizes) <= 2:
        if ds1_sizes == ds2_sizes:
            print(f"  ✅ Same image sizes")
            for size in ds1_sizes:
                print(f"     {size[0]}×{size[1]} pixels")
        else:
            print(f"  ⚠️  Different image sizes")
            print(f"     Dataset 1: {ds1_sizes}")
            print(f"     Dataset 2: {ds2_sizes}")
    else:
        print(f"  ⚠️  Variable image sizes in one or both datasets")
        print(f"     Dataset 1 has {len(ds1_sizes)} different sizes")
        print(f"     Dataset 2 has {len(ds2_sizes)} different sizes")
        print(f"     ℹ️  Models will resize to 224×224 anyway, so this is OK")
    
    # Format comparison
    print("\n3️⃣  IMAGE FORMAT:")
    ds1_formats = set(ds1_info['formats'])
    ds2_formats = set(ds2_info['formats'])
    print(f"  Dataset 1 formats: {ds1_formats}")
    print(f"  Dataset 2 formats: {ds2_formats}")
    if ds1_formats == ds2_formats:
        print(f"  ✅ Same formats")
    else:
        print(f"  ℹ️  Different formats (both supported by PIL)")
    
    # Total images
    print("\n4️⃣  DATASET SCALE:")
    print(f"  Dataset 1: {ds1_info['total_images']} images")
    print(f"  Dataset 2: {ds2_info['total_images']} images")
    ratio = ds2_info['total_images'] / ds1_info['total_images'] if ds1_info['total_images'] > 0 else 0
    print(f"  Ratio: {ratio:.1%}")
    
    # Final assessment
    print(f"\n{'='*70}")
    print("  🎯 CROSS-VALIDATION FEASIBILITY:")
    print(f"{'='*70}\n")
    
    advisable = True
    reasons = []
    
    if ds1_classes_normalized != ds2_classes_normalized:
        advisable = False
        reasons.append("❌ Different class labels detected")
    
    if len(ds2_info['sample_sizes']) == 0:
        advisable = False
        reasons.append("❌ Dataset 2 appears to have no readable images")
    
    if ds2_info['total_images'] < 100:
        advisable = False
        reasons.append("❌ Dataset 2 is too small for meaningful validation")
    
    if advisable:
        print("✅ RECOMMENDED: Cross-validation is ADVISABLE")
        print("\nReasons:")
        print("  • Compatible class structure")
        print("  • Different data source → good generalization test")
        print(f"  • Sufficient test samples ({ds2_info['total_images']} images)")
        print("\nNext Steps:")
        print("  1. Run inference on Dataset 2")
        print("  2. Compare accuracy vs Dataset 1 validation set")
        print("  3. If accuracy drops >15%, consider retraining with regularization")
    else:
        print("⚠️  CAUTION: Cross-validation may not be reliable")
        print("\nReasons:")
        for reason in reasons:
            print(f"  {reason}")

if __name__ == '__main__':
    print("\n" + "█"*70)
    print("  DATASET COMPARISON & CROSS-VALIDATION FEASIBILITY CHECKER")
    print("█"*70)
    
    # Analyze both datasets
    ds1_info = analyze_dataset(DATASET_1, "Dataset 1 (ultrasound_alisha - TRAINING)")
    ds2_test_info = analyze_dataset(DATASET_2_TEST, "Dataset 2 (ultrasound_ - TEST set)")
    ds2_train_info = analyze_dataset(DATASET_2_TRAIN, "Dataset 2 (ultrasound_ - TRAIN set)")
    
    # Combine Dataset 2 info
    ds2_info = {
        'path': DATASET_2_TEST,
        'total_images': ds2_test_info['total_images'] + ds2_train_info['total_images'],
        'classes': {**ds2_test_info['classes'], **ds2_train_info['classes']},
        'sample_sizes': ds2_test_info['sample_sizes'] + ds2_train_info['sample_sizes'],
        'formats': ds2_test_info['formats'] + ds2_train_info['formats']
    }
    
    # Compare
    compare_datasets(ds1_info, ds2_info)
    
    print(f"\n{'='*70}\n")
