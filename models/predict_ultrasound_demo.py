"""
Ultrasound Model Demo - With Explainability Features
Demonstrates the trained ResNet-18 CNN model with Grad-CAM visualization and confidence scores
This script shows what the frontend will receive and how to explain results to users
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
import cv2

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
MODEL_PATH = r"c:\coding\my projects\PCOS\models\artifacts\ultrasound_model.pth"
OUTPUT_DIR = r"c:\coding\my projects\PCOS\models\artifacts"

IMG_SIZE = 224
BATCH_SIZE = 1  # Single image for demo

# ─── DEVICE SETUP ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ─── DATA TRANSFORMS ─────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
print(f"[INFO] Loading model from: {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ─── GRAD-CAM IMPLEMENTATION ─────────────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Mapping for visualization"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Calculate Grad-CAM
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = (gradients * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ─── PREDICTION FUNCTION WITH EXPLAINABILITY ─────────────────────────────────
def predict_with_explanation(image_path, class_names=['infected', 'noninfected']):
    """
    Make prediction on an ultrasound image with explainability
    Returns: prediction, confidence, explanation dict
    """
    
    # Load and transform image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {
            'error': f'Failed to load image: {str(e)}',
            'prediction': None,
            'confidence': None
        }
    
    img_original = np.array(img)
    img_tensor = test_transforms(img).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        confidence = probabilities[0, pred_class].item()
        
        # Get all class probabilities
        class_probs = {
            class_names[i]: float(probabilities[0, i].item())
            for i in range(len(class_names))
        }
    
    # Generate Grad-CAM for visualization
    grad_cam = GradCAM(model, model.layer4[-1])
    cam = grad_cam.generate(img_tensor, pred_class)
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (img_original.shape[1], img_original.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_original, 0.5, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.5, 0)
    
    return {
        'prediction': class_names[pred_class],
        'confidence': confidence,
        'all_probabilities': class_probs,
        'predicted_class_idx': pred_class,
        'image_original': img_original,
        'heatmap': overlay,
        'cam': cam_resized,
        'explanation': {
            'model_type': 'ResNet-18 CNN',
            'method': 'Grad-CAM (Gradient-weighted Class Activation Mapping)',
            'interpretation': f"The heatmap shows which regions of the ultrasound image the model focused on to make the '{class_names[pred_class]}' prediction. Red regions are highly influential, blue regions less so.",
            'confidence_level': 'HIGH' if confidence > 0.95 else 'MODERATE' if confidence > 0.85 else 'LOW',
            'next_steps': get_next_steps(class_names[pred_class], confidence)
        }
    }

def get_next_steps(prediction, confidence):
    """Provide clinical recommendations based on prediction"""
    if prediction == 'infected':
        return [
            "✓ Schedule an appointment with a gynecologist for professional diagnosis",
            "✓ Request hormonal panel tests (LH, FSH, testosterone, insulin)",
            "✓ Ask about pelvic ultrasound confirmation",
            "✓ Discuss PCOS management options (lifestyle, medication, monitoring)",
            "✓ Consider specialist referral to endocrinologist if needed"
        ]
    else:
        return [
            "✓ Continue monitoring your reproductive health",
            "✓ Maintain regular gynecology checkups",
            "✓ Keep healthy lifestyle habits",
            "✓ Track menstrual cycle patterns",
            "✓ Report any changes in symptoms to your healthcare provider"
        ]

# ─── VISUALIZATION FUNCTION ──────────────────────────────────────────────────
def visualize_prediction(prediction_result, output_file=None):
    """Create comprehensive visualization of prediction with explainability"""
    
    if 'error' in prediction_result:
        print(f"❌ Error: {prediction_result['error']}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Ultrasound Analysis - Prediction: {prediction_result['prediction'].upper()}", 
                 fontsize=16, fontweight='bold')
    
    # Original Image
    axes[0, 0].imshow(prediction_result['image_original'])
    axes[0, 0].set_title('Original Ultrasound Image')
    axes[0, 0].axis('off')
    
    # Grad-CAM Heatmap
    axes[0, 1].imshow(prediction_result['heatmap'])
    axes[0, 1].set_title('Grad-CAM Attention Map')
    axes[0, 1].axis('off')
    
    # Confidence Scores
    probs = prediction_result['all_probabilities']
    colors = ['#2ecc71' if k == prediction_result['prediction'] else '#e74c3c' 
              for k in probs.keys()]
    axes[1, 0].barh(list(probs.keys()), list(probs.values()), color=colors)
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_title('Model Confidence by Class')
    axes[1, 0].set_xlim(0, 1)
    for i, v in enumerate(probs.values()):
        axes[1, 0].text(v + 0.01, i, f'{v:.1%}', va='center')
    
    # Explanation Text
    axes[1, 1].axis('off')
    explanation = prediction_result['explanation']
    text_content = f"""
EXPLAINABILITY REPORT

Prediction: {prediction_result['prediction'].upper()}
Confidence: {prediction_result['confidence']:.1%}
Confidence Level: {explanation['confidence_level']}

Model: {explanation['model_type']}
Method: {explanation['method']}

Interpretation:
{explanation['interpretation']}

Next Steps:
{chr(10).join(explanation['next_steps'])}
    """
    axes[1, 1].text(0.05, 0.95, text_content, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[INFO] Visualization saved to {output_file}")
    
    return fig

# ─── DEMO WITH TEST IMAGE ────────────────────────────────────────────────────
print("\n" + "="*70)
print(" ULTRASOUND MODEL DEMO - WITH EXPLAINABILITY")
print("="*70 + "\n")

# Load a test image from the ultrasound_alisha dataset
test_dataset_path = r"c:\coding\my projects\PCOS\models\ultrasound_alisha\PCOS Data"
test_infected_dir = os.path.join(test_dataset_path, "infected")

if os.path.exists(test_infected_dir):
    images = [f for f in os.listdir(test_infected_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if images:
        test_image = os.path.join(test_infected_dir, images[0])
        print(f"[INFO] Using test image: {test_image}")
        print(f"[INFO] Running prediction with Grad-CAM visualization...\n")
        
        # Get prediction with explanation
        result = predict_with_explanation(test_image)
        
        # Print results
        print("="*70)
        print(" PREDICTION RESULTS")
        print("="*70)
        print(f"\nPrediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Confidence Level: {result['explanation']['confidence_level']}")
        
        print(f"\nAll Probabilities:")
        for cls, prob in result['all_probabilities'].items():
            print(f"  {cls}: {prob:.2%}")
        
        print(f"\nModel: {result['explanation']['model_type']}")
        print(f"Explainability Method: {result['explanation']['method']}")
        
        print(f"\nInterpretation:")
        print(f"  {result['explanation']['interpretation']}")
        
        print(f"\nRecommended Next Steps:")
        for step in result['explanation']['next_steps']:
            print(f"  {step}")
        
        # Create visualization
        demo_output = os.path.join(OUTPUT_DIR, "ultrasound_demo_prediction.png")
        visualize_prediction(result, output_file=demo_output)
        
        print(f"\n✅ Demo complete!")
        print(f"[INFO] Visualization saved to: {demo_output}")
        
    else:
        print("❌ No test images found in dataset")
else:
    print(f"❌ Test dataset path not found: {test_dataset_path}")

print("\n" + "="*70)
print(" HOW TO USE THIS FOR FRONTEND INTEGRATION")
print("="*70 + """

1. BACKEND API ENDPOINT:
   POST /api/v1/predict/ultrasound
   - Accept: Image file (jpg, jpeg, png)
   - Return: JSON with prediction, confidence, probabilities, heatmap

2. FRONTEND INTEGRATION:
   - Create upload component for ultrasound images
   - Call backend API with image
   - Display results with Grad-CAM heatmap
   - Show confidence scores
   - Display next steps based on prediction

3. RESPONSE FORMAT (JSON):
   {
     "prediction": "infected|noninfected",
     "confidence": 0.9958,
     "probabilities": {
       "infected": 0.9958,
       "noninfected": 0.0042
     },
     "explanation": {
       "method": "Grad-CAM",
       "interpretation": "...",
       "confidence_level": "HIGH",
       "next_steps": [...]
     },
     "heatmap_base64": "..." // For frontend visualization
   }

4. WEARABLE DATA MODEL (FUTURE):
   - Will use XGBoost with SHAP explanations
   - Input: Heart rate, temperature, stress, sleep data
   - Output: Risk factors + SHAP force plots
   - Can be combined with ultrasound for comprehensive assessment

""" + "="*70)
