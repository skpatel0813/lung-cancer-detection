import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.cnn_model import LungCancerCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import argparse
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_image(image_path, img_size=384):
    """Enhanced preprocessing pipeline"""
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0), img.resize((img_size, img_size))

def predict_with_explanation(image_path, config_path='configs/predict.yaml'):
    """Prediction with Grad-CAM and clinical explanation for binary classification"""
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = LungCancerCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    # Preprocess image
    input_tensor, pil_img = preprocess_image(image_path, config['image_size'])
    input_tensor = input_tensor.to(device)
    rgb_img = np.array(pil_img) / 255.0
    rgb_img = np.float32(rgb_img)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    # Grad-CAM
    target_layers = model.get_cam_target_layers()
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device.type == 'cuda')
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5)

    # Explanation
    class_names = ['Normal', 'Cancer']
    explanation = generate_explanation(class_names[pred_idx], confidence, grayscale_cam, probs.cpu().numpy())

    # Show images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Prediction: {class_names[pred_idx]} ({confidence:.1%})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("\nExplanation:")
    print(explanation)

    return class_names[pred_idx], confidence, explanation

def generate_explanation(prediction, confidence, heatmap, probabilities):
    """Binary classification explanation"""
    h, w = heatmap.shape
    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    region_x = "left" if max_x < w/3 else "center" if max_x < 2*w/3 else "right"
    region_y = "upper" if max_y < h/3 else "middle" if max_y < 2*h/3 else "lower"
    focus_region = f"{region_y} {region_x}"

    if prediction == 'Normal':
        return (
            f"The model predicts a **normal** scan with {confidence:.1%} confidence.\n"
            f"Attention was focused on the {focus_region} region, showing typical lung patterns.\n\n"
            f"Probability distribution:\n"
            f"- Normal: {probabilities[0]:.1%}\n"
            f"- Cancer: {probabilities[1]:.1%}"
        )
    else:  # Cancer
        return (
            f"The model detects potential **cancerous** findings with {confidence:.1%} confidence.\n"
            f"The suspicious region is concentrated in the {focus_region}.\n\n"
            f"⚠ Clinical Note: Malignant lesions may have irregular shapes or growth patterns.\n"
            f"Consult a radiologist or pulmonologist for follow-up.\n\n"
            f"Probability distribution:\n"
            f"- Normal: {probabilities[0]:.1%}\n"
            f"- Cancer: {probabilities[1]:.1%}"
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = predict_with_explanation(sys.argv[1])
        print("\nPrediction:", result[0])
        print("Confidence:", f"{result[1]:.1%}")
    else:
        print("Usage: python predict.py <image_path>")
