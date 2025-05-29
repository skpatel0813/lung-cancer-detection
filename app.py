import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from models.cnn_model import LungCancerCNN
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
CLASS_NAMES = ['Normal', 'Cancer']
MODEL_PATH = 'models/best_model.pth'
TARGET_SIZE = 384

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LungCancerCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Grad-CAM setup
target_layer = model.block4[0]
cam = HiResCAM(model=model, target_layers=[target_layer])

# Preprocessing transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def predict(image, threshold=0.5):
    try:
        pil_img = preprocess_image(image)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        if confidence < threshold:
            return "Uncertain - Please consult a specialist", None, ""

        # Grad-CAM
        rgb_img = np.array(pil_img.resize((TARGET_SIZE, TARGET_SIZE))) / 255.0
        rgb_img = np.float32(rgb_img)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5)

        label = CLASS_NAMES[pred_idx]
        explanation = generate_explanation(label, confidence, grayscale_cam)

        return f"{label} (Confidence: {confidence:.2%})", cam_img, explanation

    except Exception as e:
        return f"Error: {str(e)}", None, ""

def generate_explanation(label, confidence, heatmap):
    h, w = heatmap.shape
    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    region_x = "left" if max_x < w/3 else "center" if max_x < 2*w/3 else "right"
    region_y = "upper" if max_y < h/3 else "middle" if max_y < 2*h/3 else "lower"

    if label == 'Normal':
        return (
            f"The model predicts a normal scan with {confidence:.0%} confidence. "
            f"Attention was focused on the {region_y} {region_x} region, showing typical lung patterns."
        )
    else:
        severity = "moderate concern" if confidence < 0.8 else "high concern"
        return (
            f"The model detects cancer-related abnormalities with {confidence:.0%} confidence ({severity}). "
            f"The most suspicious area is in the {region_y} {region_x} region (highlighted). "
            "Please consult a pulmonologist for further evaluation."
        )

# Gradio Interface
with gr.Blocks(title="Lung Cancer Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Lung Cancer Detection System")
    gr.Markdown("Upload a chest X-ray or CT scan to analyze for possible lung cancer.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Medical Image")
            threshold = gr.Slider(0.1, 0.9, value=0.5, label="Confidence Threshold")
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            label_output = gr.Label(label="Diagnosis")
            heatmap_output = gr.Image(label="Attention Heatmap")
            explanation_output = gr.Markdown(label="Explanation")

    examples = gr.Examples(
        examples=["examples/normal_1.jpg", "examples/cancer_1.jpg"],
        inputs=image_input
    )

    submit_btn.click(
        fn=predict,
        inputs=[image_input, threshold],
        outputs=[label_output, heatmap_output, explanation_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)  # Set share=True to get a public link
