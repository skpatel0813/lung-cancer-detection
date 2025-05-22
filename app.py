import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.cnn_model import LungCancerCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LungCancerCNN().to(device)
model.load_state_dict(torch.load('lung_cancer_cnn.pth', map_location=device))
model.eval()

# Grad-CAM setup
target_layer = model.conv3
cam = GradCAM(model=model, target_layers=[target_layer])

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(image):
    # Preprocess
    pil_img = image.convert("RGB")
    input_tensor = transform(pil_img.convert("L")).unsqueeze(0).to(device)

    # Prediction
    output = model(input_tensor)
    pred = torch.argmax(output, 1).item()
    prob = torch.softmax(output, 1)[0][pred].item()

    # Grad-CAM
    np_image = np.array(pil_img.resize((128, 128))) / 255.0
    np_image = np.float32(np_image)
    cam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])[0]
    cam_img = show_cam_on_image(np_image, cam_map, use_rgb=True)

    cam_img = Image.fromarray(cam_img)
    label = "Cancer" if pred == 1 else "Normal"
    return label + f" ({prob:.2f})", cam_img

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=["text", "image"],
    title="Lung Cancer Detector (X-ray/CAM)",
    description="Upload a chest X-ray. Model predicts if cancer is present and shows heatmap."
)

iface.launch()
