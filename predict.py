import torch
from torchvision import transforms
from models.cnn_model import LungCancerCNN
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
model = LungCancerCNN().to(device)
model.load_state_dict(torch.load('lung_cancer_cnn.pth', map_location=device))
model.eval()

def predict_with_explanation(image_path):
    # Load image
    image = Image.open(image_path).convert('L')
    transformed_image = transform(image).unsqueeze(0).to(device)  # Shape: (1, 1, 128, 128)

    # Convert to 3-channel image for Grad-CAM
    rgb_image = image.resize((128, 128)).convert("RGB")
    rgb_image_np = np.array(rgb_image) / 255.0
    rgb_image_np = np.float32(rgb_image_np)

    # Prediction
    with torch.no_grad():
        output = model(transformed_image)
        _, pred_class = torch.max(output, 1)
        pred_idx = pred_class.item()

    # Grad-CAM setup
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import preprocess_image

    # Use the last conv layer for Grad-CAM
    target_layer = model.conv3

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=transformed_image, targets=targets)[0]  # Shape: (128, 128)

    # Overlay CAM
    visualization = show_cam_on_image(rgb_image_np, grayscale_cam, use_rgb=True)

    # Show result
    plt.imshow(visualization)
    plt.title(f"Prediction: {'Cancer' if pred_idx == 1 else 'Normal'}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return 'Cancer' if pred_idx == 1 else 'Normal'

# Example:
# print(predict_with_explanation('path/to/image.jpg'))
