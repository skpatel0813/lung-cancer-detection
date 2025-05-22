import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.cnn_model import LungCancerCNN

# ------------------ Configuration ------------------
DATA_PATH = 'data/test'
OUTPUT_DIR = 'outputs'
HEATMAP_DIR = os.path.join(OUTPUT_DIR, 'heatmaps')
CSV_PATH = os.path.join(OUTPUT_DIR, 'predictions.csv')

os.makedirs(HEATMAP_DIR, exist_ok=True)

# ------------------ Device ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------ Data Transforms ------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------------------ Load Test Data ------------------
test_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# ------------------ Load Model ------------------
model = LungCancerCNN().to(device)
model.load_state_dict(torch.load('lung_cancer_cnn.pth', map_location=device))
model.eval()

# ------------------ Grad-CAM Setup ------------------
target_layer = model.conv3
cam = GradCAM(model=model, target_layers=[target_layer])  # for compatibility

# ------------------ Evaluation Loop ------------------
all_preds, all_labels = [], []
rows = []

print("\nEvaluating and generating Grad-CAMs...\n")

for i, (image_tensor, label_tensor) in enumerate(tqdm(test_loader)):
    image_tensor = image_tensor.to(device)
    label = label_tensor.item()
    path, _ = test_dataset.samples[i]
    file_name = os.path.basename(path)

    # Model inference
    output = model(image_tensor)
    pred_class = torch.argmax(output, 1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()
    prediction_name = idx_to_class[pred_class]
    true_name = idx_to_class[label]

    all_preds.append(pred_class)
    all_labels.append(label)

    # Raw image for CAM visualization
    raw_image = Image.open(path).convert("RGB").resize((128, 128))
    raw_image_np = np.array(raw_image) / 255.0
    raw_image_np = np.float32(raw_image_np)

    # Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    cam_image = show_cam_on_image(raw_image_np, grayscale_cam, use_rgb=True)

    # Determine most activated region
    h, w = grayscale_cam.shape
    max_y, max_x = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    zone_x = "left" if max_x < w / 3 else "center" if max_x < 2 * w / 3 else "right"
    zone_y = "upper" if max_y < h / 3 else "middle" if max_y < 2 * h / 3 else "lower"
    focus_zone = f"{zone_y}-{zone_x} region"

    # Save CAM overlay
    cam_save_path = os.path.join(HEATMAP_DIR, f"{file_name}_cam.jpg")
    plt.imsave(cam_save_path, cam_image)

    # ✅ Human-readable explanation (FIXED: label check uses lowercase)
    if prediction_name.lower() == 'cancer':
        explanation = (
            f"The model predicted **Cancer** with high confidence ({confidence:.2f}). "
            f"Strong visual activation was observed in the {focus_zone}, which may indicate abnormal tissue presence."
        )
    else:
        explanation = (
            f"The model predicted **Normal** with high confidence ({confidence:.2f}). "
            f"Model attention was mostly on the {focus_zone}, but no abnormal patterns were identified."
        )

    # Console output
    print(f"{file_name} → Predicted: {prediction_name}, True: {true_name}, Confidence: {confidence:.2f}")
    print(f"🧠 Explanation: {explanation}\n")

    # Save to CSV row
    rows.append({
        'image': file_name,
        'true_label': true_name,
        'predicted_label': prediction_name,
        'confidence': f"{confidence:.4f}",
        'focus_zone': focus_zone,
        'heatmap_path': cam_save_path,
        'explanation': explanation
    })

# ------------------ Save CSV Report ------------------
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# ------------------ Print Final Metrics ------------------
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Cancer"]))

print("=== Confusion Matrix ===")
print(confusion_matrix(all_labels, all_preds))
print(f"\n✅ All results saved to: {OUTPUT_DIR}")
