import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.cnn_model import LungCancerCNN
from utils.data_loader import get_loaders
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Lung Cancer Detection Model')
    parser.add_argument('--config', type=str, default='configs/eval.yaml', help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = LungCancerCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    # Get test loader
    _, _, test_loader = get_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['image_size'],
        num_workers=config['num_workers']
    )

    # Grad-CAM setup
    target_layers = model.get_cam_target_layers()
    cam = GradCAM(model=model, target_layers=target_layers)

    all_preds = []
    all_probs = []
    all_labels = []
    results = []

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'heatmaps'), exist_ok=True)

    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass (no gradient)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Grad-CAM on first batch (requires gradient)
        if batch_idx == 0:
            targets = [ClassifierOutputTarget(pred.item()) for pred in preds]
            grayscale_cams = cam(input_tensor=images, targets=targets)

            for i, (image, label, pred, prob, cam_img) in enumerate(zip(
                images.cpu(), labels.cpu(), preds.cpu(), probs.cpu(), grayscale_cams
            )):
                rgb_img = image.permute(1, 2, 0).numpy()
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                heatmap = show_cam_on_image(rgb_img, cam_img, use_rgb=True, image_weight=0.5)

                img_name = f"sample_{batch_idx}_{i}.png"
                img_path = os.path.join(config['output_dir'], 'heatmaps', img_name)
                plt.imsave(img_path, heatmap)

                results.append({
                    'image': img_name,
                    'true_label': test_loader.dataset.classes[label],
                    'predicted_label': test_loader.dataset.classes[pred],
                    'confidence': f"{prob[pred]:.4f}",
                    'heatmap_path': img_path,
                    'prob_normal': f"{prob[0]:.4f}",
                    'prob_cancer': f"{prob[1]:.4f}"
                })

    # Metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(
        all_labels, all_preds,
        target_names=test_loader.dataset.classes,
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(config['output_dir'], 'confusion_matrix.png'))
    plt.close()

    # ROC AUC for cancer (class 1)
    print("\n=== ROC AUC Score ===")
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    print(f"Binary ROC AUC (Cancer): {auc:.4f}")

    # Save predictions
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config['output_dir'], 'predictions.csv'), index=False)
    print(f"\n✅ Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()
