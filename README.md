# Lung Cancer Detection with Explainable AI

This project implements a high-performance deep learning pipeline for binary classification of lung histopathology images as either Normal or Cancer. It combines Convolutional Neural Networks (CNNs) with Grad-CAM-based visual explanations and provides an interactive diagnostic interface using Gradio.

## Dataset

Source: [Lung and Colon Cancer Histopathological Images – Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

- Over 25,000 histopathology images
- Multiple cancer types (lung, colon)
- This project uses a binary subset: Normal vs Lung Cancer
- Images preprocessed into balanced training, validation, and test sets

## Model Architecture

- Custom CNN built in PyTorch with:
  - Residual blocks
  - Squeeze-and-Excitation attention
  - SiLU activations
- Input size: 384×384 grayscale (converted to 3-channel)
- Output: 2-class softmax (Normal / Cancer)

## Performance

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | 99.83%    |
| Precision     | 99.88%    |
| Recall        | 99.88%    |
| F1 Score      | 99.88%    |
| ROC AUC       | 0.9999    |

Evaluated on a held-out test set of 2,400 images. See `outputs/predictions.csv` for full per-image results.

## Explainability

- Grad-CAM (HiResCAM) used to visualize attention
- Heatmaps generated for each test sample
- Each prediction includes:
  - Confidence score
  - Localized heatmap overlay
  - Human-readable explanation

Example output:

“The model detects cancer-related abnormalities with 99.7% confidence (high concern). The most suspicious area is in the middle right region. Please consult a pulmonologist for follow-up.”

## Interactive Demo

The diagnostic tool is implemented using Gradio and can be run locally:
