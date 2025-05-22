import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import LungCancerCNN
from utils.data_loader import get_loaders
from tqdm import tqdm
import os

# Set training configuration
DATA_DIR = 'data'
MODEL_PATH = 'lung_cancer_cnn.pth'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
train_loader, _ = get_loaders(DATA_DIR, batch_size=BATCH_SIZE)

# Initialize model
model = LungCancerCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("Starting training...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nTraining complete. Model saved to: {MODEL_PATH}")
