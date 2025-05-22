import torch
import torch.nn as nn
import torch.nn.functional as F

class LungCancerCNN(nn.Module):
    def __init__(self):
        super(LungCancerCNN, self).__init__()

        # Convolutional Layer Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Assuming input images are resized to 128x128
        self.fc2 = nn.Linear(256, 2)  # 2 output classes: Normal and Cancer

    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 128)

        x = self.pool1(F.relu(self.conv1(x)))  # -> (batch_size, 32, 64, 64)
        x = self.pool2(F.relu(self.conv2(x)))  # -> (batch_size, 64, 32, 32)
        x = self.pool3(F.relu(self.conv3(x)))  # -> (batch_size, 128, 16, 16)

        x = x.view(-1, 128 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits (before softmax)

        return x
