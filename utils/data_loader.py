import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_loaders(data_dir, batch_size=32, shuffle_train=True, shuffle_test=False):
    """
    Loads training and testing data from a specified directory.
    
    The directory structure should be:
    data/
    ├── train/
    │   ├── cancer/
    │   └── normal/
    └── test/
        ├── cancer/
        └── normal/

    Args:
        data_dir (str): Base directory containing 'train' and 'test' folders.
        batch_size (int): Number of samples per batch.
        shuffle_train (bool): Whether to shuffle the training data.
        shuffle_test (bool): Whether to shuffle the test data.

    Returns:
        train_loader, test_loader: PyTorch DataLoaders for training and testing.
    """

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Grayscale(),            # Ensure image is single channel
        transforms.Resize((128, 128)),     # Resize all images to 128x128
        transforms.ToTensor(),             # Convert to tensor (0-1)
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Training dataset
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)

    # Testing dataset
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, test_loader
