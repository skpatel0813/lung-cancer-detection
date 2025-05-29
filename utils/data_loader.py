import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import pydicom
from PIL import Image

class MedicalImageDataset(Dataset):
    """Dataset class for normal vs cancer images with DICOM support"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'cancer']  # Binary classification
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls]))

        print(f"[INFO] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            if path.lower().endswith('.dcm'):
                ds = pydicom.dcmread(path)
                img = ds.pixel_array
                img = self.apply_window(img, -600, 1500)
                img = (img - img.min()) / (img.max() - img.min()) * 255
                img = Image.fromarray(img.astype('uint8'))
            else:
                img = Image.open(path)

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"[ERROR] Skipping {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

    @staticmethod
    def apply_window(image, window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed = np.clip(image, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min)
        return windowed

def get_transforms(mode='train', img_size=384):
    if mode == 'train':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_loaders(data_root, batch_size=32, img_size=384, num_workers=4):
    """Get train, val, test DataLoaders assuming already split folders"""
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    train_ds = MedicalImageDataset(train_dir, transform=get_transforms('train', img_size))
    val_ds = MedicalImageDataset(val_dir, transform=get_transforms('val', img_size))
    test_ds = MedicalImageDataset(test_dir, transform=get_transforms('test', img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
