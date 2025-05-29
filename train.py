import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from models.cnn_model import LungCancerCNN
from utils.data_loader import get_loaders
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train Lung Cancer Detection Model')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load config
    args = parse_args()
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Initialize TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(config['log_dir'], timestamp))
    
    # Data loaders
    train_loader, val_loader, _ = get_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['image_size'],
        num_workers=config['num_workers']
    )
    
    # Model
    model = LungCancerCNN(num_classes=2).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] * len(train_loader),
        eta_min=config['min_lr']
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
    
    # Training loop
    best_val_acc = 0.0
    early_stop_counter = 0
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=config['use_amp']):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device, config['use_amp'])
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(config['model_dir'], 'best_model.pth'))
            print(f"✅ New best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= config['early_stop_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Cleanup
    writer.close()
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")

def validate(model, val_loader, criterion, device, use_amp=False):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return val_loss, val_acc

if __name__ == "__main__":
    main()