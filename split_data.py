import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def split_dataset(src_dir, dest_dir, test_size=0.2, val_size=0.1, random_state=42):
    """Enhanced data splitting with balanced validation set"""
    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    for class_dir in class_dirs:
        src_class_path = os.path.join(src_dir, class_dir)
        files = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
        
        # First split to separate test set
        train_val_files, test_files = train_test_split(
            files, test_size=test_size, random_state=random_state
        )
        
        # Then split train_val into train and validation
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Create destination directories
        os.makedirs(os.path.join(dest_dir, 'train', class_dir), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'val', class_dir), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'test', class_dir), exist_ok=True)
        
        # Copy files with progress bar
        print(f"\nProcessing class: {class_dir}")
        for file in tqdm(train_files, desc="Copying train files"):
            shutil.copy(
                os.path.join(src_class_path, file),
                os.path.join(dest_dir, 'train', class_dir, file)
            )
        
        for file in tqdm(val_files, desc="Copying val files"):
            shutil.copy(
                os.path.join(src_class_path, file),
                os.path.join(dest_dir, 'val', class_dir, file)
            )
        
        for file in tqdm(test_files, desc="Copying test files"):
            shutil.copy(
                os.path.join(src_class_path, file),
                os.path.join(dest_dir, 'test', class_dir, file)
            )

def main():
    parser = argparse.ArgumentParser(description='Split medical image dataset')
    parser.add_argument('--config', type=str, default='configs/split.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("Splitting dataset with configuration:")
    print(f"Source Directory: {config['source_dir']}")
    print(f"Destination Directory: {config['dest_dir']}")
    print(f"Test Size: {config['test_size']}")
    print(f"Validation Size: {config['val_size']}")
    
    split_dataset(
        src_dir=config['source_dir'],
        dest_dir=config['dest_dir'],
        test_size=config['test_size'],
        val_size=config['val_size'],
        random_state=config['random_state']
    )
    
    print("\n✅ Dataset splitting completed successfully!")

if __name__ == "__main__":
    main()