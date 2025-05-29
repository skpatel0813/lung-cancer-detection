import os
import pandas as pd
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi, create_repo
import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_hf_dataset(df, repo_id, config):
    """Create and upload dataset with Grad-CAM images and predictions"""
    df = df.copy()

    # Point the 'image' column to the heatmap image file paths
    df["image"] = df["heatmap_path"]

    # Define schema matching the CSV exactly
    features = Features({
        "image": HFImage(),
        "true_label": Value("string"),
        "predicted_label": Value("string"),
        "confidence": Value("string"),
        "focus_zone": Value("string"),
        "heatmap_path": Value("string"),
        "explanation": Value("string")
    })

    # Convert to HF dataset
    dataset = Dataset.from_pandas(df, features=features)

    # Push to Hugging Face Hub
    dataset.push_to_hub(
        repo_id=repo_id,
        split="train",
        private=config['private'],
        token=config['hf_token']
    )

    return dataset

def main():
    # Load config
    config = load_config('configs/huggingface.yaml')
    api = HfApi()

    # Attempt to create repo if it doesn't exist
    try:
        create_repo(
            repo_id=config['repo_id'],
            repo_type="dataset",
            private=config['private'],
            token=config['hf_token']
        )
    except Exception as e:
        print(f"[INFO] Repo already exists or could not be created: {e}")

    # Load predictions CSV
    if not os.path.exists(config['predictions_path']):
        print(f"[ERROR] Predictions file not found: {config['predictions_path']}")
        return

    df = pd.read_csv(config['predictions_path'])

    if df.empty:
        print("[ERROR] CSV is empty.")
        return

    print("📦 Creating and uploading dataset to Hugging Face...")
    dataset = create_hf_dataset(df, config['repo_id'], config)

    print(f"\n✅ Upload complete! View it at: https://huggingface.co/datasets/{config['repo_id']}")

if __name__ == "__main__":
    main()
