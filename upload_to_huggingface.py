import os
import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value, Image
from huggingface_hub import HfApi, HfFolder

# --------------- CONFIGURATION ----------------
HF_TOKEN = "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"
REPO_ID = "skpatel0813/lung-cancer-predictions"  # ✅ replace with your Hugging Face username/repo
CSV_PATH = "outputs/predictions.csv"
HEATMAP_DIR = "outputs/heatmaps"
# ---------------------------------------------

# Step 1: Authenticate
HfFolder.save_token(HF_TOKEN)

# Step 2: Load CSV
df = pd.read_csv(CSV_PATH)

# Step 3: Convert heatmap path to full path (for HF Image support)
df["image"] = df["heatmap_path"].apply(lambda p: os.path.abspath(p))

# Step 4: Keep only useful columns
hf_df = df[["image", "true_label", "predicted_label", "confidence", "explanation"]]

# Step 5: Define dataset features
features = Features({
    "image": Image(),  # Special Hugging Face image column
    "true_label": Value("string"),
    "predicted_label": Value("string"),
    "confidence": Value("string"),
    "explanation": Value("string"),
})

# Step 6: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(hf_df, features=features)

# Step 7: Push to Hugging Face
dataset.push_to_hub(REPO_ID)

print(f"✅ Upload complete! View at: https://huggingface.co/datasets/{REPO_ID}")
