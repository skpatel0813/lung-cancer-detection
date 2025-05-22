import os
import shutil
import random

def split_data(src_dir, train_dir, test_dir, split_ratio=0.8):
    for cls in ['cancer', 'normal']:
        src = os.path.join(src_dir, cls)
        train_dst = os.path.join(train_dir, cls)
        test_dst = os.path.join(test_dir, cls)
        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(test_dst, exist_ok=True)

        files = os.listdir(src)
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)

        for i, f in enumerate(files):
            src_file = os.path.join(src, f)
            if i < split_idx:
                shutil.copy(src_file, os.path.join(train_dst, f))
            else:
                shutil.copy(src_file, os.path.join(test_dst, f))

split_data("temp_split", "data/train", "data/test")
