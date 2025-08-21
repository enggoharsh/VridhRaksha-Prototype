import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from math import floor

# === CONFIGURATION ===
SOURCE_DIR = r"C:\users\harsh\Downloads\TrainingData\TrainingData"   # Folder with class folders (e.g., /fall, /walk/)
TARGET_DIR = r"C:\users\harsh\Downloads\TrainingData\dataset_split"  # Output folder

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Ratios must sum to 1.0"

def split_files(files):
    random.shuffle(files)
    total = len(files)

    n_train = floor(total * TRAIN_RATIO)
    n_val = floor(total * VAL_RATIO)
    n_test = total - n_train - n_val  # Ensure all files are used

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def segregate_data():
    random.seed(42)  # For reproducible splits
    class_dirs = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    for cls in tqdm(class_dirs, desc="Segregating classes"):
        src_cls_path = os.path.join(SOURCE_DIR, cls)
        files = [f for f in os.listdir(src_cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        train_files, val_files, test_files = split_files(files)

        for split, files_to_copy in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            dst_path = os.path.join(TARGET_DIR, split, cls)
            make_dir(dst_path)

            for f in files_to_copy:
                shutil.copy(os.path.join(src_cls_path, f), os.path.join(dst_path, f))

        # Optional: print counts
        print(f"[{cls}] Total: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print("✅ Dataset segregation completed successfully.")

if __name__ == "__main__":
    segregate_data()
