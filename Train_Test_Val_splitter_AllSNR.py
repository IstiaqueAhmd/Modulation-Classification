import os
import random
import shutil
from collections import defaultdict

"""
This script splits scalograms from all SNR levels into train/test/val sets.
It ensures equal distribution of each SNR level within each class.
"""

# Configuration
SNR_LEVELS = [30, 20, 10, 0, -10, -20]
OUTPUT_BASE = "Dataset(Splitted)/combined"

# Create output directories
train_dir = os.path.join(OUTPUT_BASE, "train")
test_dir = os.path.join(OUTPUT_BASE, "test")
val_dir = os.path.join(OUTPUT_BASE, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# All classes
classes = [
  "OOK", "4ASK", "8ASK",
  "BPSK", "QPSK", "8PSK", 
  "16APSK", "64QAM", 
  "AM-SSB-WC","AM-DSB-WC",
  "FM", "GMSK", "OQPSK"
]
# Set the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

print("Starting to process all SNR levels...")
print(f"SNR Levels: {SNR_LEVELS}")
print(f"Classes: {len(classes)}")
print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}\n")

# Statistics tracking
total_stats = defaultdict(lambda: defaultdict(int))

for class_name in classes:
    print(f"\nProcessing class: {class_name}")
    print("-" * 60)
    
    # Create directories for this class in each split
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # Collect files from all SNR levels for this class
    all_files_by_snr = defaultdict(list)
    
    for snr in SNR_LEVELS:
        base_path = f"Scalograms/snr_{snr}"
        class_path = os.path.join(base_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"  Warning: Class path does not exist: {class_path}")
            continue
        
        # List all scalogram files in the class folder
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.npy', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.mat'))]
        all_files_by_snr[snr] = files
        print(f"  SNR {snr:3d}: Found {len(files)} files")
    
    # Split each SNR level separately to ensure equal distribution
    class_train_count = 0
    class_val_count = 0
    class_test_count = 0
    
    for snr, files in all_files_by_snr.items():
        if len(files) == 0:
            continue
        
        # Shuffle the files for this SNR level
        random.shuffle(files)
        
        # Calculate split indices
        total_files = len(files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        
        # Create train, val, and test lists
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]
        
        # Copy files to their respective folders
        base_path = f"Scalograms/snr_{snr}"
        class_path = os.path.join(base_path, class_name)
        
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(train_dir, class_name, file)
            shutil.copy(src, dst)
            class_train_count += 1
        
        for file in val_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(val_dir, class_name, file)
            shutil.copy(src, dst)
            class_val_count += 1
        
        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(test_dir, class_name, file)
            shutil.copy(src, dst)
            class_test_count += 1
        
        total_stats[class_name][f'snr_{snr}_train'] = len(train_files)
        total_stats[class_name][f'snr_{snr}_val'] = len(val_files)
        total_stats[class_name][f'snr_{snr}_test'] = len(test_files)
    
    print(f"  Total - Train: {class_train_count}, Val: {class_val_count}, Test: {class_test_count}")
    total_stats[class_name]['total_train'] = class_train_count
    total_stats[class_name]['total_val'] = class_val_count
    total_stats[class_name]['total_test'] = class_test_count

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

overall_train = 0
overall_val = 0
overall_test = 0

for class_name in classes:
    if class_name in total_stats:
        train = total_stats[class_name]['total_train']
        val = total_stats[class_name]['total_val']
        test = total_stats[class_name]['total_test']
        total = train + val + test
        
        overall_train += train
        overall_val += val
        overall_test += test
        
        print(f"{class_name:15s} - Train: {train:4d}, Val: {val:4d}, Test: {test:4d}, Total: {total:4d}")

print("-" * 80)
print(f"{'OVERALL':15s} - Train: {overall_train:4d}, Val: {overall_val:4d}, Test: {overall_test:4d}, Total: {overall_train + overall_val + overall_test:4d}")
print("=" * 80)
print(f"\nDataset split complete! Files saved to: {OUTPUT_BASE}")
