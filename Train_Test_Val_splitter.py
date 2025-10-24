import os
import random
import shutil
from PIL import Image

# Set the SNR
snr = "30"

# Set the paths for the dataset
base_path = f"Dataset/snr_{snr}"
output_base = f"Dataset(Splitted)/snr_{snr}"

# Create output directories
train_dir = os.path.join(output_base, "train")
test_dir = os.path.join(output_base, "test")
val_dir = os.path.join(output_base, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Adjust these paths and classes
classes = ['32PSK','16APSK','32QAM','FM','GMSK','32APSK','OQPSK',
 '8ASK','BPSK','8PSK','AM-SSB-SC','4ASK','16PSK','64APSK','128QAM',
 '128APSK','AM-DSB-SC','AM-SSB-WC','64QAM','QPSK','256QAM','AM-DSB-WC',
 'OOK','16QAM']

# Set the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

for class_name in classes:
    class_path = os.path.join(base_path, class_name)
    if not os.path.exists(class_path):
        print(f"Class path does not exist: {class_path}")
        continue

    # List all images in the class folder
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff','.mat','.npy'))]

    # Shuffle the images
    random.shuffle(images)

    # Calculate split indices
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    # Create train, val, and test lists
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Create directories for this class in each split
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move images to their respective folders
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

    print(f"Processed class: {class_name}")
    print(f"Train: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_images)}")
