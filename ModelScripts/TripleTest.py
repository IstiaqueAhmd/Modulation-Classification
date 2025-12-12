"""
Stable training script for TRIPLE-stream scalogram CNN.
Key improvements:
 - Supports 3-channel input (e.g., I, Q, Magnitude or R,G,B)
 - Triple independent streams fused at the end
 - Robust dataset handling for variable channel counts
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# -------------------------
# 1) Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------
# 2) Dataset (Updated for 3 Channels)
# -------------------------
class ScalogramDataset(Dataset):
    """
    Expects .npy files. 
    __getitem__ returns (tensor: [3, H, W], label:int)
    """
    def __init__(self, root_dir, transform=None, indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.all_data = []

        # deterministically gather files per class (sorted)
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            for f in files:
                self.all_data.append((os.path.join(class_dir, f), label))

        if indices is not None:
            self.data = [self.all_data[i] for i in indices]
        else:
            self.data = self.all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        try:
            arr = np.load(file_path)  # Expects (H, W, C)
            
            # --- Robust Channel Handling for 3 Streams ---
            if arr.ndim == 2:
                # (H, W) -> (H, W, 3) (duplicate 3 times)
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 1:
                # (H, W, 1) -> (H, W, 3)
                arr = np.concatenate([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 2:
                # (H, W, 2) -> (H, W, 3)
                # If only 2 channels exist, we append a zero channel for the 3rd stream
                zeros = np.zeros_like(arr[:, :, 0:1])
                arr = np.concatenate([arr, zeros], axis=-1)
            
            # If shape is already (H, W, 3), we leave it alone.
            # If shape > 3, we take first 3.
            if arr.shape[-1] > 3:
                arr = arr[:, :, :3]
            
            # transpose to (C, H, W) -> Result is [3, H, W]
            tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
            
            if self.transform:
                tensor = self.transform(tensor)
            return tensor, label
            
        except Exception as e:
            print(f"[WARN] Error loading {file_path}: {e}. Returning zeros.")
            # Return dummy 3-channel tensor
            return torch.zeros((3, 224, 224), dtype=torch.float32), label

# -------------------------
# 3) Utilities: split & stats
# -------------------------
def create_randomized_split(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    global_train, global_val, global_test = [], [], []
    current_idx = 0

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        n = len(files)
        if n == 0:
            current_idx += 0
            continue
        indices = list(range(n))
        random.shuffle(indices)
        t_end = int(n * train_ratio)
        v_end = int(n * (train_ratio + val_ratio))
        # map to global indices
        global_train.extend([current_idx + i for i in indices[:t_end]])
        global_val.extend([current_idx + i for i in indices[t_end:v_end]])
        global_test.extend([current_idx + i for i in indices[v_end:]])
        current_idx += n

    return global_train, global_val, global_test

def compute_dataset_stats(dataset, batch_size=64, num_workers=2):
    """
    Compute per-channel mean and std using sum and sum-of-squares.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    channel_sum = None
    channel_sumsq = None
    total_pixels = 0

    print("Computing dataset mean and std...")
    for data, _ in tqdm(loader, desc="Stats"):
        # data shape: [B, C, H, W] (C will be 3)
        b, c, h, w = data.shape
        pixels = b * h * w
        if channel_sum is None:
            channel_sum = data.sum(dim=(0, 2, 3))
            channel_sumsq = (data * data).sum(dim=(0, 2, 3))
        else:
            channel_sum += data.sum(dim=(0, 2, 3))
            channel_sumsq += (data * data).sum(dim=(0, 2, 3))
        total_pixels += pixels

    mean = channel_sum / total_pixels
    mean_sq = channel_sumsq / total_pixels
    var = mean_sq - mean * mean
    # Numerical safety: clip var >= eps
    eps = 1e-6
    std = torch.sqrt(torch.clamp(var, min=eps))

    mean_list = mean.tolist()
    std_list = std.tolist()
    print(f"Computed mean: {mean_list}, std: {std_list}")
    return mean_list, std_list

# -------------------------
# 4) Model architecture (Triple Stream)
# -------------------------
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),   # default momentum
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

def make_stream():
    # Expects 1 channel input per stream
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2),
        DSConv(16, 32),
        nn.MaxPool2d(3, 2),
        DSConv(32, 64),
        DSConv(64, 64),
        DSConv(64, 32),
        nn.MaxPool2d(3, 2)
    )

class TripleStreamCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        # Three independent streams
        self.stream1 = make_stream()
        self.stream2 = make_stream()
        self.stream3 = make_stream()
        
        # Fusion: 3 streams * 32 channels each = 96 channels
        self.fuse = nn.Sequential(
            DSConv(96, 64), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        # Split into 3 individual channels. Each slice is [B, 1, H, W]
        x1 = self.stream1(x[:, 0:1]) 
        x2 = self.stream2(x[:, 1:2])
        x3 = self.stream3(x[:, 2:3])
        
        # Concatenate: [B, 32, h, w] * 3 -> [B, 96, h, w]
        x = torch.cat([x1, x2, x3], dim=1) 
        x = self.fuse(x)                 # [B, 64]
        return self.classifier(x)

# -------------------------
# 5) Training / evaluation loop (main)
# -------------------------
if __name__ == "__main__":
    # Configuration - adjust as needed
    TRAIN = True
    SNR = "30"
    BATCH_SIZE = 128
    EPOCHS = 50
    DATA_DIR = f"Dataset/Scalograms/snr_{SNR}"
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1) Create splits
    print("Creating randomized train/val/test splits ...")
    train_indices, val_indices, test_indices = create_randomized_split(DATA_DIR, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # 2) Compute normalization stats using training subset (Will now compute for 3 channels)
    temp_dataset = ScalogramDataset(DATA_DIR, transform=None, indices=train_indices)
    norm_mean, norm_std = compute_dataset_stats(temp_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 3) Define transforms (augmentations BEFORE Normalize)
    train_transform = transforms.Compose([
        # mild spatial jitter
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.RandomErasing(p=0.25, scale=(0.01, 0.08)),
        transforms.Normalize(norm_mean, norm_std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 4) Datasets and loaders
    train_dataset = ScalogramDataset(DATA_DIR, transform=train_transform, indices=train_indices)
    val_dataset = ScalogramDataset(DATA_DIR, transform=val_test_transform, indices=val_indices)
    test_dataset = ScalogramDataset(DATA_DIR, transform=val_test_transform, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # 5) Model, optimizer, scheduler, loss
    num_classes = len(train_dataset.classes)
    
    # Initialize TripleStreamCNN
    model = TripleStreamCNN(num_classes=num_classes).to(DEVICE)
    
    Number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {Number_of_parameters:,} parameters.")
    # Optimizer with smaller weight decay
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # Use ReduceLROnPlateau for more conservative LR changes
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 6) Training loop with early stopping
    if TRAIN:
        best_val_acc = 0.0
        patience = 10
        counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"Starting training for {EPOCHS} epochs ...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
            for inputs, labels in loop:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

            train_loss = running_loss / total
            train_acc = correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_loss = val_loss / total if total > 0 else 0.0
            val_acc = correct / total if total > 0 else 0.0

            # Scheduler step with metric
            scheduler.step(val_acc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Checkpoint & early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), f"best_model_snr{SNR}.pth")
                print(f"New best val acc {best_val_acc:.4f} -> saved model.")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                    break

        # Plot training curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Val')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig('training_curves.png')
        print("Saved training_curves.png")

    # 7) Evaluation (load best model if available)
    model_path = f"best_model_snr{SNR}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded best model from {model_path} for evaluation.")

    model.eval()
    all_preds = []
    all_labels = []
    print("Running evaluation on test set ...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes, digits=4))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title(f'Confusion Matrix SNR {SNR}')
    plt.tight_layout()
    plt.savefig(f'Confusion_Matrix_snr{SNR}.png')
    print(f"Saved Confusion_Matrix_snr{SNR}.png")
    print("Done.")