"""
Stable training script for dual-stream scalogram CNN.
Key improvements:
 - Correct order of transforms (augmentations before normalization)
 - Robust dataset mean/std computation (sum & sumsq)
 - Restored BatchNorm defaults (no very-low momentum)
 - Reduced weight decay and switched to ReduceLROnPlateau scheduler
 - Added mild augmentations to improve generalization
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
# 2) Balanced SNR Batch Sampler
# -------------------------
class BalancedSNRBatchSampler:
    """
    Custom batch sampler that ensures each batch contains equal samples from each SNR.
    This helps the model learn robustly across all noise conditions.
    """
    def __init__(self, snr_indices_dict, batch_size, drop_last=True):
        """
        Args:
            snr_indices_dict: Dict mapping SNR name to list of indices for that SNR
            batch_size: Total batch size (must be divisible by number of SNRs)
            drop_last: Whether to drop incomplete batches
        """
        self.snr_names = sorted(snr_indices_dict.keys())
        self.snr_indices = {snr: list(indices) for snr, indices in snr_indices_dict.items()}
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_snrs = len(self.snr_names)
        
        if batch_size % self.num_snrs != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by number of SNRs ({self.num_snrs})")
        
        self.samples_per_snr = batch_size // self.num_snrs
        
        # Calculate number of batches
        min_samples = min(len(indices) for indices in self.snr_indices.values())
        self.num_batches = min_samples // self.samples_per_snr
        if not drop_last and min_samples % self.samples_per_snr != 0:
            self.num_batches += 1
    
    def __iter__(self):
        # Shuffle indices for each SNR independently
        shuffled_snr_indices = {}
        for snr in self.snr_names:
            indices = self.snr_indices[snr].copy()
            random.shuffle(indices)
            shuffled_snr_indices[snr] = indices
        
        # Generate batches
        for batch_idx in range(self.num_batches):
            batch = []
            start_idx = batch_idx * self.samples_per_snr
            end_idx = start_idx + self.samples_per_snr
            
            # Sample equally from each SNR
            for snr in self.snr_names:
                snr_batch = shuffled_snr_indices[snr][start_idx:end_idx]
                if len(snr_batch) < self.samples_per_snr and self.drop_last:
                    continue
                batch.extend(snr_batch)
            
            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                yield batch
    
    def __len__(self):
        return self.num_batches

# -------------------------
# 3) Dataset
# -------------------------
class ScalogramDataset(Dataset):
    """
    Expects .npy files shaped (H, W, 2) where channels are last axis.
    __getitem__ returns (tensor: [2, H, W], label:int)
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
            arr = np.load(file_path)  # expected (H, W, 2)
            # If array shape varies, try to handle common cases
            if arr.ndim == 2:
                # single channel -> duplicate
                arr = np.stack([arr, arr], axis=-1)
            elif arr.shape[-1] == 1:
                arr = np.concatenate([arr, arr], axis=-1)
            # transpose to (C, H, W)
            tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
            if self.transform:
                tensor = self.transform(tensor)
            return tensor, label
        except Exception as e:
            print(f"[WARN] Error loading {file_path}: {e}. Returning zeros.")
            return torch.zeros((2, 224, 224), dtype=torch.float32), label

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
    Expects dataset that returns (tensor [C,H,W], label)
    Returns (mean_list, std_list)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    channel_sum = None
    channel_sumsq = None
    total_pixels = 0

    print("Computing dataset mean and std...")
    for data, _ in tqdm(loader, desc="Stats"):
        # data shape: [B, C, H, W]
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
# 4) Model architecture (clean BN defaults)
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

class DualStreamCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.stream_amp = make_stream()
        self.stream_phase = make_stream()
        self.fuse = nn.Sequential(
            DSConv(64, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, 2, H, W]
        xa = self.stream_amp(x[:, 0:1])  # [B, 32, h, w]
        xp = self.stream_phase(x[:, 1:2])
        x = torch.cat([xa, xp], dim=1)   # [B, 64, h, w]
        x = self.fuse(x)                 # [B, 64]
        return self.classifier(x)

# -------------------------
# 5) Training / evaluation loop (main)
# -------------------------
if __name__ == "__main__":
    # Configuration - adjust as needed
    TRAIN = True
    TEST_SNR = "10"  # SNR to use for testing (will be excluded from training)
    BATCH_SIZE = 126  # Must be divisible by number of SNRs (6 SNRs -> 21 samples per SNR)
    EPOCHS = 50
    BASE_DIR = "Dataset/Scalograms"
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1) Determine which SNR folders to use
    # Get all SNR folders - train on ALL SNRs
    all_snr_folders = sorted([d for d in os.listdir(BASE_DIR) 
                              if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith('snr_')])
    
    if not all_snr_folders:
        raise ValueError(f"No SNR folders found in {BASE_DIR}")
    
    # Train on ALL SNRs (including test SNR)
    train_val_snr_folders = all_snr_folders
    train_val_dirs = [os.path.join(BASE_DIR, snr_folder) for snr_folder in train_val_snr_folders]
    
    # Test on specified SNR
    test_snr_folder = f"snr_{TEST_SNR}"
    test_dir = os.path.join(BASE_DIR, test_snr_folder)
    model_suffix = f"all_snrs_test{TEST_SNR}"
    
    print(f"Training on ALL SNRs: {train_val_snr_folders}")
    print(f"Testing on SNR: {TEST_SNR}")

    # 2) Create splits and track SNR-specific indices
    print("Creating randomized train/val/test splits ...")
    
    # For training/validation: combine all specified directories
    all_train_indices = []
    all_val_indices = []
    train_snr_indices = {}  # Maps SNR name to train indices for balanced sampling
    val_snr_indices = {}    # Maps SNR name to val indices
    global_offset = 0
    
    for snr_folder, data_dir in zip(train_val_snr_folders, train_val_dirs):
        train_idx, val_idx, _ = create_randomized_split(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        
        # Offset indices to account for multiple directories
        offset_train_idx = [idx + global_offset for idx in train_idx]
        offset_val_idx = [idx + global_offset for idx in val_idx]
        
        all_train_indices.extend(offset_train_idx)
        all_val_indices.extend(offset_val_idx)
        
        # Store SNR-specific indices for balanced sampling
        train_snr_indices[snr_folder] = offset_train_idx
        val_snr_indices[snr_folder] = offset_val_idx
        
        # Update offset for next directory
        temp_dataset = ScalogramDataset(data_dir, transform=None, indices=None)
        global_offset += len(temp_dataset.all_data)
    
    # For testing: use only the specified test SNR
    _, _, test_indices = create_randomized_split(test_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # 3) Create combined dataset for training/validation
    # Combine datasets from all SNR folders
    combined_train_val_dataset = ScalogramDataset(train_val_dirs[0], transform=None, indices=None)
    for data_dir in train_val_dirs[1:]:
        temp_ds = ScalogramDataset(data_dir, transform=None, indices=None)
        combined_train_val_dataset.all_data.extend(temp_ds.all_data)
    combined_train_val_dataset.data = combined_train_val_dataset.all_data
    
    # Create training subset for normalization
    temp_dataset = ScalogramDataset(train_val_dirs[0], transform=None, indices=None)
    temp_dataset.all_data = combined_train_val_dataset.all_data
    temp_dataset.data = [combined_train_val_dataset.all_data[i] for i in all_train_indices]
    
    # 4) Compute normalization stats using training subset
    norm_mean, norm_std = compute_dataset_stats(temp_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 5) Define transforms (augmentations BEFORE Normalize)
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

    # 6) Datasets and loaders
    # Combined dataset for training/validation
    train_dataset = ScalogramDataset(train_val_dirs[0], transform=train_transform, indices=None)
    train_dataset.all_data = combined_train_val_dataset.all_data
    train_dataset.data = [combined_train_val_dataset.all_data[i] for i in all_train_indices]
    
    val_dataset = ScalogramDataset(train_val_dirs[0], transform=val_test_transform, indices=None)
    val_dataset.all_data = combined_train_val_dataset.all_data
    val_dataset.data = [combined_train_val_dataset.all_data[i] for i in all_val_indices]
    
    # Test dataset uses the specified TEST_SNR
    test_dataset = ScalogramDataset(test_dir, transform=val_test_transform, indices=test_indices)

    # Create balanced batch sampler for training
    train_batch_sampler = BalancedSNRBatchSampler(
        snr_indices_dict=train_snr_indices,
        batch_size=BATCH_SIZE,
        drop_last=True
    )
    
    print(f"Using balanced batch sampler: {BATCH_SIZE // len(train_snr_indices)} samples per SNR per batch")
    
    # DataLoaders - use batch_sampler for training to ensure balanced SNR representation
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # 7) Model, optimizer, scheduler, loss
    num_classes = len(train_dataset.classes)
    model = DualStreamCNN(num_classes=num_classes).to(DEVICE)
    
    Number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {Number_of_parameters:,} parameters (DualStreamCNN).")
    # Optimizer with smaller weight decay
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # Use ReduceLROnPlateau for more conservative LR changes
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 8) Training loop with early stopping
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
                torch.save(model.state_dict(), f"best_model_{model_suffix}.pth")
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
        plt.savefig(f'training_curves_{model_suffix}.png')
        print(f"Saved training_curves_{model_suffix}.png")

    # 9) Evaluation (load best model if available)
    model_path = f"best_model_{model_suffix}.pth"
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
    plt.title(f'Confusion Matrix (Test SNR: {TEST_SNR})')
    plt.tight_layout()
    plt.savefig(f'Confusion_Matrix_{model_suffix}_test{TEST_SNR}.png')
    print(f"Saved Confusion_Matrix_{model_suffix}_test{TEST_SNR}.png")
    print("Done.")
