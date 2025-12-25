"""
Multi-SNR training script for dual-stream scalogram CNN.
Key features:
 - Train on multiple SNR levels for better generalization
 - Test on specific SNR level(s)
 - Separate TRAIN mode - can test without retraining
 - Saves/loads normalization stats for consistent testing
 - Correct order of transforms (augmentations before normalization)
 - Robust dataset mean/std computation (sum & sumsq)
"""

import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------
# 2) Dataset Classes
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
            arr = np.load(file_path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr], axis=-1)
            elif arr.shape[-1] == 1:
                arr = np.concatenate([arr, arr], axis=-1)
            tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
            if self.transform:
                tensor = self.transform(tensor)
            return tensor, label
        except Exception as e:
            print(f"[WARN] Error loading {file_path}: {e}. Returning zeros.")
            return torch.zeros((2, 224, 224), dtype=torch.float32), label


class MultiSnrScalogramDataset(Dataset):
    """
    Dataset that loads scalograms from multiple SNR folders.
    Combines data from all specified SNR levels.
    """
    def __init__(self, base_dir, snr_list, transform=None, split='train', train_ratio=0.8, val_ratio=0.1, seed=42):
        self.base_dir = base_dir
        self.snr_list = snr_list
        self.transform = transform
        self.data = []
        self.classes = None
        
        random.seed(seed)
        
        for snr in snr_list:
            snr_dir = os.path.join(base_dir, f"snr_{snr}")
            if not os.path.exists(snr_dir):
                print(f"[WARN] SNR directory not found: {snr_dir}")
                continue
            
            # Get classes from first valid SNR folder
            if self.classes is None:
                self.classes = sorted([d for d in os.listdir(snr_dir) if os.path.isdir(os.path.join(snr_dir, d))])
            
            # Gather files per class with stratified split
            for label, class_name in enumerate(self.classes):
                class_dir = os.path.join(snr_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
                n = len(files)
                if n == 0:
                    continue
                
                # Create indices and shuffle
                indices = list(range(n))
                random.seed(seed)  # Reset seed for consistent splits per class/snr
                random.shuffle(indices)
                
                t_end = int(n * train_ratio)
                v_end = int(n * (train_ratio + val_ratio))
                
                if split == 'train':
                    selected_indices = indices[:t_end]
                elif split == 'val':
                    selected_indices = indices[t_end:v_end]
                else:  # test
                    selected_indices = indices[v_end:]
                
                for i in selected_indices:
                    file_path = os.path.join(class_dir, files[i])
                    self.data.append((file_path, label, snr))
        
        print(f"[{split.upper()}] Loaded {len(self.data)} samples from SNRs: {snr_list}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label, snr = self.data[idx]
        try:
            arr = np.load(file_path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr], axis=-1)
            elif arr.shape[-1] == 1:
                arr = np.concatenate([arr, arr], axis=-1)
            tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
            if self.transform:
                tensor = self.transform(tensor)
            return tensor, label
        except Exception as e:
            print(f"[WARN] Error loading {file_path}: {e}. Returning zeros.")
            return torch.zeros((2, 224, 224), dtype=torch.float32), label


# -------------------------
# 3) Utilities: stats computation & saving
# -------------------------
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
    eps = 1e-6
    std = torch.sqrt(torch.clamp(var, min=eps))

    mean_list = mean.tolist()
    std_list = std.tolist()
    print(f"Computed mean: {mean_list}, std: {std_list}")
    return mean_list, std_list


def save_normalization_stats(mean, std, filepath):
    """Save normalization stats to JSON file."""
    stats = {'mean': mean, 'std': std}
    with open(filepath, 'w') as f:
        json.dump(stats, f)
    print(f"Saved normalization stats to {filepath}")


def load_normalization_stats(filepath):
    """Load normalization stats from JSON file."""
    with open(filepath, 'r') as f:
        stats = json.load(f)
    print(f"Loaded normalization stats from {filepath}")
    return stats['mean'], stats['std']


# -------------------------
# 4) Model architecture
# -------------------------
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
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
        xa = self.stream_amp(x[:, 0:1])
        xp = self.stream_phase(x[:, 1:2])
        x = torch.cat([xa, xp], dim=1)
        x = self.fuse(x)
        return self.classifier(x)


# -------------------------
# 5) Training / evaluation loop (main)
# -------------------------
if __name__ == "__main__":
    # =====================
    # CONFIGURATION
    # =====================
    TRAIN = False                # Set to False to only run evaluation
    
    # Training: use multiple SNR levels
    TRAIN_SNRS = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    # Testing: evaluate on specific SNR level
    TEST_SNR = "-10"
    
    SCALOGRAM_BASE_DIR = "Dataset/Scalograms"
    BATCH_SIZE = 128
    EPOCHS = 100
    NUM_WORKERS = 2
    
    # Model and stats file paths
    MODEL_PATH = "best_model_multisnr.pth"
    STATS_PATH = "normalization_stats_multisnr.json"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Training SNRs: {TRAIN_SNRS}")
    print(f"Test SNR: {TEST_SNR}")


    # =====================
    # TRAINING MODE
    # =====================
    if TRAIN:
        print("\n" + "="*50)
        print("TRAINING MODE")
        print("="*50)
        
        # Load existing normalization stats or compute from training data
        if os.path.exists(STATS_PATH):
            norm_mean, norm_std = load_normalization_stats(STATS_PATH)
        else:
            # Create multi-SNR training dataset (no transform) for stats computation
            print("\nCreating multi-SNR training dataset for stats computation...")
            train_dataset_raw = MultiSnrScalogramDataset(
                SCALOGRAM_BASE_DIR, TRAIN_SNRS, transform=None, split='train'
            )
            norm_mean, norm_std = compute_dataset_stats(train_dataset_raw, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            save_normalization_stats(norm_mean, norm_std, STATS_PATH)
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.RandomErasing(p=0.25, scale=(0.01, 0.08)),
            transforms.Normalize(norm_mean, norm_std)
        ])
        
        val_transform = transforms.Compose([
            transforms.Normalize(norm_mean, norm_std)
        ])
        
        # Create proper datasets with transforms
        train_dataset = MultiSnrScalogramDataset(
            SCALOGRAM_BASE_DIR, TRAIN_SNRS, transform=train_transform, split='train'
        )
        val_dataset = MultiSnrScalogramDataset(
            SCALOGRAM_BASE_DIR, TRAIN_SNRS, transform=val_transform, split='val'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
        
        # Model, optimizer, scheduler, loss
        num_classes = len(train_dataset.classes)
        model = DualStreamCNN(num_classes=num_classes).to(DEVICE)
        
        # Check for existing weights and load if available
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"\nLoaded existing weights from {MODEL_PATH}")
        else:
            print(f"\nNo existing weights found at {MODEL_PATH}, starting with fresh weights.")
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params:,} parameters (DualStreamCNN).")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {train_dataset.classes}")
        
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with early stopping
        best_val_acc = 0.0
        patience = 10
        counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"\nStarting training for {EPOCHS} epochs ...")
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
                torch.save(model.state_dict(), MODEL_PATH)
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
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Val')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('training_curves_multisnr.png')
        plt.close()
        print("Saved training_curves_multisnr.png")
    
    # =====================
    # EVALUATION MODE
    # =====================
    print("\n" + "="*50)
    print(f"EVALUATION MODE - Testing on SNR {TEST_SNR}")
    print("="*50)
    
    # Load normalization stats
    if not os.path.exists(STATS_PATH):
        print(f"[ERROR] Normalization stats file not found: {STATS_PATH}")
        print("Please run training first (TRAIN=True) to generate normalization stats.")
        exit(1)
    
    norm_mean, norm_std = load_normalization_stats(STATS_PATH)
    
    test_transform = transforms.Compose([
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    # Create test dataset from specific SNR
    test_dataset = MultiSnrScalogramDataset(
        SCALOGRAM_BASE_DIR, [TEST_SNR], transform=test_transform, split='test'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("Please run training first (TRAIN=True) to generate the model.")
        exit(1)
    
    num_classes = len(test_dataset.classes)
    model = DualStreamCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded model from {MODEL_PATH}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\nRunning evaluation on test set (SNR {TEST_SNR}) ...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.title(f'Confusion Matrix - Test SNR {TEST_SNR}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_test_snr{TEST_SNR}.png')
    plt.close()
    print(f"Saved confusion_matrix_test_snr{TEST_SNR}.png")
    
    print("\nDone.")

