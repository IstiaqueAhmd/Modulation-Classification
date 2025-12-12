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

# ---------------------------------------------------------
# 1. ATTENTION MECHANISM (Squeeze-and-Excitation)
# ---------------------------------------------------------
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Helps the network decide 'which channels are important'.
    Critical for ignoring empty frequency bands in scalograms.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ---------------------------------------------------------
# 2. BUILDING BLOCK (MBConv: Mobile Inverted Residual)
# ---------------------------------------------------------
class MBConvBlock(nn.Module):
    """
    Modern replacement for DSConv.
    Expansion -> Depthwise -> SE -> Pointwise.
    """
    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1, kernel_size=3):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch
        hidden_dim = in_ch * expand_ratio
        
        layers = []
        # 1. Expansion Phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU(inplace=True)) # SiLU (Swish) is better for signal processing

        # 2. Depthwise Convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                                padding=kernel_size//2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU(inplace=True))

        # 3. Squeeze-and-Excitation
        layers.append(SELayer(hidden_dim))

        # 4. Pointwise Convolution (Projection)
        layers.append(nn.Conv2d(hidden_dim, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

# ---------------------------------------------------------
# 3. SPECIALIZED STREAM GENERATOR
# ---------------------------------------------------------
def make_specialized_stream(in_channels=1):
    """
    Constructs a stream backbone.
    Downgrades 224x224 -> 7x7 feature map.
    """
    return nn.Sequential(
        # Stem: Fast downsampling
        nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False), # 112x112
        nn.BatchNorm2d(24),
        nn.SiLU(inplace=True),
        
        # MBConv Blocks (Deeper and narrower than standard ResNet)
        MBConvBlock(24, 32, stride=2),           # 56x56
        MBConvBlock(32, 48, stride=2),           # 28x28
        MBConvBlock(48, 96, stride=2),           # 14x14
        MBConvBlock(96, 128, stride=2),          # 7x7
        
        # Final feature extraction
        nn.Conv2d(128, 256, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.SiLU(inplace=True)
    )

# ---------------------------------------------------------
# 4. GATED MODAL FUSION
# ---------------------------------------------------------
class GatedFusion(nn.Module):
    """
    Learns to weigh streams dynamically.
    E.g., If 'Constellation' looks messy (low SNR), 
    the model learns to lower its weight and trust 'Phase' more.
    """
    def __init__(self, num_streams, feature_dim):
        super().__init__()
        # Attention network: Takes concatenated global features -> outputs weights
        self.gate_fc = nn.Sequential(
            nn.Linear(feature_dim * num_streams, 64),
            nn.ReLU(),
            nn.Linear(64, num_streams), 
            nn.Softmax(dim=1) 
        )

    def forward(self, feats_list):
        # feats_list: [x1, x2, x3] where each is [B, C]
        
        # 1. Stack features to calculate weights
        cat_feats = torch.cat(feats_list, dim=1) # [B, C*3]
        weights = self.gate_fc(cat_feats)        # [B, 3]
        
        # 2. Apply weights to the feature vectors
        # w1*x1 + w2*x2 + w3*x3
        weighted_feats = 0
        for i, x in enumerate(feats_list):
            w = weights[:, i].unsqueeze(1) # [B, 1]
            weighted_feats += w * x
            
        return weighted_feats

# ---------------------------------------------------------
# 5. MAIN MODEL
# ---------------------------------------------------------
class TriSpectralNet(nn.Module):
    def __init__(self, num_classes=24, dropout=0.3):
        super().__init__()
        
        # Three specialized streams
        self.stream_amp = make_specialized_stream()
        self.stream_phase = make_specialized_stream()
        self.stream_const = make_specialized_stream()
        
        # Global Pooling to flatten 7x7 spatial maps to vectors
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Gated Fusion mechanism (Feature dim is 256 from the stream output)
        self.fusion = GatedFusion(num_streams=3, feature_dim=256)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, 224, 224]
        
        # 1. Process Streams Independently
        # Note: 256 channels output, 7x7 spatial size
        s1 = self.stream_amp(x[:, 0:1]) 
        s2 = self.stream_phase(x[:, 1:2])
        s3 = self.stream_const(x[:, 2:3])
        
        # 2. Global Pool (Spatial -> Vector) [B, 256]
        v1 = self.pool(s1).flatten(1)
        v2 = self.pool(s2).flatten(1)
        v3 = self.pool(s3).flatten(1)
        
        # 3. Intelligent Fusion
        # Instead of just concatenating, we weight them
        fused_vector = self.fusion([v1, v2, v3])
        
        # 4. Classification
        return self.classifier(fused_vector)


# -------------------------
# 5) Training / evaluation loop (main)
# -------------------------
if __name__ == "__main__":
    # Configuration - adjust as needed
    TRAIN = True
    SNR = "30"
    BATCH_SIZE = 32
    EPOCHS = 50
    DATA_DIR = f"Scalograms_TriChannel/snr_{SNR}"
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
    model = TriSpectralNet(num_classes=num_classes).to(DEVICE)
    
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