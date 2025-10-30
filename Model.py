import os
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ScalogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, indices=None):
        """
        Dataset that can work with a subset of indices for train/val/test splits
        
        Args:
            root_dir: Path to the scalograms folder
            transform: Transformations to apply
            indices: List of indices to use (for splitting). If None, uses all data.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.all_data = []

        # Collect all data files
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.all_data.append((os.path.join(class_dir, file), label))
        
        # If indices are provided, use only those
        if indices is not None:
            self.data = [self.all_data[i] for i in indices]
        else:
            self.data = self.all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        scalogram = np.load(file_path).transpose(2, 0, 1)
        scalogram = torch.tensor(scalogram, dtype=torch.float32)
        if self.transform:
            scalogram = self.transform(scalogram)
        return scalogram, label


def create_train_val_test_split(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test indices without copying files
    
    Args:
        root_dir: Path to the scalograms folder
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices, test_indices
    """
    random.seed(seed)
    np.random.seed(seed)
    
    classes = sorted(os.listdir(root_dir))
    all_indices = []
    
    # Collect indices per class to ensure balanced splits
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        class_indices = list(range(len(class_files)))
        random.shuffle(class_indices)
        
        # Calculate split points
        total = len(class_indices)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        # Split indices for this class
        train_idx = class_indices[:train_count]
        val_idx = class_indices[train_count:train_count + val_count]
        test_idx = class_indices[train_count + val_count:]
        
        all_indices.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'class': class_name
        })
    
    # Now create global indices
    temp_dataset = ScalogramDataset(root_dir)
    global_train_indices = []
    global_val_indices = []
    global_test_indices = []
    
    # Map local class indices to global dataset indices
    current_idx = 0
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        num_files = len(class_files)
        
        # Get the split for this class
        class_split = next(item for item in all_indices if item['class'] == class_name)
        
        # Convert local indices to global
        for local_idx in class_split['train']:
            global_train_indices.append(current_idx + local_idx)
        for local_idx in class_split['val']:
            global_val_indices.append(current_idx + local_idx)
        for local_idx in class_split['test']:
            global_test_indices.append(current_idx + local_idx)
        
        current_idx += num_files
    
    return global_train_indices, global_val_indices, global_test_indices


def calculate_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(2)
    std = torch.zeros(2)
    for inputs, _ in loader:
        for i in range(2):
            mean[i] += inputs[:, i].mean()
            std[i] += inputs[:, i].std()
    return (mean / len(loader)).tolist(), (std / len(loader)).tolist()


# ------------------------
# Utility: Depthwise Separable Conv
# ------------------------
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# ------------------------
# Residual DSConv Block
# ------------------------
class ResidualDSBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            DSConv(ch, ch),
            DSConv(ch, ch)
        )

    def forward(self, x):
        return x + self.block(x)


# ------------------------
# Squeeze-and-Excitation (SE) Channel Attention
# ------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


# ------------------------
# One Stream (I or Q)
# ------------------------
def make_stream():
    return nn.Sequential(
        # FIX 1: Removed stride=2 to capture more detail
        nn.Conv2d(1, 16, 7, 1, 3, bias=False),  # WAS: stride=2
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2),    # First downsampling: 224 -> 111

        DSConv(16, 32),
        ResidualDSBlock(32),
        nn.MaxPool2d(3, 2),    # Second downsampling: 111 -> 55

        DSConv(32, 64),
        ResidualDSBlock(64),
        SEBlock(64),
        nn.MaxPool2d(3, 2)     # Third downsampling: 55 -> 27
    )


# ------------------------
# Cross Attention Fusion
# ------------------------
class CrossAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 2, 1)
        self.key = nn.Conv2d(ch, ch // 2, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, a, p):
        # flatten spatial dimensions
        B, C, H, W = a.shape
        Q = self.query(a).flatten(2)          # [B, C/2, HW]
        K = self.key(p).flatten(2)            # [B, C/2, HW]
        V = self.value(p).flatten(2)          # [B, C, HW]
        attn = self.softmax(torch.bmm(Q.transpose(1, 2), K))  # [B, HW, HW]
        out = torch.bmm(V, attn.transpose(1, 2)).view(B, C, H, W)
        return out + a


# ------------------------
# Dual-Stream CWTNet
# ------------------------
class DualStreamCWTNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        # Renamed for clarity
        self.stream_i = make_stream()
        self.stream_q = make_stream()
        
        # FIX 2: Replaced Cross-Attention with a Merge layer
        # This 1x1 Conv will merge the concatenated 64+64=128 channels
        self.merge = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            ResidualDSBlock(64),
            SEBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x[:, 0:1] is the I-scalogram
        # x[:, 1:2] is the Q-scalogram
        
        # 1. Process streams in parallel (This is the "Dual-Stream" part)
        xi = self.stream_i(x[:, 0:1])  # I-Features
        xq = self.stream_q(x[:, 1:2])  # Q-Features

        # 2. Fuse by concatenating (This is the "Fusion" part)
        # We stack them on the channel dimension (dim=1)
        fused = torch.cat([xi, xq], dim=1)  # Shape: [B, 128, H, W]
        
        # 3. Merge the fused features
        fused = self.merge(fused)           # Shape: [B, 64, H, W]
        
        # 4. Classify
        fused = self.fuse(fused)
        return self.classifier(fused)


if __name__ == '__main__':
    # Training switch - Set to False to skip training and only evaluate
    TRAIN = True
    SNR = "20"
    
    # Dataset setup - Load directly from Scalograms folder
    data_dir = f'Scalograms/snr_{SNR}'
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    print("Creating train/val/test splits in memory...")
    train_indices, val_indices, test_indices = create_train_val_test_split(
        data_dir, train_ratio, val_ratio, test_ratio, seed=42
    )
    print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Calculate dataset stats using training data only
    temp_train = ScalogramDataset(data_dir, indices=train_indices)
    mean, std = calculate_dataset_stats(temp_train)
    print(f"Dataset stats - Mean: {mean}, Std: {std}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    # Create datasets with their respective indices
    train_dataset = ScalogramDataset(data_dir, train_transform, indices=train_indices)
    val_dataset = ScalogramDataset(data_dir, val_test_transform, indices=val_indices)
    test_dataset = ScalogramDataset(data_dir, val_test_transform, indices=test_indices)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DualStreamCWTNet(len(train_dataset.classes)).to(device)
    print(sum(p.numel() for p in model.parameters()))
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    if TRAIN:
        # Training loop with early stopping
        best_val_acc = 0.0
        patience_counter = 0
        patience = 7
        num_epochs = 50
        
        # History tracking for plotting
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            train_acc = correct / total
            val_acc = val_correct / val_total
            
            # Store history
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)

            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"Learning rate reduced to {new_lr:.6f}")

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss_avg:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss_avg:.4f} | Acc: {val_acc:.4f}\n")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f"weights_snr{SNR}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Plot training history
        print("Plotting training curves...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("Training curves saved to 'training_curves.png'")
        plt.show()
        
    else:
        print("Training skipped. Loading existing model...")

    # Final evaluation
    model.load_state_dict(torch.load(f"weights_snr{SNR}.pth"))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # Confusion matrices
    plt.figure(figsize=(15, 6))

    # Raw counts
    plt.subplot(1, 1, 1)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(f"Confusion_Matrix_snr{SNR}.png")
    plt.show()

