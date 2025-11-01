"""
==============================================================
Triple-Stream Modulation Classification Network
==============================================================
Combines three parallel streams:
1. Amplitude Scalogram (224x224x1)
2. Phase Scalogram (224x224x1)
3. Raw IQ Signal (1024x2)

The model processes all three streams independently, fuses them
using cross-attention, and performs classification.
==============================================================
"""

import os
import random
import numpy as np
import torch
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


class TripleStreamDataset(Dataset):
    def __init__(self, scalogram_dir, iq_dir, transform=None, indices=None):
        """
        Dataset that loads:
        - Amplitude scalogram from scalogram_dir
        - Phase scalogram from scalogram_dir
        - Raw IQ data from iq_dir
        
        Args:
            scalogram_dir: Path to folder with scalograms (224x224x2)
            iq_dir: Path to folder with raw IQ data (1024x2)
            transform: Transformations for scalograms
            indices: List of indices for train/val/test split
        """
        self.scalogram_dir = scalogram_dir
        self.iq_dir = iq_dir
        self.transform = transform
        self.classes = sorted(os.listdir(scalogram_dir))
        self.all_data = []

        # Collect all data files
        for label, class_name in enumerate(self.classes):
            scalogram_class_dir = os.path.join(scalogram_dir, class_name)
            iq_class_dir = os.path.join(iq_dir, class_name)
            
            if not os.path.isdir(scalogram_class_dir) or not os.path.isdir(iq_class_dir):
                continue
                
            scalogram_files = sorted([f for f in os.listdir(scalogram_class_dir) if f.endswith('.npy')])
            
            for file in scalogram_files:
                scalogram_path = os.path.join(scalogram_class_dir, file)
                iq_path = os.path.join(iq_class_dir, file)
                
                # Only add if both files exist
                if os.path.exists(iq_path):
                    self.all_data.append((scalogram_path, iq_path, label))
        
        # If indices are provided, use only those
        if indices is not None:
            self.data = [self.all_data[i] for i in indices]
        else:
            self.data = self.all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scalogram_path, iq_path, label = self.data[idx]
        
        # Load scalogram (224, 224, 2) -> separate amplitude and phase
        scalogram = np.load(scalogram_path)  # Shape: (H, W, 2)
        amplitude = scalogram[:, :, 0:1]  # Shape: (H, W, 1)
        phase = scalogram[:, :, 1:2]      # Shape: (H, W, 1)
        
        # Transpose to PyTorch format (C, H, W)
        amplitude = amplitude.transpose(2, 0, 1)  # (1, H, W)
        phase = phase.transpose(2, 0, 1)          # (1, H, W)
        
        # Load raw IQ data (1024, 2)
        iq_data = np.load(iq_path)  # Shape: (1024, 2)
        iq_data = iq_data.T  # Shape: (2, 1024) - transpose to (channels, sequence)
        
        # Convert to tensors
        amplitude = torch.tensor(amplitude, dtype=torch.float32)
        phase = torch.tensor(phase, dtype=torch.float32)
        iq_data = torch.tensor(iq_data, dtype=torch.float32)
        
        # Apply transforms to scalograms if provided
        if self.transform:
            # Stack for batch normalization, then split
            scalogram_stack = torch.cat([amplitude, phase], dim=0)
            scalogram_stack = self.transform(scalogram_stack)
            amplitude = scalogram_stack[0:1]
            phase = scalogram_stack[1:2]
        
        return amplitude, phase, iq_data, label


def create_train_val_test_split(scalogram_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test indices
    """
    random.seed(seed)
    np.random.seed(seed)
    
    classes = sorted(os.listdir(scalogram_dir))
    all_indices = []
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(scalogram_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        class_indices = list(range(len(class_files)))
        random.shuffle(class_indices)
        
        total = len(class_indices)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_idx = class_indices[:train_count]
        val_idx = class_indices[train_count:train_count + val_count]
        test_idx = class_indices[train_count + val_count:]
        
        all_indices.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'class': class_name
        })
    
    # Create global indices
    global_train_indices = []
    global_val_indices = []
    global_test_indices = []
    
    current_idx = 0
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(scalogram_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        num_files = len(class_files)
        
        class_split = next(item for item in all_indices if item['class'] == class_name)
        
        for local_idx in class_split['train']:
            global_train_indices.append(current_idx + local_idx)
        for local_idx in class_split['val']:
            global_val_indices.append(current_idx + local_idx)
        for local_idx in class_split['test']:
            global_test_indices.append(current_idx + local_idx)
        
        current_idx += num_files
    
    return global_train_indices, global_val_indices, global_test_indices


def calculate_dataset_stats(dataset):
    """Calculate mean and std for scalogram channels"""
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(2)  # For amplitude and phase
    std = torch.zeros(2)
    
    for amp, phase, iq, _ in loader:
        mean[0] += amp.mean()
        mean[1] += phase.mean()
        std[0] += amp.std()
        std[1] += phase.std()
    
    mean /= len(loader)
    std /= len(loader)
    
    return mean.tolist(), std.tolist()


# ------------------------
# CNN Blocks
# ------------------------
class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
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


class ResidualDSBlock(nn.Module):
    """Residual Block with Depthwise Separable Conv"""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            DSConv(ch, ch),
            DSConv(ch, ch)
        )

    def forward(self, x):
        return x + self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention"""
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
# Scalogram Stream (for amplitude and phase)
# ------------------------
def make_scalogram_stream():
    """Feature extraction for 2D scalogram data (224x224)"""
    return nn.Sequential(
        nn.Conv2d(1, 16, 7, 2, 3, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2),

        DSConv(16, 32),
        ResidualDSBlock(32),
        nn.MaxPool2d(3, 2),

        DSConv(32, 64),
        ResidualDSBlock(64),
        SEBlock(64),
        nn.MaxPool2d(3, 2)
    )
    # Output: [B, 64, 13, 13]


# ------------------------
# IQ Stream (for raw 1024x2 signal)
# ------------------------
class IQStream(nn.Module):
    """
    Feature extraction for raw IQ signal (1024x2)
    Uses 1D convolutions on time series data
    """
    def __init__(self):
        super().__init__()
        # Input: (B, 2, 1024) - 2 channels (I and Q), 1024 time steps
        
        # Initial 1D conv block
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2)
        )  # Output: (B, 32, 256)
        
        # Second 1D conv block
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2)
        )  # Output: (B, 64, 128)
        
        # Third 1D conv block
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2)
        )  # Output: (B, 128, 64)
        
        # Fourth 1D conv block
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(169)  # 169 = 13*13 to match scalogram stream spatial size
        )  # Output: (B, 64, 169)
        
    def forward(self, x):
        """
        Input: (B, 2, 1024)
        Output: (B, 64, 13, 13) - reshaped to match scalogram stream
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Reshape to 2D feature map to match scalogram streams
        B = x.size(0)
        x = x.view(B, 64, 13, 13)
        
        return x


# ------------------------
# Cross Attention Fusion
# ------------------------
class CrossAttention(nn.Module):
    """Cross-attention between two feature maps"""
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 2, 1)
        self.key = nn.Conv2d(ch, ch // 2, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, a, p):
        B, C, H, W = a.shape
        Q = self.query(a).flatten(2)
        K = self.key(p).flatten(2)
        V = self.value(p).flatten(2)
        attn = self.softmax(torch.bmm(Q.transpose(1, 2), K))
        out = torch.bmm(V, attn.transpose(1, 2)).view(B, C, H, W)
        return out + a


# ------------------------
# Triple-Stream Model
# ------------------------
class TripleStreamCWTNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        
        # Three independent feature extraction streams
        self.stream_amp = make_scalogram_stream()    # Amplitude scalogram
        self.stream_phase = make_scalogram_stream()  # Phase scalogram
        self.stream_iq = IQStream()                  # Raw IQ signal
        
        # Cross-attention modules for feature fusion
        self.cross_amp_phase = CrossAttention(64)
        self.cross_phase_iq = CrossAttention(64)
        self.cross_amp_iq = CrossAttention(64)
        
        # Fusion and pooling
        self.fuse = nn.Sequential(
            ResidualDSBlock(64),
            SEBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, amplitude, phase, iq):
        """
        Args:
            amplitude: (B, 1, 224, 224) - Amplitude scalogram
            phase: (B, 1, 224, 224) - Phase scalogram
            iq: (B, 2, 1024) - Raw IQ signal
        """
        # Extract features from each stream
        xa = self.stream_amp(amplitude)      # (B, 64, 13, 13)
        xp = self.stream_phase(phase)        # (B, 64, 13, 13)
        xq = self.stream_iq(iq)              # (B, 64, 13, 13)
        
        # Pairwise cross-attention fusion
        a_p = self.cross_amp_phase(xa, xp)   # Amplitude attending to Phase
        p_q = self.cross_phase_iq(xp, xq)    # Phase attending to IQ
        a_q = self.cross_amp_iq(xa, xq)      # Amplitude attending to IQ
        
        # Combine all fused features
        fused = (a_p + p_q + a_q) / 3.0
        
        # Final classification
        fused = self.fuse(fused)
        output = self.classifier(fused)
        
        return output


# ------------------------
# Training Function
# ------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for amp, phase, iq, labels in loader:
        amp, phase, iq, labels = amp.to(device), phase.to(device), iq.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(amp, phase, iq)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ------------------------
# Validation Function
# ------------------------
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for amp, phase, iq, labels in loader:
            amp, phase, iq, labels = amp.to(device), phase.to(device), iq.to(device), labels.to(device)
            
            outputs = model(amp, phase, iq)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ------------------------
# Test and Confusion Matrix
# ------------------------
def test_and_plot(model, loader, device, classes, save_path='confusion_matrix.png'):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for amp, phase, iq, labels in loader:
            amp, phase, iq, labels = amp.to(device), phase.to(device), iq.to(device), labels.to(device)
            
            outputs = model(amp, phase, iq)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return accuracy


# ------------------------
# Main Training Script
# ------------------------
if __name__ == '__main__':
    # Configuration
    SNR = "30"
    TRAIN = True
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    DROPOUT = 0.4
    
    # Paths
    scalogram_dir = f'Scalograms/snr_{SNR}'
    iq_dir = f'Dataset/snr_{SNR}'
    
    print("="*60)
    print("Triple-Stream Modulation Classification")
    print("="*60)
    print(f"SNR Level: {SNR} dB")
    print(f"Scalogram Dir: {scalogram_dir}")
    print(f"IQ Data Dir: {iq_dir}")
    
    # Create splits
    print("\nCreating train/val/test splits...")
    train_indices, val_indices, test_indices = create_train_val_test_split(
        scalogram_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Calculate stats
    print("\nCalculating dataset statistics...")
    temp_dataset = TripleStreamDataset(scalogram_dir, iq_dir, indices=train_indices)
    mean, std = calculate_dataset_stats(temp_dataset)
    print(f"Mean: {mean}, Std: {std}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    val_test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    # Datasets
    train_dataset = TripleStreamDataset(scalogram_dir, iq_dir, train_transform, train_indices)
    val_dataset = TripleStreamDataset(scalogram_dir, iq_dir, val_test_transform, val_indices)
    test_dataset = TripleStreamDataset(scalogram_dir, iq_dir, val_test_transform, test_indices)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    model = TripleStreamCWTNet(num_classes=num_classes, dropout=DROPOUT).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if TRAIN:
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'triple_stream_best_snr{SNR}.pth')
                print(f'  → Saved best model (Val Acc: {val_acc:.2f}%)')
            
            scheduler.step(val_acc)
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'triple_stream_training_curves_snr{SNR}.png', dpi=300)
        plt.close()
        print(f'\nTraining curves saved to triple_stream_training_curves_snr{SNR}.png')
        
        # Load best model for testing
        model.load_state_dict(torch.load(f'triple_stream_best_snr{SNR}.pth'))
    
    # Test evaluation
    print("\n" + "="*60)
    print("Testing")
    print("="*60)
    test_accuracy = test_and_plot(
        model, test_loader, device, 
        train_dataset.classes, 
        f'triple_stream_confusion_matrix_snr{SNR}.png'
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
