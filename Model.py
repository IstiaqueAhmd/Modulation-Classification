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
from torchvision import models

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================
#   CUSTOM 1D DATA AUGMENTATIONS (REMOVED)
# =============================================================

# (Removed AddGaussianNoise class)
# (Removed RandomTimeShift class)

# =============================================================
#   CUSTOM 1D NORMALIZE TRANSFORM
# =============================================================

class Normalize1D(nn.Module):
    """
    Normalizes a tensor with shape (C, L)
    """
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        # Ensure mean and std are tensors
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.inplace = inplace

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be normalized.
        Returns:
            Tensor: Normalized tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        # Reshape mean and std to (C, 1) for broadcasting
        mean = self.mean.view(-1, 1)
        std = self.std.view(-1, 1)
        
        # Apply normalization
        tensor.sub_(mean).div_(std)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean.tolist()}, std={self.std.tolist()})'


# =============================================================
#   DATASET & SPLITTING
# =============================================================

class IQDataset(Dataset):
    """
    Dataset for loading raw (1024, 2) I/Q data.
    """
    def __init__(self, root_dir, transform=None, indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
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
        
        # Load data, shape (1024, 2)
        iq_data = np.load(file_path).astype(np.float32)
        
        # Transpose to (2, 1024) -> (C, L) for Conv1d
        iq_data = iq_data.transpose(1, 0)
        
        # Convert to tensor
        iq_tensor = torch.tensor(iq_data, dtype=torch.float32)
        
        if self.transform:
            iq_tensor = self.transform(iq_tensor)
            
        return iq_tensor, label


def create_train_val_test_split(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits data into train/val/test indices by file.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    all_indices = []
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        # Get indices for this class's files
        class_indices = list(range(len(class_files)))
        random.shuffle(class_indices)
        
        # Calculate split points
        total = len(class_indices)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_idx = class_indices[:train_count]
        val_idx = class_indices[train_count : train_count + val_count]
        test_idx = class_indices[train_count + val_count:]
        
        all_indices.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'class': class_name
        })
    
    # Map local class indices to global dataset indices
    global_train_indices = []
    global_val_indices = []
    global_test_indices = []
    
    current_idx = 0
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
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
    """
    Calculate mean and std for I and Q channels.
    Input tensor shape is expected to be (C, L) = (2, 1024).
    """
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(2)
    std = torch.zeros(2)
    
    for inputs, _ in loader:
        # inputs shape is (B, 2, 1024)
        mean[0] += inputs[:, 0, :].mean()
        std[0] += inputs[:, 0, :].std()
        mean[1] += inputs[:, 1, :].mean()
        std[1] += inputs[:, 1, :].std()
    
    mean /= len(loader)
    std /= len(loader)
    
    # Add epsilon to prevent division by zero
    std += 1e-6 
    
    return mean.tolist(), std.tolist()


# =============================================================
#   1D DUAL-STREAM MODEL
# =============================================================

class DualStreamCNN_1D(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        
        # This function creates one 1D CNN stream
        def make_stream():
            return nn.Sequential(
                # Input: (B, 1, 1024)
                nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                # -> (B, 64, 256)

                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                # -> (B, 128, 64)

                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                # -> (B, 256, 16)

                nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                # -> (B, 512, 8)
                
                # Global pooling
                nn.AdaptiveAvgPool1d(1),
                # -> (B, 512, 1)
                nn.Flatten()
                # -> (B, 512)
            )
        
        # Create two independent streams
        self.stream_i = make_stream()
        self.stream_q = make_stream()
        
        # Classifier head
        self.classifier = nn.Sequential(
            # Input size is 512 (from I) + 512 (from Q)
            nn.Linear(512 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Input x shape: (B, 2, 1024)
        
        # Split into I and Q streams
        # x_i shape: (B, 1, 1024)
        # x_q shape: (B, 1, 1024)
        x_i = x[:, 0:1, :]
        x_q = x[:, 1:2, :]
        
        # Process each stream independently
        # features_i shape: (B, 512)
        # features_q shape: (B, 512)
        features_i = self.stream_i(x_i)
        features_q = self.stream_q(x_q)
        
        # Concatenate features
        # fused_features shape: (B, 1024)
        fused_features = torch.cat([features_i, features_q], dim=1)
        
        # Classify
        output = self.classifier(fused_features)
        return output

# =============================================================
#   MAIN TRAINING & EVALUATION SCRIPT
# =============================================================

if __name__ == '__main__':
    # --- Configuration ---
    TRAIN = True
    SNR = "30"
    
    # Point this to your *original* I/Q dataset root
    data_dir = f'Dataset/snr_{SNR}' 
    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    patience = 7 # For early stopping
    
    # --- 1. Create Splits ---
    print("Creating train/val/test splits in memory...")
    train_indices, val_indices, test_indices = create_train_val_test_split(
        data_dir, train_ratio, val_ratio, test_ratio, seed=42
    )
    print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # --- 2. Calculate Stats ---
    print("Calculating dataset statistics...")
    # Use a temporary dataset *without* augmentations to calculate stats
    temp_train = IQDataset(data_dir, transform=None, indices=train_indices)
    mean, std = calculate_dataset_stats(temp_train)
    print(f"Dataset stats - Mean: {mean}, Std: {std}")

    # --- 3. Create Transforms ---
    train_transform = transforms.Compose([
        # Add 1D augmentations *before* normalizing
        # (Removed AddGaussianNoise and RandomTimeShift)
        Normalize1D(mean, std)
    ])

    val_test_transform = transforms.Compose([
        Normalize1D(mean, std)
    ])

    # --- 4. Create Datasets & DataLoaders ---
    train_dataset = IQDataset(data_dir, train_transform, indices=train_indices)
    val_dataset = IQDataset(data_dir, val_test_transform, indices=val_indices)
    test_dataset = IQDataset(data_dir, val_test_transform, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 5. Initialize Model & Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DualStreamCNN_1D(num_classes=len(train_dataset.classes)).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # --- 6. Training Loop ---
    if TRAIN:
        best_val_acc = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

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

            # Log metrics
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            train_acc = correct / total
            val_acc = val_correct / val_total
            
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f"weights_1D_snr{SNR}.pth")
                print(f"   -> New best model saved with Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Plot training history
        print("Plotting training curves...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs_range, history['train_loss'], label='Training Loss')
        ax1.plot(epochs_range, history['val_loss'], label='Validation Loss')
        ax1.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs_range, history['train_acc'], label='Training Accuracy')
        ax2.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
        ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'training_curves_1D_snr{SNR}.png', dpi=300)
        plt.show()
        
    else:
        print("Training skipped. Loading existing model...")

    # --- 7. Final Evaluation ---
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(f"weights_1D_snr{SNR}.pth"))
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
    print("\n" + "="*60)
    print("           Classification Report")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # Confusion matrices
    print("Generating confusion matrix...")
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title(f"Confusion Matrix (Counts) - SNR {SNR} dB")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.subplot(1, 2, 2)
    cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title(f"Normalized Confusion Matrix - SNR {SNR} dB")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig(f"Confusion_Matrix_1D_snr{SNR}.png", dpi=300)
    plt.show()
    print("Evaluation complete. ✅")


