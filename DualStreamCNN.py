import os
import random
import numpy as np
import torch
import torch.nn as nn
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
torch.backends.cudnn.benchmark = True # Optimized for speed

class ScalogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.all_data = []

        # Collect all data files
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            # Sort files to ensure deterministic ordering for sequential splitting
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            for file in files:
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
        # Load [224, 224, 2] -> Transpose to [2, 224, 224]
        scalogram = np.load(file_path).transpose(2, 0, 1)
        scalogram = torch.tensor(scalogram, dtype=torch.float32)
        
        if self.transform:
            scalogram = self.transform(scalogram)
        return scalogram, label

# ---------------------------------------------------------
# FIX #1: Sequential Split (Prevents Data Leakage)
# ---------------------------------------------------------
def create_sequential_split(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    classes = sorted(os.listdir(root_dir))
    global_train_indices = []
    global_val_indices = []
    global_test_indices = []
    
    current_idx = 0
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        num_files = len(class_files)
        
        # Sequential indices (No Shuffle)
        indices = list(range(num_files))
        
        train_end = int(num_files * train_ratio)
        val_end = int(num_files * (train_ratio + val_ratio))
        
        train_local = indices[:train_end]
        val_local = indices[train_end:val_end]
        test_local = indices[val_end:]
        
        global_train_indices.extend([current_idx + i for i in train_local])
        global_val_indices.extend([current_idx + i for i in val_local])
        global_test_indices.extend([current_idx + i for i in test_local])
        
        current_idx += num_files
    
    return global_train_indices, global_val_indices, global_test_indices

# ---------------------------------------------------------
# YOUR ORIGINAL MODEL ARCHITECTURE
# ---------------------------------------------------------
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False), # Depth-wise
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), # Point-wise
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

def make_stream():
    return nn.Sequential(
        nn.Conv2d(1, 16, 7, 2, 3, bias=False),
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
            DSConv(64, 64), # 32+32 channels concatenated
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        xa = self.stream_amp(x[:, 0:1])     # amplitude channel
        xp = self.stream_phase(x[:, 1:2])   # phase channel
        x = torch.cat([xa, xp], 1)          
        return self.classifier(self.fuse(x))

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == '__main__':
    TRAIN = True
    SNR = "30"
    data_dir = f'Scalograms/snr_{SNR}'
    
    # 1. Setup Splits (Using Sequential Splitter)
    print("Creating sequential train/val/test splits...")
    train_indices, val_indices, test_indices = create_sequential_split(
        data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    print(f"Counts -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # 2. Setup Transforms (FIX #2 & #3)
    # Hardcoded normalization is faster and sufficient for MinMax scaled inputs
    norm_mean = [0.5, 0.5]
    norm_std = [0.5, 0.5]

    # Added RandomErasing for robustness against noise bursts
    train_transform = transforms.Compose([
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)) 
    ])

    val_test_transform = transforms.Compose([
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 3. Create Datasets & Loaders
    train_dataset = ScalogramDataset(data_dir, train_transform, indices=train_indices)
    val_dataset = ScalogramDataset(data_dir, val_test_transform, indices=val_indices)
    test_dataset = ScalogramDataset(data_dir, val_test_transform, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DualStreamCNN(len(train_dataset.classes)).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    if TRAIN:
        best_val_acc = 0.0
        patience = 8
        counter = 0
        epochs = 50
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            model.train()
            run_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                run_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = run_loss / len(train_loader)
            train_acc = correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            # Update History & Scheduler
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), f"best_model_snr{SNR}.pth")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Plotting
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
        print("Training complete.")

    # 6. Evaluation
    if os.path.exists(f"best_model_snr{SNR}.pth"):
        model.load_state_dict(torch.load(f"best_model_snr{SNR}.pth"))
        print("Loaded best model weights.")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix SNR {SNR}')
    plt.tight_layout()
    plt.savefig(f'Confusion_Matrix_snr{SNR}.png')
    print("Saved confusion matrix.")