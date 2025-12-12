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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# 1. DATASET CLASS
# ==========================================
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
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.all_data.append((os.path.join(class_dir, file), label))
        
        # Use subset if indices provided
        if indices is not None:
            self.data = [self.all_data[i] for i in indices]
        else:
            self.data = self.all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        # Load [H, W, 3] -> Transpose to [3, H, W]
        # Channel 0: Amp, Channel 1: Phase, Channel 2: Constellation
        scalogram = np.load(file_path).transpose(2, 0, 1)
        scalogram = torch.tensor(scalogram, dtype=torch.float32)
        
        if self.transform:
            scalogram = self.transform(scalogram)
            
        return scalogram, label

# ==========================================
# 2. UTILITY SPLIT FUNCTION
# ==========================================
def create_train_val_test_split(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    classes = sorted(os.listdir(root_dir))
    all_indices = []
    
    # Per-class balancing
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir): continue
            
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        class_indices = list(range(len(class_files)))
        random.shuffle(class_indices)
        
        total = len(class_indices)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        all_indices.append({
            'train': class_indices[:train_count],
            'val': class_indices[train_count:train_count + val_count],
            'test': class_indices[train_count + val_count:],
            'class': class_name
        })
    
    # Map back to global indices
    current_idx = 0
    global_train, global_val, global_test = [], [], []
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir): continue
        num_files = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        
        class_split = next(item for item in all_indices if item['class'] == class_name)
        
        for local_idx in class_split['train']: global_train.append(current_idx + local_idx)
        for local_idx in class_split['val']:   global_val.append(current_idx + local_idx)
        for local_idx in class_split['test']:  global_test.append(current_idx + local_idx)
        
        current_idx += num_files
    
    return global_train, global_val, global_test

def calculate_dataset_stats(dataset):
    # Modified for 3 Channels
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, _ in loader:
        for i in range(3):
            mean[i] += inputs[:, i].mean()
            std[i] += inputs[:, i].std()
    return (mean / len(loader)).tolist(), (std / len(loader)).tolist()

# ==========================================
# 3. NEURAL NETWORK BLOCKS
# ==========================================
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class ResidualDSBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(DSConv(ch, ch), DSConv(ch, ch))
    def forward(self, x):
        return x + self.block(x)

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
        return x * self.fc(x)

def make_stream():
    """Defines the structure for one input stream."""
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

def make_scalogram_stream():
    """
    Optimized for Shapes (Time-Frequency).
    Uses larger 7x7 start and standard MaxPooling.
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, 7, 2, 3, bias=False), # Large kernel for big shapes
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

def make_constellation_stream():
    """
    Optimized for Fine Dots (Density Grid).
    Uses small 3x3 kernels and Strided Convolutions (No MaxPool)
    to preserve the grid density information of QAM signals.
    """
    return nn.Sequential(
        # Stage 1: High Res Feature Extraction
        nn.Conv2d(1, 16, 3, 1, 1, bias=False), # Keep 224x224, look for dots
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        
        # Stage 2: Gentle Downsampling via Stride
        nn.Conv2d(16, 32, 3, 2, 1, bias=False), # 112x112
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        
        # Stage 3: Learn the Grid Pattern
        ResidualDSBlock(32),
        ResidualDSBlock(32),
        
        # Stage 4: Downsample
        nn.Conv2d(32, 64, 3, 2, 1, bias=False), # 56x56
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        
        # Stage 5: Final Features
        ResidualDSBlock(64),
        SEBlock(64),
        
        # Final stride to match the output size of the scalogram stream (approx 28x28)
        nn.Conv2d(64, 64, 3, 2, 1, bias=False)  # 28x28
    )

# ==========================================
# 4. TRI-STREAM NETWORK ARCHITECTURE
# ==========================================
class TriStreamNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        
        # 1. Scalogram Streams (Time/Frequency optimized)
        self.stream_amp = make_scalogram_stream()
        self.stream_phase = make_scalogram_stream()
        
        # 2. Constellation Stream (Dot/Grid optimized)
        self.stream_const = make_constellation_stream()
        
        # Fusion Block (Everything else remains the same)
        self.fuse = nn.Sequential(
            nn.Conv2d(192, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            ResidualDSBlock(128),
            SEBlock(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
    
    # Forward pass remains exactly the same
    def forward(self, x):
        xa = self.stream_amp(x[:, 0:1])
        xp = self.stream_phase(x[:, 1:2])
        xc = self.stream_const(x[:, 2:3])

        combined = torch.cat([xa, xp, xc], dim=1)
        fused = self.fuse(combined)
        return self.classifier(fused)

# ==========================================
# 5. TRAINING ROUTINE
# ==========================================
if __name__ == '__main__':
    TRAIN = True
    SNR = "30"
    
    # Folder containing Tri-Channel data
    data_dir = f'Scalograms_TriChannel/snr_{SNR}'
    
    # 1. Split Data
    print("Creating splits...")
    train_idx, val_idx, test_idx = create_train_val_test_split(data_dir)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # 2. Stats Calculation
    temp_ds = ScalogramDataset(data_dir, indices=train_idx)
    mean, std = calculate_dataset_stats(temp_ds)
    print(f"Stats (3 Channels) -> Mean: {mean}, Std: {std}")
    
    # 3. Transforms
    common_transform = transforms.Compose([transforms.Normalize(mean, std)])
    
    train_ds = ScalogramDataset(data_dir, common_transform, indices=train_idx)
    val_ds = ScalogramDataset(data_dir, common_transform, indices=val_idx)
    test_ds = ScalogramDataset(data_dir, common_transform, indices=test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    
    # 4. Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriStreamNet(len(train_ds.classes)).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    if TRAIN:
        best_acc = 0.0
        patience, counter = 7, 0
        history = {'t_loss': [], 't_acc': [], 'v_loss': [], 'v_acc': []}
        
        for epoch in range(50):
            model.train()
            t_loss, correct, total = 0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                t_loss += loss.item()
                _, pred = outputs.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            
            # Validation
            model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    v_loss += criterion(outputs, labels).item()
                    _, pred = outputs.max(1)
                    v_correct += (pred == labels).sum().item()
                    v_total += labels.size(0)
            
            t_acc = correct/total
            v_acc = v_correct/v_total
            scheduler.step(v_acc)
            
            history['t_loss'].append(t_loss/len(train_loader))
            history['t_acc'].append(t_acc)
            history['v_loss'].append(v_loss/len(val_loader))
            history['v_acc'].append(v_acc)
            
            print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f} | Val Acc {v_acc:.4f}")
            
            if v_acc > best_acc:
                best_acc = v_acc
                counter = 0
                torch.save(model.state_dict(), "tri_stream_weights.pth")
            else:
                counter += 1
                if counter >= patience:
                    print("Early Stopping!")
                    break

    # 5. Final Test
    model.load_state_dict(torch.load("tri_stream_weights.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes))
    
    # Save Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=train_ds.classes, yticklabels=train_ds.classes)
    plt.savefig("TriChannel_Confusion.png")
    print("Confusion Matrix saved.")