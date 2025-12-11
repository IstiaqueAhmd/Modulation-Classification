import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm  # <--- NEW IMPORT

# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

set_seed(42)

# ==========================================
# 2. DATASET CLASS
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
            # Sort for determinism before shuffling
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            for file in files:
                self.all_data.append((os.path.join(class_dir, file), label))
        
        # Apply indices if provided
        if indices is not None:
            self.data = [self.all_data[i] for i in indices]
        else:
            self.data = self.all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        try:
            # Load [224, 224, 2] -> Transpose to [2, 224, 224]
            scalogram = np.load(file_path).transpose(2, 0, 1)
            scalogram = torch.tensor(scalogram, dtype=torch.float32)
            
            if self.transform:
                scalogram = self.transform(scalogram)
            return scalogram, label
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((2, 224, 224)), label

# ==========================================
# 3. UTILS: SPLIT & DYNAMIC NORMALIZATION
# ==========================================
def create_randomized_split(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    classes = sorted(os.listdir(root_dir))
    global_train, global_val, global_test = [], [], []
    
    current_idx = 0
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        num_files = len(files)
        
        # Local Shuffle
        indices = list(range(num_files))
        random.shuffle(indices) 
        
        train_end = int(num_files * train_ratio)
        val_end = int(num_files * (train_ratio + val_ratio))
        
        # Map to global indices
        global_train.extend([current_idx + i for i in indices[:train_end]])
        global_val.extend([current_idx + i for i in indices[train_end:val_end]])
        global_test.extend([current_idx + i for i in indices[val_end:]])
        
        current_idx += num_files
    
    return global_train, global_val, global_test

def compute_dataset_stats(dataset):
    """
    Computes mean and std over the dataset specifically for normalization.
    """
    print("Computing dataset mean and std (this may take a moment)...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for data, _ in tqdm(loader, desc="Calculating Stats"):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    print(f"Computed Mean: {mean}, Std: {std}")
    return mean.tolist(), std.tolist()

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch, momentum=0.05), # Lower momentum for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.05),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

def make_stream():
    return nn.Sequential(
        nn.Conv2d(1, 16, 7, 2, 3, bias=False),
        nn.BatchNorm2d(16, momentum=0.05), 
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
        x = torch.cat([xa, xp], 1)          
        return self.classifier(self.fuse(x))

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    TRAIN = True
    SNR = "30"
    BATCH_SIZE = 64
    EPOCHS = 50
    data_dir = f'Scalograms/snr_{SNR}'
    
    # 1. Randomized Split
    print("Creating randomized train/val/test splits...")
    train_indices, val_indices, test_indices = create_randomized_split(
        data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # 2. Dynamic Normalization
    # Create a temporary dataset without transforms to calculate stats
    temp_dataset = ScalogramDataset(data_dir, transform=None, indices=train_indices)
    norm_mean, norm_std = compute_dataset_stats(temp_dataset)

    # 3. Setup Transforms with Calculated Stats
    train_transform = transforms.Compose([
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)) 
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 4. Create Final Datasets & Loaders
    train_dataset = ScalogramDataset(data_dir, train_transform, indices=train_indices)
    val_dataset = ScalogramDataset(data_dir, val_test_transform, indices=val_indices)
    test_dataset = ScalogramDataset(data_dir, val_test_transform, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, pin_memory=True)

    # 5. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DualStreamCNN(len(train_dataset.classes)).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 6. Training Loop
    if TRAIN:
        best_val_acc = 0.0
        patience = 10
        counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"\nStarting training for {EPOCHS} epochs...")
        
        for epoch in range(EPOCHS):
            model.train()
            run_loss = 0.0
            correct = 0
            total = 0
            
            # TQDM for Training
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
            
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                run_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

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
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print stats after progress bar clears
            print(f"Epoch {epoch+1} Results | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Early Stopping Check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), f"best_model_snr{SNR}.pth")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Plotting results
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

    # 7. Final Evaluation
    if os.path.exists(f"best_model_snr{SNR}.pth"):
        model.load_state_dict(torch.load(f"best_model_snr{SNR}.pth"))
        print("\nLoaded best model for evaluation.")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running evaluation on Test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title(f'Confusion Matrix SNR {SNR}')
    plt.tight_layout()
    plt.savefig(f'Confusion_Matrix_snr{SNR}.png')
    print("Done.")