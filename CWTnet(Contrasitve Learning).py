import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# CONFIGURATION
# =====================
TRAIN = True            
SNR = "30"              
BATCH_SIZE = 32         
NUM_EPOCHS = 30
LEARNING_RATE = 0.0003  
TEMP = 0.5              

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# ==========================================
# 1. Advanced Modules (Attention & Blocks)
# ==========================================

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention: Captures long-range dependencies along 
    Time (W) and Frequency (H) axes independently. 
    Crucial for Scalograms where X and Y axes mean different things.
    """
    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Hsigmoid()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Process Frequency (H) and Time (W) separately
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

class InvertedResidualBlock(nn.Module):
    """
    Efficient Block: Expansion -> Depthwise Conv -> Pointwise Conv -> Attention
    """
    def __init__(self, in_ch, out_ch, expand_ratio=2, stride=1):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            # Pointwise Expansion
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            # Depthwise Conv (3x3)
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Coordinate Attention Mechanism
            CoordinateAttention(hidden_dim),
            # Pointwise Linear Projection
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# ==========================================
# 2. Optimized Model Architecture
# ==========================================
class OptimizedDualStreamNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # --- Feature Extractor Builder ---
        def build_stream():
            return nn.Sequential(
                # Initial Stem: 7x7 replaced with stacked layers or rectangular kernels?
                # Using 5x5 stride 2 to capture initial wide context
                nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
                
                # Stack of Inverted Residuals with Coordinate Attention
                InvertedResidualBlock(32, 32, expand_ratio=1, stride=1),
                InvertedResidualBlock(32, 64, expand_ratio=2, stride=2),
                InvertedResidualBlock(64, 64, expand_ratio=2, stride=1),
                InvertedResidualBlock(64, 128, expand_ratio=2, stride=2),
                InvertedResidualBlock(128, 128, expand_ratio=2, stride=1),
                
                # Final Global Average Pooling
                nn.AdaptiveAvgPool2d(1)
            )

        # Separate independent streams
        self.stream_amp = build_stream()   # Amplitude (Texture/Intensity)
        self.stream_phase = build_stream() # Phase (Discontinuities/Cyclic)

        # --- Projection Head (For Contrastive Loss) ---
        # 128 * 2 streams = 256 input features
        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64) # Low dim embedding for SupCon
        )

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x is [B, 2, H, W]
        # Slice channels to process separately
        # Stream 1: Amplitude
        f_amp = self.stream_amp(x[:, 0:1]).flatten(1)
        
        # Stream 2: Phase
        f_phase = self.stream_phase(x[:, 1:2]).flatten(1)
        
        # Feature Fusion
        features = torch.cat([f_amp, f_phase], dim=1) # [B, 256]
        
        # Outputs
        logits = self.classifier(features)
        embeddings = self.projection_head(features) # Normalized in Loss function
        
        return logits, embeddings

# ==========================================
# 3. Losses & Utilities (Preserved)
# ==========================================
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # Prevent NaN if a class only has 1 sample in the batch
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        return loss

class ScalogramDataset(Dataset):
    def __init__(self, root_dir, indices=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.all_data = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir): continue
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.all_data.append((os.path.join(class_dir, file), label))
        
        if indices is not None: self.data = [self.all_data[i] for i in indices]
        else: self.data = self.all_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        # Optimization: mmap_mode='r' keeps data on disk until accessed, reducing RAM usage
        # copy() is needed because torch doesn't support negative strides from some numpy views
        scalogram = np.load(file_path, mmap_mode='r').transpose(2, 0, 1).copy()
        scalogram = torch.tensor(scalogram, dtype=torch.float32)
        return scalogram, label

def create_splits(root_dir, train_r=0.8, val_r=0.1, test_r=0.1):
    classes = sorted(os.listdir(root_dir))
    all_indices = []
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir): continue
        count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        indices = list(range(count))
        random.shuffle(indices)
        
        t = int(count * train_r)
        v = int(count * val_r)
        
        all_indices.append({'train': indices[:t], 'val': indices[t:t+v], 'test': indices[t+v:], 'name': class_name})

    g_train, g_val, g_test = [], [], []
    curr = 0
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir): continue
        count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        split = next(x for x in all_indices if x['name'] == class_name)
        
        g_train.extend([curr + x for x in split['train']])
        g_val.extend([curr + x for x in split['val']])
        g_test.extend([curr + x for x in split['test']])
        curr += count
    return g_train, g_val, g_test

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == '__main__':
    data_dir = f'Scalograms/snr_{SNR}' 
    
    print(f"--- Processing Data from: {data_dir} ---")
    
    # Check if dir exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        exit()

    train_idx, val_idx, test_idx = create_splits(data_dir)
    
    train_ds = ScalogramDataset(data_dir, indices=train_idx)
    val_ds = ScalogramDataset(data_dir, indices=val_idx)
    test_ds = ScalogramDataset(data_dir, indices=test_idx)
    
    CLASS_NAMES = train_ds.classes 
    print(f"Classes found: {CLASS_NAMES}")

    # num_workers=4 speeds up data loading significantly
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    model = OptimizedDualStreamNet(len(CLASS_NAMES)).to(device)
    
    Number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {Number_of_parameters:,} parameters.")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # AdamW is better for generalization
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) # Smooth LR decay
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=TEMP)

    # ---------------- TRAINING BLOCK ----------------
    if TRAIN:
        print(f"--- Starting Training ({NUM_EPOCHS} Epochs) ---")
        best_val_acc = 0
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                logits, embeds = model(inputs)
                
                l_ce = criterion_ce(logits, labels)
                l_con = criterion_supcon(embeds, labels)
                
                # Weighted loss: Balance classification vs clustering
                loss = l_ce + (0.7 * l_con) 
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                _, preds = logits.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            # Step Scheduler
            scheduler.step()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits, _ = model(inputs)
                    _, preds = logits.max(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/total:.3f} | Val Acc: {val_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
                print("  -> Saved Best Model")
        
        print("Training Complete.")

    # ---------------- EVALUATION BLOCK ----------------
    print("\n" + "="*40)
    print("      STARTING EVALUATION      ")
    print("="*40)

    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded weights from 'best_model.pth'")
    else:
        print("Warning: No weights found! Testing with initialized weights.")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix (SNR={SNR}dB)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()