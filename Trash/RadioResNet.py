import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models # Added torchvision for ResNet
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
# 1. Model Architecture (ResNet34)
# ==========================================

class RadioResNet(nn.Module):
    def __init__(self, num_classes):
        super(RadioResNet, self).__init__()
        
        # 1. Load Pre-trained ResNet34
        # We use weights=DEFAULT to get the best ImageNet features
        weights = models.ResNet34_Weights.DEFAULT
        self.backbone = models.resnet34(weights=weights)
        
        # 2. Extract Feature Dimension (ResNet34 outputs 512 features before FC)
        self.num_ftrs = self.backbone.fc.in_features
        
        # 3. Remove the original Classification Head
        # We replace it with Identity so we can access the raw features
        self.backbone.fc = nn.Identity()

        # 4. Projection Head (Required for SupConLoss)
        # Projects features to a lower dimension for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 128) # Embedding size
        )

        # 5. Classification Head (The actual classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),                 # Regularization
            nn.Linear(self.num_ftrs, 256),
            nn.BatchNorm1d(256),             # Stability
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        
        # Pass through ResNet Backbone
        features = self.backbone(x) # Output: [Batch, 512]
        
        # Calculate Logits (for CrossEntropy)
        logits = self.classifier(features)
        
        # Calculate Embeddings (for SupConLoss)
        embeddings = self.projection_head(features)
        
        return logits, embeddings

# ==========================================
# 2. Losses & Utilities (Preserved)
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
        # Load [H, W, 3] -> Transpose to [3, H, W]
        scalogram = np.load(file_path, mmap_mode='r')
        scalogram = scalogram.transpose(2, 0, 1).copy() 
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
# 3. Main Execution
# ==========================================
if __name__ == '__main__':
    # Ensure this directory matches your new IQA generator output
    data_dir = f'Scalograms_Complex_Log/snr_{SNR}' 
    
    print(f"--- Processing Data from: {data_dir} ---")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found. Please check your folder name.")
        exit()

    train_idx, val_idx, test_idx = create_splits(data_dir)
    
    train_ds = ScalogramDataset(data_dir, indices=train_idx)
    val_ds = ScalogramDataset(data_dir, indices=val_idx)
    test_ds = ScalogramDataset(data_dir, indices=test_idx)
    
    CLASS_NAMES = train_ds.classes 
    print(f"Classes found: {CLASS_NAMES}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # --- CHANGED: Initialize Standard ResNet Model ---
    model = RadioResNet(len(CLASS_NAMES)).to(device)
    
    Number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {Number_of_parameters:,} parameters (ResNet34).")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
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
                
                # ResNet now returns both logits and embeddings
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