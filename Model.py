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
torch.backends.cudnn.benchmark = False

class ScalogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.data.append((os.path.join(class_dir, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        scalogram = np.load(file_path).transpose(2, 0, 1)
        scalogram = torch.tensor(scalogram, dtype=torch.float32)
        if self.transform:
            scalogram = self.transform(scalogram)
        return scalogram, label


def calculate_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(2)
    std = torch.zeros(2)
    for inputs, _ in loader:
        for i in range(2):
            mean[i] += inputs[:, i].mean()
            std[i] += inputs[:, i].std()
    return (mean / len(loader)).tolist(), (std / len(loader)).tolist()


# Depthwise-Separable Conv Block
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


# One stream (amplitude OR phase)
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
    def __init__(self, num_classes, dropout=0.5):
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
        xa = self.stream_amp(x[:, 0:1])     # amplitude channel
        xp = self.stream_phase(x[:, 1:2])   # phase channel
        x = torch.cat([xa, xp], 1)          
        return self.classifier(self.fuse(x))


if __name__ == '__main__':
    # Set SNR
    snr = "-20"
    
    # Dataset setup
    train_dir = f'Dataset(Splitted)/snr_{snr}/train'
    val_dir = f'Dataset(Splitted)/snr_{snr}/val'
    test_dir = f'Dataset(Splitted)/snr_{snr}/test'

    # Calculate dataset stats
    temp_train = ScalogramDataset(train_dir)
    mean, std = calculate_dataset_stats(temp_train)
    print(f"Dataset stats - Mean: {mean}, Std: {std}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean, std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    # Create datasets
    train_dataset = ScalogramDataset(train_dir, train_transform)
    val_dataset = ScalogramDataset(val_dir, val_test_transform)
    test_dataset = ScalogramDataset(test_dir, val_test_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DualStreamCNN(len(train_dataset.classes)).to(device)
    print(sum(p.numel() for p in model.parameters()))
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Store Losses
    train_losses = []
    val_losses = []

    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    patience = 7
    num_epochs = 50

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

        epoch_loss = train_loss / len(train_loader)
        train_losses.append(epoch_loss)

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

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Calculate metrics
        train_acc = correct / total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"Learning rate reduced to {new_lr:.6f}")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f} | Acc: {val_acc:.4f}\n")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final evaluation
    model.load_state_dict(torch.load("model.pth"))
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
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")

    #Loss track
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.show()

