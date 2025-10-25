import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Custom normalization for 1D signals
class SignalNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(2, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(2, 1)
    
    def __call__(self, signal):
        # signal shape: (2, 1024)
        return (signal - self.mean) / self.std

class SignalDataset(Dataset):
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
        signal = np.load(file_path)  # Shape: (1024, 2)
        signal = torch.tensor(signal, dtype=torch.float32).T  # Transpose to (2, 1024)
        if self.transform:
            signal = self.transform(signal)
        return signal, label


def calculate_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(2)
    std = torch.zeros(2)
    for inputs, _ in loader:
        for i in range(2):
            mean[i] += inputs[:, i].mean()
            std[i] += inputs[:, i].std()
    return (mean / len(loader)).tolist(), (std / len(loader)).tolist()


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# Single Transformer Stream
class TransformerStream(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # Project 1D signal to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1024)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, return_sequence=False):
        # x shape: (batch_size, 1024) - single channel
        x = x.unsqueeze(-1)  # (batch_size, 1024, 1)
        x = self.input_projection(x)  # (batch_size, 1024, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch_size, 1024, d_model)
        
        if return_sequence:
            return x  # Return full sequence for latent attention
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        return x


# Multi-Head Latent Attention Module
class LatentAttention(nn.Module):
    def __init__(self, d_model, num_latents=32, nhead=4, dropout=0.1):
        super().__init__()
        self.num_latents = num_latents
        self.d_model = d_model
        
        # Learnable latent queries
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model))
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size = x.size(0)
        
        # Expand latent queries for batch
        queries = self.latent_queries.expand(batch_size, -1, -1)  # (batch_size, num_latents, d_model)
        
        # Cross-attention: latents attend to input sequence
        attn_output, _ = self.cross_attention(
            query=queries,
            key=x,
            value=x
        )
        
        # Residual connection and normalization
        output = self.norm(queries + attn_output)
        
        return output  # (batch_size, num_latents, d_model)


# Dual Stream Transformer with Latent Attention
class DualStreamTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, 
                 num_latents=32, dropout=0.1, use_latent_attention=True):
        super().__init__()
        self.use_latent_attention = use_latent_attention
        
        # Two separate streams for two channels
        self.stream1 = TransformerStream(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.stream2 = TransformerStream(d_model, nhead, num_layers, dim_feedforward, dropout)
        
        if use_latent_attention:
            # Latent attention for each stream
            self.latent_attn1 = LatentAttention(d_model, num_latents, nhead, dropout)
            self.latent_attn2 = LatentAttention(d_model, num_latents, nhead, dropout)
            
            # Cross-stream latent attention
            self.cross_stream_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            
            # Fusion operates on latent representations
            fusion_input_dim = num_latents * d_model * 2
        else:
            # Simple fusion without latent attention
            fusion_input_dim = d_model * 2
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 2, 1024)
        
        if self.use_latent_attention:
            # Get sequence outputs from both streams
            x1_seq = self.stream1(x[:, 0, :], return_sequence=True)  # (batch_size, 1024, d_model)
            x2_seq = self.stream2(x[:, 1, :], return_sequence=True)  # (batch_size, 1024, d_model)
            
            # Apply latent attention to compress sequences
            x1_latent = self.latent_attn1(x1_seq)  # (batch_size, num_latents, d_model)
            x2_latent = self.latent_attn2(x2_seq)  # (batch_size, num_latents, d_model)
            
            # Cross-stream attention: let stream1 latents attend to stream2 latents
            x1_enhanced, _ = self.cross_stream_attn(
                query=x1_latent,
                key=x2_latent,
                value=x2_latent
            )
            x1_latent = self.norm(x1_latent + x1_enhanced)
            
            # Flatten latent representations
            x1_flat = x1_latent.flatten(1)  # (batch_size, num_latents * d_model)
            x2_flat = x2_latent.flatten(1)  # (batch_size, num_latents * d_model)
            
            # Concatenate and fuse
            x = torch.cat([x1_flat, x2_flat], dim=1)
        else:
            # Standard pooling without latent attention
            x1 = self.stream1(x[:, 0, :])  # (batch_size, d_model)
            x2 = self.stream2(x[:, 1, :])  # (batch_size, d_model)
            x = torch.cat([x1, x2], dim=1)
        
        x = self.fusion(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # Control switch: Set to True to enable training/validation, False to skip to testing only
    ENABLE_TRAINING = True
    SNR = "30"
    # Dataset setup
    train_dir = f'Dataset(Splitted)/snr_{SNR}/train'
    val_dir = f'Dataset(Splitted)/snr_{SNR}/val'
    test_dir = f'Dataset(Splitted)/snr_{SNR}/test'

    # Calculate dataset stats
    temp_train = SignalDataset(train_dir)
    mean, std = calculate_dataset_stats(temp_train)
    print(f"Dataset stats - Mean: {mean}, Std: {std}")

    # Data transforms - using custom SignalNormalize
    train_transform = SignalNormalize(mean, std)
    val_test_transform = SignalNormalize(mean, std)

    # Create datasets
    train_dataset = SignalDataset(train_dir, train_transform)
    val_dataset = SignalDataset(val_dir, val_test_transform)
    test_dataset = SignalDataset(test_dir, val_test_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup - Using DualStreamTransformer with Latent Attention
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = DualStreamTransformer(
        num_classes=len(train_dataset.classes),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_latents=32,  # Number of learnable latent tokens
        dropout=0.1,
        use_latent_attention=True  # Set to False to disable latent attention
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    if ENABLE_TRAINING:
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
                torch.save(model.state_dict(), f"model{SNR}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    else:
        print("Training skipped. Loading pre-trained model...")
        criterion = nn.CrossEntropyLoss()

    # Final evaluation
    model.load_state_dict(torch.load(f"model{SNR}.pth"))
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
    if ENABLE_TRAINING:
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
        plt.savefig(f"confusion_matrices_{SNR}.png")
        plt.show()
    else:
        # Only confusion matrix when training is disabled
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=train_dataset.classes,
                    yticklabels=train_dataset.classes)
        plt.title("Confusion Matrix (Test Set)")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{SNR}.png")
        plt.show()

