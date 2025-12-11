"""
Dual-Stream Patch-based Transformer for 2-channel Scalograms
- Input: .npy files shaped [H, W, 2] (channel order: Amp, Phase)
- Patchify each channel separately, run lightweight transformer encoders,
  perform token-level cross-attention between streams, fuse and classify.

Usage: place scalograms under `Scalograms/snr_<SNR>/<CLASS>/*.npy`
and run `python dual_stream_patch_vit_scalogram.py`.

Designed to be efficient (no HWxHW full spatial attention) and trainable on
common GPUs for moderate dataset sizes.

"""

import os
import random
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIG
# -------------------------
ROOT_DIR = 'Scalograms'          # expected folder: Scalograms/snr_<SNR>/<CLASS>/*.npy
SNR = '30'
DATA_DIR = os.path.join(ROOT_DIR, f'snr_{SNR}')
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
PATCH_SIZE = 16                  # 224x224 -> 14x14 tokens
EMBED_DIM = 128
NUM_HEADS = 4
MLP_RATIO = 2.0
NUM_LAYERS = 6
DROPOUT = 0.1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Dataset & Utilities
# -------------------------
class ScalogramDataset(Dataset):
    """Loads .npy scalogram files: shape [H, W, 2] (amp, phase).
    Returns tensor [2, H, W] floats in range ~[0,1].
    """
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        arr = np.load(path)
        # ensure shape and float32
        if arr.ndim == 2:
            # single-channel fallback (unlikely)
            arr = np.stack([arr, arr], axis=-1)
        arr = arr.astype(np.float32)
        # transpose to [C, H, W]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label


def make_splits(root_dir: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    """Create stratified per-class splits returning lists of (filepath,label).
    """
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_items, val_items, test_items = [], [], []

    for c in classes:
        cdir = os.path.join(root_dir, c)
        files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith('.npy')]
        files.sort()
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = files[:n_train]
        val = files[n_train:n_train + n_val]
        test = files[n_train + n_val:]
        train_items += [(p, class_to_idx[c]) for p in train]
        val_items += [(p, class_to_idx[c]) for p in val]
        test_items += [(p, class_to_idx[c]) for p in test]

    random.shuffle(train_items)
    random.shuffle(val_items)
    random.shuffle(test_items)

    return train_items, val_items, test_items, classes


# Simple augmentations for scalograms (tensor [C,H,W])
class RandomTimeShift(object):
    """Shift along time axis (width) by a random offset, wrap-around.
    """
    def __init__(self, max_shift=16):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor):
        _, H, W = x.shape
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=2)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor):
        return x + torch.randn_like(x) * self.std + self.mean


# -------------------------
# Patch embedding + Transformer blocks
# -------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, embed_dim=EMBED_DIM, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C=1, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x, (H, W)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=MLP_RATIO, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        # x: [B, N, dim]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionLayer(nn.Module):
    """Allows A to attend to B: queries from A, keys/values from B."""
    def __init__(self, dim, num_heads, dropout=DROPOUT):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, a, b):
        # a, b: [B, N, dim]
        q = self.norm_q(a)
        kv = self.norm_kv(b)
        attended = self.attn(q, kv, kv)[0]
        return a + self.proj(attended)


# -------------------------
# Dual-Stream Patch Transformer
# -------------------------
class DualStreamPatchTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS):
        super().__init__()
        # Per-stream patch embedding (single-channel each)
        self.patch_amp = PatchEmbed(in_ch=1, embed_dim=embed_dim)
        self.patch_phase = PatchEmbed(in_ch=1, embed_dim=embed_dim)

        # positional embeddings (shared size will be determined at forward time)
        self.pos_embed = None

        # stacks of encoder layers for each stream
        self.amp_encoders = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.phase_encoders = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])

        # cross-attention layers to allow interaction (A->P and P->A)
        self.cross_layers = nn.ModuleList([nn.ModuleDict({
            'a_to_p': CrossAttentionLayer(embed_dim, num_heads),
            'p_to_a': CrossAttentionLayer(embed_dim, num_heads)
        }) for _ in range(num_layers)])

        # classifier head
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, 2, H, W] channels: 0=amp,1=phase
        a = x[:, 0:1]
        p = x[:, 1:2]

        a_tokens, (h, w) = self.patch_amp(a)
        p_tokens, _ = self.patch_phase(p)
        # a_tokens: [B, N, dim]

        B, N, D = a_tokens.shape
        # create / expand positional embeddings lazily
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
            self.pos_embed = nn.Parameter(torch.zeros(1, N, D)).to(a_tokens.device)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        a = a_tokens + self.pos_embed
        p = p_tokens + self.pos_embed

        # pass through layers
        for enc_a, enc_p, cross in zip(self.amp_encoders, self.phase_encoders, self.cross_layers):
            a = enc_a(a)
            p = enc_p(p)
            # cross-attention (bidirectional)
            a = cross['p_to_a'](a, p)
            p = cross['a_to_p'](p, a)

        # global pooling (mean over tokens)
        a_pool = a.mean(dim=1)
        p_pool = p.mean(dim=1)

        fused = torch.cat([a_pool, p_pool], dim=1)
        out = self.norm(fused)
        logits = self.head(out)
        return logits


# -------------------------
# Training / Evaluation
# -------------------------

def accuracy(preds, labels):
    return (preds == labels).mean()


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        running_acc += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_acc / total


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_acc += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, running_acc / total, all_preds, all_labels


# -------------------------
# Runner
# -------------------------
if __name__ == '__main__':
    assert os.path.isdir(DATA_DIR), f"Data dir not found: {DATA_DIR}"

    train_items, val_items, test_items, classes = make_splits(DATA_DIR)
    print(f"Classes: {classes}")
    print(f"Train / Val / Test: {len(train_items)} / {len(val_items)} / {len(test_items)}")

    # transforms
    train_transform = transforms.Compose([
        RandomTimeShift(max_shift=16),
        AddGaussianNoise(std=0.01)
    ])
    eval_transform = None

    train_ds = ScalogramDataset(train_items, transform=train_transform)
    val_ds = ScalogramDataset(val_items, transform=eval_transform)
    test_ds = ScalogramDataset(test_items, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = DualStreamPatchTransformer(num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_weights.pth')
            print("Saved best weights.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load('best_weights.pth'))
    test_loss, test_acc, preds, labels = eval_model(model, test_loader, criterion, DEVICE)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=classes, digits=4))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f'Confusion Matrix SNR={SNR}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_snr{SNR}.png', dpi=200)
    print('Saved confusion matrix.')
