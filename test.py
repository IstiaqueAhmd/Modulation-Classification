import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Import the model architecture and utilities from Model.py
from Model import DualStreamCNN, ScalogramDataset, calculate_dataset_stats

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================

TEST_DIR = 'Dataset(Splitted)/snr_30/test'       # Change this to your test directory
TRAIN_DIR = 'Dataset(Splitted)/snr_30/train'     # Training directory (for normalization stats)
MODEL_PATH = 'model.pth'                          # Path to your trained model
OUTPUT_NAME = 'test_snr_30'                       # Name for output files

# ============================================================================

if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"Testing Model")
    print(f"Test Directory: {TEST_DIR}")
    print(f"Model: {MODEL_PATH}")
    print(f"{'='*80}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Calculate normalization statistics from training data (same as during training)
    print("Calculating normalization statistics from training data...")
    temp_train = ScalogramDataset(TRAIN_DIR)
    mean, std = calculate_dataset_stats(temp_train)
    print(f"Using normalization - Mean: {mean}, Std: {std}\n")
    
    # Load test dataset (using same normalization as training)
    test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = ScalogramDataset(TEST_DIR, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print(f"Classes: {test_dataset.classes}")
    print(f"Total samples: {len(test_dataset)}\n")
    
    # Load model
    model = DualStreamCNN(len(test_dataset.classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("Running inference...\n")
    
    # Test the model
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Classification report
    print("Classification Report:")
    print("="*80)
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                ax=axes[0])
    axes[0].set_title(f'Confusion Matrix (Counts) - Accuracy: {accuracy:.4f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                ax=axes[1])
    axes[1].set_title(f'Confusion Matrix (Normalized) - Accuracy: {accuracy:.4f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {OUTPUT_NAME}_confusion_matrix.png")
    plt.show()
    
    print("\nTesting complete!")
