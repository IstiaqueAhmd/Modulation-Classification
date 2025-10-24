import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Import the model classes from Model.py
from Model import ScalogramDataset, DualStreamCNN, calculate_dataset_stats

def main():
    # Configuration
    snr = "10"
    model_path = "model.pth"
    test_dir = f'Dataset(Splitted)/snr_{snr}/test'
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ScalogramDataset(test_dir)
    
    # Calculate dataset stats from test data (since train/val folders are empty)
    print("Calculating dataset statistics from test data...")
    mean, std = calculate_dataset_stats(test_dataset)
    print(f"Dataset stats - Mean: {mean}, Std: {std}")
    
    # Apply transforms
    test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    # Create test dataset with transforms
    test_dataset = ScalogramDataset(test_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Load model
    print("Loading model...")
    model = DualStreamCNN(len(test_dataset.classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification Report
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrices
    plt.figure(figsize=(15, 6))
    
    # Absolute confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Normalized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Overall accuracy
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == '__main__':
    main()