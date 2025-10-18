import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import argparse

# Import the model classes from Model.py
from Model import ScalogramDataset, DualStreamCNN, DSConv, calculate_dataset_stats

def load_model(model_path, num_classes, device):
    """Load the trained model from checkpoint"""
    model = DualStreamCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test data and return predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Get predictions
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and save confusion matrix"""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot absolute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix (Absolute Counts)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, cm_normalized

def print_detailed_metrics(y_true, y_pred, class_names):
    """Print detailed classification metrics"""
    print("="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n")
    
    # Classification report
    print("Classification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"{class_name:12s}: {acc:.4f} ({acc*100:.2f}%)")

def analyze_misclassifications(y_true, y_pred, class_names, top_k=5):
    """Analyze the most common misclassifications"""
    print(f"\nTop {top_k} Most Common Misclassifications:")
    print("-" * 50)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Find misclassifications (off-diagonal elements)
    misclassifications = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclassifications.append((cm[i, j], class_names[i], class_names[j]))
    
    # Sort by frequency
    misclassifications.sort(reverse=True)
    
    for count, true_class, pred_class in misclassifications[:top_k]:
        percentage = (count / cm[class_names.index(true_class)].sum()) * 100
        print(f"{true_class:12s} → {pred_class:12s}: {count:3d} samples ({percentage:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Test the modulation classification model')
    parser.add_argument('--model_path', type=str, default='model.pth', 
                       help='Path to the trained model file')
    parser.add_argument('--snr', type=str, default='10', 
                       help='SNR level to test on')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Path to test directory (if different from default structure)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for testing')
    parser.add_argument('--save_plots', type=str, default='test_results.png',
                       help='Path to save confusion matrix plots')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set test directory
    if args.test_dir is None:
        test_dir = f'Dataset(Splitted)/snr_{args.snr}/test'
    else:
        test_dir = args.test_dir
    
    print(f"Testing on: {test_dir}")
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist!")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist!")
        return
    
    # Create test dataset to get dataset statistics
    print("Loading test dataset...")
    temp_dataset = ScalogramDataset(test_dir)
    
    # Calculate dataset stats (you might want to use the same stats from training)
    train_dir = f'Dataset(Splitted)/snr_{args.snr}/train'
    if os.path.exists(train_dir):
        print("Calculating dataset statistics from training data...")
        temp_train = ScalogramDataset(train_dir)
        mean, std = calculate_dataset_stats(temp_train)
    else:
        print("Training directory not found, calculating stats from test data...")
        mean, std = calculate_dataset_stats(temp_dataset)
    
    print(f"Dataset stats - Mean: {mean}, Std: {std}")
    
    # Create test transform
    test_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    # Create test dataset and loader
    test_dataset = ScalogramDataset(test_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Number of classes: {len(test_dataset.classes)}")
    print(f"Classes: {test_dataset.classes}")
    
    # Load model
    print("Loading trained model...")
    model = load_model(args.model_path, len(test_dataset.classes), device)
    
    # Evaluate model
    print("Evaluating model on test data...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device, test_dataset.classes)
    
    # Print detailed metrics
    print_detailed_metrics(y_true, y_pred, test_dataset.classes)
    
    # Analyze misclassifications
    analyze_misclassifications(y_true, y_pred, test_dataset.classes)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm, cm_normalized = plot_confusion_matrix(y_true, y_pred, test_dataset.classes, args.save_plots)
    
    # Save detailed results to file
    results_file = f"test_results_snr_{args.snr}.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODULATION CLASSIFICATION TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Directory: {test_dir}\n")
        f.write(f"Test Dataset Size: {len(test_dataset)} samples\n")
        f.write(f"Number of Classes: {len(test_dataset.classes)}\n")
        f.write(f"Classes: {test_dataset.classes}\n")
        f.write(f"Device: {device}\n")
        f.write("\n")
        
        accuracy = accuracy_score(y_true, y_pred)
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 60 + "\n")
        report = classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4)
        f.write(report)
        f.write("\n")
        
        f.write("Per-Class Accuracy:\n")
        f.write("-" * 40 + "\n")
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        for class_name, acc in zip(test_dataset.classes, per_class_acc):
            f.write(f"{class_name:12s}: {acc:.4f} ({acc*100:.2f}%)\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print("\nTesting completed successfully!")

if __name__ == '__main__':
    main()