import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

# Import the model architecture from Model.py
from Model import DualStreamCNN, ScalogramDataset


def test_model(test_dir, model_path, output_prefix="test_results", batch_size=64):
    """
    Test a trained model on a specific dataset directory.
    
    Args:
        test_dir: Directory containing test data
        model_path: Path to saved model weights (.pth file)
        output_prefix: Prefix for output files
        batch_size: Batch size for testing
    """
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist!")
        return
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist!")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing Model on: {test_dir}")
    print(f"Model weights: {model_path}")
    print(f"{'='*80}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load test dataset
    # Note: Using basic normalization - you may want to use the same stats as training
    test_transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5], [0.5, 0.5])  # Basic normalization
    ])
    
    test_dataset = ScalogramDataset(test_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    num_classes = len(test_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {test_dataset.classes}")
    print(f"Total test samples: {len(test_dataset)}\n")
    
    # Load model
    model = DualStreamCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Testing
    print("Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i + 1) * batch_size}/{len(test_dataset)} samples")
    
    print("Inference complete!\n")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"{'='*80}")
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*80}\n")
    
    # Classification report
    print("Classification Report:")
    print("-" * 80)
    report = classification_report(all_labels, all_preds, 
                                   target_names=test_dataset.classes,
                                   digits=4)
    print(report)
    
    # Save classification report to file
    report_dict = classification_report(all_labels, all_preds, 
                                       target_names=test_dataset.classes,
                                       output_dict=True, digits=4)
    df_report = pd.DataFrame(report_dict).transpose()
    report_file = f"{output_prefix}_classification_report.csv"
    df_report.to_csv(report_file)
    print(f"\nClassification report saved to: {report_file}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Confusion Matrix (Raw Counts)\nAccuracy: {accuracy:.4f}')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f'Confusion Matrix (Normalized)\nAccuracy: {accuracy:.4f}')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    cm_file = f"{output_prefix}_confusion_matrix.png"
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_file}")
    plt.show()
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 80)
    per_class_acc = cm_normalized.diagonal()
    for i, class_name in enumerate(test_dataset.classes):
        print(f"{class_name:20s}: {per_class_acc[i]:.4f} ({per_class_acc[i]*100:.2f}%)")
    
    # Save detailed results
    results = {
        'test_directory': test_dir,
        'model_path': model_path,
        'overall_accuracy': accuracy,
        'num_samples': len(test_dataset),
        'num_classes': num_classes,
        'classes': test_dataset.classes,
        'per_class_accuracy': dict(zip(test_dataset.classes, per_class_acc.tolist()))
    }
    
    results_file = f"{output_prefix}_summary.txt"
    with open(results_file, 'w') as f:
        f.write(f"Test Results Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Test Directory: {test_dir}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Total Samples: {len(test_dataset)}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")
        f.write(f"Per-Class Accuracy:\n")
        f.write(f"{'-'*80}\n")
        for class_name, acc in results['per_class_accuracy'].items():
            f.write(f"{class_name:20s}: {acc:.4f} ({acc*100:.2f}%)\n")
    
    print(f"\nResults summary saved to: {results_file}")
    print(f"\n{'='*80}")
    print("Testing complete!")
    print(f"{'='*80}\n")
    
    return results


def test_multiple_snr_levels(base_dir_pattern, snr_levels, model_path, output_dir="test_results"):
    """
    Test model on multiple SNR levels and generate comparative analysis.
    
    Args:
        base_dir_pattern: Pattern for test directories (e.g., "Scalograms/snr_{}/")
                         Use {} as placeholder for SNR value
        snr_levels: List of SNR levels to test
        model_path: Path to saved model weights
        output_dir: Directory to save all test results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    print(f"\n{'#'*80}")
    print(f"# TESTING MODEL ACROSS MULTIPLE SNR LEVELS")
    print(f"# SNR Levels: {snr_levels}")
    print(f"# Model: {model_path}")
    print(f"{'#'*80}\n")
    
    for snr in snr_levels:
        # Construct test directory path
        if '{}' in base_dir_pattern:
            test_dir = base_dir_pattern.format(snr)
        else:
            test_dir = base_dir_pattern.replace('SNR', str(snr))
        
        output_prefix = os.path.join(output_dir, f"snr_{snr}")
        
        try:
            results = test_model(test_dir, model_path, output_prefix)
            all_results[snr] = results
        except Exception as e:
            print(f"Error testing SNR {snr}: {e}")
            continue
    
    # Generate comparative plot
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("Generating comparative analysis...")
        print(f"{'='*80}\n")
        
        snr_values = sorted(all_results.keys())
        accuracies = [all_results[snr]['overall_accuracy'] for snr in snr_values]
        
        # Plot accuracy vs SNR
        plt.figure(figsize=(12, 6))
        plt.plot(snr_values, [acc * 100 for acc in accuracies], 
                marker='o', linewidth=2, markersize=8, color='steelblue')
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Model Accuracy vs SNR Level', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(snr_values)
        
        # Add value labels on points
        for snr, acc in zip(snr_values, accuracies):
            plt.annotate(f'{acc*100:.2f}%', 
                        xy=(snr, acc*100), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9)
        
        plt.tight_layout()
        comparison_file = os.path.join(output_dir, "snr_comparison.png")
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        print(f"SNR comparison plot saved to: {comparison_file}")
        plt.show()
        
        # Save comparison summary
        summary_file = os.path.join(output_dir, "snr_comparison_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("SNR Comparison Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Model: {model_path}\n\n")
            f.write(f"{'SNR (dB)':<12}{'Accuracy':<15}{'Samples':<15}\n")
            f.write(f"{'-'*42}\n")
            for snr in snr_values:
                acc = all_results[snr]['overall_accuracy']
                samples = all_results[snr]['num_samples']
                f.write(f"{snr:<12}{acc:.4f} ({acc*100:.2f}%){samples:>10}\n")
        
        print(f"Comparison summary saved to: {summary_file}")
    
    print(f"\n{'#'*80}")
    print("# ALL TESTING COMPLETE")
    print(f"{'#'*80}\n")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained model on different SNR levels')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Path to test directory (e.g., Dataset(Splitted)/combined/test)')
    parser.add_argument('--model_path', type=str, default='model.pth',
                       help='Path to trained model weights (.pth file)')
    parser.add_argument('--output_prefix', type=str, default='test_results',
                       help='Prefix for output files')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for testing')
    
    # Multi-SNR testing options
    parser.add_argument('--multi_snr', action='store_true',
                       help='Test on multiple SNR levels')
    parser.add_argument('--snr_pattern', type=str, default='Scalograms/snr_{}',
                       help='Pattern for SNR directories (use {} for SNR placeholder)')
    parser.add_argument('--snr_levels', type=int, nargs='+', 
                       default=[30, 20, 10, 0, -10, -20],
                       help='List of SNR levels to test')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save all test results')
    
    args = parser.parse_args()
    
    if args.multi_snr:
        # Test on multiple SNR levels
        test_multiple_snr_levels(
            base_dir_pattern=args.snr_pattern,
            snr_levels=args.snr_levels,
            model_path=args.model_path,
            output_dir=args.output_dir
        )
    else:
        # Single directory test
        if args.test_dir is None:
            print("Error: --test_dir is required for single directory testing")
            print("Example: python test.py --test_dir Scalograms/snr_10")
            print("\nOr use --multi_snr for testing multiple SNR levels:")
            print("Example: python test.py --multi_snr --snr_levels 30 20 10 0 -10 -20")
        else:
            test_model(
                test_dir=args.test_dir,
                model_path=args.model_path,
                output_prefix=args.output_prefix,
                batch_size=args.batch_size
            )
