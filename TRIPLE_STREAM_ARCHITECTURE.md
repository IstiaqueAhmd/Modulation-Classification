# Triple-Stream Model Architecture

## Overview
The model combines three parallel processing streams to classify modulation types:

### Stream 1: Amplitude Scalogram (224×224×1)
- Input: CWT amplitude scalogram
- Architecture: CNN with depthwise separable convolutions
- Output: 64 feature maps of size 13×13

### Stream 2: Phase Scalogram (224×224×1)
- Input: CWT phase scalogram
- Architecture: Identical to Stream 1
- Output: 64 feature maps of size 13×13

### Stream 3: Raw IQ Signal (1024×2)
- Input: Raw I/Q components from .npy files
- Architecture: 1D CNN for time-series processing
- Layers:
  - Conv1D(2→32): kernel=7, stride=2 → MaxPool
  - Conv1D(32→64): kernel=5 → MaxPool
  - Conv1D(64→128): kernel=3 → MaxPool
  - Conv1D(128→64): kernel=3 → AdaptiveAvgPool(169)
  - Reshape to (64, 13, 13) to match other streams
- Output: 64 feature maps of size 13×13

## Fusion Strategy
Cross-attention mechanism combines features from all three streams:
- Amplitude ⟷ Phase
- Phase ⟷ IQ
- Amplitude ⟷ IQ

The three attended feature maps are averaged and passed through:
1. Residual block with SE attention
2. Global average pooling
3. Classification head (64→128→num_classes)

## Key Features
- **Multi-modal learning**: Learns from both time-domain (IQ) and time-frequency (scalogram) representations
- **Cross-attention**: Enables feature interaction between streams
- **Efficient**: Uses depthwise separable convolutions
- **Robust**: Combines complementary representations

## Data Flow
```
Input:
├─ Amplitude Scalogram (1, 224, 224) → Stream 1 → (64, 13, 13)
├─ Phase Scalogram (1, 224, 224)     → Stream 2 → (64, 13, 13)
└─ IQ Signal (2, 1024)                → Stream 3 → (64, 13, 13)
                                               ↓
                                      Cross Attention Fusion
                                               ↓
                                      Classification Head
                                               ↓
                                         Class Prediction
```

## Training
- Dataset: Loads scalograms from `Scalograms/snr_XX/` and IQ data from `Dataset/snr_XX/`
- Automatic train/val/test split (80/10/10)
- Normalization based on training set statistics
- Adam optimizer with ReduceLROnPlateau scheduler
- Best model saved based on validation accuracy

## Usage
```bash
python TripleStreamModel.py
```

Configure SNR level and other parameters at the bottom of the script.
