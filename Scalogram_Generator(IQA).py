"""
==============================================================
RadioML Generator: I / Q / Amplitude Strategy
==============================================================

Generates 3-channel scalograms using the fundamental signal
components to avoid phase-noise artifacts.

Channel Mapping:
1. I-Component (In-Phase)
2. Q-Component (Quadrature)
3. Magnitude (Amplitude Envelope)

Output: [224 x 224 x 3] Normalized NumPy arrays.
==============================================================
"""

import os
import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

# =====================
# CONFIGURATION
# =====================
SNR_LEVELS = [30]           # List of SNRs to process (e.g. [-10, 0, 10, 30])
CLASSES = ["64QAM", "256Qam", "QPSK", "8PSK"] # Add your full list here

BASE_INPUT_DIR = "Dataset"       # Root folder containing /snr_xx/class/*.npy
BASE_OUTPUT_DIR = "Scalograms_IQA"   # Output folder
SAVE_SAMPLES = True              # Save .png images for visual debugging
NUM_SAMPLES = 5                  # Number of debug images to save per class
MAX_SCALOGRAMS = None            # Set to Integer (e.g. 1000) to limit generation for testing


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt(signal, sampling_rate=1.5e6, wavelet='cmor1.5-0.5'):
    """Compute CWT for a 1D signal."""
    sampling_period = 1 / sampling_rate
    # Scales tuned for RadioML data characteristics
    scales = np.logspace(-0.5, 1.3, num=224) 
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    return np.abs(coeffs)

def normalize_stack(cwt_i, cwt_q, cwt_amp):
    """Normalize channels independently to 0-1 and stack."""
    def norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    # Stack features
    # Result shape: [3, H, W] initially
    stacked = np.stack([norm(cwt_i), norm(cwt_q), norm(cwt_amp)], axis=0)
    return stacked

def save_debug_image(cwt_i, cwt_q, cwt_amp, save_dir, filename):
    """Save a visual comparison of the 3 channels."""
    # Convert to 0-255 uint8 for image saving
    def to_img(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)

    img_i = to_img(cwt_i)
    img_q = to_img(cwt_q)
    img_amp = to_img(cwt_amp)
    
    # Merge into an RGB image for visualization
    # OpenCV uses BGR, so we stack (Amp, Q, I) to get RGB output
    merged = cv2.merge([img_amp, img_q, img_i]) 
    
    cv2.imwrite(os.path.join(save_dir, f"{filename}_preview.png"), merged)
    
    # Also save separate channels as a strip
    strip = np.hstack([img_i, img_q, img_amp])
    cv2.imwrite(os.path.join(save_dir, f"{filename}_strip.png"), strip)


# =============================================================
#   MAIN PROCESSING LOOP
# =============================================================

def generate_iqa_scalograms(snr_list):
    
    for snr in snr_list:
        print(f"\n{'='*40}")
        print(f" Processing SNR: {snr} dB")
        print(f"{'='*40}")

        for mod_class in CLASSES:
            # Setup Paths
            input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", mod_class)
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", mod_class)
            sample_dir = os.path.join("ScalogramSamples_IQA", f"snr_{snr}", mod_class)
            
            # Verify Input Directory
            if not os.path.exists(input_dir):
                print(f"[!] Input directory not found: {input_dir}")
                continue
                
            os.makedirs(output_dir, exist_ok=True)
            if SAVE_SAMPLES:
                os.makedirs(sample_dir, exist_ok=True)

            files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
            count = 0
            sample_count = 0
            
            print(f"-> Processing {mod_class}...")

            for fname in files:
                if MAX_SCALOGRAMS and count >= MAX_SCALOGRAMS:
                    break

                # 1. Load Data
                filepath = os.path.join(input_dir, fname)
                try:
                    data = np.load(filepath)
                except Exception as e:
                    print(f"   [!] Error loading {fname}: {e}")
                    continue

                # 2. Extract Channels
                # Assuming data shape is (1024, 2). If (2, 1024), use data[0, :]
                I = data[:, 0]
                Q = data[:, 1]
                
                # 3. Compute Amplitude (Magnitude)
                Amp = np.sqrt(I**2 + Q**2)

                # 4. Generate CWTs
                cwt_i = compute_cwt(I)
                cwt_q = compute_cwt(Q)
                cwt_amp = compute_cwt(Amp)

                # 5. Resize to CNN Standard (224x224)
                # cv2.resize expects (W, H), so (224, 224)
                cwt_i = cv2.resize(cwt_i, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cwt_q = cv2.resize(cwt_q, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cwt_amp = cv2.resize(cwt_amp, (224, 224), interpolation=cv2.INTER_LANCZOS4)

                # 6. Normalize and Stack
                # Returns [3, 224, 224]
                stacked = normalize_stack(cwt_i, cwt_q, cwt_amp)
                
                # Transpose for saving: [H, W, Channels] -> [224, 224, 3]
                # (Standard image format, easier for visualizers)
                final_array = np.transpose(stacked, (1, 2, 0))

                # 7. Save Data
                save_name = os.path.splitext(fname)[0]
                np.save(os.path.join(output_dir, f"{save_name}.npy"), final_array.astype(np.float32))
                
                # 8. Save Visual Samples (First N files)
                if SAVE_SAMPLES and sample_count < NUM_SAMPLES:
                    save_debug_image(cwt_i, cwt_q, cwt_amp, sample_dir, save_name)
                    sample_count += 1
                
                count += 1

            print(f"   ✓ Generated {count} scalograms.")

    print("\nAll tasks completed successfully. ✅")

if __name__ == "__main__":
    generate_iqa_scalograms(SNR_LEVELS)