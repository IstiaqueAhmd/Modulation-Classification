"""
==============================================================
RadioML Generator: Rectangular I/Q Scalograms (64x256)
==============================================================
Optimized for "Lightweight" Radio Models.
Generates Scalograms directly from I and Q components.

Output: [64 x 256 x 2] Normalized NumPy arrays.
        - Channel 0: I-Component Scalogram
        - Channel 1: Q-Component Scalogram
==============================================================
"""

import os
import numpy as np
import pywt
import cv2

# =====================
# CONFIGURATION
# =====================
SNR_LEVELS = [30]           
CLASSES = [
  "OOK", "4ASK", "8ASK",
  "BPSK", "QPSK", "8PSK",
  "16PSK", "32PSK", "16APSK",
  "32APSK", "64APSK", "128APSK",
  "16QAM", "32QAM", "64QAM",
  "128QAM", "256QAM", "AM-SSB-WC",
  "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
  "FM", "GMSK", "OQPSK"
] # Add other classes as needed

# --- OPTIMIZED DIMENSIONS ---
IMG_HEIGHT = 64    # Scales (Frequency resolution)
IMG_WIDTH = 256    # Time (Temporal resolution) - KEEP LOW FOR LIGHTWEIGHT

BASE_INPUT_DIR = "Dataset"       
BASE_OUTPUT_DIR = "Scalograms_Rectangular_IQ" # Changed folder name
SAVE_SAMPLES = True              
NUM_SAMPLES = 5                  
MAX_SCALOGRAMS = 1000            


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt_optimized(signal, sampling_rate=1e6, wavelet='cmor1.5-0.5'):
    """
    Compute CWT with exactly IMG_HEIGHT scales.
    Returns the Magnitude of the CWT coefficients.
    """
    sampling_period = 1 / sampling_rate
    
    # Generate exactly 64 scales (Logarithmically spaced)
    scales = np.logspace(0.2, 1.2, num=IMG_HEIGHT) 
    
    # coeffs is complex, we take abs() to get the magnitude scalogram
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    return np.abs(coeffs)

def resize_width(cwt_matrix, target_width):
    """
    Resize only the width (Time axis) to target_width.
    Height is preserved.
    """
    h, w = cwt_matrix.shape
    resized = cv2.resize(cwt_matrix, (target_width, h), interpolation=cv2.INTER_AREA)
    return resized

def normalize_stack(cwt_ch1, cwt_ch2):
    """Normalize channels independently to 0-1 and stack."""
    def norm(arr):
        # Prevent division by zero
        denom = arr.max() - arr.min()
        if denom == 0: denom = 1e-8
        return (arr - arr.min()) / denom
    
    # Stack features: Result shape [2, H, W]
    return np.stack([norm(cwt_ch1), norm(cwt_ch2)], axis=0)

def save_debug_image(cwt_i, cwt_q, save_dir, filename):
    """Save visual comparison of I vs Q scalograms."""
    def to_img(arr):
        denom = arr.max() - arr.min()
        if denom == 0: denom = 1e-8
        arr = (arr - arr.min()) / denom
        return (arr * 255).astype(np.uint8)

    img_i = to_img(cwt_i)
    img_q = to_img(cwt_q)
    
    # Stack vertically: Top is I, Bottom is Q
    strip = np.vstack([img_i, img_q])
    cv2.imwrite(os.path.join(save_dir, f"{filename}_IQ.png"), strip)


# =============================================================
#   MAIN PROCESSING LOOP
# =============================================================

def generate_rectangular_scalograms(snr_list):
    
    for snr in snr_list:
        print(f"\n{'='*40}")
        print(f" Processing SNR: {snr} dB | Target Size: {IMG_HEIGHT}x{IMG_WIDTH}")
        print(f" Mode: I/Q Scalograms")
        print(f"{'='*40}")

        for mod_class in CLASSES:
            # Setup Paths
            input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", mod_class)
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", mod_class)
            sample_dir = os.path.join("ScalogramSamples_Rect_IQ", f"snr_{snr}", mod_class)
            
            if not os.path.exists(input_dir):
                print(f"Skipping {mod_class} (Source not found)")
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
                    continue

                # 2. Extract I and Q directly
                # Assuming shape is [Length, 2]
                I_sig = data[:, 0]
                Q_sig = data[:, 1]
                
                # 3. Compute CWTs directly on I and Q
                cwt_i = compute_cwt_optimized(I_sig)
                cwt_q = compute_cwt_optimized(Q_sig)

                # 4. Resize Width Only (to 256 or whatever IMG_WIDTH is)
                cwt_i = resize_width(cwt_i, IMG_WIDTH)
                cwt_q = resize_width(cwt_q, IMG_WIDTH)

                # 5. Normalize and Stack
                # Returns [2, 64, 256] -> Channel 0 is I, Channel 1 is Q
                stacked = normalize_stack(cwt_i, cwt_q)
                
                # Transpose for saving: [64, 256, 2]
                final_array = np.transpose(stacked, (1, 2, 0))

                # 6. Save Data
                save_name = os.path.splitext(fname)[0]
                np.save(os.path.join(output_dir, f"{save_name}.npy"), final_array.astype(np.float32))
                
                # 7. Save Visual Samples
                if SAVE_SAMPLES and sample_count < NUM_SAMPLES:
                    save_debug_image(cwt_i, cwt_q, sample_dir, save_name)
                    sample_count += 1
                
                count += 1

            print(f"   Generated {count} I/Q scalograms.")

    print("\nAll tasks completed successfully. ")

if __name__ == "__main__":
    generate_rectangular_scalograms(SNR_LEVELS)