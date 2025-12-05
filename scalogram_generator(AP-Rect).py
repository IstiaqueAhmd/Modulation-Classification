"""
==============================================================
RadioML Generator: Rectangular (64x256)
==============================================================
Optimized for "Lightweight" Radio Models.

Output: [64 x 256 x 2] Normalized NumPy arrays.
        - Height (64): Frequency/Scale
        - Width (256): Time
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
]

# --- OPTIMIZED DIMENSIONS ---
IMG_HEIGHT = 64   # Scales (Frequency resolution)
IMG_WIDTH = 256   # Time (Temporal resolution)

BASE_INPUT_DIR = "Dataset"       
BASE_OUTPUT_DIR = "Scalograms_Rectangular" # New output folder
SAVE_SAMPLES = True              
NUM_SAMPLES = 5                  
MAX_SCALOGRAMS = 1000            


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt_optimized(signal, sampling_rate=1e6, wavelet='cmor1.5-0.5'):
    """
    Compute CWT with exactly IMG_HEIGHT scales.
    No vertical resizing needed later!
    """
    sampling_period = 1 / sampling_rate
    
    # Generate exactly 64 scales (Logarithmically spaced to match Frequency behavior)
    scales = np.logspace(0.2, 1.2, num=IMG_HEIGHT) 
    
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    return np.abs(coeffs)

def resize_width(cwt_matrix, target_width):
    """
    Resize only the width (Time axis) to target_width.
    Height is preserved (already perfect from CWT).
    """
    # cv2.resize expects (Width, Height)
    # cwt_matrix shape is (Height, Width_Original)
    h, w = cwt_matrix.shape
    resized = cv2.resize(cwt_matrix, (target_width, h), interpolation=cv2.INTER_AREA)
    return resized

def normalize_stack(cwt_ch1, cwt_ch2):
    """Normalize channels independently to 0-1 and stack."""
    def norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    # Stack features: Result shape [2, H, W]
    return np.stack([norm(cwt_ch1), norm(cwt_ch2)], axis=0)

def save_debug_image(cwt_amp, cwt_phase, save_dir, filename):
    """Save visual comparison (Wide strip)."""
    def to_img(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)

    img_a = to_img(cwt_amp)
    img_p = to_img(cwt_phase)
    
    # Stack vertically because they are wide and short
    strip = np.vstack([img_a, img_p])
    cv2.imwrite(os.path.join(save_dir, f"{filename}_Rect.png"), strip)


# =============================================================
#   MAIN PROCESSING LOOP
# =============================================================

def generate_rectangular_scalograms(snr_list):
    
    for snr in snr_list:
        print(f"\n{'='*40}")
        print(f" Processing SNR: {snr} dB | Target Size: {IMG_HEIGHT}x{IMG_WIDTH}")
        print(f"{'='*40}")

        for mod_class in CLASSES:
            # Setup Paths
            input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", mod_class)
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", mod_class)
            sample_dir = os.path.join("ScalogramSamples_Rect", f"snr_{snr}", mod_class)
            
            if not os.path.exists(input_dir):
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

                # 2. Extract I and Q
                I = data[:, 0]
                Q = data[:, 1]
                
                # 3. Convert to Amplitude and Phase
                amp_sig = np.sqrt(I**2 + Q**2)
                phase_sig = np.unwrap(np.arctan2(Q, I))

                # 4. Compute CWTs (Optimized: Direct 64 scales)
                cwt_amp = compute_cwt_optimized(amp_sig)
                cwt_phase = compute_cwt_optimized(phase_sig)

                # 5. Resize Width Only (to 256)
                cwt_amp = resize_width(cwt_amp, IMG_WIDTH)
                cwt_phase = resize_width(cwt_phase, IMG_WIDTH)

                # 6. Normalize and Stack
                # Returns [2, 64, 256]
                stacked = normalize_stack(cwt_amp, cwt_phase)
                
                # Transpose for saving: [64, 256, 2]
                final_array = np.transpose(stacked, (1, 2, 0))

                # 7. Save Data
                save_name = os.path.splitext(fname)[0]
                np.save(os.path.join(output_dir, f"{save_name}.npy"), final_array.astype(np.float32))
                
                # 8. Save Visual Samples
                if SAVE_SAMPLES and sample_count < NUM_SAMPLES:
                    save_debug_image(cwt_amp, cwt_phase, sample_dir, save_name)
                    sample_count += 1
                
                count += 1

            print(f"   Generated {count} rect-scalograms.")

    print("\nAll tasks completed successfully. ")

if __name__ == "__main__":
    generate_rectangular_scalograms(SNR_LEVELS)