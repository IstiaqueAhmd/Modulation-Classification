"""
==============================================================
RadioML Generator: Dual Channel (Amplitude & Phase)
==============================================================

Generates 2-channel scalograms.
Channel 1: Amplitude
Channel 2: Phase (Unwrapped)

Output: [224 x 224 x 2] Normalized NumPy arrays.
==============================================================
"""

import os
import numpy as np
import pywt
import cv2

# =====================
# CONFIGURATION
# =====================
SNR_LEVELS = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]          # List of SNRs to process
CLASSES = [
  "OOK", "4ASK","BPSK", "QPSK", "8PSK","8ASK", "16APSK", "64QAM",
  "AM-SSB-WC", "AM-DSB-WC","FM","GMSK", "OQPSK"
]

BASE_INPUT_DIR = "Dataset"       # Root folder containing /snr_xx/class/*.npy
BASE_OUTPUT_DIR = "Dataset/Scalograms"   # Output folder
SAVE_SAMPLES = True              # Save .png images for visual debugging
NUM_SAMPLES = 5                  # Number of debug images to save per class
MAX_SCALOGRAMS = 200            # Set to Integer to limit generation


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt(signal, sampling_rate=1e6, wavelet='cmor1.5-0.5'):
    """Compute CWT for a 1D signal."""
    sampling_period = 1 / sampling_rate
    scales = np.logspace(0.2, 1.5, num=224) 
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    return np.abs(coeffs)

def stack_channels(cwt_ch1, cwt_ch2):
    """Stack channels without normalization. Normalization will be 
    handled by the training script with proper global statistics."""
    # Stack features: Result shape [2, H, W]
    return np.stack([cwt_ch1, cwt_ch2], axis=0)

def save_debug_image(cwt_amp, cwt_phase, save_dir, filename):
    """Save visual comparison of the 2 channels (Grayscale side-by-side)."""
    def to_img(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)

    img_a = to_img(cwt_amp)
    img_p = to_img(cwt_phase)
    
    # Stack side-by-side for visualization
    # Left: Amplitude, Right: Phase
    strip = np.hstack([img_a, img_p])
    cv2.imwrite(os.path.join(save_dir, f"{filename}_AmpPhase.png"), strip)


# =============================================================
#   MAIN PROCESSING LOOP
# =============================================================

def generate_ampphase_scalograms(snr_list):
    
    for snr in snr_list:
        print(f"\n{'='*40}")
        print(f" Processing SNR: {snr} dB")
        print(f"{'='*40}")

        for mod_class in CLASSES:
            # Setup Paths
            input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", mod_class)
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", mod_class)
            sample_dir = os.path.join("Dataset/Samples", f"snr_{snr}", mod_class)
            
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

                # 2. Compute Amplitude and Phase from raw I/Q data
                # Column 0: I (In-phase), Column 1: Q (Quadrature)
                i_sig = data[:, 0]
                q_sig = data[:, 1]
                
                # Compute amplitude: sqrt(I^2 + Q^2)
                amp_sig = np.sqrt(i_sig**2 + q_sig**2)
                # Compute phase: arctan2(Q, I) and unwrap to avoid discontinuities
                phase_sig = np.unwrap(np.arctan2(q_sig, i_sig))

                # 4. Compute CWTs
                cwt_amp = compute_cwt(amp_sig)
                cwt_phase = compute_cwt(phase_sig)

                # 5. Resize to CNN Standard (224x224)
                cwt_amp = cv2.resize(cwt_amp, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cwt_phase = cv2.resize(cwt_phase, (224, 224), interpolation=cv2.INTER_LANCZOS4)

                # 6. Stack Channels (normalization handled by training script)
                # Returns [2, 224, 224] -> Channel 0 is Amp, Channel 1 is Phase
                stacked = stack_channels(cwt_amp, cwt_phase)
                
                # Transpose for saving: [224, 224, 2]
                final_array = np.transpose(stacked, (1, 2, 0))

                # 7. Save Data
                save_name = os.path.splitext(fname)[0]
                np.save(os.path.join(output_dir, f"{save_name}.npy"), final_array.astype(np.float32))
                
                # 8. Save Visual Samples
                if SAVE_SAMPLES and sample_count < NUM_SAMPLES:
                    save_debug_image(cwt_amp, cwt_phase, sample_dir, save_name)
                    sample_count += 1
                
                count += 1

            print(f"   ✓ Generated {count} scalograms.")

    print("\nAll tasks completed successfully. ✅")

if __name__ == "__main__":
    generate_ampphase_scalograms(SNR_LEVELS)