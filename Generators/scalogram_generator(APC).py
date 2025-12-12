"""
==============================================================
RadioML Generator: Tri-Channel (Amp, Phase, Constellation)
==============================================================

Generates 3-channel inputs:
Channel 1: Amplitude Scalogram (Time-Frequency)
Channel 2: Phase Scalogram (Time-Frequency)
Channel 3: Constellation Density Map (Stat-Frequency)

Output: [224 x 224 x 3] Normalized NumPy arrays.
==============================================================
"""

import os
import numpy as np
import pywt
import cv2

# =====================
# CONFIGURATION
# =====================
SNR_LEVELS = [30]           # List of SNRs to process
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

BASE_INPUT_DIR = "Dataset"       # Root folder containing /snr_xx/class/*.npy
BASE_OUTPUT_DIR = "Scalograms_TriChannel" # Output folder
SAVE_SAMPLES = True              # Save .png images for visual debugging
NUM_SAMPLES = 5                  # Number of debug images to save per class
MAX_SCALOGRAMS = 2000            # Set to Integer to limit generation (e.g., 1000) or None for all


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt(signal, sampling_rate=1e6, wavelet='cmor1.5-0.5'):
    """Compute CWT for a 1D signal."""
    sampling_period = 1 / sampling_rate
    # Scales tailored for 224 height
    scales = np.logspace(0.2, 1.5, num=224) 
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    return np.abs(coeffs)

def compute_constellation_map(I, Q, size=224):
    """
    Generates a Constellation Density Map with ABSOLUTE SCALING.
    """
    # 1. Power Normalization (Unit Average Power)
    power = np.sqrt(np.mean(I**2 + Q**2))
    if power > 0:
        I = I / power
        Q = Q / power
    
    # 2. Fixed Range Histogram [-2, 2]
    # We use a larger bin count (e.g. 64 or 128) then resize, or just map directly.
    # Direct mapping to 224x224 bins might be too sparse for 1024 samples.
    # Let's use 224 bins directly.
    heatmap, _, _ = np.histogram2d(I, Q, bins=size, range=[[-2.5, 2.5], [-2.5, 2.5]])
    
    # 3. ABSOLUTE SCALING (The Fix)
    # 16QAM puts ~64 points in one bin (if N=1024). 
    # 256QAM puts ~4 points in one bin.
    # We define a "Saturation Limit". Any bin with > 20 hits is White (1.0).
    SATURATION_LIMIT = 20.0 
    
    heatmap = np.clip(heatmap, 0, SATURATION_LIMIT)
    heatmap = heatmap / SATURATION_LIMIT  # Scale to 0.0 - 1.0 range based on fixed limit
    
    # 4. Rotate
    return np.rot90(heatmap)

def normalize_stack_3ch(cwt_ch1, cwt_ch2, const_ch3):
    """
    Normalize channels 1 & 2 (CWT) but KEEP Channel 3's absolute scaling.
    """
    def norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    # Channel 1 (Amp) -> Normalize (Relative)
    # Channel 2 (Phase) -> Normalize (Relative)
    # Channel 3 (Const) -> USE AS IS (Absolute scaling preserved)
    return np.stack([norm(cwt_ch1), norm(cwt_ch2), const_ch3], axis=0)

def save_debug_image(cwt_amp, cwt_phase, const_map, save_dir, filename):
    """Save visual comparison of the 3 channels."""
    def to_img(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)

    img_a = to_img(cwt_amp)
    img_p = to_img(cwt_phase)
    img_c = to_img(const_map)
    
    # Stack side-by-side: Amp | Phase | Constellation
    strip = np.hstack([img_a, img_p, img_c])
    cv2.imwrite(os.path.join(save_dir, f"{filename}_TriView.png"), strip)


# =============================================================
#   MAIN PROCESSING LOOP
# =============================================================

def generate_tri_scalograms(snr_list):
    
    for snr in snr_list:
        print(f"\n{'='*40}")
        print(f" Processing SNR: {snr} dB")
        print(f"{'='*40}")

        for mod_class in CLASSES:
            # Setup Paths
            input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", mod_class)
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", mod_class)
            sample_dir = os.path.join("ScalogramSamples_TriChannel", f"snr_{snr}", mod_class)
            
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

                # 2. Extract I and Q
                I = data[:, 0]
                Q = data[:, 1]
                
                # 3. Convert to Amplitude and Phase
                amp_sig = np.sqrt(I**2 + Q**2)
                phase_sig = np.unwrap(np.arctan2(Q, I))

                # 4. Compute Features (3 Channels)
                cwt_amp = compute_cwt(amp_sig)
                cwt_phase = compute_cwt(phase_sig)
                const_map = compute_constellation_map(I, Q, size=224) # Channel 3

                # 5. Resize CWTs to 224x224 (Constellation map is already 224)
                cwt_amp = cv2.resize(cwt_amp, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cwt_phase = cv2.resize(cwt_phase, (224, 224), interpolation=cv2.INTER_LANCZOS4)

                # 6. Normalize and Stack
                # Returns [3, 224, 224]
                stacked = normalize_stack_3ch(cwt_amp, cwt_phase, const_map)
                
                # Transpose for saving: [224, 224, 3]
                final_array = np.transpose(stacked, (1, 2, 0))

                # 7. Save Data
                save_name = os.path.splitext(fname)[0]
                np.save(os.path.join(output_dir, f"{save_name}.npy"), final_array.astype(np.float32))
                
                # 8. Save Visual Samples
                if SAVE_SAMPLES and sample_count < NUM_SAMPLES:
                    save_debug_image(cwt_amp, cwt_phase, const_map, sample_dir, save_name)
                    sample_count += 1
                
                count += 1

            print(f"   ✓ Generated {count} tri-channel samples.")

    print("\nAll tasks completed successfully. ✅")

if __name__ == "__main__":
    generate_tri_scalograms(SNR_LEVELS)