"""
==============================================================
RadioML Generator: Triple-Stream (Amp, Const, SCD)
==============================================================

Generates 3-channel inputs:
Channel 1: Amplitude Scalogram (Time-Frequency)
Channel 2: Constellation Density Map (Stat-Frequency)
Channel 3: Spectral Correlation Density (Cyclostationary)

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

BASE_INPUT_DIR = "Dataset"       
BASE_OUTPUT_DIR = "Dataset/Scalograms" # Output folder
SAVE_SAMPLES = True              
NUM_SAMPLES = 5                  
MAX_SCALOGRAMS = 2000            


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt(signal, sampling_rate=1e6, wavelet='cmor1.5-0.5'):
    """Channel 1: Compute CWT (Amplitude Scalogram)."""
    sampling_period = 1 / sampling_rate
    scales = np.logspace(0.2, 1.5, num=224) 
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    return np.abs(coeffs)

def compute_constellation_map(I, Q, size=224):
    """Channel 2: Generates a Constellation Density Map."""
    # Power Normalization
    power = np.sqrt(np.mean(I**2 + Q**2))
    if power > 0:
        I = I / power
        Q = Q / power
    
    # 2D Histogram
    heatmap, _, _ = np.histogram2d(I, Q, bins=size, range=[[-2.5, 2.5], [-2.5, 2.5]])
    
    # Absolute Scaling / Saturation
    SATURATION_LIMIT = 20.0 
    heatmap = np.clip(heatmap, 0, SATURATION_LIMIT)
    heatmap = heatmap / SATURATION_LIMIT  
    
    return np.rot90(heatmap)

def compute_scd_bifreq(I, Q, size=224):
    """
    Channel 3: Computes the Bi-Frequency Spectral Correlation Map.
    This approximates the SCD by calculating the correlation matrix 
    of the FFT: S(f1, f2) = X(f1) * conj(X(f2)).
    """
    # 1. Form Complex Signal
    complex_sig = I + 1j * Q
    
    # 2. Compute FFT and Shift (Center DC)
    # We use a Hanning window to reduce spectral leakage
    window = np.hanning(len(complex_sig))
    X_f = np.fft.fftshift(np.fft.fft(complex_sig * window))
    
    # 3. Outer Product (Bi-Frequency Plane)
    # Result is an NxN matrix where diagonals represent cyclic freqs
    scd_matrix = np.abs(np.outer(X_f, np.conj(X_f)))
    
    # 4. Log Scale (Critical for SCD visualization)
    # SCD features often have high dynamic range
    scd_matrix = np.log10(scd_matrix + 1e-6)
    
    # 5. Resize to target dimension
    # Using INTER_AREA is better for shrinking dense spectral maps
    scd_resized = cv2.resize(scd_matrix, (size, size), interpolation=cv2.INTER_AREA)
    
    return scd_resized

def normalize_stack_3ch(ch1, ch2, ch3):
    """
    Normalizes and stacks the 3 independent streams.
    """
    def norm_minmax(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    # Ch1 (Amp): Relative Norm
    # Ch2 (Const): ALREADY Normalized (Absolute)
    # Ch3 (SCD): Relative Norm (Log scale is relative)
    
    return np.stack([norm_minmax(ch1), ch2, norm_minmax(ch3)], axis=0)

def save_debug_image(ch1, ch2, ch3, save_dir, filename):
    """Save visual comparison: Amp | Constellation | SCD"""
    def to_img(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)

    img_1 = to_img(ch1) # Amp
    img_2 = to_img(ch2) # Const
    img_3 = to_img(ch3) # SCD
    
    # Apply colormap to SCD for better visualization (Optional)
    img_3_color = cv2.applyColorMap(img_3, cv2.COLORMAP_JET)
    img_1_color = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
    img_2_color = cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)
    
    # Stack side-by-side
    strip = np.hstack([img_1_color, img_2_color, img_3_color])
    cv2.imwrite(os.path.join(save_dir, f"{filename}_TriStream.png"), strip)


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
            sample_dir = os.path.join("Dataset/ScalogramSamples", f"snr_{snr}", mod_class)
            
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
                
                # 3. Compute Features
                # --- Stream 1: Amplitude Scalogram ---
                amp_sig = np.sqrt(I**2 + Q**2)
                ch1_amp = compute_cwt(amp_sig)
                ch1_amp = cv2.resize(ch1_amp, (224, 224), interpolation=cv2.INTER_LANCZOS4)

                # --- Stream 2: Constellation Map ---
                ch2_const = compute_constellation_map(I, Q, size=224) 

                # --- Stream 3: Spectral Correlation (SCD) ---
                ch3_scd = compute_scd_bifreq(I, Q, size=224)

                # 4. Normalize and Stack
                # Returns [3, 224, 224]
                stacked = normalize_stack_3ch(ch1_amp, ch2_const, ch3_scd)
                
                # Transpose for saving: [224, 224, 3]
                final_array = np.transpose(stacked, (1, 2, 0))

                # 5. Save Data
                save_name = os.path.splitext(fname)[0]
                np.save(os.path.join(output_dir, f"{save_name}.npy"), final_array.astype(np.float32))
                
                # 6. Save Visual Samples
                if SAVE_SAMPLES and sample_count < NUM_SAMPLES:
                    save_debug_image(ch1_amp, ch2_const, ch3_scd, sample_dir, save_name)
                    sample_count += 1
                
                count += 1

            print(f"   ✓ Generated {count} triple-stream samples.")

    print("\nAll tasks completed successfully. ✅")

if __name__ == "__main__":
    generate_tri_scalograms(SNR_LEVELS)