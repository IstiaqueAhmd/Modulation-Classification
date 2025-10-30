"""
==============================================================
RadioML 2018.01A → Continuous Wavelet Scalogram Generator
==============================================================

Generates I/Q channel scalograms from I/Q data using
physically consistent wavelet parameters (1.5 MHz sampling).
Each output file: [224×224×2] NumPy array (CWT_I, CWT_Q).
--------------------------------------------------------------
Author : Istiaque (2025)
Updated : Oct 2025 (Modified for I/Q CWT)
==============================================================
"""

import os
import numpy as np
import pywt
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =====================
# CONFIGURATION
# =====================
SNR_LEVELS = [30]      # SNRs to process
CLASSES = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']


BASE_INPUT_DIR = "Dataset"      # Root where /snr_xx/class/*.npy are stored
BASE_OUTPUT_DIR = "Scalograms"  # Where scalograms will be saved
SAVE_SAMPLES = True           # Save example images for sanity check
NUM_SAMPLES = 5               # # of sample images per modulation per SNR
MAX_SCALOGRAMS = 5            # Limit per class (None = all)


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt(signal, sampling_rate=1.5e6, wavelet='cmor1.5-0.5'):
    """Compute CWT for a 1D signal using correct physical sampling."""
    sampling_period = 1 / sampling_rate
    scales = np.logspace(-1, 1.3, num=200)  # tuned for radio bands
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    coeffs = np.abs(coeffs)
    return coeffs


def normalize_stack(cwt_I, cwt_Q):
    """Normalize I and Q CWTs jointly (0–1 range)."""
    stacked = np.stack([cwt_I, cwt_Q], axis=0)
    stacked = (stacked - stacked.min()) / (stacked.max() - stacked.min() + 1e-8)
    return stacked


def save_sample_images(cwt_I, cwt_Q, sample_dir, base_filename, snr):
    """Save grayscale I-channel & Q-channel sample images."""
    i_path = os.path.join(sample_dir, f"{base_filename}_I_snr{snr}.png")
    q_path = os.path.join(sample_dir, f"{base_filename}_Q_snr{snr}.png")

    # Scale to 0–255 for saving as grayscale PNG
    i_img = (255 * (cwt_I - cwt_I.min()) / (cwt_I.max() - cwt_I.min() + 1e-8)).astype(np.uint8)
    q_img = (255 * (cwt_Q - cwt_Q.min()) / (cwt_Q.max() - cwt_Q.min() + 1e-8)).astype(np.uint8)

    cv2.imwrite(i_path, i_img)
    cv2.imwrite(q_path, q_img)

# =============================================================
#   MAIN GENERATION FUNCTION
# =============================================================

def generate_wavelet_scalograms(data_type, snr,
                                max_scalograms=None,
                                save_samples=False,
                                num_samples=5):
    input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", data_type)
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", data_type)
    sample_dir = os.path.join("ScalogramSamples", f"snr_{snr}", data_type)

    os.makedirs(output_dir, exist_ok=True)
    if save_samples:
        os.makedirs(sample_dir, exist_ok=True)

    scalogram_count = 0
    sample_count = 0

    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    for filename in files:
        if max_scalograms is not None and scalogram_count >= max_scalograms:
            break

        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)

        # Separate I and Q components
        I, Q = data[:, 0], data[:, 1]
        
        # --- MODIFICATION: Compute CWT on I and Q directly ---
        
        # Compute CWTs
        cwt_I = compute_cwt(I)
        cwt_Q = compute_cwt(Q)

        # Resize for CNN input
        cwt_I = cv2.resize(cwt_I, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cwt_Q = cv2.resize(cwt_Q, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        # Normalize jointly
        stacked = normalize_stack(cwt_I, cwt_Q)
        
        # --- END MODIFICATION ---

        stacked = np.transpose(stacked, (1, 2, 0))  # H×W×C

        # Save scalogram
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}_snr{snr}.npy")
        np.save(output_path, stacked.astype(np.float32))
        scalogram_count += 1

        # Save sample visualization
        if save_samples and sample_count < num_samples:
            # Pass cwt_I and cwt_Q to the updated save function
            save_sample_images(cwt_I, cwt_Q, sample_dir, base_filename, snr)
            sample_count += 1

    print(f"[✓] {data_type} | SNR {snr} → {scalogram_count} scalograms generated.")


# =============================================================
#   MAIN SCRIPT EXECUTION
# =============================================================

if __name__ == "__main__":
    for snr in SNR_LEVELS:
        print(f"\n{'='*60}")
        print(f" Processing SNR Level: {snr} dB ")
        print(f"{'='*60}\n")
        for modulation in CLASSES:
            generate_wavelet_scalograms(
                modulation,
                snr,
                max_scalograms=MAX_SCALOGRAMS,
                save_samples=SAVE_SAMPLES,
                num_samples=NUM_SAMPLES
            )

    print("\nAll scalograms generated successfully ✅")