"""
==============================================================
RadioML 2018.01A → Continuous Wavelet Scalogram Generator
==============================================================

Generates amplitude–phase scalograms from I/Q data using
physically consistent wavelet parameters (1.5 MHz sampling).
Each output file: [224×224×2] NumPy array (amplitude, phase).
--------------------------------------------------------------
Author : Istiaque (2025)
Updated : Oct 2025
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
SNR_LEVELS = [30,20,10,0]   # SNRs to process
CLASSES = [
    "OOK", "4ASK", "8ASK",
    "BPSK", "QPSK", "8PSK",
    "16APSK", "64QAM",
    "AM-SSB-WC", "AM-DSB-WC",
    "FM", "GMSK", "OQPSK"
]

BASE_INPUT_DIR = "Dataset"       # Root where /snr_xx/class/*.npy are stored
BASE_OUTPUT_DIR = "Scalograms"   # Where scalograms will be saved
SAVE_SAMPLES = True              # Save example images for sanity check
NUM_SAMPLES = 5                  # # of sample images per modulation per SNR
MAX_SCALOGRAMS = 1000            # Limit per class (None = all)


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


def normalize_stack(cwt_amp, cwt_phase):
    cwt_amp = (cwt_amp - cwt_amp.min()) / (cwt_amp.max() - cwt_amp.min() + 1e-8)
    cwt_phase = (cwt_phase - cwt_phase.min()) / (cwt_phase.max() - cwt_phase.min() + 1e-8)
    return np.stack([cwt_amp, cwt_phase], axis=0)


def save_sample_images(cwt_amp, cwt_phase, sample_dir, base_filename, snr):
    """Save grayscale amplitude & phase sample images."""
    amp_path = os.path.join(sample_dir, f"{base_filename}_amp_snr{snr}.png")
    phase_path = os.path.join(sample_dir, f"{base_filename}_phase_snr{snr}.png")

    # Scale to 0–255 for saving as grayscale PNG
    amp_img = (255 * (cwt_amp - cwt_amp.min()) / (cwt_amp.max() - cwt_amp.min() + 1e-8)).astype(np.uint8)
    phase_img = (255 * (cwt_phase - cwt_phase.min()) / (cwt_phase.max() - cwt_phase.min() + 1e-8)).astype(np.uint8)

    cv2.imwrite(amp_path, amp_img)
    cv2.imwrite(phase_path, phase_img)

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
        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        phase = np.unwrap(phase)

        # Compute CWTs
        cwt_amp = compute_cwt(amplitude)
        cwt_phase = compute_cwt(phase)

        # Resize for CNN input
        cwt_amp = cv2.resize(cwt_amp, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cwt_phase = cv2.resize(cwt_phase, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        # Normalize jointly
        stacked = normalize_stack(cwt_amp, cwt_phase)
        stacked = np.transpose(stacked, (1, 2, 0))  # H×W×C

        # Save scalogram
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}_snr{snr}.npy")
        np.save(output_path, stacked.astype(np.float32))
        scalogram_count += 1

        # Save sample visualization
        if save_samples and sample_count < num_samples:
            save_sample_images(cwt_amp, cwt_phase, sample_dir, base_filename, snr)
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
