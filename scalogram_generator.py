"""
==============================================================
RadioML 2018.01A -> Complex CWT Scalogram Generator
==============================================================

Generates amplitude–phase scalograms from I/Q data using
physically consistent wavelet parameters (1.5 MHz sampling).

This version computes the CWT on the COMPLEX I+jQ signal
and extracts amplitude/phase from the resulting complex
coefficients to avoid 'arctan2' phase-wrapping artifacts.

Each output file: [224×224×2] NumPy array (amplitude, phase).
--------------------------------------------------------------
Author : Istiaque (2025)
Updated : Oct 2025 (Gemini modification for complex CWT)
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
SNR_LEVELS = [30]  # SNRs to process
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
SAVE_SAMPLES = True             # Save example images for sanity check
NUM_SAMPLES = 5                 # # of sample images per modulation per SNR
MAX_SCALOGRAMS = None           # Limit per class (None = all)
TARGET_SIZE = (224, 224)        # Target H, W for CNN input


# =============================================================
#   HELPER FUNCTIONS (MODIFIED FOR COMPLEX CWT)
# =============================================================

def compute_complex_cwt(signal, sampling_rate=1.5e6, wavelet='cmor1.5-0.5'):
    """
    Compute CWT for a 1D *complex* signal and return complex coefficients.
    """
    sampling_period = 1 / sampling_rate
    # These scales were tuned in the original script
    scales = np.logspace(-1, 1.3, num=200) 
    
    # pywt.cwt handles complex input (I+jQ) automatically
    # This returns [num_scales, signal_length] complex coefficients
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    
    return coeffs


def normalize_scalograms(cwt_amp, cwt_phase):
    """
    Normalize amplitude (MinMax 0-1) and phase ([-pi, pi] -> 0-1) 
    independently.
    """
    # Normalize amplitude using Min-Max
    amp_norm = (cwt_amp - cwt_amp.min()) / (cwt_amp.max() - cwt_amp.min() + 1e-8)
    
    # Normalize phase from [-pi, pi] to [0, 1]
    phase_norm = (cwt_phase + np.pi) / (2 * np.pi)
    
    # Stack: (C, H, W)
    stacked = np.stack([amp_norm, phase_norm], axis=0)
    return stacked


def save_sample_images(cwt_amp, cwt_phase, sample_dir, base_filename, snr):
    """Save grayscale amplitude & phase sample images."""
    amp_path = os.path.join(sample_dir, f"{base_filename}_amp_snr{snr}.png")
    phase_path = os.path.join(sample_dir, f"{base_filename}_phase_snr{snr}.png")

    # Scale raw amplitude to 0–255 for saving as grayscale PNG
    amp_img = (255 * (cwt_amp - cwt_amp.min()) / (cwt_amp.max() - cwt_amp.min() + 1e-8)).astype(np.uint8)
    
    # Scale raw phase (now [-pi, pi]) to 0-255
    phase_img = (255 * (cwt_phase + np.pi) / (2 * np.pi + 1e-8)).astype(np.uint8)

    cv2.imwrite(amp_path, amp_img)
    cv2.imwrite(phase_path, phase_img)

# =============================================================
#   MAIN GENERATION FUNCTION (MODIFIED FOR COMPLEX CWT)
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
        
        # --- KEY CHANGE: Create complex signal ---
        complex_signal = I + 1j * Q

        # --- KEY CHANGE: Compute CWT on complex signal ---
        # This returns complex-valued coefficients
        complex_coeffs = compute_complex_cwt(complex_signal)

        # --- KEY CHANGE: Extract amp/phase from complex coefficients ---
        cwt_amp = np.abs(complex_coeffs)
        cwt_phase = np.angle(complex_coeffs) # Range [-pi, pi]

        # Resize for CNN input
        # Note: CV2 expects (H, W), so we resize the [scales, time] matrix
        cwt_amp = cv2.resize(cwt_amp, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        cwt_phase = cv2.resize(cwt_phase, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # Save sample visualization (uses raw resized amp/phase)
        if save_samples and sample_count < num_samples:
            base_filename = os.path.splitext(filename)[0]
            save_sample_images(cwt_amp, cwt_phase, sample_dir, base_filename, snr)
            sample_count += 1

        # Normalize independently
        # Returns shape (C, H, W) = (2, 224, 224)
        stacked = normalize_scalograms(cwt_amp, cwt_phase)
        
        # Transpose to H×W×C (224, 224, 2) for consistency with original script
        stacked = np.transpose(stacked, (1, 2, 0)) 

        # Save scalogram
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}_snr{snr}.npy")
        np.save(output_path, stacked.astype(np.float32))
        scalogram_count += 1

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

    print("\nAll complex CWT scalograms generated successfully ✅")