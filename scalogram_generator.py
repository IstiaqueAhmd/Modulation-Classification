"""
==============================================================
RadioML 2018.01A → Continuous Wavelet Scalogram Generator
==============================================================

Generates 1-channel amplitude scalograms from I/Q data using
physically consistent wavelet parameters (1.5 MHz sampling).
Each output file: [224×224×1] NumPy array (amplitude).
--------------------------------------------------------------
Author : Istiaque (2025)
Updated : Oct 2025
==============================================================
"""

import os
import numpy as np
import pywt
import cv2

# =====================
# CONFIGURATION
# =====================
SNR_LEVELS = [30]     # SNRs to process
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
MAX_SCALOGRAMS = 4000            # Limit per class (None = all)


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt(signal, sampling_rate=1.5e6, wavelet='cmor1.5-0.5'):
    """
    Compute CWT for a 1D (real or complex) signal.
    Returns the complex coefficients.
    """
    sampling_period = 1 / sampling_rate
    scales = np.logspace(-1, 1.3, num=200)  # tuned for radio bands
    
    # Return the raw complex coefficients
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    
    return coeffs


def save_sample_image(cwt_amp_resized, sample_dir, base_filename, snr):
    """Save a single grayscale amplitude sample image."""
    amp_path = os.path.join(sample_dir, f"{base_filename}_amp_snr{snr}.png")

    # Scale the 224x224 input to 0–255 for saving as grayscale PNG
    amp_min = cwt_amp_resized.min()
    amp_max = cwt_amp_resized.max()
    amp_img = (255 * (cwt_amp_resized - amp_min) / (amp_max - amp_min + 1e-8)).astype(np.uint8)

    cv2.imwrite(amp_path, amp_img)

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
        
        # 1. Create the complex signal
        complex_signal = I + 1j * Q

        # 2. Compute CWT on the complex signal (returns complex coeffs)
        complex_coeffs = compute_cwt(complex_signal)

        # 3. Extract amplitude scalogram (discarding phase)
        cwt_amp = np.abs(complex_coeffs)

        # 4. Resize for CNN input
        cwt_amp_resized = cv2.resize(cwt_amp, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        # 5. Normalize amplitude (Min-Max to 0-1)
        amp_min = cwt_amp_resized.min()
        amp_max = cwt_amp_resized.max()
        cwt_amp_norm = (cwt_amp_resized - amp_min) / (amp_max - amp_min + 1e-8)

        # 6. Reshape to HxWx1
        final_scalogram = np.expand_dims(cwt_amp_norm, axis=-1)

        # Save scalogram
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}_snr{snr}.npy")
        np.save(output_path, final_scalogram.astype(np.float32))
        scalogram_count += 1

        # Save sample visualization
        if save_samples and sample_count < num_samples:
            # Pass the resized (224x224) amplitude data
            save_sample_image(cwt_amp_resized, sample_dir, base_filename, snr)
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