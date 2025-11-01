"""
=====================================================================
RadioML 2018.01A → Enhanced Complex Wavelet Scalogram Generator (v2)
=====================================================================

Generates amplitude–phase (and optional frequency) scalograms
from I/Q signals using complex Morlet wavelets tuned for
fine-grained digital modulation discrimination.

Each output file: [224×224×2 or ×3] NumPy array
(amplitude, phase [, inst_freq]).
---------------------------------------------------------------------
Author  : Istiaque (2025)
Updated : Nov 2025
=====================================================================
"""

import os
import numpy as np
import pywt
import cv2

# =====================
# CONFIGURATION
# =====================
SNR_LEVELS = [30]  # full range (adjust as needed)
BASE_INPUT_DIR = "Dataset"
BASE_OUTPUT_DIR = "Scalograms_v2"
SAVE_SAMPLES = True
NUM_SAMPLES = 5
MAX_SCALOGRAMS = 1000
INCLUDE_INST_FREQ = True  # adds 3rd channel

SAMPLING_RATE = 1.5e6
WAVELET = "cmor0.8-0.2"  # sharper freq resolution
NUM_SCALES = 256


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_cwt(signal_complex, sampling_rate=1.5e6, wavelet=WAVELET, num_scales=NUM_SCALES):
    """Compute complex CWT and return amplitude and phase components."""
    sampling_period = 1 / sampling_rate
    scales = np.logspace(-1, 1.5, num=num_scales)
    coeffs, freqs = pywt.cwt(signal_complex, scales, wavelet, sampling_period=sampling_period)
    cwt_amp = np.abs(coeffs)
    cwt_phase = np.angle(coeffs)
    return cwt_amp, cwt_phase


def compute_inst_freq(signal_complex):
    """Instantaneous frequency from unwrapped phase derivative."""
    phase = np.unwrap(np.angle(signal_complex))
    inst_freq = np.diff(phase, prepend=phase[0])
    return inst_freq


def normalize_independent(*arrays):
    """Normalize each array independently to [0,1]."""
    normed = []
    for arr in arrays:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        normed.append(arr)
    return normed


def save_sample_images(channels, names, sample_dir, base_filename, snr):
    """Save each channel as grayscale image."""
    for img, name in zip(channels, names):
        img_norm = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)
        path = os.path.join(sample_dir, f"{base_filename}_{name}_snr{snr}.png")
        cv2.imwrite(path, img_norm)


# =============================================================
#   MAIN GENERATION FUNCTION
# =============================================================

def generate_scalograms(data_type, snr,
                        include_inst_freq=INCLUDE_INST_FREQ,
                        max_scalograms=MAX_SCALOGRAMS,
                        save_samples=SAVE_SAMPLES,
                        num_samples=NUM_SAMPLES):
    input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", data_type)
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", data_type)
    sample_dir = os.path.join("ScalogramSamples_v2", f"snr_{snr}", data_type)

    os.makedirs(output_dir, exist_ok=True)
    if save_samples:
        os.makedirs(sample_dir, exist_ok=True)

    scalogram_count = 0
    sample_count = 0

    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    for filename in files:
        if max_scalograms is not None and scalogram_count >= max_scalograms:
            break

        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)
        I, Q = data[:, 0], data[:, 1]
        signal_complex = I + 1j * Q

        # Compute wavelet transform
        cwt_amp, cwt_phase = compute_cwt(signal_complex, SAMPLING_RATE)

        # Optional instantaneous frequency channel
        if include_inst_freq:
            inst_freq = compute_inst_freq(signal_complex)
            cwt_freq, _ = compute_cwt(inst_freq)
            # resize later

        # Resize for CNN
        cwt_amp = cv2.resize(cwt_amp, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cwt_phase = cv2.resize(cwt_phase, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        if include_inst_freq:
            cwt_freq = cv2.resize(cwt_freq, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        # Normalize independently
        if include_inst_freq:
            cwt_amp, cwt_phase, cwt_freq = normalize_independent(cwt_amp, cwt_phase, cwt_freq)
            stacked = np.stack([cwt_amp, cwt_phase, cwt_freq], axis=-1)
        else:
            cwt_amp, cwt_phase = normalize_independent(cwt_amp, cwt_phase)
            stacked = np.stack([cwt_amp, cwt_phase], axis=-1)

        # Save as .npy
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}_snr{snr}.npy")
        np.save(output_path, stacked.astype(np.float32))
        scalogram_count += 1

        # Save samples
        if save_samples and sample_count < num_samples:
            if include_inst_freq:
                save_sample_images(
                    [cwt_amp, cwt_phase, cwt_freq],
                    ["amp", "phase", "freq"],
                    sample_dir, base_filename, snr
                )
            else:
                save_sample_images(
                    [cwt_amp, cwt_phase],
                    ["amp", "phase"],
                    sample_dir, base_filename, snr
                )
            sample_count += 1

    print(f"[✓] {data_type} | SNR {snr} → {scalogram_count} scalograms generated.")


# =============================================================
#   MAIN SCRIPT
# =============================================================

if __name__ == "__main__":
    # Full 24-class RadioML set
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

    for snr in SNR_LEVELS:
        print(f"\n{'='*60}")
        print(f" Processing SNR Level: {snr} dB ")
        print(f"{'='*60}\n")
        for modulation in CLASSES:
            generate_scalograms(modulation, snr)

    print("\nAll scalograms generated successfully ✅")
