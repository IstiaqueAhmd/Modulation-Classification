"""
==============================================================
RadioML 2018.01A → I/Q Eye Diagram Generator
==============================================================

Generates I-channel and Q-channel eye diagrams from I/Q data.
Requires a known Samples-Per-Symbol (SPS) rate.
Each output file: [224×224×2] NumPy array (Eye_I, Eye_Q).
--------------------------------------------------------------
Author : Istiaque (2025)
Updated : Oct 2025 (Modified for Eye Diagrams)
==============================================================
"""

import os
import numpy as np
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

# --- Eye Diagram Specific Configuration ---
SAMPLES_PER_SYMBOL = 8      # CRITICAL: Must be correct for the dataset
SYMBOLS_TO_PLOT = 2         # How many symbols to overlay (2 is standard)
RESOLUTION = 224            # Output image resolution (H and W)
# ----------------------------------------

BASE_INPUT_DIR = "Dataset"      # Root where /snr_xx/class/*.npy are stored
BASE_OUTPUT_DIR = "EyeDiagrams" # Where eye diagrams will be saved
SAVE_SAMPLES = True           # Save example images for sanity check
NUM_SAMPLES = 5               # # of sample images per modulation per SNR
MAX_EYE_DIAGRAMS = 1000         # Limit per class (None = all)


# =============================================================
#   HELPER FUNCTIONS
# =============================================================

def compute_eye_diagram(signal, sps, symbols_to_plot=2, resolution=224):
    """
    Generate a 2D eye diagram histogram from a 1D signal.
    """
    # 1. Calculate segment length
    segment_len = sps * symbols_to_plot
    
    # 2. Normalize the entire signal's amplitude to [0, 1] for histogram bins
    # We add a small epsilon to prevent division by zero if signal is flat
    signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    
    # 3. Create all x and y coordinates for the histogram
    # We use non-overlapping segments, striding by `sps`
    num_segments = (len(signal_norm) - segment_len) // sps
    
    # Create x-coordinates (time axis, 0 to 1)
    x_coords = np.tile(np.linspace(0, 1, segment_len), num_segments)
    
    # Create y-coordinates (amplitude axis, 0 to 1)
    y_coords = np.zeros(num_segments * segment_len)
    for i in range(num_segments):
        start = i * sps
        end = start + segment_len
        y_coords[i * segment_len : (i + 1) * segment_len] = signal_norm[start:end]

    # 4. Create the 2D histogram
    # This is the "plotting" step
    hist, _, _ = np.histogram2d(
        x_coords,
        y_coords,
        bins=[resolution, resolution],
        range=[[0, 1], [0, 1]]
    )
    
    # 5. Log-scale for better contrast (hits are not linear)
    hist = np.log1p(hist)
    
    # 6. Normalize the final image to [0, 1]
    hist = (hist - hist.min()) / (hist.max() - hist.min() + 1e-8)
    
    # 7. Transpose because np.histogram2d swaps x and y axes
    return hist.T


def normalize_stack(eye_I, eye_Q):
    """Normalize I and Q eye diagrams jointly (0–1 range)."""
    stacked = np.stack([eye_I, eye_Q], axis=0)
    # No need for min/max normalization, compute_eye_diagram already did it
    return stacked


def save_sample_images(eye_I, eye_Q, sample_dir, base_filename, snr):
    """Save grayscale I-channel & Q-channel eye diagrams."""
    i_path = os.path.join(sample_dir, f"{base_filename}_eye_I_snr{snr}.png")
    q_path = os.path.join(sample_dir, f"{base_filename}_eye_Q_snr{snr}.png")

    # Scale to 0–255 for saving as grayscale PNG
    # Already normalized 0-1 from the helper function
    i_img = (255 * eye_I).astype(np.uint8)
    q_img = (255 * eye_Q).astype(np.uint8)

    cv2.imwrite(i_path, i_img)
    cv2.imwrite(q_path, q_img)

# =============================================================
#   MAIN GENERATION FUNCTION
# =============================================================

def generate_eye_diagrams(data_type, snr,
                           max_diagrams=None,
                           save_samples=False,
                           num_samples=5):
    input_dir = os.path.join(BASE_INPUT_DIR, f"snr_{snr}", data_type)
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"snr_{snr}", data_type)
    sample_dir = os.path.join("EyeDiagramSamples", f"snr_{snr}", data_type)

    os.makedirs(output_dir, exist_ok=True)
    if save_samples:
        os.makedirs(sample_dir, exist_ok=True)

    diagram_count = 0
    sample_count = 0

    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    for filename in files:
        if max_diagrams is not None and diagram_count >= max_diagrams:
            break

        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)

        # Separate I and Q components
        I, Q = data[:, 0], data[:, 1]
        
        # --- MODIFICATION: Compute Eye Diagrams ---
        
        eye_I = compute_eye_diagram(
            I, 
            sps=SAMPLES_PER_SYMBOL, 
            symbols_to_plot=SYMBOLS_TO_PLOT, 
            resolution=RESOLUTION
        )
        
        eye_Q = compute_eye_diagram(
            Q, 
            sps=SAMPLES_PER_SYMBOL, 
            symbols_to_plot=SYMBOLS_TO_PLOT, 
            resolution=RESOLUTION
        )

        # Normalize jointly (optional, as they are already 0-1)
        # This function just stacks them
        stacked = normalize_stack(eye_I, eye_Q)
        
        # --- END MODIFICATION ---

        stacked = np.transpose(stacked, (1, 2, 0))  # H×W×C

        # Save scalogram
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}_snr{snr}.npy")
        np.save(output_path, stacked.astype(np.float32))
        diagram_count += 1

        # Save sample visualization
        if save_samples and sample_count < num_samples:
            save_sample_images(eye_I, eye_Q, sample_dir, base_filename, snr)
            sample_count += 1

    print(f"[✓] {data_type} | SNR {snr} → {diagram_count} eye diagrams generated.")


# =============================================================
#   MAIN SCRIPT EXECUTION
# =============================================================

if __name__ == "__main__":
    for snr in SNR_LEVELS:
        print(f"\n{'='*60}")
        print(f" Processing SNR Level: {snr} dB ")
        print(f"{'='*60}\n")
        for modulation in CLASSES:
            generate_eye_diagrams(
                modulation,
                snr,
                max_diagrams=MAX_EYE_DIAGRAMS,
                save_samples=SAVE_SAMPLES,
                num_samples=NUM_SAMPLES
            )

    print("\nAll eye diagrams generated successfully ✅")