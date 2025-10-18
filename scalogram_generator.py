#import necessary libraries
import os
import numpy as np
import scipy.io as sio
import pywt
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

""" 
This code generates Amplitude and Phase scalograms using I/Q data.
"""

# Configuration parameters
MAX_SCALOGRAMS = 1000  # Set to None to process all available scalograms, or specify a number
SAVE_SAMPLES = True   # Set to False if you don't want to save sample images
NUM_SAMPLES = 5       # Number of sample images to save (only used if SAVE_SAMPLES is True)
SNR = 10

def generateWaveletTransform(data_type, snr, max_scalograms=None, save_samples=False, num_samples=5):
    input_dir = f'Dataset/snr_{snr}/{data_type}'
    output_dir = f'Scalograms/snr_{snr}/{data_type}'
    samples_dir = f'ScalogramSamples/snr_{snr}/{data_type}'

    os.makedirs(output_dir, exist_ok=True)
    if save_samples:
        os.makedirs(samples_dir, exist_ok=True)

    sample_count = 0
    scalogram_count = 0


    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            # Check if we've reached the maximum number of scalograms to process
            if max_scalograms is not None and scalogram_count >= max_scalograms:
                break
                
            frame_path = os.path.join(input_dir, filename)
            data = np.load(frame_path)

            I = data[:, 0]  # First Col: I component (1024 samples)
            Q = data[:, 1]  # Second Col: Q component (1024 samples) 

            amplitude = np.sqrt(I ** 2 + Q ** 2)
            phase = np.arctan2(Q, I)

            wavelet = 'cmor1.5-0.5'
            scales = np.logspace(0.5, 2, num=200)

            def compute_cwt(signal):
                coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / 1000)
                coeffs = np.abs(coeffs)
                scaler = MinMaxScaler()
                return scaler.fit_transform(coeffs)

            cwt_amplitude = compute_cwt(amplitude)
            cwt_phase = compute_cwt(phase)

            cwt_amplitude = cv2.resize(cwt_amplitude, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            cwt_phase = cv2.resize(cwt_phase, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            stacked_scalogram = np.stack([cwt_amplitude, cwt_phase], axis=-1)

            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')
            np.save(output_path, stacked_scalogram.astype(np.float32))
            scalogram_count += 1

            if save_samples and sample_count < num_samples:
                amp_img_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_amp.png")
                phase_img_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_phase.png")

                plt.imsave(amp_img_path, cwt_amplitude, cmap='gray', vmin=0, vmax=1)
                plt.imsave(phase_img_path, cwt_phase, cmap='gray', vmin=0, vmax=1)

                sample_count += 1
    if save_samples:
        print(f"Stacked wavelet transforms saved for {data_type}: {scalogram_count} scalograms, including {sample_count} raw sample images.")
    else:
        print(f"Stacked wavelet transforms saved for {data_type}: {scalogram_count} scalograms.")


# Run for multiple modulation types
classes = [
  "OOK", "4ASK", "8ASK",
  "BPSK", "QPSK", "8PSK", 
  "16APSK", "64QAM", 
  "AM-SSB-WC","AM-DSB-WC",
  "FM", "GMSK", "OQPSK"
]

for data_type in classes:
    generateWaveletTransform(data_type, SNR, max_scalograms=MAX_SCALOGRAMS, save_samples=SAVE_SAMPLES, num_samples=NUM_SAMPLES)

