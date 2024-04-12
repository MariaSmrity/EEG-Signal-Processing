import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
from scipy.signal import spectrogram
import seaborn as sns
import pandas as pd
# Load the EEG data from the .mat file
eeg_data = loadmat('ourEEG.mat')

# Extract the EEG signal from the loaded data
eeg_signal = eeg_data['data']  # 'data' is the variable name containing EEG data

# Sampling frequency of the EEG signal (Hz)
fs = 1000  # Assuming sampling frequency is 1000 Hz, change accordingly

# Subtract the mean of the EEG signal to make it oscillate around zero
eeg_signal = eeg_signal - np.mean(eeg_signal)

# Parameters for noisy spectrum
nperseg_noisy = 128
noverlap_noisy = 64
nfft_noisy = 128

# Parameters for smoother spectrum
nperseg_smooth = 512
noverlap_smooth = 256
nfft_smooth = 1024

# Calculate power spectrum estimates using Welch's method with parameters for noisy spectrum
f_noisy, Pxx_noisy = welch(eeg_signal, fs=fs, nperseg=nperseg_noisy, noverlap=noverlap_noisy, nfft=nfft_noisy,axis = 0)

# Calculate power spectrum estimates using Welch's method with parameters for smoother spectrum
f_smooth, Pxx_smooth = welch(eeg_signal, fs=fs, nperseg=nperseg_smooth, noverlap=noverlap_smooth, nfft=nfft_smooth, axis =0)

"""# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.semilogy(f_noisy, Pxx_noisy)
plt.title('Power Spectral Density - Noisy Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(f_smooth, Pxx_smooth)
plt.title('Power Spectral Density - Smooth Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid(True)

plt.tight_layout()
plt.show()"""


#---------------Task 2
#df = sns.load_dataset('ourEEG')

# Parameters for time-frequency analysis with relatively high variance (noisy)
nperseg_noisy = 128
noverlap_noisy = 64

# Parameters for time-frequency analysis with smoother output
nperseg_smooth = 512
noverlap_smooth = 256
# Perform time-frequency analysis using spectrogram with parameters for noisy spectrum
f_noisy, t_noisy, Sxx_noisy = spectrogram(eeg_signal, fs=fs, nperseg=nperseg_noisy, noverlap=noverlap_noisy,axis=0)
print(Sxx_noisy.shape, f_noisy.shape, t_noisy.shape)
# Perform time-frequency analysis using spectrogram with parameters for smoother spectrum
f_smooth, t_smooth, Sxx_smooth = spectrogram(eeg_signal, fs=fs, nperseg=nperseg_smooth, noverlap=noverlap_smooth,axis=0)


# Transpose Sxx_noisy to rearrange the dimensions
#Sxx_noisy_transposed = np.transpose(Sxx_noisy, axes=(1, 0, 2))
#print(Sxx_noisy_transposed.shape)
print(Sxx_noisy.shape)
# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.pcolormesh(t_noisy, f_noisy, 10 * np.log10(Sxx_noisy[:,0,:,]), shading='gouraud') # Displaying first segment
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.title('Time-Frequency Analysis - Noisy Spectrum')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.subplot(2, 1, 2)
plt.pcolormesh(t_smooth, f_smooth, 10 * np.log10(Sxx_smooth[:,0,:,]), shading='gouraud')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.title('Time-Frequency Analysis - Smooth Spectrum')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()