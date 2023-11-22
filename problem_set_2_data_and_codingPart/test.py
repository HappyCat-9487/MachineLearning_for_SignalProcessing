from scipy.io import loadmat
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import wiener
from scipy import signal
import matplotlib.pyplot as plt

data_signal = loadmat(r"/Users/luchengliang/ML_sp/Assignment/problem_set_2_data/problem2_5_signal.mat")
s = data_signal['signal']

data_noise = loadmat(r"/Users/luchengliang/ML_sp/Assignment/problem_set_2_data/problem2_5_noise.mat")
omega = data_noise['noise']

x = np.add(s, omega)

def xcorr(x, y, k):
    N = min(len(x),len(y))
    r_xy = (1/N) * signal.correlate(x,y,'full') # reference implementation is unscaled
    return r_xy[N-k-1:N+k]

def plot_frequecy_response():
    
    filter_coeffs = theta
    
    w, h = signal.freqz(filter_coeffs)
    
    freq, Pxx = signal.welch(x, fs=8000)
    
    #plot filter freq response
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title('Filter Frequency Response')
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.ylabel('dB')
    plt.grid()

    # plot the input signal  freq response
    plt.subplot(2, 1, 2)
    plt.title('Input Signal Frequecy Response')
    plt.semilogy(freq, Pxx)
    plt.xlabel('Hz')
    plt.ylabel('dB/Hz')
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_freq_response():
    # Compute frequency responses
    freq_ori, H_x = signal.freqz(x)
    freq_dhat, H_dhat = signal.freqz(dhat)
    freq_theta, H_theta = signal.freqz(theta, worN=L)

    # Plot the magnitude and phase responses
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.semilogx(freq_ori, 20 * np.log10(np.abs(H_x)), label='Original Signal')
    plt.semilogx(freq_dhat, 20 * np.log10(np.abs(H_dhat)), label='Filtered Signal')
    plt.title('Magnitude Response')
    plt.xlabel('Frequency (radians/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.semilogx(freq_ori, np.angle(H_x), label='Original Signal')
    plt.semilogx(freq_dhat, np.angle(H_dhat), label='Filtered Signal')
    plt.semilogx(freq_theta, np.angle(H_theta), label='Filter')
    plt.title('Phase Response')
    plt.xlabel('Frequency (radians/sample)')
    plt.ylabel('Phase (radians)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Filter length
L = 100

# Compute Wiener filter
r_uu = xcorr(x, x, L-1)
R_uu = toeplitz(r_uu[L-1:])
r_du = xcorr(s, x, L-1)
theta = np.linalg.solve(R_uu, r_du[L-1:])

# Filter noisy signal
dhat = signal.lfilter(theta, 1, x) 

plot_freq_response()