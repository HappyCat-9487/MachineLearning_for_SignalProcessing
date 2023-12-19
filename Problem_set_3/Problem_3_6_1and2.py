#%%
#%%
import os
from scipy.io import loadmat
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
#%%

# Load bladerunner data
data = loadmat('./problem3_6.mat')
t = data['t'].flatten()
y = data['y'].flatten()

# Calculate the center of the signal
max_value = np.max(y)
ind = np.argmax(y)
tmax = t[ind]
max_value_abbrev = round(max_value, 3)

#Initial figure and plot the signal and its center point
fig, ax = plt.subplots()
ax.plot(t, y, label='clean')
ax.axvline(tmax, color='r', linestyle='--', label='Center')
ax.scatter(tmax, max_value, color='red', label=f'Center at x={tmax}, y={max_value_abbrev}')
ax.set_xlabel('time in sec')
ax.set_ylabel('amplitude')
ax.legend()
ax.grid()


#Initial another figure
fig, ax = plt.subplots()
y_original = y
ax.plot(t, y_original, label='clean', zorder=1)

# data parameters
np.random.seed(0)
N = t.size
percent_outlier = 0.1
snr = 10  # dB

# learning parameters
sigma = 0.004
C = 1e-2

# add white Gaussian noise
noise = np.random.randn(N)
noise *= (np.sum(y**2)/np.sum(noise**2)/10**(snr/10))**0.5
y += noise
ax.plot(t, y, label='with noise', zorder=0)

# Calculate the new center of the signal with noise
max_value_mednoise = np.max(y)
ind_mednoise = np.argmax(y)
tmax_mednoise = t[ind_mednoise]
tmax_mednoise_abbrev = round(tmax_mednoise, 1)
max_value_mednoise_abbrev = round(max_value_mednoise, 3)

# finish figure
ax.axvline(tmax_mednoise, color='r', linestyle='--', label='Center')
ax.scatter(tmax_mednoise, max_value_mednoise, color='red', label=f'Center at x={tmax_mednoise_abbrev}, y={max_value_mednoise_abbrev}')
ax.set_xlabel('time in sec')
ax.set_ylabel('amplitude')
ax.legend()
ax.grid()
#%%
# unbiased L2 Kernel Ridge Regression (KRR-L2)
# build kernel matrix
pair_dist = np.abs(t.reshape(-1, 1) - t.reshape(1, -1))
K = np.exp(-1/(sigma**2)*pair_dist**2) 
A = C*np.identity(N) + K  
sol = np.linalg.solve(A, y)

# Generate regressor
# NOTE: this loop can be optimized
samples = t[-1]
x = np.arange(0, samples + 0.2, 0.2)
M2 = len(x)
z0 = np.zeros(M2)
for k in range(M2):
    z0[k] = 0
    for j in range(N):
        value = np.exp(-1/(sigma**2)*(t[j] - x[k])**2)
        z0[k] += sol[j]*value

# Get the center of the chirp
max_value_re = np.max(z0)
ind_re = np.argmax(z0)
tmax_re = x[ind_re]
tmax_re_abbrev = round(tmax_re, 1)
max_value_re_abbrev = round(max_value_re, 3)
# Compute SNR
SNR = np.var(z0) / (0.2 ** 2 * np.var(np.random.rand(1, len(z0))))

# plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.set_xlabel('time in sec')
ax1.set_ylabel('amplitude')
ax1.plot(t, y, label='input signal', color='orange')
ax1.axvline(tmax_mednoise, color='r', linestyle='--', label='Center')
ax1.scatter(tmax_mednoise, max_value_mednoise, color='red', label=f'Center at x={tmax_mednoise_abbrev}, y={max_value_mednoise_abbrev}')
ax1.legend()
ax1.set_title(f'SNR: {SNR:.2f}')
ax1.grid()

ax2.set_xlabel('time in sec')
ax2.set_ylabel('amplitude')
ax2.plot(x, z0, label='reconstructed signal', color='purple')
ax2.axvline(tmax_re, color='r', linestyle='--', label='Center')
ax2.scatter(tmax_re, max_value_re, color='red', label=f'Center at x={tmax_re_abbrev}, y={max_value_re_abbrev}')
ax2.legend()
ax2.set_title(f'SNR: {SNR:.2f}')
ax2.grid()

plt.subplots_adjust(hspace=0.3)
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('time in sec')
ax.set_ylabel('amplitude')
ax.plot(x, z0, 'r', linewidth=1, label='Estimated signal')
ax.plot(t, y, '.', markeredgecolor=0.3*np.array([1, 1, 1]), markersize=5, label='Sampled signal')
ax.legend()
ax.set_title(f'SNR: {SNR:.2f}')
ax.grid()