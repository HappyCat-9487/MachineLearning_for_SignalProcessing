#%%
import numpy as np
from scipy.io import loadmat
import librosa
import librosa.display
import matplotlib.pyplot as plt
from random import randrange
from sklearn.svm import SVR
#%%
def awgn(signal, snr):
    x_watts = signal ** 2
    # Set a target SNR
    target_snr_db = snr
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    y_volts = signal + noise_volts    
    return y_volts

#Load the data
data = loadmat('./problem3_6.mat')
t = data['t'].flatten()
y = data['y'].flatten()
x = t

# parameters
N=t.size
snr = 10 #dB
percent_outlier = 0.1

# learning parameters
epsilon=0.01
kernel_type='Gaussian'
kernel_params=2
C=1e3

# Add white Gaussian noise
y_noised = awgn(y, snr)

# convert data to proper dimensions in order to fit requirements of the library
x_col = x.reshape(( np.size(x), 1))
y_row = np.copy(y_noised)
t_col = t.reshape(( np.size(t), 1))

t_col = np.around(t_col, decimals=4)
x_col = np.around(x_col, decimals=4)
y_row = np.around(y_row, decimals=4)

# ---------- Support Vectore Regression -----------
gamma = 1/(np.square(kernel_params)) # gamma needs to be calculated in order to use 'Gaussian' kernel, which is not available in the library
regressor = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)

regressor.fit(x_col,y_row)
y_pred = regressor.predict(t_col)


# Find outliers using threshold
threshold = 10 * epsilon
outsider = np.zeros(len(t))
for i, sv_index in enumerate(regressor.support_):
    j = np.where(x_col == x_col[sv_index])[0][0]
    if abs(y_row[sv_index] - y_pred[j]) > threshold:
        outsider[i] = 1

outsider = outsider.astype(bool)

# Get center of the chirp
max_value = np.max(y_pred)
ind = np.argmax(y_pred)
t_max = x[ind]
tmax_abbrev = round(t_max, 1)
max_value_abbrev = round(max_value, 3)

SNR = np.var(y_pred) / np.var(0.2 ** 2 * np.random.randn(len(y_pred)))


# plot
plt.figure(figsize=(16,10))
plt.stem(x_col[regressor.support_], y_row[regressor.support_], linefmt = 'none', markerfmt='yo', label='support vector', basefmt=" "
         , use_line_collection=True)
plt.stem(x_col, y_row,  linefmt = 'none', markerfmt='k.', label='noised values', basefmt=" ", use_line_collection=True)
plt.plot(t_col, y_pred, color = 'red') 
# Plot support vectors and outliers
plt.plot(x_col[outsider], y_row[outsider], 'o', markerfacecolor='none', markersize=15, color='r', label='Outliers')
#plt.plot(x_col[regressor.support_], y_row[regressor.support_], 'o', markerfacecolor='none', markersize=7, color='g', label='Support Vectors')
plt.axvline(t_max, color='r', linestyle='--', label='Center')
plt.scatter(t_max, max_value, color='red', s=50, label=f'Center at x={tmax_abbrev}, y={max_value_abbrev}')
plt.title("Support Vector Regression, C = %d, " % C + f'SNR: {SNR:.2f}')
plt.xlabel("Time in (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# %%
