#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.fft import dct, idct
from scipy.io import loadmat

#%%
#Load data from problem3_1.mat
data = loadmat('/Users/luchengliang/ML_sp/Problem Set 3/problem3_1.mat')
n = data['n'].flatten()
x = data['x'].flatten()

#%%
# Signal setup
N = 2**5  # number of observations to make
l = 2**9  # signal length

B = np.zeros((N, l))
for i in range(N):
    B[i, n[i]] = 1

# Since it is sparse in the IDCT domain, i.e. B*x = B*Phi*X = BF*X,
# where X sparse,  BF = B*Phi; and Phi is the DCT matrix, Phi = dctmtx(l);.
# Equivalently, since IDCT = transpose of DCT using idct we can write
BF = idct(B, norm='ortho')

lambda_ = 0.005
model = Lasso(lambda_, fit_intercept=False)
model.fit(BF, x)
solsB = model.coef_

# create IST solution
nsteps = 100000
t_ = np.zeros((l, nsteps))
mu = 0.1
for k in range(1, nsteps):
    e = x - BF@t_[:, k-1] 
    t_tilde = t_[:, k-1] + mu*BF.T@e 
    t_[:, k] = np.sign(t_tilde)*np.maximum(abs(t_tilde) - lambda_*mu, 0) 
solsIST = t_[:, -1]

# Get K, ai, and mi
sols = solsIST*np.sqrt(2 / len(solsIST))  # normalize according to DCT specification
a = sols[np.nonzero(sols)]  # get ai values
k = np.count_nonzero(sols)  # get number of non-zero components in DCT domain
m = np.where(sols)[0]  # get positions of non-zero components
print("a_i values:", a)
print("K status:", k)
print("m_j values:", m)

# Reconstruct the multitone signal for verification
t = np.arange(0, l)  # time axis
s_multitone = np.zeros(l)
for i in range(k):
    s_multitone += a[i] * np.cos((np.pi * (2 * m[i] - 1) * t) / (2 * l))
# `s_multitone` now contains the reconstructed multitone signal

# plot solutions
fig, ax= plt.subplots(1, 1, figsize=(6, 3))
ax.stem(solsB, markerfmt='bo', label='sklearn Lasso', basefmt=' ')
ax.stem(solsIST, markerfmt='ro', label='IST', basefmt=' ')
ax.legend()
ax.set_title('Solutions')

# Take the inverse IDCT (i.e. the DCT) in order to compute the estimated signal. 
x_hat = dct(solsIST, norm='ortho')
b_hat = dct(solsB, norm='ortho')

fig, ax= plt.subplots(3, 1, figsize=(12, 8))
ax[0].plot(b_hat)
ax[0].plot(n, x, 'r.')
ax[0].set_title('Estimated in time domain by LASSO')
ax[1].plot(x_hat)
ax[1].plot(n, x, 'r.')
ax[1].set_title('Estimated in time domain by IST')
ax[2].plot(s_multitone)
ax[2].set_title('Estimated signal reconstructed using multitone formula')
fig.tight_layout()
plt.show()
