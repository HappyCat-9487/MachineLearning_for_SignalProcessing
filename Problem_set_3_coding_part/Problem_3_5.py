#%%
import numpy as np
import matplotlib.pyplot as plt

# Set the parameters

q = 1
dt = 0.1
s = 0.21
F = np.array([
    [1, dt],
    [0, 1],
])
Q = q*np.array([
    [s**2, 0],
    [0,    0]
])

H = np.array([1, 0])
R = s**2*np.identity(2)
m0 = np.array([[0], [1]])
P0 = np.identity(2)

# Simulate data

np.random.seed(1)

steps = 10
X = np.zeros((len(F), steps))
Y = np.zeros((len(H), steps))
x = m0
for k in range(steps):
    x = F@x + s*np.random.randn(len(F), 1)
    y = H@x + s*np.random.randn(1, 1)
    X[:, k] = x[:, 0]  
    Y[:, k] = y[:, 0]


# Kalman filter

m = m0
P = P0
kf_m = np.zeros((len(m), Y.shape[1]))
kf_P = np.zeros((len(P), P.shape[1], Y.shape[1]))
for k in range(Y.shape[1]):
    m = F@m
    P = F@P@F.T + Q

    e = Y[:, k].reshape(-1, 1) - H@m
    S = H@P@H.T + R
    K = P@H.T@np.linalg.inv(S)
    m = m + K@e
    P = P - K@S@K.T

    kf_m[:, k] = m[:, 0]
    kf_P[:, :, k] = P

a = np.arange(1,11)
plt.figure()
plt.plot(a, X[0, :], '-')
plt.plot(a, Y[0, :], 'o')
plt.plot(a, kf_m[0, :], '-')
plt.legend(['True Trajectory', 'Measurements', 'Filter Estimate'])
plt.xlabel('$Steps$')
plt.ylabel('$Position$')
'''
plt.figure()
plt.plot(X[0, :], X[1, :], '-')
plt.plot(Y[0, :], Y[1, :], 'o')
plt.plot(kf_m[0, :], kf_m[1, :], '-')
plt.legend(['True Trajectory', 'Measurements', 'Filter Estimate'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
'''