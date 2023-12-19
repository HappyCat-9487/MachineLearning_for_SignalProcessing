#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%%
def column_switch(A, A_hat):
    # Compute differences and normalize columns
    diff = np.abs(A - A_hat)
    norm_diff = diff / np.linalg.norm(diff, axis=0)

    # Check if columns need to be switched in A_hat
    for i in range(A.shape[1] - 1):
        if np.sum(norm_diff[:, i]) > np.sum(norm_diff[:, i + 1]):
            # Switch columns in A_hat
            A_hat[:, [i, i + 1]] = A_hat[:, [i + 1, i]]

    return A_hat

#%%
def error_cal(A, A_hat):
    
    #Check if columns need to be switched in A_hat
    A_hat = column_switch(A, A_hat)
    
    # Normalize columns of A and A_hat
    A_normalized = A / np.linalg.norm(A, axis=0)
    A_hat_normalized = A_hat / np.linalg.norm(A_hat, axis=0)

    # Compute the absolute difference between normalized matrices
    diff = np.abs(A_normalized - A_hat_normalized)

    # Compute the sum of coefficients from the absolute difference
    error = np.sum(diff)
    return error

#%%
# Independent Component Analysis function
def ICA(x, mu, num_components, iters, mode, A):
    # Random initialization
    W = np.random.rand(num_components, num_components)
    N = np.size(x, 0)

    if mode=='superGauss':
        phi = lambda u : 2*np.tanh(u)
    elif mode=='subGauss':
        phi = lambda u : u-np.tanh(u)
    else:
        print("Unknown mode")
        return W

    errors = []
    
    for i in range(iters):
        u = W @ x.T 
        dW = (np.eye(num_components) - phi(u) @ u.T/N) @ W 
        # Uniform distribution, so take average for E[]
        # Update
        W = W + mu*dW
        A_hat = W.T
        error = error_cal(A, A_hat)
        errors.append(error)
        
    return W, errors

#%%
def Find_the_Source(s):
    
    # Mix signals
    A = np.array([[3, 1], [1, 1]])
    x = (A@s).T 
    r = s.T

    # calculate ica
    mu = 0.1
    components = 2
    iterations = 100

    # Mean across the first (column) axis
    col_means = np.mean(x, axis=0)
    x = x - col_means

    # run ICA
    W, errors = ICA(x, mu, components, iterations, 'subGauss', A)

    # Normalize unmixing matrix
    W = np.divide(W, np.max(W))

    # Compute unmixed signals
    y = (W@x.T).T


    # Plotting
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)  # Subplot 1: Source Data
    plt.plot(r[:, 0], r[:, 1], '.')
    plt.xlabel('Source 1')
    plt.ylabel('Source 2')
    plt.title('Source Data')

    plt.subplot(2, 2, 2)  # Subplot 2: Generated Data
    plt.plot(x[:, 0], x[:, 1], '.')
    plt.xlabel('Generated data 1')
    plt.ylabel('Generated data 2')
    plt.title('Generated Data')

    plt.subplot(2, 2, 3)  # Subplot 3: Data Projection on ICA Axis
    plt.plot(y[:, 0], y[:, 1], '.')
    plt.xlabel('Estimated source 1')
    plt.ylabel('Estimated source 2')
    plt.title('Data Projection on ICA Axis')

    iters = np.arange(1, iterations + 1)
    plt.subplot(2, 2, 4)  # Subplot 4: Error by Iteration
    plt.plot(iters, errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error between A and A hat')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

#%%
# generate data

N = 5000

# Define two non-gaussian uniform components
s1 = np.random.rand(N)
s2 = np.random.rand(N)
s = np.array(([s1, s2]))

# Define one non-gaussian uniform component and one beta component
s1b = np.random.rand(N)
s2b = np.random.beta(0.1, 0.1, size=N)
sb = np.array(([s1b, s2b]))

# Define one non-gaussian uniform component and one gaussian component
s1n = np.random.rand(N)
s2n = np.random.normal(size=N)
sn = np.array(([s1n, s2n]))

#Define multivariate normal distribution with
#μ = (0, 1), Σ = [2 0.25; 0.25 1]
mean = [0, 1]
covariance = [[2, 0.25], [0.25, 1]]
sm_r = np.random.multivariate_normal(mean, covariance, N)
sm = sm_r.T

Find_the_Source(s)
Find_the_Source(sb)
Find_the_Source(sn)
Find_the_Source(sm)
