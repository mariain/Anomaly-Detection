import numpy as np

def estimategaussian(X):
    m, n = X.shape
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))

    mu = np.mean(X, axis=0)
    mu = mu.T
    sigma2 = np.var(X, axis=0)
    sigma2 = sigma2.T
    return mu, sigma2