import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and standardize the dataset
iris = load_iris()
X = iris.data
y = iris.target
X = StandardScaler().fit_transform(X)

# Step 1: Compute pairwise distances
def compute_pairwise_distances(X):
    sum_X = np.sum(np.square(X), 1)
    return -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]

# Step 2: Compute conditional probability P_j|i using binary search on β
def Hbeta(D, beta=1.0):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    return H, P / sumP

# Step 3: Create symmetric joint probability matrix P
def x2p(X, tol=1e-5, perplexity=30.0):
    n = X.shape[0]
    D = compute_pairwise_distances(X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        betamin, betamax = -np.inf, np.inf
        Di = np.delete(D[i], i)
        H, thisP = Hbeta(Di, beta[i])
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                beta[i] = beta[i] * 2. if betamax == np.inf else (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                beta[i] = beta[i] / 2. if betamin == -np.inf else (beta[i] + betamin) / 2.
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    return (P + P.T) / (2 * n)

# Step 4: Reduce dimensionality with PCA
def PCA(X, no_dims=50):
    X -= np.mean(X, axis=0)
    cov = np.dot(X.T, X)
    U, _, _ = np.linalg.svd(cov)
    return np.dot(X, U[:, :no_dims])

# Step 5: Full t-SNE implementation
def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000, learning_rate=200.0):
    X = PCA(X, initial_dims)
    n, d = X.shape
    Y = np.random.randn(n, no_dims)
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    P = x2p(X, perplexity=perplexity)
    P *= 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)

    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1 / (1 + compute_pairwise_distances(Y))
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q
        for i in range(n):
            dY[i] = 4 * np.sum(((PQ[:, i] * num[:, i])[:, None]) * (Y[i] - Y), axis=0)

        gains = (gains + 0.2) * (np.sign(dY) != np.sign(iY)) + (gains * 0.8) * (np.sign(dY) == np.sign(iY))
        gains[gains < 0.01] = 0.01
        iY = 0.8 * iY - learning_rate * gains * dY
        Y += iY
        Y -= np.mean(Y, axis=0)

        if iter == 100:
            P /= 4.0  # stop early exaggeration

        if iter % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print(f"Iter {iter}: KL Divergence = {C:.4f}")

    return Y

# Run t-SNE
Y = tsne(X, no_dims=2, perplexity=30.0, max_iter=1000)

# Visualize the result
plt.figure(figsize=(8, 6))
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap='viridis', edgecolor='k', s=60)
plt.title("t-SNE on Iris Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.show()
