import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
iris = load_iris()
X = iris.data[:50] # Use a subset to keep training fast and clear
y = iris.target[:50]

# Normalize the input features
X = StandardScaler().fit_transform(X)

# 1. Compute pairwise squared Euclidean distances
def pairwise_distances(X):
    sum_X = np.sum(np.square(X), axis=1)
    dists = -2 * np.dot(X, X.T) + sum_X[:, np.newaxis] + sum_X[np.newaxis, :]
    return dists

# 2. Compute conditional probability matrix P in high-D space
def compute_p_matrix(X, sigma=1.0):
    N = X.shape[0]
    dists = pairwise_distances(X)
    P = np.exp(-dists / (2 * sigma**2))
    np.fill_diagonal(P, 0)  # No self-similarity
    P = P / (P.sum(axis=1, keepdims=True) + 1e-8)  # Row-wise normalization
    return P

# 3. Compute conditional probability matrix Q in low-D space
def compute_q_matrix(Y):
    dists = pairwise_distances(Y)
    Q = np.exp(-np.clip(dists, 0, 100))  # Avoid huge exponents
    np.fill_diagonal(Q, 0)
    Q = Q / (Q.sum(axis=1, keepdims=True) + 1e-8)
    return Q

# 4. Compute the gradient of the KL divergence
def compute_gradient(P, Q, Y):
    N, dim = Y.shape
    dY = np.zeros_like(Y)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = Y[i] - Y[j]
            dY[i] += 2 * (P[i, j] - Q[i, j]) * diff
    dY = np.clip(dY, -10, 10)  # Clip gradients to prevent explosion
    return dY

# 5. Main SNE optimization function
def sne(X, dim=2, sigma=1.0, n_iter=1000, learning_rate=0.1):
    N = X.shape[0]
    Y = np.random.randn(N, dim) * 0.001  # Small random init
    P = compute_p_matrix(X, sigma)

    for it in range(n_iter):
        Q = compute_q_matrix(Y)
        grad = compute_gradient(P, Q, Y)
        Y += learning_rate * grad

        if it % 100 == 0 or it == n_iter - 1:
            loss = np.sum(P * np.log((P + 1e-8) / (Q + 1e-8)))
            print(f"Iter {it}: KL Divergence = {loss:.4f}")
    return Y

# Run the SNE algorithm
Y_2d = sne(X, dim=2, sigma=1.0, n_iter=1000, learning_rate=0.1)

# 6. Visualize the 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(Y_2d[:, 0], Y_2d[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("2D Projection using SNE (from scratch)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()
