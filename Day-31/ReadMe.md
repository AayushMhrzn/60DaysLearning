# Day 31 – t-SNE from Scratch

**t-SNE (t-distributed Stochastic Neighbor Embedding)** is a powerful **non-linear dimensionality reduction technique** designed for the **visualization of high-dimensional data** in 2D or 3D. Unlike PCA (linear), t-SNE preserves the **local structure** of the data.

implemented t-SNE from scratch using the Iris dataset, reducing 4D data to 2D and visualizing its clusters.

---

##  Mathematical Theory Behind t-SNE

### 1. **High-Dimensional Similarity (P)**
We define similarity between high-dimensional points using **conditional probabilities**.

For a point \( x_i \), we define:

`P_{j|i} = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σₖ₍ₖ≠ᵢ₎ exp(-||xᵢ - x_k||² / 2σᵢ²)`

This is the **probability that \( x_i \) picks \( x_j \) as its neighbor**, assuming a Gaussian centered at \( x_i \).

We make it **symmetric**:

`P_{ij} = (P_{j|i} + P_{i|j}) / 2n`

We tune \( \sigma_i \) for each point using **binary search** to match a desired **perplexity**, which reflects the effective number of neighbors.

---

### 2. **Low-Dimensional Similarity (Q)**
In the 2D space, we define similarity using a **Student-t distribution** (with 1 DOF):

`Q_{ij} = (1 + ||yᵢ - yⱼ||²)^(-1) / Σₖₗ₍ₖ≠ₗ₎ (1 + ||y_k - y_l||²)^(-1)`

The heavy tail ensures that dissimilar points are **pushed far apart**.

---

### 3. **Cost Function (KL Divergence)**
We minimize the **Kullback-Leibler divergence** between \( P \) and \( Q \):

`C = KL(P || Q) = Σ₍ᵢ≠ⱼ₎ P_{ij} * log(P_{ij} / Q_{ij})`

This is optimized using **gradient descent**.

---

##  Full Code Summary

###  Pairwise Distance Matrix
```python
def compute_pairwise_distances(X):
    sum_X = np.sum(np.square(X), 1)
    return -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]
```
- Efficient computation of squared Euclidean distances using broadcasting.

### Gaussian Conditional Probabilities
```python
def Hbeta(D, beta=1.0):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    return H, P / sumP
```
- Computes entropy H and conditional probabilities 𝑃𝑗∣𝑖 for a given 𝛽(inverse variance).

### Binary Search for Perplexity
```python
def x2p(X, tol=1e-5, perplexity=30.0):
    ...
    # For each point, find beta such that entropy matches log(perplexity)
```
- Adjusts Gaussian width per point to match target perplexity.

### PCA for Dimensionality Reduction
```python
def PCA(X, no_dims=50):
    X -= np.mean(X, axis=0)
    cov = np.dot(X.T, X)
    U, _, _ = np.linalg.svd(cov)
    return np.dot(X, U[:, :no_dims])
``` 
- Reduces original dimension using top 50 principal components.

### Low-Dimensional Similarity and KL Minimization
```python
def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000, learning_rate=200.0):
    ...
    Y = np.random.randn(n, no_dims)
    for iter in range(max_iter):
        ...
        Q = num / np.sum(num)
        PQ = P - Q
        dY[i] = 4 * np.sum(((PQ[:, i] * num[:, i])[:, None]) * (Y[i] - Y), axis=0)
        ...
```
**Function arguments:**
- `X`: input high-dimensional dataset (n samples, d features)
- `no_dims`: final target dimension (usually 2 or 3 for visualization)
- `initial_dims`: number of dimensions after PCA (e.g., 50)
- `perplexity`: controls the size of the local neighborhood
- `max_iter`: number of gradient descent iterations
- `learning_rate`: controls how fast Y gets updated

##  Why PCA First?

PCA is used to:
- **Denoise** the data
- **Speed up** t-SNE (by reducing dimensions from hundreds to 30–50)
- Avoid **curse of dimensionality**

```python
    Y = np.random.randn(n, no_dims)
```
- Randomly initialize low-dimensional embeddings Y (shape [n, 2] if no_dims=2)

```python
dY = np.zeros_like(Y)
iY = np.zeros_like(Y)
gains = np.ones_like(Y)
```
- `dY`: Gradient of the cost function (used in backprop update)
- `iY`: Previous step (momentum term)
- `gains`: Adaptive learning rates for each parameter (inspired by Adagrad)

```python
 P = x2p(X, perplexity=perplexity)
 P *= 4.0  # early exaggeration
 P = np.maximum(P, 1e-12)
 ```
 - Calculates the symmetric joint probability matrix P in the high-dimensional space based on distances, using Gaussian kernel with tuned bandwidth per point (based on desired perplexity).
 - Temporarily multiply `P` by 4 to create stronger attraction forces at the beginning. Helps form better clusters initially
- Avoid log(0) errors later in KL divergence calculation by clamping very small values.

```python
for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1 / (1 + compute_pairwise_distances(Y))
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        PQ = P - Q
```
- Begin **gradient descent** loop
- Compute low-dimensional similarity Q using Student-t distribution
- Normalize num to turn it into a probability matrix Q, and clamp small values.
- Calculate the **difference between high-dim and low-dim similarities** P-Q. This is the **error signal** for gradient descent.

```python
        for i in range(n):
            dY[i] = 4 * np.sum(((PQ[:, i] * num[:, i])[:, None]) * (Y[i] - Y), axis=0)
```
- This is the core gradient computation for 𝑦𝑖. This is derived from:
`∂C/∂yᵢ = 4 * Σⱼ (Pᵢⱼ − Qᵢⱼ) * qᵢⱼ * (yᵢ − yⱼ)`

```python
    gains = (gains + 0.2) * (np.sign(dY) != np.sign(iY)) + (gains * 0.8) * (np.sign(dY) == np.sign(iY))
        gains[gains < 0.01] = 0.01
        iY = 0.8 * iY - learning_rate * gains * dY
        Y += iY
        Y -= np.mean(Y, axis=0)
```
➡ Implements **adaptive learning rate**:
- If the sign of gradient has changed → increase gain (was moving wrong way)
- If sign is same → decrease gain (it's converging)
➡ Ensure minimum learning rate for stability
➡ Update `iY
➡ Apply the gradient update to positions
➡ Center `Y` around origin (zero mean) to prevent it from drifting off

## Conclusion 
| Step      | Concept                         | Purpose              |
| --------- | ------------------------------- | -------------------- |
| PCA       | Linear dimensionality reduction | Preprocess & denoise |
| `x2p`     | Gaussian similarities in high-D | Build matrix P       |
| `num → Q` | t-distribution in low-D         | Build matrix Q       |
| `P - Q`   | Error signal                    | Drives updates       |
| `dY`      | Gradient of KL divergence       | Controls movement    |
| `gains`   | Adaptive learning rate          | Speeds up learning   |
| `iY`      | Momentum-based update           | Stabilizes descent   |
