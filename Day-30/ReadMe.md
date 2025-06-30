# Day 30 – Stochastic Neighbor Embedding (SNE) 
dived deep into **Stochastic Neighbor Embedding (SNE)** — a non-linear, unsupervised dimensionality reduction technique used for **visualizing high-dimensional data in 2D or 3D** while preserving local neighborhood structures.

---

## What is SNE?

SNE maps high-dimensional data (H-dim) into a lower-dimensional space (L-dim, typically 2D or 3D) such that **similar points stay close together** and dissimilar ones remain far apart.

---

## Why Use SNE?

- **Data visualization**: When you want to project high-dimensional data into 2D for human understanding.
- **Clustering insight**: To reveal natural groupings in data.
- **Manifold learning**: Understand how data lies on a lower-dimensional manifold.

---

## High-D vs Low-D Space

| Concept       | High-Dimensional Space (Input) | Low-Dimensional Space (Output) |
|---------------|-------------------------------|---------------------------------|
| Dimension     | D (e.g., 784 for MNIST)        | d (e.g., 2 or 3 for plotting)   |
| Similarities  | Measured using Gaussians       | Measured using Student-t (in t-SNE) or Gaussian |
| Probabilities | `P_ij`: Similarity of i→j      | `Q_ij`: Similarity of i→j in low-D |
| Goal          | Match `P_ij` and `Q_ij`        | Learn `Y_i` such that this happens |

---

## Core Idea

- Compute pairwise similarity in **high-dimensional space**:  
  \( P_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma^2)} \)

- Compute pairwise similarity in **low-dimensional space**:  
  \( Q_{j|i} = \frac{\exp(-||y_i - y_j||^2)}{\sum_{k \neq i} \exp(-||y_i - y_k||^2)} \)

- Minimize the **Kullback-Leibler divergence** between these distributions:  
  \[
  C = \sum_i \sum_j P_{j|i} \log\left(\frac{P_{j|i}}{Q_{j|i}}\right)
  \]

---

## How It Works – Steps

### 1. Compute High-D Similarities `P`
```python
dists = pairwise_distances(X)
P = np.exp(-dists / (2 * sigma**2))
P = P / (P.sum(axis=1, keepdims=True) + 1e-8)
```
- Gaussian kernel-based similarity.
- Normalized row-wise to get conditional probabilities.

### 2. Initialize Low-D Embedding Y
```python
Y = np.random.randn(N, dim) * 0.001
```
Start with random positions in 2D.

### 3. Compute Low-D Similarities Q
```python
dists = pairwise_distances(Y)
Q = np.exp(-np.clip(dists, 0, 100))
Q = Q / (Q.sum(axis=1, keepdims=True) + 1e-8)
```
- Compute similarities in 2D using the same method.
- Clipping avoids numerical instability.

### 4. Compute Gradient of Loss
```python
for i in range(N):
    for j in range(N):
        diff = Y[i] - Y[j]
        grad[i] += 2 * (P[i, j] - Q[i, j]) * diff
```
- Pulls similar points together and pushes dissimilar points apart.
- Gradient is clipped to prevent explosion.

### 5. Update Positions
```python
Y += learning_rate * grad
```
- Gradual optimization to minimize KL divergence.

## Why KL Divergence?
Measures the difference between the high-D and low-D similarity distributions.
As Q gets closer to P, KL divergence decreases.

## How Is It Unsupervised?
The algorithm doesn't use any labels.
It learns to preserve the local geometry of the data by purely comparing pairwise similarities.

## When Does Q Break?
You must prevent:

Division by zero (Q.sum(...) + 1e-8)
Overflow in exponentials (use np.clip)
Large gradients (clip them!)

## Difference from t-SNE?

| Feature       | SNE               | t-SNE                        |
| ------------- | ----------------- | ---------------------------- |
| Low-D kernel  | Gaussian          | Student-t (heavy-tailed)     |
| Symmetry      | Asymmetric `P_ij` | Symmetric `P_ij`             |
| Cost function | KL divergence     | KL divergence (symmetric P)  |
| Speed         | Slower            | Faster with Barnes-Hut, etc. |
    
