# Day 29 – Principal Component Analysis (PCA) from Scratch

Principal Component Analysis (PCA) is an unsupervised **dimensionality reduction** technique used to reduce the number of features in a dataset while preserving as much variance (information) as possible. It is commonly used for **data visualization**, **preprocessing**, and **noise reduction**.

---

## Why PCA?

In high-dimensional datasets, many features may be correlated or redundant. PCA helps:
- Reduce feature space to 2D/3D for visualization
- Improve training speed and reduce overfitting
- Capture the most meaningful patterns in the data

---

## PCA Theory and Math

PCA identifies **principal components**—new axes formed by **linear combinations** of the original features—that capture the maximum variance.

### Step-by-Step Mathematical Process

1. **Standardize the dataset** (mean center):
   \[
   X_{\text{centered}} = X - \bar{X}
   \]
   where \( \bar{X} \) is the mean of each feature.

2. **Compute the Covariance Matrix**:
   \[
   \text{Cov}(X) = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}
   \]

3. **Eigen Decomposition**:
   Find eigenvalues and eigenvectors of the covariance matrix.
   \[
   \text{Cov}(X)v = \lambda v
   \]
   - \( v \): eigenvectors (principal axes)
   - \( \lambda \): eigenvalues (amount of variance explained)

4. **Sort eigenvectors by eigenvalues** (descending order)

5. **Select top-k eigenvectors** to form the **projection matrix**.

6. **Project the data** into the lower-dimensional space:
   \[
   X_{\text{reduced}} = X_{\text{centered}} \cdot W_k
   \]
   where \( W_k \) contains the top-k eigenvectors.

---

## PCA From Scratch on Iris Dataset
```python
# 3. Covariance Matrix
cov_mat = np.cov(X_meaned.T)
```
Explanation: Transpose needed because np.cov() expects variables in rows. This matrix tells us how features vary with each other.

```python
# 4. Eigen Decomposition
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
```
Explanation: Eigenvectors are directions of maximum variance; eigenvalues tell how much variance they capture.

```python
# 5. Sort eigenvectors by eigenvalues (descending)
sorted_idx = np.argsort(eigen_vals)[::-1]
eigen_vals = eigen_vals[sorted_idx]
eigen_vecs = eigen_vecs[:, sorted_idx]
```
Explanation: We want the top principal components — the directions with most information (largest eigenvalues).

```python
# 6. Choose top k eigenvectors
k = 2
eigen_vecs_k = eigen_vecs[:, :k]

# 7. Project the data
X_reduced = X_meaned @ eigen_vecs_k
```

Explanation: @ is matrix multiplication. We project the original 4D data onto the new 2D space defined by the top 2 principal components.