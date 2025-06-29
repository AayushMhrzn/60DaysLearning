import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load Dataset
data = load_iris()
X = data.data  # shape (150, 4)
y = data.target  # class labels for coloring

# 2. Standardize (mean-center)
X_meaned = X - np.mean(X, axis=0)

# 3. Covariance Matrix
cov_mat = np.cov(X_meaned.T)

# 4. Eigen Decomposition
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# 5. Sort eigenvalues and eigenvectors
sorted_idx = np.argsort(eigen_vals)[::-1]
eigen_vals = eigen_vals[sorted_idx]
eigen_vecs = eigen_vecs[:, sorted_idx]

# 6. Choose top k eigenvectors (k=2)
k = 2
eigen_vecs_k = eigen_vecs[:, :k]

# 7. Project the data
X_reduced = X_meaned @ eigen_vecs_k

# 8. Visualization
plt.figure(figsize=(8,6))
for class_val in np.unique(y):
    plt.scatter(X_reduced[y == class_val, 0], X_reduced[y == class_val, 1], label=data.target_names[class_val])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Iris Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

explained_variance_ratio = eigen_vals / np.sum(eigen_vals)
print("Explained variance ratio (first 2 PCs):", explained_variance_ratio[:2])

plt.plot(np.cumsum(explained_variance_ratio), marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.grid(True)
plt.show()