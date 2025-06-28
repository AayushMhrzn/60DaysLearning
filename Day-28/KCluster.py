import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1. Load and Prepare Data

iris = load_iris()
X = iris.data  # (150, 4)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 2. Helper Functions

def euclidean(a, b):
    return np.linalg.norm(a - b)

def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean(x, c) for c in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        points = X[clusters == i]
        if len(points) == 0:
            new_centroids.append(np.zeros(X.shape[1]))
        else:
            new_centroids.append(np.mean(points, axis=0))
    return np.array(new_centroids)

def compute_wcss(X, clusters, centroids):
    wcss = 0
    for i in range(len(centroids)):
        points = X[clusters == i]
        wcss += np.sum((points - centroids[i]) ** 2)
    return wcss


# 3. Main K-Means Function

def k_means(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    wcss = compute_wcss(X, clusters, centroids)
    return clusters, centroids, wcss


# 4. Elbow Method

wcss_list = []
K_range = range(1, 10)

for k in K_range:
    _, _, wcss = k_means(X_scaled, k)
    wcss_list.append(wcss)

plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss_list, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()


# 5. Final Clustering & Visualization

optimal_k = 3
clusters, centroids, _ = k_means(X_scaled, optimal_k)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set1", optimal_k)
for i in range(optimal_k):
    plt.scatter(
        X_pca[clusters == i, 0],
        X_pca[clusters == i, 1],
        label=f'Cluster {i}',
        s=50,
        alpha=0.7,
        color=palette[i]
    )

# Plot centroids
centroids_2d = pca.transform(centroids)
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    s=200,
    marker='X',
    color='black',
    label='Centroids'
)

plt.title("K-Means Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()
