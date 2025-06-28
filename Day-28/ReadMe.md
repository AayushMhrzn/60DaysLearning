# K-Means Clustering from Scratch
to implement K-Means Clustering from scratch using the Iris dataset (without using labels). It includes both the core algorithm and visualizations like the Elbow Method and 2D PCA projection.

## What is Clustering?
Clustering is an unsupervised learning technique where the goal is to group similar data points together based on their features, without access to ground-truth labels.

Each group is called a cluster, and members of the same cluster are more similar to each other than to members of other clusters.


## What is K-Means?
K-Means is one of the most popular clustering algorithms. The objective is to:

- Partition the dataset into K clusters
- Each cluster has a centroid (mean of its points)
- Assign each data point to the cluster with the nearest centroid
- Repeat until convergence (centroids don’t move)

K-Means is unsupervised because:
It does not require labels or prior knowledge of class boundaries.
It works purely by trying to minimize intra-cluster variance.

## Mathematics Behind K-Means
### Objective Function
The goal is to minimize:

WCSS (Within-Cluster Sum of Squares):

WCSS=∑𝑖=1𝑘∑𝑥∈𝐶𝑖∥𝑥−𝜇𝑖∥2


### Algorithm Steps
- Initialization: Select k random points as initial centroids.

- Assignment Step: Assign each point to the nearest centroid using Euclidean distance.

- Update Step: For each cluster, compute the new centroid as the mean of all points assigned to that cluster.

- Repeat: Until centroids do not change (or max iterations reached).

### When K ≠ Number of Classes
In real datasets:
The true number of natural classes is unknown.
We choose K using methods like the Elbow Method or Silhouette Score.

If K > true number of classes, the algorithm will:
- Split natural clusters into smaller parts
- Over-cluster the data
- Still minimize intra-cluster variance

## Code Walkthrough
Below is a breakdown of the full implementation and what each section does.

- Euclidean Distance Function
```python
def euclidean(a, b):
    return np.linalg.norm(a - b)
```
Calculates the straight-line (L2) distance between two vectors.

- Initialize Centroids
```python
def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]
```
Randomly selects k distinct points from the dataset to serve as initial centroids.

- Assign Clusters
```python
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean(x, c) for c in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)
```
For each data point, compute the distance to every centroid, and assign it to the closest one.

- Update Centroids
```python
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        points = X[clusters == i]
        new_centroids.append(np.mean(points, axis=0) if len(points) > 0 else np.zeros(X.shape[1]))
    return np.array(new_centroids)
```
For each cluster, compute the new centroid as the mean of all assigned points.

- Compute WCSS (Cost Function)
```python
def compute_wcss(X, clusters, centroids):
    wcss = 0
    for i in range(len(centroids)):
        points = X[clusters == i]
        wcss += np.sum((points - centroids[i]) ** 2)
    return wcss
```
Calculates the total intra-cluster variance to evaluate clustering performance.

- Main K-Means Function
```python
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
```
The main loop that repeatedly assigns points and updates centroids until convergence or maximum iterations.

