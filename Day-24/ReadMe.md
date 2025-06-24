# Day 24: K-Nearest Neighbors (KNN) from Scratch using Python
Today’s focus is on understanding the K-Nearest Neighbors (KNN) algorithm, implementing it from scratch, applying it to the Iris dataset, and visualizing the results for 2D features.

## What is KNN?
**K-Nearest Neighbors** is a **supervised classification algorithm** that classifies a new point based on the **majority class of its k closest points** in the training set.

It's a **lazy learning** method:
- No model is built during training.
- All computation happens at prediction time.

##  How Does KNN Work?

1. Choose `k` (number of neighbors).
2. For a new data point:
   - Compute distance to all training points (usually Euclidean).
   - Find the `k` nearest neighbors.
   - Assign the **majority class** among those neighbors.

##  Euclidean Distance Formula
For two points x1 and x2:
`distance = sqrt(Σ (x1ᵢ - x2ᵢ)²)`

## Why Is KNN Called Non-Parametric?

Because it **doesn't make assumptions** about the underlying data distribution (unlike Naive Bayes or Logistic Regression).

---

## Choosing `k`

- Small `k`: More flexible but may overfit (noisy).
- Large `k`: Smoother boundary but may underfit.

> A good rule of thumb: Start with odd `k` (like 3 or 5) to avoid ties.

---

## Intuition Behind Decision Boundary

- Each region in the feature space is assigned to the class most common among its `k` nearest neighbors.
- Boundaries are **non-linear and data-driven**.

---

## `_predict(self, x)` Method
This method predicts the class for a **single test sample `x`** using the K-Nearest Neighbors algorithm.

```python
distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
```
- For the given test sample x, we compute its distance to all training samples.
- This uses the Euclidean distance function we defined earlier.
- distances will be a list of length equal to the number of training samples.

```python
k_indices = np.argsort(distances)[:self.k]
```
- We sort the distances in ascending order and pick the indices of the k smallest ones.
- These indices point to the k nearest neighbors in the training set.

```python
k_nearest_labels = [self.y_train[i] for i in k_indices]
```
- We get the labels of the k nearest training points using the indices we just computed.
- These are the classes that will be considered for majority voting.

```python
most_common = Counter(k_nearest_labels).most_common(1)
```
- We use Python's Counter to count the frequency of each label among the k neighbors.
- .most_common(1) returns a list with the most frequent label and its count.

Example:
If `k_nearest_labels = [0, 1, 1, 2, 1]` →
`most_common = [(1, 3)]` → class `1` appears most (3 times).

```python
return most_common[0][0]
```
- We return the label (not the count) of the most common class — this is the final prediction for this sample.

In summary, the _predict() method:

- Calculates distances from test point to training points.
- Selects the top k closest neighbors.
- Performs majority voting to decide the predicted class.
