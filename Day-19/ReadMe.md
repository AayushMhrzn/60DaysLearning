# Day 19: Random Forest Classifier from Scratch

Implemented a Random Forest Classifier from scratch using NumPy focusing on ensemble learning to improve prediction accuracy.

---

## Decision Tree (DT)
A Decision Tree splits data at nodes based on the feature that gives the **best impurity reduction** (e.g., Gini).

> Already explained in Day 18:
- `Node` class
- Gini impurity calculation
- Best split finder
- Recursive tree building

### Limitations of a Single Decision Tree
- **Overfitting**: Trees tend to memorize noise from the training set.
- **Instability**: A small change in data can lead to a completely different tree.
- **High Variance**: Poor generalization on unseen data.

---

## Random Forest: The Fix
Random Forest is an **ensemble method** that combines predictions from multiple decision trees. It reduces overfitting and increases accuracy.

### Key Ideas:
1. **Bootstrap Aggregation (Bagging)**:
   - Train each tree on a random subset (with replacement) of the data.
2. **Feature Randomness**:
   - At each split in a tree, use only a random subset of features.

### Math Behind Random Forest
Suppose we train \( T \) decision trees \( f_1, f_2, ..., f_T \). The final prediction is made by:

- **Classification (majority vote)**:
  - **Classification (majority vote)**:  
  `ŷ = mode(f₁(x), f₂(x), ..., fₜ(x))`

- **Regression (average prediction)**:  
  `ŷ = (1/T) * Σₜ₌₁ᵀ fₜ(x)`

Randomness in both data and feature selection reduces **variance**, improving generalization.

---

## Code Explanation

> Parts already explained in Day 18: `Node`, `gini_impurity`, `best_split`, `build_tree`, `predict_tree`, etc.

Below is a breakdown of the **Random Forest logic**:

### `bootstrap_sample(X, y)`
```python
indices = np.random.choice(len(X), size=len(X), replace=True)
```
- Selects a random sample of the same size as the dataset with replacement.
- Enables diversity across trees.

### RandomForest class
```python
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
```
- n_estimators: Number of trees in the forest
- max_depth: Maximum depth of each tree

### fit(X, y)
```python
for _ in range(self.n_estimators):
    X_sample, y_sample = bootstrap_sample(X, y)
    tree = build_tree(X_sample, y_sample, max_depth=self.max_depth)
    self.trees.append(tree)
```
- Each tree is trained on a bootstrapped dataset.

### predict(X)
```python
tree_preds = np.array([predict_batch(X, tree) for tree in self.trees])
```
- Predicts on all trees.
```python
return [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
```
- Takes majority vote across all trees for each sample.
