import numpy as np
from sklearn.datasets import load_iris
from collections import Counter
import matplotlib.pyplot as plt

# Gini Impurity
def gini_impurity(y):
    counts = Counter(y)
    impurity = 1 - sum((count / len(y)) ** 2 for count in counts.values())
    return impurity

# Split Function
def split(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

# Best Split Finder
def best_split(X, y):
    best_gain = 0
    best_feat, best_thresh = None, None
    current_impurity = gini_impurity(y)
    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            p = len(y_left) / len(y)
            gain = current_impurity - (p * gini_impurity(y_left) + (1 - p) * gini_impurity(y_right))

            if gain > best_gain:
                best_gain = gain
                best_feat, best_thresh = feature, threshold
    return best_feat, best_thresh

# Node Class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

# Tree Builder
def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1 or depth == max_depth:
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    feat, thresh = best_split(X, y)
    if feat is None:
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    X_left, X_right, y_left, y_right = split(X, y, feat, thresh)
    left = build_tree(X_left, y_left, depth + 1, max_depth)
    right = build_tree(X_right, y_right, depth + 1, max_depth)
    return Node(feature=feat, threshold=thresh, left=left, right=right)

# Prediction Functions
def predict_tree(x, tree):
    if tree.is_leaf():
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict_tree(x, tree.left)
    else:
        return predict_tree(x, tree.right)

def predict_batch(X, tree):
    return np.array([predict_tree(x, tree) for x in X])

# Load Dataset and Train Tree
iris = load_iris()
X, y = iris.data, iris.target

# Use only 2 features for simplicity
#X = X[:, :2]  # Sepal length & sepal width

# Build decision tree
my_tree = build_tree(X, y, max_depth=3)

# Predict
y_pred = predict_batch(X, my_tree)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy * 100:.2f}%")
