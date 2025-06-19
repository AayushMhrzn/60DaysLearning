import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ---- Node class for tree ----
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

# ---- Gini impurity ----
def gini_impurity(y):
    counts = Counter(y)
    impurity = 1 - sum((count / len(y)) ** 2 for count in counts.values())
    return impurity

# ---- Split the dataset ----
def split(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

# ---- Best split based on Gini ----
def best_split(X, y, max_features=None):
    best_gain = 0
    best_feat, best_thresh = None, None
    current_impurity = gini_impurity(y)
    n_features = X.shape[1]

    features = np.random.choice(n_features, max_features or n_features, replace=False)

    for feature in features:
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

# ---- Build a decision tree ----
def build_tree(X, y, depth=0, max_depth=3, max_features=None):
    if len(set(y)) == 1 or depth == max_depth:
        return Node(value=Counter(y).most_common(1)[0][0])

    feature, threshold = best_split(X, y, max_features)
    if feature is None:
        return Node(value=Counter(y).most_common(1)[0][0])

    X_left, X_right, y_left, y_right = split(X, y, feature, threshold)
    left = build_tree(X_left, y_left, depth + 1, max_depth, max_features)
    right = build_tree(X_right, y_right, depth + 1, max_depth, max_features)
    return Node(feature=feature, threshold=threshold, left=left, right=right)

# ---- Predict single and batch ----
def predict_tree(x, tree):
    if tree.is_leaf():
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict_tree(x, tree.left)
    else:
        return predict_tree(x, tree.right)

def predict_batch(X, tree):
    return np.array([predict_tree(x, tree) for x in X])

# ---- Random Forest Class ----
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=3, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X, y):
        m = X.shape[0]
        indices = np.random.choice(m, m, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = build_tree(X_sample, y_sample, max_depth=self.max_depth, max_features=self.max_features)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([predict_batch(X, tree) for tree in self.trees])
        final_preds = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final_preds)

# ---- Run on Iris dataset ----
if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[:, :2]  # Use 2 features for simplicity
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForest(n_estimators=10, max_depth=3, max_features=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Random Forest Accuracy: {acc * 100:.2f}%")
