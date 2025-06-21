#Step 1: Imports and Load Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#Step 2: Reuse Binary Linear SVM
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b += self.lr * y[idx]

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

#Step 3: One-vs-Rest SVM Class
class OvRSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            clf = LinearSVM(self.lr, self.lambda_param, self.n_iters)
            clf.fit(X, y_binary)
            self.classifiers[cls] = clf

    def predict(self, X):
        scores = []
        for cls in self.classes:
            clf = self.classifiers[cls]
            score = clf.decision_function(X)
            scores.append(score)
        scores = np.array(scores)
        return self.classes[np.argmax(scores, axis=0)]

#Step 4: Train and Evaluate
# Load data
iris = load_iris()
X = iris.data[:, :2]  # Use only 2 features for visualization
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train OvR SVM
model = OvRSVM()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

#visualize
def plot_decision_boundary(X, y, model, title="Multiclass SVM (OvR) Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.rainbow, edgecolor='k')
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)

    handles, _ = scatter.legend_elements()
    unique_labels = np.unique(y)
    label_names = [iris.target_names[i] for i in unique_labels]
    plt.legend(handles=handles, labels=label_names)

    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_train, y_train, model)
