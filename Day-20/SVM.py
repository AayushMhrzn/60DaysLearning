# Step 1: Import Libraries and Prepare Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Step 2: Load and Preprocess Data
iris = load_iris()
X = iris.data[:, :2]               # use first two features
y = iris.target

# Filter for binary classification (class 0 and 1 only)
binary_mask = y < 2
X = X[binary_mask]
y = y[binary_mask]

# Convert labels from (0, 1) to (-1, 1) for SVM
y = np.where(y == 0, -1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: SVM Class Definition with SGD
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # regularization strength (1/C)
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
                    # only apply regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # misclassified point, update both weights and bias
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b += self.lr * y[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

#Step 4: Train and Evaluate the Model
svm = LinearSVM()
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy :.2f}%")

# Step 5: Visualize Decision Boundary
def plot_decision_boundary(X, y, model):
    def decision_boundary(x):
        return -(model.w[0] * x + model.b) / model.w[1]

    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1 = decision_boundary(x0)
    plt.plot(x0, x1, "k--", label="Decision Boundary")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title("Linear SVM from Scratch")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_train, y_train, svm)
