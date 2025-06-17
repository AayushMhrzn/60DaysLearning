import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. Load Iris dataset and select only 2 features for visualization
iris = datasets.load_iris()
X = iris.data[:, :2]  # Only use sepal length & width
y = iris.target

# 2. One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# 3. Train-test split
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
y_test = np.argmax(y_test_encoded, axis=1)

# 4. Normalize features
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
#X_test = (X_test - X_mean) / X_std

# 5. Add bias term
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# 6. Initialize parameters
n_classes = y_train_encoded.shape[1]
n_features = X_train_b.shape[1]
np.random.seed(42)
theta = np.random.randn(n_features, n_classes)

# 7. Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 8. Train the model
learning_rate = 0.1
n_iterations = 1000
m = X_train.shape[0]

for i in range(n_iterations):
    logits = X_train_b.dot(theta)
    y_pred = softmax(logits)
    gradients = (1/m) * X_train_b.T.dot(y_pred - y_train_encoded)
    theta -= learning_rate * gradients

# 9. Prediction function
def predict(X):
    X_scaled = (X - X_mean) / X_std
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    logits = X_b.dot(theta)
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

# 10. Visualize decision boundaries
def plot_decision_boundary(X, y, model_predict):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.rainbow)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Decision Boundaries (Softmax Regression)')
    plt.show()

# Plot
plot_decision_boundary(X, y, predict)

# Accuracy Check
y_test_pred = predict(X_test)
accuracy = np.mean(y_test_pred == y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
