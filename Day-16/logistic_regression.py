# logistic_regression.py
import numpy as np
import matplotlib.pyplot as plt

# 1. Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Generate dummy binary classification data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = (X > 1).astype(int)  # class 0 if X <= 1, class 1 otherwise

# Normalize features
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std

# Add bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape (100, 2)

# 3. Initialize weights
theta = np.random.randn(2, 1)

# 4. Hyperparameters
learning_rate = 0.1
n_iterations = 1000

# 5. Gradient Descent
for i in range(n_iterations):
    z = X_b.dot(theta)
    y_pred = sigmoid(z)
    gradients = X_b.T.dot(y_pred - y) / X.shape[0]
    theta -= learning_rate * gradients

# 6. Predict & Visualize
def predict(X):
    X_scaled = (X - X_mean) / X_std
    X_input = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    return sigmoid(X_input.dot(theta)) >= 0.5

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data")
x_plot = np.linspace(-2, 2, 100).reshape(-1, 1)
y_plot = sigmoid(np.c_[np.ones((100, 1)), x_plot].dot(theta))
plt.plot(x_plot, y_plot, color="red", label="Decision Boundary")
plt.title("Logistic Regression")
plt.xlabel("Feature X")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()
plt.show()

# 7. Print parameters
print(f"Learned weights: bias = {theta[0][0]:.2f}, weight = {theta[1][0]:.2f}")
