# linear_regression.py
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate dummy data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Normalize the data (optional but good practice)
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std

# 3. Add a bias term (column of ones)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape: (100, 2)

# 4. Initialize parameters
theta = np.random.randn(2, 1)  # [bias, weight]

# 5. Hyperparameters
learning_rate = 0.1
n_iterations = 1000

# 6. Gradient Descent Loop
for iteration in range(n_iterations):
    gradients = 2 / X_b.shape[0] * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

# 7. Predictions
predictions = X_b.dot(theta)

# 8. Plot results
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data")
plt.plot(X, predictions, color="red", label="Best fit line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

# 9. Print parameters
print(f"Learned parameters: bias = {theta[0][0]:.2f}, weight = {theta[1][0]:.2f}")
