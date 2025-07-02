import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Load and normalize data
digits = load_digits()
X = digits.data
X = MinMaxScaler().fit_transform(X)

# Split data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Hyperparameters
input_dim = X.shape[1]       # 64 for digits dataset
hidden_dim = 32              # increased from 16
learning_rate = 0.05
epochs = 1000

# Initialize weights and biases
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, input_dim) * 0.01
b2 = np.zeros((1, input_dim))

# Training loop
losses = []
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Loss (Mean Squared Error)
    loss = np.mean((A2 - X_train) ** 2)
    losses.append(loss)

    # Backpropagation
    dA2 = 2 * (A2 - X_train) / X_train.shape[0]
    dZ2 = dA2 * sigmoid_deriv(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Gradient descent update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss = {loss:.5f}")

# Inference
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

# Plot original vs reconstructed images
n = 10
plt.figure(figsize=(18, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.axis('off')
    
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(A2_test[i].reshape(8, 8), cmap='gray')
    plt.axis('off')

plt.suptitle("Top: Original | Bottom: Reconstructed", fontsize=14)
plt.show()
