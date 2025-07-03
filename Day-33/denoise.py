import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load and normalize the digits dataset
digits = load_digits()
X = digits.data / 16.0  # Normalize pixel values to [0,1]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Add Gaussian noise
def add_noise(X, noise_factor=0.5):
    noise = np.random.normal(0, noise_factor, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0., 1.)

X_train_noisy = add_noise(X_train)
X_test_noisy = add_noise(X_test)

# Sigmoid activation and derivative
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Mean Squared Error loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Parameters
input_dim = 64
hidden_dim = 32
learning_rate = 0.1
epochs = 100
batch_size = 64

# Initialize weights and biases
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, input_dim) * 0.01
b2 = np.zeros((1, input_dim))

# Training
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(X_train_noisy), batch_size):
        x_batch = X_train_noisy[i:i+batch_size]
        y_batch = X_train[i:i+batch_size]

        # Forward pass
        z1 = np.dot(x_batch, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        output = sigmoid(z2)

        # Loss
        loss = mse_loss(y_batch, output)
        epoch_loss += loss

        # Backpropagation
        d_loss = 2 * (output - y_batch) / y_batch.shape[0]
        d_output = d_loss * sigmoid_deriv(z2)
        dW2 = np.dot(a1.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, W2.T) * sigmoid_deriv(z1)
        dW1 = np.dot(x_batch.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

# Test set reconstruction
def reconstruct(X_input):
    a1 = sigmoid(np.dot(X_input, W1) + b1)
    out = sigmoid(np.dot(a1, W2) + b2)
    return out

reconstructed = reconstruct(X_test_noisy)

# Plotting original, noisy, and reconstructed images
n = 10
plt.figure(figsize=(18, 6))
for i in range(n):
    # Original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_test_noisy[i].reshape(8, 8), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(reconstructed[i].reshape(8, 8), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
