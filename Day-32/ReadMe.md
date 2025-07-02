#  Day 32: Autoencoders from Scratch (Unsupervised Deep Learning)

learned and implemented a **vanilla Autoencoder** from scratch using only **NumPy**, trained on the **MNIST** dataset. explored:

##  What is an Autoencoder?

An **Autoencoder** is a type of **neural network** used to learn **compressed representations** of input data. It tries to **reconstruct the input** from a lower-dimensional space.

It is composed of two parts:

| Part     | Role                                               |
|----------|----------------------------------------------------|
| Encoder  | Compresses the input to a lower-dimensional code   |
| Decoder  | Reconstructs the input from the code               |

---

##  Applications of Autoencoders

Autoencoders are used for:

| Application              | Description                             |
| ------------------------ | --------------------------------------- |
| Dimensionality Reduction | Replace PCA with a non-linear version   |
| Denoising Autoencoders   | Reconstruct clean data from noisy input |
| Anomaly Detection        | Large reconstruction error = anomaly    |
| Image Compression        | Compress and decompress images          |
| Latent Space Clustering  | Cluster latent features (e.g., digits)  |


---

##  Architecture Overview

### Input:
MNIST image: `28x28` pixels → flattened to `784` vector

### Encoder:
```python
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)
```
- Compress input from 784 → 32 dimensions

### Decoder:
```python
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)
```
- Reconstruct from 32 → 784 dimensions

## Math and Theory

1. Forward Propagation
- Encoder: 
Hidden (compressed) representation:
Z1 = X · W1 + b1
A1 = ReLU(Z1)

- Decoder:
Reconstructed output:
Z2 = A1 · W2 + b2
A2 = Sigmoid(Z2)

2. Loss Function (Binary Cross Entropy)
Used because MNIST images are binary-normalized [0,1]:
`Loss = -1/n ∑ [X * log(X') + (1 - X) * log(1 - X')]`

In code:
```python
loss = -np.mean(X * np.log(A2 + 1e-8) + (1 - X) * np.log(1 - A2 + 1e-8))
```

3. Backpropagation
We compute gradients for:

dW2, db2 for decoder
dW1, db1 for encoder

---
 ## Example: MNIST Autoencoding
Input:
1000 MNIST images (flattened to shape [1000, 784])

Encoder Output:
Compressed to shape [1000, 32]

Decoder Output:
Reconstructed images [1000, 784]

We reshape to 28x28 and visualize:

First row: Original images
Second row: Reconstructed images

Result:
After a few epochs, outputs start looking like original digits.

##  Key Concepts
🔹 Latent Vector:
Output of encoder (A1) is the compressed representation.
Like a feature vector — can be visualized or clustered.

🔹 Unsupervised:
No labels required.
Model learns from the structure of input data itself.