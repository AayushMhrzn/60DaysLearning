# Day 33: Denoising Autoencoders from Scratch using NumPy

 ## What is a Denoising Autoencoder?
A Denoising Autoencoder (DAE) is a type of autoencoder that learns to reconstruct clean input data from corrupted (noisy) versions. Unlike traditional autoencoders, DAEs are trained with input data that has been intentionally corrupted, with the goal of learning robust representations.

It’s useful in:

- Noise removal
- Robust feature learning
- Data pre-processing and compression

## Key Concepts
An autoencoder has two main parts:

Encoder: Compresses input data x into a lower-dimensional latent representation z.
Decoder: Reconstructs the input x' from the latent z.

## Why Denoising Autoencoders?
DAEs encourage the model to learn more robust and meaningful latent representations. This is useful in real-world tasks like:

- Image noise reduction
- Pre-training for deep models
- Anomaly detection
- Compressed sensing

## Denoising Objective
We train the model to minimize the reconstruction error:
`Loss = ||x - x'||²`
However, input to the encoder is a corrupted version x_noisy:

`x_noisy → Encoder → z → Decoder → x'`
The model still tries to output the clean x, not the noisy x_noisy.

 ## Math Behind It
- Add Noise to Input
We add Gaussian noise:

`x_noisy = x + N(0, σ²)`

- Forward Propagation
Let:
W1, b1 be weights and biases of encoder.
W2, b2 be weights and biases of decoder.

- Encoder:
`z = sigmoid(x_noisy @ W1 + b1)`

- ecoder:
`x_hat = sigmoid(z @ W2 + b2)`

- Loss Function
Mean Squared Error:
`L = (1/N) * Σ ||x - x_hat||²`

- Backpropagation
We update weights using gradients of loss L w.r.t. each weight. The gradients are derived via the chain rule through both encoder and decoder layers.