# Day 10: Introduction to Convolutional Neural Networks (CNN)

On Day 10 of the Challenge, we explored the fundamentals of **Convolutional Neural Networks (CNNs)**, which are widely used for image classification, object detection, and other vision-related tasks.

---

## What is a CNN?

CNNs are a special type of deep learning model specifically designed to process grid-like data, such as images. Their architecture mimics how humans recognize visual patterns using layers that learn spatial hierarchies.

### Basic CNN Architecture:

1. **Convolutional Layer**:

   * Applies filters (kernels) to the input image to extract features like edges, corners, textures.
   * Each filter slides over the image and produces a feature map.

2. **Activation Layer (ReLU)**:

   * Introduces non-linearity.
   * Converts negative values to zero to speed up training and increase accuracy.

3. **Pooling Layer (Max Pooling)**:

   * Reduces spatial size of feature maps.
   * Helps control overfitting and computation.

4. **Flatten Layer**:

   * Converts 2D feature maps into 1D feature vector before passing into dense layers.

5. **Fully Connected Layer (Dense)**:

   * Classifies the image based on the extracted features.

---

## EXAMPLE-Tiny CNN for Digit Recognition

We implemented a CNN using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

### CNN Model Summary

* **Input Shape**: (28, 28, 1) grayscale images
* **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU
* **MaxPooling2D**: 2x2 pool size
* **Flatten**: 2D to 1D
* **Dense Layer**: 128 neurons, ReLU
* **Output Dense**: 10 classes, Softmax

### Dataset Used:

* **MNIST**: 70,000 grayscale images of handwritten digits (0–9)

### Code Overview:

1. Load and preprocess MNIST data
2. Build CNN architecture
3. Train on training data
4. Evaluate on test data
