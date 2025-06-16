# Day 16: Binary Classification with Regression (From Scratch)

This lesson covers the theory and implementation of two foundational machine learning algorithms:

* **Linear Regression** – used for predicting continuous values
* **Logistic Regression** – used for binary classification

implemented both models from scratch using NumPy (without relying on scikit-learn or ML libraries) to understand the internal mechanics.

---

## Linear Regression

### Objective:

Fit a straight line that best represents the relationship between input `x` and target `y`.

### Formula:

$y = w \cdot x + b$

### Loss Function (MSE):

$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

### Optimization:

Use **Gradient Descent** to update weights `w` and bias `b` to minimize MSE.

### Gradients:

$\frac{\partial L}{\partial w} = -\frac{2}{n} \sum x_i (y_i - \hat{y}_i)$
$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum (y_i - \hat{y}_i)$

---
#### Step-by-step Breakdown:

1. **Generate Dummy Regression Data**
```python
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```
- Inputs between 0 and 2.
- Targets follow the true line \( y = 4 + 3x \) plus Gaussian noise.

2. **Add Bias Term to Input**
```python
X_b = np.c_[np.ones((100, 1)), X]
```
- Adds column of 1s so bias term \( w_0 \) can be learned as part of weights.

3. **Initialize Parameters**
```python
theta = np.random.randn(2, 1)
```
- Random initialization of weights (one for bias and one for input).

4. **Gradient Descent Loop**
```python
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients
```
- Compute gradients of the MSE loss.
- Update weights by subtracting gradient scaled by learning rate.

5. **Model Prediction**
```python
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)
```
- Predicts outputs for new values \( x = 0 \) and \( x = 2 \) using learned weights.

6. **Visualization**
```python
plt.plot(X_new, y_predict, ...)
plt.scatter(X, y)
```
- Displays original data points and the fitted regression line.

7. **Final Output**
```python
print(f"Learned weights: bias = {theta[0][0]:.2f}, weight = {theta[1][0]:.2f}")
```
- Shows the values of learned parameters.

---

## Logistic Regression

### Objective:

Predict binary class labels (0 or 1). The model outputs probabilities which are squashed using the **Sigmoid** function.

### Formula:

$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where } z = w \cdot x + b$

### Loss Function (Binary Cross-Entropy):

$L = - \frac{1}{n} \sum \left( y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right)$

### Optimization:

Use **Gradient Descent** to minimize the cross-entropy loss.

---
#### Step-by-step Breakdown:

1. **Sigmoid Function**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
- The sigmoid squashes any real number to a probability between 0 and 1.
- Used as the activation function in logistic regression.

2. **Generate Dummy Binary Classification Data**
```python
X = 2 * np.random.rand(100, 1)
y = (X > 1).astype(int)
```
- Random 1D inputs between 0 and 2.
- Class is 0 if \( x \leq 1 \), otherwise 1 — a simple linearly separable setup.

3. **Normalization**
```python
X = (X - X_mean) / X_std
```
- Feature scaling ensures gradient descent converges faster and more reliably.

4. **Add Bias Term**
```python
X_b = np.c_[np.ones((X.shape[0], 1)), X]
```
- Adds a column of ones to handle the bias \( w_0 \) in the weight vector.

5. **Initialize Weights**
```python
theta = np.random.randn(2, 1)
```
- Random weights for bias and feature.

6. **Gradient Descent Loop**
```python
for i in range(n_iterations):
    z = X_b.dot(theta)
    y_pred = sigmoid(z)
    gradients = X_b.T.dot(y_pred - y) / X.shape[0]
    theta -= learning_rate * gradients
```
- Perform 1000 steps of gradient descent:
  - Compute prediction \( \hat{y} \)
  - Calculate gradients of the loss w.r.t. weights
  - Update weights using learning rate

7. **Prediction Function**
```python
def predict(X):
    ...
    return sigmoid(...) >= 0.5
```
- Predicts class (0 or 1) for new input after normalization and sigmoid thresholding.

8. **Visualization**
```python
plt.scatter(X, y)
plt.plot(...)
```
- Plots input data points and the sigmoid-based decision boundary.

9. **Final Output**
```python
print(f"Learned weights: bias = {theta[0][0]:.2f}, weight = {theta[1][0]:.2f}")
```
- Displays learned model parameters.
