# Day 20: Support Vector Machines (SVM) from Scratch

Understanding and implementing a **Linear Support Vector Machine** from scratch using NumPy.

---

## What is SVM?

SVM is a **supervised machine learning algorithm** that finds the best decision boundary (hyperplane) to separate data points of different classes. Its goal is to **maximize the margin** between the hyperplane and the nearest data points (support vectors).

---

## Why is it called “Support Vector Machine”?

- **Support Vectors**: Data points closest to the decision boundary. These are the most "informative" examples.
- **Machine**: Refers to the model that uses these vectors to learn the boundary.

---

## What is “Linear” SVM?

If the classes can be separated using a **straight line (2D)** or **hyperplane (nD)**, we call it a **Linear SVM**. If not, we use **non-linear kernels** — which will be covered in a later days.

---

## Mathematical Foundation

### Hard Margin SVM (linearly separable)

Find a hyperplane \( w^T x + b = 0 \) such that:
y_i * (w^T * x_i + b) ≥ 1 for all i

Objective (minimize the norm of the weight vector):
(1/2) * ||w||^2


---

### Soft Margin SVM (allowing misclassification)

For non-linearly separable data:

\[
y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \text{where } \xi_i \geq 0
\]

Objective:
(1/2) * ||w||^2 + C * Σ max(0, 1 - y_i * (w^T * x_i + b))

Where:
- \( \xi_i \): slack variable (margin violation)
- \( C \): regularization parameter (higher = less tolerance for errors)
- `max(0, 1 - y_i * (w^T * x_i + b))` is the **hinge loss** (penalizes misclassified or near-boundary points)
---

## Prediction Rule

\[
\hat{y} = \text{sign}(w^T x + b)
\]

---

## SVM vs Logistic Regression vs Perceptron

| Criteria           | SVM                      | Logistic Regression         | Perceptron               |
|--------------------|--------------------------|-----------------------------|--------------------------|
| Loss Function      | Hinge Loss               | Cross-Entropy Loss          | Perceptron Loss          |
| Margin             | Maximizes margin         | No margin maximization      | No margin maximization   |
| Output             | +1 or -1 (sign-based)    | Probability (sigmoid)       | +1 or -1 (sign)          |
| Regularization     | Yes (via λ or C)         | Yes                         | Rarely                   |
| Robust to Outliers | Somewhat (via margin)    | Less robust                 | Not robust               |
---

## LinearSVM Class — Explained
```python
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # regularization strength (1/C)
        self.n_iters = n_iters
        self.w = None
        self.b = None
```
- learning_rate: Controls how much we adjust the weights per update.
- lambda_param: Regularization term that prevents overfitting 
- n_iters: Number of training iterations.
- self.w: Weight vector (learned during training).
- self.b: Bias term (intercept).

```python
def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
```
- fit(X, y): Trains the model using Stochastic Gradient Descent.
- n_samples, n_features: Get number of samples and features.
- self.w: Initialize weights to zeros.
- self.b: Initialize bias to zero.

```python
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b += self.lr * y[idx]
```
- Iterate over the dataset for n_iters times.
- For each sample x_i, calculate the margin condition: yi(w⊤*xi+b)≥1
- If this holds, the point is correctly classified and outside the margin.
- If correctly classified and outside margin: Only update weights using regularization term. This prevents overfitting and ensures weights don't grow too large.
- If misclassified or within the margin: Update weights using gradient of hinge loss.
- Update equation: w=w-n(2*lambda*w-yi*xi)
- Update bias: b=b+n*yi

```python
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
```
- For prediction:Compute raw scores 𝑤𝑥+𝑏
- Return the sign: If positive → class +1,  If negative → class -1