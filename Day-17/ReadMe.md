# Day 17: Multiclass Classification :Softmax Regression from Scratch

### What is Multiclass Classification?
Multiclass classification is a type of classification task where the goal is to classify input data into **one of three or more classes**. Each input sample belongs to only one class out of the possible multiple classes.

**Example:**
- Classifying animals into **Cat**, **Dog**, or **Elephant**.
- Classifying handwritten digits from 0 to 9.

### Common Approaches

1. **One-vs-Rest (OvR)**:
   - Build a separate binary classifier for each class.
   - For class `i`, the model learns to distinguish class `i` vs all others.
   - During prediction, all models are evaluated and the class with the highest score is chosen.

2. **Softmax Regression (Multinomial Logistic Regression):**
   - Generalizes binary logistic regression to handle multiple classes.
   - Uses the **softmax function** to convert raw scores (logits) into class probabilities.
   - The predicted class is the one with the highest probability.

### Softmax Function:
Given logits `z` for each class, the softmax function computes:

```
softmax(z_i) = exp(z_i) / sum(exp(z_j))   for all j
```
- Output is a probability distribution over all `K` classes.
- Ensures all probabilities sum to 1.

### Loss Function: Cross Entropy
The cross-entropy loss for multiclass classification is:
```
Loss = - sum_over_i ( sum_over_k ( y_ik * log(yhat_ik) ) )
```
Where:
- `y_ik = 1` if sample `i` belongs to class `k`, else 0  
- `yhat_ik` is the predicted probability of class `k` for sample `i`

---

## Code: Softmax Regression (Multiclass Classifier from Scratch)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
```
### Step 1: Prepare Data (Iris Dataset - Iris Flower classification with 3 classes from 4 features)
```python
# 1. Load Iris dataset
iris = datasets.load_iris()
X = iris.data 
y = iris.target

# One-hot encode labels
ohe = OneHotEncoder(sparse=False)
y_encoded = ohe.fit_transform(y.reshape(-1, 1))
```
- We load dataset with 4 informative features and 3 classes.
- One-hot encoding is used to convert class labels like `1` or `2` into vectors like `[0 1 0]` or `[0 0 1]`.

### Step 2: Normalize Data
```python
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
```
- Features are normalized (mean = 0, std = 1) for stable and faster convergence.

### Step 3: Add Bias Term
```python
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape: (m, n+1)
```
- A bias column (intercept term) is added to the input matrix.

### Step 4: Initialize Parameters
```python
n_classes = y_encoded.shape[1]
n_features = X_b.shape[1]
np.random.seed(42)
theta = np.random.randn(n_features, n_classes)  # shape: (n+1, K)
```
- `theta` holds weights for each class. Shape = (features + 1, number of classes).

### Step 5: Define Softmax Function
```python
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```
- Converts raw logits into probability distributions.
- Subtracting max for numerical stability.

### Step 6: Train using Gradient Descent
```python
learning_rate = 0.1
n_iterations = 1000
m = X_b.shape[0]

for i in range(n_iterations):
    logits = X_b.dot(theta)              # shape: (m, K)
    y_pred = softmax(logits)             # shape: (m, K)
    gradients = (1/m) * X_b.T.dot(y_pred - y_encoded)
    theta -= learning_rate * gradients
```
- Compute predictions using softmax.
- Calculate gradients of the cross-entropy loss.
- Update weights using gradient descent.

### Step 7: Define Predict Function
```python
def predict(X_new):
    X_new = (X_new - X_mean) / X_std
    X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
    logits = X_new_b.dot(theta)
    probs = softmax(logits)
    return np.argmax(probs, axis=1)
```
- Normalizes new input.
- Applies learned weights.
- Returns class with highest probability.

### Step 8: Evaluate Model
```python
preds = predict(X)
accuracy = np.mean(preds == y)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
- Compares predicted and true labels.
- Computes classification accuracy.

---

## Visualize Decision Boundaries

- Creates a 2D plot to visualize how the model divides space into different class regions.
- Helps validate model’s decision surface.

---

