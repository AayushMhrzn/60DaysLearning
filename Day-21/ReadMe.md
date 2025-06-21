# Day 21: Multi-Class SVM  (One-vs-Rest)

implemented a **multi-class Support Vector Machine (SVM)** using the **One-vs-Rest (OvR)** strategy. This README explains the math, logic, and Python implementation line by line.

---

### What is One-vs-Rest (OvR) SVM?

SVM is inherently a **binary classifier**. For **multi-class classification**, we extend it using the **OvR strategy**:

- For `K` classes, we train `K` binary classifiers.
- Each classifier predicts **"class i vs all other classes"**.
- During prediction, we choose the classifier that has the **highest confidence score** (or output value).

---

### Mathematical Formulation

Given input \( (x_i, y_i) \), with \( y_i \in \{-1, 1\} \), the **primal soft-margin SVM** optimization problem:
`min_w,b (1/2)||w||^2 + C * sum_i max(0, 1 - y_i (w^T x_i + b))`

- The term `max(0, 1 - y_i(w^T x_i + b))` is the **hinge loss**.
- `C = 1/λ` is the regularization parameter.

---

### Code Walkthrough

- checkout day20 for explanation upto LinearSVM class 

### One-vs-Rest SVM Class Explained

We define a wrapper class `OvRSVM` that implements the **One-vs-Rest** strategy for **multi-class classification** using multiple binary SVM classifiers.

```python
class OvRSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate                      # Learning rate for gradient descent
        self.lambda_param = lambda_param             # Regularization strength (1/C)
        self.n_iters = n_iters                       # Number of training iterations
        self.classifiers = {}                        # Dictionary to store one classifier per class
```
#### fit() Method — Training Multiple Binary Classifiers
```python
    def fit(self, X, y):
        self.classes = np.unique(y)                  # Get all unique class labels

        for cls in self.classes:                     # Loop over each class (e.g., 0, 1, 2)
            y_binary = np.where(y == cls, 1, -1)     # Create binary labels: 1 for current class, -1 for others

            clf = LinearSVM(self.lr, self.lambda_param, self.n_iters)  # Create a new binary SVM
            clf.fit(X, y_binary)                      # Train the SVM for class 'cls'
            self.classifiers[cls] = clf               # Store the trained model in dictionary
```
- This effectively creates one SVM per class — each trained to distinguish "class X vs all others".

#### predict() Method — Pick Class with Highest Score

```python
    def predict(self, X):
        scores = []                                   # Store decision scores from each classifier

        for cls in self.classes:
            clf = self.classifiers[cls]               # Get trained classifier for class 'cls'
            score = clf.decision_function(X)          # Compute decision function value (w·x + b)
            scores.append(score)

        scores = np.array(scores)                     # Shape: (n_classes, n_samples)

        return self.classes[np.argmax(scores, axis=0)]  # Return class with highest score for each sample
```
- For each input sample, this method gets the raw decision scores from each classifier.
- The class with the maximum score is selected as the prediction.

### Summary: Linear vs Multi-class SVM
| Type        | Description                           |
| ----------- | ------------------------------------- |
| Binary SVM  | Separates two classes with a margin   |
| Multi-class | Uses OvR strategy (1 model per class) |
| Accuracy    | Depends on data separability          |

