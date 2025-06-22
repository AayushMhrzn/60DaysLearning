#  Day 22 — Non-Linear SVM with Kernel Trick (RBF Kernel)

 dived deeper into **Support Vector Machines (SVM)**, focusing on the **non-linear case** using the **kernel trick**, specifically the **Radial Basis Function (RBF)** kernel.

---

##  Why Linear SVM Fails?

When data is **not linearly separable** — meaning it can't be separated with a straight line — linear SVMs won't work well. Example: Iris dataset has two classes (`versicolor` and `virginica`) that **overlap** in lower dimensions.

---

## What is a Kernel?

A **kernel** is a function that **implicitly maps** the input data into a **higher-dimensional space**, where it **might become linearly separable**.

Instead of computing this transformation explicitly, we use a **kernel function** that computes the **inner product** in the high-dimensional space **without ever going there**.


## Common Kernel Types

| Kernel Type | Formula | Description |
|-------------|---------|-------------|
| Linear      | `K(x, x') = xᵀ x'` | Default for linear SVM |
| Polynomial  | `K(x, x') = (xᵀ x' + c)^d` | Adds polynomial interaction |
| RBF (Gaussian) | `K(x, x') = exp(-γ ||x - x'||²)` | Most popular. Non-linear. Infinite dimension mapping |
| Sigmoid     | `K(x, x') = tanh(α xᵀ x' + c)` | Related to neural nets |

---

##  RBF (Gaussian) Kernel

The **Radial Basis Function (RBF)** kernel computes similarity between two points based on their distance:
`K(x, x') = exp(-γ * ||x - x'||²)`

- `γ` (gamma): Controls the **radius** of influence for a single training point.
- Smaller γ → broader influence → smoother decision boundaries.
- Larger γ → sharp curves → risk of overfitting.

---

## SVM Objective with Kernel

For soft-margin non-linear SVM, the optimization becomes:
`Minimize: (1/2) ||w||² + C * Σ max(0, 1 - yᵢ * f(xᵢ))`

Where:
- `C` = regularization parameter
- `f(xᵢ)` = prediction using kernel: `Σ αⱼ yⱼ K(xⱼ, xᵢ) + b`

---

##  Why Feature Scaling?

- RBF kernel uses Euclidean distance.
- Unscaled features distort distance measurements.
- StandardScaler standardizes data to zero mean and unit variance.

## SVC Parameters

```python
clf = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr')
```

| Param                           | Meaning                                                                    |
| ------------------------------- | -------------------------------------------------------------------------- |
| `kernel='rbf'`                  | Use radial basis function                                                  |
| `C=1.0`                         | Regularization (smaller C → larger margin, more misclassification allowed) |
| `gamma='scale'`                 | Auto-calculated γ value based on data                                      |
| `decision_function_shape='ovr'` | One-vs-Rest strategy for multi-class                                       |



