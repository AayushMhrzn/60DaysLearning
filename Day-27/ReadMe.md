# Day 27: Model Calibration & Multiclass ROC Curves

Understand model calibration and its need.
Plot multiclass ROC curves for both calibrated and uncalibrated classifiers.
Compare performance visually using AUC (Area Under Curve).

## What is Model Calibration?

Model calibration refers to the process of adjusting a classifier’s predicted probabilities to better reflect the true likelihood of outcomes.
A model is well-calibrated if, among all predictions with a predicted probability of 0.8, about 80% of them are actually correct.

## Why do we need calibration?
Many ML models (like Random Forests or SVMs) can predict class probabilities. However:

These probabilities might not represent the true confidence of the prediction.
E.g., a model might say "90% confidence" but is only right 60% of the time.
Calibration aligns predicted probability with actual likelihood.

### How do we calibrate?
We use a wrapper class in sklearn:
```python
CalibratedClassifierCV(estimator=..., method='sigmoid', cv=3)
```
method='sigmoid': Uses Platt scaling (logistic regression).
cv=3: 3-fold cross-validation to avoid overfitting calibration step.

### ROC Curve and AUC (for Multiclass)
The ROC (Receiver Operating Characteristic) curve shows the tradeoff between TPR (recall) and FPR (fallout) for different thresholds.
For binary classification, ROC is straightforward.
For multiclass classification, we plot a separate ROC for each class using one-vs-rest strategy.

## Why separate ROC for each class?
Because:
Each class has its own characteristics (some may be harder to predict).
Helps analyze which class the model handles poorly.
Useful for imbalanced datasets and risk-sensitive applications.

- Used Random Forest classifier (RF)
- Calibrated it using Platt scaling
- Compared original and calibrated models using ROC curves per class

Each ROC curve shows:

How well the model distinguishes each class from the rest
Effect of calibration on prediction confidence

Calibration doesn't always improve AUC — but in real-world tasks where decision thresholds matter, having a well-calibrated model is crucial.