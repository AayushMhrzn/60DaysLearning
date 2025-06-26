# Day 26: ROC Curve & AUC (Receiver Operating Characteristic)

explored **ROC Curves** and **AUC** (Area Under the Curve), powerful evaluation tools for **binary classifiers**, especially when dealing with **imbalanced data** or probabilistic outputs.

---

## What is ROC?

- **ROC Curve (Receiver Operating Characteristic)** is a plot of:
  - **True Positive Rate (TPR)** = Sensitivity = TP / (TP + FN)
  - **False Positive Rate (FPR)** = FP / (FP + TN)
  
- It shows how well a classifier separates positive and negative classes at various threshold levels.

### Why ROC?
- Helps to **visually compare** classifiers.
- Useful when **classes are imbalanced**.
- Shows **trade-off** between sensitivity and specificity.

---

## What is AUC?

- **AUC (Area Under Curve)**: Area under the ROC curve.
  - Ranges from **0 to 1**.
  - **AUC = 1** → Perfect classifier.
  - **AUC = 0.5** → Random guessing.
  - Higher AUC = Better Model!

---
## When to Use ROC & AUC?

- Binary classification with imbalanced datasets.

- When you care about ranking or probability scores.

- To compare different models or hyperparameters visually.

compared all three models on **ROC and AUC** using a real dataset.
## Models Used
- Logistic Regression
- K-Nearest Neighbors (K=5)
- Support Vector Machine (RBF Kernel)

## Key Notes:

predict_proba() gives the probability scores required for ROC.

roc_curve() returns FPR, TPR values at different thresholds.

auc() calculates area under the curve.

All models were trained on scaled features.

Scaling is essential especially for distance-based methods like SVM and KNN.


## Summary
| Concept            | Description                               |
| ------------------ | ----------------------------------------- |
| ROC Curve          | TPR vs FPR at different thresholds        |
| AUC Score          | Area under the ROC curve                  |
| Perfect Classifier | AUC = 1.0                                 |
| Random Classifier  | AUC = 0.5                                 |
| Scaling            | Required for SVM & KNN for better results |
