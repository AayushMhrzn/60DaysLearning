# Day 25: Model Validation & Evaluation – Visualization & Explanation

focus on **evaluating classifier performance** using multiple validation tools and **visualizing** them.

---

## Dataset Used: Iris Dataset (3-class classification)

We’ll use `scikit-learn` utilities to:
- Split the dataset
- Train a classifier (KNN)
- Visualize and explain:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
  - Cross Validation Score

---

## Why Model Validation is Important?

Validation helps answer:
- **How good is my model?**
- **Where is it making mistakes?**
- **Is it biased towards a class?**
- **How generalizable is it to unseen data?**

---

## Evaluation Metrics Explained
### Accuracy

- Percentage of correctly classified samples.
- Easy to understand, but misleading with imbalanced data.

### Confusion Matrix

- A grid showing true vs predicted classes.
- Helps identify:
Misclassifications
Class-specific errors

|        | Pred 0 | Pred 1 | Pred 2 |
| ------ | ------ | ------ | ------ |
| True 0 | 10     | 0      | 0      |
| True 1 | 1      | 8      | 1      |
| True 2 | 0      | 2      | 9      |

###  Classification Report
Includes:

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = Harmonic mean of precision and recall

### Cross-Validation

-Splits dataset into k parts and evaluates on each.
-Gives more stable and unbiased performance measure.


| Metric                | Purpose                           |
| --------------------- | --------------------------------- |
| Accuracy              | Overall correct predictions       |
| Confusion Matrix      | Class-level prediction breakdown  |
| Classification Report | Precision, Recall, F1 per class   |
| Cross Validation      | More stable evaluation over folds |

