import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
n_classes = len(np.unique(y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Binarize test labels (always for all classes)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Train original Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_score_rf = rf.predict_proba(X_test)

# Train calibrated model
calibrated_rf = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                                       method='sigmoid', cv=3)
calibrated_rf.fit(X_train, y_train)
y_score_cal = calibrated_rf.predict_proba(X_test)

# Prepare plots
plt.figure(figsize=(10, 7))

for i in range(n_classes):
    if np.sum(y_test_bin[:, i]) == 0:
        print(f" Class {i} not present in y_test, skipping ROC for this class.")
        continue

    fpr_rf, tpr_rf, _ = roc_curve(y_test_bin[:, i], y_score_rf[:, i])
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, '--', label=f'RF - Class {class_names[i]} (AUC = {auc_rf:.2f})')

    fpr_cal, tpr_cal, _ = roc_curve(y_test_bin[:, i], y_score_cal[:, i])
    auc_cal = auc(fpr_cal, tpr_cal)
    plt.plot(fpr_cal, tpr_cal, '-', label=f'Calibrated RF - Class {class_names[i]} (AUC = {auc_cal:.2f})')

# Plot diagonal
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.title('ROC Curves: Random Forest vs Calibrated RF (Multi-Class)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
