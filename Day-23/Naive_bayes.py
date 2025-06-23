# Step 1: Import Libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 2: Gaussian Naive Bayes Classifier
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(self._log_gaussian_density(c, x))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _log_gaussian_density(self, class_label, x):
        mean = self.mean[class_label]
        var = self.var[class_label]
        eps = 1e-9  # to avoid log(0)
        return -0.5 * np.log(2 * np.pi * var + eps) - ((x - mean) ** 2) / (2 * var + eps)


# Step 3: Load Dataset and Select First 2 Features
iris = load_iris()
X = iris.data[:, :2]   #  Only use first 2 features (e.g., sepal length & sepal width)
y = iris.target

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: (Important) Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the Naive Bayes model
model = NaiveBayes()
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")



def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names.tolist())
    plt.grid(True)
    plt.show()
plot_decision_boundary(model, X_train, y_train, title="Naive Bayes on Iris (2 Features)")
