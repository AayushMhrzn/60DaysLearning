import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris data
iris = load_iris()
X = iris.data[:,:2]
y = iris.target
feature_names = iris.feature_names
class_labels = iris.target_names

# Create a range for plotting PDFs
def plot_gaussian(x_range, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(- ((x_range - mean) ** 2) / (2 * std ** 2))

# Plot for each feature
plt.figure(figsize=(16, 10))
for i in range(X.shape[1]):
    plt.subplot(2, 1, i+1)
    x_min, x_max = X[:, i].min() - 0.5, X[:, i].max() + 0.5
    x_range = np.linspace(x_min, x_max, 1000)

    for cls in np.unique(y):
        X_cls = X[y == cls, i]
        mu = X_cls.mean()
        sigma = X_cls.std()

        pdf = plot_gaussian(x_range, mu, sigma)
        plt.plot(x_range, pdf, label=f"{class_labels[cls]} (μ={mu:.2f}, σ={sigma:.2f})")

    plt.title(f"Gaussian Distribution: {feature_names[i]}")
    plt.xlabel(feature_names[i])
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)

plt.suptitle("Gaussian Distributions of Iris Features per Class", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
