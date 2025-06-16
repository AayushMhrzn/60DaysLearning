import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic linear data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
true_w = 3.5
true_b = 1.2
y = true_w * X + true_b + np.random.randn(100, 1) * 0.5

# Visualize the data
plt.scatter(X, y)
plt.title("Linear Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
 