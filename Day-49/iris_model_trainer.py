# iris_model_trainer.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'Day-49/iris_classifier_knn_model.joblib')  

print("Model saved successfully!")