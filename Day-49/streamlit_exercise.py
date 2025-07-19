import streamlit as st
import joblib
from sklearn.datasets import load_iris

iris = load_iris()

st.title("Iris Flower Predictor")

# Inputs
sepal_length = st.number_input('Enter sepal length')
sepal_width = st.number_input('Enter sepal width')
petal_length = st.number_input('Enter petal length')
petal_width = st.number_input('Enter petal width')

flower_btn = st.button('Predict Flower')

# Load saved model
model = joblib.load('Day-49/iris_classifier_knn_model.joblib')

# Predict
sample = [[sepal_length, sepal_width, petal_length, petal_width]]
if flower_btn:
    prediction = model.predict(sample)
    st.success(f'Your flower is: {iris.target_names[prediction[0]]}')
