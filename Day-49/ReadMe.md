# Day 49 - Streamlit Basics & Flower Prediction App

#### What is Streamlit?

Streamlit is an open-source Python library that allows you to build beautiful and interactive web applications quickly—mainly used for machine learning and data science projects.

> No need to learn frontend frameworks like HTML, CSS, or JavaScript. Everything is done using simple Python scripts.

---

### Key Streamlit Features:

* `st.write()`, `st.header()`, `st.text()` for text display
* `st.number_input()`, `st.text_input()`, `st.button()` for input forms
* `st.error()`, `st.success()`, `st.warning()` to show alerts and results
* Easy to integrate with ML models using `joblib`, `pickle`, or `sklearn`

---

###  How to Run a Streamlit App:

First pip install streamlit. Then 

```bash
streamlit run your_file.py
```

Make sure you are in the correct folder or provide a relative path:

```bash
streamlit run Day-49/streamlit_exercise.py
```

---

### Exercise: **Iris Flower Predictor App**

#### 1. Model Creation (Before Using Streamlit)

* We used **scikit-learn** to train a **KNN Classifier** on the Iris dataset
* Saved the trained model using `joblib.dump()`
* File saved as: `iris_classifier_knn_model.joblib`

#### 2. Streamlit App Workflow

```python
import streamlit as st
import joblib
from sklearn.datasets import load_iris

# Load dataset and model
iris = load_iris()
model = joblib.load('../models/iris_classifier_knn_model.joblib')

# Input fields
petal_length = st.number_input('Enter petal length')
petal_width = st.number_input('Enter petal width')
speial_length = st.number_input('Enter sepal length')
sepal_width = st.number_input('Enter sepal width')

# Predict button
if st.button('Predict Flower'):
    sample = [[petal_length, petal_width, speial_length, sepal_width]]
    pred = model.predict(sample)
    st.success(f"Your flower is: {iris.target_names[pred[0]]}")
```
#### 3. Run the file
- use `streamlit run app.py` on terminal.
- open the local host provided where your web app will load up.

--- 

### Learnings:

* Learned to use Streamlit widgets: `number_input`, `button`, `write`, `header`, `success`, etc.
* Integrated a trained ML model to make real-time predictions via a web UI
* Understood layout control using Streamlit and how to chain logic based on button clicks

---
