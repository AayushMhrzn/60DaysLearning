# 📘 Day 3: NumPy Basics & Array Manipulations

## 🔹 1. NumPy Array vs Python List

- **Performance**: NumPy arrays are faster due to internal optimizations and vectorized operations.
- **Memory Efficiency**: NumPy consumes less memory because of fixed data types and contiguous memory allocation.
- **Functionality**: NumPy supports complex operations like broadcasting, slicing, reshaping, and mathematical functions that are cumbersome with Python lists.

---

## 🔹 2. Array Creation & Data Types

### ➤ Common Array Creation Functions:
- `zeros`, `ones`, `full`: Create arrays initialized with fixed values.
- `arange`, `linspace`: Generate sequences with specified intervals or number of points.
- `random`: Generate arrays with random numbers.
- `identity`, `eye`: Create identity matrices.

### ➤ Common NumPy Data Types:
- **Integers**: `int8`, `int16`, `int32`, `int64`
- **Floats**: `float16`, `float32`, `float64`
- **Booleans**: `bool_`
- **Complex Numbers**: `complex64`, `complex128`

NumPy allows you to explicitly define the data type for performance and memory control.

---

## 🔹 3. Array Manipulations

### ➤ Indexing & Slicing:
- Access elements in 1D, 2D, and multi-dimensional arrays using standard slicing (`[start:stop:step]`).
- Boolean and fancy indexing enable advanced filtering.

### ➤ Flattening & Reshaping:
- `flatten`/`ravel`: Convert multi-dimensional array to 1D.
- `reshape`: Change the shape of an array without changing the data.

### ➤ Joining Arrays:
- Combine arrays using concatenation, stacking (horizontal, vertical, depth-wise).

### ➤ Splitting Arrays:
- Split arrays into multiple sub-arrays along specified axes.

### ➤ Broadcasting:
- Allows operations on arrays of different shapes by virtually expanding the smaller array without actual memory duplication.
- Essential for efficient element-wise operations.

---

## ✅ Summary:

- NumPy provides a high-performance array object and tools to work with large data efficiently.
- It is faster, more memory-efficient, and more versatile than Python lists.
- Learning NumPy is foundational for data science, ML, and scientific computing in Python.

---
