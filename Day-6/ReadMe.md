# Day 6 – 2D Image Transformations Using Matrix Multiplication

## Overview

In this task, we learn how to manipulate images in 2D space by applying various geometric transformations. These transformations are mathematically represented using matrices. Applying these matrices to the coordinates of an image allows us to translate, rotate, scale, or distort the image in a controlled manner.

---

## Why Use Matrix Multiplication?

Matrix multiplication provides a compact and efficient way to apply linear transformations and translations to image points. Using **homogeneous coordinates** (adding a 1 to coordinate vectors), we can combine multiple transformations into a single matrix operation.

---

## Types of 2D Transformations

### 1. Translation

**What:** Moves an image by shifting all pixels horizontally and/or vertically.

**Matrix:**
```
A = | 1  0  tx |
    | 0  1  ty |
    | 0  0  1  |
```

**Explanation:**  
`tx`, `ty` are translation distances along x and y axes.

---

### 2. Euclidean Transformation

**What:** Combination of rotation and translation without changing shape or size.

**Matrix:**

```
E = | cosθ  -sinθ  tx |
    | sinθ   cosθ  ty |
    |  0       0    1 |
```


**Explanation:**  
Rotates by `θ` radians and translates by `(tx, ty)`.

---

### 3. Affine Transformation

**What:** General linear transformation including rotation, translation, scaling, and shearing.

**Matrix:**

```
A = | a11  a12  tx |
    | a21  a22  ty |
    |  0    0    1 |
```

**Explanation:**  
The `aij` values control rotation, scaling, and shearing effects.

---

### 4. Similarity Transformation

**What:** Combination of uniform scaling, rotation, and translation.

**Matrix:**

```
S = |  scosθ  -ssinθ   tx |
    |  ssinθ   scosθ   ty |
    |   0        0      1 |
```

**Explanation:**  
Scales by `s`, rotates by `θ`, and translates by `(tx, ty)`.

---

### 5. Projective (Homography) Transformation

**What:** Most general linear transformation that can map any quadrilateral to another quadrilateral, allowing perspective distortion.

**Matrix:**

```
P = | h11  h12  h13 |
    | h21  h22  h23 |
    | h31  h32  h33 |
```

**Explanation:**  
This 3×3 matrix can encode perspective warping and transformations such as skew and camera-like projections.

---

## How We Apply These Transformations

1. Convert image pixel coordinates into **homogeneous coordinates**: `(x, y, 1)`
2. Multiply each coordinate vector by the transformation matrix
3. Convert back from homogeneous coordinates to 2D by dividing by the third element (for projective)
4. Use the new coordinates to construct the transformed image

---


