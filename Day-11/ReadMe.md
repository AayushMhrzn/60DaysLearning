# Day 11: Understanding Camera Matrix and 3D Transformations

The objective is to understand the fundamental concepts of 3D transformations, the camera matrix, and how 3D world points are projected to a 2D image plane using these matrices. also visualized this transformation with a cube example using Python and matplotlib.

---

## 1. Camera Matrix

The **camera matrix** (also known as the **projection matrix**) is used to convert 3D world coordinates into 2D image coordinates.

### Intrinsic Matrix (K)

The **intrinsic matrix** encodes the internal parameters of the camera:

```
K = [ fx  0  cx ]
    [ 0  fy  cy ]
    [ 0   0   1 ]
```

* **fx, fy**: Focal lengths in pixels (usually fx = fy).
* **cx, cy**: Principal point (typically the center of the image).

### Extrinsic Matrix \[R|t]

The **extrinsic matrix** transforms coordinates from the world to the camera coordinate system.

* **R**: Rotation matrix (3x3)
* **t**: Translation vector (3x1)

Combining both:

```
P = K * [R | t]  --> (3x4 matrix)
```

This matrix projects a 3D point `X = [x, y, z, 1]^T` in homogeneous coordinates to a 2D point `x' = [u, v, 1]^T`.

---

## 2. 3D to 2D Projection

To project a 3D point to 2D image plane:

1. Add homogeneous coordinate: `[x, y, z, 1]`
2. Multiply with the projection matrix: `x' = P * X`
3. Normalize by the third coordinate to get pixel coordinates:

```
[u, v] = [x'/z', y'/z']
```

---

## 3. Code Explanation

built a cube in 3D and used a manually defined camera matrix to project it into a 2D image:

* A cube with 8 vertices was defined.
* An intrinsic matrix `K` and extrinsic matrix `[R|t]` were created.
* The cube vertices were projected using the projection matrix `P = K * [R|t]`.
* The original 3D cube and the 2D projected image were plotted side-by-side using `matplotlib`.

---

## 4. Visualization

* Left: 3D cube in world coordinates
* Right: 2D projection from the camera's viewpoint
