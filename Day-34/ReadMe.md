# Day 34 – Singular Value Decomposition (SVD) for Image Compression
Understand the theory and math behind SVD and use it to compress a grayscale image by reconstructing it with fewer singular values.

## What is SVD?
Singular Value Decomposition (SVD) is a linear algebra technique used to factorize any real or complex matrix A into three matrices:

`A = U * Σ * Vᵀ`
Where:

- A is the original matrix of shape (m × n)
- U is an orthogonal matrix of shape (m × m)
- Σ (Sigma) is a diagonal matrix (m × n) with non-negative real values (called singular values)
- Vᵀ is the transpose of an orthogonal matrix of shape (n × n)
These singular values tell us the importance (energy) of each component.

## Why Use SVD?
Dimensionality reduction: Keep only top-k singular values to approximate original matrix.
Noise reduction: Small singular values often represent noise.
Compression: Store much less data than original matrix.

## Use Case: Image Compression
In image processing, a grayscale image is just a 2D matrix of pixel intensities (0–1 or 0–255). We can apply SVD to compress it by reconstructing it with only top k singular values.

## SVD Math Recap
Given an image matrix A ∈ ℝ^{m×n}, SVD factorizes it as:

A = U Σ Vᵀ
Let’s approximate A using only top k singular values:

`A_k ≈ U_k Σ_k V_kᵀ`
Where:

U_k: first k columns of U
Σ_k: top-left k×k part of Σ
V_kᵀ: first k rows of Vᵀ

The higher the k, the better the approximation.

## code
```python
U, S, VT = np.linalg.svd(image, full_matrices=False)
```
- Perform SVD using NumPy's built-in function:
U: left singular vectors (256×256)
S: singular values (256,)
VT: right singular vectors transposed (256×256)


```python
def reconstruct_image(U, S, VT, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    return np.dot(U_k, np.dot(S_k, VT_k))
```
- This function reconstructs image using top k singular values and vectors.

## Output
With k=5: very blurry, but some shapes visible
With k=50: decent quality
With k=150: nearly indistinguishable from original!

## Compression Ratio
If original image is size m×n = 256×256 = 65536 pixels, and we only store:

U_k: m×k
S_k: k
V_k: k×n
- Total storage = k × (m + n + 1)

So for k=50:

Storage = 50 × (256 + 256 + 1) = 50 × 513 = 25,650
Compression = 25,650 / 65,536 ≈ 0.39 → 61% reduction!

## Summary
- SVD is a matrix decomposition method useful for compression and dimensionality reduction
- Singular values reflect the importance of components
- Using fewer components can reconstruct the original image with less storage
- It’s a great intuitive introduction to dimensionality reduction