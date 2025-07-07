#  PyTorch Tensor Operations - DAY 37
n this notebook, we started with constructing simple tensors and manipulating them

## 1. Converting Images to Batched tensors
An image is made up of pixel arrays that represent the intensity of pixels in grayscale or the color values in RGB format. When working with deep learning models, it's often necessary to convert these images into tensors, which are the primary data structures used in PyTorch for handling and processing data.

Tensors: In PyTorch, tensors are multi-dimensional arrays similar to NumPy arrays, but with additional capabilities for GPU acceleration and automatic differentiation. Tensors are the fundamental building blocks for representing data and parameters in neural networks.
Batches: Batching is a technique where multiple data samples (images, in this case) are grouped together into a single tensor. This allows efficient processing of multiple samples simultaneously, to take advantage of the parallel processing capabilities of modern hardware.

- Images were loaded using `cv2.imread()` both in color and grayscale.
- Normalization was done by dividing pixel values by 255.0.
- Batch was created using `torch.stack()`.
- Permutation was applied to follow PyTorch format `[N, C, H, W]`.

```python
batch_input = batch_tensor.permute(0, 3, 1, 2)  # from [N, H, W, C] to [N, C, H, W]
```

---
## 2. Introduction to Tensors and its Operations
We have seen the importance of tensors, now will understand it from ground up. Tensor is simply a fancy name given to matrices. If you are familiar with NumPy arrays, understanding and using PyTorch Tensors will be very easy. A scalar value is represented by a 0-dimensional Tensor. Similarly, a column/row matrix is represented using a 1-D Tensor and so on. Some examples of Tensors with different dimensions are shown for you to visualize and understand.

### Tensor Creation

| Dimension | Example Code |
|----------|---------------|
| 0D (Scalar) | `torch.tensor(5)` |
| 1D (Vector) | `torch.ones(5)` |
| 2D (Matrix) | `torch.zeros(3, 2)` |
| 3D (Tensor) | `torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])` |

---

## Access Tensor Elements

- Access single item: `tensor[1, 0]` or `tensor[1][0]`
- Slice rows/columns:

```python
tensor[0, :]  # first row
tensor[:, 1]  # second column
tensor[1:3]   # rows 1 to 2
```

---

## Data Types

- PyTorch automatically chooses dtype (e.g., `float32`, `int64`).
- You can specify or convert:

```python
tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
int_tensor = tensor.type(torch.int64)
```

---

##  NumPy ↔ Tensor Conversion

```python
# Tensor to NumPy
np_array = tensor.numpy()

# NumPy to Tensor
torch_tensor = torch.from_numpy(np_array)
```

---

##  Tensor Arithmetic

```python
# Element-wise | a and b are two tensors respectively
a + b
a - b
a * b
a / b

# Matrix multiplication
torch.mm(tensor1, tensor2)

# With scalars
a * 2
a / 2
```

---

##  Broadcasting
- a is a 1-dimensional tensor with shape ([ 3 ]).
- b is a scalar tensor with shape ([ 1 ]).
- When adding a and b, PyTorch broadcasts b to match the shape of a, resulting in ([ 1 + 4, 2 + 4, 3 + 4 ]).

```python
# Create two 1-dimensional tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4])

# adding a scalar to a vector
result = a + b
```
- Result of Broadcasting:
 tensor([5, 6, 7])

 - Broadcasting allows PyTorch to perform element-wise operations on tensors of

a is a 2-dimensional tensor with shape ([1, 3]).
b is a 2-dimensional tensor with shape ([3, 1]).
When adding a and b, PyTorch broadcasts both tensors to the common shape ([3, 3]), resulting in:

 
|1+4	2+4	    3+4|
|1+5	2+5	    3+5|
|1+6	2+6	    3+6|
 
