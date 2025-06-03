import numpy as np


# 1. Indexing & Slicing

# 1D Indexing
arr_1d = np.array([10, 20, 30, 40, 50])
print("1D Indexing:", arr_1d[2])        # Output: 30
print("1D Slicing:", arr_1d[1:4])       # Output: [20 30 40]

# 2D Indexing
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Indexing:", arr_2d[1][2])   # Output: 6
print("2D Slicing:\n", arr_2d[:, 1:])   # Output: [[2 3]
                                        #          [5 6]]


# 2. Flatten & Ravel

# Flatten returns a copy
flat = arr_2d.flatten()
print("\nFlatten:", flat)               # Output: [1 2 3 4 5 6]

# Ravel returns a view (faster & memory efficient)
rav = arr_2d.ravel()
print("Ravel:", rav)                    # Output: [1 2 3 4 5 6]


# 3. Reshape

reshaped = np.arange(6).reshape(2, 3)
print("\nReshaped:\n", reshaped)
# Output:
# [[0 1 2]
#  [3 4 5]]


# 4. Join (Concatenate & Stack)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

# Concatenate along axis 0
concat = np.concatenate((a, b), axis=0)
print("\nConcatenate:\n", concat)
# Output:
# [[1 2]
#  [3 4]
#  [5 6]]

# Horizontal Stack (columns)
h_stack = np.hstack((a, a))
print("\nHorizontal Stack:\n", h_stack)
# Output:
# [[1 2 1 2]
#  [3 4 3 4]]

# Vertical Stack (rows)
v_stack = np.vstack((a, b))
print("\nVertical Stack:\n", v_stack)
# Output:
# [[1 2]
#  [3 4]
#  [5 6]]

# Depth Stack
depth_stack = np.dstack((a, a))
print("\nDepth Stack:\n", depth_stack)
# Output:
# [[[1 1]
#   [2 2]]
#
#  [[3 3]
#   [4 4]]]


# 5. Split

arr = np.array([1, 2, 3, 4, 5, 6])
a,b,c = np.split(arr, 3)
print("\nSplit:", "a=",a,"b=",b,"c=",c)
# Output: Split: a= [1 2] b= [3 4] c= [5 6]


# 6. Broadcasting

x = np.array([[1], [2], [3]])   # Shape (3,1)
y = np.array([10, 20, 30])      # Shape (3,)
broadcasted_sum = x + y
print("\nBroadcasting Result:\n", broadcasted_sum)
# Output:
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
