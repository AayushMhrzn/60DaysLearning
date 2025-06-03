import sys
import numpy as np

py_list = list(range(1000))
np_array = np.arange(1000)

print("Memory used by list:", sys.getsizeof(py_list))  # size of list object
print("Memory used by numpy array:", np_array.nbytes)  # actual data memory


# 1. Zeros
arr_zeros = np.zeros((2, 3))
print("Zeros:\n", arr_zeros)
# Output:
# [[0. 0. 0.]
#  [0. 0. 0.]]

# 2. Ones
arr_ones = np.ones((3, 2))
print("\nOnes:\n", arr_ones)
# Output:
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

# 3. Full
arr_full = np.full((2, 2), 7)
print("\nFull with 7:\n", arr_full)
# Output:
# [[7 7]
#  [7 7]]

# 4. Identity Matrix
identity = np.eye(3)
print("\nIdentity Matrix:\n", identity)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 5. Arange
arr_range = np.arange(0, 10, 2)
print("\nArange (step 2):\n", arr_range)
# Output:
# [0 2 4 6 8]

# 6. Linspace
arr_linspace = np.linspace(0, 1, 5)
print("\nLinspace (0 to 1):\n", arr_linspace)
# Output:
# [0.   0.25 0.5  0.75 1.  ]

# 7. Random Values
arr_rand = np.random.rand(2, 2)
print("\nRandom values:\n", arr_rand)
# Output (will vary each time):
# [[0.49329822 0.80101934]
#  [0.00124845 0.35280199]]

# Different dtype examples
bool_arr = np.array([1, 0, 1], dtype=np.bool_)
int16_arr = np.array([1000, 2000], dtype=np.int16)
float_arr = np.array([1.5, 2.3], dtype=np.float64)
complex_arr = np.array([1+2j, 3+4j], dtype=np.complex_)

print("Boolean Array:", bool_arr)
print("Int16 Array:", int16_arr)
print("Float64 Array:", float_arr)
print("Complex Array:", complex_arr)
