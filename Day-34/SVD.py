import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# Load and preprocess image
image = color.rgb2gray(data.astronaut())  # grayscale image
image = resize(image, (256, 256), anti_aliasing=True)

# Display original image
plt.figure(figsize=(6, 6))
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Perform SVD
U, S, VT = np.linalg.svd(image, full_matrices=False)
print("Original shape:", image.shape)
print("U shape:", U.shape, "S shape:", S.shape, "VT shape:", VT.shape)

# Function to reconstruct image using top k singular values
def reconstruct_image(U, S, VT, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    return np.dot(U_k, np.dot(S_k, VT_k))

# Try different ranks for reconstruction
ranks = [5, 20, 50, 100, 150]

plt.figure(figsize=(15, 6))
for i, k in enumerate(ranks):
    recon = reconstruct_image(U, S, VT, k)
    plt.subplot(1, len(ranks), i + 1)
    plt.imshow(recon, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')
plt.suptitle("Image Reconstruction using SVD with Different Ranks")
plt.tight_layout()
plt.show()
