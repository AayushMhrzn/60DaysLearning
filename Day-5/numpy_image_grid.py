import numpy as np
import cv2

# Loading 4 images
img1 = cv2.imread('cat1.jpg')
img2 = cv2.imread('cat2.jpg')
img3 = cv2.imread('cat3.jpg')
img4 = cv2.imread('cat4.jpg')

# Check if it's a NumPy array
print(type(img1))               # Output: <class 'numpy.ndarray'>

# Check the shape (Height x Width x Channels)
print(img1.shape)               # Example: (400, 400, 3)

# Check data type of pixel values
print(img1.dtype)               # Example: uint8

# View the raw pixel values (optional)
print(img1)

images = [img1, img2, img3, img4]

# Ensure all images are the same shape
H, W, C = images[0].shape
print(H,W,C)

# Grid size (2x2 for 4 images)
grid_size = int(np.ceil(np.sqrt(len(images))))

# Create an empty canvas for the grid
grid_img = np.zeros((grid_size * H, grid_size * W, C), dtype=np.uint8)

# Place each image in the grid
for idx, img in enumerate(images):
    row = idx // grid_size
    col = idx % grid_size
    grid_img[row*H:(row+1)*H, col*W:(col+1)*W,:] = img

# Show the final grid
cv2.imshow("Image Grid", grid_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
