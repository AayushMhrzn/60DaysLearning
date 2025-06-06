import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return img

# def to_homogeneous(coords):
#     """Convert Nx2 array of coordinates to Nx3 homogeneous coordinates"""
#     ones = np.ones((coords.shape[0], 1))
#     return np.hstack([coords, ones]) #converts each 2D coordinate (x, y) into homogeneous coordinates (x, y, 1).

# def from_homogeneous(coords):
#     """Convert Nx3 homogeneous coordinates back to Nx2"""
#     return coords[:, :2] / coords[:, 2].reshape(-1, 1)

def apply_transform(img, matrix):
    """
    Applies a 3x3 transformation matrix to an image.
    Uses cv2.warpPerspective to support all transforms.
    """
    rows, cols = img.shape[:2]
    transformed_img = cv2.warpPerspective(img, matrix, (cols, rows))
    return transformed_img
