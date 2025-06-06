from base_transform import load_image, apply_transform
import numpy as np
import cv2

def affine_transform(img, matrix_2x3):
    """
    matrix_2x3 is a 2x3 affine transform matrix:
    [[a11, a12, tx],
     [a21, a22, ty]]
    """
    rows, cols = img.shape[:2]
    return cv2.warpAffine(img, matrix_2x3, (cols, rows))

if __name__ == "__main__":
    img = load_image('Day-6/cat1.jpg')
    # Example: shear + scale + translate
    M = np.array([
        [1.2, 0.3, 30],
        [0.1, 1.1, 40]
    ], dtype=np.float32)
    transformed = affine_transform(img, M)
    cv2.imshow("Affine Transform", transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
