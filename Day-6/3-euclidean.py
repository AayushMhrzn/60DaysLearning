from base_transform import load_image, apply_transform
import numpy as np
import cv2
import math

def euclidean_transform(img, theta_deg, tx, ty):
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    E = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [0,      0,     1]
    ], dtype=np.float32)
    return apply_transform(img, E)

if __name__ == "__main__":
    img = load_image('Day-6/cat1.jpg')
    transformed = euclidean_transform(img, 45, 50, 50)
    cv2.imshow("Euclidean Transform", transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
