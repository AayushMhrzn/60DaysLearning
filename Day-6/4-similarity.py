from base_transform import load_image, apply_transform
import numpy as np
import cv2
import math

def similarity_transform(img, scale, theta_deg, tx, ty):
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    S = np.array([
        [scale * cos_t, -scale * sin_t, tx],
        [scale * sin_t,  scale * cos_t, ty],
        [0,             0,              1]
    ], dtype=np.float32)
    return apply_transform(img, S)

if __name__ == "__main__":
    img = load_image('Day-6/cat1.jpg')
    transformed = similarity_transform(img, 1.5, 30, 20, 30)
    cv2.imshow("Similarity Transform", transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
