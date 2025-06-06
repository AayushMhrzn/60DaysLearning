from base_transform import load_image, apply_transform
import numpy as np
import cv2

def translate_image(img, tx, ty):
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)
    return apply_transform(img, T)

if __name__ == "__main__":
    img = load_image('Day-6/cat1.jpg')
    translated = translate_image(img, 100, 50)
    cv2.imshow("Translated Image", translated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
