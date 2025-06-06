from base_transform import load_image, apply_transform
import numpy as np
import cv2
import matplotlib.pyplot as plt

def projective_transform(img, src_pts, dst_pts):
    """
    src_pts and dst_pts are 4x2 arrays representing
    source and destination quadrilaterals.
    """
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return apply_transform(img, H)

if __name__ == "__main__":
    img = load_image('Day-6/cat1.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols = img.shape[:2]

    # Source points (corners of the image)
    src_points = np.array([
        [0, 0],
        [cols - 1, 0],
        [cols - 1, rows - 1],
        [0, rows - 1]
    ], dtype=np.float32)

# Define 4 destination points (where we want the above points to map)
    dst_points = np.array([
        [0, 0],
        [cols*0.8, rows*0.2],
        [cols*0.7, rows*0.9],
        [cols*0.2, rows*0.8]
    ], dtype=np.float32)
    
    transformed = projective_transform(img_rgb, src_points, dst_points)
    #transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    
    # Show original and warped images side by side
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.scatter(src_points[:, 0], src_points[:, 1], c='red', marker='o')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Warped Image')
    plt.imshow(transformed)
    plt.scatter(dst_points[:, 0], dst_points[:, 1], c='red', marker='o')
    plt.axis('off')

    plt.show()
#     cv2.imshow("Projective Transform", transformed)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()







