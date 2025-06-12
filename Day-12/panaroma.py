import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load input images
img1 = cv2.imread("Day-12/left.jpg")   # left image
img2 = cv2.imread("Day-12/right.jpg")  # right image

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 1. Detect ORB features and compute descriptors.
orb = cv2.ORB_create(5000)
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# 2. Match features using Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:200], None, flags=2)


# 3. Extract location of good matches
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

# 4. Find Homography matrix using RANSAC
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# 5. Warp right image (img2) to left image (img1)'s plane
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

# Get the canvas dimensions
pts_img2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
pts_img2_transformed = cv2.perspectiveTransform(pts_img2, H)
pts_combined = np.concatenate((pts_img2_transformed, np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)), axis=0)

[x_min, y_min] = np.int32(pts_combined.min(axis=0).ravel())
[x_max, y_max] = np.int32(pts_combined.max(axis=0).ravel())

# Translation matrix to avoid negative coordinates
translation = [-x_min, -y_min]
H_translate = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

# Warp the second image
result = cv2.warpPerspective(img2, H_translate @ H, (x_max - x_min, y_max - y_min))
result[translation[1]:height1+translation[1], translation[0]:width1+translation[0]] = img1


cv2.imshow("matched points",img_matches)
cv2.imshow("Stitched Panorama", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
