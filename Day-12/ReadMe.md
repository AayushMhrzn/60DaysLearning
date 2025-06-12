# Day 12 - Image Stitching Using Homography

The goal was to learn how to **stitch two overlapping images** taken from close proximity into a single panorama. The task involved understanding and implementing the following key concepts:

---

### 1. Feature Extraction

We use a feature detector like **ORB (Oriented FAST and Rotated BRIEF)** to find keypoints in both images. These keypoints represent unique, detectable points such as corners or blobs.

```python
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
```

### 2. Feature Matching

We match these keypoints using a matcher like **BruteForce-Hamming** for binary descriptors:

```python
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

### 3. Homography and RANSAC

We extract the matched point coordinates from both images:

```python
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
```

Using these points, we estimate a **homography matrix H** using **RANSAC** to reject outliers:

```python
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
```

**Homography** is a 3x3 matrix that maps points from one image plane to another, considering perspective distortion.

### 4. Warping and Stitching

warp one image (img1) to the plane of the other using the homography:

```python
result = cv2.warpPerspective(img1, H, (width_sum, height))
```

Then paste the second image directly into the result:

```python
result[0:img2.shape[0], 0:img2.shape[1]] = img2
```

### Why Homography?

Homography is used because the two images are assumed to lie on the **same plane** or under **pure camera rotation**. Homography allows mapping one set of image points to another even with perspective changes.

The result is a stitched panorama that combines both overlapping images into a seamless view.

---

