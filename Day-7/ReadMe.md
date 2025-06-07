# Day 7 – Real-Time Edge Detection and Filters in OpenCV

###  Edge Detection
- **Edge Detection** identifies points in an image where brightness changes sharply.
- We used **Canny Edge Detection**, which is a multi-stage process:
  1. **Noise Reduction** (via Gaussian blur)
  2. **Gradient Calculation**
  3. **Non-maximum Suppression**
  4. **Hysteresis Thresholding**

- **OpenCV Function:** `cv2.Canny(image, threshold1, threshold2)`
- You can tune these thresholds using **trackbars** in real-time.

###  Preprocessing Techniques for Better Edges
- **Grayscale conversion**: Simplifies the image to a single channel.
- **Gaussian Blur**: Reduces noise and smooths the image.
- **Bilateral Filter**: Preserves edges while smoothing — useful before edge detection.

---

##  Real-Time Trackbars
- We created a UI using `cv2.createTrackbar()` to dynamically adjust threshold values.
- This helped in fine-tuning edge detection output interactively.

---

##  Filters in Image Processing

###  What is a Filter?
- A **filter** is a small matrix (kernel) that is slid over the image to modify pixel values.
- This operation is called **convolution**.
- The filter enhances, blurs, sharpens, or detects edges.

###  How it Works:
- The kernel (e.g. 3×3) is multiplied with the pixel values of a small neighborhood.
- The sum of these multiplications replaces the central pixel value.

###  Common Filters
| Filter Type  | Kernel Matrix Example              | Effect             |
|--------------|------------------------------------|---------------------|
| Blur         | `1/9 * np.ones((3, 3))`            | Softens image       |
| Sharpen      | `[[0,-1,0],[-1,5,-1],[0,-1,0]]`     | Highlights edges    |
| Edge Detect  | `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` | Detects boundaries  |
| Emboss       | `[[-2,-1,0],[-1,1,1],[0,1,2]]`      | Adds 3D relief      |

###  Code Concepts
- **`cv2.filter2D()`** applies the kernel to the image.

---

