# Day 8 – Line Detection in Video

The goal of this task is to detect straight lines in a video feed (such as road lanes) using image processing techniques. This is done by applying **Canny edge detection**, then **Hough Line Transform**, and finally overlaying the detected lines onto the original video frame.

---

### 1. Reading the Video

We load a video file using OpenCV’s `VideoCapture`, which allows us to process it frame by frame.

---

### 2. Preprocessing

Before detecting lines, each frame is processed with the following steps:

- **Grayscale Conversion**:  
  Converts the color image to grayscale to reduce complexity.

- **Gaussian Blur**:  
  Applies a blur to reduce noise and avoid detecting false edges.

- **Canny Edge Detection**:  
  Detects sharp changes in pixel intensity, i.e., edges.

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
```
---

### 3. Region of Interest (ROI) Masking with a Polygon
We define a specific region in the frame where lines are most likely to appear — for example, the bottom triangle of a road scene.
```python
mask = np.zeros_like(edges)  # Creates a black image of same shape
height, width = edges.shape
polygon = np.array([[(0, height), (width, height), (width // 2, height // 2)]])
cv2.fillPoly(mask, polygon, 255)  # Fills triangle with white (255)
masked_edges = cv2.bitwise_and(edges, mask)
```
# Why np.zeros_like(edges)?
- It creates a blank (black) image of the same dimensions as edges.

- It’s used to create a mask so that we can isolate only the region we care about.

# Why cv2.bitwise_and?
- It performs an AND operation between the mask and the edge-detected image.

- Only the edges inside the white polygon area are kept; everything else is zeroed out (masked).
![Alt text](Day-8/masking.png)

---

### 4. Hough Line Transform
Detects lines based on votes in parameter space.
` lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, maxLineGap=50) `
- 1: Distance resolution in pixels.

- np.pi/180: Angle resolution in radians.

- 50: Minimum number of votes (intersections) to detect a line.

- maxLineGap: Maximum allowed gap between line segments to treat them as a single line.

---

### 5. Drawing Detected Lines
```python
line_img = np.zeros_like(frame)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
```
- A new blank image is created (line_img).

- Detected lines are drawn with green color and thickness of 3.

---

### 6. Combining with Original Frame
`combined = cv2.addWeighted(frame, 0.8, line_img, 1, 1)`