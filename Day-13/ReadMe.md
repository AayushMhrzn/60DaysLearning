# Day 13 – Object Tracking using OpenCV

The goal of this task is to understand **object tracking** in videos using OpenCV's built-in tracking algorithms. By the end of this task, you'll have practical experience in tracking a moving object across frames using bounding boxes and various tracking algorithms.

---

## What is Object Tracking?

Object tracking refers to the process of **locating a moving object** (or multiple objects) over time using a camera. It differs from object detection in that:

* **Detection** locates objects in individual frames.
* **Tracking** links detections across video frames to form a temporal trajectory.

---

## OpenCV Object Trackers

OpenCV provides multiple tracking algorithms under the `cv2.Tracker` module:

| Tracker    | Description                                         |
| ---------- | --------------------------------------------------- |
| BOOSTING   | Based on AdaBoost. Old and less accurate.           |
| MIL        | Multiple Instance Learning.                         |
| KCF        | Kernelized Correlation Filters. Fast and accurate.  |
| TLD        | Tracks and Learns. Handles occlusion.               |
| MEDIANFLOW | Good for predictable motion. Fails under occlusion. |
| GOTURN     | Deep learning-based tracker. High accuracy.         |
| MOSSE      | Fastest. Robust under lighting changes.             |
| CSRT       | Discriminative correlation filter. Very accurate.   |

In this , **GOTURN** is demonstrated.

---

## Pipeline Overview

### 1. Load Assets

Assets are downloaded and extracted using Python's `urllib` and `zipfile` modules.

### 2. Helper Functions

* `drawRectangle`: Draws a bounding box.
* `displayRectangle`: Shows a frame with a bounding box.
* `drawText`: Adds overlay text like FPS and tracker name.

### 3. Tracker Selection

Choose a tracker from the list. Example:

```python
tracker_type = tracker_types[6]  # Goturn
```

### 4. Read Video

Video is loaded using `cv2.VideoCapture` and an output writer is initialized using XVID codec.

### 5. Define Initial Bounding Box

A bounding box is defined for the object to track. This can be hardcoded or selected interactively.

### 6. Initialize Tracker

```python
tracker.init(frame, bbox)
```

This sets the initial frame and object position for tracking.

### 7. Track Object Frame-by-Frame

For each frame:

* Update tracker with the current frame.
* Draw updated bounding box.
* Calculate and display FPS.
* Save the frame to an output video.

---

## Tracker Metrics

* **FPS**: Indicates performance.
* **Accuracy**: Depends on tracker choice and video conditions (motion blur, occlusion, lighting).

---

## Output

* Tracked object displayed using a bounding box.
* Final output video displayed in notebook using base64 encoding.

---
