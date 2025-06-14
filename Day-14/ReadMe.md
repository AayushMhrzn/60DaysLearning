# Day 14 - Object Detection with OpenCV

Explored object detection using OpenCV's DNN module, the model used is the SSD MobileNet v2, which performs object detection on real-world images. The task is to detect and label various objects in an image with bounding boxes.

---

## What is SSD MobileNet?

* **SSD (Single Shot MultiBox Detector)** is a real-time object detection algorithm that detects multiple objects in a single forward pass through the network.
* **MobileNet** is a lightweight neural network optimized for mobile and edge devices. It serves as the backbone for the SSD detector.
* This combination provides a balance between accuracy and speed, making it ideal for real-time applications.

* **Model Files:** From TensorFlow's Object Detection Model Zoo

---

## Assets

downloaded additional images (e.g., street.jpg) and label files from an asset ZIP file hosted online using `urllib` and `zipfile`.

---

## COCO Class Labels

* The model is trained on the **COCO dataset**, which has 90 object classes.
* These class labels (e.g., person, car, dog, etc.) are loaded from `coco_class_labels.txt`.

---

## Object Detection Workflow

### Step 1: Load the Model

```python
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
```

Loads the pre-trained SSD model and its configuration into OpenCV.

### Step 2: Create a Blob

```python
blob = cv2.dnn.blobFromImage(im, size=(300, 300))
```

This converts the image into a 4D blob that is suitable for input to the network.

### Step 3: Perform Inference

```python
net.setInput(blob)
objects = net.forward()
```

A forward pass detects objects in the image and returns a set of detections.

### Step 4: Post-process and Display Results

* The detection results include class IDs, confidence scores, and bounding box coordinates.
* loop through the detections, filter out low-confidence ones, and draw rectangles + labels on the original image.

The final result is an image with multiple detected objects, each bounded by a rectangle and labeled with its class name. This provides a clear and interpretable visual output of object detection.

---

