# Day 9 – Real-time Face Detection using OpenCV DNN

## Objective

This project demonstrates how to use OpenCV's Deep Neural Network (DNN) module to perform real-time face detection on a video feed. The detection is powered by a pre-trained deep learning model (SSD with ResNet-10), and the system is capable of displaying the confidence and inference time for each detection.

---

## What is DNN Face Detection?

The OpenCV DNN module allows running deep learning models without requiring frameworks like TensorFlow or PyTorch. In this task, we use a Caffe-based model trained for detecting faces in images using a **Single Shot Multibox Detector (SSD)** with a **ResNet-10** base network.

---

## Key Files Used

* **`deploy.prototxt`**: Defines the architecture of the network.
* **`res10_300x300_ssd_iter_140000_fp16.caffemodel`**: Pre-trained weights of the model.

These are standard files provided by OpenCV's model zoo.

---

## Breakdown of the Code

### 1. **Video Input Initialization**

The script checks for a command-line argument to select the video source:

* If an argument is passed (e.g., a path to a video), it uses that.
* If not, it defaults to the webcam.

```python
source = cv2.VideoCapture(s)
```

### 2. **Model Loading**

```python
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
```

The pre-trained model is loaded from disk using OpenCV's DNN interface.

### 3. **Preprocessing the Frame**

Each frame is converted to a 4D blob using:

```python
blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), mean, swapRB=False, crop=False)
```

This:

* Resizes the image to 300x300 (model input size)
* Subtracts mean values for normalization
* Returns a blob that can be passed into the network

### 4. **Running Inference**

The blob is set as input and the model is run:

```python
net.setInput(blob)
detections = net.forward()
```

`detections` is a 4D array containing detection results.

### 5. **Processing Detections**

For each detection:

* Confidence score is checked
* Bounding box coordinates are calculated based on image size
* A green rectangle and confidence label are drawn

```python
if confidence > conf_threshold:
    x_top_left = int(detections[0, 0, i, 3] * frame_width)
    ...
```

### 6. **Inference Time Display**

```python
t, _ = net.getPerfProfile()
label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
```

This provides an idea of the speed of processing per frame.

---
