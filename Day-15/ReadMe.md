# Day 15 - Pose Estimation using OpenCV DNN (Deep Neural Networks)

In this exercise, **human pose estimation** using a pre-trained Caffe model and OpenCV's DNN module. Pose estimation refers to the process of detecting the position of a person’s joints and limbs from an image. It is commonly used in sports analytics, fitness apps, AR/VR systems, and motion capture.

---

## Key Concepts

* **Pose Estimation**: The task of localizing body parts (like elbows, knees, etc.) and connecting them to form a full human skeleton.
* **Pre-trained Caffe Model**: used a model trained on the MPI (Multi-Person Image) dataset for estimating the human body pose.
* **OpenCV DNN Module**: Provides support for various deep learning frameworks (like TensorFlow, Caffe, Darknet) and allows us to do inference without needing TensorFlow or PyTorch installed.

---

## Pipeline Summary

1. **Download Assets**: The notebook fetches necessary images, model files, and configurations using a Dropbox URL.
2. **Load Model**:

   * `pose_deploy_linevec_faster_4_stages.prototxt` defines the network architecture.
   * `pose_iter_160000.caffemodel` contains the pre-trained weights.
3. **Read and Preprocess Image**:

   * Input image is read and converted from BGR to RGB.
   * The image is converted to a blob and resized to (368×368) for feeding into the network.
4. **Run Inference**:

   * Perform a forward pass to get confidence maps for 15 keypoints (e.g., shoulders, elbows, knees).
   * Each map represents the confidence of the keypoint’s location.
5. **Post-processing**:

   * Extract the most confident point from each map and scale them to the original image dimensions.
   * Draw circles and labels for each keypoint.
   * Connect the keypoints using predefined POSE PAIRS to visualize the skeleton.

---

## Key Variables

* `nPoints = 15`: Number of keypoints predicted by the model.
* `POSE_PAIRS`: Defines how points should be connected (e.g., \[0,1] for head to neck).
* `threshold = 0.1`: Minimum confidence to accept a keypoint.

---

## Output

The final visualization includes:

* The keypoints (joints) detected on the person’s image.
* The skeletal structure connecting those points.

---
