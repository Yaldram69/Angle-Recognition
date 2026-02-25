# Angle Recognition (Object Orientation Detection)

Real-time **object orientation (angle) detection** using **OpenCV + PyTorch + Mask R-CNN** on a webcam feed.

This project detects objects using a pretrained **Mask R-CNN (ResNet50-FPN)** model from `torchvision`, extracts segmentation masks, and computes each detected object's **rotation angle** (in degrees) using OpenCV’s `minAreaRect`. The result is displayed live with:

- Axis-aligned bounding box
- Rotated bounding box
- Angle label (e.g., `37.4°`)

---

## Features

- Real-time webcam inference
- Instance segmentation with pretrained Mask R-CNN
- Orientation/angle estimation from object masks
- Rotated rectangle visualization
- Angle annotation on live video stream

---

## Tech Stack

- Python
- OpenCV (`cv2`)
- NumPy
- PyTorch
- Torchvision

---

## How It Works

1. Capture frames from webcam
2. Run Mask R-CNN inference
3. Filter detections by confidence threshold
4. Convert predicted masks to binary masks
5. Fit a minimum-area rotated rectangle (`cv2.minAreaRect`)
6. Convert OpenCV angle format to a human-readable orientation angle
7. Draw bounding boxes and angle labels on the frame

---

## Project Structure

```text
Angle Recognition/
└── test.py
