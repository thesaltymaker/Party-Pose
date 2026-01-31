# Party-Pose Model Configuration

## Required Models

Place all models in the `models/` subdirectory.

### Pose Detection (Two-Stage Pipeline)
1. **pose_detection_128x128_float32.onnx**
   - BlazePose detector (stage 1)
   - Input: 128x128 RGB image
   - Output: Body bounding boxes with confidence scores
   - Source: PINTO model zoo

2. **pose_landmark_upper_body_256x256_float32.onnx**
   - BlazePose upper body landmarks (stage 2)
   - Input: 256x256 RGB image (cropped body region)
   - Output: Upper body keypoints (25 or 33 landmarks)
   - Source: PINTO model zoo

### Face Detection & Mesh (Two-Stage Pipeline)
3. **det_2.5g.onnx**
   - SCRFD face detector (stage 1)
   - Input: 320x320 image
   - Output: Face bounding boxes

4. **face_landmark_192x192.onnx**
   - MediaPipe face mesh (stage 2)
   - Input: 256x256 RGB image (cropped face region)
   - Output: 478 face landmarks

### Hand Detection & Landmarks (Two-Stage Pipeline)
5. **MediaPipeHandDetector.onnx**
   - BlazePalm hand detector (stage 1)
   - Input: 256x256 RGB image
   - Output: Hand bounding boxes

6. **hand_landmark.onnx**
   - MediaPipe hand landmarks (stage 2)
   - Input: 224x224 RGB image (cropped hand region)
   - Output: 21 hand keypoints per hand

## Pipeline Architecture

All three features use the same two-stage detection approach:

1. **Stage 1 - Detection**: Run lightweight detector on full frame to find region of interest (ROI)
2. **Stage 2 - Landmarks**: Crop ROI, run detailed landmark model on cropped region

This approach provides:
- Better accuracy (focused on relevant regions)
- Better performance (landmark models run on smaller crops)
- Robustness (detectors handle varying positions/scales)

## Model Sources

- BlazePose models: [PINTO Model Zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- MediaPipe models: Converted from MediaPipe TFLite models
- SCRFD: Face detection model
