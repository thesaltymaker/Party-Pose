# Poser - Real-time Pose Detection Party App

## Overview
Lightweight, high-performance pose detection application for party entertainment.
Real-time video processing with configurable overlays and minimal latency.
No data persistence - pure ephemeral fun!

## Requirements

### Functional Requirements
- Real-time webcam video capture and display
- Pose landmark detection and overlay visualization
- Configurable processing: face, body, hands (independently toggle)
- Mirror mode for natural user interaction
- Region of Interest (ROI) visualization for each model
- Performance monitoring (FPS counter)
- Command-line driven configuration
- Clean, immediate visual feedback
- Working python code writen to the disk

### Technical Requirements
- **Backend**: Python 3.10+
- **Core Libraries**: OpenCV (cv2), ONNX Runtime, NumPy
- **Models**: Pre-converted MediaPipe ONNX models in `models/` directory
- **GPU Processing**: Keep frames on GPU memory throughout pipeline
- **CUDA**: ONNX Runtime with CUDA execution provider
- **Performance Target**: >30 FPS on RTX 3060/5080
- **No Persistence**: Zero database, zero file storage

### MediaPipe ONNX Models
The models follow MediaPipe's two-stage detection pipeline:

**Face Pipeline:**
- `face_detector_opset15.onnx` - Detects faces, returns bounding boxes
- `face_landmarks_detector_opset15.onnx` - Extracts 468 face landmarks
- `face_blendshapes_opset15.onnx` - Facial expression blendshapes (52 coefficients)

**Hand Pipeline:**
- `hand_detector_opset15.onnx` - Detects hands, returns bounding boxes
- `hand_landmarks_detector_opset15.onnx` - Extracts 21 hand landmarks per hand

**Body Pipeline:**
- `pose_landmarks_detector_opset15.onnx` - Full body pose with 33 landmarks

### Pipeline Architecture
Each modality uses a two-stage approach:
1. **Detector** - Fast model that finds ROI (Region of Interest) bounding boxes
2. **Landmarks** - Precise model that extracts detailed landmarks within ROI

Exception: Body pose uses single-stage full-frame detection.

### Why ONNX?
- **Dependency Hell Solved**: MediaPipe + TensorFlow version conflicts eliminated
- **Flexibility**: Can swap/upgrade models independently
- **Performance**: ONNX Runtime GPU inference is optimized
- **Portability**: Models work across different inference engines

## Architecture

### Pipeline Flow (GPU-resident)
```
Webcam → GPU Upload → Detector (ROI) → Landmarks (within ROI) → Overlay → Display
         ↑_______________________________________________________________|
                      (Frame stays in GPU memory)
```

### Processing Flow Details

**Face Processing:**
```
Frame → face_detector → [face bounding boxes] 
      → face_landmarks_detector (per face ROI) → [468 landmarks per face]
      → face_blendshapes (optional, per face) → [52 expression coefficients]
```

**Hand Processing:**
```
Frame → hand_detector → [hand bounding boxes (left/right)]
      → hand_landmarks_detector (per hand ROI) → [21 landmarks per hand]
```

**Body Processing:**
```
Frame → pose_landmarks_detector → [33 body landmarks + visibility scores]
```

### Components
1. **Video Capture Module** (OpenCV CUDA backend)
2. **ONNX Model Manager** (lazy loading, two-stage pipeline management)
3. **Detector Models** (fast ROI localization)
4. **Landmark Models** (precise keypoint extraction)
5. **Inference Engine** (ONNX Runtime with CUDAExecutionProvider)
6. **Rendering Pipeline** (OpenCV CUDA drawing operations)
7. **CLI Argument Parser** (configuration management)

### Model Structure
```
models/
├── face_detector_opset15.onnx                  # Stage 1: Face detection
├── face_landmarks_detector_opset15.onnx        # Stage 2: Face landmarks
├── face_blendshapes_opset15.onnx               # Optional: Expressions
├── hand_detector_opset15.onnx                  # Stage 1: Hand detection
├── hand_landmarks_detector_opset15.onnx        # Stage 2: Hand landmarks
└── pose_landmarks_detector_opset15.onnx        # Single stage: Body pose
```

## Command-Line Interface
```bash
python poser.py \
  --mirror              # Flip frame horizontally (default: False)
  --no-face            # Disable face detection
  --no-body            # Disable body pose detection
  --no-hands           # Disable hand detection
  --show-roi           # Show Region of Interest boxes for detectors
  --fps                # Display FPS counter
  --face-expressions   # Enable face blendshapes (expression detection)
  --camera 0           # Camera device index (default: 0)
  --width 1280         # Frame width (default: 1280)
  --height 720         # Frame height (default: 720)
```

### Usage Examples
```bash
# Full features, mirrored, with FPS
python poser.py --mirror --show-roi --fps

# Body pose only, high performance mode
python poser.py --no-face --no-hands --fps

# Party selfie mode (mirrored, all features including expressions)
python poser.py --mirror --face-expressions

# Debug mode (ROI boxes, FPS, no face processing)
python poser.py --show-roi --fps --no-face

# Hands-only dance mode
python poser.py --no-face --no-body --mirror --fps
```

## Features

### Core Features
1. **Selective Processing**: Toggle face/body/hands independently
2. **Mirror Mode**: Natural interaction for selfie-style use
3. **ROI Visualization**: See detector bounding boxes
4. **FPS Counter**: Real-time performance monitoring
5. **GPU-Resident Processing**: Minimal CPU↔GPU transfer overhead
6. **Face Expressions**: Optional blendshape coefficients for facial analysis

### Visual Overlays
- Skeleton lines connecting landmarks (configurable colors per model)
- Landmark points (circles/dots with different sizes)
- ROI bounding boxes from detectors (when --show-roi enabled)
- FPS counter (corner overlay)
- Model status indicators (which models are active)
- Color scheme:
  - Face: Green landmarks, cyan bounding boxes
  - Hands: Yellow/Orange landmarks (left/right differentiation)
  - Body: Blue landmarks and skeleton

## Technical Constraints

### GPU Memory Strategy
1. **Upload Once**: Capture frame → GPU upload → never download until display
2. **ROI Processing**: Extract ROI on GPU, run landmark inference
3. **Memory Pooling**: Pre-allocate inference buffers for each model
4. **Zero Copy**: Use cv2.cuda.GpuMat throughout pipeline

### Two-Stage Pipeline Details
**Efficiency Consideration:**
- Detectors are lightweight, run on full frame
- Landmark models are heavier, but run only on small ROI crops
- Net result: Much faster than running landmark models on full frames

**Implementation:**
```python
# Pseudocode for two-stage pipeline
frame_gpu = capture_frame()  # GpuMat

# Stage 1: Detection
roi_boxes = detector_model.infer(frame_gpu)  # Fast, full frame

# Stage 2: Landmarks (per ROI)
for roi in roi_boxes:
    roi_crop = frame_gpu[roi.y:roi.y+h, roi.x:roi.x+w]  # GPU crop
    landmarks = landmark_model.infer(roi_crop)  # Precise, small region
    landmarks_image_space = transform_to_image_coords(landmarks, roi)
```

### Performance Requirements
- Minimize CPU↔GPU transfers (primary bottleneck)
- Use ONNX Runtime's CUDA execution provider
- Leverage OpenCV CUDA operations for preprocessing
- Target: <5ms CPU↔GPU transfer per frame
- Target: >30 FPS on RTX 3060, >60 FPS on RTX 5080
- Detector models: <5ms inference
- Landmark models: <10ms inference per ROI

### Model Loading Strategy
- Lazy load: Only initialize requested model pipelines
- Pair management: Keep detector + landmark models together
- GPU memory allocation: Pre-allocate based on enabled features
- Session management: One ONNX session per model
- Error handling: Graceful fallback if model missing/corrupt

### Model Input/Output Specifications
**Note**: Exact tensor shapes and preprocessing need to be validated from model metadata.

Expected patterns (verify with actual models):
- **Detectors**: Input [1, 3, H, W], Output: [N, 6] (x, y, w, h, confidence, class)
- **Face Landmarks**: Input [1, 3, 192, 192], Output: [1, 468, 3] (x, y, z)
- **Hand Landmarks**: Input [1, 3, 224, 224], Output: [1, 21, 3] (x, y, z)
- **Pose Landmarks**: Input [1, 3, 256, 256], Output: [1, 33, 5] (x, y, z, visibility, presence)
- **Face Blendshapes**: Input [face_landmarks], Output: [1, 52] (blendshape coefficients)

### Landmark Topology (for skeleton drawing)
**Face**: 468 landmarks (MediaPipe Face Mesh topology)
- Eyes, eyebrows, nose, mouth, face oval
- Use MediaPipe's predefined connection indices

**Hands**: 21 landmarks per hand
- Wrist (0), Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
- Connections: Wrist to all finger bases, then sequential finger joints

**Body**: 33 landmarks
- Pose (0-24): Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, etc.
- Hands/Face (25-32): Additional points for hand/face alignment
- Use MediaPipe Pose connection topology

## Success Criteria
- ✅ Smooth real-time display (>30 FPS)
- ✅ Clean, responsive CLI interface
- ✅ GPU memory stays <4GB on RTX 3060
- ✅ CPU usage <20% during operation
- ✅ All command-line options functional
- ✅ Robust error handling
- ✅ No frame drops or stuttering
- ✅ ROI visualization works correctly
- ✅ Multi-face, multi-hand detection works
- ✅ Party-ready: fun, fast, reliable!

## Non-Requirements (Out of Scope)
- ❌ Data persistence (SQLite, CSV, JSON export)
- ❌ Recording/playback functionality
- ❌ Web interface or REST API
- ❌ Multi-camera support
- ❌ Gesture recognition beyond landmarks
- ❌ Network streaming
- ❌ Configuration files (CLI only)
- ❌ Custom model training/fine-tuning

## Dependencies to Avoid
- ❌ TensorFlow (version conflicts)
- ❌ MediaPipe (dependency hell with TF)
- ❌ PyTorch (unnecessary overhead)

## Key Dependencies
- ✅ OpenCV with CUDA support (cv2.cuda)
- ✅ ONNX Runtime GPU (onnxruntime-gpu)
- ✅ NumPy
- ✅ argparse (stdlib)

## Error Handling
- Missing model files: Clear error message, list expected models
- Camera access failure: Graceful exit with troubleshooting tips
- GPU/CUDA unavailable: Fallback to CPU with performance warning
- Invalid arguments: Help text and usage examples
- Model inference errors: Skip frame, log warning, continue
- ROI extraction failures: Skip that detection, continue with others

## Development Notes
- Use type hints throughout
- Comprehensive docstrings
- Clear variable naming (no abbreviations in critical paths)
- Profile GPU memory usage during development
- Test on both RTX 5080 and RTX 3060
- Validate FPS on different resolution settings
- Test with multiple faces/hands in frame
- Verify ROI coordinate transformations are correct
- Document MediaPipe model tensor formats

## MediaPipe Model Documentation References
- Input preprocessing: Normalize to [0, 1] or [-1, 1]? (verify from model)
- BGR vs RGB: MediaPipe typically expects RGB
- Coordinate system: Normalized [0, 1] or pixel coordinates?
- Visibility/presence scores: How to interpret and use
