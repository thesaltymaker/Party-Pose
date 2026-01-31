# Party-Pose

Real-time holistic body tracking using ONNX Runtime with GPU acceleration. Tracks body pose (33 landmarks), face mesh (478 landmarks), and hand landmarks (21 per hand) simultaneously.

![Party-Pose Demo](screenshots/demo.png)
<!-- Replace screenshots/demo.png with your actual screenshot -->

## Features

- **Body Pose Detection**: Upper body tracking with shoulders, arms, and torso
- **Face Mesh**: 478-point facial landmark tracking with optional "skelly" filled mode
- **Hand Tracking**: Dual hand detection with 21 landmarks per hand
- **GPU Accelerated**: CUDA and optional TensorRT support via ONNX Runtime
- **Multi-person**: Supports tracking multiple people simultaneously
- **Santa Hat Mode**: Fun holiday overlay feature

## Screenshots

| Normal Mode | Skelly Mode |
|-------------|-------------|
| ![Normal](screenshots/normal.png) | ![Skelly](screenshots/skelly.png) |

<!--
To add your screenshots:
1. Create a 'screenshots' folder: mkdir screenshots
2. Take screenshots and save them as:
   - screenshots/demo.png (main hero image)
   - screenshots/normal.png (normal tracking mode)
   - screenshots/skelly.png (skelly mode with filled face)
3. Recommended size: 640x480 or 1280x720
-->

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Webcam

## Installation

```bash
# Clone the repository
git clone https://github.com/thesaltymaker/Party-Pose.git
cd Party-Pose

# Create and activate virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # Linux/Mac
# or: env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Model Files

Download the required ONNX models and place them in the `models/` directory:

```
models/
├── pose_detection_128x128_float32.onnx
├── pose_landmark_full_body.onnx
├── det_2.5g.onnx (SCRFD face detection)
├── face_landmark_192x192.onnx
├── MediaPipeHandDetector.onnx
└── hand_landmark.onnx
```

## Usage

```bash
# Basic usage (default camera)
python Party-Pose.py

# Specify camera device
python Party-Pose.py --camera 0

# Set resolution
python Party-Pose.py --width 1920 --height 1080

# Enable TensorRT acceleration
python Party-Pose.py --tensorrt

# Mirror output (selfie mode)
python Party-Pose.py --mirror

# Enable skelly mode (filled face)
python Party-Pose.py --skelly

# Disable specific features (saves computation)
python Party-Pose.py --no-pose
python Party-Pose.py --no-face
python Party-Pose.py --no-hands

# Show pose ROI debug boxes
python Party-Pose.py --show-pose-roi

# Holiday fun
python Party-Pose.py --santa-hat
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--camera` | Camera device index (default: 0) |
| `--width` | Camera capture width (default: 1920) |
| `--height` | Camera capture height (default: 1080) |
| `--display-width` | Display window width |
| `--display-height` | Display window height |
| `--tensorrt` | Enable TensorRT acceleration |
| `--mirror` | Mirror the output horizontally |
| `--skelly` | Enable filled face skeleton mode |
| `--santa-hat` | Draw Santa hats on detected faces |
| `--no-pose` | Disable body pose tracking |
| `--no-face` | Disable face mesh tracking |
| `--no-hands` | Disable hand tracking |
| `--no-debug` | Hide FPS and debug info |
| `--show-pose-roi` | Show pose detection bounding boxes |

## Configuration

Key parameters can be adjusted at the top of `Party-Pose.py`:

```python
# Detection thresholds
POSE_CONF_THR = 0.7      # Pose detection confidence
POSE_IOU_THR = 0.2       # Pose NMS IoU threshold
POSE_ROI_SCALE = 2.0     # ROI expansion factor (downward)
FACE_SCORE_THR = 0.50    # Face detection threshold
HAND_SCORE_THR = 0.85    # Hand detection threshold
```

## Controls

- Press `q` or `ESC` to quit

## Architecture

The pipeline processes each frame through:

1. **Face Detection** (SCRFD) → Face Mesh landmarks
2. **Body Detection** (BlazePose) → Body landmarks
3. **Hand Detection** (MediaPipe Palm) → Hand landmarks

Hand detections that overlap with face regions are automatically suppressed to prevent false positives.

## Performance

Typical performance on NVIDIA GPUs:
- GTX 1080: ~30 FPS at 1080p
- RTX 3070: ~45 FPS at 1080p
- With TensorRT: +20-30% improvement

## Troubleshooting

**Camera not detected:**
```bash
# List available cameras
ls /dev/video*
# Try different camera index
python Party-Pose.py --camera 1
```

**CUDA not available:**
```bash
# Check ONNX Runtime providers
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**Low FPS:**
- Reduce resolution: `--width 1280 --height 720`
- Disable unused features: `--no-hands` or `--no-face`
- Enable TensorRT: `--tensorrt`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the pose and hand models
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [InsightFace](https://github.com/deepinsight/insightface) for SCRFD face detection
