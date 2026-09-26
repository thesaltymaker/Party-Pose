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
- RTX 4070 mobile: ~31 FPS at 1080p
- With TensorRT: +20-30% improvement

Jetson Orin Nano Super (`python poser.py --platform orin --fps`, 1080p CSI camera, all models on TensorRT):
- ~30 FPS with one detected person, 20–25 FPS with two. Each extra detected body costs about 9 ms/frame.
- Needs `sudo jetson_clocks` (resets on reboot). Without it the GPU governor downclocks to about 408 MHz and inference runs about 2× slower.
- `--fps` also prints per-stage timings (`[STAGES ms/frame]`), detections per frame and each model's execution provider (`[PROVIDERS]`).
- Benchmark tools: `tools/orin_bench/`. Details: `docs/orin-fps-issue-2.md`.

## Jetson Orin setup and deployment

- The Orin runs the modular app, `poser.py` with `src/`. Its Python environment is built by `tools/orin_setup/bootstrap_env.sh`, not by `requirements.txt`. See `tools/orin_setup/README.md`.
- Deploy only through git: push from the laptop, `git pull` on the Orin. Never copy the repo directory onto the Orin.
- Enable the pre-push hook once per clone (`git config core.hooksPath tools/git-hooks`). Every push then takes a snapshot of the Orin's environment, models and camera settings first.

## Debugging detections

```bash
python poser.py --platform orin --fps --dump-detections /tmp/ppdump --dump-every 1
```

Every `--dump-every` seconds this saves the frame with the overlay and the detector boxes and scores drawn, the raw frame, and a JSON of detector scores and landmark presence scores. It was used to find and fix the false positives in issue #3; see `docs/orin-false-positives-issue-3.md`.

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
- Jetson Orin: run `sudo jetson_clocks`, then check `[STAGES ms/frame]` and `[PROVIDERS]` in the `--fps` output (see `tools/orin_bench/README.md`)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the pose and hand models
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [InsightFace](https://github.com/deepinsight/insightface) for SCRFD face detection
