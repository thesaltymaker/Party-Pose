# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Activate environment (if not already activated)
source env/bin/activate

# Run the application
python poser.py [flags]

# Common flags:
#   --mirror          : Mirror output (selfie mode)
#   --no-face         : Disable face tracking
#   --no-hands        : Disable hand tracking  
#   --show-roi        : Show detection bounding boxes
#   --fps             : Show FPS counter
#   --camera N        : Use camera device N (default: 0)
#   --width W         : Set capture width (default: 1920)
#   --height H        : Set capture height (default: 1080)

# Press 'q' or ESC to quit
```

## Development Workflow

### Running Tests & Validation
There is no formal test suite, but you can validate functionality:

```bash
# Basic functionality test
python poser.py --width 640 --height 480 --duration 5  # Add --duration flag if implemented

# Check model availability
ls models/  # Should contain required ONNX models

# Validate CUDA availability
python -c "import onnxruntime; print('Providers:', onnxruntime.get_available_providers())"

# Run diagnostic scripts
python opencv_check.py      # Check OpenCV CUDA support
python test_env.py          # Test environment setup
```

### Common Debug Commands
```bash
# List available cameras
ls /dev/video*

# Check ONNX Runtime session options
python -c "import onnxruntime as ort; sess = ort.InferenceSession('models/face_detector.onnx'); print(sess.get_providers())"

# Profile performance (add timing to poser.py if needed)
```

## High-Level Architecture

Party-Pose follows a GPU-resident pipeline architecture designed for real-time performance (>30 FPS). The system avoids CPU-GPU round-trips by keeping frames as `cv2.cuda.GpuMat` from capture through processing.

### Core Pipeline Flow
```
Webcam → GPU Upload → [Mirror Flip] → 
Face Processor → Hand Processor → Body Processor → 
Renderer (single GPU↔CPU transfer) → Display
```

### Key Architectural Decisions

1. **GPU Residency Principle**: Frames remain as `cv2.cuda.GpuMat` objects throughout processing. Only the final frame is downloaded to CPU for display via `cv2.imshow()`.

2. **Two-Stage Detection Strategy**:
   - **Face**: Lightweight detector (128×128) → ROI crop → Landmark model (256×256) → Optional blendshapes (146×2)
   - **Hands**: Detector (192×192) → ROI crop → Landmark model (224×224) 
   - **Body**: Single-stage landmark model (256×256) - no separate detector needed

3. **Modular Design (`src/` directory)**:
   - `model_manager.py`: Lazy loads ONNX sessions only for requested modalities
   - `preprocessor.py`: Handles GPU operations (resize, crop, color conversion, normalization)
   - `*_processor.py`: Face, hand, and body processing modules
   - `renderer.py`: Draws landmarks and connections on CPU frame (single download/upload per frame)
   - `types.py`: Dataclasses for bounding boxes and landmarks
   - `config.py`: Parses CLI args into immutable configuration
   - `fps_counter.py`: Rolling-window FPS calculation
   - `topology.py`: MediaPipe connection constants (requires fixing - see FIXES_NEEDED.md)
   - `person_detector.py`: Handles person-level detection and association
   - `video_capture.py`: Manages webcam capture and GPU upload

### Model Specifications
All models expect NHWC float32 input in range [0,1]:
- `face_detector`: [1, 128, 128, 3] → regressors + classificators (stride 8/16 anchors)
- `face_landmarks_detector`: [1, 256, 256, 3] → 478 landmarks × 3 coords
- `face_blendshapes`: [1, 146, 2] → 52 expression coefficients
- `hand_detector`: [1, 192, 192, 3] → regressors [2016,18], classificators [2016,1]
- `hand_landmarks_detector`: [1, 224, 224, 3] → 21 landmarks (flat [1,63]), handedness, presence
- `pose_landmarks_detector`: [1, 256, 256, 3] → 33 landmarks × 5 (x,y,z,vis,presence), segmentation

### Critical Implementation Notes

1. **Coordinate Systems**: 
   - Preprocessor outputs convert landmarks to **pixel space** (not normalized [0,1])
   - Renderer expects pixel-space coordinates - **do not** multiply by frame dimensions
   - BoundingBox uses (x, y, w, h) format in pixel space, not corner coordinates

2. **Detection Decoding**:
   - Face/hand detectors use SSD-style anchor grids with strides 8 and 16
   - Outputs decoded relative to normalized anchor centers, then scaled to pixel coords
   - NMS filters overlapping boxes using IoU threshold

3. **Person Association Logic**:
   - Head boxes associated to bodies by centroid containment
   - Hand boxes associated to bodies by centroid within body bbox (± margins)
   - Each result gets a `person_id` for tracking across modalities

### Known Issues & Fixes Needed

Refer to `FIXES_NEEDED.md` for 15 specific bugs requiring attention. Key areas:

1. **topology.py**: 
   - `FACE_CONNECTIONS` empty - needs MediaPipe FaceMesh tessellation pairs
   - `CANONICAL_BLENDSHAPE_INDICES` empty - needs 146 landmark indices for blendshapes

2. **config.py**:
   - `--fps` flag reads `args.show_fps` but stores in `args.fps`
   - `store_false` args (`--no-face`, etc.) default to `None` instead of `True`

3. **model_manager.py**:
   - CUDA provider options incorrectly passed as `SessionOptions` object instead of dict

4. **Processor Files**:
   - `model_manager.run_session()` calls don't exist - use `get_session().run()`
   - Incorrect bounding box field names (use x,y,w,h not x1,y1,x2,y2)
   - Anchor grid calculations use stride instead of grid size
   - ROI bbox not properly unpacked from `crop_roi_nhwc`
   - Landmark coordinates already in pixel space - no additional scaling needed

5. **renderer.py**:
   - 4 separate GPU↔CPU transfers per frame - should download once, draw all, upload once
   - Incorrect coordinate scaling in drawing functions

### Performance Characteristics

**Targets**:
- >30 FPS on RTX 3060, >60 FPS on RTX 5080
- GPU memory <4 GB, CPU usage <20%
- Detector inference <5ms, landmark inference <10ms per ROI

**Optimization Techniques**:
- Lazy model loading (only load requested modalities)
- Single GPU↔CPU transfer per frame for rendering
- ROI-based processing (landmark models run only on detected regions)
- CUDA Execution Provider for ONNX Runtime
- OpenCV CUDA operations for frame preprocessing

### File Organization

- `poser.py`: Main application entry point (modular refactor)
- `Party-Pose.py`: Original monolithic implementation (do not modify)
- `src/`: Core modules (see above)
- `models/`: Directory for ONNX model files
- `screenshots/`: Demo screenshots
- `*.md`: Documentation files
- `requirements.txt`: Python dependencies
- `activate.sh`: Environment activation helper
- `FIXES_NEEDED.md`: List of bugs to fix (priority for development)
- `poser_crew.py`: CrewAI multi-agent orchestration for LLM-driven code generation

### Getting Started with Development

1. **Fix Critical Issues First**: Address items in `FIXES_NEEDED.md` starting with topology.py and config.py
2. **Verify Model Availability**: Ensure required ONNX models are in `models/` directory
3. **Test Incrementally**: Make small changes and test with `python poser.py --width 640 --height 480`
4. **Monitor Performance**: Watch for FPS drops indicating regressions
5. **Validate Fixes**: Ensure landmarks render correctly in proper positions

### Important Constraints

- **Do not modify** `Party-Pose.py` - it's the reference implementation
- Focus development on `poser.py` and `src/` modules
- Maintain GPU residency principle - minimize CPU↔GPU transfers
- Keep configuration immutable after startup
- All model inputs must be NHWC float32 [0,1]
- Renderer operates on CPU frame after single download from GPU