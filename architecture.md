# Poser — System Architecture

## Current pipeline (2026-10, issue #13)

Everything below section a. describes the original design, where each processor ran its own SSD detector. This section is how `poser.py` and `src/` work now. Measurements behind each choice: `docs/issue-13-party-readiness.md`, `docs/lessons-learned.md`, `docs/orin-performance-tuning.md`.

### Threads

| Thread | Work | Notes |
|---|---|---|
| `clip-reader` (`--video` only) | reads the raw clip from disk one frame ahead | drops each frame from the file cache (shared CPU/GPU memory on the Orin) |
| `poser-infer` (`--pipeline`) | `infer()`: read frame, upload to the GPU, all models, tracking, smoothing, ghosts | one-slot queue to the main thread |
| main | `render()`: draw on the CPU (one GPU download, or a black canvas with `--black-bg`), `imshow` | GTK windows must stay in one thread |
| `poser-detect` (`--detect-ahead` only) | reads frame N+1 and runs the person detector while `poser-infer` works on frame N | not used in the service: the Orin GPU is shared and the other models slow down |

### One frame in `infer()`

1. **Capture**: `VideoCaptureModule.read_frame()` uploads into a `GpuMat` (a ring of 6 with `--detect-ahead`), applies white balance on the Orin, and flips it with `--mirror` (all drawing follows the flipped picture).
2. **Person detector** (`PersonDetector`, YOLOX tiny 288x480 with NMS inside, `--detector FILE --detector-bgr --detector-area`): body, head and hand boxes. With `--detect-every N` it runs only every Nth frame while every live, non-still person was followed by their pose last frame (`_detect_now`); otherwise every frame.
3. **Following** (`--no-follow` turns it off): each confirmed track with a pose last frame gets the pose model on the crop from that pose, with or without a detector box (`body_rois`). Detector boxes mostly inside a followed person start no second track (`_follow_boxes`, `_boxes_not_followed`).
4. **Tracking** (`PersonTracker`): stable IDs from box IoU and centre distance with velocity; confirmed after 3 matches, kept 15 missed frames, revived within 5 s. `--drop-still`: a track whose box centres hardly moved over ~10 s (a prop such as the lamp) gets no models. `--min-body-height 0.40`: people far back are not drawn.
5. **Pose** (`BodyProcessor`, BlazePose full 256x256): for people not followed yet, on the detector box. With `--pose-batch N` all followed people run in one model call.
6. **Face** (`FaceProcessor`, 256x256, 478 points): the crop is a square around the pose's face points (0-10), not the head box (head boxes are missing in ~1/3 of frames). `--face-keep LOGIT` keeps a face found last frame down to a lower face logit while the head is not turned away. Faces run in batches (`--face-batch N`, or a fixed batch of 2).
7. **Hands** (`HandProcessor`, 224x224, 21 points): cropped from the hand's last landmarks, else from the pose's wrist and finger points. A hand is kept only if its wrist is within 0.5 shoulder widths of the person's pose wrist. `--hand-batch N`: all hands in batched calls.
8. **Every Nth frame per person** (`--hand-every N`, `--face-every N`, `ghosts.skip_turn`): a part found last frame skips its model on N-1 of N frames; people take turns, and a person's face runs on the frames their hands skip, so each frame has about the same number of model calls. A part not found last frame always runs.
9. **Smoothing** (`PoseSmoother`): One Euro filter per landmark on the clip's or camera's clock (`--smooth-body/face/hand MIN_CUTOFF,BETA`); `--score-smooth ALPHA` averages body joint presence/visibility so one-frame dips don't hide joints. `JointHold` (`--hold-joints`) can hold lost joints (off).
10. **Ghosts** (`Ghosts`, `--ghost-frames N`): a part missing this frame is redrawn from its last smoothed result, dimming over N frames, moved with the head (faces) or the pose wrist (hands, on skipped frames). A part skipped on purpose (step 8) is redrawn at full brightness. Moved ghost faces are a lazy sequence (`_MovedLandmarks`).
11. Counters for replays: `[STAGES]` ms and counts per frame, `[FLICKER]` blinks (gaps of 1-5 frames per part), `[MOTION]` pops/jitter/lag (`--motion-stats`), `[NEW-ID]`, `[BENCH]`.

### GPU residency and model sessions

- Crops never leave the GPU: `GpuModel` (one crop), `GpuBatch` (n crops, one TensorRT session per batch size n), `GpuFaceBatch`, `GpuDetectorInput` (`src/gpu_model.py`) write the model input into a fixed continuous GPU buffer bound with ONNX Runtime I/O binding, and outputs go to fixed GPU `OrtValue`s.
- On the Orin the landmark sessions use TensorRT FP16 with CUDA graphs: each session records its kernel launches once and replays them, which needs fixed buffer addresses and one batch size per session.
- `_load_sessions()` loads every session (detector, landmark models, each batch size up to N) before the first frame. Loading one on first use froze the picture for 1-3 s.
- `run_bound()` holds `FIRST_RUNS_LOCK` for a session's first 3 runs (when its CUDA graph is recorded); the `--detect-ahead` thread holds it for each frame, because a GPU allocation in another thread breaks a recording in CUDA's global capture mode.
- OpenCV's CUDA calls run on the default stream and ONNX Runtime on its own: every bound run waits for OpenCV first (`waitForCompletion`).

### Service settings (Orin, `tools/orin_setup/service/party-pose.env.example`)

`--platform orin --fullscreen --fps --mirror --drop-still --black-bg --skelly --pipeline --detector yolox_t_body_head_hand_post_0299_0.4522_1x3x288x480_nhwc.onnx --detector-bgr --detector-area --face-keep -8 --score-smooth 0.05 --ghost-frames 8 --smooth-body 0.3,0.5 --smooth-face 0.3,0.5 --smooth-hand 0.3,0.5 --pose-batch 4 --hand-batch 4 --face-batch 4 --detect-every 3 --hand-every 2 --face-every 2`

New options are off by default in the code, so tests and older commands keep their behaviour; the service file holds the chosen values.

## a. System Architecture

### Overview

Poser is a single-process, real-time GPU pipeline that reads webcam frames, runs pose inference via ONNX Runtime, and renders landmark overlays back to the screen at >30 FPS. All configuration is supplied at startup via CLI flags; there is no runtime state persistence.

### Top-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MAIN LOOP  (poser.py)                              │
│                                                                             │
│   VideoCaptureModule                                                        │
│          │  GpuMat (BGR)                                                    │
│          ▼                                                                  │
│   [Mirror Flip — GPU]  (optional, --mirror)                                 │
│          │  GpuMat (BGR)                                                    │
│          ├─────────────────────────────────────┐                            │
│          │                                     │                            │
│   FaceProcessor     HandProcessor    BodyProcessor                          │
│   (if --face)       (if --hands)     (if --body)                            │
│          │               │                │                                 │
│          ▼               ▼                ▼                                 │
│   List[FaceResult]  List[HandResult]  Optional[BodyResult]                  │
│          └───────────────┴────────────────┘                                 │
│                          │                                                  │
│                     Renderer                                                │
│                          │  GpuMat (BGR + overlays)                         │
│                          ▼                                                  │
│                    cv2.imshow()                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Description |
|---|---|
| GPU Residency | Frames live as GpuMat from capture through display. CPU transfer is limited to the minimal array required for ONNX inference input. |
| Lazy Loading | ModelManager opens ONNX sessions only for the modalities requested at startup. |
| Two-Stage Detection | Lightweight SSD detectors (BlazeFace / BlazePalm) run on the full frame; heavy landmark models run only on the small expanded ROI crop. |
| Single-Stage Body | pose_landmarks_detector runs on the full frame — no separate body detector is required. |
| Immutable Config | Config is built once from argparse at startup and never mutated. |

---

## b. Core Components

### 1. VideoCaptureModule
Owns the camera handle. Reads raw BGR frames and uploads them to GPU memory. Applies a GPU-side horizontal flip when mirror mode is active.

### 2. Config
Frozen value object populated by argparse. Carries all user preferences: enabled modalities, camera index, resolution, and display flags. Passed by reference to all processors.

### 3. ModelManager
Single source of truth for ONNX InferenceSession objects. Performs lazy loading — a session is created only the first time it is requested. Validates that all required model files exist before the camera is opened, raising a descriptive error listing every missing file.

### 4. Preprocessor
Stateless utility component. Handles all frame transformation prior to inference: BGR→RGB conversion on GPU, ROI crop on GPU with configurable padding, resize to model input dimensions, and conversion to a normalized float32 array in NHWC layout.

### 5. FaceProcessor
Orchestrates the three-stage face pipeline: detection → landmarks → optional blendshapes. Returns one FaceResult per detected face.

### 6. HandProcessor
Orchestrates the two-stage hand pipeline: detection → landmarks. Returns one HandResult per detected hand, up to two hands.

### 7. BodyProcessor
Runs the single-stage pose landmark model on the full frame. Returns a BodyResult when the model's presence score meets the threshold, otherwise None.

### 8. Renderer
Draws all overlays directly onto the GpuMat using OpenCV CUDA drawing operations. Reads skeleton connection topology from topology.py. Owns all color constants. Draws ROI bounding boxes (--show-roi) and FPS counter (--fps) when those flags are active.

### 9. FPSCounter
Maintains a rolling window of frame timestamps. Returns the rolling-window average FPS.

### 10. topology.py
Module-level constants only: MediaPipe connection index lists for face mesh (478 landmarks), hand skeleton (21 landmarks), and body skeleton (39 landmarks). Imported directly by Renderer.

---

## c. Data Flow

### Frame Lifecycle

```
Camera (V4L2 / USB)
  │  Raw BGR frame  (CPU memory)
  ▼
cv2.VideoCapture.read()
  │  Upload to GPU
  ▼
GpuMat (BGR) ──────────► [cv2.cuda.flip — if --mirror]
  │
  ├─── FaceProcessor ──────────────────────────────────────────────────┐
  │      │                                                             │
  │      ├─ Full frame → resize 128×128 → normalize                   │
  │      ├─ face_detector inference                                    │
  │      │       regressors    [1, 896, 16]                            │
  │      │       classificators [1, 896, 1]                            │
  │      ├─ Anchor decode + sigmoid + NMS → BoundingBox list           │
  │      └─ Per detected face:                                         │
  │             Crop + pad ROI on GPU → resize 256×256 → normalize     │
  │             face_landmarks inference                               │
  │                 Identity      [N, 1, 1, 1434]  478 landmarks × 3   │
  │                 Identity_1    [N, 1, 1]         presence score      │
  │                 Identity_2    [N, 1]            face score          │
  │             Transform landmarks → image pixel coords               │
  │             [if --face-expressions]                                │
  │             face_blendshapes inference                             │
  │                 input:  146 landmark subset  [1, 146, 2]           │
  │                 output: blendshape coeffs    [52]                  │
  │                                                                    │
  ├─── HandProcessor ──────────────────────────────────────────────────┤
  │      │                                                             │
  │      ├─ Full frame → resize 192×192 → normalize                   │
  │      ├─ hand_detector inference                                    │
  │      │       Identity    [1, 2016, 18]                             │
  │      │       Identity_1  [1, 2016,  1]                             │
  │      ├─ Anchor decode + NMS → BoundingBox list (max 2 hands)       │
  │      └─ Per detected hand:                                         │
  │             Crop + pad ROI on GPU → resize 224×224 → normalize     │
  │             hand_landmarks inference                               │
  │                 Identity    [1, 63]   21 landmarks × 3 (screen)    │
  │                 Identity_1  [1,  1]   handedness score             │
  │                 Identity_2  [1,  1]   presence score               │
  │                 Identity_3  [1, 63]   world coords (not rendered)  │
  │                                                                    │
  └─── BodyProcessor ──────────────────────────────────────────────────┘
         │
         ├─ Full frame → resize 256×256 → normalize
         ├─ pose_landmarks inference
         │       Identity    [1, 195]           39 landmarks × 5 values
         │       Identity_1  [1,   1]           body presence score
         │       Identity_2  [1, 256, 256, 1]   segmentation mask (unused)
         │       Identity_4  [1, 117]           world coords (unused)
         └─ If presence < 0.5 → return None

Results: List[FaceResult], List[HandResult], Optional[BodyResult]
  │
  ▼
Renderer.draw_faces / draw_hands / draw_body / draw_fps
  │  GpuMat (BGR + overlays)
  ▼
cv2.imshow()  ←  GpuMat.download() to CPU for display window
```

### Coordinate Transformation Chain

All models output coordinates in **pixel space of their input tensor** (e.g., values in 0–256 for a 256×256 input). Coordinates must be transformed back to image space before rendering.

**Detector output → image pixel coordinates (full-frame inference):**
```
x_image = ( x_model / model_input_width  ) × frame_width
y_image = ( y_model / model_input_height ) × frame_height
```

**Landmark output within ROI → image pixel coordinates:**
```
x_norm  = x_model / model_input_width        (normalized within crop)
y_norm  = y_model / model_input_height

x_image = crop_bbox.x  +  x_norm × crop_bbox.width
y_image = crop_bbox.y  +  y_norm × crop_bbox.height
```

**Mirror correction (applied last, before drawing):**
```
x_display = frame_width − x_image
y_display = y_image
```

### ROI Expansion

When cropping a detected bounding box for landmark inference, expand it on each side and clamp to frame boundaries. The padding differs per model: face 0 (the head box already includes hair; 25% padding pushed real faces' scores near the rejection threshold, see issue #3), body 0.1 (letterboxed), hand 0.8. The formula below shows 25%.

```
pad_x      = bbox.width  × 0.25
pad_y      = bbox.height × 0.25
crop_x     = max(0,            bbox.x − pad_x)
crop_y     = max(0,            bbox.y − pad_y)
crop_width = min(frame_width  − crop_x,  bbox.width  + 2 × pad_x)
crop_height= min(frame_height − crop_y,  bbox.height + 2 × pad_y)
```

### Event State Definitions

#### Visibility State — Body Landmarks Only

The pose model Identity output encodes 39 landmarks × 5 values: `x, y, z, visibility, presence`. Visibility and presence are sigmoid-activated in [0, 1].

| Condition | Rendering Action |
|---|---|
| `presence < 0.5` | Landmark not in frame — skip entirely, skip connected skeleton segments |
| `visibility < 0.5` | Landmark occluded — draw at 30% opacity |
| `visibility ≥ 0.5` | Landmark visible — draw normally |

Threshold rationale: 0.5 is the industry-standard midpoint for sigmoid-based confidence; it balances false positives (jitter from low-confidence draws) against false negatives (missing valid landmarks).

#### Out of Bounds State — All Modalities

A landmark is **out of bounds** when its normalized image-space position falls outside the padded boundary **[−0.05, 1.05]**. The 0.05 buffer accommodates legitimate edge-of-frame detections without drawing off-screen artefacts. Out-of-bounds landmarks are not drawn; skeleton segments connecting to them are also skipped.

```
is_out_of_bounds  ←  x_norm < −0.05  OR  x_norm > 1.05
                   OR  y_norm < −0.05  OR  y_norm > 1.05

where  x_norm = landmark.x / frame_width
       y_norm = landmark.y / frame_height
```

#### SSD Anchor Decoding — Both Detectors

BlazeFace and BlazePalm use SSD-style anchor grids. The regressor output encodes offsets relative to pre-computed anchor centres.

```
confidence = sigmoid( classificator[i] )

center_x = regressor[i, 0] / input_size  +  anchor_center_x
center_y = regressor[i, 1] / input_size  +  anchor_center_y
width    = regressor[i, 2] / input_size
height   = regressor[i, 3] / input_size
```

| Parameter | Face Detector | Hand Detector |
|---|---|---|
| Input size | 128 | 192 |
| Anchor count | 896 | 2016 |
| Acceptance threshold | ≥ 0.5 | ≥ 0.5 |
| NMS IoU threshold | 0.3 | 0.3 |

---

## d. Technical Implementation Details

### ONNX Runtime Session Configuration

Each InferenceSession is created with:
- Graph optimization level: **ORT_ENABLE_ALL** — enables all offline and online graph optimizations
- Memory pattern: **enabled** — allows runtime to reuse memory allocations across inference calls
- Intra-op thread count: **1** — the GPU does the computation; excess CPU threads add contention without benefit
- Execution provider order: **CUDAExecutionProvider** first, **CPUExecutionProvider** as fallback

CUDAExecutionProvider options:
- `arena_extend_strategy`: kNextPowerOfTwo — grows GPU arena in power-of-two steps to reduce fragmentation
- `gpu_mem_limit`: 2 GB per session — prevents a single model from exhausting VRAM
- `cudnn_conv_algo_search`: EXHAUSTIVE — one-time search at session creation; best algorithm is cached
- `do_copy_in_default_stream`: True — synchronizes copies on the default CUDA stream to avoid race conditions

On `--platform orin`, TensorrtExecutionProvider is placed first (FP16, engine cache in `models/trt_cache`), with CUDA and CPU as fallbacks. Per-model TensorRT options live in `ModelManager._TRT_EXTRA_OPTS`. The person detector uses them to keep its NMS post-processing off TensorRT while the backbone runs in TensorRT (see `docs/orin-fps-issue-2.md`).

### Actual Model Tensor Specifications

All models use **NHWC float32** input in **RGB** channel order, normalized to **[0.0, 1.0]**.

| Model | Input Name | Input Shape | Key Outputs |
|---|---|---|---|
| face_detector_opset15.onnx | `input` | `[1, 128, 128, 3]` | `regressors [1,896,16]`  `classificators [1,896,1]` |
| face_landmarks_detector_opset15.onnx | `input_12` | `[N, 256, 256, 3]` | `Identity [N,1,1,1434]` (478 lm×3)  `Identity_1 [N,1,1]` presence  `Identity_2 [N,1]` score |
| face_blendshapes_opset15.onnx | `serving_default_input_points:0` | `[1, 146, 2]` | `StatefulPartitionedCall:0 [52]` |
| hand_detector_opset15.onnx | `input_1` | `[1, 192, 192, 3]` | `Identity [1,2016,18]`  `Identity_1 [1,2016,1]` |
| hand_landmarks_detector_opset15.onnx | `input_1` | `[1, 224, 224, 3]` | `Identity [1,63]` (21 lm×3)  `Identity_1 [1,1]` handedness  `Identity_2 [1,1]` score  `Identity_3 [1,63]` world coords |
| pose_landmarks_detector_opset15.onnx | `input_1` | `[1, 256, 256, 3]` | `Identity [1,195]` (39 lm×5)  `Identity_1 [1,1]` presence  `Identity_2 [1,256,256,1]` seg mask  `Identity_4 [1,117]` world coords |

> **Landmark count corrections vs. spec**: The face landmarks model outputs **478** landmarks (MediaPipe v2 face mesh), not 468. The pose model outputs **39** landmarks (33 body + 6 face/hand auxiliary alignment points); use indices 0–32 for skeleton drawing.

### Output Tensor Reshaping

| Model | Raw Output Shape | Reshape Target | Interpretation |
|---|---|---|---|
| face_landmarks | `[N, 1, 1, 1434]` | `[N, 478, 3]` | x, y, z in [0, 256] input pixels |
| hand_landmarks | `[1, 63]` | `[21, 3]` | x, y, z in [0, 224] input pixels |
| pose_landmarks | `[1, 195]` | `[39, 5]` | x, y, z, visibility, presence |

### Blendshapes Input Preparation

The blendshapes model expects **146 specific face landmarks** (a canonical subset of the 478 total), supplied as normalized (x, y) pairs in [0, 1] within the 256×256 landmark input space. The 146 canonical indices are defined as a constant in `src/topology.py`.

Preparation steps:
1. Select the 146 rows from the [478, 3] landmark array using the canonical index list
2. Take only columns 0 and 1 (x, y; discard z)
3. Divide by 256.0 to normalize from model pixel space to [0, 1]
4. Add batch dimension → shape [1, 146, 2]

### Preprocessing Pipeline (Per Inference Call)

1. **BGR → RGB** — GPU color conversion (cv2.cuda.cvtColor)
2. **Resize** — GPU resize to model input dimensions (cv2.cuda.resize)
3. **Download** — Minimal GPU→CPU transfer for ONNX Runtime input
4. **Normalize** — Divide by 255.0 on CPU → float32 in [0.0, 1.0]
5. **Add batch dim** — HWC → NHWC

### GPU Memory Budget

| Component | Estimated VRAM |
|---|---|
| Frame buffers (1280×720 BGR, ×3) | ~9 MB |
| face_detector session | ~30 MB |
| face_landmarks session | ~50 MB |
| face_blendshapes session | ~20 MB |
| hand_detector session | ~40 MB |
| hand_landmarks session | ~80 MB |
| pose_landmarks session | ~100 MB |
| **Total (all modalities enabled)** | **~330 MB** |

Well within the 4 GB RTX 3060 constraint.

### CPU Fallback Behaviour

If CUDAExecutionProvider is unavailable at session creation, ModelManager logs a warning, drops to CPUExecutionProvider, and disables cv2.cuda operations in favor of standard OpenCV. FPS will be significantly lower. The warning is printed to stderr before the main loop starts.

### Rendering Color Scheme

All colors are BGR tuples for OpenCV drawing functions.

| Element | Color | BGR Value |
|---|---|---|
| Face landmarks | Green | `(0, 255, 0)` |
| Face ROI box | Cyan | `(255, 255, 0)` |
| Left hand landmarks | Yellow | `(0, 255, 255)` |
| Right hand landmarks | Orange | `(0, 165, 255)` |
| Hand ROI box | Muted yellow | `(0, 200, 200)` |
| Body landmarks | Blue | `(255, 0, 0)` |
| Body skeleton | Dark blue | `(200, 50, 50)` |
| FPS counter text | Green | `(0, 255, 0)` |

---

## e. Component Interfaces

### Result Data Types

**BoundingBox**

| Field | Type | Description |
|---|---|---|
| x | float | Top-left x in image pixel space |
| y | float | Top-left y in image pixel space |
| w | float | Width in pixels |
| h | float | Height in pixels |
| confidence | float | Detection confidence [0, 1] |

**Landmark**

| Field | Type | Description |
|---|---|---|
| x | float | Pixel x in image space |
| y | float | Pixel y in image space |
| z | float | Relative depth |
| visibility | float | [0, 1]; body landmarks only; default 1.0 |
| presence | float | [0, 1]; body landmarks only; default 1.0 |

**FaceResult**

| Field | Type | Description |
|---|---|---|
| bbox | BoundingBox | Detected face region |
| landmarks | List[Landmark] | 478 points in image pixel space |
| blendshapes | float[52] or None | Expression coefficients; None when --face-expressions not set |

**HandResult**

| Field | Type | Description |
|---|---|---|
| bbox | BoundingBox | Detected hand region |
| landmarks | List[Landmark] | 21 points in image pixel space |
| handedness | str | "Left" or "Right" |
| handedness_score | float | Handedness confidence [0, 1] |

**BodyResult**

| Field | Type | Description |
|---|---|---|
| landmarks | List[Landmark] | 39 points; indices 0–32 used for skeleton |
| presence | float | Whole-body presence score [0, 1] |

---

### Config

| Field | Type | Source flag |
|---|---|---|
| camera | int | --camera |
| width | int | --width |
| height | int | --height |
| mirror | bool | --mirror |
| face | bool | (default True; --no-face disables) |
| body | bool | (default True; --no-body disables) |
| hands | bool | (default True; --no-hands disables) |
| show_roi | bool | --show-roi |
| show_fps | bool | --fps |
| face_expressions | bool | --face-expressions |

---

### ModelManager

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `__init__` | models_dir: Path | — | Stores directory; does not load any sessions |
| `validate` | required: List[str] | — | Raises FileNotFoundError listing all missing files |
| `get_session` | name: str | InferenceSession | Creates and caches session on first call |

Model name keys: `face_detector`, `face_landmarks`, `face_blendshapes`, `hand_detector`, `hand_landmarks`, `pose_landmarks`

---

### VideoCaptureModule

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `__init__` | config: Config | — | Opens camera; raises RuntimeError on failure |
| `read_frame` | — | GpuMat | BGR frame; raises RuntimeError on camera failure |
| `release` | — | — | Releases camera handle |

---

### Preprocessor

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `full_frame_nhwc` | frame_gpu: GpuMat, target_w: int, target_h: int | ndarray [1,H,W,3] float32 | BGR→RGB on GPU, resize on GPU, normalize on CPU |
| `crop_roi_nhwc` | frame_gpu: GpuMat, bbox: BoundingBox, target_w: int, target_h: int, pad_fraction=0.25 | (ndarray [1,H,W,3], BoundingBox) | Returns array and actual crop box (after padding + clamp) |
| `to_image_space` | lms_model: ndarray [N,≥2], model_w: int, model_h: int, crop_bbox: BoundingBox, frame_w: int, frame_h: int, mirror: bool | List[Landmark] | Transforms pixel coords from model space → image space |

---

### FaceProcessor

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `__init__` | model_manager: ModelManager, enable_blendshapes: bool | — | |
| `process` | frame_gpu: GpuMat, frame_w: int, frame_h: int, mirror: bool | List[FaceResult] | Full face pipeline |
| `_decode_detections` | regressors [1,896,16], classificators [1,896,1], frame_w, frame_h | List[BoundingBox] | Anchor decode + sigmoid + NMS |
| `_run_blendshapes` | landmarks_raw [478,3] | ndarray[52] or None | Extracts 146-point subset, normalizes, runs session |

Constants: `DETECTOR_W=128`, `DETECTOR_H=128`, `LANDMARK_W=256`, `LANDMARK_H=256`

---

### HandProcessor

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `__init__` | model_manager: ModelManager | — | |
| `process` | frame_gpu: GpuMat, frame_w: int, frame_h: int, mirror: bool | List[HandResult] | Full hand pipeline; max 2 results |

Constants: `DETECTOR_W=192`, `DETECTOR_H=192`, `LANDMARK_W=224`, `LANDMARK_H=224`, `MAX_HANDS=2`

---

### BodyProcessor

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `__init__` | model_manager: ModelManager | — | |
| `process` | frame_gpu: GpuMat, frame_w: int, frame_h: int, mirror: bool | BodyResult or None | Returns None when presence < PRESENCE_THRESHOLD |

Constants: `INPUT_W=256`, `INPUT_H=256`, `PRESENCE_THRESHOLD=0.5`, `VISIBILITY_THRESHOLD=0.5`

---

### Renderer

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `__init__` | show_roi: bool | — | |
| `draw_faces` | frame: GpuMat, results: List[FaceResult] | — | Draws landmarks, skeleton, optional ROI box |
| `draw_hands` | frame: GpuMat, results: List[HandResult] | — | Left/right color differentiation |
| `draw_body` | frame: GpuMat, result: BodyResult or None | — | Applies visibility/presence opacity rules |
| `draw_fps` | frame: GpuMat, fps: float | — | Corner text overlay |

**Universal drawing rules applied by all draw methods:**
- Skip any landmark where `is_out_of_bounds` is True
- Skip any landmark where `presence < 0.5` (body only)
- Draw at 30% opacity where `visibility < 0.5` (body only)
- Skip skeleton segment if either endpoint is skipped

---

### FPSCounter

| Method | Inputs | Returns | Notes |
|---|---|---|---|
| `__init__` | window=30 | — | Rolling deque of 30 frame timestamps |
| `tick` | — | — | Record current timestamp; call once per displayed frame |
| `get_fps` | — | float | `window / (newest_ts − oldest_ts)` |

---

### topology.py — Constants

| Constant | Type | Description |
|---|---|---|
| `FACE_CONNECTIONS` | List[Tuple[int,int]] | 478-point MediaPipe face mesh edge pairs |
| `HAND_CONNECTIONS` | List[Tuple[int,int]] | 21-point hand skeleton edge pairs |
| `BODY_CONNECTIONS` | List[Tuple[int,int]] | 33-point body skeleton edge pairs (indices 0–32 only) |
| `CANONICAL_BLENDSHAPE_INDICES` | List[int] | 146 indices into the 478 face landmarks for blendshape input |

---

## f. Project File Structure

```
poser/
├── poser.py                                   # Entry point: CLI parsing, main loop, error handling
│
├── src/
│   ├── __init__.py
│   ├── types.py                               # BoundingBox, Landmark, FaceResult, HandResult, BodyResult
│   ├── config.py                              # Config dataclass + parse_args()
│   ├── model_manager.py                       # ONNX session lifecycle: lazy load, validate, cache
│   ├── video_capture.py                       # VideoCaptureModule — GpuMat output, mirror flip
│   ├── preprocessor.py                        # Preprocessor — GPU crop/resize, normalize, coord transform
│   ├── face_processor.py                      # FaceProcessor — detect → landmarks → blendshapes
│   ├── hand_processor.py                      # HandProcessor — detect → landmarks
│   ├── body_processor.py                      # BodyProcessor — single-stage full-frame pose
│   ├── renderer.py                            # Renderer — skeleton, ROI boxes, FPS overlay
│   ├── fps_counter.py                         # FPSCounter — rolling deque average
│   └── topology.py                            # MediaPipe connection index constants
│
├── models/
│   ├── face_detector_opset15.onnx             # BlazeFace    input [1,128,128,3]
│   ├── face_landmarks_detector_opset15.onnx   # FaceMesh     input [N,256,256,3]  → 478 landmarks
│   ├── face_blendshapes_opset15.onnx          # Blendshapes  input [1,146,2]      → 52 coefficients
│   ├── hand_detector_opset15.onnx             # BlazePalm   input [1,192,192,3]
│   ├── hand_landmarks_detector_opset15.onnx   # HandMesh     input [1,224,224,3]  → 21 landmarks
│   └── pose_landmarks_detector_opset15.onnx   # BlazePose    input [1,256,256,3]  → 39 landmarks
│
├── requirements.txt
└── CLAUDE.md
```

### Module Dependency Graph

```
poser.py
  ├── src/config.py          (parse_args)
  ├── src/model_manager.py   (validate, get_session)
  ├── src/video_capture.py
  ├── src/face_processor.py
  │     ├── src/model_manager.py
  │     ├── src/preprocessor.py
  │     └── src/topology.py    (CANONICAL_BLENDSHAPE_INDICES)
  ├── src/hand_processor.py
  │     ├── src/model_manager.py
  │     └── src/preprocessor.py
  ├── src/body_processor.py
  │     ├── src/model_manager.py
  │     └── src/preprocessor.py
  ├── src/renderer.py
  │     └── src/topology.py    (FACE_CONNECTIONS, HAND_CONNECTIONS, BODY_CONNECTIONS)
  ├── src/fps_counter.py
  └── src/types.py             (shared by all modules)
```

---

## g. Optional Orin audio service (2026-10)

Separate from the pose pipeline: a user systemd unit, `party-pose-audio.service`, installed only on the Orin by `tools/orin_setup/install_audio_service.sh` (never by the Party-Pose install, never on the laptop). It shares no code or process with `poser.py` and runs at `Nice=10`.

```
~/Projects/Video_Projects/Music/*.{webm,m4a,mp3,flac,ogg,opus,wav}      (copied by hand, not in git)
        |
tools/orin_setup/audio/party_pose_audio.py   (stdlib Python, polls every 2 s)
  Controller.tick():   pactl list short sinks --> find_usb_sink(SINK_MATCH)
      speakers present, no player  --> set sink volume --> MpvPlayer(tracks rotated to resume point)
      speakers gone,    player     --> remember playlist-pos, stop mpv
      player exited on its own     --> restart, at most every 5 s
  button thread:  /dev/input/by-id/*USB_Speaker*-event* --> parse_key_presses --> Controller.press
      163 Next -> playlist-next   165 Previous -> playlist-prev   164 Play/Pause -> cycle pause
        |
mpv --no-video --audio-device=pulse/<usb sink> --loop-playlist=inf --input-ipc-server=$XDG_RUNTIME_DIR/party-pose-audio.sock
```

- **Modes** (`MODE` in `~/.config/party-pose/party-pose-audio.env`): `sorted` (default), `random`, `playlist` (`playlist.m3u` in the Music folder). Other keys: `MUSIC_DIR`, `VOLUME` (set on the sink at every connect), `SINK_MATCH`.
- **Resume** after a replug restarts at the track that was playing (kept in memory only); `random` reshuffles. The folder is read each time playback starts, so new files need a service restart.
- **Install pieces:** user unit in `~/.config/systemd/user/`, env file, linger for the user (boot start with no login; the Orin's `pulseaudio.socket` is already user socket-activated), and a udev rule (`/etc/udev/rules.d/99-party-pose-audio.rules`) that lets group `audio` read the speaker's input device. `--no-udev`, `--no-enable`, `--force`, `--uninstall` are the options.
- **Tests:** `test_audio_service.py` (laptop, `mpv` and `pactl` stubbed). Hardware behaviour is checked on the Orin; see the audio section of `tools/orin_setup/README.md`.
- **Files:** `tools/orin_setup/audio/{party_pose_audio.py, party-pose-audio.service.in, party-pose-audio.env.example, 99-party-pose-audio-buttons.rules}`, `tools/orin_setup/install_audio_service.sh`.
