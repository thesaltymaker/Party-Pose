# Orin fix integration evaluation

Evaluated against the laptop's actual modular `src/` code and its `Party-Pose.py` reference
(read directly, not assumed), during the 2026-07-25 Orin capability/performance test. See
`docs/superpowers/specs/2026-07-25-orin-archive-and-platform-config-design.md` for full
context. Archived source: `Party-Pose.orin_monolith.py`.

| # | Fix | Verdict |
|---|-----|---------|
| 1 | Canvas black-screen bug (camera frame never copied onto the display canvas) | **N/A** — laptop's modular render path already handles frame display correctly. This was a regression introduced during the Orin's own hand-patching, not present in the laptop's reference or `src/`. |
| 2 | `preprocess_for_detection` undefined `box` | **N/A** — laptop's reference `Party-Pose.py` already has the correct version of this function (no `box` reference, no crop-by-box at the whole-frame detection stage). |
| 3 | Undefined `ensure_rgb` | **N/A** — same reasoning as #2; laptop's `crop_for_hand_lmk`/`crop_for_facemesh` never call a function like this. |
| 4 | GStreamer caps syntax fix (`format=(string)BGR` not `format=BGR(string)`) | **New feature to add**, not a bug port — `src/video_capture.py` has no CSI/GStreamer path at all today, only `cv2.VideoCapture(config.camera)`. If CSI support is added, use the corrected caps syntax from the start. |
| 5 | `NightModeDetector` (gain/exposure-based grayscale switch) | **New feature to port**, contingent on #4 — it's IMX219/`v4l2-ctl` specific and only makes sense once CSI capture exists in `src/`. |

## Open follow-ups

- **TensorRT execution provider**: the Orin's onnxruntime build exposed and used
  `TensorrtExecutionProvider` automatically (seen in the Orin run's provider list:
  `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`).
  `src/model_manager.py` only ever requests `CUDAExecutionProvider`/CPU. Adding TensorRT to
  the provider list (Orin-only) and/or pre-converting models to TensorRT engines for faster
  cold start needs per-model compatibility testing — some ONNX ops may lack TensorRT kernels
  and require per-node CPU/CUDA fallback. Not yet verified which of the five models
  (`person_detector`, `face_landmarks`, `face_blendshapes`, `hand_landmarks`, `pose_landmarks`)
  actually work correctly under TensorRT on the Orin.
