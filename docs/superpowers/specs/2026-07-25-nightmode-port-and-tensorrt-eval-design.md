# Night-mode port and TensorRT per-model compatibility evaluation

## Context

Follow-up to `docs/superpowers/specs/2026-07-25-orin-archive-and-platform-config-design.md`
and `docs/orin-fix-integration-eval.md`. That earlier pass added a `--platform orin|laptop|auto`
config and, while implementing it, actually built the real GStreamer/CSI pipeline inside
`src/video_capture.py`'s `"orin"` branch (not just a placeholder, as the eval doc's wording
implied — confirmed by re-reading the applied file). The one piece from
`Party-Pose.orin_monolith.py` still unported is the `NightModeDetector`.

Separately, `src/model_manager.py` was updated to try `TensorrtExecutionProvider` first when
`platform == "orin"`, but this has never been verified against any of the 5 models — they've
only ever run under `CUDAExecutionProvider` on the Orin.

## Goals

1. Port night-mode grayscale switching into `src/video_capture.py`, running early enough in
   the pipeline that both detection and display see the same desaturated frame during night
   mode — matching the user's explicit call that detection needs the same image, and that the
   gain/exposure poll must not cost FPS.
2. Enable TensorRT engine caching in `src/model_manager.py` so the "build once, reuse after"
   assumption is real, not just how ONNX Runtime happens to behave.
3. Produce a factual per-model TensorRT compatibility report (pass/fail, shape match, cold vs
   warm timing) — no further `model_manager.py` provider-selection logic changes in this pass;
   that's a follow-up decision once the report exists.

## Non-goals (this pass)

- No per-model provider-selection logic in `model_manager.py` (e.g. "skip TensorRT for model
  X") — the report informs that decision later, doesn't make it.
- No numeric/regression comparison of TensorRT vs CUDA output values — shape-match only, per
  the earlier scope decision.
- No calibration of the night-mode gain/exposure thresholds against a real dark/IR-lit test —
  still an open item from the previous pass.

## Design

### A. Night mode in `src/video_capture.py`

- Port `NightModeDetector` and `query_gain_exposure` from `Party-Pose.orin_monolith.py`
  essentially as-is: same 30-frame poll interval (avoids spawning `v4l2-ctl` every frame), same
  gain/exposure threshold constants (still uncalibrated against a real dark test — unchanged
  from before).
- `VideoCaptureModule.__init__`: `self._night_mode = NightModeDetector()` only when
  `config.platform == "orin"`; otherwise `self._night_mode = None`, so laptop/webcam use never
  spawns a `v4l2-ctl` subprocess.
- `read_frame()`: after `self._gpu_frame.upload(frame)` and the existing mirror-flip step, if
  `self._night_mode is not None and self._night_mode.update()` is true, apply
  `cv2.cuda.cvtColor` BGR→GRAY then GRAY→BGR on `self._gpu_frame` in place. Stays GPU-resident
  (no CPU download/upload round-trip), preserving the project's GPU-residency principle.
  Because this happens before `read_frame()` returns, every downstream consumer — detection
  and the renderer alike — sees the same desaturated frame during night mode.
- No change to `release()`.

### B. TensorRT engine caching + compatibility report

- `src/model_manager.py`: extend `trt_provider_opts` with `trt_engine_cache_enable=True` and
  `trt_engine_cache_path` pointing at `models/trt_cache/` (already covered by the `trt_cache/`
  pattern in `.gitignore`, so cached engine files never get committed). This is what makes the
  "build once, reuse after" claim actually true — without it ONNX Runtime rebuilds the engine
  from scratch on every process start.
- A standalone script, run on the Orin only (not added to the laptop's `test_config.py` pytest
  suite — it needs real TensorRT hardware and isn't something CI-style tooling can run):
  - For each of the 5 models (`person_detector`, `face_landmarks`, `face_blendshapes`,
    `hand_landmarks`, `pose_landmarks`): construct via `ModelManager(models_dir,
    platform="orin")`, build a synthetic input matching the model's documented input shape
    (from `CLAUDE.md`'s "Model Specifications" section), run inference **twice** — first run
    is cold (builds + caches the TensorRT engine), second run is warm (loads the cached
    engine) — timing both separately.
  - Catch and record any exception from either run.
  - Compare output tensor shapes against a `CUDAExecutionProvider` run of the same
    model/input, to confirm TensorRT didn't silently produce a different-shaped (and thus
    clearly wrong) output.
- Output: `docs/orin-tensorrt-compatibility.md` — one row per model: pass/fail, error text (if
  failed), shape-match confirmation, cold load time, warm load time, inference time.

## Testing

- Night mode: no new automated test beyond what already exists (`NightModeDetector`'s logic
  was already unit-testable in principle, but the monolith's version had no tests either;
  matching that baseline — manual verification via a real dark/IR-lit run is the actual
  validation, tracked as the pre-existing uncalibrated-thresholds follow-up).
- TensorRT: the compatibility script itself *is* the test — its output is the report. No
  pytest wrapper, since it requires the Orin's hardware/TensorRT install to mean anything.
