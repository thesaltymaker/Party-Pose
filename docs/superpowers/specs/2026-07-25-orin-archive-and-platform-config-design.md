# Orin monolith archival, fix-integration evaluation, and platform config

## Context

Party-Pose exists in two diverged states:

- **Laptop** (this repo): modular rewrite (`poser.py` + `src/`), never tested against Jetson
  CSI camera hardware. `Party-Pose.py` at the repo root is an untouchable reference
  implementation (do not modify per `CLAUDE.md`).
- **Orin** (`thesa-orin1`, 192.168.2.168): a hand-patched monolithic `Party-Pose.py` using a
  GStreamer/CSI camera pipeline (`nvarguscamerasrc`) plus IR night-vision hardware (IMX219
  sensor + IR LED illuminator ring with its own onboard photoresistor — no software control
  path over the LEDs exists on the Jetson).

A live debugging session against the Orin found and fixed five issues in its monolith:

1. Canvas was zeroed every frame and the camera image was never copied onto it (black
   display, only skeleton overlays drawn).
2. `preprocess_for_detection` referenced an undefined `box` variable — leftover from a
   refactor; this is the whole-frame detection stage, not an ROI crop stage.
3. `ensure_rgb` was called in two places but never defined anywhere in the file — crashed
   the app as soon as a hand was detected.
4. The GStreamer caps strings used backwards syntax (`format=BGR(string)` instead of the
   valid `format=(string)BGR`). This silently collapsed the capture to a single-channel
   (grayscale-looking) frame even though the sensor is a full-color IMX219 — confirmed via
   pixel-channel-mean divergence and an independent vision-model description of a saved
   frame.
5. Added a `NightModeDetector` that polls `/dev/video0` gain/exposure via `v4l2-ctl` every
   30 frames (not every frame, to avoid subprocess overhead) and desaturates the display feed
   to grayscale once both are past a threshold — because color reconstruction under pure IR
   illumination is meaningless (every Bayer sub-pixel sees the same non-visible wavelength).
   Thresholds are unverified against a real dark/IR-lit test and will need calibration.

Reading the laptop's own `Party-Pose.py` reference and `src/` modules during this session
showed that bugs #1–#3 do **not** exist there — they were regressions introduced specifically
during the Orin's own hand-patching, not inherited from a shared ancestor. Bugs #4 and #5 are
genuinely new capability (CSI camera support does not exist at all in the laptop's modular
code yet).

The user also wants a platform config so the same codebase can target either machine, since
GPU acceleration setup differs: the Orin's onnxruntime build exposes and uses
`TensorrtExecutionProvider` automatically; the laptop's `src/model_manager.py` only ever
requests `CUDAExecutionProvider`/CPU. Not all models are yet confirmed to work correctly
under TensorRT on the Orin — some ONNX ops may lack TensorRT kernels and need per-node
fallback, which requires real per-model testing, not just a config flip.

## Goals

1. Archive the now-fixed Orin monolith under a clear name, in both places (Orin machine and
   laptop repo), before any merge/port work begins — so it can be returned to if the ported
   modular version underperforms on Orin hardware for detection quality.
2. Produce a written evaluation of each of the five fixes against the laptop's actual modular
   architecture (not the untouchable reference file) — what applies, what doesn't, what's net
   -new — without porting any code in this pass.
3. Add a `--platform {orin,laptop,auto}` config so `src/video_capture.py` and
   `src/model_manager.py` can select the right camera backend and ONNX execution provider
   order per machine.

## Non-goals (this pass)

- Do not port/write the CSI camera pipeline or `NightModeDetector` into `src/` yet — that is
  follow-up work once the evaluation doc is reviewed.
- Do not change model file variants per platform.
- Do not decide `TensorrtExecutionProvider` per-model compatibility here — flagged as an open
  follow-up action item in the evaluation doc only.
- Do not modify `Party-Pose.py` (the laptop's reference implementation) — constraint from
  `CLAUDE.md`.

## Design

### A. Archival

- On the Orin (192.168.2.168), rename the fixed monolith in place:
  `/home/thesa/Projects/Party-Pose/Party-Pose.py` → `Party-Pose.orin_monolith.py`.
- Copy that same file into this laptop repo at `Party-Pose.orin_monolith.py` (repo root,
  alongside `Party-Pose.py` and `poser.march.28.works.py`).
- Commit to git and push to `thesaltymaker/Party-Pose` on GitHub (confirm immediately before
  the actual `git push`, per standing risk-action norms).
- The `.bak_*` snapshots created during the live-debugging session stay on the Orin only —
  they're incremental undo points from that session, not archival-worthy, and are not copied
  to the laptop or committed.

### B. Fix-integration evaluation doc

A new markdown file, `docs/orin-fix-integration-eval.md`, in the laptop repo. One entry per
fix, each rated against the laptop's actual `src/` code (confirmed by reading it, not assumed):

| # | Fix | Verdict |
|---|-----|---------|
| 1 | Canvas black-screen bug | N/A — laptop's modular render path already handles frame display correctly; this was an Orin-copy-only regression. |
| 2 | `preprocess_for_detection` undefined `box` | N/A — laptop's reference/`src/` never had this bug. |
| 3 | Undefined `ensure_rgb` | N/A — same reasoning as #2. |
| 4 | GStreamer caps syntax fix | Applies as **new feature**, not a bug port — laptop's `src/video_capture.py` only does plain `cv2.VideoCapture(config.camera)`; CSI/GStreamer support doesn't exist yet. If added, use the corrected caps syntax from the start. |
| 5 | `NightModeDetector` | **New feature to port**, contingent on #4 landing first — it's IMX219/`v4l2-ctl` specific and only makes sense once CSI capture exists. |

Plus an explicit "Open follow-ups" section noting:
- Whether to add `TensorrtExecutionProvider` to `model_manager.py`'s provider list (Orin-only
  branch) and/or pre-convert models to TensorRT engines for faster cold start — needs
  per-model compatibility testing, not a blanket flag.

No code changes to `src/` happen as part of writing this doc.

### C. Platform config

- `src/config.py`: add `platform: str = "laptop"` field to the `Config` dataclass, and a
  `--platform {orin,laptop,auto}` CLI argument (default `"auto"`).
- Add `detect_platform() -> str` in `config.py`: checks `platform.machine() == "aarch64"` and
  `/proc/device-tree/model` containing `"NVIDIA Jetson"` → `"orin"`, else `"laptop"`. Only
  runs when the CLI arg is `"auto"` (the default) — an explicit `--platform orin` or
  `--platform laptop` always wins.
- `src/video_capture.py`: `VideoCaptureModule.__init__` branches on `config.platform` —
  `"orin"` builds the corrected GStreamer/CSI pipeline string; `"laptop"` keeps today's plain
  `cv2.VideoCapture(config.camera)`. (Building the actual CSI pipeline code is deferred until
  the evaluation doc's fix #4 is acted on — this pass only adds the branch point.)
- `src/model_manager.py`: `get_session()`'s provider list branches on a `platform` value
  passed in at `ModelManager` construction — `"orin"` tries `TensorrtExecutionProvider` first
  (falling back to CUDA/CPU exactly as today), `"laptop"` unchanged.
- No model-file-variant switching, no night-mode-enable switching in this pass — matches the
  earlier scope decision (camera backend + GPU provider only).

## Testing

- Archival: verify the renamed file exists and runs (`python3 -m py_compile`) on both the
  Orin and in the laptop repo post-copy.
- Evaluation doc: no code, so no test — reviewed by the user before any follow-up porting
  work is planned.
- Platform config: `detect_platform()` gets a unit test (mock `platform.machine()` and the
  device-tree file check) for both branches; `Config`/CLI parsing gets a test confirming
  `--platform` defaults to `"auto"` and explicit values override detection.
