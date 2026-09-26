# Orin FPS with a person in view (issue #2)

Status: fixes merged to `main`. The target of 30 FPS is met when about one body is detected per frame with `jetson_clocks` on. It is not met when the scene yields two or more detected bodies.

## Problem

On the Jetson Orin Nano Super, `python poser.py --platform orin --fps` dropped to 8–16 FPS with a person in view. With all models disabled it ran at 30.8 FPS, so capture and render were not the bottleneck.

## Method

1. `poser.py` records per-stage timing (capture, person, face, hands, body, render), detection counts per frame and each session's execution provider. They print as 60-frame averages with `--fps` (`[STAGES ms/frame]`, `[PROVIDERS]`).
2. `tools/orin_bench/bench_models.py` times each piece in isolation. `tools/orin_bench/run_poser.sh` runs the app for a fixed time and samples GPU/CPU clocks. See `tools/orin_bench/README.md`.
3. Compare stage times inside the app with isolated times. A large gap points to something outside the model (clocks, contention).
4. For a before/after comparison, run old and new code back to back in the same scene. Detection counts change FPS more than any code change does.

## Findings

| # | Cause | Evidence | Action |
|---|---|---|---|
| 1 | Dynamic clocks. `nvhost_podgov` kept the GPU at 408–510 MHz (max 1020) and `schedutil` dropped CPU cores to 729 MHz, because poser's load comes in bursts (GR3D busy 40–90%). | Sampled `/sys/class/devfreq/17000000.gpu/cur_freq` during runs. In-app inference was about 2× the isolated benchmark. | Run `sudo jetson_clocks` (needs root and resets on reboot; making it persistent is a follow-up). |
| 2 | person_detector ran on the CUDA EP because it was excluded from TensorRT. | 10.7 ms on CUDA against 7.2 ms on TensorRT in isolation at default clocks; 8.1 against 3.1 ms with clocks pinned. | Backbone now runs in TensorRT (see below). |
| 3 | `Preprocessor.to_image_space` looped over landmarks with numpy scalar indexing. | 4.3 ms per 478-point face on the Orin. | Vectorized: 0.6 ms, output equal within 3e-5 px. |
| 4 | Stale TensorRT engines (built Aug 16, likely before the board became "Super") log "different models of devices". | Fresh engines remove the warning but run at the same speed. | No change. Delete `models/trt_cache` to rebuild (about 7 min cold start). |

### person_detector and TensorRT

The YOLOX model has NMS built in. The earlier "hang" was a very slow build, 6+ minutes, which then fails with `TRT-16198 - Layers missing empty tensor support`. Excluding only `NonMaxSuppression` still fails, because the ops after NMS form their own TensorRT subgraph with empty tensors.

Fix, in `ModelManager._TRT_EXTRA_OPTS`: exclude every op type that appears only after NMS (`Cast, Squeeze, Shape, Relu, Gather, GatherND` plus the default `NonMaxSuppression, NonZero, RoiAlign`), and set `trt_min_subgraph_size=20` so leftover fragments stay on CUDA. The backbone (Conv/Mul/Sigmoid/Concat/...) contains none of those types, so it stays one TensorRT engine. `bench_models.py post-ops` recomputes the list, which also shows `Div`; `Div` is covered by the min-subgraph setting. Detections on a real frame match the CUDA EP to within 0.4 px.

## Results

Same scene (one person on the couch, plus a second figure in the kitchen background), `jetson_clocks` on, old and new code run back to back:

| Stage (ms/frame) | Before | After |
|---|---|---|
| capture | 2.1–2.8 | 2.2–3.0 (7–10 at ~1 body: waiting on the 30 FPS camera) |
| person detector | 16.2–17.4 | 8.2–10.7 |
| face | ~8.5 per run | ~4.8 per run |
| body | ~4.7 per run | ~4.0 per run |
| download + render | 9.2–11.9 | 7.4–12.0 |
| **FPS** | **15–20** | **20–31; 29–31 at ~1 body/frame** |

The first baseline, at default clocks with the original code, was 14–16 FPS.

Final verification with `tools/orin_bench/run_poser.sh` (merged code, `jetson_clocks`, ~1.2 bodies/frame): **28.5–31.6 FPS** (22 samples, 21 at 29.4 or higher). Per frame: person 8.4–10.1, face 6.1–6.7, body 4.3–5.0, render 7.9–8.8 ms. The GPU stayed at 1020 MHz for all 72 clock samples.

## Remaining cost

- With one person the floor is about 30 ms/frame: person ~10 + face ~5 + body ~4 + render ~9 + capture ~2.
- Each extra detected body adds about 9 ms/frame (face + body landmark runs). The scene averaged about 2 bodies per frame. Separating real people from false positives is tracked in #3.
- Render (1080p download + draw + `imshow`) is about 9 ms. Overlapping it with inference would need a pipelined loop; not done.
