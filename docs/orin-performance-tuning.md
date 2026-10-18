# Performance tuning on the Jetson Orin: what worked

A playbook from tuning Party-Pose on the Orin Nano (JetPack 6, CUDA 12.6, TensorRT through ONNX Runtime 1.23,
OpenCV 4.10 with CUDA). Two people dancing, 1080p camera: 20.4 FPS at the start, **35.6 FPS** with all of the
below (`--mirror --pipeline`). Day-by-day notes are in `lessons-learned.md`; this file is the method and the
techniques, with what each one gave.

## 1. Measure first, the same way every time

| Step | How | Why |
|---|---|---|
| Pin the clocks | `sudo jetson_clocks` (resets on reboot); check `/sys/class/devfreq/17000000.gpu/cur_freq` = 1020000000 | Without it the GPU idles at 408-510 MHz under bursty load and every model takes ~2x |
| Replay a recorded clip, not live people | `tools/orin_bench/record_video.py` (raw frames, ~11 GB/min), `poser.py --video CLIP`, `tools/orin_bench/clip_bench.sh` | Live FPS depends on who is dancing; a clip repeats within 0.2 FPS |
| Check the benchmark's own time | The `capture` stage of a replay should be ~2 ms, like the camera | A synchronous 6 MB read per frame added 10 ms and hid every gain |
| Nothing else on the GPU | `systemctl is-active party-pose`, `pgrep -af poser.py` before each run | A restarted service running alongside made every run ~30% slower |
| Interleave A/B runs | A, B, A, B back to back, not all A then all B | Drift between steps minutes apart flipped a +0.7 FPS change to -0.6 |
| Same arguments as production | Include `--mirror`, `--pipeline`, `--black-bg --skelly` | What is measured must be what is shown |
| Check outputs, not just speed | Tests compare the fast path with the old one on the **same** engine (exact to 1e-3); clip runs compare hands found and face-score lines | A TensorRT FP16 vs FP32 comparison differs by up to 15 model pixels on noise and hides real bugs |

## 2. Find where the time goes

- **Per-stage timings in the app** (`--fps` prints `[STAGES ms/frame]`): which stage, and how it scales with people.
- **Per-call micro-benchmarks on the device** (`bench_stages.py`, `bench_iobinding.py`, `bench_face_detector.py`): model run vs crop vs copy, batch 1 vs 2. Then split a slow call into its parts: the crop's parts added up to half its time, the rest was allocation.
- **`nsys` timeline** (installed with JetPack: `nsys profile -t cuda,osrt,nvtx --sample=none --delay 15 --duration 30`, then `nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum`): kernel time / wall time is the real GPU use (jtop's GR3D % is not). It showed 23% GPU compute, ~900 kernel launches a frame at ~12 us each, ~4 ms/frame of synchronizes.
- **`py-spy record --threads --idle -f raw`**: which thread is the bottleneck and what it waits on (the model thread spent 52% inside `session.run`; the drawing thread was idle 74%).
- Half-busy GPU and CPU at low power (~20 W of 35 W, 65 C) means **waiting**: launches, syncs and copies, not lack of compute.

## 3. Techniques, in the order of what they gave

| Technique | Where | Gain (two people) |
|---|---|---|
| **Overlap drawing with the next frame's models** (`--pipeline`: models in a worker thread, drawing and `imshow` in the main thread, one-slot queue) | `poser.py` `_run_pipelined` | +2-3 FPS (drawing ~8 ms was serial) |
| **CUDA graphs** for small TensorRT models (`trt_cuda_graph_enable`): the model's hundreds of kernel launches are recorded once and replayed as one | `ModelManager._TRT_EXTRA_OPTS`, `src/gpu_model.py` | hand 1.40 -> 1.11 ms, body 2.31 -> 1.84 ms, 2 faces 5.3 -> 2.9 ms |
| **Keep crops on the GPU, bind them to the model** (ONNX Runtime I/O binding to a continuous `cv2.cuda.createContinuous` buffer; outputs to fixed GPU `OrtValue`s): no download, no CPU float conversion, no ONNX Runtime upload | `src/gpu_model.py` | hand 1.68 -> 1.40 ms, body 2.64 -> 2.31 ms; required for CUDA graphs |
| Hand + body together (GPU crops + graphs) | | 30.8 -> 34.5 FPS (no `--mirror`) |
| **Batch the same model across people** (all faces in one run) | `FaceProcessor.process_many`, `GpuFaceBatch` | 2 faces 2 x 2.36 -> 2.89 ms (+2.1 FPS) |
| Faces + detector input on the GPU | `GpuFaceBatch`, `GpuDetectorInput` | 32.6 -> 35.6 FPS (`--mirror`) |
| **Shrink before converting** (resize the 1080p frame, then BGR->RGB on the small image) | `Preprocessor.full_frame_nhwc` | -1.2 ms, +1.4 FPS |
| **Reuse buffers instead of allocating per call** (GPU images, host image, float array) | `Preprocessor.crop_rotated_nhwc` | +0.7 FPS (allocation was half the crop time) |
| **Pass arrays, not Python objects** (processors hand the [N, 2] point array to smoothing; no rebuilding from 478 Landmark objects) | `pts` on results, `src/smoothing.py` | smoothing 0.94 -> 0.55 ms (3 people, laptop) |
| One model run per hand per frame (retry a lost hand on the next frame) | `HandProcessor.process` | hand time 2.5 -> 1.9 ms/person, 92% -> 89% found |
| FP16 TensorRT engines with a cache (`trt_fp16_enable`, `trt_engine_cache_enable`) | `ModelManager` | already on before; keep |

| **Load every session at startup** (each batch size too) | `poser._load_sessions` | no 1-3 s freeze when a new number of people appears; replays no longer count the load |
| **Batch per batch size** (`--pose-batch 4 --hand-batch 4 --face-batch 4`, one TensorRT session per size) | `GpuBatch` | +2.2 FPS with 3-4 people |
| **Hands and faces every 2nd frame per person** (`--hand-every 2 --face-every 2`) | `ghosts.skip_turn` | +1.2-1.4 FPS, fewer blinks |
| **Person detector every 3rd frame while everyone is followed** (`--detect-every 3`) | `poser._detect_now` | +1.2 FPS |
| Python fixes found with py-spy (ghost face moved when read, `is_still()` every 10th box) | `poser._MovedLandmarks`, `Track.is_still` | +2 FPS with 4 people |

What did not pay: moving the crop to the CPU (2.2 ms vs 0.8 on the GPU); a second CUDA graph per batch size (wrong outputs, see below).

## 4. Pitfalls that took time to find

- **CUDA streams: OpenCV and ONNX Runtime do not wait for each other.** OpenCV's `cv2.cuda` calls run on the default stream; ONNX Runtime runs on its own non-blocking stream, and an upload from ordinary (pageable) CPU memory can return before it lands. Wait (`cv2.cuda.Stream.Null().waitForCompletion()`) before each bound run: without it 8 of 60 test runs read a stale input.
- **A CUDA graph replays the addresses of its first run.** One set of input/output buffers per graph-enabled session (`gpu_model()` keeps one per model). Every run of that session must go through the binding.
- **One batch size per graph.** A second graph for another batch size in the same session (`gpu_graph_id` run option) gave outputs off by 215 model pixels with no error. Run a fixed batch (faces always at 2; one face leaves the second slot's output unused).
- **Give dynamic-batch TensorRT models a fixed shape range** (`trt_profile_min/opt/max_shapes`). Otherwise the first new batch size rebuilds the engine (~6 min on the Orin, app frozen).
- **The Orin's OpenCV build may not write into a `dst` argument.** Always use the returned `GpuMat`; check the returned `cudaPtr()` when the buffer must stay the same (a bound input).
- **`GpuMat` rows are padded.** A model input must be one block: `cv2.cuda.createContinuous`. `convertTo(type, dst, alpha, beta)` is the argument order that writes into `dst` in OpenCV 4.10.
- **Shared CPU/GPU memory.** Copies are memory-to-memory (no PCIe); their time is mostly the synchronize each one forces. Large file reads fill the same memory as page cache: drop it (`posix_fadvise(DONTNEED)`) or a later GPU allocation fails (NvMap error 12).
- **Provider options read back from a session come back as strings ORT rejects**; build them again (as `ModelManager` does) when creating another session.
- **`jtop` shows the system OpenCV (4.8.0, no CUDA)**; the app's venv loads the CUDA build through `00-opencv-cuda.pth`. Check with the venv's python.
- **Commands over ssh**: `pkill -f poser.py` and `grep poser.py` match their own `bash -c`; use `pgrep -f "^/home/thesa/Projects/Party-Pose/env/bin/python poser.py"`. Long runs: `nohup ... & disown` and a done marker written even on failure.

## 5. Where the time is now (three dancers, the service arguments, ~29 ms model thread, 2026-10-18)

body ~8.2 ms (3 poses in one run), hands ~6.2, face ~6.3, person detector ~4.2 (every 3rd frame while everyone is followed), capture ~2.1, smoothing ~1.3; drawing ~9.7 ms in the other thread. 45% of the model thread is inside `run_with_iobinding`; much of the rest is waiting for Python's lock (three Python threads). A parallel detector stage (`--detect-ahead`) works but the GPU is shared: the other models run 25-35% slower. Next candidates: fewer model runs (a fixed batch across hands and faces), and less Python per frame in the drawing thread.
