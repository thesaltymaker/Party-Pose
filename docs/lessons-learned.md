# Lessons learned

## Jetson Orin performance (issue #2, 2026-09-26)

- **Pin the clocks before measuring anything.** Under a bursty real-time load, `nvhost_podgov` keeps the Orin GPU at 408–510 MHz of 1020, and `schedutil` drops CPU cores to 729 MHz. Every model then takes about 2× its standalone time. Run `sudo jetson_clocks`; it resets on reboot. Isolated benchmarks saturate the GPU and hide this.
- **Compare in-app stage time with isolated time.** A large gap means the cost is outside the model: clocks, contention, or Python glue.
- **Detection count moves FPS more than code changes do.** Each detected body adds a face run and a body run, about 9 ms. Log detections per frame next to timings, and compare old and new code back to back in the same scene. An early "15 → 27 FPS" result turned out to be the scene changing, not the fix.
- **A slow TensorRT build is not a hang.** The person detector's build "hung" for 300+ s, then failed after about 6 min with `Layers missing empty tensor support`. Give builds a 10-minute timeout and read the error.
- **Split models with built-in NMS by op type.** ORT's `trt_op_types_to_exclude` works per op type, not per node. Excluding the op types that occur only after NMS, plus `trt_min_subgraph_size`, keeps the backbone in TensorRT and puts the post-processing on CUDA/CPU. `tools/orin_bench/bench_models.py post-ops` lists those types.
- **Python loops over landmarks are expensive on the Orin's CPU.** Numpy scalar indexing in a 478-point loop cost 4.3 ms per face. Vectorize with `.tolist()`.
- **The "different models of devices" TensorRT warning was harmless here.** Rebuilt engines ran at the same speed; check this rather than assuming.
- **Measure GPU clocks from sysfs** (`/sys/class/devfreq/17000000.gpu/cur_freq`). tegrastats on this JetPack shows only GR3D percent.
- **Tooling:** `devops_task` commands that run past its time limit report FAIL with no output but keep running on the remote host. Background long jobs with `nohup ... &` and poll a log, otherwise duplicate runs pile up and compete for the GPU.

## False positives (issue #3, 2026-09-26)

- **Save the raw frame, not just the overlay.** Overlay frames have meshes drawn on the very crops you want to re-score. `--dump-detections` saves both.
- **Replay offline before choosing a threshold.** Cropping dumped boxes from raw frames and running the landmark model on the laptop gave score ranges for real vs false detections in minutes, without touching the Orin.
- **Label by eye with a contact sheet.** A grid of crops with their scores printed on each tile made it obvious which scores were real faces. Labelling by position broke as soon as people moved.
- **Check what a model already outputs.** The face landmark model had a face score all along (`Identity_1`); the code only read the landmarks.
- **Crop padding changes model scores a lot.** With 25% padding, real faces scored near the decision boundary; without padding they scored far above it.
- **Some false positives can't be fixed per frame.** A static lamp scores the same as a seated person on every signal the models give. It needs information across frames (tracking, calibration).
- **Lighting changes the false positives.** In bright sun the ball and lamp were not detected at all. Test in the light you will run in.

## Keeping the Orin recoverable (2026-09-26)

- **An agent copying a whole repo tree destroyed the Orin's Python environment.** The laptop's `env/` (Python 3.12) overwrote the Orin's (Python 3.10). `env/` is not in git, so nothing recorded how it was built. Deploy through git only, and never copy directories to the Orin.
- **pip's HTTP cache can hold wheels you can no longer find.** The original onnxruntime-gpu wheel was recovered from `~/.cache/pip/http` by searching cached responses for the zip header. Keep wheels you install from outside PyPI.
- **Snapshot before changes, automatically.** `tools/orin_setup/snapshot.sh` runs from the pre-push hook; hard links keep snapshots cheap, and read-only directories stop `rm -rf`. See `tools/orin_setup/README.md`.
- **pip will quietly upgrade numpy.** Installing onnx pulled numpy 2, which broke the OpenCV build compiled against numpy 1.x (`_ARRAY_API not found`).
- **Check `pgrep -f` isn't matching itself.** Over ssh, `pgrep -f 'python poser'` matched the ssh command line and reported a run that had never started. Use `ps -eo args | grep '^[p]ython poser'`.
