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

## Tracking, props and faces (issues #6/#7, 2026-09-27)

- **Get labels from the people in the room, and make them easy to read.** Thresholds chosen from unlabelled runs were wrong twice; one labelled 60 s run was worth more than several unlabelled ones. Track numbers must be readable from ~10 ft (text ~1/14 of frame height) and centred on the box: a top-left label next to a lamp's label was read as the lamp's number.
- **A test that needs people must be on the screen.** Two headless face-crop tests ran while nobody was performing, and their result was the opposite of the real one (padding looked best; with people facing the camera the unpadded crop kept 37% of faces vs 0-5% padded). Show the app or a big "TEST RUNNING – do X" with a countdown.
- **Frame-to-frame movement measures jitter, not people.** The pose model's skeleton on a lamp shakes more than a still person's. Measure over a longer window: the 90th-percentile distance of the box centre from its median over ~10 s separated dancers (0.08-0.93 of box height) and a person standing still (0.055) from a prop (0.037). People sitting very still (0.025) look like props.
- **Head-box rate and face scores didn't separate props either.** The lamp got a false head (0.47) on top of it; head and body scores of props and people overlap. Dancers' faces score low (logit < 8) from blur and turns.
- **Fixed ignore zones work until the unit moves.** A venue file of prop boxes removed the lamp and chair, but the owner moves the unit and the light changes through the day; the filter has to learn by itself (`--drop-still`).
- **A box that flips between a small and a big size breaks velocity prediction.** The centre jump looks like movement, the prediction misses the next box, and a new ID starts while the old track is alive. Fall back to the last box when the predicted box doesn't match; revive a lost track when half of the smaller box overlaps.
- **A higher-scoring box inside a person's box can be a duplicate.** A 0.56 box on part of a person, 90% inside their 0.51 box, took their head. When one box is >= 80% inside another, keep the one the head fits.
- **Faces work for the dancer; nobody else may be facing the camera.** With one dancer and kids doing homework, the dancer's face was found in 69-88% of frames and the kids' in 0-7%. Ask who was where before tuning on a "low" number.
- **Tooling:** `pkill -f poser.py` over ssh kills its own `bash -c`; kill by PID. A devops call the owner rejected still ran twice. `Failed to create CaptureSession` means another process (often the service) holds the camera. Killing the service's process with TERM exits 0, so `Restart=on-failure` leaves it stopped.

## Smoothing, pose crops and hands (2026-10-11)

- **Jumpy skeletons came from the crop, not only the detector.** The pose model (BlazePose GHUM) expects MediaPipe's crop: a square centred on the hips, sized from its own alignment points (33, 34), rotated upright, black outside the frame, taken from the previous frame's landmarks. Given the person box letterboxed instead, it drew a skeleton too small for the person (low shoulders, short arms) and lost the pose in a third of frames. With MediaPipe's crop the error against MediaPipe fell from 0.22 to 0.086 shoulder widths, at the same cost.
- **Measure against a reference before changing a model's input.** A 10 s laptop recording and MediaPipe 1.1 (`mediapipe` in a separate venv) as the reference turned "looks bad" into numbers in minutes, and checked each fix through the app's own code. Kept local in `testdata/pose-eval/` (git-ignored; it shows a person).
- **Check which output is which.** The hand model's outputs are landmarks, presence, handedness; the code read handedness as presence, so hands were kept by side and one hand was rarely found. A crop with no hand in it shows which output is presence (0.000 vs 0.3).
- **Crop hands from the body, then from the hand itself.** The detector's hand boxes found 40% of hands on the laptop and almost none at the Orin's distance. Crops from the pose wrist/index/pinky, then from the hand's last landmarks (MediaPipe Holistic), found 92%. Hands MediaPipe "missed" that we found were real (blur, frame edge): look at them before calling them false.
- **One Euro smoothing on landmarks, not on crops.** Smoothing the landmarks per person (speed in object sizes per second) made the skeleton steady without visible lag, at 1.3 ms per frame on the Orin (still ~31 FPS). Smoothing the head box would move the face off-centre in its crop, and the face model loses faces when that happens (issue #6/#7).
- **Log why a new ID starts.** `[NEW-ID]` names the nearest old track (live/lost, distance, size, overlap). On the first run it showed two boxes on one person (`live missed=0 overlap 0.4`), not a lost person — a cause no one had guessed.
- **Small artefacts matter to the viewer.** The pose model's finger points (17-22) drawn on top of the hand model's hands looked wrong; the body now stops at the wrists.

## Speed on the Orin: replay a clip, change one thing at a time (2026-10-11)

- **Benchmark on a recorded clip, not live.** FPS depends on who is dancing; a 60 s raw clip of two people (`tools/orin_bench/record_video.py`, replayed with `poser.py --video`) gives runs that repeat within 0.2 FPS. `tools/orin_bench/clip_bench.sh` runs it with the service's arguments.
- **Check the benchmark's own costs first.** The first replays showed 23 FPS; 10 ms of every frame was the clip reader (a sync 6 MB read per frame, and the page cache it filled made a later GPU allocation fail with NvMap error 12). Reading ahead in a thread and dropping frames from the cache gave the real baseline: 27.6 FPS.
- **Interleave A/B runs.** Measured in sequence, reusing crop buffers looked 0.6 FPS worse; interleaved with its control it was 0.7 FPS better. Steps run minutes apart drift.
- **What helped (two people, `--pipeline`):** drawing frame N while frame N+1's models run (+2-3 FPS; drawing is ~8 ms the GPU used to wait for), shrinking the frame before converting it to RGB for the person detector (+1.4), reusing crop buffers (+0.7), one face-model run for all faces (+2.1). 27.6 -> 30.5 FPS; 24.8 without `--pipeline`.
- **Give TensorRT a fixed batch range.** A model with a dynamic batch rebuilds its engine (~6 min on the Orin, app frozen) the first time a new batch size appears; set `trt_profile_min/opt/max_shapes`.
- **A half-busy GPU and CPU means waiting, not too little power.** At 21 FPS the GPU was ~60% and the CPU ~50% busy at ~20 W of 35 W: each crop and model run waits for the previous one. On the Orin the CPU and GPU share memory, so copies are cheap; the waits at each download and model run are what cost.
- **`jtop` reports the system OpenCV (4.8.0, no CUDA).** The app's venv loads the CUDA build (4.10.0) through `00-opencv-cuda.pth`; check with the venv's python, not jtop.
- **Profile before guessing (`nsys`, `py-spy` on the clip).** The GPU computed only ~23% of the time: ~900 kernel launches a frame at ~12 us each, about what a launch costs the CPU, plus waits at every crop download and model input copy. Keeping crops on the GPU (I/O binding to a fixed continuous buffer) and letting TensorRT replay each model as a CUDA graph (`trt_cuda_graph_enable`) took hand + body from 12.4 to 8.8 ms a frame: 30.8 -> 34.5 FPS with two people, same outputs.
- **OpenCV and ONNX Runtime use different CUDA streams.** OpenCV works on the default stream; ONNX Runtime's stream doesn't wait for it, and an upload from ordinary CPU memory can return before it lands. Without a wait before each run, 8 of 60 test runs read a stale input. A CUDA graph also replays its first run's addresses: one buffer set per model.
- **Compare a fast path with the same engine, not a different precision.** TensorRT FP16 and an FP32 reference differ by up to 15 model pixels on noise input; against the same FP16 engine the GPU path matched to 0.001.
- **Faces and detector on the GPU too.** Face crops cut on the GPU into one bound batch-of-2 buffer, replayed as one CUDA graph (2 faces: 6.4 -> 2.9 ms); detector input prepared on the GPU (5.4 -> 4.4 ms; no graph, part of it runs outside TensorRT). Two people, `--mirror --pipeline`: 32.6 -> 35.6 FPS. A second CUDA graph for another batch size in the same session (`gpu_graph_id`) gave wrong outputs with no error: keep one fixed batch per graph.
- **A benchmark is only valid with nothing else on the GPU.** One rerun showed everything 30% slower: the service had been restarted and ran alongside. Check `systemctl is-active party-pose` and `pgrep` before trusting numbers.
- **The method and every technique with its measured gain are collected in [orin-performance-tuning.md](orin-performance-tuning.md)** — start there for the next tuning attempt.
