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
