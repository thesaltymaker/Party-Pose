# Orin benchmark tools

These are diagnostic scripts used for issue #2 (see `docs/orin-fps-issue-2.md`). Run them on the Orin from anywhere; they locate the repo themselves. Output (logs, grabbed frames, scratch TensorRT caches) goes to `/tmp/ppbench` by default.

For comparable numbers, pin the clocks first with `sudo jetson_clocks`. Otherwise the GPU governor downclocks to about 408 MHz under poser's bursty load.

## run_poser.sh: whole app

```sh
tools/orin_bench/run_poser.sh after_fix                  # 90 s run, all modalities
DURATION=60 tools/orin_bench/run_poser.sh noface --no-face
```

It writes `<name>.log`, `<name>.freq` (GPU/CPU clock samples) and `<name>.tegra` (tegrastats) to the output directory. When the run ends it prints the FPS values, the last `[STAGES ms/frame]` lines, `[PROVIDERS]`, and a histogram of GPU clocks. The stage lines include detection counts per frame; FPS depends heavily on how many bodies are detected.

## bench_models.py: pieces in isolation

```sh
B=tools/orin_bench/bench_models.py
python $B grab                                      # save a real camera frame (camera must be free)
python $B pre                                       # GPU preprocessing per model
python $B landmarks                                 # to_image_space cost per face/body/hand
python $B models person_detector face_landmarks     # sessions exactly as poser.py creates them
python $B models person_detector --provider cuda    # compare with the CUDA EP
python $B models person_detector --provider trt --no-extra-opts --cache-tag full   # full-model TRT (fails after ~6 min)
python $B post-ops                                  # op types found only after NMS in the person detector
```

After a `grab`, `models person_detector` also prints the detections on the real frame, so you can check that two providers give the same output. The `trt` provider builds engines in `<out-dir>/trt_cache_<tag>` and leaves `models/trt_cache` alone.
