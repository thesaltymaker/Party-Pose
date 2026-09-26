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
