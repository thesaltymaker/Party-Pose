"""Time one model run at batch 1 (the app's model file) and at batch N (the .dynbatch.onnx copy from
batch_model.py), TensorRT FP16 + CUDA graph + I/O binding as in the app, inputs and outputs on the GPU.

    env/bin/python tools/orin_bench/bench_batch.py pose 1 2 3 4
    env/bin/python tools/orin_bench/bench_batch.py hand 1 2 4 6 8

Each batch size gets its own engine cache in /tmp/bench_batch_cache/<model>_<N> (first run builds it, minutes).
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parents[2] / 'models'
FILES = {'pose': ('pose_landmarks_detector_opset15', 256, ['Identity', 'Identity_1']),
         'hand': ('hand_landmarks_detector_opset15', 224, ['Identity', 'Identity_1', 'Identity_2'])}


def session(name: str, n: int) -> ort.InferenceSession:
    stem, size, _ = FILES[name]
    opts = {'trt_max_workspace_size': 2 * 1024 ** 3, 'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
            'trt_engine_cache_path': f'/tmp/bench_batch_cache/{name}_{n}', 'trt_cuda_graph_enable': True}
    if n > 1:
        shape = f'input_1:{n}x{size}x{size}x3'
        opts.update(trt_profile_min_shapes=shape, trt_profile_opt_shapes=shape, trt_profile_max_shapes=shape)
    Path(opts['trt_engine_cache_path']).mkdir(parents=True, exist_ok=True)
    return ort.InferenceSession(str(MODELS / (stem + ('.onnx' if n == 1 else '.dynbatch.onnx'))),
                                providers=[('TensorrtExecutionProvider', opts), 'CUDAExecutionProvider'])


def main() -> None:
    name, sizes = sys.argv[1], [int(a) for a in sys.argv[2:]]
    _, size, outs = FILES[name]
    for n in sizes:
        s = session(name, n)
        b = s.io_binding()
        x = ort.OrtValue.ortvalue_from_numpy(np.random.default_rng(0).random((n, size, size, 3), dtype=np.float32),
                                             'cuda', 0)
        b.bind_ortvalue_input('input_1', x)
        meta = {o.name: o.shape for o in s.get_outputs()}
        vals = [ort.OrtValue.ortvalue_from_shape_and_type([n] + [d for d in meta[o][1:]], np.float32, 'cuda', 0)
                for o in outs]
        for o, v in zip(outs, vals):
            b.bind_ortvalue_output(o, v)
        for _ in range(20):
            s.run_with_iobinding(b)
            [v.numpy() for v in vals]
        times = []
        for _ in range(300):
            t = time.perf_counter()
            s.run_with_iobinding(b)
            [v.numpy() for v in vals]
            times.append((time.perf_counter() - t) * 1000)
        med = float(np.median(times))
        print(f'{name} batch {n}: median {med:.2f} ms/run, {med / n:.2f} ms per crop', flush=True)


if __name__ == '__main__':
    main()
