"""Does keeping a crop on the GPU and handing it to ONNX Runtime by address (I/O binding) beat downloading it?

(a) now: warp + RGB on the GPU, download, float on the CPU, session.run(numpy) (ORT copies it back to the GPU)
(b) GPU: warp + RGB + float/255 on the GPU into a continuous buffer, io_binding.bind_input(GPU pointer),
    outputs bound to the CPU
(c) as (b), in a TensorRT session with CUDA graphs (trt_cuda_graph_enable): the model's kernel launches are
    recorded once and replayed as one; outputs go to fixed GPU buffers (needed for the replay), then to the CPU

Prints the median ms of each for the hand (224) and body (256) models and the largest difference between
their landmark outputs. Synthetic frame, no camera. Run on the Orin with the service stopped:

    env/bin/python tools/orin_bench/bench_iobinding.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import detect_platform  # noqa: E402
from src.model_manager import ModelManager  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402


def bench(fn, n=300, warm=30) -> float:
    for _ in range(warm):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main() -> None:
    mm = ModelManager(REPO_ROOT / 'models', platform=detect_platform())
    rng = np.random.default_rng(0)
    frame = cv2.cuda_GpuMat()
    frame.upload(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))

    for model, size, inp, outs in (('hand_landmarks', 224, 'input_1', ['Identity', 'Identity_1', 'Identity_2']),
                                   ('pose_landmarks', 256, 'input_1', ['Identity', 'Identity_1'])):
        sess = mm.get_session(model)
        M = Preprocessor.roi_affine(900, 500, 300 if size == 224 else 600, 0.3, size)
        Minv = cv2.invertAffineTransform(M)

        def now():
            return sess.run(outs, {inp: Preprocessor.crop_rotated_nhwc(frame, M, size)})

        warp = cv2.cuda_GpuMat(size, size, cv2.CV_8UC3)
        rgb = cv2.cuda_GpuMat(size, size, cv2.CV_8UC3)
        flt = cv2.cuda.createContinuous(size, size, cv2.CV_32FC3, cv2.cuda_GpuMat())
        assert flt.isContinuous(), 'float buffer must be one block for ORT'
        binding = sess.io_binding()
        binding.bind_input(inp, 'cuda', 0, np.float32, [1, size, size, 3], flt.cudaPtr())
        for o in outs:
            binding.bind_output(o, 'cpu')

        def gpu():
            nonlocal warp, rgb
            warp = cv2.cuda.warpAffine(frame, Minv, (size, size), warp, flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
            rgb = cv2.cuda.cvtColor(warp, cv2.COLOR_BGR2RGB, rgb)
            out = rgb.convertTo(cv2.CV_32FC3, flt, 1 / 255.0, 0.0)
            assert out.cudaPtr() == flt.cudaPtr(), 'convertTo did not write into the bound buffer'
            sess.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()

        a, b = now(), gpu()
        diff = float(np.max(np.abs(a[0] - b[0])))
        line = (f'{model:15s} now {bench(now):.2f} ms | GPU-resident {bench(gpu):.2f} ms '
                f'(landmark diff {diff:.4f})')

        if 'TensorrtExecutionProvider' in sess.get_providers():
            # As ModelManager builds them (options read back from a session come back as strings ORT rejects).
            trt_opts = {'trt_max_workspace_size': 2 * 1024 ** 3, 'trt_fp16_enable': True,
                        'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(REPO_ROOT / 'models' / 'trt_cache'),
                        'trt_cuda_graph_enable': True}
            gsess = ort.InferenceSession(sess._model_path, sess_options=sess.get_session_options(),
                                         providers=[('TensorrtExecutionProvider', trt_opts),
                                                    'CUDAExecutionProvider', 'CPUExecutionProvider'])
            assert gsess.get_providers()[0] == 'TensorrtExecutionProvider', gsess.get_providers()
            gbind = gsess.io_binding()
            gbind.bind_input(inp, 'cuda', 0, np.float32, [1, size, size, 3], flt.cudaPtr())
            gout = []
            for o, meta in zip(outs, [x for x in gsess.get_outputs() if x.name in outs]):
                v = ort.OrtValue.ortvalue_from_shape_and_type([d if isinstance(d, int) else 1 for d in meta.shape],
                                                              np.float32, 'cuda', 0)
                gbind.bind_ortvalue_output(o, v)
                gout.append(v)

            def graph():
                nonlocal warp, rgb
                warp = cv2.cuda.warpAffine(frame, Minv, (size, size), warp, flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
                rgb = cv2.cuda.cvtColor(warp, cv2.COLOR_BGR2RGB, rgb)
                rgb.convertTo(cv2.CV_32FC3, flt, 1 / 255.0, 0.0)
                gsess.run_with_iobinding(gbind)
                return [v.numpy() for v in gout]

            c = graph()
            gdiff = float(np.max(np.abs(a[0] - c[0])))
            line += f' | CUDA graph {bench(graph):.2f} ms (landmark diff {gdiff:.4f})'
        print(line, flush=True)


if __name__ == '__main__':
    main()
