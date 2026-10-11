"""Face model (batched) and person detector: current path vs crops kept on the GPU (I/O binding), and for the
face also CUDA graphs with one graph per batch size (ONNX Runtime run option gpu_graph_id).

Prints median ms and the largest output difference against the current path. Synthetic frame, no camera.
Run on the Orin with the service stopped:

    env/bin/python tools/orin_bench/bench_face_detector.py
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
from src.face_processor import FaceProcessor  # noqa: E402
from src.model_manager import ModelManager  # noqa: E402
from src.person_detector import PersonDetector  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402
from src.types import BoundingBox  # noqa: E402


def bench(fn, n=200, warm=20) -> float:
    for _ in range(warm):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def face_crop_rect(head: BoundingBox, frame_w: int, frame_h: int):
    """Preprocessor.crop_roi_nhwc's rectangle for pad_fraction=0 (the face crop)."""
    x, y = max(0, head.x), max(0, head.y)
    w, h = min(frame_w - x, head.w), min(frame_h - y, head.h)
    return int(x), int(y), int(w), int(h)


def main() -> None:
    platform = detect_platform()
    mm = ModelManager(REPO_ROOT / 'models', platform=platform)
    rng = np.random.default_rng(0)
    frame = cv2.cuda_GpuMat()
    frame.upload(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))
    stream = cv2.cuda.Stream.Null()

    # ---------------- face ----------------
    heads = [BoundingBox(600, 200, 140, 170, 0.9), BoundingBox(1200, 260, 120, 150, 0.8)]
    fp = FaceProcessor(mm)
    sess = mm.get_session('face_landmarks')
    outs = ['Identity', 'Identity_1']

    def face_now():
        crops = [Preprocessor.crop_roi_nhwc(frame, h, 256, 256, pad_fraction=0.0, frame_w=1920, frame_h=1080)[0]
                 for h in heads]
        return sess.run(outs, {'input_12': np.concatenate(crops)})

    ref = face_now()

    batch = cv2.cuda.createContinuous(4 * 256, 256, cv2.CV_32FC3, cv2.cuda_GpuMat())
    small = [cv2.cuda_GpuMat(256, 256, cv2.CV_8UC3) for _ in range(2)]
    rgb = [cv2.cuda_GpuMat(256, 256, cv2.CV_8UC3) for _ in range(2)]

    def fill():
        for k, h in enumerate(heads):
            x, y, w, hh = face_crop_rect(h, 1920, 1080)
            small[k] = cv2.cuda.resize(cv2.cuda_GpuMat(frame, (x, y, w, hh)), (256, 256), small[k])
            rgb[k] = cv2.cuda.cvtColor(small[k], cv2.COLOR_BGR2RGB, rgb[k])
            slot = batch.rowRange(k * 256, (k + 1) * 256)
            out = rgb[k].convertTo(cv2.CV_32FC3, slot, 1 / 255.0, 0.0)
            assert out.cudaPtr() == slot.cudaPtr()
        stream.waitForCompletion()

    def bind(session, n):
        b = session.io_binding()
        b.bind_input('input_12', 'cuda', 0, np.float32, [n, 256, 256, 3], batch.cudaPtr())
        vals = []
        for o in session.get_outputs():
            if o.name in outs:
                shape = [n] + [d for d in o.shape[1:]]
                v = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, 'cuda', 0)
                b.bind_ortvalue_output(o.name, v)
                vals.append(v)
        return b, vals

    b2, v2 = bind(sess, 2)

    def face_gpu():
        fill()
        sess.run_with_iobinding(b2)
        return [v.numpy() for v in v2]

    got = face_gpu()
    d = max(float(np.abs(g.reshape(r.shape) - r).max()) for g, r in zip(got, ref))
    line = f'face x2   now {bench(face_now):.2f} ms | GPU crops {bench(face_gpu):.2f} ms (diff {d:.4f})'

    if 'TensorrtExecutionProvider' in sess.get_providers():
        opts = {'trt_max_workspace_size': 2 * 1024 ** 3, 'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(REPO_ROOT / 'models' / 'trt_cache'),
                'trt_profile_min_shapes': 'input_12:1x256x256x3', 'trt_profile_opt_shapes': 'input_12:2x256x256x3',
                'trt_profile_max_shapes': 'input_12:4x256x256x3', 'trt_cuda_graph_enable': True}
        gs = ort.InferenceSession(str(REPO_ROOT / 'models' / mm._MODEL_FILES['face_landmarks']),
                                  providers=[('TensorrtExecutionProvider', opts), 'CUDAExecutionProvider'])
        assert gs.get_providers()[0] == 'TensorrtExecutionProvider'
        g2, gv2 = bind(gs, 2)
        g1, gv1 = bind(gs, 1)
        ro2, ro1 = ort.RunOptions(), ort.RunOptions()
        ro2.add_run_config_entry('gpu_graph_id', '2')
        ro1.add_run_config_entry('gpu_graph_id', '1')

        def face_graph2():
            fill()
            gs.run_with_iobinding(g2, ro2)
            return [v.numpy() for v in gv2]

        def face_graph1():
            fill()
            gs.run_with_iobinding(g1, ro1)
            return [v.numpy() for v in gv1]

        try:
            got2 = face_graph2()
            got1 = face_graph1()   # second graph, other batch size, same session
            got2b = face_graph2()  # back to the first graph
            d2 = max(float(np.abs(g.reshape(r.shape) - r).max()) for g, r in zip(got2b, ref))
            d1 = float(np.abs(got1[0].reshape(-1)[:1434] - ref[0].reshape(2, -1)[0]).max())
            line += (f' | CUDA graph x2 {bench(face_graph2):.2f} ms (diff {d2:.4f}), '
                     f'x1 {bench(face_graph1):.2f} ms (diff vs face 1 {d1:.4f})')
        except Exception as e:  # noqa: BLE001
            line += f' | CUDA graph per batch size failed: {str(e).splitlines()[0][:160]}'
    print(line, flush=True)

    # ---------------- person detector ----------------
    det = PersonDetector(mm)
    dsess = mm.get_session('person_detector')
    dname = dsess.get_inputs()[0].name
    dout = 'batchno_classid_score_x1y1x2y2'

    def det_now():
        arr = Preprocessor.full_frame_nhwc(frame, det.INPUT_W, det.INPUT_H, normalize=False)
        return dsess.run([dout], {dname: arr})[0]

    scale, pad_x, pad_y = Preprocessor.letterbox_params(1920, 1080, det.INPUT_W, det.INPUT_H)
    nw, nh = int(round(1920 * scale)), int(round(1080 * scale))
    dbuf = cv2.cuda.createContinuous(det.INPUT_H, det.INPUT_W, cv2.CV_32FC3, cv2.cuda_GpuMat())
    dsmall, drgb, dpad = (cv2.cuda_GpuMat(nh, nw, cv2.CV_8UC3), cv2.cuda_GpuMat(nh, nw, cv2.CV_8UC3),
                          cv2.cuda_GpuMat(det.INPUT_H, det.INPUT_W, cv2.CV_8UC3))
    db = dsess.io_binding()
    db.bind_input(dname, 'cuda', 0, np.float32, [1, det.INPUT_H, det.INPUT_W, 3], dbuf.cudaPtr())
    db.bind_output(dout, 'cpu')

    def det_gpu():
        nonlocal dsmall, drgb, dpad
        dsmall = cv2.cuda.resize(frame, (nw, nh), dsmall)
        drgb = cv2.cuda.cvtColor(dsmall, cv2.COLOR_BGR2RGB, drgb)
        dpad = cv2.cuda.copyMakeBorder(drgb, pad_y, det.INPUT_H - nh - pad_y, pad_x, det.INPUT_W - nw - pad_x,
                                       cv2.BORDER_CONSTANT, dpad, (0, 0, 0))
        out = dpad.convertTo(cv2.CV_32FC3, dbuf, 1.0, 0.0)
        assert out.cudaPtr() == dbuf.cudaPtr()
        stream.waitForCompletion()
        dsess.run_with_iobinding(db)
        return db.copy_outputs_to_cpu()[0]

    a, b = det_now(), det_gpu()
    dd = float(np.abs(a - b).max()) if a.shape == b.shape else float('nan')
    print(f'detector  now {bench(det_now):.2f} ms | GPU input {bench(det_gpu):.2f} ms '
          f'(outputs {a.shape} vs {b.shape}, diff {dd:.4f})', flush=True)


if __name__ == '__main__':
    main()
