"""Per-call cost of the per-person pipeline pieces on the Orin, to decide on batching and GPU-resident crops.

Uses the app's ModelManager (same providers and TensorRT options) on a synthetic 1920x1080 frame; no camera,
no images of people. Run with the service stopped, so nothing else uses the GPU:

    env/bin/python tools/orin_bench/bench_stages.py [--out-dir /tmp/ppbench]

Models and the app's TensorRT cache are copied into --out-dir, so new engines (the face model at batch 2) are
built there and the app's cache is left as it is. A first build can take minutes.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import detect_platform  # noqa: E402
from src.model_manager import ModelManager  # noqa: E402
from src.person_detector import PersonDetector  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402
from src.smoothing import PoseSmoother  # noqa: E402
from src.types import BodyResult, FaceResult, HandResult, BoundingBox, Landmark  # noqa: E402


def bench(fn, n=200, warm=20) -> float:
    """Median ms per call."""
    for _ in range(warm):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp/ppbench')
    args = ap.parse_args()

    models = Path(args.out_dir) / 'models'
    models.mkdir(parents=True, exist_ok=True)
    for f in (REPO_ROOT / 'models').glob('*.onnx'):
        if not (models / f.name).exists():
            (models / f.name).symlink_to(f)
    if (REPO_ROOT / 'models' / 'trt_cache').is_dir() and not (models / 'trt_cache').exists():
        shutil.copytree(REPO_ROOT / 'models' / 'trt_cache', models / 'trt_cache')

    platform = detect_platform()
    mm = ModelManager(models, platform=platform)
    print(f'platform={platform}', flush=True)

    rng = np.random.default_rng(0)
    frame = cv2.cuda_GpuMat()
    frame.upload(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))
    M224 = Preprocessor.roi_affine(900, 500, 300, 0.3, 224)
    M256 = Preprocessor.roi_affine(900, 500, 600, 0.1, 256)

    rows = []

    def row(name, ms, note=''):
        rows.append((name, ms, note))
        print(f'{name:44s} {ms:7.2f} ms  {note}', flush=True)

    row('crop 224 (warp + RGB + download + float)', bench(lambda: Preprocessor.crop_rotated_nhwc(frame, M224, 224)))
    row('crop 256 (warp + RGB + download + float)', bench(lambda: Preprocessor.crop_rotated_nhwc(frame, M256, 256)))

    hand = mm.get_session('hand_landmarks')
    x224 = Preprocessor.crop_rotated_nhwc(frame, M224, 224)
    row('hand model run (batch 1)', bench(lambda: hand.run(['Identity', 'Identity_1', 'Identity_2'], {'input_1': x224})))

    body = mm.get_session('pose_landmarks')
    x256 = Preprocessor.crop_rotated_nhwc(frame, M256, 256)
    row('body model run (batch 1)', bench(lambda: body.run(['Identity', 'Identity_1'], {'input_1': x256})))

    face = mm.get_session('face_landmarks')
    f1 = rng.random((1, 256, 256, 3), dtype=np.float32)
    f2 = rng.random((2, 256, 256, 3), dtype=np.float32)
    row('face model run (batch 1)', bench(lambda: face.run(['Identity', 'Identity_1'], {'input_12': f1})))
    print('building / loading face engine for batch 2 (can take minutes the first time)...', flush=True)
    t = time.perf_counter()
    face.run(['Identity', 'Identity_1'], {'input_12': f2})
    print(f'  first batch-2 run took {time.perf_counter() - t:.1f} s', flush=True)
    b2 = bench(lambda: face.run(['Identity', 'Identity_1'], {'input_12': f2}))
    row('face model run (batch 2, both faces)', b2, f'= {b2 / 2:.2f} ms per face')
    row('face model run (batch 1, after batch 2)', bench(lambda: face.run(['Identity', 'Identity_1'], {'input_12': f1})),
        'engine with a batch range may be slower at batch 1')

    person = PersonDetector(mm)
    row('person detector (whole frame)', bench(lambda: person.process(frame, 1920, 1080), n=100))

    lms478 = rng.random((478, 5)).astype(np.float32) * 256
    lms21 = rng.random((21, 5)).astype(np.float32) * 224
    row('landmarks to frame: face 478 pts', bench(lambda: Preprocessor.to_image_space_affine(lms478, M256)))
    row('landmarks to frame: hand 21 pts', bench(lambda: Preprocessor.to_image_space_affine(lms21, M224)))

    def people(n):
        def mk(k, ox):
            pts = np.c_[ox + rng.random(k) * 100, 300 + rng.random(k) * 100]
            return [Landmark(x, y, 0.0) for x, y in pts.tolist()], pts
        faces, hands, bodies = [], [], []
        for p in range(n):
            lm, pts = mk(478, p * 500)
            faces.append(FaceResult(BoundingBox(0, 0, 1, 1, 1), lm, person_id=p, pts=pts))
            lm, pts = mk(39, p * 500)
            bodies.append(BodyResult(lm, person_id=p, pts=pts))
            for h in range(2):
                lm, pts = mk(21, p * 500 + h * 200)
                hands.append(HandResult(BoundingBox(0, 0, 1, 1, 1), lm, person_id=p, pts=pts))
        return faces, hands, bodies

    ps, clock = PoseSmoother(), [0.0]
    data = people(2)

    def smooth():
        clock[0] += 1 / 26
        ps.smooth(*data, clock[0])
    row('smoothing, 2 people', bench(smooth))

    print('\nSUMMARY ' + ' | '.join(f'{n}={ms:.2f}' for n, ms, _ in rows), flush=True)


if __name__ == '__main__':
    main()
