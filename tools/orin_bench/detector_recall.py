"""How often the person detector finds people in a recorded clip (no drawing, nothing saved).

  env/bin/python tools/orin_bench/detector_recall.py CLIP [--detector FILE] [--bgr] [--stretch] [--gain G] [--gamma G]

Runs the detector (TensorRT on the Orin, as the app) on every frame, mirrored as the service shows it, with the
app's input (letterboxed, RGB, 0-255) or a variant, keeps body boxes as the app does (score >= 0.3, nested boxes
dropped) and prints: body boxes per frame, the share of frames with at least k boxes, frames with a big box
(>= 0.7 of the frame height: someone close to the camera), frames with a lamp-like box (narrow, centre in the
right sixth: x > 1600 of 1920), the share of frames with a box centred in each sixth of the width (320 px bands
at 1920), and the detector's ms per frame. Half-size clips (testdata/clips_half) give nearly the same boxes.
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from poser import _drop_nested_bodies
from src.config import detect_platform
from src.model_manager import ModelManager
from src.person_detector import PersonDetector
from src.preprocessor import Preprocessor
from src.raw_video import RawVideoReader
from src.types import BoundingBox


def main():
    p = argparse.ArgumentParser()
    p.add_argument('clip')
    p.add_argument('--detector', default='')
    p.add_argument('--bgr', action='store_true', help='feed BGR (as the model zoo demo) instead of RGB')
    p.add_argument('--stretch', action='store_true', help='resize to the input without keeping the aspect ratio')
    p.add_argument('--gain', type=float, default=1.0, help='multiply pixel values (saturating at 255)')
    p.add_argument('--gamma', type=float, default=1.0, help='brighten dark pixels: 255 * (v / 255) ** (1 / gamma)')
    p.add_argument('--area', action='store_true', help='shrink with INTER_AREA (averages pixels) instead of INTER_LINEAR')
    p.add_argument('--no-mirror', dest='mirror', action='store_false')
    p.add_argument('--gray', action='store_true', help='grayscale, as the app shows the picture in night mode')
    a = p.parse_args()

    if detect_platform() != 'orin' and hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()  # CUDA/cuDNN from pip packages (other machines used for recall checks)
    mm = ModelManager(REPO_ROOT / 'models', platform=detect_platform(), detector=a.detector)
    sess = mm.get_session('person_detector')
    _, in_h, in_w, _ = sess.get_inputs()[0].shape
    lut = np.clip(np.arange(256) * a.gain, 0, 255)
    lut = (255 * (lut / 255) ** (1 / a.gamma)).astype(np.uint8)
    reader = RawVideoReader(REPO_ROOT / a.clip)
    fw, fh = reader.width, reader.height
    if a.stretch:
        sx, sy, px, py = in_w / fw, in_h / fh, 0, 0
    else:
        s, px, py = Preprocessor.letterbox_params(fw, fh, in_w, in_h)
        sx = sy = s
        nw, nh = int(round(fw * s)), int(round(fh * s))
    img = np.zeros((in_h, in_w, 3), np.uint8)
    interp = cv2.INTER_AREA if a.area else cv2.INTER_LINEAR

    counts = collections.Counter()
    bands = collections.Counter()
    big = lamp = heads = n = 0
    ms = []
    while True:
        try:
            frame = reader.read()
        except EOFError:
            break
        n += 1
        if a.mirror:
            frame = cv2.flip(frame, 1)
        if a.gray:
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if a.stretch:
            img = cv2.resize(frame, (in_w, in_h), interpolation=interp)
        else:
            img[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=interp)
        x = img if a.bgr else img[..., ::-1]
        if a.gain != 1.0 or a.gamma != 1.0:
            x = cv2.LUT(np.ascontiguousarray(x), lut)
        x = np.ascontiguousarray(x, dtype=np.float32)[None]
        t = time.perf_counter()
        raw = sess.run(['batchno_classid_score_x1y1x2y2'], {'input': x})[0]
        ms.append((time.perf_counter() - t) * 1000)

        bodies, head_boxes = [], []
        for _, cls, score, x1, y1, x2, y2 in raw:
            if score < PersonDetector.CONF_THRESHOLD:
                continue
            fx1, fy1 = max(0.0, (x1 - px) / sx), max(0.0, (y1 - py) / sy)
            fx2, fy2 = min(float(fw), (x2 - px) / sx), min(float(fh), (y2 - py) / sy)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            box = BoundingBox(x=fx1, y=fy1, w=fx2 - fx1, h=fy2 - fy1, confidence=float(score))
            if int(cls) == PersonDetector.CLASS_BODY:
                bodies.append(box)
            elif int(cls) == PersonDetector.CLASS_HEAD:
                head_boxes.append(box)
        bodies = _drop_nested_bodies(bodies, head_boxes)
        heads += len(head_boxes)
        counts[len(bodies)] += 1
        big += any(b.h >= 0.7 * fh for b in bodies)
        lamp += any(b.x + b.w / 2 > fw * 5 / 6 and b.w / b.h < 0.45 for b in bodies)
        for band in {min(5, int((b.x + b.w / 2) * 6 // fw)) for b in bodies}:
            bands[band] += 1

    total = sum(k * v for k, v in counts.items())
    at_least = ' '.join(f'>={k}:{100 * sum(v for c, v in counts.items() if c >= k) / n:.0f}%' for k in (1, 2, 3, 4))
    band_txt = ' '.join(f'{b * 320}:{100 * bands[b] / n:.0f}%' for b in range(6))
    warm = sorted(ms[20:]) or sorted(ms)
    print(f'[RECALL] {a.clip} detector={a.detector or "default"} input={in_w}x{in_h} bgr={a.bgr} '
          f'stretch={a.stretch} area={a.area} gain={a.gain} gamma={a.gamma} frames={n}')
    print(f'[RECALL] bodies/frame={total / n:.2f} {at_least} big={100 * big / n:.0f}% lamp={100 * lamp / n:.0f}% '
          f'heads/frame={heads / n:.2f} | bands {band_txt} | det ms mean={sum(warm) / len(warm):.2f} '
          f'p90={warm[int(0.9 * len(warm))]:.2f}')


if __name__ == '__main__':
    main()
