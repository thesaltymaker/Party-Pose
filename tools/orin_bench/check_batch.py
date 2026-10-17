"""Compare --pose-batch / --hand-batch results with one model run per crop on real clip frames (the app's models
and sessions: TensorRT FP16 on the Orin). Prints the largest and median differences of the drawn points in frame
pixels, and how often one path found a pose or hand the other did not.

    env/bin/python tools/orin_bench/check_batch.py testdata/clips/three_people [POSE_SLOTS HAND_SLOTS]

Every 5th frame: the detector's body boxes give each person's pose crop (one run each, as for a new person); the
next frame runs the poses from those crops one at a time and batched, and the hands from the found poses.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.body_processor import BodyProcessor  # noqa: E402
from src.config import detect_platform  # noqa: E402
from src.hand_processor import HandProcessor  # noqa: E402
from src.model_manager import ModelManager  # noqa: E402
from src.person_detector import PersonDetector  # noqa: E402
from src.raw_video import RawVideoReader  # noqa: E402


def main() -> None:
    clip = sys.argv[1]
    pose_slots = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    hand_slots = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    mm = ModelManager(Path(__file__).resolve().parents[2] / 'models', platform=detect_platform())
    det = PersonDetector(mm)
    body1, bodyn = BodyProcessor(mm), BodyProcessor(mm, pose_slots)
    hand1, handn = HandProcessor(mm), HandProcessor(mm, hand_slots)
    video = RawVideoReader(clip)
    gpu = cv2.cuda_GpuMat()
    rois = []
    pose_d, hand_d = [], []
    pose_odd = hand_odd = poses = hands = 0
    for i in range(video.count):
        gpu.upload(video.read())
        w, h = video.width, video.height
        if rois:
            one = [body1.process(gpu, _box(r), w, h, False, roi=r) for r in rois]
            many = bodyn.process_many(gpu, rois)
            for a, b in zip(one, many):
                poses += 1
                if (a is None) != (b is None):
                    pose_odd += 1
                elif a is not None:
                    pose_d.append(float(np.abs(a.pts[:33] - b.pts[:33]).max()))
            found = [a for a, b in zip(one, many) if a is not None and b is not None]
            h1 = [hand1.process(gpu, a.landmarks, (None, None)) for a in found]
            hn = handn.process_many(gpu, [(a.landmarks, (None, None)) for a in found])
            for pa, pb in zip(h1, hn):
                for a, b in zip(pa, pb):
                    hands += 1
                    if (a is None) != (b is None):
                        hand_odd += 1
                    elif a is not None:
                        hand_d.append(float(np.abs(a.pts - b.pts).max()))
            rois = []
        if i % 5 == 0:
            boxes = det.process(gpu, w, h).body_boxes
            rois = [b.roi for b in (body1.process(gpu, box, w, h, False) for box in boxes) if b is not None]
    for name, d, odd, n in (('pose', pose_d, pose_odd, poses), ('hand', hand_d, hand_odd, hands)):
        d = np.array(d or [0.0])
        print(f'{name}: {n} crops, found by one path only {odd}, point difference px: max {d.max():.2f} '
              f'median {np.median(d):.3f} p99 {np.percentile(d, 99):.2f}', flush=True)


def _box(roi):
    from src.types import BoundingBox
    return BoundingBox(roi[0] - roi[2] / 2, roi[1] - roi[2] / 2, roi[2], roi[2], 1.0)


if __name__ == '__main__':
    main()
