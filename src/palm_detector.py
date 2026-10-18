"""MediaPipe's palm detector, used to find a lost hand again (--palm-refind).

A hand cropped from the pose's wrist, pinky and index points is often outside the crop when the arms cross the
body (the pose's wrist point lies up the forearm). MediaPipe's own hand tracker finds a lost hand with its palm
detector instead: here it runs on a square crop of the person's upper body, and each palm found near one of the
person's arms gives that side's hand crop the way MediaPipe makes it (wrist and middle finger base keypoints).

Model: hand_detector_opset15.onnx, input [1, 192, 192, 3] RGB in [0, 1], outputs Identity [1, 2016, 18] (box
centre, size and 7 keypoints, in model pixels relative to the anchor) and Identity_1 [1, 2016, 1] (score logits).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from src.gpu_model import gpu_model
from src.hand_processor import Roi, _rect
from src.preprocessor import Preprocessor
from src.types import Landmark

SIZE = 192
SCORE = 0.5  # MediaPipe's min_detection_confidence
OVERLAP = 0.3  # IoU above which the lower-scored of two palms is dropped (MediaPipe's NMS threshold)
ARM_FACTOR = 0.5  # a palm's wrist must be within this many shoulder widths of the person's forearm line


@dataclass
class Palm:
    """One palm in frame pixels: score, box centre, box width and height, wrist and middle finger base points."""
    score: float
    centre: np.ndarray
    w: float
    h: float
    wrist: np.ndarray
    middle: np.ndarray


def palm_anchors() -> np.ndarray:
    """Anchor centres [2016, 2] in model pixels: MediaPipe's SSD anchors for the 192 palm model (strides 8, 16, 16,
    16; the three stride-16 layers share one grid, 2 anchors per layer and cell)."""
    out = []
    for stride, per_cell in ((8, 2), (16, 6)):
        n = SIZE // stride
        ys, xs = np.mgrid[0:n, 0:n]
        c = (np.stack([xs, ys], axis=-1).reshape(-1, 2) + 0.5) * stride
        out.append(np.repeat(c, per_cell, axis=0))
    return np.concatenate(out).astype(np.float32)


_ANCHORS = palm_anchors()


def decode_palms(boxes: np.ndarray, logits: np.ndarray, M: np.ndarray, score: float = SCORE,
                 overlap: float = OVERLAP) -> List[Palm]:
    """Palms from the model's outputs ([2016, 18] and [2016] or [2016, 1]), mapped to the frame through the crop's
    roi_affine matrix M. Highest score first; a palm overlapping a higher-scored one by more than `overlap` is
    dropped."""
    s = 1 / (1 + np.exp(-np.clip(logits.reshape(-1), -100, 100)))
    keep = np.flatnonzero(s >= score)
    keep = keep[np.argsort(-s[keep])]
    k = math.hypot(M[0, 0], M[1, 0])
    out: List[Palm] = []
    taken = []  # (x0, y0, x1, y1) in model pixels
    for i in keep:
        r = boxes[i]
        cx, cy = r[0] + _ANCHORS[i, 0], r[1] + _ANCHORS[i, 1]
        box = (cx - r[2] / 2, cy - r[3] / 2, cx + r[2] / 2, cy + r[3] / 2)
        if any(_iou(box, t) > overlap for t in taken):
            continue
        taken.append(box)
        pts = np.array([[cx, cy], r[4:6] + _ANCHORS[i], r[8:10] + _ANCHORS[i]]) @ M[:, :2].T + M[:, 2]
        out.append(Palm(float(s[i]), pts[0], float(r[2] * k), float(r[3] * k), pts[1], pts[2]))
    return out


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def palm_roi(p: Palm) -> Roi:
    """MediaPipe's hand crop from a palm: upright from wrist to middle finger base, 2.6x the box's longer side,
    moved half the box height towards the fingers."""
    angle = Preprocessor.roi_from_alignment(p.wrist, p.middle)[3]
    return _rect(float(p.centre[0]), float(p.centre[1]), p.w, p.h, angle, 2.6, -0.5)


def _xy(lm: Landmark) -> np.ndarray:
    return np.array([lm.x, lm.y])


def _shoulder_width(pose: Sequence[Landmark]) -> float:
    return float(np.hypot(*(_xy(pose[11]) - _xy(pose[12]))))


def upper_body_roi(pose: Sequence[Landmark], scale: float) -> Roi:
    """An unrotated square crop of the upper body: centred a quarter of the way from the shoulders' to the hips'
    midpoint, `scale` shoulder widths wide (at least half the shoulder-hip distance, for people seen side on)."""
    sh = (_xy(pose[11]) + _xy(pose[12])) / 2
    hip = (_xy(pose[23]) + _xy(pose[24])) / 2
    width = max(_shoulder_width(pose), 0.5 * float(np.hypot(*(hip - sh))))
    c = sh + 0.25 * (hip - sh)
    return float(c[0]), float(c[1]), scale * width, 0.0


def _arm_distance(p: np.ndarray, pose: Sequence[Landmark], side: int) -> float:
    """Distance from p to the side's elbow-wrist segment extended half a forearm past the wrist."""
    e, w = _xy(pose[13 + side]), _xy(pose[15 + side])
    end = w + 0.5 * (w - e)
    d = end - e
    t = float(np.clip(np.dot(p - e, d) / max(float(np.dot(d, d)), 1e-9), 0, 1))
    return float(np.hypot(*(p - (e + t * d))))


def assign_palms(palms: Sequence[Palm], pose: Sequence[Landmark], sides: Sequence[int],
                 factor: float = ARM_FACTOR) -> Dict[int, Palm]:
    """At most one palm per side in `sides`: each palm whose wrist is within factor x shoulder width (at least
    10 px) of one of this person's forearm lines, nearest pairs first; a palm goes to one side only."""
    max_dist = max(10.0, factor * _shoulder_width(pose))
    pairs = sorted((_arm_distance(p.wrist, pose, s), s, i) for i, p in enumerate(palms) for s in sides)
    out: Dict[int, Palm] = {}
    used = set()
    for d, s, i in pairs:
        if d <= max_dist and s not in out and i not in used:
            out[s] = palms[i]
            used.add(i)
    return out


class PalmDetector:
    """The palm model on one person's upper body (crop `SCALE` shoulder widths wide), palms given to their arms."""

    SCALE = 4.0

    def __init__(self, model_manager) -> None:
        self.model_manager = model_manager
        self._model = None  # created on first use (loads the session)

    def find(self, frame_gpu, pose: Sequence[Landmark], sides: Sequence[int]) -> Dict[int, Palm]:
        if self._model is None:
            self._model = gpu_model(self.model_manager, 'palm_detector', 'input_1', ['Identity', 'Identity_1'], SIZE)
        M = Preprocessor.roi_affine(*upper_body_roi(pose, self.SCALE), SIZE)
        boxes, logits = self._model.run(frame_gpu, M)
        return assign_palms(decode_palms(boxes[0], logits[0], M), pose, sides)
