"""Smoothing of body crops and landmarks across frames, per person.

The detector's body box flips between a small and a big box (Orin: 117x239 and 220x510, each held for a few
frames; issue #6), so the pose crop moved and every landmark jumped even when the person stood still. The body
crop is the union of the person's boxes from the last CROP_WINDOW seconds: it stays on the big box through
flips, grows at once when the box grows, and covers a fast dancer (the pose model letterboxes the crop to a
square, so extra width on a tall box costs nothing). The head box is not smoothed: the face model loses faces
when the head is off-centre in the crop (unpadded crop 37% kept vs 0-5% padded, issue #6/#7).

Landmarks of faces, hands and bodies go through a One Euro filter (Casiez et al. 2012, the filter MediaPipe
uses): a low-pass filter whose cutoff rises with speed, so a still person is smoothed hard and a fast dancer
gets little lag.

Speed is measured in object sizes per second (box height for boxes, landmark extent for landmarks), so near
and far people get the same smoothing. A filter that hasn't been fed for `max_gap` seconds (person lost) starts
again from the new value instead of sliding across the screen.

Hands have no track of their own and the model's left/right flips, so each hand is matched to the nearest hand
of the same person from earlier frames by wrist position.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Hashable, List, Tuple

import numpy as np

from src.types import BodyResult, BoundingBox, FaceResult, HandResult

MIN_CUTOFF = 0.3  # Hz: cutoff when still (lower = steadier, more lag on slow movement)
BETA = 1.5        # cutoff added per (object size / second) of speed (higher = less lag when fast)
D_CUTOFF = 1.0    # Hz: cutoff for the speed estimate
CROP_WINDOW = 0.3  # s of body boxes joined into the crop; longer than a flip is held (3-6 frames at 26 FPS)


def _alpha(cutoff, dt: float):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuro:
    def __init__(self, x: np.ndarray, t: float) -> None:
        self.x = x.astype(np.float64)
        self.dx = np.zeros_like(self.x)
        self.t = t

    def __call__(self, x: np.ndarray, t: float, scale: float, min_cutoff: float, beta: float) -> np.ndarray:
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += _alpha(D_CUTOFF, dt) * ((x - self.x) / dt - self.dx)
        cutoff = min_cutoff + beta * np.abs(self.dx) / max(scale, 1.0)
        self.x += _alpha(cutoff, dt) * (x - self.x)
        return self.x


class Smoother:
    """One filter per key. Call with the key's raw values each frame; returns the smoothed values."""

    def __init__(self, min_cutoff: float = MIN_CUTOFF, beta: float = BETA, max_gap: float = 0.5) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.max_gap = max_gap
        self.filters: Dict[Hashable, _OneEuro] = {}

    def __call__(self, key: Hashable, x: np.ndarray, t: float, scale: float) -> np.ndarray:
        f = self.filters.get(key)
        if f is None or t - f.t > self.max_gap or f.x.shape != x.shape:
            self.filters[key] = _OneEuro(x, t)
            return x
        return f(x, t, scale, self.min_cutoff, self.beta).copy()

    def prune(self, t: float) -> None:
        """Forget filters not fed for max_gap seconds."""
        self.filters = {k: f for k, f in self.filters.items() if t - f.t <= self.max_gap}


def _points(landmarks) -> np.ndarray:
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float64)


def _write(landmarks, pts: np.ndarray) -> None:
    for lm, (x, y, z) in zip(landmarks, pts.tolist()):
        lm.x, lm.y, lm.z = x, y, z


def _extent(pts: np.ndarray) -> float:
    span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    return float(span.mean())


class PoseSmoother:
    """Steadies each person's body crop and smooths the landmarks of every face, hand and body drawn."""

    def __init__(self) -> None:
        self.smoother = Smoother()
        self._hand_slots = 0
        self._crops: Dict[Hashable, Deque[Tuple[float, BoundingBox]]] = {}

    def crop(self, key: Hashable, b: BoundingBox, t: float) -> BoundingBox:
        """The union of this key's boxes from the last CROP_WINDOW seconds, including b."""
        boxes = self._crops.setdefault(key, deque())
        boxes.append((t, b))
        while t - boxes[0][0] > CROP_WINDOW:
            boxes.popleft()
        x0 = min(bb.x for _, bb in boxes)
        y0 = min(bb.y for _, bb in boxes)
        x1 = max(bb.x + bb.w for _, bb in boxes)
        y1 = max(bb.y + bb.h for _, bb in boxes)
        return BoundingBox(x0, y0, x1 - x0, y1 - y0, b.confidence)

    def smooth(self, faces: List[FaceResult], hands: List[HandResult], bodies: List[BodyResult],
               t: float) -> None:
        """Replaces the landmark positions in place with smoothed ones. Call once per frame."""
        for kind, results in (('face', faces), ('body', bodies)):
            for r in results:
                self._smooth_landmarks((kind, r.person_id), r.landmarks, t)
        used = set()
        for hand in hands:
            key = self._hand_key(hand, t, used)
            used.add(key)
            self._smooth_landmarks(key, hand.landmarks, t)
        self.smoother.prune(t)
        self._crops = {k: v for k, v in self._crops.items() if t - v[-1][0] <= CROP_WINDOW}

    def _smooth_landmarks(self, key: Hashable, landmarks, t: float) -> None:
        if not landmarks:
            return
        pts = _points(landmarks)
        _write(landmarks, self.smoother(key, pts, t, _extent(pts)))

    def _hand_key(self, hand: HandResult, t: float, used: set) -> Hashable:
        """The key of this person's nearest hand from earlier frames (wrist within one hand size), else a new one."""
        pts = _points(hand.landmarks)
        wrist, size = pts[0, :2], max(_extent(pts), 1.0)
        best, best_d = None, size
        for key, f in self.smoother.filters.items():
            if key[0] != 'hand' or key[1] != hand.person_id or key in used or t - f.t > self.smoother.max_gap:
                continue
            d = float(np.hypot(*(f.x[0, :2] - wrist)))
            if d < best_d:
                best, best_d = key, d
        if best is None:
            self._hand_slots += 1
            best = ('hand', hand.person_id, self._hand_slots)
        return best
