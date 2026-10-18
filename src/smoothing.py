"""Smoothing of landmarks across frames, per person.

The body crop follows the person's own landmarks (BodyProcessor), not the detector's box, which flips between
a small and a big box. The head box is not smoothed: the face model loses faces when the head is off-centre in
the crop (unpadded crop 37% kept vs 0-5% padded, issue #6/#7).

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
from typing import Dict, Hashable, List

import numpy as np

from src.types import BodyResult, FaceResult, HandResult

MIN_CUTOFF = 0.3  # Hz: cutoff when still (lower = steadier, more lag on slow movement)
BETA = 1.5        # cutoff added per (object size / second) of speed (higher = less lag when fast)
D_CUTOFF = 1.0    # Hz: cutoff for the speed estimate


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

    def __call__(self, key: Hashable, x: np.ndarray, t: float, scale: float, params=None) -> np.ndarray:
        """params: (min_cutoff, beta) for this key instead of the smoother's own."""
        f = self.filters.get(key)
        if f is None or t - f.t > self.max_gap or f.x.shape != x.shape:
            self.filters[key] = _OneEuro(x, t)
            return x
        min_cutoff, beta = params or (self.min_cutoff, self.beta)
        return f(x, t, scale, min_cutoff, beta).copy()

    def prune(self, t: float) -> None:
        """Forget filters not fed for max_gap seconds."""
        self.filters = {k: f for k, f in self.filters.items() if t - f.t <= self.max_gap}


def _points(result) -> np.ndarray:
    """[N, 2] x, y: the processor's array if it set one (building it from Landmark objects is most of the time on
    the Orin), else from the landmarks (held faces)."""
    if result.pts is not None and len(result.pts) == len(result.landmarks):
        return result.pts
    return np.array([(lm.x, lm.y) for lm in result.landmarks], dtype=np.float64)


def _write(landmarks, pts: np.ndarray) -> None:
    for lm, (x, y) in zip(landmarks, pts.tolist()):
        lm.x, lm.y = x, y


def _extent(pts: np.ndarray) -> float:
    span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    return float(span.mean())


class PoseSmoother:
    """Smooths the landmarks of every face, hand and body drawn.

    params: {'face' | 'hand' | 'body': (min_cutoff, beta)} for kinds filtered differently from MIN_CUTOFF/BETA.
    score_alpha > 0: body landmark presence and visibility are averaged over frames too (exponential moving average
    with this weight for the new value, as MediaPipe smooths visibility), so a one-frame dip doesn't hide a joint
    or flip it dim.
    """

    def __init__(self, params=None, score_alpha: float = 0.0) -> None:
        self.smoother = Smoother()
        self._hand_slots = 0
        self.params = params or {}
        self.score_alpha = score_alpha
        self.scores = {}  # body key -> (time, [N, 2] presence and visibility averages)

    def smooth(self, faces: List[FaceResult], hands: List[HandResult], bodies: List[BodyResult],
               t: float) -> None:
        """Replaces the landmark positions in place with smoothed ones. Call once per frame."""
        for kind, results in (('face', faces), ('body', bodies)):
            for r in results:
                self._smooth_landmarks((kind, r.person_id), r, t)
        if self.score_alpha > 0:
            for r in bodies:
                self._smooth_scores(('body', r.person_id), r, t)
        used = set()
        for hand in hands:
            key = self._hand_key(hand, t, used)
            used.add(key)
            self._smooth_landmarks(key, hand, t)
        self.smoother.prune(t)
        self.scores = {k: v for k, v in self.scores.items() if t - v[0] <= self.smoother.max_gap}

    def _smooth_landmarks(self, key: Hashable, result, t: float) -> None:
        if not result.landmarks:
            return
        pts = _points(result)
        _write(result.landmarks, self.smoother(key, pts, t, _extent(pts), self.params.get(key[0])))

    def _smooth_scores(self, key: Hashable, result, t: float) -> None:
        new = np.array([(lm.presence, lm.visibility) for lm in result.landmarks])
        old = self.scores.get(key)
        if old is not None and t - old[0] <= self.smoother.max_gap and old[1].shape == new.shape:
            new = old[1] + self.score_alpha * (new - old[1])
        self.scores[key] = (t, new)
        for lm, (p, v) in zip(result.landmarks, new.tolist()):
            lm.presence, lm.visibility = p, v

    def _hand_key(self, hand: HandResult, t: float, used: set) -> Hashable:
        """The key of this person's nearest hand from earlier frames (wrist within one hand size), else a new one."""
        pts = _points(hand)
        wrist, size = pts[0], max(_extent(pts), 1.0)
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
