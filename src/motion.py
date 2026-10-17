"""Measures what the owner sees inside a drawn part: joints popping in and out, shaking, and trailing (issue #13).

[FLICKER] counts whole parts blinking out. This counts, for each part drawn in two frames in a row:
- pops: body, hand or face points that are drawn in one frame and not the next (body points with presence < 0.5
  are hidden unless held, as in the renderer; any point off the frame), and body connections doing the same;
- dim flips: body points switching between dim (visibility < 0.5) and full colour;
- jitter: per frame, the mean movement of the part's drawn points after removing the part's own movement (the
  median point movement), in thousandths of the person's body height, over frames where the person moves slowly;
  the median and 90th percentile over those frames, plus the same for the raw model points (what smoothing removed);
- lag: for points moving fast, how far the drawn point is behind the model's point, in frames of that movement.
Faces are measured on their outline points (what --skelly fills), which keeps this cheap enough to leave on in
replays. Ghosts (fade < 1) count for pops but not for jitter or lag (their points are old). Printed as one [MOTION] line at
the end of a --video run, next to [FLICKER].
"""
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from src.renderer import _BODY_SKIP_LM
from src.topology import BODY_CONNECTIONS, FACE_CONTOUR_INDICES

SLOW = 0.01  # person movement per frame, in body heights, below which a frame counts for jitter (~0.3/s at 30 fps)
FAST = 0.03  # point movement per frame, in body heights, above which a point counts for lag (~1/s at 30 fps)
_BODY_POINTS = [i for i in range(33) if i not in _BODY_SKIP_LM]
_BODY_CONNS = np.array([c for c in BODY_CONNECTIONS if c[0] not in _BODY_SKIP_LM and c[1] not in _BODY_SKIP_LM])


def _kind(part: str) -> str:
    return 'hand' if part.startswith('hand') else part


def _state(kind: str, result, pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """Per point: 0 not drawn, 1 drawn dim, 2 drawn."""
    inside = ((pts[:, 0] >= -0.05 * w) & (pts[:, 0] <= 1.05 * w) & (pts[:, 1] >= -0.05 * h) & (pts[:, 1] <= 1.05 * h))
    state = inside.astype(np.int8) * 2
    if kind == 'body':
        lms = result.landmarks
        for i in range(len(state)):
            if i in _BODY_SKIP_LM or (lms[i].presence < 0.5 and lms[i].fade >= 1.0):
                state[i] = 0
            elif state[i] and lms[i].visibility < 0.5 and lms[i].fade >= 1.0:
                state[i] = 1
    return state


class MotionStats:
    def __init__(self) -> None:
        self.prev: Dict = {}  # key -> (drawn pts, raw pts or None, state, fade, held mask)
        self.pairs: Dict[str, int] = defaultdict(int)
        self.pops: Dict[str, int] = defaultdict(int)
        self.conn_pops: Dict[str, int] = defaultdict(int)
        self.dim_flips: Dict[str, int] = defaultdict(int)
        self._jitter: Dict[str, List[float]] = defaultdict(list)
        self._raw_jitter: Dict[str, List[float]] = defaultdict(list)
        self._lag: Dict[str, List[float]] = defaultdict(list)

    def update(self, drawn: Dict, w: int, h: int) -> None:
        """drawn: (track id, part) -> the result drawn this frame (ghosts included). Call once per frame."""
        cur = {}
        scale = {}  # track id -> body height this frame
        for key, result in drawn.items():
            kind = _kind(key[1])
            lms = result.landmarks
            pick = FACE_CONTOUR_INDICES if kind == 'face' and len(lms) >= 468 else range(len(lms))
            pts = np.array([(lms[i].x, lms[i].y) for i in pick], dtype=np.float64)
            raw = result.pts[pick] if result.pts is not None and len(result.pts) == len(lms) else None
            held = np.array([lms[i].fade < 1.0 for i in pick]) if kind == 'body' else np.zeros(len(pts), bool)
            cur[key] = (pts, raw, _state(kind, result, pts, w, h), result.fade, held)
            if kind == 'body' and len(pts) >= 33:
                body = (raw if raw is not None else pts)[11:33]
                scale[key[0]] = max(float(np.ptp(body[:, 1])), float(np.ptp(body[:, 0])), 1.0)
        for key, (pts, raw, state, fade, held) in cur.items():
            old = self.prev.get(key)
            if old is None or len(old[0]) != len(pts):
                continue
            kind = _kind(key[1])
            o_pts, o_raw, o_state, o_fade, o_held = old
            self.pairs[kind] += 1
            self.pops[kind] += int(np.sum((state > 0) != (o_state > 0)))
            self.dim_flips[kind] += int(np.sum((state > 0) & (o_state > 0) & (state != o_state)))
            if kind == 'body':
                on, o_on = state > 0, o_state > 0
                a, b = _BODY_CONNS.T
                self.conn_pops[kind] += int(np.sum((on[a] & on[b]) != (o_on[a] & o_on[b])))
            s = scale.get(key[0])
            if s is None or fade < 1.0 or o_fade < 1.0 or raw is None or o_raw is None:
                continue
            both = (state > 0) & (o_state > 0) & ~held & ~o_held
            if both.sum() < 3:
                continue
            d_raw = raw[both] - o_raw[both]
            if np.hypot(*np.median(d_raw, axis=0)) / s < SLOW:
                for out, a, b in ((self._jitter, pts, o_pts), (self._raw_jitter, raw, o_raw)):
                    d = a[both] - b[both]
                    out[kind].append(1000 * float(np.hypot(*(d - np.median(d, axis=0)).T).mean()) / s)
            speed = np.hypot(*d_raw.T)
            fast = speed / s > FAST
            if fast.any():
                behind = np.hypot(*(pts[both] - raw[both]).T)
                self._lag[kind].extend((behind[fast] / speed[fast]).tolist())
        self.prev = cur

    @staticmethod
    def _pct(values: List[float], q: float) -> Optional[float]:
        return float(np.percentile(values, q)) if values else None

    def jitter(self, kind: str, q: float = 50) -> Optional[float]:
        return self._pct(self._jitter.get(kind, []), q)

    def raw_jitter(self, kind: str, q: float = 50) -> Optional[float]:
        return self._pct(self._raw_jitter.get(kind, []), q)

    def lag(self, kind: str) -> Optional[float]:
        return self._pct(self._lag.get(kind, []), 50)

    def report(self) -> str:
        f = lambda v: '-' if v is None else f'{v:.1f}'
        parts = []
        for kind in sorted(self.pairs):
            n = max(self.pairs[kind], 1)
            parts.append(f'{kind}: pairs={self.pairs[kind]} pops/100={100 * self.pops[kind] / n:.1f} '
                         f'dim/100={100 * self.dim_flips[kind] / n:.1f} conn/100={100 * self.conn_pops[kind] / n:.1f} '
                         f'jitter={f(self.jitter(kind))}/{f(self.jitter(kind, 90))} '
                         f'raw={f(self.raw_jitter(kind))}/{f(self.raw_jitter(kind, 90))} lag={f(self.lag(kind))}f')
        return '[MOTION] ' + ' | '.join(parts)
