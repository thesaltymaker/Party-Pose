"""Per-track statistics printed as text, for tuning false-positive filters without saving images (issue #6/#7).

`--track-report SECONDS` prints one `[TRACK]` line per track seen in the last interval. With `--show-ids` the
track number is drawn on screen, so the viewer can say which tracks are people and which are false positives.

Movement is measured two ways, both relative to the box height so near and far people compare:
  box_move  median frame-to-frame movement of the box centre
  limb_move median frame-to-frame movement of the pose model's shoulders, elbows, wrists and hips
A static prop should score near 0 on both; a seated person facing away mostly on limb_move.

Frame-to-frame movement turned out to measure detector and pose-model jitter, not the person. Over a
longer time a prop's box stays put while a person's drifts, so two more figures cover the last ~10 s:
  spread10s  90th percentile distance of the box centre from its median, relative to box height
  size10s    90th percentile change of the box height from its median, relative to box height
and strong_faces counts frames with a face logit >= STRONG_FACE_LOGIT.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from src.tracker import centre_spread
from src.types import BodyResult, BoundingBox

# MediaPipe pose indices: shoulders, elbows, wrists, hips.
LIMB_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24)
RECENT_FRAMES = 300       # ~10 s at 30 FPS
STRONG_FACE_LOGIT = 8.0   # people reached 13-17 on the Orin; lamp and chair at most 6.1


@dataclass
class _Stats:
    frames: int = 0
    scores: List[float] = field(default_factory=list)
    boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    box_moves: List[float] = field(default_factory=list)
    limb_moves: List[float] = field(default_factory=list)
    body_presence: List[float] = field(default_factory=list)
    face_logits: List[float] = field(default_factory=list)
    heads: int = 0
    strong_faces: int = 0
    recent: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=RECENT_FRAMES))
    still: Optional[bool] = None  # tracker's still decision (--drop-still)
    last_limbs: Optional[List[Tuple[float, float]]] = None
    last_centre: Optional[Tuple[float, float]] = None
    seen_since_report: bool = False


class TrackStats:
    def __init__(self, max_samples: int = 3000) -> None:
        self._stats: Dict[int, _Stats] = {}
        self._max = max_samples

    def update(self, track_id: int, box: BoundingBox, had_head: bool,
               face_logit: Optional[float], body: Optional[BodyResult],
               still: Optional[bool] = None) -> None:
        st = self._stats.setdefault(track_id, _Stats())
        st.still = still
        st.frames += 1
        st.seen_since_report = True
        h = max(box.h, 1.0)
        centre = (box.x + box.w / 2, box.y + box.h / 2)
        if st.last_centre is not None:
            st.box_moves.append(math.dist(centre, st.last_centre) / h)
        st.last_centre = centre
        st.recent.append((centre[0], centre[1], box.h))
        st.scores.append(box.confidence)
        st.boxes.append((box.x, box.y, box.w, box.h))
        st.heads += int(had_head)
        if face_logit is not None:
            st.face_logits.append(face_logit)
            st.strong_faces += int(face_logit >= STRONG_FACE_LOGIT)
        if body is not None:
            st.body_presence.append(body.presence)
            limbs = [(body.landmarks[i].x, body.landmarks[i].y) for i in LIMB_LANDMARKS
                     if i < len(body.landmarks)]
            if st.last_limbs is not None and len(limbs) == len(st.last_limbs) and limbs:
                st.limb_moves.append(statistics.mean(math.dist(a, b) for a, b in zip(limbs, st.last_limbs)) / h)
            st.last_limbs = limbs
        else:
            st.last_limbs = None
        for lst in (st.scores, st.boxes, st.box_moves, st.limb_moves, st.body_presence, st.face_logits):
            if len(lst) > self._max:
                del lst[: len(lst) - self._max]

    def report(self, alive_ids, min_frames: int = 15) -> List[str]:
        """One line per track seen since the last report (with at least min_frames); forgets expired tracks."""
        lines = []
        for tid in sorted(self._stats):
            st = self._stats[tid]
            if not st.seen_since_report or st.frames < min_frames:
                continue
            x, y, w, h = (statistics.median(v) for v in zip(*st.boxes))
            face = (f'faces={len(st.face_logits)} logit_med={statistics.median(st.face_logits):.1f} '
                    f'max={max(st.face_logits):.1f} '
                    f'kept={sum(v >= 0 for v in st.face_logits) / len(st.face_logits):.0%} '
                    f'logit_p25={_pct(st.face_logits, 0.25):.1f} p75={_pct(st.face_logits, 0.75):.1f}'
                    ) if st.face_logits else 'faces=0'
            lines.append(
                f'[TRACK] #{tid} frames={st.frames} box=({x:.0f},{y:.0f} {w:.0f}x{h:.0f}) '
                f'score={statistics.median(st.scores):.2f} '
                f'box_move={_med(st.box_moves):.4f} limb_move={_med(st.limb_moves):.4f} '
                f'pose={_med(st.body_presence):.2f} head={st.heads / st.frames:.0%} {face} '
                f'strong_faces={st.strong_faces} {_spread(st.recent)}'
                + (f' still={"yes" if st.still else "no"}' if st.still is not None else ''))
            st.seen_since_report = False
        for tid in [t for t in self._stats if t not in alive_ids]:
            del self._stats[tid]
        return lines


def _med(v: List[float]) -> float:
    return statistics.median(v) if v else float('nan')


def _spread(recent) -> str:
    if len(recent) < 30:
        return 'spread10s=nan size10s=nan'
    hs = [h for _, _, h in recent]
    mh = max(statistics.median(hs), 1.0)
    s = sorted(abs(h - mh) / mh for h in hs)
    return f'spread10s={centre_spread(recent):.3f} size10s={s[int(0.9 * (len(s) - 1))]:.3f}'


def _pct(v: List[float], q: float) -> float:
    s = sorted(v)
    return s[int(q * (len(s) - 1))]
