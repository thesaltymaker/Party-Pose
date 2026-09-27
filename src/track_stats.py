"""Per-track statistics printed as text, for tuning false-positive filters without saving images (issue #6/#7).

`--track-report SECONDS` prints one `[TRACK]` line per track seen in the last interval. With `--show-ids` the
track number is drawn on screen, so the viewer can say which tracks are people and which are false positives.

Movement is measured two ways, both relative to the box height so near and far people compare:
  box_move  median frame-to-frame movement of the box centre
  limb_move median frame-to-frame movement of the pose model's shoulders, elbows, wrists and hips
A static prop should score near 0 on both; a seated person facing away mostly on limb_move.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.types import BodyResult, BoundingBox

# MediaPipe pose indices: shoulders, elbows, wrists, hips.
LIMB_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24)


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
    last_limbs: Optional[List[Tuple[float, float]]] = None
    last_centre: Optional[Tuple[float, float]] = None
    seen_since_report: bool = False


class TrackStats:
    def __init__(self, max_samples: int = 3000) -> None:
        self._stats: Dict[int, _Stats] = {}
        self._max = max_samples

    def update(self, track_id: int, box: BoundingBox, had_head: bool,
               face_logit: Optional[float], body: Optional[BodyResult]) -> None:
        st = self._stats.setdefault(track_id, _Stats())
        st.frames += 1
        st.seen_since_report = True
        h = max(box.h, 1.0)
        centre = (box.x + box.w / 2, box.y + box.h / 2)
        if st.last_centre is not None:
            st.box_moves.append(math.dist(centre, st.last_centre) / h)
        st.last_centre = centre
        st.scores.append(box.confidence)
        st.boxes.append((box.x, box.y, box.w, box.h))
        st.heads += int(had_head)
        if face_logit is not None:
            st.face_logits.append(face_logit)
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
                    f'max={max(st.face_logits):.1f}') if st.face_logits else 'faces=0'
            lines.append(
                f'[TRACK] #{tid} frames={st.frames} box=({x:.0f},{y:.0f} {w:.0f}x{h:.0f}) '
                f'score={statistics.median(st.scores):.2f} '
                f'box_move={_med(st.box_moves):.4f} limb_move={_med(st.limb_moves):.4f} '
                f'pose={_med(st.body_presence):.2f} head={st.heads / st.frames:.0%} {face}')
            st.seen_since_report = False
        for tid in [t for t in self._stats if t not in alive_ids]:
            del self._stats[tid]
        return lines


def _med(v: List[float]) -> float:
    return statistics.median(v) if v else float('nan')
