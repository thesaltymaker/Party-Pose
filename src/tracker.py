"""Person tracking: stable IDs for body boxes across frames (issue #6).

The detector orders body boxes by score, which changes every frame, so using the box index as the
person ID made colours swap. `PersonTracker.update()` matches each frame's body boxes to the tracks
from earlier frames and returns a track ID per box. IDs are never reused, so a colour
(`PERSON_COLORS[id % 8]`) stays with a person while they stay in frame.

Matching is greedy on a score of box overlap (IoU) against each track's predicted box (last box moved
by its smoothed velocity, so two people crossing keep their IDs), with a centre-distance fallback for
fast movement where boxes no longer overlap. A track that gets no match stays alive for `max_missed`
frames, so a one-frame detector miss doesn't reset it. A track is `confirmed` once it has been matched in
`min_hits` frames; the app only draws confirmed tracks, so a false positive that flickers for a frame or
two never gets a skeleton (and never uses up a colour).

A confirmed track that expires is kept as "lost" for `max_lost` frames. A new box in the same spot gets the
lost track back, with its ID, colour and movement history, instead of a new ID. A far person with a weak detector
score otherwise got a new ID every few seconds. "Same spot" is measured as the share of the smaller box that
lies inside the other (>= `revive_overlap`), not IoU: that person's box also flips between a small and a big
box (IoU 0.25).

Still tracks (`--drop-still`, issue #7): props (a lamp, a bar stool, a chair) are detected as people, and no
per-frame score separates them from people (issue #3), and they move when the unit is moved. Over ~10 s a
prop's box stays put while a moving person's drifts. `Track.is_still()` is true once the track has
STILL_FRAMES matched frames and the 90th-percentile distance of its box centre from the median, relative to
box height, is below STILL_SPREAD. Orin (Just Dance, 3 min): dancers 0.08-0.93, a person standing still
0.055, a prop 0.037. A person sitting very still (0.025 at a desk) is dropped too; accepted for party use.
A dropped track is shown again after about 1 s of movement (10% of the window moved).

Each track keeps a short history of box centres for venue calibration (issue #7).
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from src.types import BoundingBox

STILL_FRAMES = 300     # ~10 s at 30 FPS
STILL_SPREAD = 0.045   # box centre spread (fraction of box height) below which a track counts as still


def centre_spread(samples) -> float:
    """90th-percentile distance of the (cx, cy) centres from their median, relative to the median height."""
    xs, ys, hs = zip(*samples)
    mx, my = statistics.median(xs), statistics.median(ys)
    mh = max(statistics.median(hs), 1.0)
    d = sorted(math.dist((x, y), (mx, my)) / mh for x, y in zip(xs, ys))
    return d[int(0.9 * (len(d) - 1))]


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def overlap_of_smaller(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection area as a share of the smaller box's area (1.0 when one box lies inside the other)."""
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    smaller = min(a.w * a.h, b.w * b.h)
    return ix * iy / smaller if smaller > 0 else 0.0


def _centre(b: BoundingBox) -> Tuple[float, float]:
    return b.x + b.w / 2, b.y + b.h / 2


@dataclass
class Track:
    id: int
    box: BoundingBox
    vx: float = 0.0          # smoothed centre velocity, pixels per frame
    vy: float = 0.0
    missed: int = 0          # frames since the last match
    age: int = 1             # frames since the track started
    hits: int = 1            # frames in which the track was matched
    confirmed: bool = False  # matched in at least min_hits frames; stays True until the track expires
    face_hits: int = 0       # frames in which the app's face model clearly found a face in this track's head box
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))
    recent: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=STILL_FRAMES))

    def is_still(self) -> bool:
        """True once the box hasn't moved beyond detector jitter over the last STILL_FRAMES matched frames."""
        return len(self.recent) == STILL_FRAMES and centre_spread(self.recent) < STILL_SPREAD

    def predicted(self) -> BoundingBox:
        """The last box moved by the velocity over the frames since it was seen."""
        steps = self.missed + 1
        return BoundingBox(self.box.x + self.vx * steps, self.box.y + self.vy * steps,
                           self.box.w, self.box.h, self.box.confidence)


class PersonTracker:
    """Assigns stable IDs to body boxes. Call `update()` once per frame, even with no boxes."""

    def __init__(self, max_missed: int = 15, min_iou: float = 0.2, max_centre_dist: float = 0.6,
                 velocity_smoothing: float = 0.5, min_hits: int = 3, max_lost: int = 150,
                 revive_overlap: float = 0.5) -> None:
        """
        min_hits: frames a track must be matched in before it is confirmed (drawn).
        max_missed: frames a track survives without a match (15 = 0.5 s at 30 FPS).
        max_lost: frames since last seen that an expired confirmed track can still be revived (150 = 5 s).
        revive_overlap: minimum share of the smaller of (new box, lost track's last box) inside the other,
            to revive the lost track.
        min_iou: minimum overlap between a box and a track's predicted box to match.
        max_centre_dist: fallback match when boxes don't overlap enough: centre distance, as a fraction
            of the track box's diagonal.
        velocity_smoothing: weight of the newest movement in the velocity estimate (0..1).
        """
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.max_centre_dist = max_centre_dist
        self.velocity_smoothing = velocity_smoothing
        self.min_hits = min_hits
        self.max_lost = max_lost
        self.revive_overlap = revive_overlap
        self.tracks: List[Track] = []
        self.lost: List[Track] = []  # expired confirmed tracks that a box in the same spot can revive
        self._next_id = 0

    def _match_score(self, track: Track, box: BoundingBox) -> Optional[float]:
        """Higher is better; None if the box can't belong to the track.

        IoU matches score in (1, 2]; centre-distance matches in (0, 1], so any overlap match wins over
        a distance-only one. If the box doesn't match the predicted box, it is compared with the last box:
        a still person whose box flips between a small and a big box makes the centre jump, which the
        velocity takes as movement, so the prediction alone missed them (Orin, issue #6).
        """
        score = self._score_against(track.predicted(), box)
        return score if score is not None else self._score_against(track.box, box)

    def _score_against(self, ref: BoundingBox, box: BoundingBox) -> Optional[float]:
        overlap = iou(ref, box)
        if overlap >= self.min_iou:
            return 1.0 + overlap
        (px, py), (bx, by) = _centre(ref), _centre(box)
        diag = (ref.w ** 2 + ref.h ** 2) ** 0.5
        dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5 / diag if diag > 0 else float('inf')
        if dist <= self.max_centre_dist:
            return 1.0 - dist / self.max_centre_dist * 0.999
        return None

    def update(self, boxes: List[BoundingBox]) -> List[int]:
        """Match this frame's body boxes to tracks. Returns the track ID for each box, in order."""
        return [t.id for t in self.update_tracks(boxes)]

    def update_tracks(self, boxes: List[BoundingBox]) -> List[Track]:
        """Like `update()`, but returns the Track for each box (for `confirmed` and `age`)."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for bi, box in enumerate(boxes):
                score = self._match_score(track, box)
                if score is not None:
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)

        out: List[Optional[Track]] = [None] * len(boxes)
        matched_tracks = set()
        for _, ti, bi in pairs:
            if ti in matched_tracks or out[bi] is not None:
                continue
            matched_tracks.add(ti)
            track = self.tracks[ti]
            self._advance(track, boxes[bi])
            out[bi] = track

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.missed += 1
                track.age += 1

        for track in self.lost:
            track.missed += 1
            track.age += 1
        self.lost = [t for t in self.lost if t.missed <= self.max_lost]
        self.lost += [t for t in self.tracks if t.missed > self.max_missed and t.confirmed]
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        revive = sorted(((overlap_of_smaller(t.box, box), li, bi) for li, t in enumerate(self.lost)
                         for bi, box in enumerate(boxes) if out[bi] is None), reverse=True)
        revived = set()
        for overlap, li, bi in revive:
            if overlap < self.revive_overlap:
                break
            if li in revived or out[bi] is not None:
                continue
            revived.add(li)
            track = self.lost[li]
            track.vx = track.vy = 0.0
            self._advance(track, boxes[bi])
            track.vx = track.vy = 0.0
            self.tracks.append(track)
            out[bi] = track
        self.lost = [t for li, t in enumerate(self.lost) if li not in revived]

        for bi, box in enumerate(boxes):
            if out[bi] is None:
                track = Track(id=self._next_id, box=box, confirmed=self.min_hits <= 1)
                track.history.append(_centre(box))
                track.recent.append((*_centre(box), box.h))
                self._next_id += 1
                self.tracks.append(track)
                out[bi] = track

        return out  # type: ignore[return-value]

    def _advance(self, track: Track, box: BoundingBox) -> None:
        (ox, oy), (nx, ny) = _centre(track.box), _centre(box)
        steps = track.missed + 1
        a = self.velocity_smoothing
        track.vx = (1 - a) * track.vx + a * (nx - ox) / steps
        track.vy = (1 - a) * track.vy + a * (ny - oy) / steps
        track.box = box
        track.missed = 0
        track.age += 1
        track.hits += 1
        if track.hits >= self.min_hits:
            track.confirmed = True
        track.history.append((nx, ny))
        track.recent.append((nx, ny, box.h))
