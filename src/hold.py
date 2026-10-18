"""Hold a lost body joint at its last place for a few frames, dimming, instead of hiding it at once (--hold-joints).

The renderer hides body joints with presence < 0.5, so an elbow or knee whose presence dips for a frame or two pops
out and back in. With a hold, a joint that was drawn in the last `frames` - 1 frames and is now lost is drawn at its
last (smoothed) place, moved with the body (the middle of the shoulders and hips), at (frames - missed) / frames
brightness, as part ghosts are (src/ghosts.py). The joint's own landmark gets that brightness as its `fade`, and its
last visibility, so a joint drawn dim stays dim.
Call after smoothing, on this frame's bodies only (not on ghosts).
"""
from typing import Dict, List

import numpy as np

from src.types import BodyResult

_CENTRE = (11, 12, 23, 24)  # shoulders and hips


def _centre(lms) -> np.ndarray:
    return np.array([(lms[i].x, lms[i].y) for i in _CENTRE]).mean(axis=0)


class JointHold:
    def __init__(self, frames: int = 0) -> None:
        self.frames = frames
        # track id -> ([33, 2] joint places relative to the centre, [33] frames missed, [33] last visibility)
        self.last: Dict[int, tuple] = {}

    def apply(self, bodies: List[BodyResult]) -> None:
        if self.frames <= 0:
            return
        last = {}
        for body in bodies:
            lms = body.landmarks
            if len(lms) < 33:
                continue
            centre = _centre(lms)
            places, missed, vis = self.last.get(body.person_id,
                                                (np.zeros((33, 2)), np.full(33, self.frames), np.ones(33)))
            places, missed, vis = places.copy(), missed.copy(), vis.copy()
            for i in range(33):
                lm = lms[i]
                if lm.presence >= 0.5:
                    places[i] = (lm.x - centre[0], lm.y - centre[1])
                    missed[i] = 0
                    vis[i] = lm.visibility
                    continue
                missed[i] += 1
                if missed[i] < self.frames:
                    lm.x, lm.y = float(places[i, 0] + centre[0]), float(places[i, 1] + centre[1])
                    lm.fade = (self.frames - missed[i]) / self.frames
                    lm.visibility = float(vis[i])  # dim joints stay dim while held
            last[body.person_id] = (places, missed, vis)
        self.last = last
