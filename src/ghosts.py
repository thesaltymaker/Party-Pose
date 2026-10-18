"""Ghosts: redraw a tracked person's missing body, face or hand, dimming, instead of letting it flicker (issue #13).

Each part is keyed by (track id, part), so a ghost keeps its track's colour and never moves to another person.
When a part is missing, its last (smoothed) result is drawn again at (frames - missed) / frames brightness (80, 60, 40,
20% for 5 frames), then dropped. A new detection for the key replaces the ghost at once. A person who leaves
loses all their parts the same way, so nothing of them stays longer than `frames` frames.
A part whose model was skipped on purpose this frame (--hand-every, --face-every) is redrawn at full brightness and
does not age.
"""
from typing import Any, Dict, Hashable, List, Tuple

Key = Tuple[int, Hashable]


def skip_turn(every: int, frame_no: int, slot: int, have_last: bool) -> bool:
    """True when a person's hands or face skip the model this frame: every `every`-th frame is run, people (slot =
    their place in the frame's list) taking turns so each frame has the same number of runs; a part not found last
    frame is always run."""
    return every > 1 and have_last and (frame_no + slot) % every != 0


class Ghosts:
    def __init__(self, frames: int = 5) -> None:
        self.frames = frames
        self.last: Dict[Key, Any] = {}  # key -> last result seen
        self.missed: Dict[Key, int] = {}  # key -> frames since it was seen

    def fresh(self, key: Key) -> bool:
        """True when the part was drawn from a model result last frame (or redrawn on a skipped frame)."""
        return key in self.last and self.missed[key] == 0

    def update(self, seen: Dict[Key, Any], skipped=()) -> List[Tuple[Key, Any, float]]:
        """seen: this frame's results by key; skipped: keys whose model was not run on purpose. Returns the ghosts
        to draw: (key, last result, brightness)."""
        ghosts = []
        for key in list(self.last):
            if key in seen:
                continue
            if key in skipped and self.frames:
                ghosts.append((key, self.last[key], 1.0))
                continue
            self.missed[key] += 1
            if self.missed[key] >= self.frames:
                del self.last[key], self.missed[key]
            else:
                ghosts.append((key, self.last[key], (self.frames - self.missed[key]) / self.frames))
        for key, result in seen.items():
            self.last[key] = result
            self.missed[key] = 0
        return ghosts
