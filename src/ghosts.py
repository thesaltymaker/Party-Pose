"""Ghosts: redraw a tracked person's missing body, face or hand, dimming, instead of letting it flicker (issue #13).

Each part is keyed by (track id, part), so a ghost keeps its track's colour and never moves to another person.
When a part is missing, its last (smoothed) result is drawn again at (frames - missed) / frames brightness (80, 60, 40,
20% for 5 frames), then dropped. A new detection for the key replaces the ghost at once. A person who leaves
loses all their parts the same way, so nothing of them stays longer than `frames` frames.
"""
from typing import Any, Dict, Hashable, List, Tuple

Key = Tuple[int, Hashable]


class Ghosts:
    def __init__(self, frames: int = 5) -> None:
        self.frames = frames
        self.last: Dict[Key, Any] = {}  # key -> last result seen
        self.missed: Dict[Key, int] = {}  # key -> frames since it was seen

    def update(self, seen: Dict[Key, Any]) -> List[Tuple[Key, Any, float]]:
        """seen: this frame's results by key. Returns the ghosts to draw: (key, last result, brightness)."""
        ghosts = []
        for key in list(self.last):
            if key in seen:
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
