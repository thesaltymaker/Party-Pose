"""Counts how often a drawn part of a tracked person blinks out (issue #13).

A blink is a gap of 1 to MAX_GAP frames in which a (track id, part) was not drawn, between two frames in which
it was. Longer gaps are the person leaving or turning away, not flicker. Printed as one [FLICKER] line at the
end of a --video run, so a clip replay gives a before/after figure.
"""
from collections import defaultdict
from typing import Dict, Hashable, Iterable, Tuple

MAX_GAP = 5


class FlickerCounter:
    def __init__(self) -> None:
        self.frame = 0
        self.last: Dict[Tuple[int, str], int] = {}  # (track id, part) -> last frame it was drawn
        self.drawn: Dict[str, int] = defaultdict(int)  # part -> frames drawn
        self.blinks: Dict[str, Dict[int, int]] = {}  # part -> {gap length: count}

    def update(self, drawn: Iterable[Tuple[Hashable, str]]) -> None:
        """drawn: the (track id, part) keys drawn this frame. Call once per frame."""
        self.frame += 1
        for key in drawn:
            part = key[1]
            self.drawn[part] += 1
            gap = self.frame - self.last.get(key, self.frame) - 1
            if 1 <= gap <= MAX_GAP:
                counts = self.blinks.setdefault(part, {})
                counts[gap] = counts.get(gap, 0) + 1
            self.last[key] = self.frame

    def report(self) -> str:
        parts = []
        for part in sorted(self.drawn):
            counts = self.blinks.get(part, {})
            gaps = ' '.join(f'{g}f:{n}' for g, n in sorted(counts.items()))
            parts.append(f'{part}: drawn={self.drawn[part]} blinks={sum(counts.values())}'
                         + (f' ({gaps})' if gaps else ''))
        return '[FLICKER] ' + ' | '.join(parts)
