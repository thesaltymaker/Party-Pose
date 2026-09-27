"""Venue ignore zones (issue #7): fixed spots where the detector keeps finding a "person" that is a prop.

On the Orin, a table lamp, a bar stool and a chair were detected as people every few frames. Their scores
(body 0.31-0.64, head 0.32-0.47) overlap real people's, but the boxes come back in the same place and size.
A venue file lists those boxes; a track whose box matches one (IoU >= min_iou) is not drawn.

File format (JSON, camera pixels, x/y = top-left):
    {"ignore": [{"name": "lamp", "box": [669, 423, 128, 241]}, ...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.tracker import iou
from src.types import BoundingBox


@dataclass(frozen=True)
class IgnoreZone:
    name: str
    box: BoundingBox


def load_venue(path: str) -> List[IgnoreZone]:
    data = json.loads(Path(path).read_text())
    return [IgnoreZone(z.get('name', f'zone{i}'), BoundingBox(*map(float, z['box']), 1.0))
            for i, z in enumerate(data.get('ignore', []))]


def ignored_by(box: BoundingBox, zones: List[IgnoreZone], min_iou: float = 0.5):
    """The zone this box matches, or None."""
    for zone in zones:
        if iou(box, zone.box) >= min_iou:
            return zone
    return None
