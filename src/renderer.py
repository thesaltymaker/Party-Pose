from __future__ import annotations
import cv2
import numpy as np
from typing import List
from src.types import BoundingBox, Landmark, FaceResult, HandResult, BodyResult
from src.topology import FACE_CONNECTIONS, HAND_CONNECTIONS, BODY_CONNECTIONS

PERSON_COLORS = [
    (0,   255,   0),   # Green
    (0,   165, 255),   # Orange
    (255, 255,   0),   # Cyan
    (255,   0, 255),   # Magenta
    (0,   255, 255),   # Yellow
    (180, 105, 255),   # Pink
    (0,   255, 128),   # Lime
    (255,   0, 128),   # Purple
]
COLOR_ROI  = (200, 200, 200)  # Grey for all ROI boxes
COLOR_FPS  = (0,   255,   0)  # Green

# MediaPipe Pose landmarks 0-10 are face points (nose, eyes, ears, mouth).
# Suppressed in draw_body so the dedicated face mesh is used instead.
_BODY_FACE_LM = frozenset(range(11))

class Renderer:
    """Renderer: draws pose detection results onto a CPU numpy frame (single download per loop)."""

    def __init__(self, show_roi: bool = False):
        self.show_roi = show_roi

    def draw_faces(self, cpu_frame: np.ndarray, results: List[FaceResult],
                   frame_w: int, frame_h: int, cs_x: float = 1.0, cs_y: float = 1.0) -> None:
        for result in results:
            color = PERSON_COLORS[result.person_id % len(PERSON_COLORS)]
            for lm in result.landmarks:
                x, y = int(lm.x * cs_x), int(lm.y * cs_y)
                if not (-0.05 * frame_w <= x <= 1.05 * frame_w and -0.05 * frame_h <= y <= 1.05 * frame_h):
                    continue
                cv2.circle(cpu_frame, (x, y), 2, color, -1)
            for conn in FACE_CONNECTIONS:
                start = result.landmarks[conn[0]]
                end   = result.landmarks[conn[1]]
                px, py = int(start.x * cs_x), int(start.y * cs_y)
                ex, ey = int(end.x   * cs_x), int(end.y   * cs_y)
                if (-0.05 * frame_w <= px <= 1.05 * frame_w and -0.05 * frame_h <= py <= 1.05 * frame_h and
                        -0.05 * frame_w <= ex <= 1.05 * frame_w and -0.05 * frame_h <= ey <= 1.05 * frame_h):
                    cv2.line(cpu_frame, (px, py), (ex, ey), color, 1)
            if self.show_roi:
                bbox = result.bbox
                cv2.rectangle(cpu_frame,
                              (int(bbox.x * cs_x), int(bbox.y * cs_y)),
                              (int((bbox.x + bbox.w) * cs_x), int((bbox.y + bbox.h) * cs_y)),
                              COLOR_ROI, 2)

    def draw_hands(self, cpu_frame: np.ndarray, results: List[HandResult],
                   frame_w: int, frame_h: int, cs_x: float = 1.0, cs_y: float = 1.0) -> None:
        for result in results:
            color = PERSON_COLORS[result.person_id % len(PERSON_COLORS)]
            for lm in result.landmarks:
                x, y = int(lm.x * cs_x), int(lm.y * cs_y)
                if not (-0.05 * frame_w <= x <= 1.05 * frame_w and -0.05 * frame_h <= y <= 1.05 * frame_h):
                    continue
                cv2.circle(cpu_frame, (x, y), 3, color, -1)
            for conn in HAND_CONNECTIONS:
                start = result.landmarks[conn[0]]
                end   = result.landmarks[conn[1]]
                px, py = int(start.x * cs_x), int(start.y * cs_y)
                ex, ey = int(end.x   * cs_x), int(end.y   * cs_y)
                if (-0.05 * frame_w <= px <= 1.05 * frame_w and -0.05 * frame_h <= py <= 1.05 * frame_h and
                        -0.05 * frame_w <= ex <= 1.05 * frame_w and -0.05 * frame_h <= ey <= 1.05 * frame_h):
                    cv2.line(cpu_frame, (px, py), (ex, ey), color, 2)
            if self.show_roi:
                bbox = result.bbox
                cv2.rectangle(cpu_frame,
                              (int(bbox.x * cs_x), int(bbox.y * cs_y)),
                              (int((bbox.x + bbox.w) * cs_x), int((bbox.y + bbox.h) * cs_y)),
                              COLOR_ROI, 2)

    def draw_body(self, cpu_frame: np.ndarray, results: List[BodyResult],
                  frame_w: int, frame_h: int, cs_x: float = 1.0, cs_y: float = 1.0) -> None:
        for result in results:
            color = PERSON_COLORS[result.person_id % len(PERSON_COLORS)]
            dim_color = tuple(int(c * 0.3) for c in color)
            for i, lm in enumerate(result.landmarks[:33]):
                if i in _BODY_FACE_LM:
                    continue
                x, y = int(lm.x * cs_x), int(lm.y * cs_y)
                if not (-0.05 * frame_w <= x <= 1.05 * frame_w and -0.05 * frame_h <= y <= 1.05 * frame_h):
                    continue
                if lm.presence < 0.5:
                    continue
                pt_color = dim_color if lm.visibility < 0.5 else color
                cv2.circle(cpu_frame, (x, y), 4, pt_color, -1)
            for conn in BODY_CONNECTIONS:
                if conn[0] in _BODY_FACE_LM or conn[1] in _BODY_FACE_LM:
                    continue
                start = result.landmarks[conn[0]]
                end   = result.landmarks[conn[1]]
                if start.presence < 0.5 or end.presence < 0.5:
                    continue
                px, py = int(start.x * cs_x), int(start.y * cs_y)
                ex, ey = int(end.x   * cs_x), int(end.y   * cs_y)
                if (-0.05 * frame_w <= px <= 1.05 * frame_w and -0.05 * frame_h <= py <= 1.05 * frame_h and
                        -0.05 * frame_w <= ex <= 1.05 * frame_w and -0.05 * frame_h <= ey <= 1.05 * frame_h):
                    cv2.line(cpu_frame, (px, py), (ex, ey), color, 2)

    def draw_fps(self, cpu_frame: np.ndarray, fps: float) -> None:
        cv2.putText(cpu_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FPS, 2)
