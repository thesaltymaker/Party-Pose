from __future__ import annotations
import cv2
import numpy as np
from typing import List
from src.types import BoundingBox, Landmark, FaceResult, HandResult, BodyResult
from src.topology import (FACE_CONNECTIONS, HAND_CONNECTIONS, BODY_CONNECTIONS, FACE_CONTOUR_INDICES,
                          SKELLY_CUTOUTS, SANTA_HAT_LEFT, SANTA_HAT_RIGHT)

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
COLOR_SKELLY = (200, 230, 245)  # Bone white (Party-Pose.py FACE_MESH_COLOR)
SANTA_HAT_RED = (0, 0, 200)
SANTA_HAT_WHITE = (255, 255, 255)

# MediaPipe Pose landmarks 0-10 are face points (nose, eyes, ears, mouth).
# Suppressed in draw_body so the dedicated face mesh is used instead.
_BODY_FACE_LM = frozenset(range(11))

class Renderer:
    """Renderer: draws pose detection results onto a CPU numpy frame (single download per loop)."""

    def __init__(self, show_roi: bool = False, skelly: bool = False, santa_hat: bool = False):
        self.show_roi = show_roi
        self.skelly = skelly
        self.santa_hat = santa_hat

    def draw_faces(self, cpu_frame: np.ndarray, results: List[FaceResult],
                   frame_w: int, frame_h: int, cs_x: float = 1.0, cs_y: float = 1.0) -> None:
        for result in results:
            if self.skelly:
                _fill_skelly_face(cpu_frame, result.landmarks, cs_x, cs_y)
            if self.santa_hat:
                _draw_santa_hat(cpu_frame, result.landmarks, cs_x, cs_y, result.person_id)
            if self.skelly:
                continue
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

    def draw_ids(self, cpu_frame: np.ndarray, labels, cs_x: float = 1.0, cs_y: float = 1.0) -> None:
        """Draw '#<track id>' at the top-left of each person's body box (--show-ids).

        Text height scales with the frame (about 1/14 of it) so the numbers can be read from across a room.
        """
        scale = max(1.0, cpu_frame.shape[0] / 400)
        thick = max(2, int(scale * 2))
        for track_id, box in labels:
            color = PERSON_COLORS[track_id % len(PERSON_COLORS)]
            text_h = int(30 * scale)
            org = (int(box.x * cs_x) + 4, max(text_h, int(box.y * cs_y) + text_h))
            cv2.putText(cpu_frame, f'#{track_id}', org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick * 3)
            cv2.putText(cpu_frame, f'#{track_id}', org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def draw_fps(self, cpu_frame: np.ndarray, fps: float) -> None:
        cv2.putText(cpu_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FPS, 2)


def _points(landmarks, indices, cs_x: float, cs_y: float) -> np.ndarray:
    return np.array([(landmarks[i].x * cs_x, landmarks[i].y * cs_y) for i in indices], dtype=np.int32)


def _fill_skelly_face(cpu_frame: np.ndarray, landmarks, cs_x: float, cs_y: float) -> None:
    """--skelly: fill the face outline in bone white, leaving eyes, nose and mouth showing (Party-Pose.py SKELLY_MODE).

    The mask covers only the face's bounding box, not the whole frame, so it stays cheap at full-screen size.
    """
    if len(landmarks) < 468:
        return
    outline = _points(landmarks, FACE_CONTOUR_INDICES, cs_x, cs_y)
    h, w = cpu_frame.shape[:2]
    x0, y0 = np.maximum(outline.min(axis=0), 0)
    x1, y1 = np.minimum(outline.max(axis=0) + 1, (w, h))
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    offset = np.array([x0, y0], np.int32)
    cv2.fillPoly(mask, [outline - offset], 255)
    cv2.fillPoly(mask, [_points(landmarks, c, cs_x, cs_y) - offset for c in SKELLY_CUTOUTS], 0)
    cpu_frame[y0:y1, x0:x1][mask > 0] = COLOR_SKELLY


def _draw_santa_hat(canvas: np.ndarray, landmarks, cs_x: float, cs_y: float, person_id: int) -> None:
    """--santa-hat: a floppy Santa hat on the temples (landmarks 103 and 332), ported from Party-Pose.py.

    The tip flops right for even person ids and left for odd ones.
    """
    if len(landmarks) < 468:
        return
    (lx, ly), (rx, ry) = _points(landmarks, (SANTA_HAT_LEFT, SANTA_HAT_RIGHT), cs_x, cs_y).tolist()
    center_x, center_y = (lx + rx) // 2, (ly + ry) // 2
    hat_width = int(np.hypot(rx - lx, ry - ly))
    if hat_width < 4:
        return

    hat_initial_height = int(hat_width * 0.6)
    hat_flop_length = int(hat_width * 1.1)
    hat_thickness = int(hat_width * 0.2)
    base_extension = int(hat_width * 0.15)
    brim_height = int(hat_width * 0.15)
    pompom_radius = int(hat_width * 0.12)
    flop = 1 if person_id % 2 == 0 else -1

    base_left = (lx - base_extension, ly)
    base_right = (rx + base_extension, ry)
    peak_x, peak_y = center_x + int(hat_width * 0.2 * flop), center_y - hat_initial_height
    mid_x, mid_y = center_x + int(hat_flop_length * 0.5 * flop), center_y - int(hat_initial_height * 0.4)
    tip_x, tip_y = center_x + int(hat_flop_length * flop), center_y

    top_edge = np.array([base_left,
                         (peak_x - int(hat_thickness * 0.7), peak_y - int(hat_thickness * 0.5)),
                         (mid_x, mid_y - int(hat_thickness * 0.4)),
                         (tip_x, tip_y)], dtype=np.int32)
    bottom_edge = np.array([(tip_x, tip_y),
                            (mid_x, mid_y + int(hat_thickness * 0.4)),
                            (peak_x + int(hat_thickness * 0.7), peak_y + int(hat_thickness * 0.5)),
                            base_right], dtype=np.int32)
    cv2.fillPoly(canvas, [np.vstack([top_edge, bottom_edge])], SANTA_HAT_RED)
    cv2.polylines(canvas, [top_edge, bottom_edge], False, (0, 0, 150), 2, cv2.LINE_AA)

    # White fluffy brim
    puff_radius = int(brim_height * 0.8)
    for i in range(13):
        t = i / 12
        bx = int(base_left[0] * (1 - t) + base_right[0] * t)
        by = int(base_left[1] * (1 - t) + base_right[1] * t) + int(brim_height * 0.3 * np.sin(t * np.pi * 3))
        cv2.circle(canvas, (bx, by), puff_radius, SANTA_HAT_WHITE, -1, cv2.LINE_AA)

    # White fluffy pompom
    for i in range(8):
        angle = i / 8 * 2 * np.pi
        cv2.circle(canvas, (tip_x + int(pompom_radius * 0.3 * np.cos(angle)), tip_y + int(pompom_radius * 0.3 * np.sin(angle))),
                   int(pompom_radius * 0.8), SANTA_HAT_WHITE, -1, cv2.LINE_AA)
    cv2.circle(canvas, (tip_x, tip_y), pompom_radius, SANTA_HAT_WHITE, -1, cv2.LINE_AA)
