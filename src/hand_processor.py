from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Sequence, Tuple
from src.types import BoundingBox, HandResult, Landmark
from src.model_manager import ModelManager
from src.preprocessor import Preprocessor

Roi = Tuple[float, float, float, float]  # (cx, cy, size, angle), as Preprocessor.roi_affine takes

# Pose landmarks per side: (wrist, pinky, index). Side 0 is the pose model's left, side 1 its right.
_POSE_HAND = ((15, 17, 19), (16, 18, 20))
# Hand landmarks MediaPipe uses for the next frame's crop: wrist, thumb base, and the base two joints of each finger.
_PALM = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]


def _rect(cx: float, cy: float, w: float, h: float, angle: float, scale: float, shift_y: float) -> Roi:
    """MediaPipe's RectTransformation: move the centre by shift_y x h along the rect's own up axis, then make it
    square on its longer side and scale it."""
    cx -= h * shift_y * math.sin(angle)
    cy += h * shift_y * math.cos(angle)
    return cx, cy, max(w, h) * scale, angle


def hand_roi_from_pose(pose: Sequence[Landmark], side: int) -> Roi:
    """MediaPipe Holistic's hand crop from the pose model's wrist, pinky and index points, at 2.0x, not Holistic's
    2.7x: Holistic then re-crops with a second model we don't have. At 2.7x the hand was too small in the crop and
    the pose's left-hand points are less exact, so 67% of left hands were found (98% right); at 2.0x 91% / 96%."""
    w, p, i = (np.array([pose[k].x, pose[k].y]) for k in _POSE_HAND[side])
    middle = (2 * i + p) / 3
    size = 2 * float(np.hypot(*(middle - w)))
    angle = Preprocessor.roi_from_alignment(w, middle)[3]
    return _rect(middle[0], middle[1], size, size, angle, 2.0, -0.1)


def hand_roi_from_landmarks(pts: np.ndarray) -> Roi:
    """MediaPipe's next-frame hand crop from 21 hand points [21, 2]: the palm's box, upright from wrist to middle
    finger base, twice as big."""
    angle = Preprocessor.roi_from_alignment(pts[0], pts[9])[3]
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, s], [-s, c]])  # frame -> upright
    up = pts[_PALM] @ rot.T
    lo, hi = up.min(axis=0), up.max(axis=0)
    cx, cy = ((lo + hi) / 2) @ rot  # back to the frame
    return _rect(cx, cy, hi[0] - lo[0], hi[1] - lo[1], angle, 2.0, -0.1)


class HandProcessor:
    """Runs the hand landmark model on each of a person's two hands.

    Crops follow MediaPipe Holistic: from the hand's own landmarks in the previous frame, else from the pose model's
    wrist, pinky and index points. One model run per hand per frame (Orin FPS): a hand lost at its previous crop
    is cropped from the pose on the next frame. The detector's hand boxes found 40% of hands on the laptop (0% when the person
    was shrunk to a third, as at the Orin's 13-16 ft); these crops found 92% (12% shrunk).

    The model's outputs are landmarks (Identity), presence (Identity_1) and handedness (Identity_2). Reading
    handedness as presence dropped most hands of one side.
    """

    LANDMARK_SIZE = 224
    CONF_THRESHOLD = 0.5
    WRIST_VISIBILITY = 0.5  # pose wrist visibility below this: no crop from the pose for that side

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def process(self, frame_gpu, pose: Sequence[Landmark], prev_rois: Sequence[Optional[Roi]]) -> List[Optional[HandResult]]:
        """One HandResult or None per side. prev_rois: each side's HandResult.roi from the previous frame, or None."""
        out: List[Optional[HandResult]] = []
        for side in (0, 1):
            roi = prev_rois[side]
            if roi is None and pose[_POSE_HAND[side][0]].visibility >= self.WRIST_VISIBILITY:
                roi = hand_roi_from_pose(pose, side)
            out.append(self._run(frame_gpu, roi) if roi is not None else None)
        return out

    def _run(self, frame_gpu, roi: Roi) -> Optional[HandResult]:
        M = Preprocessor.roi_affine(*roi, self.LANDMARK_SIZE)
        arr = Preprocessor.crop_rotated_nhwc(frame_gpu, M, self.LANDMARK_SIZE)
        lms, presence, handedness = self.model_manager.get_session('hand_landmarks').run(
            ['Identity', 'Identity_1', 'Identity_2'], {'input_1': arr})
        presence, handedness = float(presence[0, 0]), float(handedness[0, 0])
        if presence < self.CONF_THRESHOLD:
            return None
        lms = lms.reshape(21, 3)
        landmarks = Preprocessor.to_image_space_affine(np.c_[lms, np.ones((21, 2), np.float32)], M)
        pts = lms[:, :2] @ M[:, :2].T + M[:, 2]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return HandResult(
            bbox=BoundingBox(float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]), presence),
            landmarks=landmarks,
            handedness='Right' if handedness > 0.5 else 'Left',
            handedness_score=handedness,
            presence=presence,
            roi=hand_roi_from_landmarks(pts),
            pts=pts,
        )
