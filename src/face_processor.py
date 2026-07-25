from __future__ import annotations
import numpy as np
from typing import Optional
from src.types import BoundingBox, FaceResult
from src.model_manager import ModelManager
from src.preprocessor import Preprocessor
from src.topology import CANONICAL_BLENDSHAPE_INDICES


class FaceProcessor:
    """Runs face landmark inference on a pre-detected head bounding box.

    The detection stage has been moved to PersonDetector (YOLOX).
    This class only handles the landmark (and optional blendshape) step.
    """

    LANDMARK_W = 256
    LANDMARK_H = 256

    def __init__(self, model_manager: ModelManager, enable_blendshapes: bool = False) -> None:
        self.model_manager = model_manager
        self.enable_blendshapes = enable_blendshapes

    def process(
        self,
        frame_gpu,
        head_bbox: BoundingBox,
        frame_w: int,
        frame_h: int,
        mirror: bool,
    ) -> Optional[FaceResult]:
        roi_arr, crop_bbox = Preprocessor.crop_roi_nhwc(
            frame_gpu, head_bbox, self.LANDMARK_W, self.LANDMARK_H,
            pad_fraction=0.25, frame_w=frame_w, frame_h=frame_h,
        )

        session = self.model_manager.get_session('face_landmarks')
        landmarks_raw = session.run(['Identity'], {'input_12': roi_arr})[0]
        landmarks_raw = landmarks_raw.reshape(-1, 478, 3)

        landmarks = Preprocessor.to_image_space(
            landmarks_raw[0], self.LANDMARK_W, self.LANDMARK_H,
            crop_bbox, frame_w, frame_h, mirror,
        )

        blendshapes = None
        if self.enable_blendshapes and landmarks_raw.size > 0:
            blendshapes = self._run_blendshapes(landmarks_raw[0])

        return FaceResult(bbox=head_bbox, landmarks=landmarks, blendshapes=blendshapes)

    def _run_blendshapes(self, landmarks_raw: np.ndarray) -> Optional[np.ndarray]:
        if not CANONICAL_BLENDSHAPE_INDICES:
            return None
        subset = landmarks_raw[CANONICAL_BLENDSHAPE_INDICES, :2] / 256.0
        subset = np.expand_dims(subset, axis=0)
        session = self.model_manager.get_session('face_blendshapes')
        blendshapes = session.run(['StatefulPartitionedCall:0'], {'serving_default_input_points:0': subset})[0]
        return blendshapes[0] if blendshapes.size > 0 else None
