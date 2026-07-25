from __future__ import annotations
import numpy as np
from typing import List
from src.types import BoundingBox, HandResult
from src.model_manager import ModelManager
from src.preprocessor import Preprocessor


class HandProcessor:
    """Runs hand landmark inference on pre-detected hand bounding boxes.

    The detection stage has been moved to PersonDetector (YOLOX).
    This class only handles the landmark step for each provided hand bbox.
    """

    LANDMARK_W = 224
    LANDMARK_H = 224
    CONF_THRESHOLD = 0.5

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def process(
        self,
        frame_gpu,
        hand_bboxes: List[BoundingBox],
        frame_w: int,
        frame_h: int,
        mirror: bool,
    ) -> List[HandResult]:
        results = []
        lm_session = self.model_manager.get_session('hand_landmarks')

        for bbox in hand_bboxes:
            roi_arr, crop_bbox = Preprocessor.crop_roi_nhwc(
                frame_gpu, bbox, self.LANDMARK_W, self.LANDMARK_H,
                pad_fraction=0.8, frame_w=frame_w, frame_h=frame_h,
            )
            lm_outputs = lm_session.run(
                ['Identity', 'Identity_1', 'Identity_2'], {'input_1': roi_arr}
            )
            landmarks_raw_flat = lm_outputs[0]
            handedness_score   = float(lm_outputs[1][0, 0])
            presence_score     = float(lm_outputs[2][0, 0])

            if presence_score < self.CONF_THRESHOLD:
                continue

            landmarks_raw = landmarks_raw_flat.reshape(21, 3)
            landmarks = Preprocessor.to_image_space(
                landmarks_raw, self.LANDMARK_W, self.LANDMARK_H,
                crop_bbox, frame_w, frame_h, mirror,
            )
            handedness = 'Right' if handedness_score > 0.5 else 'Left'
            results.append(HandResult(
                bbox=crop_bbox,
                landmarks=landmarks,
                handedness=handedness,
                handedness_score=handedness_score,
            ))
        return results
