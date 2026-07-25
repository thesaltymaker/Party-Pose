from __future__ import annotations
import numpy as np
from typing import Optional
from src.types import BoundingBox, BodyResult
from src.model_manager import ModelManager
from src.preprocessor import Preprocessor


class BodyProcessor:
    """Runs pose landmark inference on a pre-detected body bounding box.

    The detection stage has been moved to PersonDetector (YOLOX).
    The body crop is letterboxed into the model's 256×256 input to preserve
    aspect ratio for tall person bounding boxes.
    """

    INPUT_W = 256
    INPUT_H = 256
    PRESENCE_THRESHOLD = 0.5

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def process(
        self,
        frame_gpu,
        body_bbox: BoundingBox,
        frame_w: int,
        frame_h: int,
        mirror: bool,
    ) -> Optional[BodyResult]:
        arr, virtual_bbox = Preprocessor.crop_roi_nhwc(
            frame_gpu, body_bbox, self.INPUT_W, self.INPUT_H,
            pad_fraction=0.1, frame_w=frame_w, frame_h=frame_h,
            letterbox=True,
        )

        session = self.model_manager.get_session('pose_landmarks')
        outputs = session.run(['Identity', 'Identity_1'], {'input_1': arr})

        body_presence = float(outputs[1][0, 0])
        if body_presence < self.PRESENCE_THRESHOLD:
            return None

        lms = outputs[0].reshape(39, 5)
        landmarks = Preprocessor.to_image_space(
            lms, self.INPUT_W, self.INPUT_H, virtual_bbox, frame_w, frame_h, mirror,
        )
        return BodyResult(landmarks=landmarks, presence=body_presence)
