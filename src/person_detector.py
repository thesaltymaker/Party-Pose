from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.types import BoundingBox
from src.gpu_model import GpuDetectorInput
from src.preprocessor import Preprocessor
from src.model_manager import ModelManager


@dataclass
class PersonDetections:
    body_boxes: List[BoundingBox] = field(default_factory=list)
    head_boxes: List[BoundingBox] = field(default_factory=list)
    hand_boxes: List[BoundingBox] = field(default_factory=list)


class PersonDetector:
    """First-stage detector: YOLOX-Body-Head-Hand.

    Replaces the separate face_detector and hand_detector models with a single
    pass that returns body, head, and hand bounding boxes for all people in frame.

    Output tensor 'batchno_classid_score_x1y1x2y2': shape [60, 7]
        columns: [batch_no, class_id, score, x1, y1, x2, y2]
        x1/y1/x2/y2 are in model input pixel space (320×256 for the default model; set from the model's input).
        class_id: 0=Body  1=Head  2=Hand
    """

    INPUT_W = 320  # the default model's input; replaced by the loaded model's own size on first use (--detector)
    INPUT_H = 256
    CONF_THRESHOLD = 0.3
    CLASS_BODY = 0
    CLASS_HEAD = 1
    CLASS_HAND = 2

    def __init__(self, model_manager: ModelManager, bgr: bool = False, area: bool = False) -> None:
        self.model_manager = model_manager
        self.bgr = bgr  # feed BGR, the colour order the model was trained on (--detector-bgr); default RGB
        self.interpolation = cv2.INTER_AREA if area else cv2.INTER_LINEAR  # --detector-area: average, less grain
        self._gpu: Optional[GpuDetectorInput] = None  # created on first use (loads the session)

    def process(self, frame_gpu, frame_w: int, frame_h: int) -> PersonDetections:
        session = self.model_manager.get_session('person_detector')
        if self._gpu is None:
            _, self.INPUT_H, self.INPUT_W, _ = session.get_inputs()[0].shape  # NHWC [1, H, W, 3]
            self._gpu = GpuDetectorInput(session, 'input', 'batchno_classid_score_x1y1x2y2', self.INPUT_W, self.INPUT_H,
                                         self.bgr, self.interpolation)
        if self._gpu.on_gpu:
            raw = self._gpu.run(frame_gpu, frame_w, frame_h)
        else:
            arr = Preprocessor.full_frame_nhwc(frame_gpu, self.INPUT_W, self.INPUT_H, normalize=False,
                                               interpolation=self.interpolation)
            if self.bgr:
                arr = np.ascontiguousarray(arr[..., ::-1])
            raw = session.run(['batchno_classid_score_x1y1x2y2'], {'input': arr})[0]
        # raw: [60, 7]  —  [batch_no, class_id, score, x1, y1, x2, y2]

        scale, pad_x, pad_y = Preprocessor.letterbox_params(frame_w, frame_h, self.INPUT_W, self.INPUT_H)

        result = PersonDetections()
        for row in raw:
            _, class_id, score, x1, y1, x2, y2 = row
            if score < self.CONF_THRESHOLD:
                continue
            fx1 = max(0.0, (x1 - pad_x) / scale)
            fy1 = max(0.0, (y1 - pad_y) / scale)
            fx2 = min(float(frame_w), (x2 - pad_x) / scale)
            fy2 = min(float(frame_h), (y2 - pad_y) / scale)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            bbox = BoundingBox(x=fx1, y=fy1, w=fx2 - fx1, h=fy2 - fy1, confidence=float(score))
            class_id = int(class_id)
            if class_id == self.CLASS_BODY:
                result.body_boxes.append(bbox)
            elif class_id == self.CLASS_HEAD:
                result.head_boxes.append(bbox)
            elif class_id == self.CLASS_HAND:
                result.hand_boxes.append(bbox)

        return result
