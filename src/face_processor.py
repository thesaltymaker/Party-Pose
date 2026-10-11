from __future__ import annotations
import numpy as np
from typing import List, Optional
from src.types import BoundingBox, FaceResult
from src.model_manager import ModelManager
from src.gpu_model import gpu_face_batch
from src.preprocessor import Preprocessor
from src.topology import CANONICAL_BLENDSHAPE_INDICES


class FaceProcessor:
    """Runs face landmark inference on a pre-detected head bounding box.

    The detection stage has been moved to PersonDetector (YOLOX).
    This class only handles the landmark (and optional blendshape) step.
    """

    LANDMARK_W = 256
    LANDMARK_H = 256
    # Face landmark model's own face score (sigmoid of Identity_1). Measured on Orin dumps
    # (issue #3): with the unpadded head crop, frontal faces score well above 0.5; the black
    # ball, fists, lampshade and backs of heads score below it.
    PRESENCE_THRESHOLD = 0.5

    # Faces per model run. On the Orin two faces in one run cost 2.89 ms against 2 x 2.36 ms; the TensorRT engine
    # is built for batches of 1-4 (ModelManager), so more people are split into runs of up to 4.
    MAX_BATCH = 4

    def __init__(self, model_manager: ModelManager, enable_blendshapes: bool = False) -> None:
        self.model_manager = model_manager
        self.enable_blendshapes = enable_blendshapes
        self.last_logit: Optional[float] = None  # face score logit of the last process() call, kept or not
        self.last_logits: List[float] = []       # the same for each head of the last process_many() call

    def process(
        self,
        frame_gpu,
        head_bbox: BoundingBox,
        frame_w: int,
        frame_h: int,
        mirror: bool,
    ) -> Optional[FaceResult]:
        face = self.process_many(frame_gpu, [head_bbox], frame_w, frame_h, mirror)[0]
        self.last_logit = self.last_logits[0]
        return face

    def process_many(self, frame_gpu, head_bboxes: List[BoundingBox], frame_w: int, frame_h: int,
                     mirror: bool) -> List[Optional[FaceResult]]:
        """One FaceResult (or None when the face score is too low) per head box, in order. All heads go through
        the face model together, up to MAX_BATCH per run."""
        if not head_bboxes:
            self.last_logits = []
            return []
        gpu = gpu_face_batch(self.model_manager)
        raws, logits = [], []
        if gpu.on_gpu:
            rects = [Preprocessor.crop_rect(head, 0.0, frame_w, frame_h) for head in head_bboxes]
            crop_boxes = [BoundingBox(x, y, w, h, head.confidence) for (x, y, w, h), head in zip(rects, head_bboxes)]
            for i in range(0, len(rects), gpu.SLOTS):
                lms, lg = gpu.run(frame_gpu, rects[i:i + gpu.SLOTS])
                raws.extend(lms)
                logits.extend(float(v) for v in lg)
        else:
            crops = [Preprocessor.crop_roi_nhwc(frame_gpu, head, self.LANDMARK_W, self.LANDMARK_H,
                                                pad_fraction=0.0, frame_w=frame_w, frame_h=frame_h)
                     for head in head_bboxes]
            crop_boxes = [box for _, box in crops]
            session = self.model_manager.get_session('face_landmarks')
            for i in range(0, len(crops), self.MAX_BATCH):
                batch = np.concatenate([arr for arr, _ in crops[i:i + self.MAX_BATCH]])
                landmarks_raw, presence_logit = session.run(['Identity', 'Identity_1'], {'input_12': batch})
                raws.extend(landmarks_raw.reshape(-1, 478, 3))
                logits.extend(float(v) for v in presence_logit.reshape(-1))
        self.last_logits = logits
        return [self._result(head, crop_bbox, raw, logit, frame_w, frame_h, mirror)
                for head, crop_bbox, raw, logit in zip(head_bboxes, crop_boxes, raws, logits)]

    def _result(self, head_bbox: BoundingBox, crop_bbox: BoundingBox, raw: np.ndarray, logit: float,
                frame_w: int, frame_h: int, mirror: bool) -> Optional[FaceResult]:
        presence = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))
        if presence < self.PRESENCE_THRESHOLD:
            return None
        landmarks = Preprocessor.to_image_space(raw, self.LANDMARK_W, self.LANDMARK_H,
                                                crop_bbox, frame_w, frame_h, mirror)
        blendshapes = self._run_blendshapes(raw) if self.enable_blendshapes else None
        pts = np.c_[crop_bbox.x + raw[:, 0] / self.LANDMARK_W * crop_bbox.w,
                    crop_bbox.y + raw[:, 1] / self.LANDMARK_H * crop_bbox.h]
        return FaceResult(bbox=head_bbox, landmarks=landmarks, blendshapes=blendshapes, presence=presence, pts=pts)

    def _run_blendshapes(self, landmarks_raw: np.ndarray) -> Optional[np.ndarray]:
        if not CANONICAL_BLENDSHAPE_INDICES:
            return None
        subset = landmarks_raw[CANONICAL_BLENDSHAPE_INDICES, :2] / 256.0
        subset = np.expand_dims(subset, axis=0)
        session = self.model_manager.get_session('face_blendshapes')
        blendshapes = session.run(['StatefulPartitionedCall:0'], {'serving_default_input_points:0': subset})[0]
        return blendshapes[0] if blendshapes.size > 0 else None
