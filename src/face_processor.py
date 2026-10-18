from __future__ import annotations
import numpy as np
from typing import List, Optional
from src.types import BoundingBox, FaceResult
from src.model_manager import ModelManager
from src.gpu_model import gpu_batch, gpu_face_batch
from src.preprocessor import Preprocessor
from src.topology import CANONICAL_BLENDSHAPE_INDICES


# Face crop from the pose (issue #13): a square around the pose's face points (0-10: nose, eyes, ears, mouth), this
# many times their larger spread. On two Orin clips it found 81% / 92% of faces, the detector's head box 50% / 58%
# (the head box is missing in a third of frames; 1.5x: 55% / 82%, 2.5x: 70% / 89%).
FACE_FROM_POSE_SCALE = 2.0


def face_box_from_pose(pose, scale: float = FACE_FROM_POSE_SCALE) -> BoundingBox:
    """The face crop for a person from their pose landmarks: a square around points 0-10, `scale` x their spread."""
    xs = [pose[i].x for i in range(11)]
    ys = [pose[i].y for i in range(11)]
    side = scale * max(max(xs) - min(xs), max(ys) - min(ys))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return BoundingBox(cx - side / 2, cy - side / 2, side, side, 1.0)


# --face-keep: a face kept last frame is kept while its logit stays at or above the hold value and the head is less
# than this turned (head_turn). Rejected faces in the evening and dance clips are mostly real faces, blurred or seen
# from below (logit -1 to -13); side-on heads and backs of heads give head_turn >= 0.4.
FACE_HOLD_TURN = 0.4


def head_turn(pose) -> float:
    """How far the head is turned, from the pose: the nose's x off the ears' middle, in ear distances (0 facing the
    camera, about +-0.5 side on). 0 when the ears are less than a pixel apart."""
    ears = pose[7].x - pose[8].x
    if abs(ears) < 1:
        return 0.0
    return (pose[0].x - (pose[7].x + pose[8].x) / 2) / abs(ears)


def face_min_logit(pose, held: bool, hold_logit: float) -> float:
    """The face logit a person's face needs this frame: 0 (presence 0.5), or hold_logit when it was kept last frame
    and the pose's head is not turned away (--face-keep)."""
    if held and pose is not None and abs(head_turn(pose)) < FACE_HOLD_TURN:
        return min(0.0, hold_logit)
    return 0.0


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

    # Faces per model run. On the Orin two faces in one run take 2.89 ms against 2 x 2.36 ms; the TensorRT engine
    # is built for batches of 1-4 (ModelManager), so more people are split into runs of up to 4.
    MAX_BATCH = 4

    def __init__(self, model_manager: ModelManager, enable_blendshapes: bool = False, batch: int = 0) -> None:
        self.model_manager = model_manager
        self.batch = batch  # --face-batch: up to this many faces per run, a session per batch size (0 = GpuFaceBatch)
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
                     mirror: bool, min_logits: Optional[List[float]] = None) -> List[Optional[FaceResult]]:
        """One FaceResult (or None when the face score is too low) per head box, in order. All heads go through
        the face model together, up to MAX_BATCH per run. min_logits: the logit each face needs (face_min_logit);
        without it, presence PRESENCE_THRESHOLD."""
        if not head_bboxes:
            self.last_logits = []
            return []
        gpu = gpu_batch(self.model_manager, 'face_landmarks', 'input_12', ['Identity', 'Identity_1'], self.LANDMARK_W,
                        self.batch) if self.batch else gpu_face_batch(self.model_manager)
        raws, logits = [], []
        if gpu.on_gpu:
            rects = [Preprocessor.crop_rect(head, 0.0, frame_w, frame_h) for head in head_bboxes]
            crop_boxes = [BoundingBox(x, y, w, h, head.confidence) for (x, y, w, h), head in zip(rects, head_bboxes)]
            if self.batch:
                lms, lg = gpu.run_rects(frame_gpu, rects)
                raws.extend(lms.reshape(-1, 478, 3))
                logits.extend(float(v) for v in lg.reshape(-1))
            else:
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
        mins = min_logits or [None] * len(raws)
        return [self._result(head, crop_bbox, raw, logit, frame_w, frame_h, mirror, m)
                for head, crop_bbox, raw, logit, m in zip(head_bboxes, crop_boxes, raws, logits, mins)]

    def _result(self, head_bbox: BoundingBox, crop_bbox: BoundingBox, raw: np.ndarray, logit: float,
                frame_w: int, frame_h: int, mirror: bool, min_logit: Optional[float] = None) -> Optional[FaceResult]:
        presence = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))
        if presence < self.PRESENCE_THRESHOLD if min_logit is None else logit < min_logit:
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
