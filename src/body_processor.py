from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from src.types import BoundingBox, BodyResult
from src.model_manager import ModelManager
from src.gpu_model import GpuModel, gpu_model
from src.preprocessor import Preprocessor


class BodyProcessor:
    """Runs pose landmark inference for one person.

    The pose model (BlazePose GHUM) expects MediaPipe's crop: a square centred on the hips, sized and rotated
    from the model's own alignment landmarks (33 = hip centre, 34 = body-size point), black outside the frame.
    The person box letterboxed into the model instead gave a skeleton too small for the person, with low
    shoulders and short arms (laptop, 99 frames vs MediaPipe: median error 0.22 shoulder widths; with the ROI
    from the previous frame's landmarks, 0.093).

    With `roi` from this person's previous frame (BodyResult.roi), the model runs once on it. Otherwise (new
    person, or the ROI centre is far from the body box) it runs on the letterboxed box first to find the
    alignment landmarks, then on the ROI from them.
    """

    INPUT_W = 256
    INPUT_H = 256
    PRESENCE_THRESHOLD = 0.5

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager
        self._model: Optional[GpuModel] = None  # created on first use (loads the session)

    def process(
        self,
        frame_gpu,
        body_bbox: BoundingBox,
        frame_w: int,
        frame_h: int,
        mirror: bool,
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[BodyResult]:
        if roi is None or not self._near(roi, body_bbox):
            arr, virtual_bbox = Preprocessor.crop_roi_nhwc(
                frame_gpu, body_bbox, self.INPUT_W, self.INPUT_H,
                pad_fraction=0.1, frame_w=frame_w, frame_h=frame_h,
                letterbox=True,
            )
            lms, presence = self._outputs(self._gpu().run_array(arr))
            if presence < self.PRESENCE_THRESHOLD:
                return None
            first = Preprocessor.to_image_space(lms, self.INPUT_W, self.INPUT_H, virtual_bbox, frame_w, frame_h)
            roi = Preprocessor.roi_from_alignment((first[33].x, first[33].y), (first[34].x, first[34].y))

        M = Preprocessor.roi_affine(*roi, self.INPUT_W)
        lms, presence = self._outputs(self._gpu().run(frame_gpu, M))
        if presence < self.PRESENCE_THRESHOLD:
            return None
        landmarks = Preprocessor.to_image_space_affine(lms, M)
        next_roi = Preprocessor.roi_from_alignment((landmarks[33].x, landmarks[33].y),
                                                   (landmarks[34].x, landmarks[34].y))
        return BodyResult(landmarks=landmarks, presence=presence, roi=next_roi,
                          pts=lms[:, :2] @ M[:, :2].T + M[:, 2])

    def _gpu(self) -> GpuModel:
        if self._model is None:
            self._model = gpu_model(self.model_manager, 'pose_landmarks', 'input_1', ['Identity', 'Identity_1'],
                                   self.INPUT_W)
        return self._model

    @staticmethod
    def _outputs(outputs) -> Tuple[np.ndarray, float]:
        return outputs[0].reshape(39, 5), float(outputs[1][0, 0])

    @staticmethod
    def _near(roi, box: BoundingBox) -> bool:
        """The ROI centre (hips) lies inside the body box grown by half its size on each side."""
        cx, cy = roi[0], roi[1]
        return (box.x - box.w / 2 <= cx <= box.x + 1.5 * box.w) and (box.y - box.h / 2 <= cy <= box.y + 1.5 * box.h)
