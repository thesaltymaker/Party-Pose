from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple
from src.types import BoundingBox, Landmark

class Preprocessor:
    """Stateless preprocessor for GPU-resident pose detection pipeline."""

    @staticmethod
    def letterbox_params(frame_w: int, frame_h: int, target_w: int, target_h: int) -> Tuple[float, int, int]:
        """Return (scale, pad_x, pad_y) for letterbox resize into target (model-pixel units)."""
        scale = min(target_w / frame_w, target_h / frame_h)
        new_w = int(round(frame_w * scale))
        new_h = int(round(frame_h * scale))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        return scale, pad_x, pad_y

    @staticmethod
    def full_frame_nhwc(
        frame_gpu: cv2.cuda_GpuMat,
        target_w: int,
        target_h: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """Letterbox the full frame to target dims and return a [1, H, W, 3] float32 array.

        normalize=True  → values in [0, 1]  (MediaPipe models)
        normalize=False → values in [0, 255] (YOLOX model)
        """
        frame_w, frame_h = frame_gpu.size()
        scale, pad_x, pad_y = Preprocessor.letterbox_params(frame_w, frame_h, target_w, target_h)
        new_w = int(round(frame_w * scale))
        new_h = int(round(frame_h * scale))

        rgb_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2RGB)
        resized_gpu = cv2.cuda.resize(rgb_gpu, (new_w, new_h))
        resized = resized_gpu.download()

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        arr = padded.astype(np.float32)
        if normalize:
            arr = arr / 255.0
        return np.expand_dims(arr, axis=0)

    @staticmethod
    def crop_roi_nhwc(
        frame_gpu: cv2.cuda_GpuMat,
        bbox: BoundingBox,
        target_w: int,
        target_h: int,
        pad_fraction: float = 0.25,
        frame_w: int = 0,
        frame_h: int = 0,
        letterbox: bool = False,
    ) -> Tuple[np.ndarray, BoundingBox]:
        """Crop a bbox ROI from the GPU frame and resize to target dimensions.

        When letterbox=False (default): stretches the crop to target_w × target_h.
            Returns the actual crop bbox; to_image_space divides by crop dims.

        When letterbox=True: letterboxes the crop into target_w × target_h with zero padding.
            Returns a virtual bbox that encodes both the outer crop offset and the inner
            letterbox padding so that to_image_space maps model coords back to frame space.
            Use this for tall/wide crops (e.g. body bboxes) to avoid aspect-ratio distortion.

        Always normalizes to [0, 1] — all landmark models (MediaPipe) expect this range.
        """
        pad_x = bbox.w * pad_fraction
        pad_y = bbox.h * pad_fraction
        crop_x = max(0, bbox.x - pad_x)
        crop_y = max(0, bbox.y - pad_y)
        crop_w = min(frame_w - crop_x, bbox.w + 2 * pad_x)
        crop_h = min(frame_h - crop_y, bbox.h + 2 * pad_y)

        roi_gpu = cv2.cuda_GpuMat(frame_gpu, (int(crop_x), int(crop_y), int(crop_w), int(crop_h)))
        rgb_gpu = cv2.cuda.cvtColor(roi_gpu, cv2.COLOR_BGR2RGB)

        if letterbox:
            scale = min(target_w / crop_w, target_h / crop_h)
            new_w = int(round(crop_w * scale))
            new_h = int(round(crop_h * scale))
            inner_pad_x = (target_w - new_w) // 2
            inner_pad_y = (target_h - new_h) // 2
            resized_gpu = cv2.cuda.resize(rgb_gpu, (new_w, new_h))
            resized = resized_gpu.download()
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            padded[inner_pad_y:inner_pad_y + new_h, inner_pad_x:inner_pad_x + new_w] = resized
            arr = padded.astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            # Virtual bbox: to_image_space will compute
            #   x_frame = virtual_bbox.x + (x_model / target_w) * virtual_bbox.w
            #           = (crop_x - inner_pad_x/scale) + x_model/scale
            #           = crop_x + (x_model - inner_pad_x) / scale  ✓
            virtual_bbox = BoundingBox(
                x=crop_x - inner_pad_x / scale,
                y=crop_y - inner_pad_y / scale,
                w=target_w / scale,
                h=target_h / scale,
                confidence=bbox.confidence,
            )
            return arr, virtual_bbox
        else:
            resized_gpu = cv2.cuda.resize(rgb_gpu, (target_w, target_h))
            arr = resized_gpu.download()
            arr = arr.astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            actual_bbox = BoundingBox(
                x=crop_x, y=crop_y, w=crop_w, h=crop_h, confidence=bbox.confidence
            )
            return arr, actual_bbox

    @staticmethod
    def to_image_space(
        lms_model: np.ndarray,
        model_w: int,
        model_h: int,
        crop_bbox: BoundingBox,
        frame_w: int,
        frame_h: int,
        mirror: bool = False,  # unused: frame is already flipped by video_capture
    ) -> List[Landmark]:
        landmarks = []
        for i in range(lms_model.shape[0]):
            x_norm = lms_model[i, 0] / model_w
            y_norm = lms_model[i, 1] / model_h
            x_image = crop_bbox.x + x_norm * crop_bbox.w
            y_image = crop_bbox.y + y_norm * crop_bbox.h
            z = lms_model[i, 2] if lms_model.shape[1] >= 3 else 0.0
            visibility = lms_model[i, 3] if lms_model.shape[1] >= 4 else 1.0
            presence = lms_model[i, 4] if lms_model.shape[1] >= 5 else 1.0

            landmarks.append(
                Landmark(
                    x=x_image,
                    y=y_image,
                    z=z,
                    visibility=visibility,
                    presence=presence
                )
            )
        return landmarks
