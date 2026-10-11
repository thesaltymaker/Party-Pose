from __future__ import annotations
import math

import cv2
import numpy as np
from typing import List, Tuple
from src.types import BoundingBox, Landmark

# crop_rotated_nhwc buffers per crop size: [warped GpuMat, RGB GpuMat, downloaded uint8, float32 output]
_CROP_BUFS: dict = {}


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

        # Resize first: converting the whole 1080p frame cost 1.3 ms on the Orin, the small image almost nothing.
        resized_gpu = cv2.cuda.resize(frame_gpu, (new_w, new_h))
        resized = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2RGB).download()

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
    def roi_from_alignment(centre, size_point, scale: float = 1.25) -> Tuple[float, float, float, float]:
        """MediaPipe's pose ROI from the two alignment points (pose landmarks 33 and 34): a square centred on
        the hips, 2 x their distance x `scale` wide, rotated so the hips-to-size-point line points up.
        Returns (cx, cy, size, angle in radians)."""
        (x0, y0), (x1, y1) = centre, size_point
        angle = math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0)
        angle -= 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))
        return float(x0), float(y0), 2 * math.hypot(x1 - x0, y1 - y0) * scale, angle

    @staticmethod
    def roi_affine(cx: float, cy: float, size: float, angle: float, target: int) -> np.ndarray:
        """2x3 matrix from model pixels (target x target) to frame pixels for a rotated square ROI."""
        c, s, k = math.cos(angle), math.sin(angle), size / target
        h = target / 2
        return np.array([[c * k, -s * k, cx - (c - s) * h * k],
                         [s * k, c * k, cy - (s + c) * h * k]])

    @staticmethod
    def crop_rotated_nhwc(frame_gpu: cv2.cuda_GpuMat, M: np.ndarray, target: int) -> np.ndarray:
        """Crop the ROI given by `roi_affine` into a [1, target, target, 3] RGB float array in [0, 1].
        Outside the frame is black, as in MediaPipe.

        The GPU images, the downloaded image and the returned array are reused for every crop of this size
        (allocating them per call was about half the crop time on the Orin), so the array is only valid until
        the next crop of the same size; the models read it straight away. Call from one thread.
        """
        bufs = _CROP_BUFS.get(target)
        if bufs is None:
            bufs = _CROP_BUFS[target] = [cv2.cuda_GpuMat(target, target, cv2.CV_8UC3),
                                         cv2.cuda_GpuMat(target, target, cv2.CV_8UC3),
                                         np.empty((target, target, 3), np.uint8),
                                         np.empty((1, target, target, 3), np.float32)]
        warp, rgb, host, out = bufs
        # Use the returned GpuMats: on the Orin's OpenCV build dst is not always written in place.
        warp = bufs[0] = cv2.cuda.warpAffine(frame_gpu, cv2.invertAffineTransform(M), (target, target), warp,
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb = bufs[1] = cv2.cuda.cvtColor(warp, cv2.COLOR_BGR2RGB, rgb)
        host = bufs[2] = rgb.download(host)
        np.multiply(host, np.float32(1 / 255), out=out[0], casting='unsafe')
        return out

    @staticmethod
    def to_image_space_affine(lms_model: np.ndarray, M: np.ndarray) -> List[Landmark]:
        """Landmarks [N, 5] (x, y, z, visibility, presence in model pixels) to frame pixels through `roi_affine`."""
        xy = lms_model[:, :2] @ M[:, :2].T + M[:, 2]
        zs = lms_model[:, 2] * math.hypot(M[0, 0], M[1, 0])
        return [Landmark(x=x, y=y, z=z, visibility=v, presence=p)
                for (x, y), z, v, p in zip(xy.tolist(), zs.tolist(), lms_model[:, 3].tolist(),
                                           lms_model[:, 4].tolist())]

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
        # Vectorized: per-element numpy scalar indexing cost ~4 ms per 478-point face on the Orin
        n, cols = lms_model.shape[0], lms_model.shape[1]
        xs = (crop_bbox.x + lms_model[:, 0] / model_w * crop_bbox.w).tolist()
        ys = (crop_bbox.y + lms_model[:, 1] / model_h * crop_bbox.h).tolist()
        zs = lms_model[:, 2].tolist() if cols >= 3 else [0.0] * n
        vis = lms_model[:, 3].tolist() if cols >= 4 else [1.0] * n
        pres = lms_model[:, 4].tolist() if cols >= 5 else [1.0] * n
        return [
            Landmark(x=x, y=y, z=z, visibility=v, presence=p)
            for x, y, z, v, p in zip(xs, ys, zs, vis, pres)
        ]
