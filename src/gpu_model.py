"""Run a landmark model on a crop that never leaves the GPU.

Profile of a two-people replay on the Orin (nsys): the GPU computed only ~23% of the time. The models are
hundreds of tiny TensorRT kernels (~900 launches a frame, ~12 us each, about what a launch costs the CPU), and
every crop went GPU -> CPU (download, float) -> GPU (ONNX Runtime's input copy) with a wait at each step.

GpuModel keeps the crop on the GPU: rotate, RGB and float/255 run into one continuous GPU buffer that is bound
to the model's input once (ONNX Runtime I/O binding), and the outputs go to fixed GPU buffers. Because the
addresses never change, the Orin's TensorRT sessions for these models record their kernel launches once as a
CUDA graph and replay them as one (ModelManager: trt_cuda_graph_enable). Orin, crop + run: hand 1.68 -> 1.11 ms,
body 2.64 -> 1.84 ms, same outputs.

A session with CUDA graphs must only be run this way, and through one GpuModel: the graph is recorded with the
first run's buffer addresses, so a second GpuModel's buffers would be ignored on replay. Get it with gpu_model(),
which keeps one per model. Without CUDA (CPU-only), it falls back to cropping to numpy and session.run. Use it
from one thread.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from src.preprocessor import Preprocessor


def gpu_model(model_manager, model: str, input_name: str, outputs: List[str], size: int) -> 'GpuModel':
    """The one GpuModel for this model (created on first use)."""
    models = model_manager.__dict__.setdefault('_gpu_models', {})
    if model not in models:
        models[model] = GpuModel(model_manager, model, input_name, outputs, size)
    return models[model]


class GpuModel:
    def __init__(self, model_manager, model: str, input_name: str, outputs: List[str], size: int) -> None:
        self.session = model_manager.get_session(model)
        self.input_name, self.outputs, self.size = input_name, outputs, size
        self.on_gpu = any(p in self.session.get_providers()
                          for p in ('TensorrtExecutionProvider', 'CUDAExecutionProvider'))
        if not self.on_gpu:
            return
        self._opencv_stream = cv2.cuda.Stream.Null()
        self._warp = cv2.cuda_GpuMat(size, size, cv2.CV_8UC3)
        self._rgb = cv2.cuda_GpuMat(size, size, cv2.CV_8UC3)
        self._input = cv2.cuda.createContinuous(size, size, cv2.CV_32FC3, cv2.cuda_GpuMat())
        self._bound_ptr = self._input.cudaPtr()
        self._binding = self.session.io_binding()
        self._binding.bind_input(input_name, 'cuda', 0, np.float32, [1, size, size, 3], self._bound_ptr)
        meta = {o.name: o.shape for o in self.session.get_outputs()}
        self._out = []
        for name in outputs:
            shape = [d if isinstance(d, int) else 1 for d in meta[name]]
            value = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, 'cuda', 0)
            self._binding.bind_ortvalue_output(name, value)
            self._out.append(value)

    def input_ptr(self) -> int:
        return self._input.cudaPtr()

    def run(self, frame_gpu: cv2.cuda_GpuMat, M: np.ndarray) -> List[np.ndarray]:
        """The model's outputs for the rotated crop given by Preprocessor.roi_affine (black outside the frame)."""
        if not self.on_gpu:
            return self.session.run(self.outputs, {self.input_name: Preprocessor.crop_rotated_nhwc(frame_gpu, M, self.size)})
        # Use the returned GpuMats: on the Orin's OpenCV build dst is not always written in place.
        self._warp = cv2.cuda.warpAffine(frame_gpu, cv2.invertAffineTransform(M), (self.size, self.size), self._warp,
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self._rgb = cv2.cuda.cvtColor(self._warp, cv2.COLOR_BGR2RGB, self._rgb)
        out = self._rgb.convertTo(cv2.CV_32FC3, self._input, 1 / 255.0, 0.0)
        if out.cudaPtr() != self._bound_ptr:
            raise RuntimeError('crop was not written into the bound GPU input buffer')
        return self._run()

    def run_array(self, arr: np.ndarray) -> List[np.ndarray]:
        """The model's outputs for a [1, size, size, 3] float array already prepared on the CPU."""
        if not self.on_gpu:
            return self.session.run(self.outputs, {self.input_name: arr})
        self._input.upload(np.ascontiguousarray(arr[0], dtype=np.float32))  # same size and type: same buffer
        if self._input.cudaPtr() != self._bound_ptr:
            raise RuntimeError('upload moved the bound GPU input buffer')
        return self._run()

    def _run(self) -> List[np.ndarray]:
        # OpenCV's crop and upload run on CUDA's default stream; ONNX Runtime runs on its own stream, which doesn't
        # wait for it, and an upload from ordinary CPU memory can return before the copy lands. Without this wait
        # about 1 run in 25 read a stale input (test_gpu_model.py).
        self._opencv_stream.waitForCompletion()
        self.session.run_with_iobinding(self._binding)
        return [v.numpy() for v in self._out]
