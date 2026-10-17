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


def gpu_face_batch(model_manager) -> 'GpuFaceBatch':
    """The one GpuFaceBatch (created on first use)."""
    models = model_manager.__dict__.setdefault('_gpu_models', {})
    if 'face_landmarks' not in models:
        models['face_landmarks'] = GpuFaceBatch(model_manager)
    return models['face_landmarks']


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


def gpu_batch(model_manager, model: str, input_name: str, outputs: List[str], size: int, slots: int) -> 'GpuBatch':
    """The one GpuBatch for this model (created on first use)."""
    models = model_manager.__dict__.setdefault('_gpu_models', {})
    if model + '_batch' not in models:
        models[model + '_batch'] = GpuBatch(model_manager, model, input_name, outputs, size, slots)
    return models[model + '_batch']


class GpuBatch:
    """GpuModel for up to `slots` rotated crops in one run (--pose-batch, --hand-batch), on the model's
    .dynbatch.onnx copy (tools/orin_bench/batch_model.py). Like GpuFaceBatch it always runs the full batch, so
    the Orin's TensorRT session replays one CUDA graph; unused slots keep their last crop and are ignored.
    More crops than slots run `slots` at a time."""

    def __init__(self, model_manager, model: str, input_name: str, outputs: List[str], size: int, slots: int) -> None:
        self.session = model_manager.get_batch_session(model, slots)
        self.input_name, self.outputs, self.size, self.slots = input_name, outputs, size, slots
        self.on_gpu = any(p in self.session.get_providers()
                          for p in ('TensorrtExecutionProvider', 'CUDAExecutionProvider'))
        if not self.on_gpu:
            return
        self._stream = cv2.cuda.Stream.Null()
        self._warp = [cv2.cuda_GpuMat(size, size, cv2.CV_8UC3) for _ in range(slots)]
        self._rgb = [cv2.cuda_GpuMat(size, size, cv2.CV_8UC3) for _ in range(slots)]
        self._input = cv2.cuda.createContinuous(slots * size, size, cv2.CV_32FC3, cv2.cuda_GpuMat())
        self._binding = self.session.io_binding()
        self._binding.bind_input(input_name, 'cuda', 0, np.float32, [slots, size, size, 3], self._input.cudaPtr())
        meta = {o.name: o.shape for o in self.session.get_outputs()}
        self._out = []
        for name in outputs:
            v = ort.OrtValue.ortvalue_from_shape_and_type([slots] + [d if isinstance(d, int) else 1 for d in meta[name][1:]],
                                                          np.float32, 'cuda', 0)
            self._binding.bind_ortvalue_output(name, v)
            self._out.append(v)

    def run(self, frame_gpu: cv2.cuda_GpuMat, Ms: List[np.ndarray]) -> List[np.ndarray]:
        """The model's outputs [len(Ms), ...] for the rotated crops given by Preprocessor.roi_affine."""
        outs = []
        for i in range(0, len(Ms), self.slots):
            outs.append(self._run(frame_gpu, Ms[i:i + self.slots]))
        return [np.concatenate(o) for o in zip(*outs)] if outs else []

    def _run(self, frame_gpu, Ms) -> List[np.ndarray]:
        n, size = len(Ms), self.size
        if not self.on_gpu:
            batch = np.zeros((self.slots, size, size, 3), np.float32)
            for k, M in enumerate(Ms):
                batch[k] = Preprocessor.crop_rotated_nhwc(frame_gpu, M, size)[0]
            return [o[:n] for o in self.session.run(self.outputs, {self.input_name: batch})]
        for k, M in enumerate(Ms):
            # Use the returned GpuMats: on the Orin's OpenCV build dst is not always written in place.
            self._warp[k] = cv2.cuda.warpAffine(frame_gpu, cv2.invertAffineTransform(M), (size, size), self._warp[k],
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self._rgb[k] = cv2.cuda.cvtColor(self._warp[k], cv2.COLOR_BGR2RGB, self._rgb[k])
            slot = self._input.rowRange(k * size, (k + 1) * size)
            if self._rgb[k].convertTo(cv2.CV_32FC3, slot, 1 / 255.0, 0.0).cudaPtr() != slot.cudaPtr():
                raise RuntimeError('crop was not written into the bound GPU batch buffer')
        self._stream.waitForCompletion()  # ONNX Runtime's stream doesn't wait for OpenCV's (see GpuModel._run)
        self.session.run_with_iobinding(self._binding)
        return [v.numpy()[:n] for v in self._out]


class GpuFaceBatch:
    """The face model on up to SLOTS head crops per run, cut, resized, RGB- and float-converted on the GPU into one
    bound batch buffer (the same pixels as Preprocessor.crop_roi_nhwc with pad_fraction=0).

    It always runs at batch SLOTS, so the Orin's TensorRT session replays one CUDA graph: a second graph for
    another batch size in the same session gave wrong outputs (off by 215 model pixels, no error). With one face
    the second slot keeps its last crop and its output is ignored; more faces run SLOTS at a time.
    Orin, two faces: 6.40 ms -> 2.87 ms, same outputs.
    """

    SLOTS = 2
    SIZE = 256

    def __init__(self, model_manager) -> None:
        self.session = model_manager.get_session('face_landmarks')
        self.on_gpu = any(p in self.session.get_providers()
                          for p in ('TensorrtExecutionProvider', 'CUDAExecutionProvider'))
        if not self.on_gpu:
            return
        n, size = self.SLOTS, self.SIZE
        self._stream = cv2.cuda.Stream.Null()
        self._small = [cv2.cuda_GpuMat(size, size, cv2.CV_8UC3) for _ in range(n)]
        self._rgb = [cv2.cuda_GpuMat(size, size, cv2.CV_8UC3) for _ in range(n)]
        self._input = cv2.cuda.createContinuous(n * size, size, cv2.CV_32FC3, cv2.cuda_GpuMat())
        self._binding = self.session.io_binding()
        self._binding.bind_input('input_12', 'cuda', 0, np.float32, [n, size, size, 3], self._input.cudaPtr())
        meta = {o.name: o.shape for o in self.session.get_outputs()}
        self._out = []
        for name in ('Identity', 'Identity_1'):
            v = ort.OrtValue.ortvalue_from_shape_and_type([n] + [d if isinstance(d, int) else 1 for d in meta[name][1:]],
                                                          np.float32, 'cuda', 0)
            self._binding.bind_ortvalue_output(name, v)
            self._out.append(v)

    def run(self, frame_gpu: cv2.cuda_GpuMat, rects) -> tuple:
        """rects: up to SLOTS (x, y, w, h) frame rectangles from Preprocessor.crop_rect. Returns
        (landmarks [n, 478, 3] in model pixels, face logits [n])."""
        size = self.SIZE
        for k, (x, y, w, h) in enumerate(rects):
            roi = cv2.cuda_GpuMat(frame_gpu, (int(x), int(y), int(w), int(h)))
            # Use the returned GpuMats: on the Orin's OpenCV build dst is not always written in place.
            self._small[k] = cv2.cuda.resize(roi, (size, size), self._small[k])
            self._rgb[k] = cv2.cuda.cvtColor(self._small[k], cv2.COLOR_BGR2RGB, self._rgb[k])
            slot = self._input.rowRange(k * size, (k + 1) * size)
            if self._rgb[k].convertTo(cv2.CV_32FC3, slot, 1 / 255.0, 0.0).cudaPtr() != slot.cudaPtr():
                raise RuntimeError('face crop was not written into the bound GPU batch buffer')
        self._stream.waitForCompletion()  # ONNX Runtime's stream doesn't wait for OpenCV's (see GpuModel._run)
        self.session.run_with_iobinding(self._binding)
        n = len(rects)
        return self._out[0].numpy().reshape(self.SLOTS, 478, 3)[:n], self._out[1].numpy().reshape(-1)[:n]


class GpuDetectorInput:
    """The person detector's input prepared on the GPU (shrink, RGB, letterbox padding, float 0-255; the same
    values as Preprocessor.full_frame_nhwc(normalize=False)) into a bound buffer. Its output has a different
    number of rows every frame, so it goes to the CPU; part of the model runs outside TensorRT, so no CUDA graph.
    Orin: 5.38 -> 4.36 ms, same output.
    """

    def __init__(self, session, input_name: str, output_name: str, width: int, height: int, bgr: bool = False) -> None:
        self.session, self.output_name, self.width, self.height = session, output_name, width, height
        self.bgr = bgr  # keep the frame's BGR order instead of converting to RGB
        self.on_gpu = any(p in session.get_providers() for p in ('TensorrtExecutionProvider', 'CUDAExecutionProvider'))
        if not self.on_gpu:
            return
        self._stream = cv2.cuda.Stream.Null()
        self._small = self._rgb = self._padded = None
        self._input = cv2.cuda.createContinuous(height, width, cv2.CV_32FC3, cv2.cuda_GpuMat())
        self._binding = session.io_binding()
        self._binding.bind_input(input_name, 'cuda', 0, np.float32, [1, height, width, 3], self._input.cudaPtr())
        self._binding.bind_output(output_name, 'cpu')

    def run(self, frame_gpu: cv2.cuda_GpuMat, frame_w: int, frame_h: int) -> np.ndarray:
        scale, pad_x, pad_y = Preprocessor.letterbox_params(frame_w, frame_h, self.width, self.height)
        new_w, new_h = int(round(frame_w * scale)), int(round(frame_h * scale))
        if self._small is None or self._small.size() != (new_w, new_h):
            self._small = cv2.cuda_GpuMat(new_h, new_w, cv2.CV_8UC3)
            self._rgb = cv2.cuda_GpuMat(new_h, new_w, cv2.CV_8UC3)
            self._padded = cv2.cuda_GpuMat(self.height, self.width, cv2.CV_8UC3)
        # Use the returned GpuMats: on the Orin's OpenCV build dst is not always written in place.
        self._small = cv2.cuda.resize(frame_gpu, (new_w, new_h), self._small)
        self._rgb = self._small if self.bgr else cv2.cuda.cvtColor(self._small, cv2.COLOR_BGR2RGB, self._rgb)
        self._padded = cv2.cuda.copyMakeBorder(self._rgb, pad_y, self.height - new_h - pad_y, pad_x,
                                               self.width - new_w - pad_x, cv2.BORDER_CONSTANT, self._padded, (0, 0, 0))
        if self._padded.convertTo(cv2.CV_32FC3, self._input, 1.0, 0.0).cudaPtr() != self._input.cudaPtr():
            raise RuntimeError('detector input was not written into the bound GPU buffer')
        self._stream.waitForCompletion()  # ONNX Runtime's stream doesn't wait for OpenCV's (see GpuModel._run)
        self.session.run_with_iobinding(self._binding)
        return self._binding.copy_outputs_to_cpu()[0]
