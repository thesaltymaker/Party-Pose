"""GpuModel must give the same outputs as cropping to numpy and calling session.run (uses the real models)."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from src.config import detect_platform
from src.face_processor import FaceProcessor
from src.gpu_model import gpu_model
from src.person_detector import PersonDetector
from src.types import BoundingBox
from src.model_manager import ModelManager
from src.preprocessor import Preprocessor

MODELS = Path(__file__).parent / 'models'
pytestmark = pytest.mark.skipif(
    not (MODELS / 'hand_landmarks_detector_opset15.onnx').exists()
    or 'CUDAExecutionProvider' not in ort.get_available_providers()
    or cv2.cuda.getCudaEnabledDeviceCount() == 0,
    reason='needs the model files and CUDA')


@pytest.fixture(scope='module')
def mm():
    return ModelManager(MODELS, platform=detect_platform())


@pytest.fixture(scope='module')
def frame():
    g = cv2.cuda_GpuMat()
    g.upload(np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8))
    return g


def _plain(mm, model, outs, arr):
    """The same model run the old way (numpy in, session.run). A CUDA-graph session only runs through I/O binding,
    so this is a separate session: on the Orin TensorRT FP16 with the same cached engine, so outputs must match
    exactly (an FP32 reference differs by up to 15 model pixels on noise input); elsewhere plain CUDA."""
    if 'TensorrtExecutionProvider' in mm.get_session(model).get_providers():
        providers = [('TensorrtExecutionProvider', {
            'trt_max_workspace_size': 2 * 1024 ** 3, 'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(mm.models_dir / 'trt_cache')}), 'CUDAExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(str(mm.models_dir / mm._MODEL_FILES[model]),
                                providers=providers).run(outs, {'input_1': arr})


@pytest.mark.parametrize('model,size,outs', [
    ('hand_landmarks', 224, ['Identity', 'Identity_1', 'Identity_2']),
    ('pose_landmarks', 256, ['Identity', 'Identity_1']),
])
def test_gpu_crop_gives_the_same_outputs(mm, frame, model, size, outs):
    gm = gpu_model(mm, model, 'input_1', outs, size)
    for cx, angle in ((900, 0.3), (300, -1.2), (1800, 2.5)):  # the last one partly outside the frame
        M = Preprocessor.roi_affine(cx, 500, 400, angle, size)
        ref = _plain(mm, model, outs, Preprocessor.crop_rotated_nhwc(frame, M, size).copy())
        got = gm.run(frame, M)
        for r, g in zip(ref, got):
            assert g.shape == r.shape
            assert np.allclose(g, r, atol=1e-3), (model, cx, float(np.abs(g - r).max()))


def test_cpu_array_input_gives_the_same_outputs(mm):
    gm = gpu_model(mm, 'pose_landmarks', 'input_1', ['Identity', 'Identity_1'], 256)
    arr = np.random.default_rng(1).random((1, 256, 256, 3), dtype=np.float32)
    ref = _plain(mm, 'pose_landmarks', ['Identity', 'Identity_1'], arr)
    got = gm.run_array(arr)
    assert np.allclose(got[0], ref[0], atol=1e-3)


def test_input_buffer_stays_at_one_gpu_address(mm, frame):
    # CUDA graphs replay with the addresses they were recorded with.
    gm = gpu_model(mm, 'hand_landmarks', 'input_1', ['Identity', 'Identity_1', 'Identity_2'], 224)
    ptr = gm.input_ptr()
    gm.run(frame, Preprocessor.roi_affine(900, 500, 300, 0.0, 224))
    gm.run_array(np.zeros((1, 224, 224, 3), np.float32))
    assert gm.input_ptr() == ptr


def test_one_gpu_model_per_model(mm):
    # A CUDA graph replays the addresses of the first run, so a model must not get a second set of buffers.
    a = gpu_model(mm, 'pose_landmarks', 'input_1', ['Identity', 'Identity_1'], 256)
    assert gpu_model(mm, 'pose_landmarks', 'input_1', ['Identity', 'Identity_1'], 256) is a


def _reference_session(mm, model):
    """The model's own engine run the old way (see _plain)."""
    if 'TensorrtExecutionProvider' in mm.get_session(model).get_providers():
        opts = {'trt_max_workspace_size': 2 * 1024 ** 3, 'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(mm.models_dir / 'trt_cache'), **mm._TRT_EXTRA_OPTS.get(model, {})}
        opts.pop('trt_cuda_graph_enable', None)
        providers = [('TensorrtExecutionProvider', opts), 'CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    path = mm.models_dir / (mm._TRT_MODEL_FILE_OVERRIDES.get(model) if mm.platform == 'orin' and model in
                            mm._TRT_MODEL_FILE_OVERRIDES else mm._MODEL_FILES[model])
    return ort.InferenceSession(str(path), providers=providers)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_faces_on_the_gpu_match_the_numpy_path(mm, frame, n):
    heads = [BoundingBox(600, 200, 140, 170, 0.9), BoundingBox(1200, 260, 120, 150, 0.8),
             BoundingBox(1840, 1000, 140, 160, 0.7)][:n]  # the third one runs past the frame's corner
    fp = FaceProcessor(mm)
    fp.process_many(frame, heads, 1920, 1080, False)
    got = fp.last_logits
    ref_sess = _reference_session(mm, 'face_landmarks')
    crops = [Preprocessor.crop_roi_nhwc(frame, h, 256, 256, pad_fraction=0.0, frame_w=1920, frame_h=1080)[0]
             for h in heads]
    ref = [float(ref_sess.run(['Identity_1'], {'input_12': np.concatenate([c, c])})[0].reshape(-1)[0]) for c in crops]
    # Orin: the same TensorRT engine, so the same scores. Laptop (CUDA): the batch's layout changes the
    # convolution algorithm, which moves results a little (up to ~0.2).
    tol = 1e-3 if 'TensorrtExecutionProvider' in ref_sess.get_providers() else 0.5
    assert np.allclose(got, ref, atol=tol), (got, ref)


def test_detector_input_on_the_gpu_matches_the_numpy_path(mm, frame):
    det = PersonDetector(mm)
    det.process(frame, 1920, 1080)
    got = det._gpu.run(frame, 1920, 1080)
    arr = Preprocessor.full_frame_nhwc(frame, det.INPUT_W, det.INPUT_H, normalize=False)
    ref = _reference_session(mm, 'person_detector').run(['batchno_classid_score_x1y1x2y2'], {'input': arr})[0]
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-3), float(np.abs(got - ref).max())


def test_detector_bgr_input_on_the_gpu_matches_the_numpy_path(mm, frame):
    det = PersonDetector(mm, bgr=True)
    det.process(frame, 1920, 1080)
    got = det._gpu.run(frame, 1920, 1080)
    arr = Preprocessor.full_frame_nhwc(frame, det.INPUT_W, det.INPUT_H, normalize=False)[..., ::-1]
    ref = _reference_session(mm, 'person_detector').run(['batchno_classid_score_x1y1x2y2'],
                                                        {'input': np.ascontiguousarray(arr)})[0]
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-3), float(np.abs(got - ref).max())
