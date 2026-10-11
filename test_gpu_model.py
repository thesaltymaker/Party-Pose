"""GpuModel must give the same outputs as cropping to numpy and calling session.run (uses the real models)."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from src.config import detect_platform
from src.gpu_model import gpu_model
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
