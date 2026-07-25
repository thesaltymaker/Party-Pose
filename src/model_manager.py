import sys
from pathlib import Path
from typing import Dict, List
import onnxruntime as ort

class ModelManager:
    person_detector = "person_detector"
    face_landmarks = "face_landmarks"
    face_blendshapes = "face_blendshapes"
    hand_landmarks = "hand_landmarks"
    pose_landmarks = "pose_landmarks"

    _MODEL_FILES = {
        person_detector: "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.onnx",
        face_landmarks: "face_landmarks_detector_opset15.onnx",
        face_blendshapes: "face_blendshapes_opset15.onnx",
        hand_landmarks: "hand_landmarks_detector_opset15.onnx",
        pose_landmarks: "pose_landmarks_detector_opset15.onnx",
    }

    _TRT_MODEL_FILE_OVERRIDES = {
        person_detector: "yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.shapeinferred.onnx",
    }

    # person_detector's baked-in NonMaxSuppression op hangs (300s+) during TensorRT
    # engine build on the Orin. Skip TensorRT for it, use CUDA/CPU fallback instead.
    _TRT_EXCLUDED_MODELS = {person_detector}

    def __init__(self, models_dir: Path, platform: str = "laptop") -> None:
        self.models_dir = models_dir
        self.platform = platform
        self._sessions: Dict[str, ort.InferenceSession] = {}

    def validate(self, required: List[str]) -> None:
        missing_files = []
        for name in required:
            if name not in self._MODEL_FILES:
                continue
            model_path = self.models_dir / self._MODEL_FILES[name]
            if not model_path.exists():
                missing_files.append(str(model_path))
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")

    def get_session(self, name: str) -> ort.InferenceSession:
        if name in self._sessions:
            return self._sessions[name]

        model_path = (self.models_dir / self._TRT_MODEL_FILE_OVERRIDES[name]) \
                     if self.platform == "orin" and name in self._TRT_MODEL_FILE_OVERRIDES \
                     else self.models_dir / self._MODEL_FILES[name]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.intra_op_num_threads = 1

        cuda_provider_opts = {
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 ** 3,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
        trt_provider_opts = {
            'trt_max_workspace_size': 2 * 1024 ** 3,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(self.models_dir / 'trt_cache'),
        }

        # Provider selection logic
        if self.platform == "orin" and name not in self._TRT_EXCLUDED_MODELS and 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers = [('TensorrtExecutionProvider', trt_provider_opts)]
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', cuda_provider_opts))
            providers.append('CPUExecutionProvider')
        else:
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = [('CUDAExecutionProvider', cuda_provider_opts), 'CPUExecutionProvider']
            else:
                print(f"Warning: CUDAExecutionProvider not available, using CPUExecutionProvider only", file=sys.stderr)

        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
        self._sessions[name] = session
        return session
