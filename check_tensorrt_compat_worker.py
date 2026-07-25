import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.model_manager import ModelManager


def build_synthetic_input(session):
    """Generate synthetic zero-filled float32 inputs for an ONNX Runtime session."""
    return {inp.name: np.zeros([d if isinstance(d, int) and d > 0 else 1 for d in inp.shape], dtype=np.float32) for inp in session.get_inputs()}


def test_one_model(name: str, models_dir) -> dict:
    """Test a single model across TRT and CUDA backends."""
    try:
        trt_manager = ModelManager(models_dir, platform="orin")
        trt_session = trt_manager.get_session(name)
        inputs = build_synthetic_input(trt_session)

        start = time.perf_counter()
        trt_outputs_cold = trt_session.run(None, inputs)
        trt_cold_seconds = time.perf_counter() - start

        start = time.perf_counter()
        trt_outputs_warm = trt_session.run(None, inputs)
        trt_warm_seconds = time.perf_counter() - start

        cuda_manager = ModelManager(models_dir, platform="laptop")
        cuda_session = cuda_manager.get_session(name)
        cuda_outputs = cuda_session.run(None, inputs)

        cuda_output_shapes = [list(o.shape) for o in cuda_outputs]
        trt_output_shapes = [list(o.shape) for o in trt_outputs_warm]
        shapes_match = (cuda_output_shapes == trt_output_shapes)

        return {
            "name": name,
            "status": "PASS" if shapes_match else "FAIL",
            "error": None,
            "trt_cold_seconds": trt_cold_seconds,
            "trt_warm_seconds": trt_warm_seconds,
            "cuda_output_shapes": cuda_output_shapes,
            "trt_output_shapes": trt_output_shapes,
            "shapes_match": shapes_match,
        }
    except Exception as e:
        return {
            "name": name,
            "status": "ERROR",
            "error": str(e),
            "trt_cold_seconds": None,
            "trt_warm_seconds": None,
            "cuda_output_shapes": None,
            "trt_output_shapes": None,
            "shapes_match": False,
        }


if __name__ == '__main__':
    model_name = sys.argv[1]
    models_dir = Path(__file__).parent / 'models'
    result = test_one_model(model_name, models_dir)
    print(json.dumps(result))
