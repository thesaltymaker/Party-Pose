import json
import subprocess
import sys
from pathlib import Path

MODEL_NAMES = ["person_detector", "face_landmarks", "face_blendshapes", "hand_landmarks", "pose_landmarks"]


def run_model_test_with_timeout(name: str, timeout_seconds: int = 300) -> dict:
    """Run model test with timeout and parse result."""
    worker = str(Path(__file__).parent / "check_tensorrt_compat_worker.py")
    try:
        completed = subprocess.run([sys.executable, worker, name],
                                   capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "ERROR",
            "error": f"timed out after {timeout_seconds}s",
            "trt_cold_seconds": None,
            "trt_warm_seconds": None,
            "cuda_output_shapes": None,
            "trt_output_shapes": None,
            "shapes_match": False
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
            "shapes_match": False
        }

    lines = completed.stdout.splitlines()
    last_line = ""
    for line in reversed(lines):
        if line.strip():
            last_line = line
            break

    try:
        result = json.loads(last_line)
    except Exception as e:
        error_msg = f"Failed to parse JSON: {str(e)}. Last line: {last_line}"
        return {
            "name": name,
            "status": "ERROR",
            "error": error_msg + "\n" + completed.stderr[-2000:],
            "trt_cold_seconds": None,
            "trt_warm_seconds": None,
            "cuda_output_shapes": None,
            "trt_output_shapes": None,
            "shapes_match": False
        }

    result.setdefault("name", name)
    result.setdefault("status", "ERROR")
    result.setdefault("error", None)
    for key, default in {
        "trt_cold_seconds": None,
        "trt_warm_seconds": None,
        "cuda_output_shapes": None,
        "trt_output_shapes": None,
        "shapes_match": False
    }.items():
        result.setdefault(key, default)

    return result


def main():
    """Generate Orin TensorRT compatibility report"""
    results = [run_model_test_with_timeout(name) for name in MODEL_NAMES]
    for r in results:
        cold = f"{r['trt_cold_seconds']:.3f}" if r.get('trt_cold_seconds') is not None else "n/a"
        warm = f"{r['trt_warm_seconds']:.3f}" if r.get('trt_warm_seconds') is not None else "n/a"
        print(f"{r['name']:20s} {r['status']:6s} cold={cold}  warm={warm}  match={r['shapes_match']}")
    header = "# Orin TensorRT compatibility report\n"
    table_header = "| Model | Status | Cold (s) | Warm (s) | Shapes Match | Error |\n"
    separator = "|---|---|---|---|---|---|\n"
    rows = []
    for r in results:
        model = r['name']
        status = r['status']
        cold = f"{r['trt_cold_seconds']:.3f}" if r.get('trt_cold_seconds') is not None else "n/a"
        warm = f"{r['trt_warm_seconds']:.3f}" if r.get('trt_warm_seconds') is not None else "n/a"
        error = (r.get('error') or '').replace("\n", " ")
        rows.append(f"| {model} | {status} | {cold} | {warm} | {r['shapes_match']} | {error} |\n")
    markdown = header + table_header + separator + ''.join(rows)
    out_path = Path(__file__).parent / 'docs' / 'orin-tensorrt-compatibility.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)


if __name__ == '__main__':
    main()
