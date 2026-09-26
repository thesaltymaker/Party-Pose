"""
Party-Pose diagnostic micro-benchmarks for the Jetson Orin (issue #2).

Runs pieces of the pipeline in isolation so their cost can be compared with the
per-stage timing that `poser.py --fps` prints from the live app.

Modes (run from anywhere; results and TensorRT caches go to --out-dir):
  grab:      python tools/orin_bench/bench_models.py --out-dir /tmp/ppbench grab
  pre:       python tools/orin_bench/bench_models.py pre
  landmarks: python tools/orin_bench/bench_models.py landmarks
  models:    python tools/orin_bench/bench_models.py models person_detector --provider trt
  post-ops:  python tools/orin_bench/bench_models.py post-ops

`grab` saves a real camera frame; `models person_detector` then also prints the
detections on that frame so different providers can be checked for equal output.
"""

import argparse
import collections
import time
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.model_manager import ModelManager
from src.preprocessor import Preprocessor
from src.types import BoundingBox

def bench(fn, n=100, warm=10):
    """Returns average ms per call."""
    for _ in range(warm):
        fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) * 1000 / n

def cmd_grab(args):
    pipeline = (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Could not open CSI camera.")
        return
    for _ in range(60):
        cap.read()
    ret, frame = cap.read()
    if ret:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_dir / "frame.png"), frame)
        small = cv2.resize(frame, (480, 270))
        cv2.imwrite(str(args.out_dir / "frame_small.jpg"), small)
        print(f"Captured frame: {frame.shape}")
    cap.release()

def cmd_pre(args):
    frame = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    bbox_face = BoundingBox(x=800, y=200, w=200, h=250, confidence=1.0)
    bbox_body = BoundingBox(x=700, y=100, w=500, h=900, confidence=1.0)
    def run_full(): Preprocessor.full_frame_nhwc(gpu_frame, 320, 256, normalize=False)
    def run_face(): Preprocessor.crop_roi_nhwc(gpu_frame, bbox_face, 256, 256, 0.25, 1920, 1080)
    def run_body(): Preprocessor.crop_roi_nhwc(gpu_frame, bbox_body, 256, 256, 0.1, 1920, 1080, letterbox=True)
    print(f"Full frame: {bench(run_full):.2f} ms")
    print(f"Face crop:   {bench(run_face):.2f} ms")
    print(f"Body crop:   {bench(run_body):.2f} ms")

def cmd_landmarks(args):
    shapes = {'face': (478, 3), 'body': (39, 5), 'hand': (21, 3)}
    bbox = BoundingBox(x=700, y=100, w=500, h=900, confidence=1.0)
    for name, shape in shapes.items():
        arr = np.random.rand(*shape).astype(np.float32)
        def run(): Preprocessor.to_image_space(arr, 256, 256, bbox, 1920, 1080)
        print(f"{name:5}: {bench(run, n=200, warm=20):.2f} ms")

def cmd_models(args):
    if args.provider == 'app':
        mm = ModelManager(args.models_dir, platform='orin')

    for name in args.names:
        start_create = time.perf_counter()
        if args.provider == 'app':
            session = mm.get_session(name)
        else:
            model_path = str(args.models_dir / ModelManager._MODEL_FILES[name])
            if args.provider == 'trt':
                model_path = str(args.models_dir / ModelManager._TRT_MODEL_FILE_OVERRIDES.get(name, ModelManager._MODEL_FILES[name]))
                opts = {'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': str(args.out_dir / f'trt_cache_{args.cache_tag}')}
                if not args.no_extra_opts:
                    opts.update(ModelManager._TRT_EXTRA_OPTS.get(name, {}))
                if args.exclude_ops:
                    opts['trt_op_types_to_exclude'] = args.exclude_ops
                if args.min_subgraph:
                    opts['trt_min_subgraph_size'] = args.min_subgraph
                providers = [('TensorrtExecutionProvider', opts), 'CUDAExecutionProvider', 'CPUExecutionProvider']
            elif args.provider == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)

        print(f"[{name}] Created in {time.perf_counter()-start_create:.2f}s. Provider: {session.get_providers()[0]}")
        input_name = session.get_inputs()[0].name
        in_shape = list(session.get_inputs()[0].shape)
        for i in range(len(in_shape)):
            if not isinstance(in_shape[i], int):
                in_shape[i] = 1
        scale = 255.0 if name == 'person_detector' else 1.0
        dummy_in = (np.random.rand(*in_shape).astype(np.float32) * scale)
        def run_inf(): session.run(None, {input_name: dummy_in})
        print(f"  Inference: {bench(run_inf):.2f} ms")

        if name == 'person_detector' and (args.out_dir / "frame.png").exists():
            img = cv2.imread(str(args.out_dir / "frame.png"))
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            arr = Preprocessor.full_frame_nhwc(gpu_img, 320, 256, normalize=False)
            out = session.run(['batchno_classid_score_x1y1x2y2'], {'input': arr})[0]
            print('real-frame detections (class score x1 y1 x2 y2):')
            for row in out:
                if row[2] >= 0.3:
                    print(f"    Det: Class {int(row[1])}, Score {row[2]:.3f}, Box {row[3]:.1f} {row[4]:.1f} {row[5]:.1f} {row[6]:.1f}")

def cmd_post_ops(args):
    """Identifies post-NMS operations that are safe to pass to trt_op_types_to_exclude."""
    import onnx
    import collections

    model_path = args.models_dir / ModelManager._TRT_MODEL_FILE_OVERRIDES['person_detector']
    model = onnx.load(str(model_path))
    graph = model.graph

    # Node-level BFS to find downstream nodes from NMS
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)

    nms_nodes = [n for n in graph.node if 'NonMaxSuppression' in n.op_type]
    stack = []
    for nms in nms_nodes:
        for output in nms.output:
            stack.append(output)

    seen = set()
    downstream_nodes = set()

    while stack:
        curr_out = stack.pop()
        if curr_out in seen:
            continue
        seen.add(curr_out)
        downstream_nodes.add(curr_out)
        for consumer in consumers[curr_out]:
            for out_tensor in consumer.output:
                stack.append(out_tensor)

    # Split nodes into downstream and backbone based on tensor reachability
    downstream = [n for n in graph.node if 'NonMaxSuppression' in n.op_type or any(o in downstream_nodes for o in n.output)]
    backbone = [n for n in graph.node if n not in downstream]

    # Print required counts and set difference
    print(f"Downstream Op Counts: {collections.Counter(n.op_type for n in downstream)}")
    print(f"Backbone Op Counts:   {collections.Counter(n.op_type for n in backbone)}")
    print(f"Only downstream (safe for trt_op_types_to_exclude): {sorted(set(n.op_type for n in downstream) - set(n.op_type for n in backbone))}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ppbench"))
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    subparsers = parser.add_subparsers(dest="mode")

    p_grab = subparsers.add_parser("grab")

    p_pre = subparsers.add_parser("pre")

    subparsers.add_parser("landmarks")

    p_mod = subparsers.add_parser("models")
    p_mod.add_argument("names", nargs="+")
    p_mod.add_argument("--provider", choices=['app', 'cuda', 'cpu', 'trt'], default='app')
    p_mod.add_argument("--no-extra-opts", action="store_true")
    p_mod.add_argument("--exclude-ops", type=str)
    p_mod.add_argument("--min-subgraph", type=int)
    p_mod.add_argument("--cache-tag", type=str, default="bench")

    subparsers.add_parser("post-ops")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "grab": cmd_grab(args)
    elif args.mode == "pre": cmd_pre(args)
    elif args.mode == "landmarks": cmd_landmarks(args)
    elif args.mode == "models": cmd_models(args)
    elif args.mode == "post-ops": cmd_post_ops(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
