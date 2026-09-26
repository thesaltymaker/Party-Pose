import sys
import json
import time
from pathlib import Path
from typing import List, Optional
import cv2
from src.config import parse_args
from src.model_manager import ModelManager
from src.video_capture import VideoCaptureModule
from src.person_detector import PersonDetector, PersonDetections
from src.face_processor import FaceProcessor
from src.hand_processor import HandProcessor
from src.body_processor import BodyProcessor
from src.renderer import Renderer
from src.fps_counter import FPSCounter
from src.types import BoundingBox, FaceResult, HandResult, BodyResult


def _find_head_for_body(body: BoundingBox, head_boxes: List[BoundingBox]) -> Optional[BoundingBox]:
    """Return the most confident head box whose centroid falls inside the body bbox, or None."""
    inside = [
        h for h in head_boxes
        if body.x <= h.x + h.w / 2 <= body.x + body.w
        and body.y <= h.y + h.h / 2 <= body.y + body.h
    ]
    return max(inside, key=lambda h: h.confidence) if inside else None


def _find_hands_for_body(body: BoundingBox, hand_boxes: List[BoundingBox]) -> List[BoundingBox]:
    """Return hand boxes whose centroid falls within the body bbox (with lateral margin)."""
    margin_x = body.w * 0.5
    margin_y = body.h * 0.15
    return [
        h for h in hand_boxes
        if (body.x - margin_x <= h.x + h.w / 2 <= body.x + body.w + margin_x
            and body.y - margin_y <= h.y + h.h / 2 <= body.y + body.h + margin_y)
    ]


def _dump_frame(dump_dir: Path, n: int, frame, detections, faces, hands, bodies, cs_x: float, cs_y: float) -> None:
    """Debug: save the frame with detector boxes + scores drawn, and all scores as JSON."""
    def rec(b):
        return {'score': round(b.confidence, 3), 'bbox': [round(v) for v in (b.x, b.y, b.w, b.h)]}

    boxes = {'body': detections.body_boxes, 'head': detections.head_boxes, 'hand': detections.hand_boxes} if detections else {}
    data = {cls: [rec(b) for b in bs] for cls, bs in boxes.items()}
    data['face_results'] = [{'person_id': f.person_id, 'presence': round(f.presence, 3), 'bbox': rec(f.bbox)['bbox']} for f in faces]
    data['hand_results'] = [{'person_id': h.person_id, 'presence': round(h.presence, 3), 'bbox': rec(h.bbox)['bbox']} for h in hands]
    data['body_results'] = [{'person_id': b.person_id, 'presence': round(b.presence, 3)} for b in bodies]
    (dump_dir / f'frame_{n:04d}.json').write_text(json.dumps(data, indent=1))

    out = frame.copy()
    colors = {'body': (0, 0, 255), 'head': (0, 255, 0), 'hand': (255, 128, 0)}
    for cls, bs in boxes.items():
        for b in bs:
            p1 = (int(b.x * cs_x), int(b.y * cs_y))
            p2 = (int((b.x + b.w) * cs_x), int((b.y + b.h) * cs_y))
            cv2.rectangle(out, p1, p2, colors[cls], 2)
            cv2.putText(out, f'{cls} {b.confidence:.2f}', (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)
    cv2.imwrite(str(dump_dir / f'frame_{n:04d}.jpg'), out)


def main():
    config = parse_args()
    models_dir = Path(__file__).parent / 'models'
    model_manager = ModelManager(models_dir, platform=config.platform)

    any_modality = config.face or config.hands or config.body
    required_models = []
    if any_modality:
        required_models.append('person_detector')
    if config.face:
        required_models.append('face_landmarks')
        if config.face_expressions:
            required_models.append('face_blendshapes')
    if config.hands:
        required_models.append('hand_landmarks')
    if config.body:
        required_models.append('pose_landmarks')

    try:
        model_manager.validate(required_models)
    except FileNotFoundError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    try:
        capture = VideoCaptureModule(config)
    except RuntimeError as e:
        print(f'Error opening camera: {e}', file=sys.stderr)
        sys.exit(1)

    person_proc = PersonDetector(model_manager) if any_modality else None
    face_proc   = FaceProcessor(model_manager, config.face_expressions) if config.face else None
    hand_proc   = HandProcessor(model_manager) if config.hands else None
    body_proc   = BodyProcessor(model_manager) if config.body else None

    renderer    = Renderer(show_roi=config.show_roi)
    fps_counter = FPSCounter()

    display_w = config.width  if config.width  > 0 else capture.width
    display_h = config.height if config.height > 0 else capture.height
    scale_display = (display_w != capture.width or display_h != capture.height)

    # Accumulated ms per stage (capture, person, face, hands, body, render); printed as 60-frame averages with --fps.
    # Counts track extra work (false/extra detections) since face/body models run once per detected body.
    stage_ms = {k: 0.0 for k in ('capture', 'person', 'face', 'hands', 'body', 'render')}
    counts = {k: 0 for k in ('bodies', 'heads', 'hands', 'faces_run', 'hands_run')}

    dump_dir = Path(config.dump_detections) if config.dump_detections else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    dump_n, last_dump = 0, 0.0

    try:
        while True:
            t = time.perf_counter()
            frame_gpu = capture.read_frame()
            frame_w, frame_h = frame_gpu.size()
            stage_ms['capture'] += (time.perf_counter() - t) * 1000

            all_face_results: List[FaceResult] = []
            all_hand_results: List[HandResult] = []
            all_body_results: List[BodyResult] = []
            detections = None

            if person_proc:
                t = time.perf_counter()
                detections = person_proc.process(frame_gpu, frame_w, frame_h)
                stage_ms['person'] += (time.perf_counter() - t) * 1000
                counts['bodies'] += len(detections.body_boxes)
                counts['heads'] += len(detections.head_boxes)
                counts['hands'] += len(detections.hand_boxes)

                for person_id, body_bbox in enumerate(detections.body_boxes):
                    head_bbox    = _find_head_for_body(body_bbox, detections.head_boxes)
                    person_hands = _find_hands_for_body(body_bbox, detections.hand_boxes)

                    if face_proc and head_bbox is not None:
                        t = time.perf_counter()
                        face = face_proc.process(frame_gpu, head_bbox, frame_w, frame_h, config.mirror)
                        stage_ms['face'] += (time.perf_counter() - t) * 1000
                        counts['faces_run'] += 1
                        if face is not None:
                            face.person_id = person_id
                            all_face_results.append(face)

                    if hand_proc and person_hands:
                        t = time.perf_counter()
                        for hand in hand_proc.process(frame_gpu, person_hands, frame_w, frame_h, config.mirror):
                            hand.person_id = person_id
                            all_hand_results.append(hand)
                        stage_ms['hands'] += (time.perf_counter() - t) * 1000
                        counts['hands_run'] += len(person_hands)

                    if body_proc:
                        t = time.perf_counter()
                        body = body_proc.process(frame_gpu, body_bbox, frame_w, frame_h, config.mirror)
                        stage_ms['body'] += (time.perf_counter() - t) * 1000
                        if body is not None:
                            body.person_id = person_id
                            all_body_results.append(body)

            # Timing for the final stage: GPU download, drawing, imshow, and waitKey.
            t_render = time.perf_counter()
            # Single GPU→CPU download for all drawing
            cpu_frame = frame_gpu.download()
            if config.black_bg:
                cpu_frame[:] = 0

            # Scale frame to display size before drawing so lines/points are native resolution
            if scale_display:
                cpu_frame = cv2.resize(cpu_frame, (display_w, display_h))
                cs_x = display_w / frame_w
                cs_y = display_h / frame_h
            else:
                cs_x = cs_y = 1.0

            dump_now = dump_dir and time.monotonic() - last_dump >= config.dump_every
            if dump_now:
                cv2.imwrite(str(dump_dir / f'frame_{dump_n + 1:04d}_raw.jpg'), cpu_frame)

            renderer.draw_faces(cpu_frame, all_face_results, display_w, display_h, cs_x, cs_y)
            renderer.draw_hands(cpu_frame, all_hand_results, display_w, display_h, cs_x, cs_y)
            renderer.draw_body(cpu_frame, all_body_results, display_w, display_h, cs_x, cs_y)

            if dump_now:
                dump_n += 1
                last_dump = time.monotonic()
                _dump_frame(dump_dir, dump_n, cpu_frame, detections, all_face_results, all_hand_results, all_body_results, cs_x, cs_y)

            fps_counter.tick()
            if config.show_fps:
                renderer.draw_fps(cpu_frame, fps_counter.get_fps())
                fps_counter._print_counter = getattr(fps_counter, "_print_counter", 0) + 1
                if fps_counter._print_counter % 60 == 0:
                    print(f"[FPS] {fps_counter.get_fps():.1f}", flush=True)

            cv2.imshow('Poser', cpu_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            stage_ms['render'] += (time.perf_counter() - t_render) * 1000

            # Log 60-frame averages and execution provider info (TensorRT vs CUDA fallback), then reset accumulators.
            if config.show_fps and fps_counter._print_counter % 60 == 0:
                n = 60
                print('[STAGES ms/frame] ' + ' '.join(f'{k}={v / n:.1f}' for k, v in stage_ms.items())
                      + ' | per frame: ' + ' '.join(f'{k}={v / n:.2f}' for k, v in counts.items()), flush=True)
                if fps_counter._print_counter == 60:
                    for name, sess in model_manager._sessions.items():
                        print(f'[PROVIDERS] {name}: {sess.get_providers()}', flush=True)
                stage_ms = dict.fromkeys(stage_ms, 0.0)
                counts = dict.fromkeys(counts, 0)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f'Runtime error: {e}', file=sys.stderr)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
