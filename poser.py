import sys
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
    """Return the head box whose centroid falls inside the body bbox, or the nearest one."""
    if not head_boxes:
        return None
    inside = [
        h for h in head_boxes
        if body.x <= h.x + h.w / 2 <= body.x + body.w
        and body.y <= h.y + h.h / 2 <= body.y + body.h
    ]
    pool = inside if inside else head_boxes
    return max(pool, key=lambda h: h.confidence)


def _find_hands_for_body(body: BoundingBox, hand_boxes: List[BoundingBox]) -> List[BoundingBox]:
    """Return hand boxes whose centroid falls within the body bbox (with lateral margin)."""
    margin_x = body.w * 0.5
    margin_y = body.h * 0.15
    return [
        h for h in hand_boxes
        if (body.x - margin_x <= h.x + h.w / 2 <= body.x + body.w + margin_x
            and body.y - margin_y <= h.y + h.h / 2 <= body.y + body.h + margin_y)
    ]


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

    try:
        while True:
            frame_gpu = capture.read_frame()
            frame_w, frame_h = frame_gpu.size()

            all_face_results: List[FaceResult] = []
            all_hand_results: List[HandResult] = []
            all_body_results: List[BodyResult] = []

            if person_proc:
                detections = person_proc.process(frame_gpu, frame_w, frame_h)

                for person_id, body_bbox in enumerate(detections.body_boxes):
                    head_bbox    = _find_head_for_body(body_bbox, detections.head_boxes)
                    person_hands = _find_hands_for_body(body_bbox, detections.hand_boxes)

                    if face_proc and head_bbox is not None:
                        face = face_proc.process(frame_gpu, head_bbox, frame_w, frame_h, config.mirror)
                        if face is not None:
                            face.person_id = person_id
                            all_face_results.append(face)

                    if hand_proc and person_hands:
                        for hand in hand_proc.process(frame_gpu, person_hands, frame_w, frame_h, config.mirror):
                            hand.person_id = person_id
                            all_hand_results.append(hand)

                    if body_proc:
                        body = body_proc.process(frame_gpu, body_bbox, frame_w, frame_h, config.mirror)
                        if body is not None:
                            body.person_id = person_id
                            all_body_results.append(body)

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

            renderer.draw_faces(cpu_frame, all_face_results, display_w, display_h, cs_x, cs_y)
            renderer.draw_hands(cpu_frame, all_hand_results, display_w, display_h, cs_x, cs_y)
            renderer.draw_body(cpu_frame, all_body_results, display_w, display_h, cs_x, cs_y)

            fps_counter.tick()
            if config.show_fps:
                renderer.draw_fps(cpu_frame, fps_counter.get_fps())

            cv2.imshow('Poser', cpu_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f'Runtime error: {e}', file=sys.stderr)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
