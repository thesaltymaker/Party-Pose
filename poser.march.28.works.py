import sys
from pathlib import Path
import cv2
from src.config import parse_args
from src.model_manager import ModelManager
from src.video_capture import VideoCaptureModule
from src.face_processor import FaceProcessor
from src.hand_processor import HandProcessor
from src.body_processor import BodyProcessor
from src.renderer import Renderer
from src.fps_counter import FPSCounter

def main():
    config = parse_args()
    models_dir = Path(__file__).parent / 'models'
    model_manager = ModelManager(models_dir)

    required_models = []
    if config.face:
        required_models.extend(['face_detector', 'face_landmarks'])
        if config.face_expressions:
            required_models.append('face_blendshapes')
    if config.hands:
        required_models.extend(['hand_detector', 'hand_landmarks'])
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

    face_proc = FaceProcessor(model_manager, config.face_expressions) if config.face else None
    hand_proc = HandProcessor(model_manager) if config.hands else None
    body_proc = BodyProcessor(model_manager) if config.body else None

    renderer = Renderer(show_roi=config.show_roi)
    fps_counter = FPSCounter()

    try:
        while True:
            frame_gpu = capture.read_frame()
            frame_w, frame_h = frame_gpu.size()  # actual dims from GpuMat (cols, rows)

            face_results = face_proc.process(frame_gpu, frame_w, frame_h, config.mirror) if face_proc else []
            hand_results = hand_proc.process(frame_gpu, frame_w, frame_h, config.mirror) if hand_proc else []
            body_result  = body_proc.process(frame_gpu, frame_w, frame_h, config.mirror) if body_proc else None

            # Single GPU→CPU download for all drawing
            cpu_frame = frame_gpu.download()

            renderer.draw_faces(cpu_frame, face_results, frame_w, frame_h)
            renderer.draw_hands(cpu_frame, hand_results, frame_w, frame_h)
            renderer.draw_body(cpu_frame, body_result, frame_w, frame_h)

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
