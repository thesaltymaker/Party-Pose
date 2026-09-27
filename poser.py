import sys
import json
import math
import signal
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from src.config import parse_args
from src.model_manager import ModelManager
from src.video_capture import VideoCaptureModule
from src.person_detector import PersonDetector, PersonDetections
from src.face_processor import FaceProcessor
from src.hand_processor import HandProcessor
from src.body_processor import BodyProcessor
from src.renderer import Renderer
from src.fps_counter import FPSCounter
from src.tracker import PersonTracker
from src.track_stats import TrackStats
from src.types import BoundingBox, FaceResult, HandResult, BodyResult


def _head_fit(body: BoundingBox, head: BoundingBox) -> Optional[float]:
    """How well a head box fits a body box as that body's head (lower is better), or None if it can't be.

    The head's centre must be inside the body box horizontally and in its top part (a false-positive box
    around a person has the person's head in its middle), and the head must not be tiny next to the body
    (a box around a whole group).
    """
    hx, hy = head.x + head.w / 2, head.y + head.h / 2
    if not (body.x <= hx <= body.x + body.w):
        return None
    if not (body.y <= hy <= body.y + 0.45 * body.h):
        return None
    if head.w < 0.15 * body.w:
        return None
    return abs(hx - (body.x + body.w / 2)) / body.w + abs(hy - (body.y + 0.1 * body.h)) / body.h


def _assign_heads(bodies: List[BoundingBox], heads: List[BoundingBox]) -> List[Optional[BoundingBox]]:
    """Give each body at most one head and each head to at most one body (issue #6).

    Bodies are served in the order given (the app passes the oldest track first, so a person's head stays
    with the same track and colour); each takes the best-fitting free head.
    """
    out: List[Optional[BoundingBox]] = [None] * len(bodies)
    free = list(heads)
    for bi, body in enumerate(bodies):
        fits = [(fit, hi) for hi, head in enumerate(free) if (fit := _head_fit(body, head)) is not None]
        if fits:
            out[bi] = free.pop(min(fits)[1])
    return out


def _find_hands_for_body(body: BoundingBox, hand_boxes: List[BoundingBox]) -> List[BoundingBox]:
    """Return hand boxes whose centroid falls within the body bbox (with lateral margin)."""
    margin_x = body.w * 0.5
    margin_y = body.h * 0.15
    return [
        h for h in hand_boxes
        if (body.x - margin_x <= h.x + h.w / 2 <= body.x + body.w + margin_x
            and body.y - margin_y <= h.y + h.h / 2 <= body.y + body.h + margin_y)
    ]


def _dump_frame(dump_dir: Path, n: int, frame, detections, faces, hands, bodies, cs_x: float, cs_y: float,
                track_ids: Optional[List[int]] = None) -> None:
    """Debug: save the frame with detector boxes + scores drawn, and all scores as JSON."""
    def rec(b):
        return {'score': round(b.confidence, 3), 'bbox': [round(v) for v in (b.x, b.y, b.w, b.h)]}

    boxes = {'body': detections.body_boxes, 'head': detections.head_boxes, 'hand': detections.hand_boxes} if detections else {}
    data = {cls: [rec(b) for b in bs] for cls, bs in boxes.items()}
    for entry, track_id in zip(data.get('body', []), track_ids or []):
        entry['track_id'] = track_id
    data['face_results'] = [{'person_id': f.person_id, 'presence': round(f.presence, 3), 'bbox': rec(f.bbox)['bbox']} for f in faces]
    data['hand_results'] = [{'person_id': h.person_id, 'presence': round(h.presence, 3), 'bbox': rec(h.bbox)['bbox']} for h in hands]
    data['body_results'] = [{'person_id': b.person_id, 'presence': round(b.presence, 3)} for b in bodies]
    (dump_dir / f'frame_{n:04d}.json').write_text(json.dumps(data, indent=1))

    out = frame.copy()
    colors = {'body': (0, 0, 255), 'head': (0, 255, 0), 'hand': (255, 128, 0)}
    for cls, bs in boxes.items():
        for i, b in enumerate(bs):
            label = f'{cls} {b.confidence:.2f}'
            if cls == 'body' and track_ids:
                label = f'#{track_ids[i]} ' + label
            p1 = (int(b.x * cs_x), int(b.y * cs_y))
            p2 = (int((b.x + b.w) * cs_x), int((b.y + b.h) * cs_y))
            cv2.rectangle(out, p1, p2, colors[cls], 2)
            cv2.putText(out, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)
    cv2.imwrite(str(dump_dir / f'frame_{n:04d}.jpg'), out)


def _drop_nested_bodies(body_boxes: List[BoundingBox], head_boxes: List[BoundingBox] = (),
                        max_inside: float = 0.5, duplicate_inside: float = 0.8) -> List[BoundingBox]:
    """Drop body boxes that lie mostly inside another body box.

    A box >= max_inside inside a higher-scoring box is dropped. The detector's NMS keeps these because their
    IoU with the bigger box is low; on the Orin they were a lamp beside a person and duplicate boxes on a
    raised arm.

    A box >= duplicate_inside inside another box is a duplicate even when it scores higher (Orin: a 0.56 box
    on the left part of a person, 90% inside their 0.51 box). Only one of the pair is kept: the outer box,
    unless only the inner box fits a head (then the outer box is a false positive around a person).
    """
    def inside(a: BoundingBox, b: BoundingBox) -> float:
        ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        return ix * iy / (a.w * a.h)

    def fits_head(body: BoundingBox) -> bool:
        return any(_head_fit(body, h) is not None for h in head_boxes)

    drop = set()
    for i, a in enumerate(body_boxes):
        for j, b in enumerate(body_boxes):
            if i == j:
                continue
            share = inside(a, b)
            if b.confidence > a.confidence and share >= max_inside:
                drop.add(i)
            elif share >= duplicate_inside and i not in drop and j not in drop:
                drop.add(j if fits_head(a) and not fits_head(b) else i)
    return [a for i, a in enumerate(body_boxes) if i not in drop]


def _claim_hands(body: BoundingBox, hands: List[BoundingBox]) -> List[BoundingBox]:
    """This body's hands, removed from the pool so no other body gets them."""
    own = _find_hands_for_body(body, hands)
    for h in own:
        hands.remove(h)
    return own


def _screen_size() -> Optional[tuple]:
    """(width, height) of the monitor from `xrandr`, or None if it can't be read.

    Uses the connected output's current mode (e.g. `HDMI-0 connected primary 2560x1440+0+0`), falling back
    to the whole X screen's size.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(['xrandr', '--current'], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError) as e:
        print(f'[DISPLAY] xrandr failed: {e}', flush=True)
        return None
    m = (re.search(r' connected (?:primary )?(\d+)x(\d+)\+', out)
         or re.search(r'current (\d+) x (\d+)', out))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _fit(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple:
    """Largest (w, h) with the source aspect ratio that fits in max_w x max_h."""
    scale = min(max_w / src_w, max_h / src_h)
    return round(src_w * scale), round(src_h * scale)


# With --require-face, a track counts as a person after FACE_VERIFY_HITS frames with a face scored at least
# FACE_VERIFY_PRESENCE. Not yet tuned on live Orin scores: a logit of +5 (0.993), taken from issue #3's offline
# replay, was never reached live. The [STAGES] log prints the live face logits (face_logit=) to set it from.
FACE_VERIFY_PRESENCE = 0.95
FACE_VERIFY_HITS = 2

WINDOW_NAME = 'Poser'
QUIT_KEYS = (ord('q'), 27)  # q or Escape


class _Terminated(Exception):
    """Raised by the SIGTERM handler so the main loop unwinds through `finally` and releases the camera."""


def _raise_terminated(signum, frame):
    raise _Terminated()


def _is_quit_key(key: int) -> bool:
    """True for q or Escape. `key` is the raw cv2.waitKey() result (-1 when no key was pressed)."""
    return key != -1 and (key & 0xFF) in QUIT_KEYS


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

    tracker     = PersonTracker()
    track_stats = TrackStats() if config.track_report > 0 else None
    last_report = time.monotonic()
    require_face = config.require_face and face_proc is not None
    renderer    = Renderer(show_roi=config.show_roi, skelly=config.skelly, santa_hat=config.santa_hat)
    fps_counter = FPSCounter()

    display_w = config.width  if config.width  > 0 else capture.width
    display_h = config.height if config.height > 0 else capture.height
    # Full screen without --width/--height: draw at the monitor's resolution (e.g. 2560x1440), not the camera's.
    # The Orin's OpenCV (GTK) shows images unscaled in the top-left of a full-screen window, so the frame
    # itself must be screen-sized. The window's reported size is not used: before the window manager makes it
    # full screen, OpenCV reports a small default size (it shrank the picture to 1/16 of the screen).
    if config.fullscreen and config.width <= 0 and config.height <= 0:
        screen = _screen_size()
        if screen:
            display_w, display_h = _fit(capture.width, capture.height, *screen)
            print(f'[DISPLAY] screen {screen[0]}x{screen[1]}, drawing at {display_w}x{display_h}', flush=True)
        else:
            print('[DISPLAY] could not read the screen size (xrandr); pass --width/--height', flush=True)
    scale_display = (display_w != capture.width or display_h != capture.height)
    display_gpu = cv2.cuda_GpuMat()
    # --black-bg shows none of the camera image, so draw on this instead of resizing and downloading the frame.
    black_frame = np.zeros((display_h, display_w, 3), np.uint8) if config.black_bg else None

    # Accumulated ms per stage (capture, person, face, hands, body, render); printed as 60-frame averages with --fps.
    # Counts track extra work (false/extra detections) since face/body models run once per detected body.
    stage_ms = {k: 0.0 for k in ('capture', 'person', 'face', 'hands', 'body', 'render')}
    counts = {k: 0 for k in ('bodies', 'people', 'drawn', 'heads', 'hands', 'faces_run', 'hands_run')}
    face_logits: List[float] = []  # face model logits of drawn faces, for tuning FACE_VERIFY_PRESENCE

    dump_dir = Path(config.dump_detections) if config.dump_detections else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    dump_n, last_dump = 0, 0.0

    if config.fullscreen:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # `systemctl stop party-pose` and `pkill -TERM -f poser.py` send SIGTERM; exit cleanly like Esc does (issue #8).
    signal.signal(signal.SIGTERM, _raise_terminated)
    exit_code = 0

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
            track_ids: List[int] = []
            id_labels = []  # (track id, body box) for --show-ids

            if person_proc:
                t = time.perf_counter()
                detections = person_proc.process(frame_gpu, frame_w, frame_h)
                detections.body_boxes = _drop_nested_bodies(detections.body_boxes, detections.head_boxes)
                stage_ms['person'] += (time.perf_counter() - t) * 1000
                counts['bodies'] += len(detections.body_boxes)
                counts['heads'] += len(detections.head_boxes)
                counts['hands'] += len(detections.hand_boxes)

                # Stable per-person IDs (and so colours) across frames (issue #6).
                # Only confirmed tracks (seen in a few frames) are processed and drawn, oldest first, so a
                # flickering false positive gets no skeleton and can't take a real person's head or hands.
                tracks = tracker.update_tracks(detections.body_boxes)
                track_ids = [tr.id for tr in tracks]
                people = sorted(((tr, box) for tr, box in zip(tracks, detections.body_boxes) if tr.confirmed),
                                key=lambda p: (-p[0].age, p[0].id))
                free_hands = list(detections.hand_boxes)
                heads = _assign_heads([box for _, box in people], detections.head_boxes)
                counts['people'] += len(people)
                for (track, body_bbox), head_bbox in zip(people, heads):
                    person_id = track.id
                    track.note_head(head_bbox is not None)
                    head_info = dict(head_rate=track.head_rate(), head_ok=track.head_ok)
                    # --head-filter: skip tracks that rarely have a head (props), before the face and body
                    # models run on them.
                    if config.head_filter and not track.head_ok:
                        if track_stats:
                            track_stats.update(track.id, body_bbox, head_bbox is not None, None, None, **head_info)
                        continue
                    person_hands = _claim_hands(body_bbox, free_hands)
                    face_logit = body = None

                    if face_proc and head_bbox is not None:
                        t = time.perf_counter()
                        face = face_proc.process(frame_gpu, head_bbox, frame_w, frame_h, config.mirror)
                        stage_ms['face'] += (time.perf_counter() - t) * 1000
                        counts['faces_run'] += 1
                        if face is not None:
                            face.person_id = person_id
                            all_face_results.append(face)
                            face_logit = math.log(face.presence / max(1e-12, 1 - face.presence))
                            face_logits.append(face_logit)
                            if face.presence >= FACE_VERIFY_PRESENCE:
                                track.face_hits += 1

                    # A body is only drawn once the face model has clearly seen a real face in its own head box
                    # in a few frames. The lamp and other static false positives never show one (issue #3).
                    if require_face and track.face_hits < FACE_VERIFY_HITS:
                        if track_stats:
                            track_stats.update(track.id, body_bbox, head_bbox is not None, face_logit, None,
                                               **head_info)
                        continue
                    counts['drawn'] += 1
                    id_labels.append((person_id, body_bbox))

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

                    if track_stats:
                        track_stats.update(track.id, body_bbox, head_bbox is not None, face_logit, body,
                                           **head_info)

                if track_stats and time.monotonic() - last_report >= config.track_report:
                    last_report = time.monotonic()
                    for line in track_stats.report({t.id for t in tracker.tracks + tracker.lost}):
                        print(line, flush=True)

            # Timing for the final stage: GPU download, drawing, imshow, and waitKey.
            t_render = time.perf_counter()
            # Scale to display size on the GPU before the single GPU→CPU download, so lines/points are drawn
            # at native resolution without a CPU resize.
            if black_frame is not None:
                black_frame.fill(0)
                cpu_frame = black_frame
                cs_x = display_w / frame_w
                cs_y = display_h / frame_h
            elif scale_display:
                # Use the returned GpuMat: on the Orin's OpenCV build the dst argument is not written in place.
                display_gpu = cv2.cuda.resize(frame_gpu, (display_w, display_h), display_gpu)
                cpu_frame = display_gpu.download()
                cs_x = display_w / frame_w
                cs_y = display_h / frame_h
            else:
                cpu_frame = frame_gpu.download()
                cs_x = cs_y = 1.0

            dump_now = dump_dir and time.monotonic() - last_dump >= config.dump_every
            if dump_now:
                cv2.imwrite(str(dump_dir / f'frame_{dump_n + 1:04d}_raw.jpg'), cpu_frame)

            renderer.draw_faces(cpu_frame, all_face_results, display_w, display_h, cs_x, cs_y)
            renderer.draw_hands(cpu_frame, all_hand_results, display_w, display_h, cs_x, cs_y)
            renderer.draw_body(cpu_frame, all_body_results, display_w, display_h, cs_x, cs_y)
            if config.show_ids:
                renderer.draw_ids(cpu_frame, id_labels, cs_x, cs_y)

            if dump_now:
                dump_n += 1
                last_dump = time.monotonic()
                _dump_frame(dump_dir, dump_n, cpu_frame, detections, all_face_results, all_hand_results, all_body_results, cs_x, cs_y,
                            track_ids)

            fps_counter.tick()
            if config.show_fps:
                renderer.draw_fps(cpu_frame, fps_counter.get_fps())
                fps_counter._print_counter = getattr(fps_counter, "_print_counter", 0) + 1
                if fps_counter._print_counter % 60 == 0:
                    print(f"[FPS] {fps_counter.get_fps():.1f}", flush=True)

            cv2.imshow(WINDOW_NAME, cpu_frame)

            if _is_quit_key(cv2.waitKey(1)):
                print('[EXIT] quit key pressed', flush=True)
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
                if face_logits:
                    fl = sorted(face_logits)
                    print(f'[FACES] logit min={fl[0]:.1f} median={fl[len(fl) // 2]:.1f} max={fl[-1]:.1f} n={len(fl)}',
                          flush=True)
                face_logits.clear()
                stage_ms = dict.fromkeys(stage_ms, 0.0)
                counts = dict.fromkeys(counts, 0)
    except KeyboardInterrupt:
        print('[EXIT] interrupted', flush=True)
    except _Terminated:
        print('[EXIT] SIGTERM received', flush=True)
    except RuntimeError as e:
        # Non-zero so the systemd service restarts after e.g. a camera read failure.
        print(f'Runtime error: {e}', file=sys.stderr)
        exit_code = 1
    finally:
        # Ignore a second SIGTERM while the camera (nvargus) is being released.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        capture.release()
        cv2.destroyAllWindows()
        print('[EXIT] camera released', flush=True)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
