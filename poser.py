import sys
import json
import dataclasses
import math
import queue
import signal
import threading
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from src.config import parse_args
from src.model_manager import ModelManager
from src.video_capture import VideoCaptureModule
from src.person_detector import PersonDetector, PersonDetections
from src.face_processor import FaceProcessor, face_box_from_pose
from src.hand_processor import HandProcessor
from src.body_processor import BodyProcessor
from src.renderer import Renderer
from src.fps_counter import FPSCounter
from src.tracker import PersonTracker, iou, overlap_of_smaller
from src.smoothing import PoseSmoother
from src.track_stats import TrackStats
from src.flicker import FlickerCounter
from src.ghosts import Ghosts
from src.types import BoundingBox, FaceResult, HandResult, BodyResult, Landmark


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


def _moved_face(face: FaceResult, old_head: BoundingBox, new_head: BoundingBox) -> FaceResult:
    """The face found in old_head, moved and scaled to where the same head is now (a ghost face, issue #13)."""
    ocx, ocy = old_head.x + old_head.w / 2, old_head.y + old_head.h / 2
    ncx, ncy = new_head.x + new_head.w / 2, new_head.y + new_head.h / 2
    s = new_head.w / old_head.w if old_head.w > 0 else 1.0
    landmarks = [Landmark(ncx + (lm.x - ocx) * s, ncy + (lm.y - ocy) * s, lm.z * s, lm.visibility, lm.presence)
                 for lm in face.landmarks]
    return FaceResult(bbox=new_head, landmarks=landmarks, blendshapes=face.blendshapes,
                      person_id=face.person_id, presence=face.presence)


def _face_box(body: Optional[BodyResult], head: Optional[BoundingBox], frame_w: int, frame_h: int
              ) -> Optional[BoundingBox]:
    """Where to run the face model for a person: around the pose's face points when there is a pose (issue #13: a
    body always has a head, and the detector's head box is missing in a third of frames), else the head box."""
    if body is not None:
        box = face_box_from_pose(body.landmarks)
        if box.w >= 1 and box.x < frame_w and box.y < frame_h and box.x + box.w > 0 and box.y + box.h > 0:
            return box
    return head


def _pose_box(body: BodyResult, frame_w: int, frame_h: int, grow: float = 0.1) -> BoundingBox:
    """A person's box from their pose: around the visible body landmarks (0-32), `grow` x its size wider on each
    side (the detector's box has a margin too), clipped to the frame."""
    pts = [lm for lm in body.landmarks[:33] if lm.visibility >= 0.5] or body.landmarks[:33]
    x0, x1 = min(p.x for p in pts), max(p.x for p in pts)
    y0, y1 = min(p.y for p in pts), max(p.y for p in pts)
    dx, dy = grow * (x1 - x0), grow * (y1 - y0)
    x0, y0 = max(0.0, x0 - dx), max(0.0, y0 - dy)
    x1, y1 = min(float(frame_w), x1 + dx), min(float(frame_h), y1 + dy)
    return BoundingBox(x0, y0, x1 - x0, y1 - y0, body.presence)


def _follow_boxes(poses: List[tuple]) -> dict:
    """Track id -> pose box of the people to follow, from (track id, pose box) oldest track first.

    Two tracks on one person (two detector boxes on one dancer) end up with the same pose; a pose box with
    IoU >= 0.5 against an older one is that person found twice and is left out. Two people's poses don't
    overlap that much unless one hides the other.
    """
    out = {}
    for tid, box in poses:
        if all(iou(box, other) < 0.5 for other in out.values()):
            out[tid] = box
    return out


def _boxes_not_followed(body_boxes: List[BoundingBox], pose_boxes) -> List[BoundingBox]:
    """The detector body boxes that are not a followed person: a box sharing >= 0.4 of the smaller of it and a
    pose box is that person (a box on part of them, a second box on them, or a box around them), so it must
    not start a new track. laptop_test: a box on the person's raised arm and the sofa shared 0.41-0.8 and got
    its own ID at 0.5. two_people: a box on the other person shared >= 0.3 with a pose box in 1.7% of boxes,
    mostly while both people were followed (then it doesn't matter)."""
    pose_boxes = list(pose_boxes)
    return [b for b in body_boxes if all(overlap_of_smaller(b, p) < 0.4 for p in pose_boxes)]


def _big_enough(box: BoundingBox, frame_h: int, min_frac: float) -> bool:
    """A person is near enough to draw if their box is at least min_frac of the frame height (0 = off). Orin, frame
    1080 high: dancers 0.56-0.89, a child near the camera 0.47, people at the back of the room below that."""
    return min_frac <= 0 or box.h >= min_frac * frame_h


def _hand_on_own_wrist(hand: HandResult, pose, side: int, factor: float) -> bool:
    """A hand is this person's if its wrist (hand landmark 0) is within factor x shoulder width (pose 11-12, at
    least 10 px) of the pose wrist of that side (15 or 16). A crop chained from the hand's last landmarks can jump
    to another person's hand when hands cross bodies."""
    wrist = pose[15 + side]
    max_dist = max(10.0, factor * math.hypot(pose[11].x - pose[12].x, pose[11].y - pose[12].y))
    return math.hypot(hand.landmarks[0].x - wrist.x, hand.landmarks[0].y - wrist.y) <= max_dist


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


def _screen_canvas(screen_w: int, screen_h: int, w: int, h: int) -> tuple:
    """A black screen-sized image and the centred w x h area of it the picture is drawn in (issue #12).

    The full-screen window shows images unscaled from its top-left, and any part of the screen the image doesn't
    cover shows the window background (white): a 16:9 picture on a 1920x1200 monitor left a band. The area is a
    view, so drawing in it never reaches the border, which stays black.
    """
    canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
    x, y = (screen_w - w) // 2, (screen_h - h) // 2
    return canvas, canvas[y:y + h, x:x + w]


# With --require-face, a track counts as a person after FACE_VERIFY_HITS frames with a face scored at least
# FACE_VERIFY_PRESENCE. Not yet tuned on live Orin scores: a logit of +5 (0.993), taken from issue #3's offline
# replay, was never reached live. The [STAGES] log prints the live face logits (face_logit=) to set it from.
FACE_VERIFY_PRESENCE = 0.95
FACE_VERIFY_HITS = 2

# A hand is kept if its wrist is within this many shoulder widths of the pose wrist (_hand_on_own_wrist).
# two_people replay: median 0.08, 95% under 0.23; 0.5 rejected 1.9% of found hands (0.75: 1.3%, 1.0: 0.9%).
HAND_WRIST_FACTOR = 0.5

WINDOW_NAME = 'Poser'
QUIT_KEYS = (ord('q'), 27)  # q or Escape


class _Terminated(Exception):
    """Raised by the SIGTERM handler so the main loop unwinds through `finally` and releases the camera."""


def _raise_terminated(signum, frame):
    raise _Terminated()


def _is_quit_key(key: int) -> bool:
    """True for q or Escape. `key` is the raw cv2.waitKey() result (-1 when no key was pressed)."""
    return key != -1 and (key & 0xFF) in QUIT_KEYS


def _run_pipelined(produce, consume) -> None:
    """--pipeline: run produce() in a worker thread and consume() its results here, one frame apart, so the next
    frame's models run while this one is drawn. consume returns False to stop. An error in produce is raised here.
    The display stays in this (the main) thread; GTK windows must be used from one thread."""
    results: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()

    def put(item) -> None:
        while not stop.is_set():
            try:
                results.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def worker() -> None:
        try:
            while not stop.is_set():
                put((True, produce()))
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            put((False, e))

    thread = threading.Thread(target=worker, name='poser-infer', daemon=True)
    thread.start()
    try:
        while True:
            try:
                ok, item = results.get(timeout=0.5)  # with a timeout so Ctrl+C / SIGTERM are handled
            except queue.Empty:
                continue
            if not ok:
                raise item
            if not consume(item):
                return
    finally:
        stop.set()
        thread.join(timeout=2)


def main():
    config = parse_args()
    models_dir = Path(__file__).parent / 'models'
    model_manager = ModelManager(models_dir, platform=config.platform, detector=config.detector)

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

    person_proc = PersonDetector(model_manager, config.detector_bgr) if any_modality else None
    face_proc   = FaceProcessor(model_manager, config.face_expressions, config.face_batch) if config.face else None
    hand_proc   = HandProcessor(model_manager, config.hand_batch) if config.hands else None
    body_proc   = BodyProcessor(model_manager, config.pose_batch) if config.body else None

    tracker     = PersonTracker()
    smoother    = PoseSmoother() if config.smooth else None  # smooths landmarks per person across frames
    body_rois   = {}  # track id -> pose crop (cx, cy, size, angle) from that person's last body landmarks
    hand_rois   = {}  # track id -> (left, right) hand crops from that person's last hand landmarks
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
    # The monitor is read at every start, so the unit can be plugged into any TV or monitor (issue #12).
    canvas = picture = None  # screen-sized black image, and the area of it the picture is drawn in
    if config.fullscreen and config.width <= 0 and config.height <= 0:
        screen = _screen_size()
        if screen:
            display_w, display_h = _fit(capture.width, capture.height, *screen)
            if (display_w, display_h) != tuple(screen):
                canvas, picture = _screen_canvas(*screen, display_w, display_h)
            print(f'[DISPLAY] screen {screen[0]}x{screen[1]}, drawing at {display_w}x{display_h}'
                  + (', centred on a black screen-sized canvas' if canvas is not None else ''), flush=True)
        else:
            print('[DISPLAY] could not read the screen size (xrandr); pass --width/--height', flush=True)
    scale_display = (display_w != capture.width or display_h != capture.height)
    display_gpu = cv2.cuda_GpuMat()
    # --black-bg shows none of the camera image, so draw on this instead of resizing and downloading the frame.
    if config.black_bg:
        black_frame = picture if picture is not None else np.zeros((display_h, display_w, 3), np.uint8)
    else:
        black_frame = None

    # Accumulated ms per stage (capture, person, face, hands, body, smooth, render); printed as 60-frame averages with --fps.
    # Counts track extra work (false/extra detections) since face/body models run once per detected body.
    stage_ms = {k: 0.0 for k in ('capture', 'person', 'face', 'hands', 'body', 'smooth', 'render')}
    counts = {k: 0 for k in ('bodies', 'people', 'still', 'drawn', 'heads', 'hands', 'faces_run', 'ghosts',
                             'hands_found', 'small', 'hands_off')}
    ghosts = Ghosts(config.ghost_frames)  # last results of each person's parts, redrawn dimming when missing
    flicker = FlickerCounter()  # [FLICKER] blink counts per part, printed at the end of a --video run
    frame_no = 0
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

    def infer():
        """Capture a frame and run the models, tracking and smoothing on it."""
        nonlocal frame_no, body_rois, hand_rois, last_report
        frame_no += 1
        t = time.perf_counter()
        frame_gpu = capture.read_frame()
        t_frame = time.monotonic()
        frame_w, frame_h = frame_gpu.size()
        stage_ms['capture'] += (time.perf_counter() - t) * 1000

        all_face_results: List[FaceResult] = []
        all_hand_results: List[HandResult] = []
        all_body_results: List[BodyResult] = []
        detections = None
        track_ids: List[int] = []
        id_labels = []  # (track id, body box) for --show-ids
        seen = {}  # (track id, part) -> this frame's result (a face with its head box), for ghosts
        heads_now = {}  # track id -> face crop this frame, to move a ghost face with the head

        if person_proc:
            t = time.perf_counter()
            detections = person_proc.process(frame_gpu, frame_w, frame_h)
            detections.body_boxes = _drop_nested_bodies(detections.body_boxes, detections.head_boxes)
            stage_ms['person'] += (time.perf_counter() - t) * 1000
            counts['bodies'] += len(detections.body_boxes)
            counts['heads'] += len(detections.head_boxes)
            counts['hands'] += len(detections.hand_boxes)

            # Follow found people with the pose model, as MediaPipe does: a confirmed person whose pose was found
            # last frame gets the pose model on the crop from that pose, with or without a detector box (the
            # detector misses a moving dancer in most frames). Found again: they stay matched with a box from the
            # pose, and detector boxes on them start no second track. Not found: they need a detector box again.
            pre_bodies = {}  # track id -> this frame's pose (or None) for people already run here
            poses = []
            if body_proc and config.follow:
                t = time.perf_counter()
                follow = [tr for tr in sorted(tracker.tracks, key=lambda tr: (-tr.age, tr.id))
                          if tr.confirmed and body_rois.get(tr.id) is not None
                          and not (config.drop_still and tr.is_still())]
                if config.pose_batch:  # all followed people in one model run
                    found = body_proc.process_many(frame_gpu, [body_rois[tr.id] for tr in follow])
                for i, tr in enumerate(follow):
                    roi = body_rois[tr.id]
                    if config.pose_batch:
                        body = pre_bodies[tr.id] = found[i]
                    else:
                        roi_box = BoundingBox(roi[0] - roi[2] / 2, roi[1] - roi[2] / 2, roi[2], roi[2], 1.0)
                        body = pre_bodies[tr.id] = body_proc.process(frame_gpu, roi_box, frame_w, frame_h,
                                                                     config.mirror, roi=roi)
                    body_rois[tr.id] = body.roi if body is not None else None
                    if body is not None:
                        box = _pose_box(body, frame_w, frame_h)
                        if _big_enough(box, frame_h, config.min_body_height):  # too far back: not followed
                            poses.append((tr.id, box))
                stage_ms['body'] += (time.perf_counter() - t) * 1000
            followed = _follow_boxes(poses)
            for tid in pre_bodies:
                if tid not in followed:
                    pre_bodies[tid] = None  # pose lost, or a second track on a followed person
                    body_rois[tid] = None
            free_boxes = _boxes_not_followed(detections.body_boxes, followed.values())

            # Stable per-person IDs (and so colours) across frames (issue #6).
            # Only confirmed tracks (seen in a few frames) are processed and drawn, oldest first, so a
            # flickering false positive gets no skeleton and can't take a real person's head or hands.
            tracks = tracker.update_tracks(free_boxes, followed=followed)
            track_ids = [tr.id for tr in tracks]
            live_ids = {tr.id for tr in tracker.tracks}
            body_rois = {k: v for k, v in body_rois.items() if k in live_ids}
            hand_rois = {k: v for k, v in hand_rois.items() if k in live_ids}
            for line in tracker.events:
                print(line, flush=True)
            tracker.events.clear()
            # --drop-still (issue #7): a track that hasn't moved for ~10 s is a prop (or someone very
            # still); it gets no head, hands, or face and body models.
            confirmed = [(tr, box) for tr, box in zip(tracks, free_boxes) if tr.confirmed]
            confirmed += [(tr, followed[tr.id]) for tr in tracker.tracks if tr.id in followed]
            people = sorted(((tr, box) for tr, box in confirmed if not (config.drop_still and tr.is_still())),
                            key=lambda p: (-p[0].age, p[0].id))
            counts['still'] += len(confirmed) - len(people)
            # --min-body-height: people far back in the room get no models and are not drawn.
            near = [(tr, box) for tr, box in people if _big_enough(box, frame_h, config.min_body_height)]
            counts['small'] += len(people) - len(near)
            people = near
            heads = _assign_heads([box for _, box in people], detections.head_boxes)
            counts['people'] += len(people)
            # Every person's pose first: their face crop comes from it (issue #13).
            bodies = [None] * len(people)
            if body_proc:
                t = time.perf_counter()
                for k, (track, body_bbox) in enumerate(people):
                    if track.id in pre_bodies:  # one pose run per person per frame
                        bodies[k] = pre_bodies[track.id]
                        continue
                    bodies[k] = body_proc.process(frame_gpu, body_bbox, frame_w, frame_h, config.mirror,
                                                  roi=body_rois.get(track.id))
                    body_rois[track.id] = bodies[k].roi if bodies[k] is not None else None
                stage_ms['body'] += (time.perf_counter() - t) * 1000
            face_boxes = [_face_box(body, head, frame_w, frame_h) for body, head in zip(bodies, heads)]
            # Every person's face goes through the face model in one run (cheaper per face on the Orin).
            faces_by_person = {}  # index in people -> (FaceResult or None, face logit)
            if face_proc:
                with_box = [k for k, box in enumerate(face_boxes) if box is not None]
                if with_box:
                    t = time.perf_counter()
                    faces = face_proc.process_many(frame_gpu, [face_boxes[k] for k in with_box], frame_w, frame_h,
                                                   config.mirror)
                    faces_by_person = dict(zip(with_box, zip(faces, face_proc.last_logits)))
                    stage_ms['face'] += (time.perf_counter() - t) * 1000
                    counts['faces_run'] += len(with_box)
            def keep_hands(person_id, body, hands):
                # A hand is kept only near this person's pose wrist; a rejected one is searched again from
                # the pose wrist next frame.
                kept = [h if h is not None and _hand_on_own_wrist(h, body.landmarks, side, HAND_WRIST_FACTOR)
                        else None for side, h in enumerate(hands)]
                counts['hands_off'] += sum(h is not None for h in hands) - sum(h is not None for h in kept)
                hands = kept
                hand_rois[person_id] = tuple(h.roi if h is not None else None for h in hands)
                for side, hand in enumerate(hands):
                    if hand is not None:
                        hand.person_id = person_id
                        all_hand_results.append(hand)
                        seen[(person_id, f'hand{side}')] = hand
                counts['hands_found'] += sum(h is not None for h in hands)

            hand_people = []  # --hand-batch: (person id, pose) whose hands run after the loop
            for k, ((track, body_bbox), head_bbox) in enumerate(zip(people, heads)):
                person_id = track.id
                still = track.is_still()
                face = face_logit = None
                body = bodies[k]
                if face_boxes[k] is not None:
                    heads_now[person_id] = face_boxes[k]

                if k in faces_by_person:
                    face, face_logit = faces_by_person[k]  # logit also for rejected faces, in [TRACK] lines
                    if face is not None:
                        face.person_id = person_id
                        all_face_results.append(face)
                        seen[(person_id, 'face')] = (face, face_boxes[k])
                        face_logit = math.log(face.presence / max(1e-12, 1 - face.presence))
                        face_logits.append(face_logit)
                        if face.presence >= FACE_VERIFY_PRESENCE:
                            track.face_hits += 1

                # A body is only drawn once the face model has clearly seen a real face in its own face crop
                # in a few frames. The lamp and other static false positives never show one (issue #3).
                if require_face and track.face_hits < FACE_VERIFY_HITS:
                    if track_stats:
                        track_stats.update(track.id, body_bbox, head_bbox is not None, face_logit, None,
                                           still=still)
                    continue
                counts['drawn'] += 1
                id_labels.append((person_id, body_bbox))

                if body is not None:
                    body.person_id = person_id
                    all_body_results.append(body)
                    seen[(person_id, 'body')] = body

                # Hands are cropped from the pose's wrists (or the hand's last landmarks), so they need the body.
                if hand_proc and body is not None:
                    if config.hand_batch:
                        hand_people.append((person_id, body))  # all people's hands in one run, after this loop
                    else:
                        t = time.perf_counter()
                        keep_hands(person_id, body, hand_proc.process(frame_gpu, body.landmarks,
                                                                      hand_rois.get(person_id, (None, None))))
                        stage_ms['hands'] += (time.perf_counter() - t) * 1000

                if track_stats:
                    track_stats.update(track.id, body_bbox, head_bbox is not None, face_logit, body,
                                       still=still)

            if hand_people:
                t = time.perf_counter()
                found = hand_proc.process_many(frame_gpu, [(body.landmarks, hand_rois.get(pid, (None, None)))
                                                           for pid, body in hand_people])
                for (pid, body), hands in zip(hand_people, found):
                    keep_hands(pid, body, hands)
                stage_ms['hands'] += (time.perf_counter() - t) * 1000

            if track_stats and time.monotonic() - last_report >= config.track_report:
                last_report = time.monotonic()
                for line in track_stats.report({t.id for t in tracker.tracks + tracker.lost}):
                    print(line, flush=True)

        if smoother:
            t = time.perf_counter()
            smoother.smooth(all_face_results, all_hand_results, all_body_results, t_frame)
            stage_ms['smooth'] += (time.perf_counter() - t) * 1000

        # Ghosts (issue #13): a tracked person's part missing this frame is drawn again from its last smoothed
        # result, dimming, in the track's colour. Copies with their own brightness, so the result being drawn
        # by the other thread (--pipeline) is never changed.
        ghosted = ghosts.update(seen)
        for (person_id, part), item, fade in ghosted:
            if part == 'face':
                face, old_head = item
                if person_id in heads_now:
                    face = _moved_face(face, old_head, heads_now[person_id])
                all_face_results.append(dataclasses.replace(face, fade=fade))
            elif part == 'body':
                all_body_results.append(dataclasses.replace(item, fade=fade))
            else:
                all_hand_results.append(dataclasses.replace(item, fade=fade))
        counts['ghosts'] += len(ghosted)
        flicker.update(list(seen) + [key for key, _, _ in ghosted])

        if black_frame is not None:
            frame_gpu = None            # not drawn
        elif config.pipeline:
            frame_gpu = frame_gpu.clone()  # read_frame() reuses its buffer for the next frame
        return (frame_gpu, frame_w, frame_h, all_face_results, all_hand_results, all_body_results, id_labels,
                detections, track_ids)

    def render(out) -> bool:
        """Draw and show one frame's results; False when the quit key was pressed."""
        nonlocal display_gpu, dump_n, last_dump, rendered, label_t0
        (frame_gpu, frame_w, frame_h, all_face_results, all_hand_results, all_body_results, id_labels,
         detections, track_ids) = out
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
        if picture is not None and cpu_frame is not picture:
            picture[:] = cpu_frame
            cpu_frame = picture

        dump_now = dump_dir and time.monotonic() - last_dump >= config.dump_every
        if dump_now:
            cv2.imwrite(str(dump_dir / f'frame_{dump_n + 1:04d}_raw.jpg'), cpu_frame)

        renderer.draw_faces(cpu_frame, all_face_results, display_w, display_h, cs_x, cs_y)
        renderer.draw_hands(cpu_frame, all_hand_results, display_w, display_h, cs_x, cs_y)
        renderer.draw_body(cpu_frame, all_body_results, display_w, display_h, cs_x, cs_y)
        if config.show_ids:
            renderer.draw_ids(cpu_frame, id_labels, cs_x, cs_y)
        if config.test_label:
            label_t0 = label_t0 or time.monotonic()
            renderer.draw_banner(cpu_frame, f'{config.test_label}  {int(time.monotonic() - label_t0)} s')

        if dump_now:
            dump_n += 1
            last_dump = time.monotonic()
            _dump_frame(dump_dir, dump_n, cpu_frame, detections, all_face_results, all_hand_results, all_body_results, cs_x, cs_y,
                        track_ids)

        fps_counter.tick()
        rendered += 1
        if config.show_fps:
            renderer.draw_fps(cpu_frame, fps_counter.get_fps())
            fps_counter._print_counter = getattr(fps_counter, "_print_counter", 0) + 1
            if fps_counter._print_counter % 60 == 0:
                print(f"[FPS] {fps_counter.get_fps():.1f}", flush=True)

        cv2.imshow(WINDOW_NAME, canvas if canvas is not None else cpu_frame)

        if _is_quit_key(cv2.waitKey(1)):
            print('[EXIT] quit key pressed', flush=True)
            return False

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
            # Cleared in place: with --pipeline the worker thread adds to these dicts.
            for k in stage_ms:
                stage_ms[k] = 0.0
            for k in counts:
                counts[k] = 0
        return True

    if config.switch_ms > 0:
        sys.setswitchinterval(config.switch_ms / 1000)
    rendered, t_start = 0, time.monotonic()
    label_t0 = None  # when the first picture was shown, for the --test-label seconds counter
    try:
        if config.pipeline:
            _run_pipelined(infer, render)
        else:
            while render(infer()):
                pass
    except KeyboardInterrupt:
        print('[EXIT] interrupted', flush=True)
    except _Terminated:
        print('[EXIT] SIGTERM received', flush=True)
    except EOFError:
        # --video: the whole clip has been processed; one line to compare runs on the same clip.
        seconds = time.monotonic() - t_start
        print(f'[BENCH] frames={rendered} seconds={seconds:.1f} fps={rendered / seconds:.2f} '
              f'pipeline={config.pipeline} smooth={config.smooth}', flush=True)
        print(flicker.report(), flush=True)
        print('[EXIT] end of video', flush=True)
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
