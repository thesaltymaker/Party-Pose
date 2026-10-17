"""Why faces drop: replay a clip through the app's detector, tracker and pose model and score the face model on
many face crops per person and frame.

Crops (all squares; outside the frame is black, as in MediaPipe), named CENTRE-SCALE-ROT:
- app: the app's crop (face_box_from_pose, cut off at the frame edge and stretched, no rotation);
- centre: mid = middle of the pose's face points 0-10 (the app), face = mean of nose, eyes and mouth (0, 2, 5, 9,
  10; ears left out: one ear goes out of view when the head turns), nose = point 0;
- scale: times the larger spread of points 0-10;
- rot: 0 = upright, eyes = turned so the pose's eye line (2-5) is level, as MediaPipe turns its face crop;
- track: MediaPipe's own way: from the last frame's face landmarks when the face was kept (1.5x their spread,
  turned by the eye corners 33-263, moved by the pose nose's motion), else face-2.0-eyes.

Per row: the logits, the face points' visibility, a head turn value from the pose (nose x against the ears'
middle, in ear distances: 0 facing the camera, +-0.5 side on), the spread in pixels, the eye line angle.
Writes one JSON line per person and frame to OUT and prints a summary per track.
    env/bin/python tools/orin_bench/face_crops.py testdata/clips_half/three_dancers /tmp/fc_three_dancers.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from poser import _assign_heads, _big_enough, _drop_nested_bodies  # noqa: E402
from src.body_processor import BodyProcessor  # noqa: E402
from src.config import detect_platform  # noqa: E402
from src.face_processor import FaceProcessor, face_box_from_pose  # noqa: E402
from src.model_manager import ModelManager  # noqa: E402
from src.person_detector import PersonDetector  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402
from src.tracker import PersonTracker  # noqa: E402
from src.video_capture import VideoCaptureModule  # noqa: E402

SCALES = (1.5, 2.0, 2.5)
CENTRES = ('mid', 'face', 'nose')
ROTS = ('0', 'eyes')


def eye_angle(pose) -> float:
    """Angle of the pose's eye line (2 -> 5 or back, pointing right in the image), radians."""
    dx, dy = pose[5].x - pose[2].x, pose[5].y - pose[2].y
    if dx < 0:
        dx, dy = -dx, -dy
    return math.atan2(dy, dx)


def pose_crop(pose, centre: str, scale: float, rot: str):
    """(cx, cy, side, angle) of a square face crop from the pose's face points."""
    pts = np.array([(pose[i].x, pose[i].y) for i in range(11)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    side = scale * float(max(hi - lo))
    if centre == 'mid':
        c = (lo + hi) / 2
    elif centre == 'face':
        c = pts[[0, 2, 5, 9, 10]].mean(axis=0)
    else:
        c = pts[0]
    return float(c[0]), float(c[1]), side, eye_angle(pose) if rot == 'eyes' else 0.0


def landmark_crop(pts: np.ndarray, scale: float = 1.5):
    """MediaPipe's face crop from its own last landmarks [478, 2] (frame pixels): turned by the eye corners
    (33, 263), centred on the landmarks' box in that turned frame, `scale` x its larger side."""
    d = pts[263] - pts[33]
    if d[0] < 0:
        d = -d
    a = math.atan2(d[1], d[0])
    c, s = math.cos(a), math.sin(a)
    r = pts @ np.array([[c, -s], [s, c]])  # into the turned frame
    lo, hi = r.min(axis=0), r.max(axis=0)
    m = (lo + hi) / 2
    cx, cy = m[0] * c - m[1] * s, m[0] * s + m[1] * c
    return float(cx), float(cy), scale * float(max(hi - lo)), a


def run_face(session, frame_gpu, crop):
    """Face logit and landmarks [478, 2] in frame pixels for a (cx, cy, side, angle) crop."""
    cx, cy, side, a = crop
    M = Preprocessor.roi_affine(cx, cy, side, a, 256)
    arr = Preprocessor.crop_rotated_nhwc(frame_gpu, M, 256)
    lms, logit = session.run(['Identity', 'Identity_1'], {'input_12': arr})
    raw = lms.reshape(478, 3)[:, :2]
    return float(logit.reshape(-1)[0]), raw @ M[:, :2].T + M[:, 2]


def head_turn(pose) -> float:
    ears = pose[7].x - pose[8].x
    if abs(ears) < 1:
        return 0.0
    return (pose[0].x - (pose[7].x + pose[8].x) / 2) / abs(ears)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('clip')
    ap.add_argument('out')
    ap.add_argument('--no-mirror', dest='mirror', action='store_false')
    ap.add_argument('--min-body-height', type=float, default=0.40)
    args = ap.parse_args()

    platform = detect_platform()
    mm = ModelManager(REPO_ROOT / 'models', platform=platform)
    cap = VideoCaptureModule(SimpleNamespace(platform=platform, camera=0, mirror=args.mirror, video=args.clip))
    person, face, body = PersonDetector(mm), FaceProcessor(mm), BodyProcessor(mm)
    session = mm.get_session('face_landmarks')
    tracker = PersonTracker()
    body_rois, last_face = {}, {}  # track id -> pose roi; track id -> (face landmarks, pose nose) when kept
    rows, frame_no = [], 0
    with open(args.out, 'w') as out:
        while True:
            try:
                frame = cap.read_frame()
            except EOFError:
                break
            frame_no += 1
            fw, fh = frame.size()
            det = person.process(frame, fw, fh)
            det.body_boxes = _drop_nested_bodies(det.body_boxes, det.head_boxes)
            tracks = tracker.update_tracks(det.body_boxes)
            people = sorted(((t, b) for t, b in zip(tracks, det.body_boxes)
                             if t.confirmed and _big_enough(b, fh, args.min_body_height)),
                            key=lambda p: (-p[0].age, p[0].id))
            heads = _assign_heads([b for _, b in people], det.head_boxes)
            for (track, box), head in zip(people, heads):
                pid = track.id
                res = body.process(frame, box, fw, fh, args.mirror, roi=body_rois.get(pid))
                body_rois[pid] = res.roi if res is not None else None
                if res is None:
                    last_face.pop(pid, None)
                    continue
                pose = res.landmarks
                fb = face_box_from_pose(pose)
                row = {'frame': frame_no, 'id': pid, 'body_h': round(box.h), 'cx': round(box.x + box.w / 2),
                       'head': head is not None, 'vis': [round(pose[i].visibility, 2) for i in range(11)],
                       'turn': round(head_turn(pose), 2), 'spread': round(fb.w / 2.0, 1),
                       'eye_deg': round(math.degrees(eye_angle(pose)), 1)}
                lg = {}
                face.process_many(frame, [fb], fw, fh, args.mirror)
                lg['app'] = face.last_logits[0]
                for c in CENTRES:
                    for s in SCALES:
                        for r in ROTS:
                            lg[f'{c}-{s}-{r}'] = run_face(session, frame, pose_crop(pose, c, s, r))[0]
                if pid in last_face:
                    pts, nose = last_face[pid]
                    cx, cy, side, a = landmark_crop(pts)
                    crop = (cx + pose[0].x - nose[0], cy + pose[0].y - nose[1], side, a)
                else:
                    crop = pose_crop(pose, 'face', 2.0, 'eyes')
                lg['track'], pts = run_face(session, frame, crop)
                if lg['track'] >= 0:
                    last_face[pid] = (pts, (pose[0].x, pose[0].y))
                else:
                    last_face.pop(pid, None)
                row['logit'] = {k: round(v, 2) for k, v in lg.items()}
                out.write(json.dumps(row) + '\n')
                rows.append(row)
    cap.release()
    print(summary(rows, frame_no))


def summary(rows, frames: int) -> str:
    if not rows:
        return f'{frames} frames, no person'
    names = list(rows[0]['logit'])
    lines = [f'{frames} frames, {len(rows)} person-frames']
    ids = sorted({r['id'] for r in rows}, key=lambda i: -sum(r['id'] == i for r in rows))
    groups = [('all', rows)] + [(f'id{i}', [r for r in rows if r['id'] == i]) for i in ids
                                if sum(r['id'] == i for r in rows) >= 30]
    lines.append('kept at logit >= 0 (presence 0.5), % of person-frames with a pose:')
    lines.append(f'{"crop":16s}' + ''.join(f'{g:>9s}' for g, _ in groups))
    lines.append(f'{"n":16s}' + ''.join(f'{len(rs):9d}' for _, rs in groups))
    for k in names:
        lines.append(f'{k:16s}' + ''.join(f'{100 * np.mean([r["logit"][k] >= 0 for r in rs]):8.0f}%' for _, rs in groups))
    for g, rs in groups:
        sp = np.array([r['spread'] for r in rs])
        tn = np.array([abs(r['turn']) for r in rs])
        lines.append(f'{g}: spread px median {np.median(sp):.0f} (p10 {np.percentile(sp, 10):.0f}), |turn| median '
                     f'{np.median(tn):.2f}, nose vis median {np.median([r["vis"][0] for r in rs]):.2f}, '
                     f'|eye angle| > 20 deg {100 * np.mean([abs(r["eye_deg"]) > 20 for r in rs]):.0f}%')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
