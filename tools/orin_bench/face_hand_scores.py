"""Why faces and hands drop: replay a clip through the app's pipeline and log every face and hand score.

Per frame and confirmed person (as the app, mirrored):
- face: is there a head box; the face score (logit, kept at >= 0) for the head box scaled 0.8/0.9/1.0/1.2, and for a
  square crop from the pose's face points (0-10: nose, eyes, ears, mouth) sized 1.5/2.0/2.5x their spread, which
  also works when the detector found no head box;
- hands: per side, the hand model's presence before the 0.5 cutoff, where the crop came from (the hand's last
  landmarks, the pose wrist, or none), the wrist's visibility, and whether the wrist is over the torso.

Writes one JSON line per person and frame to OUT and prints a summary. Run on the Orin with the service stopped:
    env/bin/python tools/orin_bench/face_hand_scores.py testdata/clips/face_walk /tmp/fh_face_walk.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from poser import _assign_heads, _drop_nested_bodies  # noqa: E402
from src.body_processor import BodyProcessor  # noqa: E402
from src.config import detect_platform  # noqa: E402
from src.face_processor import FaceProcessor  # noqa: E402
from src.hand_processor import HandProcessor, _POSE_HAND  # noqa: E402
from src.model_manager import ModelManager  # noqa: E402
from src.person_detector import PersonDetector  # noqa: E402
from src.tracker import PersonTracker  # noqa: E402
from src.types import BoundingBox  # noqa: E402
from src.video_capture import VideoCaptureModule  # noqa: E402

HEAD_SCALES = (0.8, 0.9, 1.0, 1.2)
POSE_SCALES = (1.5, 2.0, 2.5)
HAND_KEEP = 0.5  # HandProcessor.CONF_THRESHOLD in the app


def scaled(box: BoundingBox, s: float) -> BoundingBox:
    """The box scaled by s about its centre."""
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    return BoundingBox(cx - box.w * s / 2, cy - box.h * s / 2, box.w * s, box.h * s, box.confidence)


def pose_face_box(pose, s: float) -> BoundingBox:
    """A square around the pose's face points (0-10), s times their larger spread."""
    pts = np.array([(pose[i].x, pose[i].y) for i in range(11)])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    side = s * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return BoundingBox(cx - side / 2, cy - side / 2, side, side, 1.0)


def over_torso(pose, side: int) -> bool:
    """True when the wrist lies inside the shoulders-hips quadrilateral (11, 12, 24, 23)."""
    quad = np.array([(pose[i].x, pose[i].y) for i in (11, 12, 24, 23)])
    p = np.array([pose[_POSE_HAND[side][0]].x, pose[_POSE_HAND[side][0]].y])
    signs = [np.cross(quad[(k + 1) % 4] - quad[k], p - quad[k]) for k in range(4)]
    return all(v >= 0 for v in signs) or all(v <= 0 for v in signs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('clip')
    ap.add_argument('out')
    ap.add_argument('--no-mirror', dest='mirror', action='store_false')
    args = ap.parse_args()

    platform = detect_platform()
    mm = ModelManager(REPO_ROOT / 'models', platform=platform)
    cap = VideoCaptureModule(SimpleNamespace(platform=platform, camera=0, mirror=args.mirror, video=args.clip))
    person, face, body = PersonDetector(mm), FaceProcessor(mm), BodyProcessor(mm)
    hand = HandProcessor(mm)
    hand.CONF_THRESHOLD = 0.0  # every hand's presence is returned; the app's 0.5 cutoff is applied below
    tracker = PersonTracker()
    body_rois, hand_rois = {}, {}
    rows = []
    frame_no = 0
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
            people = sorted(((t, b) for t, b in zip(tracks, det.body_boxes) if t.confirmed),
                            key=lambda p: (-p[0].age, p[0].id))
            heads = _assign_heads([b for _, b in people], det.head_boxes)
            for (track, box), head in zip(people, heads):
                pid = track.id
                row = {'frame': frame_no, 'id': pid, 'body_h': round(box.h), 'head': head is not None}
                res = body.process(frame, box, fw, fh, args.mirror, roi=body_rois.get(pid))
                body_rois[pid] = res.roi if res is not None else None
                row['pose'] = res is not None
                crops, names = [], []
                if head is not None:
                    row['head_wh'] = (round(head.w), round(head.h))
                    crops += [scaled(head, s) for s in HEAD_SCALES]
                    names += [f'head{s}' for s in HEAD_SCALES]
                if res is not None:
                    crops += [pose_face_box(res.landmarks, s) for s in POSE_SCALES]
                    names += [f'pose{s}' for s in POSE_SCALES]
                if crops:
                    face.process_many(frame, crops, fw, fh, args.mirror)
                    row['face'] = {n: round(v, 2) for n, v in zip(names, face.last_logits)}
                if res is not None:
                    prev = hand_rois.get(pid, (None, None))
                    found = hand.process(frame, res.landmarks, prev)
                    row['hands'] = []
                    for side in (0, 1):
                        wrist = res.landmarks[_POSE_HAND[side][0]]
                        src = 'hand' if prev[side] is not None else ('pose' if wrist.visibility >= hand.WRIST_VISIBILITY
                                                                     else 'none')
                        h = found[side]
                        row['hands'].append({'src': src, 'presence': None if h is None else round(h.presence, 3),
                                             'wrist_vis': round(wrist.visibility, 2), 'over_torso': over_torso(res.landmarks, side)})
                    hand_rois[pid] = tuple(h.roi if h is not None and h.presence >= HAND_KEEP else None for h in found)
                out.write(json.dumps(row) + '\n')
                rows.append(row)
    cap.release()
    print(summary(rows, frame_no))


def summary(rows, frames: int) -> str:
    n = len(rows)
    if not n:
        return f'{frames} frames, no confirmed person'
    pct = lambda k: f'{100 * k / n:.0f}%'  # noqa: E731
    lines = [f'{frames} frames, {n} person-frames, ids {sorted({r["id"] for r in rows})}',
             f'head box {pct(sum(r["head"] for r in rows))}  pose {pct(sum(r["pose"] for r in rows))}']
    kept = lambda r, k: r.get('face', {}).get(k, -99) >= 0  # noqa: E731
    for k in [f'head{s}' for s in HEAD_SCALES] + [f'pose{s}' for s in POSE_SCALES]:
        with_head = sum(kept(r, k) for r in rows if r['head'])
        no_head = sum(kept(r, k) for r in rows if not r['head'])
        lines.append(f'face {k:8s} found {pct(with_head + no_head)}  (with head box {with_head}, without {no_head})')
    for s in POSE_SCALES:
        either = sum(kept(r, 'head1.0') or kept(r, f'pose{s}') for r in rows)
        lines.append(f'face head1.0, else pose{s}: found {pct(either)}')
    for side in (0, 1):
        hs = [r['hands'][side] for r in rows if 'hands' in r]
        for torso in (False, True):
            sel = [h for h in hs if h['over_torso'] == torso]
            if not sel:
                continue
            p = [h['presence'] for h in sel if h['presence'] is not None]
            bins = np.histogram(p, bins=[0, 0.1, 0.3, 0.4, 0.5, 1.01])[0] if p else []
            no_crop = sum(h['src'] == 'none' for h in sel)
            lines.append(f'hand{side} {"over torso" if torso else "away     "} n={len(sel)} found(>=0.5) '
                         f'{100 * sum(v >= HAND_KEEP for v in p) / len(sel):.0f}%  no crop {no_crop}  '
                         f'presence bins <0.1/0.1-0.3/0.3-0.4/0.4-0.5/>=0.5: {"/".join(str(int(b)) for b in bins)}')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
