"""Record a raw camera clip for repeatable benchmarks: `poser.py --video CLIP` then replays the same frames.

Frames are saved as the camera gives them (before the app's white-balance gain and mirroring, which replay applies
again), uncompressed (src/raw_video.py): 1080p is ~6.2 MB a frame, ~11 GB a minute. The clip shows people:
keep it in the git-ignored testdata/ folder on the machine that recorded it.

Run with the service stopped (it holds the camera). The monitor shows a full-screen preview with a countdown, so
the people in the room know when to perform:

    env/bin/python tools/orin_bench/record_video.py testdata/clips/two_people --seconds 60 --countdown 10
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import detect_platform  # noqa: E402
from src.raw_video import RawVideoWriter  # noqa: E402
from src.video_capture import VideoCaptureModule  # noqa: E402

WINDOW = 'Party-Pose recording'


def _show(frame, text: str, colour) -> None:
    view = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    scale = view.shape[0] / 300
    cv2.putText(view, text, (20, int(60 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(8 * scale))
    cv2.putText(view, text, (20, int(60 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, int(3 * scale))
    cv2.imshow(WINDOW, view)
    cv2.waitKey(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('clip', help='output path without extension (writes CLIP.bgr and CLIP.json)')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--countdown', type=float, default=10)
    ap.add_argument('--camera', type=int, default=0)
    args = ap.parse_args()

    Path(args.clip).parent.mkdir(parents=True, exist_ok=True)
    cap = VideoCaptureModule(SimpleNamespace(platform=detect_platform(), camera=args.camera, mirror=False, video=None))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        t0 = time.monotonic()
        while (left := args.countdown - (time.monotonic() - t0)) > 0:
            ok, frame = cap._cap.read()
            if ok:
                _show(frame, f'GET READY {left:.0f}', (0, 255, 255))
        with RawVideoWriter(args.clip, cap.width, cap.height) as w:
            t0 = time.monotonic()
            while (left := args.seconds - (time.monotonic() - t0)) > 0:
                ok, frame = cap._cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                w.write(frame)
                if w.count % 3 == 0:  # preview every third frame, so showing it doesn't slow the recording
                    _show(frame, f'RECORDING - dance! {left:.0f}', (0, 0, 255))
            w.fps = w.count / (time.monotonic() - t0)
        print(f'[RECORD] {w.count} frames {cap.width}x{cap.height} at {w.fps:.1f} fps -> {args.clip}.bgr', flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3:
            ok, frame = cap._cap.read()
            if ok:
                _show(frame, 'DONE - thank you', (0, 255, 0))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
