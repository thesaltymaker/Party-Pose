"""Record a raw camera clip for repeatable benchmarks: `poser.py --video CLIP` then replays the same frames.

Frames are saved as the camera gives them (before the app's white-balance gain and mirroring, which replay applies
again), uncompressed (src/raw_video.py): 1080p is ~6.2 MB a frame, ~11 GB a minute. The clip shows people:
keep it in the git-ignored testdata/ folder on the machine that recorded it.

Run with the service stopped (it holds the camera). The monitor shows a full-screen preview with a countdown, so
the people in the room know when to perform:

    env/bin/python tools/orin_bench/record_video.py testdata/clips/two_people --seconds 60 --countdown 10

--label is shown on the monitor (what to do). Once a second the camera's gain and exposure, the picture's mean
brightness and its share of blown-out pixels are logged; they are saved in CLIP.json with --light and --lux.
"""
import argparse
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import detect_platform  # noqa: E402
from src.raw_video import RawVideoWriter  # noqa: E402
from src.video_capture import VideoCaptureModule, query_gain_exposure  # noqa: E402

WINDOW = 'Party-Pose recording'


def _show(frame, text: str, colour) -> None:
    view = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    scale = view.shape[0] / 300
    cv2.putText(view, text, (20, int(60 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(8 * scale))
    cv2.putText(view, text, (20, int(60 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, int(3 * scale))
    cv2.imshow(WINDOW, view)
    cv2.waitKey(1)


def frame_brightness(frame) -> tuple:
    """(mean brightness 0-255, % of pixels blown out at >= 250) of a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float((gray >= 250).mean() * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('clip', help='output path without extension (writes CLIP.bgr and CLIP.json)')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--countdown', type=float, default=10)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--label', default='dance!', help='what to do, shown on the monitor')
    ap.add_argument('--light', default='', help='the room light, saved with the clip (e.g. "lamps on, blinds shut")')
    ap.add_argument('--lux', type=float, default=None, help='a light meter reading at the dancers, saved with the clip')
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
                _show(frame, f'GET READY {left:.0f}: {args.label}', (0, 255, 255))
        with RawVideoWriter(args.clip, cap.width, cap.height) as w:
            log = []  # once a second: camera gain and exposure, picture brightness
            stop = threading.Event()

            def sample_camera() -> None:  # v4l2-ctl takes ~20 ms: not in the recording loop
                while not stop.wait(1.0):
                    gain, exposure = query_gain_exposure()
                    log.append({'t': round(time.monotonic() - t0, 1), 'gain': gain, 'exposure_us': exposure})

            t0 = time.monotonic()
            sampler = threading.Thread(target=sample_camera, daemon=True)
            sampler.start()
            next_luma = 0
            while (left := args.seconds - (time.monotonic() - t0)) > 0:
                ok, frame = cap._cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                w.write(frame)
                if w.count % 3 == 0:  # preview every third frame, so showing it doesn't slow the recording
                    _show(frame, f'RECORDING {left:.0f}: {args.label}', (0, 0, 255))
                if log and next_luma < len(log):  # brightness of one frame per camera sample
                    log[-1]['luma'], log[-1]['saturated_pct'] = (round(v, 2) for v in frame_brightness(frame[::4, ::4]))
                    next_luma = len(log)
            w.fps = w.count / (time.monotonic() - t0)
            stop.set()
            sampler.join(timeout=2)
            w.meta = {'label': args.label, 'light': args.light, 'lux': args.lux,
                      'recorded': time.strftime('%Y-%m-%d %H:%M:%S'), 'brightness': log}
        print(f'[RECORD] {w.count} frames {cap.width}x{cap.height} at {w.fps:.1f} fps -> {args.clip}.bgr', flush=True)
        for k in ('gain', 'exposure_us', 'luma', 'saturated_pct'):
            vals = sorted(e[k] for e in log if e.get(k) is not None)
            if vals:
                print(f'[LIGHT] {k}: min={vals[0]} median={vals[len(vals) // 2]} max={vals[-1]}', flush=True)
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
