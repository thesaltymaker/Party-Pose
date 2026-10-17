import numpy as np
import pytest

from src.raw_video import RawVideoReader, RawVideoWriter


def _frames(n, h=6, w=8):
    return [np.full((h, w, 3), i, np.uint8) for i in range(n)]


def test_frames_read_back_in_order(tmp_path):
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        for f in _frames(5):
            w.write(f)
    r = RawVideoReader(path)
    assert (r.width, r.height, r.count) == (8, 6, 5)
    for i in range(5):
        assert (r.read() == i).all()


def test_end_of_clip_raises_eoferror(tmp_path):
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        w.write(_frames(1)[0])
    r = RawVideoReader(path)
    r.read()
    with pytest.raises(EOFError):
        r.read()


def test_wrong_frame_size_is_rejected(tmp_path):
    with RawVideoWriter(tmp_path / 'clip', 8, 6) as w:
        with pytest.raises(ValueError):
            w.write(np.zeros((6, 9, 3), np.uint8))


def test_fps_of_the_recording_is_kept(tmp_path):
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        for f in _frames(3):
            w.write(f)
        w.fps = 29.5
    assert RawVideoReader(path).fps == 29.5


def test_app_camera_reads_the_clip_and_stops_at_its_end(tmp_path):
    from types import SimpleNamespace
    from src.video_capture import VideoCaptureModule
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        for f in _frames(2):
            w.write(f)
    cap = VideoCaptureModule(SimpleNamespace(platform='laptop', camera=0, mirror=False, video=str(path)))
    assert (cap.width, cap.height) == (8, 6)
    assert (cap.read_frame().download() == 0).all()
    assert (cap.read_frame().download() == 1).all()
    with pytest.raises(EOFError):
        cap.read_frame()
    cap.release()


def test_video_option():
    from unittest.mock import patch
    from src.config import parse_args
    with patch('sys.argv', ['poser.py']):
        assert parse_args().video is None
    with patch('sys.argv', ['poser.py', '--video', 'testdata/clip']):
        assert parse_args().video == 'testdata/clip'


def test_reader_drops_each_frame_from_the_file_cache(tmp_path, monkeypatch):
    # Orin: replaying an 11 GB clip filled the shared CPU/GPU memory with file cache and the next run's GPU
    # allocation failed (NvMap error 12). Each frame read is dropped from the cache.
    import os
    calls = []
    monkeypatch.setattr(os, 'posix_fadvise', lambda fd, off, n, advice: calls.append((off, n, advice)))
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        for f in _frames(2):
            w.write(f)
    r = RawVideoReader(path)
    r.read()
    r.read()
    assert calls == [(0, 144, os.POSIX_FADV_DONTNEED), (144, 144, os.POSIX_FADV_DONTNEED)]


def test_frames_are_read_ahead_in_the_background(tmp_path):
    # A camera hands over frames captured in the background; reading 6 MB per frame from disk inside read() made
    # the replay's capture stage 11.6 ms on the Orin against ~3 ms live.
    import time
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        for f in _frames(6):
            w.write(f)
    r = RawVideoReader(path, ahead=3)
    deadline = time.monotonic() + 2
    while r.buffered() < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert r.buffered() == 3
    assert [int(r.read()[0, 0, 0]) for _ in range(6)] == list(range(6))
    with pytest.raises(EOFError):
        r.read()


def test_extra_info_is_saved_with_the_clip(tmp_path):
    # record_video.py stores the light and the camera's gain/exposure log next to the frames.
    import json
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        w.write(_frames(1)[0])
        w.meta['light'] = 'lamps on'
        w.meta['brightness'] = [{'t': 0.0, 'gain': 170, 'exposure_us': 29999, 'luma': 41.2, 'saturated_pct': 0.1}]
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['light'] == 'lamps on' and meta['brightness'][0]['gain'] == 170
    assert meta['count'] == 1 and RawVideoReader(path).count == 1


def test_brightness_of_a_frame():
    from tools.orin_bench.record_video import frame_brightness
    frame = np.zeros((10, 10, 3), np.uint8)
    frame[:5] = 255
    luma, saturated = frame_brightness(frame)
    assert round(luma) == 128 and saturated == 50.0
