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
