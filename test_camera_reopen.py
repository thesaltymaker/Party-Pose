from types import SimpleNamespace

import numpy as np
import pytest

import src.video_capture as vc


class FakeCapture:
    """cv2.VideoCapture stand-in: each open takes the next list of read results (a frame, or None for a failed read)."""
    opens = []

    def __init__(self, *args):
        self.args = args
        self.reads = FakeCapture.opens.pop(0)
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {vc.cv2.CAP_PROP_FRAME_WIDTH: 8, vc.cv2.CAP_PROP_FRAME_HEIGHT: 6}[prop]

    def read(self):
        frame = self.reads.pop(0) if self.reads else None
        return frame is not None, frame

    def release(self):
        self.released = True


def _frame(value):
    return np.full((6, 8, 3), value, np.uint8)


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(vc.cv2, 'VideoCapture', FakeCapture)
    monkeypatch.setattr(vc.time, 'sleep', lambda s: None)

    def make(*opens):
        FakeCapture.opens = list(opens)
        return vc.VideoCaptureModule(SimpleNamespace(video=None, platform='laptop', camera=0, mirror=False))
    return make


def test_failed_read_opens_the_camera_again(capture):
    cap = capture([_frame(1), None], [_frame(2)])
    first = cap._cap
    assert (cap.read_frame().download() == 1).all()
    assert (cap.read_frame().download() == 2).all()
    assert first.released and cap._cap is not first and cap._cap.args == (0,)


def test_gives_up_after_the_tries(capture):
    cap = capture([None], *[[None]] * vc.VideoCaptureModule.REOPEN_TRIES)
    with pytest.raises(RuntimeError, match='Camera read failed'):
        cap.read_frame()
    assert FakeCapture.opens == []
