"""--detect-ahead: the next frame is read and the person detector run on it in another thread, one frame ahead."""
import threading
import time

import numpy as np
import pytest

from poser import _Ahead
from src import gpu_model


def test_items_come_in_order():
    n = iter(range(100))
    ahead = _Ahead(lambda: next(n))
    try:
        assert [ahead.get() for _ in range(10)] == list(range(10))
    finally:
        ahead.close()


def test_next_item_is_made_while_the_caller_works_on_this_one():
    made_during_work = []
    busy = threading.Event()

    def produce():
        made_during_work.append(busy.is_set())
        return 0

    ahead = _Ahead(produce)
    try:
        for _ in range(5):
            ahead.get()
            busy.set()
            time.sleep(0.02)
            busy.clear()
    finally:
        ahead.close()
    assert any(made_during_work)


def test_error_in_produce_is_raised_by_get():
    def produce():
        raise EOFError('end of clip')

    ahead = _Ahead(produce)
    try:
        with pytest.raises(EOFError):
            ahead.get()
    finally:
        ahead.close()


def test_thread_stops_on_close():
    ahead = _Ahead(lambda: 0)
    ahead.get()
    ahead.close()
    assert not ahead._thread.is_alive()


class _Session:
    def __init__(self):
        self.locked = []

    def run_with_iobinding(self, binding):
        self.locked.append(gpu_model.FIRST_RUNS_LOCK.locked())


def test_first_runs_of_a_session_hold_the_lock_later_runs_do_not():
    # A CUDA graph is recorded in a session's first runs; no other thread may allocate GPU memory meanwhile.
    s = _Session()
    for _ in range(gpu_model.FIRST_RUNS + 2):
        gpu_model.run_bound(s, None)
    assert s.locked == [True] * gpu_model.FIRST_RUNS + [False, False]


def test_each_session_counts_its_own_first_runs():
    a, b = _Session(), _Session()
    for _ in range(gpu_model.FIRST_RUNS):
        gpu_model.run_bound(a, None)
    gpu_model.run_bound(b, None)
    assert b.locked == [True]


def test_capture_can_keep_its_last_frames_in_separate_gpu_buffers(tmp_path):
    # --detect-ahead: up to 6 frames are in use at once (read ahead, queued, models, drawing); no copy per frame.
    from types import SimpleNamespace
    from src.raw_video import RawVideoWriter
    from src.video_capture import VideoCaptureModule
    path = tmp_path / 'clip'
    with RawVideoWriter(path, 8, 6) as w:
        for i in range(4):
            w.write(np.full((6, 8, 3), i, np.uint8))
    cap = VideoCaptureModule(SimpleNamespace(platform='laptop', camera=0, mirror=False, video=str(path)))
    cap.keep_frames(3)
    frames = [cap.read_frame() for _ in range(3)]
    assert [int(f.download()[0, 0, 0]) for f in frames] == [0, 1, 2]
    ptrs = [f.cudaPtr() for f in frames]
    assert len(set(ptrs)) == 3
    assert cap.read_frame().cudaPtr() == ptrs[0]  # the oldest buffer is reused
    cap.release()
