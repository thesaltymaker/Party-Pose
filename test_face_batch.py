import numpy as np
import pytest

from src.face_processor import FaceProcessor
from src.preprocessor import Preprocessor
from src.types import BoundingBox


class _Session:
    """Face model stand-in: logit +5 for even crop ids, -5 for odd ones (id = the crop's fill value)."""
    def __init__(self):
        self.batches = []

    def run(self, _names, feeds):
        x = feeds['input_12']
        self.batches.append(len(x))
        ids = x[:, 0, 0, 0].astype(int)
        lms = np.zeros((len(x), 1, 1, 1434), np.float32)
        lms[:, 0, 0, 0::3] = 128
        lms[:, 0, 0, 1::3] = 128
        logit = np.where(ids % 2 == 0, 5.0, -5.0).astype(np.float32).reshape(-1, 1, 1, 1)
        return [lms, logit]


class _Models:
    def __init__(self):
        self.session = _Session()

    def get_session(self, _name):
        return self.session


@pytest.fixture
def proc(monkeypatch):
    monkeypatch.setattr(Preprocessor, 'crop_roi_nhwc', staticmethod(
        lambda frame, bbox, w, h, **kw: (np.full((1, h, w, 3), bbox.confidence, np.float32),
                                         BoundingBox(bbox.x, bbox.y, 256, 256, 1.0))))
    return FaceProcessor(_Models())


def _heads(n):
    return [BoundingBox(100 * i, 50, 80, 80, float(i)) for i in range(n)]


def test_faces_of_two_people_run_as_one_batch(proc):
    faces = proc.process_many(None, _heads(2), 1920, 1080, False)
    assert proc.model_manager.session.batches == [2]
    assert faces[0] is not None and faces[1] is None          # odd crop id scored -5
    assert proc.last_logits == [5.0, -5.0]
    assert faces[0].landmarks[0].x == pytest.approx(0 + 128)  # mapped through its own crop
    assert faces[0].pts.shape == (478, 2)


def test_more_than_four_faces_are_split_into_batches_of_four(proc):
    faces = proc.process_many(None, _heads(5), 1920, 1080, False)
    assert proc.model_manager.session.batches == [4, 1]
    assert [f is not None for f in faces] == [True, False, True, False, True]
    assert faces[4].landmarks[0].x == pytest.approx(400 + 128)


def test_no_heads_runs_nothing(proc):
    assert proc.process_many(None, [], 1920, 1080, False) == []
    assert proc.model_manager.session.batches == []


def test_single_face_still_works(proc):
    face = proc.process(None, _heads(1)[0], 1920, 1080, False)
    assert face is not None and proc.last_logit == 5.0
