import math

import numpy as np
import pytest

from src.body_processor import BodyProcessor
from src.preprocessor import Preprocessor
from src.types import BoundingBox


def test_roi_from_alignment_upright_person():
    # Hip centre (point 33) at (300, 400), body-size point (34) straight above it, 200 px away.
    cx, cy, size, angle = Preprocessor.roi_from_alignment((300, 400), (300, 200))
    assert (cx, cy) == (300, 400)
    assert size == pytest.approx(2 * 200 * 1.25)
    assert angle == pytest.approx(0.0)


def test_roi_affine_maps_model_centre_and_top_into_the_frame():
    M = Preprocessor.roi_affine(300, 400, 500, 0.0, 256)
    def to_frame(u, v):
        return tuple(M @ np.array([u, v, 1.0]))
    assert to_frame(128, 128) == pytest.approx((300, 400))
    assert to_frame(128, 0) == pytest.approx((300, 150))   # top of the crop is above the hips
    assert to_frame(256, 128) == pytest.approx((550, 400))


def test_roi_for_a_person_lying_sideways_is_rotated():
    # Body-size point to the right of the hips: the person's head points right, so the crop's top maps right.
    cx, cy, size, angle = Preprocessor.roi_from_alignment((300, 400), (500, 400))
    assert angle == pytest.approx(math.pi / 2)
    M = Preprocessor.roi_affine(cx, cy, size, angle, 256)
    x, y = M @ np.array([128, 0, 1.0])
    assert x > cx + 200 and y == pytest.approx(cy)


def test_landmarks_map_back_through_the_roi():
    M = Preprocessor.roi_affine(300, 400, 512, 0.0, 256)
    lms = np.array([[128, 128, 5, 3, 4], [0, 0, 0, 0, 0]], dtype=np.float32)
    out = Preprocessor.to_image_space_affine(lms, M)
    assert (out[0].x, out[0].y, out[1].x, out[1].y) == pytest.approx((300, 400, 44, 144))
    assert (out[0].z, out[0].visibility, out[0].presence) == pytest.approx((10, 3, 4))  # z scaled like x


class _Session:
    def get_providers(self):
        return ['CPUExecutionProvider']

    """Pose model stand-in: hips at the crop centre, body-size point 64 px above, presence 0.9."""
    def __init__(self):
        self.runs = 0

    def run(self, _names, _feeds):
        self.runs += 1
        lms = np.zeros((39, 5), dtype=np.float32)
        lms[:, :2] = 128
        lms[34, 1] = 64
        return [lms.reshape(1, 195), np.array([[0.9]], dtype=np.float32)]


class _Models:
    def __init__(self):
        self.session = _Session()

    def get_session(self, _name):
        return self.session


@pytest.fixture
def body(monkeypatch):
    calls = []
    monkeypatch.setattr(Preprocessor, 'crop_roi_nhwc', staticmethod(
        lambda frame, bbox, w, h, **kw: (calls.append('box') or np.zeros((1, 256, 256, 3), np.float32),
                                         BoundingBox(bbox.x, bbox.y, 256, 256, 1.0))))
    monkeypatch.setattr(Preprocessor, 'crop_rotated_nhwc', staticmethod(
        lambda frame, M, size: calls.append('roi') or np.zeros((1, 256, 256, 3), np.float32)))
    proc = BodyProcessor(_Models())
    return proc, calls


def test_new_person_runs_on_the_box_then_on_the_roi(body):
    proc, calls = body
    r = proc.process(None, BoundingBox(100, 100, 200, 400, 0.9), 640, 480, False)
    assert calls == ['box', 'roi']
    assert r.roi is not None


def test_tracked_person_runs_once_on_the_previous_roi(body):
    proc, calls = body
    r = proc.process(None, BoundingBox(100, 100, 200, 400, 0.9), 640, 480, False, roi=(200, 300, 500, 0.0))
    assert calls == ['roi']
    assert r.roi == pytest.approx((200, 300, 2 * 64 / 256 * 500 * 1.25, 0.0))


def test_previous_roi_far_from_the_box_is_not_used(body):
    proc, calls = body
    proc.process(None, BoundingBox(100, 100, 200, 400, 0.9), 640, 480, False, roi=(2000, 300, 500, 0.0))
    assert calls == ['box', 'roi']


def test_body_points_match_its_landmarks(body):
    proc, _ = body
    r = proc.process(None, BoundingBox(100, 100, 200, 400, 0.9), 640, 480, False)
    assert r.pts.shape == (39, 2)
    assert r.pts[34].tolist() == pytest.approx([r.landmarks[34].x, r.landmarks[34].y])
