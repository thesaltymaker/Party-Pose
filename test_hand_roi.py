import numpy as np
import pytest

from src.hand_processor import HandProcessor, hand_roi_from_landmarks, hand_roi_from_pose
from src.preprocessor import Preprocessor
from src.types import Landmark


def _pose(wrist=(100, 200), index=(100, 100), pinky=(100, 100), side=0, visibility=5.0):
    lms = [Landmark(0.0, 0.0, 0.0, visibility=-5.0) for _ in range(39)]
    w, p, i = (15, 17, 19) if side == 0 else (16, 18, 20)
    lms[w] = Landmark(*wrist, 0.0, visibility=visibility)
    lms[p] = Landmark(*pinky, 0.0)
    lms[i] = Landmark(*index, 0.0)
    return lms


def test_hand_roi_from_pose_points_up_from_the_wrist():
    # Hand centre 2/3 of the way to the index from the pinky: (100, 100), 100 px above the wrist.
    cx, cy, size, angle = hand_roi_from_pose(_pose(), 0)
    assert angle == pytest.approx(0.0)
    assert (cx, cy) == pytest.approx((100, 80))       # shifted 0.1 x 200 towards the fingers
    assert size == pytest.approx(200 * 2.0)  # 2.7 (Holistic, which then re-crops) left the hand too small


def test_hand_roi_from_landmarks_is_square_around_the_palm():
    pts = np.zeros((21, 2))
    pts[[0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]] = [(100, 200), (80, 190), (70, 170), (60, 150), (80, 140),
                                                      (80, 120), (100, 100), (100, 80), (120, 140), (120, 120),
                                                      (140, 150), (140, 130)]
    cx, cy, size, angle = hand_roi_from_landmarks(pts)
    assert angle == pytest.approx(0.0)
    assert size == pytest.approx(120 * 2.0)            # taller side (80..200) x 2
    assert (cx, cy) == pytest.approx((100, 140 - 12))  # box centre, shifted 0.1 x 120 towards the fingers


class _Session:
    def get_providers(self):
        return ['CPUExecutionProvider']

    """Hand model stand-in. Outputs in the model's order: landmarks, presence (Identity_1), handedness (Identity_2)."""
    def __init__(self, presences):
        self.presences = list(presences)
        self.runs = 0

    def run(self, _names, _feeds):
        self.runs += 1
        lms = np.zeros((1, 63), np.float32)
        lms[0, 0::3] = 112
        lms[0, 1::3] = 112
        p = self.presences.pop(0) if self.presences else 0.9
        return [lms, np.array([[p]], np.float32), np.array([[0.1]], np.float32)]


class _Models:
    def __init__(self, session):
        self.session = session

    def get_session(self, _name):
        return self.session


@pytest.fixture
def crops(monkeypatch):
    rois = []
    monkeypatch.setattr(Preprocessor, 'crop_rotated_nhwc', staticmethod(
        lambda frame, M, size: rois.append(tuple(np.round(M @ [size / 2, size / 2, 1]))) or np.zeros((1, size, size, 3))))
    return rois


def test_hand_is_kept_by_its_presence_score_not_its_handedness(crops):
    # Bug: Identity_2 (handedness) was read as presence, so one side's hands were dropped.
    proc = HandProcessor(_Models(_Session([0.9])))
    left, right = proc.process(None, _pose(), [None, None])
    assert left is not None and left.presence == pytest.approx(0.9)


def test_previous_hand_roi_is_used_first(crops):
    proc = HandProcessor(_Models(_Session([])))
    proc.process(None, _pose(), [(300, 300, 100, 0.0), None])
    assert crops[0] == (300, 300)


def test_lost_previous_hand_is_cropped_from_the_pose_on_the_next_frame(crops):
    # One hand-model run per hand per frame: a miss at the previous ROI is not retried in the same frame.
    session = _Session([0.1])
    proc = HandProcessor(_Models(session))
    left, _ = proc.process(None, _pose(), [(300, 300, 100, 0.0), None])
    assert left is None and session.runs == 1
    left, _ = proc.process(None, _pose(), [None, None])
    assert crops == [(300, 300), (100, 80)]
    assert left is not None and left.roi is not None


def test_hand_points_match_its_landmarks(crops):
    left, _ = HandProcessor(_Models(_Session([]))).process(None, _pose(), [None, None])
    assert left.pts.shape == (21, 2)
    assert left.pts[0].tolist() == pytest.approx([left.landmarks[0].x, left.landmarks[0].y])


def test_hidden_wrist_without_a_previous_hand_is_skipped(crops):
    session = _Session([])
    proc = HandProcessor(_Models(session))
    assert proc.process(None, _pose(visibility=-3.0), [None, None]) == [None, None]
    assert session.runs == 0
