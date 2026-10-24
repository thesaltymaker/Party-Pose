"""--face-keep: a face kept last frame stays while its logit is above a lower value and the head is not turned away."""
import numpy as np

from src.face_processor import NIGHT_FACE_HOLD, NIGHT_FACE_NEW, FaceProcessor, face_min_logit, head_turn
from src.types import BoundingBox, Landmark


def _pose(nose_x=500.0, ears=(450.0, 550.0)):
    lms = [Landmark(500, 300, 0, 0.9, 0.9) for _ in range(33)]
    lms[0] = Landmark(nose_x, 300, 0, 0.9, 0.9)
    lms[7] = Landmark(ears[0], 290, 0, 0.9, 0.9)
    lms[8] = Landmark(ears[1], 290, 0, 0.9, 0.9)
    return lms


def test_head_turn_facing_and_side_on():
    assert head_turn(_pose()) == 0.0                      # nose between the ears
    assert abs(head_turn(_pose(nose_x=540)) - 0.4) < 1e-9  # nose 0.4 ear distances off the middle
    assert abs(head_turn(_pose(nose_x=440)) + 0.6) < 1e-9
    assert head_turn(_pose(ears=(500, 500.5))) == 0.0      # ears on top of each other: no value


def test_new_face_needs_logit_zero():
    assert face_min_logit(_pose(), False, -8.0) == 0.0


def test_held_face_keeps_down_to_hold_logit():
    assert face_min_logit(_pose(), True, -8.0) == -8.0


def test_turned_head_needs_logit_zero_again():
    assert face_min_logit(_pose(nose_x=545), True, -8.0) == 0.0  # side on or the back of the head


def test_hold_zero_or_no_pose_is_off():
    assert face_min_logit(_pose(), True, 0.0) == 0.0
    assert face_min_logit(None, True, -8.0) == 0.0


def test_result_uses_min_logit():
    fp = FaceProcessor(None)
    box = BoundingBox(0, 0, 256, 256, 1.0)
    raw = np.zeros((478, 3), np.float32)
    assert fp._result(box, box, raw, -3.0, 640, 480, False) is None             # presence 0.05 < 0.5
    assert fp._result(box, box, raw, -3.0, 640, 480, False, -8.0) is not None    # held face
    assert fp._result(box, box, raw, -9.0, 640, 480, False, -8.0) is None
    assert fp._result(box, box, raw, 0.5, 640, 480, False, 0.0) is not None


def test_night_new_face_needs_night_logit():
    assert face_min_logit(_pose(), False, -8.0, night=True) == NIGHT_FACE_NEW


def test_night_held_face_keeps_down_to_night_hold():
    assert face_min_logit(_pose(), True, -8.0, night=True) == NIGHT_FACE_HOLD


def test_night_turned_head_needs_night_new_logit():
    assert face_min_logit(_pose(nose_x=545), True, -8.0, night=True) == NIGHT_FACE_NEW


def test_night_without_pose_uses_night_new_logit():
    assert face_min_logit(None, True, -8.0, night=True) == NIGHT_FACE_NEW
