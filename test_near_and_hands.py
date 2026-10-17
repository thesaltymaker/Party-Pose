"""Only people near the camera are drawn, and a hand is kept only on its own person's wrist."""
from unittest.mock import patch

import numpy as np

from poser import _big_enough, _hand_on_own_wrist
from src.types import BoundingBox, HandResult, Landmark


def _tall(h):
    return BoundingBox(100, 0, 200, h, 0.9)


def test_far_people_are_too_small():
    # Orin, frame 1080 high: dancers 0.56-0.89, a child near the camera 0.47, people at the back below 0.4.
    assert _big_enough(_tall(0.56 * 1080), 1080, 0.40)
    assert _big_enough(_tall(0.47 * 1080), 1080, 0.40)
    assert not _big_enough(_tall(0.30 * 1080), 1080, 0.40)


def test_min_body_height_zero_is_off():
    assert _big_enough(_tall(5), 1080, 0.0)


def _pose(l_wrist=(400, 500), r_wrist=(800, 500), l_sh=(500, 200), r_sh=(700, 200)):
    lms = [Landmark(0, 0, 0, 0.9, 0.9) for _ in range(33)]
    for k, (x, y) in ((15, l_wrist), (16, r_wrist), (11, l_sh), (12, r_sh)):
        lms[k] = Landmark(x, y, 0, 0.9, 0.9)
    return lms


def _hand(wx, wy):
    pts = np.array([[wx, wy]] + [[wx + 5, wy - 5 * i] for i in range(1, 21)], np.float32)
    lms = [Landmark(float(x), float(y), 0, 1, 1) for x, y in pts]
    return HandResult(bbox=BoundingBox(wx, wy - 100, 10, 100, 0.9), landmarks=lms, pts=pts)


def test_hand_on_own_wrist_is_kept():
    # shoulders 200 px apart: a hand wrist up to factor x 200 px from the pose wrist is this person's
    assert _hand_on_own_wrist(_hand(430, 520), _pose(), 0, 0.5)
    assert _hand_on_own_wrist(_hand(830, 480), _pose(), 1, 0.5)


def test_hand_on_another_persons_wrist_is_rejected():
    # the left-side crop chained onto someone else's hand 300 px away
    assert not _hand_on_own_wrist(_hand(700, 500), _pose(), 0, 0.5)
    assert not _hand_on_own_wrist(_hand(400, 500), _pose(), 1, 0.5)


def test_foreshortened_shoulders_keep_a_floor():
    # side-on: shoulders 2 px apart, the hand 8 px from the pose wrist is still kept
    pose = _pose(l_sh=(600, 200), r_sh=(602, 200))
    assert _hand_on_own_wrist(_hand(408, 500), pose, 0, 0.5)
    assert not _hand_on_own_wrist(_hand(450, 500), pose, 0, 0.5)


def test_min_body_height_flag():
    from src.config import parse_args
    with patch("sys.argv", ["poser.py"]):
        assert parse_args().min_body_height == 0.40
    with patch("sys.argv", ["poser.py", "--min-body-height", "0"]):
        assert parse_args().min_body_height == 0.0
