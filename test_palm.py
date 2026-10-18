"""A lost hand is found again with MediaPipe's palm detector on the person's upper body."""
import math

import numpy as np

from src.palm_detector import Palm, assign_palms, decode_palms, palm_anchors, palm_roi, upper_body_roi
from src.preprocessor import Preprocessor
from src.types import Landmark


def test_anchors_are_mediapipes_palm_grid():
    a = palm_anchors()
    assert a.shape == (2016, 2)
    # 24x24 cells of 8 px, 2 anchors each, then 12x12 cells of 16 px, 6 anchors each; centres in model pixels
    assert np.allclose(a[0], [4, 4]) and np.allclose(a[1], [4, 4]) and np.allclose(a[2], [12, 4])
    assert np.allclose(a[24 * 2], [4, 12])
    assert np.allclose(a[1152], [8, 8]) and np.allclose(a[1157], [8, 8]) and np.allclose(a[1158], [24, 8])
    assert np.allclose(a[-1], [184, 184])


def _raw(k, box=(0, 0, 20, 30), wrist=(0, 10), middle=(0, -10), logit=3.0):
    boxes = np.zeros((2016, 18), np.float32)
    boxes[k, :4] = box
    boxes[k, 4:6] = wrist
    boxes[k, 8:10] = middle  # keypoint 2: middle finger base
    logits = np.full(2016, -10, np.float32)
    logits[k] = logit
    return boxes, logits


def test_decode_maps_the_palm_through_the_crop():
    # crop: 384 px square centred at (500, 400), unrotated -> 2 frame px per model px
    M = Preprocessor.roi_affine(500, 400, 384, 0, 192)
    k = 1152 + (6 * 12 + 6) * 6  # stride-16 cell (6, 6): centre (104, 104) model px
    palms = decode_palms(*_raw(k), M)
    assert len(palms) == 1
    p = palms[0]
    assert abs(p.score - 1 / (1 + math.exp(-3))) < 1e-6
    assert np.allclose(p.centre, [308 + 208, 208 + 208])
    assert np.allclose((p.w, p.h), (40, 60))
    assert np.allclose(p.wrist, [516, 436]) and np.allclose(p.middle, [516, 396])


def test_decode_drops_low_scores_and_overlaps():
    M = Preprocessor.roi_affine(96, 96, 192, 0, 192)
    boxes, logits = _raw(1152, box=(0, 0, 40, 40))
    boxes[1153], logits[1153] = boxes[1152] + np.r_[1, 1, [0] * 16], 2.0  # the same palm, a bit lower score
    boxes[0, 2:4], logits[0] = (10, 10), -1.0  # below 0.5
    palms = decode_palms(boxes, logits, M)
    assert len(palms) == 1 and abs(palms[0].score - 1 / (1 + math.exp(-3))) < 1e-6


def test_palm_roi_is_mediapipes_hand_crop():
    # wrist straight below the middle finger base: upright; 2.6 x the longer side, moved half a box up
    p = Palm(0.9, np.array([100.0, 200.0]), 40.0, 60.0, np.array([100.0, 220.0]), np.array([100.0, 180.0]))
    cx, cy, size, angle = palm_roi(p)
    assert abs(angle) < 1e-9 and abs(size - 156) < 1e-9
    assert abs(cx - 100) < 1e-9 and abs(cy - 170) < 1e-9


def _pose(l_sh=(500, 200), r_sh=(700, 200), l_hip=(520, 500), r_hip=(680, 500), l_el=(450, 350), r_el=(750, 350),
          l_wr=(480, 450), r_wr=(720, 450)):
    lms = [Landmark(0, 0, 0, 0.9, 0.9) for _ in range(33)]
    for k, (x, y) in ((11, l_sh), (12, r_sh), (23, l_hip), (24, r_hip), (13, l_el), (14, r_el), (15, l_wr), (16, r_wr)):
        lms[k] = Landmark(x, y, 0, 0.9, 0.9)
    return lms


def test_upper_body_crop_is_sized_from_the_shoulders():
    cx, cy, size, angle = upper_body_roi(_pose(), 3.0)
    assert (cx, angle) == (600, 0) and size == 600
    assert 200 < cy < 500  # between the shoulders and the hips


def test_upper_body_crop_keeps_a_size_when_side_on():
    # shoulders 20 px apart (side on): sized from half the shoulder-hip distance instead
    _, _, size, _ = upper_body_roi(_pose(l_sh=(590, 200), r_sh=(610, 200), l_hip=(590, 500), r_hip=(610, 500)), 3.0)
    assert size == 450


def _palm(wx, wy, score=0.9):
    return Palm(score, np.array([wx, wy - 20.0]), 40.0, 40.0, np.array([wx, wy]), np.array([wx, wy - 40.0]))


def test_palms_go_to_the_nearest_arm():
    # a palm just past each pose wrist; only the lost sides take one
    pose = _pose()
    left, right = _palm(485, 500), _palm(715, 500)
    assert assign_palms([right, left], pose, (0, 1)) == {0: left, 1: right}
    assert assign_palms([right, left], pose, (1,)) == {1: right}


def test_palm_on_the_forearm_line_goes_to_that_side():
    # arms across the body: the pose wrist lies up the forearm; the palm sits past it on the elbow-wrist line
    pose = _pose(l_el=(450, 300), l_wr=(560, 350))
    p = _palm(640, 390)
    assert assign_palms([p], pose, (0, 1)) == {0: p}


def test_palms_far_from_both_arms_are_dropped():
    # another person's hand 300 px from this person's arms
    assert assign_palms([_palm(1050, 450)], _pose(), (0, 1)) == {}


def test_one_palm_per_side():
    pose = _pose()
    near, nearer = _palm(470, 500, 0.8), _palm(482, 470, 0.95)
    assert assign_palms([near, nearer], pose, (0,)) == {0: nearer}
