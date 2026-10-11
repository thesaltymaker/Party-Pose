import random

import numpy as np

from src.smoothing import PoseSmoother, Smoother
from src.types import BoundingBox, HandResult, Landmark

FPS = 26.0


def _noisy_still(smoother, key, n=120, noise=4.0, scale=100.0):
    rng = random.Random(1)
    out = []
    for f in range(n):
        x = np.array([[500 + rng.gauss(0, noise), 300 + rng.gauss(0, noise)]])
        out.append(smoother(key, x, f / FPS, scale)[0, 0])
    return np.array(out[20:])


def test_still_point_jitter_is_reduced():
    raw_std = 4.0
    out = _noisy_still(Smoother(), 'a')
    assert out.std() < raw_std / 3


def test_fast_movement_has_little_lag():
    # A dancer's hand moving 2 box heights per second (200 px/s at scale 100).
    s = Smoother()
    for f in range(60):
        x = np.array([[100 + 200 * f / FPS, 300.0]])
        y = s('a', x, f / FPS, 100.0)
    assert abs(y[0, 0] - x[0, 0]) < 10  # < 0.1 box height behind


def test_new_key_and_gap_restart_the_filter():
    s = Smoother(max_gap=0.5)
    s('a', np.array([[0.0, 0.0]]), 0.0, 100.0)
    # A different key starts from its own first value.
    assert s('b', np.array([[500.0, 500.0]]), 0.04, 100.0)[0, 0] == 500.0
    # The same key after a long gap (person lost) jumps straight to the new value.
    assert s('a', np.array([[300.0, 0.0]]), 2.0, 100.0)[0, 0] == 300.0


def _hand(wx, wy):
    lms = [Landmark(wx + i, wy + i, 0.0) for i in range(21)]
    return HandResult(bbox=BoundingBox(wx, wy, 50, 50, 1.0), landmarks=lms, person_id=7)


def test_hands_keep_their_filter_when_handedness_flips():
    ps = PoseSmoother()
    for f in range(30):
        left, right = _hand(100, 400), _hand(600, 400)
        left.handedness, right.handedness = ('Left', 'Right') if f % 2 else ('Right', 'Left')
        hands = [left, right] if f % 3 else [right, left]
        ps.smooth([], hands, [], f / FPS)
        # Each hand stays near its own place; a mixed-up filter would pull it toward the other hand.
        assert abs(left.landmarks[0].x - 100) < 5
        assert abs(right.landmarks[0].x - 600) < 5


def _covers(outer, inner):
    return (outer.x <= inner.x + 1e-6 and outer.y <= inner.y + 1e-6
            and outer.x + outer.w >= inner.x + inner.w - 1e-6 and outer.y + outer.h >= inner.y + inner.h - 1e-6)


def test_crop_stays_steady_when_the_box_flips_for_several_frames():
    # Orin: one person's box flips between 117x239 and 220x510, each size held for a few frames.
    rng = random.Random(2)
    small, big = BoundingBox(1540, 433, 117, 239, 0.4), BoundingBox(1512, 421, 220, 510, 0.4)
    ps, crops, f, use_big = PoseSmoother(), [], 0, True
    while f < 200:
        for _ in range(rng.randint(3, 6)):
            box = big if use_big else small
            crop = ps.crop(1, box, f / FPS)
            assert _covers(crop, box)
            crops.append((crop.x, crop.y, crop.w, crop.h))
            f += 1
        use_big = not use_big
    crops = np.array(crops[20:])
    assert (crops.max(axis=0) - crops.min(axis=0)).max() < 1e-6  # crop never changes


def test_crop_always_covers_a_fast_dancer_and_shrinks_back():
    ps = PoseSmoother()
    for f in range(40):
        box = BoundingBox(100 + 15 * f, 300, 200, 500, 0.8)  # 390 px/s sideways, ~0.8 box heights per second
        assert _covers(ps.crop(1, box, f / FPS), box)
    # Arms up: the box grows at once, the crop grows with it.
    grown = BoundingBox(600, 200, 400, 600, 0.8)
    assert _covers(ps.crop(1, grown, 40 / FPS), grown)
    # Standing still in a small box for a second: the crop shrinks back to it.
    still = BoundingBox(700, 300, 200, 500, 0.8)
    for f in range(41, 70):
        crop = ps.crop(1, still, f / FPS)
    assert (crop.x, crop.y, crop.w, crop.h) == (700, 300, 200, 500)
