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


def test_smoothing_starts_from_the_processor_points_when_given():
    # Processors pass their [N, 2] point array, so smoothing doesn't rebuild it from Landmark objects.
    from src.types import BodyResult
    body = BodyResult(landmarks=[Landmark(0.0, 0.0, 7.0) for _ in range(3)], person_id=1)
    body.pts = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    PoseSmoother().smooth([], [], [body], 0.0)
    assert [(lm.x, lm.y, lm.z) for lm in body.landmarks] == [(10, 20, 7), (30, 40, 7), (50, 60, 7)]
