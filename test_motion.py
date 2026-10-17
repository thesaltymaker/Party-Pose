import numpy as np

from src.motion import MotionStats
from src.types import BodyResult, HandResult, BoundingBox, Landmark

W, H = 1920, 1080


def _body(dx=0.0, noise=None, presence=None, visibility=None, raw_dx=None, fade=1.0, scale=1.0):
    """A standing body ~500 px tall at x 900-1000; dx moves it, noise[i] adds to landmark i."""
    pts = np.array([[900 + (i % 5) * 25 * scale, 400 + i * 15 * scale] for i in range(33)], float)
    pts[:, 0] += dx
    if noise is not None:
        pts += noise
    raw = pts.copy()
    if raw_dx is not None:
        raw[:, 0] += raw_dx - dx
    lms = [Landmark(x, y, 0.0) for x, y in pts]
    for i, p in (presence or {}).items():
        lms[i].presence = p
    for i, v in (visibility or {}).items():
        lms[i].visibility = v
    return BodyResult(landmarks=lms, presence=1.0, person_id=1, pts=raw, fade=fade)


def _run(frames):
    ms = MotionStats()
    for drawn in frames:
        ms.update(drawn, W, H)
    return ms


def test_a_joint_hidden_for_one_frame_pops_twice():
    k = (1, 'body')
    ms = _run([{k: _body()}, {k: _body(presence={13: 0.2})}, {k: _body()}])
    assert ms.pops['body'] == 2  # gone, back
    assert ms.conn_pops['body'] == 4  # 11-13 and 13-15, gone and back
    assert ms.pairs['body'] == 2


def test_a_joint_turning_dim_and_back_is_a_dim_flip_not_a_pop():
    k = (1, 'body')
    ms = _run([{k: _body()}, {k: _body(visibility={13: 0.2})}, {k: _body()}])
    assert ms.pops['body'] == 0
    assert ms.dim_flips['body'] == 2


def test_skipped_face_and_finger_points_of_the_body_never_pop():
    k = (1, 'body')
    ms = _run([{k: _body()}, {k: _body(presence={0: 0.1, 18: 0.1})}, {k: _body()}])
    assert ms.pops['body'] == 0


def test_a_whole_part_leaving_is_not_a_joint_pop():
    k = (1, 'body')
    ms = _run([{k: _body()}, {}, {k: _body()}])
    assert ms.pops['body'] == 0 and ms.pairs['body'] == 0


def test_a_hand_point_leaving_the_frame_pops():
    k = (1, 'hand0')
    lms = lambda x0: [Landmark(x0 + i * 5, 500.0, 0.0) for i in range(21)]
    hand = lambda x0: HandResult(bbox=BoundingBox(0, 0, 1, 1, 1.0), landmarks=lms(x0), person_id=1)
    ms = _run([{k: hand(1900)}, {k: hand(1950)}])
    assert ms.pops['hand'] > 0


def test_still_body_jitter_is_measured_and_a_slow_rigid_walk_has_none():
    k = (1, 'body')
    rng = np.random.default_rng(0)
    shaky = _run([{k: _body(noise=rng.normal(0, 5, (33, 2)))} for _ in range(30)])
    walk = _run([{k: _body(dx=2.0 * f)} for f in range(30)])  # 2 px/frame: slow
    assert shaky.jitter('body') > 3  # per mille of body height per frame
    assert walk.jitter('body') < 0.01
    assert shaky.raw_jitter('body') > 3  # raw equals drawn here (no smoothing)


def test_lag_in_frames_for_fast_movement():
    k = (1, 'body')
    # Moving 40 px/frame (fast); drawn 3 frames (120 px) behind the raw landmarks.
    ms = _run([{k: _body(dx=40.0 * f, raw_dx=40.0 * f + 120)} for f in range(20)])
    assert abs(ms.lag('body') - 3.0) < 0.2


def test_ghosts_count_for_pops_but_not_for_jitter_or_lag():
    k = (1, 'body')
    ms = _run([{k: _body()}, {k: _body(fade=0.8)}, {k: _body(fade=0.6)}])
    assert ms.pairs['body'] == 2
    assert ms.jitter('body') is None and ms.lag('body') is None


def test_report_line_starts_with_motion():
    k = (1, 'body')
    line = _run([{k: _body()}, {k: _body(presence={13: 0.2})}]).report()
    assert line.startswith('[MOTION] body: pairs=1 pops/100=100.0 dim/100=0.0 conn/100=200.0')
