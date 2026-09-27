from src.track_stats import TrackStats
from src.types import BodyResult, BoundingBox, Landmark


def _body(arm_x):
    lms = [Landmark(100.0, 100.0, 0.0) for _ in range(33)]
    lms[15] = Landmark(arm_x, 200.0, 0.0)  # left wrist
    return BodyResult(landmarks=lms, presence=0.8)


def test_static_prop_and_moving_person():
    ts = TrackStats()
    for f in range(30):
        ts.update(1, BoundingBox(10, 10, 100, 400, 0.4), False, None, _body(100.0))            # lamp
        ts.update(2, BoundingBox(500, 10, 100, 400, 0.5), True, 9.0, _body(100.0 + (f % 2) * 40))  # waving
    lines = ts.report({1, 2})
    assert len(lines) == 2
    lamp, person = lines
    assert lamp.startswith('[TRACK] #1 ') and 'box_move=0.0000 limb_move=0.0000' in lamp and 'faces=0' in lamp
    assert person.startswith('[TRACK] #2 ') and 'limb_move=0.0000' not in person and 'logit_med=9.0' in person
    assert 'head=100%' in person


def test_report_skips_short_and_forgets_expired_tracks():
    ts = TrackStats()
    for _ in range(5):
        ts.update(3, BoundingBox(0, 0, 50, 100, 0.3), False, None, None)
    assert ts.report({3}) == []          # too few frames
    assert ts.report(set()) == []        # expired: forgotten
    for _ in range(20):
        ts.update(3, BoundingBox(0, 0, 50, 100, 0.3), False, None, None)
    assert ts.report({3})[0].startswith('[TRACK] #3 frames=20 ')


def test_report_shows_head_filter_decision():
    ts = TrackStats()
    for _ in range(20):
        ts.update(4, BoundingBox(0, 0, 50, 100, 0.3), True, None, None, head_rate=0.8, head_ok=True)
        ts.update(5, BoundingBox(200, 0, 50, 100, 0.3), False, None, None, head_rate=0.1, head_ok=False)
    shown, hidden = ts.report({4, 5})
    assert 'head2s=80% show=yes' in shown
    assert 'head2s=10% show=no' in hidden
