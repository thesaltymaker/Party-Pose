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


def test_report_shows_10s_spread_and_strong_faces():
    # A prop's box stays put (only detector jitter); a person's drifts over seconds.
    import math
    ts = TrackStats()
    for f in range(300):
        j = (f % 3) - 1  # +-1 px jitter
        ts.update(6, BoundingBox(669 + j, 423, 128, 241, 0.5), False, 1.0, None)
        ts.update(7, BoundingBox(700 + 60 * math.sin(f / 30), 500, 200, 330, 0.7), True,
                  12.0 if f % 10 == 0 else 2.0, None)
    lamp, person = ts.report({6, 7})
    assert 'strong_faces=0 ' in lamp and 'strong_faces=30 ' in person
    lamp_spread = float(lamp.split('spread10s=')[1].split()[0])
    person_spread = float(person.split('spread10s=')[1].split()[0])
    assert lamp_spread < 0.01 < 0.1 < person_spread


def test_report_shows_still_decision():
    ts = TrackStats()
    for _ in range(20):
        ts.update(4, BoundingBox(0, 0, 50, 100, 0.3), True, None, None, still=True)
        ts.update(5, BoundingBox(200, 0, 50, 100, 0.3), True, None, None, still=False)
    lamp, dancer = ts.report({4, 5})
    assert lamp.endswith(' still=yes') and dancer.endswith(' still=no')


def test_report_shows_share_of_kept_faces():
    ts = TrackStats()
    for f in range(20):
        ts.update(8, BoundingBox(0, 0, 50, 100, 0.5), True, -3.0 if f % 4 else 2.0, None)
    [line] = ts.report({8})
    assert 'kept=25%' in line and 'logit_p25=-3.0' in line
