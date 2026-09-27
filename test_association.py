from poser import _assign_heads, _claim_hands, _drop_nested_bodies
from src.types import BoundingBox


def _box(x, y, w, h, conf=0.9):
    return BoundingBox(x=x, y=y, w=w, h=h, confidence=conf)


def test_head_inside_body_is_chosen():
    body = _box(100, 100, 200, 400)
    head = _box(150, 110, 60, 60)
    assert _assign_heads([body], [head]) == [head]


def test_head_outside_body_is_not_attached():
    # Issue #3: a black ball far from the body was picked as its head and got a face mesh.
    body = _box(1710, 755, 210, 324)
    ball = _box(1690, 658, 156, 148, conf=0.63)  # centroid (1768, 732) is above the body box
    assert _assign_heads([body], [ball]) == [None]


def test_head_near_top_centre_wins():
    body = _box(0, 0, 500, 1000)
    off = _box(10, 150, 80, 80)
    centred = _box(210, 40, 80, 80)
    assert _assign_heads([body], [off, centred]) == [centred]


def test_false_positive_box_around_a_person_does_not_take_their_head():
    # Issue #6 (Orin, 3 people, ~10 body boxes): a big false-positive box containing a person's head got
    # the head, passed the face check, and was drawn as a person.
    person = _box(600, 300, 250, 700)
    around = _box(300, 0, 900, 1080, conf=0.4)  # head sits in the middle of this box
    head = _box(690, 310, 70, 80)
    assert _assign_heads([around, person], [head]) == [None, head]


def test_each_head_goes_to_one_body():
    a = _box(100, 100, 200, 400)
    b = _box(110, 90, 200, 420)   # duplicate box on the same person
    head = _box(170, 110, 60, 60)
    assert sorted(h is head for h in _assign_heads([a, b], [head])) == [False, True]


def test_body_mostly_inside_higher_scoring_body_is_dropped():
    # Issue #3 (Orin dump 3f frame 91): a 0.31 box on the lamp beside a person, 67% inside the person's box.
    person = _box(1140, 340, 380, 680, conf=0.83)
    lamp = _box(1080, 340, 130, 660, conf=0.31)
    assert _drop_nested_bodies([lamp, person]) == [person]


def test_separate_bodies_are_kept():
    a = _box(100, 100, 200, 400, conf=0.9)
    b = _box(600, 100, 200, 400, conf=0.4)
    assert _drop_nested_bodies([a, b]) == [a, b]


def test_hands_go_to_only_one_body():
    person = _box(100, 100, 200, 400)
    around = _box(50, 50, 400, 500, conf=0.4)
    hands = [_box(90, 250, 40, 40)]
    assert len(_claim_hands(person, hands)) == 1
    assert _claim_hands(around, hands) == []


def test_fit_keeps_aspect_ratio():
    from poser import _fit
    assert _fit(1920, 1080, 2560, 1440) == (2560, 1440)
    assert _fit(1920, 1080, 2560, 1600) == (2560, 1440)


def test_screen_size_prefers_connected_output(monkeypatch):
    import subprocess
    from poser import _screen_size
    out = ('Screen 0: minimum 8 x 8, current 4480 x 1440, maximum 32767 x 32767\n'
           'HDMI-0 connected primary 2560x1440+0+0 (normal left inverted right x axis y axis) 597mm x 336mm\n'
           '   2560x1440     59.95*+\n')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ''))
    assert _screen_size() == (2560, 1440)
    out2 = 'Screen 0: minimum 8 x 8, current 2560 x 1440, maximum 32767 x 32767\n'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, out2, ''))
    assert _screen_size() == (2560, 1440)


def test_head_stays_with_the_older_track_when_both_fit():
    # Bodies come oldest track first; the head must not hop to a newer duplicate box that fits slightly better
    # (that made face colours flicker on the Orin).
    older = _box(100, 100, 220, 400)
    newer = _box(110, 100, 200, 400)
    head = _box(180, 110, 60, 60)
    assert _assign_heads([older, newer], [head]) == [head, None]
