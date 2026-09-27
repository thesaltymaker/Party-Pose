from poser import _find_head_for_body, _drop_nested_bodies
from src.types import BoundingBox


def _box(x, y, w, h, conf=0.9):
    return BoundingBox(x=x, y=y, w=w, h=h, confidence=conf)


def test_head_inside_body_is_chosen():
    body = _box(100, 100, 200, 400)
    head = _box(150, 110, 60, 60)
    assert _find_head_for_body(body, [head]) is head


def test_head_outside_body_is_not_attached():
    # Issue #3: a black ball far from the body was picked as its head and got a face mesh.
    body = _box(1710, 755, 210, 324)
    ball = _box(1690, 658, 156, 148, conf=0.63)  # centroid (1768, 732) is above the body box
    assert _find_head_for_body(body, [ball]) is None


def test_highest_confidence_head_inside_wins():
    body = _box(0, 0, 500, 500)
    low = _box(10, 10, 50, 50, conf=0.4)
    high = _box(100, 10, 50, 50, conf=0.8)
    assert _find_head_for_body(body, [low, high]) is high


def test_body_mostly_inside_higher_scoring_body_is_dropped():
    # Issue #3 (Orin dump 3f frame 91): a 0.31 box on the lamp beside a person, 67% inside the person's box.
    person = _box(1140, 340, 380, 680, conf=0.83)
    lamp = _box(1080, 340, 130, 660, conf=0.31)
    assert _drop_nested_bodies([lamp, person]) == [person]


def test_separate_bodies_are_kept():
    a = _box(100, 100, 200, 400, conf=0.9)
    b = _box(600, 100, 200, 400, conf=0.4)
    assert _drop_nested_bodies([a, b]) == [a, b]


def test_head_and_hands_go_to_only_one_body():
    # Issue #6: a head inside both a person's box and a false-positive box around them got two face meshes.
    from poser import _claim_parts
    person = _box(100, 100, 200, 400)
    around = _box(50, 50, 400, 500, conf=0.4)
    heads = [_box(150, 110, 60, 60)]
    hands = [_box(90, 250, 40, 40)]
    head, own_hands = _claim_parts(person, heads, hands)
    assert head is not None and len(own_hands) == 1
    assert _claim_parts(around, heads, hands) == (None, [])


def test_fit_keeps_aspect_ratio():
    from poser import _fit
    assert _fit(1920, 1080, 2560, 1440) == (2560, 1440)
    assert _fit(1920, 1080, 2560, 1600) == (2560, 1440)
