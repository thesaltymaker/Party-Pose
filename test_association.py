from poser import _find_head_for_body
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
