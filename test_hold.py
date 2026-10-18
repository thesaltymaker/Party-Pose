import numpy as np

from src.hold import JointHold
from src.renderer import Renderer, PERSON_COLORS
from src.types import BodyResult, Landmark


def _body(dx=0.0, low=(), person_id=1):
    """33 joints; shoulders 11/12 and hips 23/24 around (200, 225); elbow 13 at (150, 250)."""
    lms = [Landmark(200.0 + dx, 200.0, 0.0) for _ in range(33)]
    lms[11], lms[12] = Landmark(150.0 + dx, 150.0, 0.0), Landmark(250.0 + dx, 150.0, 0.0)
    lms[23], lms[24] = Landmark(170.0 + dx, 300.0, 0.0), Landmark(230.0 + dx, 300.0, 0.0)
    lms[13] = Landmark(150.0 + dx, 250.0, 0.0)
    for i in low:
        lms[i].presence = 0.1
        lms[i].x, lms[i].y = 999.0, 999.0  # the model's guess for a lost joint: anywhere
    return BodyResult(landmarks=lms, presence=1.0, person_id=person_id)


def test_lost_joint_is_held_at_its_last_place_dimming_then_dropped():
    h = JointHold(frames=4)
    h.apply([_body()])
    fades, places = [], []
    for _ in range(4):
        b = _body(low=[13])
        h.apply([b])
        fades.append(b.landmarks[13].fade)
        places.append((b.landmarks[13].x, b.landmarks[13].y))
    assert fades == [0.75, 0.5, 0.25, 1.0]
    assert places[:3] == [(150.0, 250.0)] * 3
    assert b.landmarks[13].presence < 0.5  # 4th frame: not held, hidden as before


def test_held_joint_moves_with_the_body():
    h = JointHold(frames=4)
    h.apply([_body()])
    b = _body(dx=10.0, low=[13])
    h.apply([b])
    assert (b.landmarks[13].x, b.landmarks[13].y) == (160.0, 250.0)


def test_joint_found_again_is_drawn_fresh():
    h = JointHold(frames=4)
    h.apply([_body()])
    h.apply([_body(low=[13])])
    b = _body(dx=5.0)
    h.apply([b])
    assert b.landmarks[13].fade == 1.0 and b.landmarks[13].x == 155.0


def test_a_dim_joint_stays_dim_while_held():
    h = JointHold(frames=4)
    b = _body()
    b.landmarks[13].visibility = 0.2
    h.apply([b])
    b = _body(low=[13])
    b.landmarks[13].visibility = 0.9  # the lost joint's own score means nothing
    h.apply([b])
    assert b.landmarks[13].visibility == 0.2


def test_zero_frames_holds_nothing():
    h = JointHold(frames=0)
    h.apply([_body()])
    b = _body(low=[13])
    h.apply([b])
    assert b.landmarks[13].fade == 1.0 and b.landmarks[13].x == 999.0


def test_holds_are_per_person():
    h = JointHold(frames=4)
    h.apply([_body(person_id=1)])
    b2 = _body(person_id=2, low=[13])
    h.apply([b2])
    assert b2.landmarks[13].fade == 1.0


def test_renderer_draws_a_held_joint_dimmed():
    b = _body()
    b.person_id = 3
    b.landmarks[13].presence, b.landmarks[13].fade = 0.1, 0.5
    frame = np.zeros((400, 400, 3), np.uint8)
    Renderer().draw_body(frame, [b], 400, 400)
    expected = tuple(int(c * 0.5) for c in PERSON_COLORS[3])
    assert tuple(frame[250, 150]) == expected  # the held elbow
    assert tuple(frame[200, 150]) == expected  # middle of the shoulder-elbow line
