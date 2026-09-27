import numpy as np

from src.renderer import Renderer, COLOR_SKELLY, SANTA_HAT_RED
from src.topology import FACE_CONTOUR_INDICES, SKELLY_CUTOUTS, SANTA_HAT_LEFT, SANTA_HAT_RIGHT
from src.types import BoundingBox, FaceResult, Landmark

W, H = 400, 400


def _face(person_id=0):
    """478 landmarks: outline on a circle of radius 100 around (200, 250), each cut-out a small ring
    inside it (eyes, nose, mouth at different spots), temples at (150, 200) and (250, 200)."""
    lms = [Landmark(200.0, 250.0, 0.0) for _ in range(478)]
    for k, i in enumerate(FACE_CONTOUR_INDICES):
        a = 2 * np.pi * k / len(FACE_CONTOUR_INDICES)
        lms[i] = Landmark(200 + 100 * np.cos(a), 250 + 100 * np.sin(a), 0.0)
    for (cx, cy), cut in zip([(165, 225), (235, 225), (200, 255), (200, 295)], SKELLY_CUTOUTS):
        for k, i in enumerate(cut):
            a = 2 * np.pi * k / len(cut)
            lms[i] = Landmark(cx + 12 * np.cos(a), cy + 12 * np.sin(a), 0.0)
    lms[SANTA_HAT_LEFT] = Landmark(150.0, 200.0, 0.0)
    lms[SANTA_HAT_RIGHT] = Landmark(250.0, 200.0, 0.0)
    return FaceResult(bbox=BoundingBox(100, 150, 200, 200, 1.0), landmarks=lms, person_id=person_id)


def test_skelly_fills_face_and_leaves_eye_showing():
    frame = np.zeros((H, W, 3), np.uint8)
    Renderer(skelly=True).draw_faces(frame, [_face()], W, H)
    assert tuple(frame[320, 250]) == COLOR_SKELLY   # cheek, inside the outline
    assert tuple(frame[225, 165]) == (0, 0, 0)      # left eye centre is cut out
    assert tuple(frame[20, 20]) == (0, 0, 0)        # outside the face untouched


def test_skelly_scales_to_display_size():
    frame = np.zeros((2 * H, 2 * W, 3), np.uint8)
    Renderer(skelly=True).draw_faces(frame, [_face()], 2 * W, 2 * H, 2.0, 2.0)
    assert tuple(frame[640, 500]) == COLOR_SKELLY
    assert tuple(frame[450, 330]) == (0, 0, 0)


def test_skelly_face_partly_off_screen_does_not_crash():
    face = _face()
    for lm in face.landmarks:
        lm.x -= 250  # left half of the face off the frame
    frame = np.zeros((H, W, 3), np.uint8)
    Renderer(skelly=True).draw_faces(frame, [face], W, H)
    assert tuple(frame[320, 0]) == COLOR_SKELLY


def test_santa_hat_above_temples_flops_by_person():
    even = np.zeros((H, W, 3), np.uint8)
    odd = np.zeros((H, W, 3), np.uint8)
    Renderer(santa_hat=True).draw_faces(even, [_face(0)], W, H)
    Renderer(santa_hat=True).draw_faces(odd, [_face(1)], W, H)
    assert tuple(even[160, 210]) == SANTA_HAT_RED     # hat body above the temple line
    red_even = np.argwhere((even == SANTA_HAT_RED).all(axis=2))[:, 1].mean()
    red_odd = np.argwhere((odd == SANTA_HAT_RED).all(axis=2))[:, 1].mean()
    assert red_even > 200 > red_odd                  # even ids flop right, odd ids left
