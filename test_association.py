from poser import _assign_heads, _drop_nested_bodies
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


def test_partial_duplicate_box_inside_a_person_is_dropped():
    # Orin run, 3 people: a 0.56 box on the left part of a person, 90% inside their 0.51 box, was tracked
    # as a second person (#1/#20) and took the person's head.
    person = _box(678, 393, 357, 481, conf=0.51)
    part = _box(663, 406, 147, 314, conf=0.56)
    head = _box(690, 410, 70, 80)
    assert _drop_nested_bodies([part, person], [head]) == [person]


def test_nested_box_without_heads_keeps_the_bigger_box():
    person = _box(678, 393, 357, 481, conf=0.51)
    part = _box(663, 406, 147, 314, conf=0.56)
    assert _drop_nested_bodies([part, person], []) == [person]


def test_higher_scoring_person_inside_false_positive_box_keeps_the_person():
    # The head sits in the middle of the big box, so only the person's box fits it.
    person = _box(600, 300, 250, 700, conf=0.8)
    around = _box(300, 0, 900, 1080, conf=0.4)
    head = _box(690, 310, 70, 80)
    assert _drop_nested_bodies([person, around], [head]) == [person]


def test_two_near_identical_boxes_keep_one():
    a = _box(100, 100, 200, 400, conf=0.6)
    b = _box(102, 98, 200, 404, conf=0.6)
    assert len(_drop_nested_bodies([a, b], [])) == 1


def test_ghost_face_follows_the_head():
    # Issue #13: a missing face's ghost is redrawn where the head is now, scaled with it.
    from poser import _moved_face
    from src.types import FaceResult, Landmark
    old_head = _box(100, 100, 80, 100)                      # centre (140, 150)
    face = FaceResult(bbox=old_head, person_id=3,
                      landmarks=[Landmark(140, 150, 0), Landmark(160, 170, 2)])
    new_head = _box(300, 200, 160, 200)                     # centre (380, 300), twice as big
    moved = _moved_face(face, old_head, new_head)
    assert [(lm.x, lm.y, lm.z) for lm in moved.landmarks] == [(380, 300, 0), (420, 340, 4)]
    assert moved.person_id == 3 and moved.bbox == new_head
    assert face.landmarks[1].x == 160                       # the stored face is not changed


def test_picture_is_centred_on_a_black_screen_sized_canvas():
    # Issue #12: a 16:9 picture on a 1920x1200 (16:10) monitor left a white band; the window must be screen-sized.
    import numpy as np
    from poser import _screen_canvas
    canvas, view = _screen_canvas(1920, 1200, 1920, 1080)
    assert canvas.shape == (1200, 1920, 3) and view.shape == (1080, 1920, 3)
    view[:] = 255
    assert not canvas[:60].any() and not canvas[1140:].any()   # 60 px black above and below
    assert canvas[60:1140].all()


def test_picture_on_a_portrait_screen_is_centred_vertically():
    from poser import _fit, _screen_canvas
    w, h = _fit(1920, 1080, 1080, 1920)
    canvas, view = _screen_canvas(1080, 1920, w, h)
    view[:] = 255
    assert canvas.shape == (1920, 1080, 3) and (w, h) == (1080, 608)
    top = (1920 - 608) // 2
    assert not canvas[:top].any() and canvas[top:top + 608].all() and not canvas[top + 608:].any()


def test_drawing_in_the_picture_area_does_not_reach_the_border():
    import cv2
    from poser import _screen_canvas
    canvas, view = _screen_canvas(1920, 1200, 1920, 1080)
    cv2.line(view, (0, 0), (1919, 1079), (255, 255, 255), 2)
    assert canvas[60, 0].any() and not canvas[:60].any() and not canvas[1140:].any()


def _pose_with_face(x0=100.0, y0=50.0):
    from src.types import Landmark
    lms = [Landmark(0.0, 0.0, 0.0) for _ in range(33)]
    for i in range(11):
        lms[i] = Landmark(x0 + 4 * i, y0 + i, 0.0)  # nose, eyes, ears, mouth: 40 wide, 10 high
    return lms


def test_face_box_from_pose_is_a_square_around_the_face_points():
    # Issue #13 clips: crops from the pose's face points at 2x found 81% / 92% of faces, the head box 50% / 58%.
    from src.face_processor import face_box_from_pose
    box = face_box_from_pose(_pose_with_face(), 2.0)
    assert (box.w, box.h) == (80, 80) and (box.x + box.w / 2, box.y + box.h / 2) == (120, 55)


def test_face_crop_is_the_head_box_else_the_last_pose():
    from poser import _face_box
    head = _box(90, 20, 60, 70)
    pose = _pose_with_face()
    assert _face_box(head, pose, 1920, 1080) == head            # a head box: as before
    assert _face_box(None, pose, 1920, 1080).w == 80            # none: around the last pose's face points
    assert _face_box(None, None, 1920, 1080) is None
    assert _face_box(None, _pose_with_face(x0=2500.0), 1920, 1080) is None  # outside the picture
