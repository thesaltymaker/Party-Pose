"""Following a found person with the pose model (MediaPipe style): the detector only finds people."""
from unittest.mock import patch

from poser import _boxes_not_followed, _follow_boxes, _pose_box
from src.tracker import PersonTracker
from src.types import BodyResult, BoundingBox, Landmark


def _box(cx, cy=500, w=200, h=500, conf=0.8):
    return BoundingBox(x=cx - w / 2, y=cy - h / 2, w=w, h=h, confidence=conf)


def _confirmed(t, box):
    [track] = t.update_tracks([box])
    for _ in range(t.min_hits - 1):
        t.update_tracks([box])
    assert track.confirmed
    return track


def test_followed_track_lives_without_detector_boxes():
    t = PersonTracker()
    track = _confirmed(t, _box(500))
    for f in range(40):
        assert t.update_tracks([], followed={track.id: _box(500 + 5 * f)}) == []
        assert [tr.id for tr in t.tracks] == [track.id]
    assert track.missed == 0 and track.box.x == _box(695).x


def test_followed_track_does_not_take_other_boxes():
    t = PersonTracker()
    track = _confirmed(t, _box(500))
    [other] = t.update_tracks([_box(520)], followed={track.id: _box(500)})
    assert other.id != track.id and track.missed == 0


def test_track_no_longer_followed_needs_boxes_and_expires():
    t = PersonTracker(max_missed=5)
    track = _confirmed(t, _box(500))
    for _ in range(10):
        t.update_tracks([], followed={track.id: _box(500)})
    for _ in range(6):
        t.update_tracks([])
    assert track not in t.tracks and track in t.lost


def test_overlapping_detector_boxes_are_the_followed_person():
    pose = _box(500, w=220, h=520)
    part = BoundingBox(400, 250, 150, 300, 0.6)       # a box on the upper left of the same dancer
    whole = _box(510, w=240, h=540)                    # a box around all of them
    arm = BoundingBox(280, 250, 200, 500, 0.6)         # laptop_test: a box on a raised arm and the sofa, 0.45 inside
    other = _box(800)                                  # someone beside them
    assert _boxes_not_followed([part, whole, arm, other], [pose]) == [other]


def test_duplicate_detector_boxes_make_one_track():
    # Orin: two boxes (IoU 0.4) on one dancer each became a track and a skeleton. While the dancer is followed,
    # neither box starts a track, and frames without any box don't lose them.
    t = PersonTracker()
    track = _confirmed(t, _box(500))
    pose = _box(500, w=220, h=520)
    for f in range(30):
        boxes = [BoundingBox(400, 250, 200, 400, 0.6), BoundingBox(430, 300, 200, 450, 0.5)] if f % 3 == 0 else []
        t.update_tracks(_boxes_not_followed(boxes, [pose]), followed={track.id: pose})
    assert [tr.id for tr in t.tracks] == [track.id]


def test_one_person_found_twice_is_followed_once():
    a = _box(500, w=220, h=520)
    same = _box(505, w=215, h=515)
    other = _box(800)
    assert _follow_boxes([(3, a), (7, same), (9, other)]) == {3: a, 9: other}


def test_pose_box_covers_visible_landmarks():
    lms = [Landmark(100, 200, 0, 0.9, 0.9), Landmark(300, 600, 0, 0.9, 0.9), Landmark(900, 900, 0, 0.1, 0.1)]
    box = _pose_box(BodyResult(landmarks=lms), 1920, 1080)
    assert (box.x, box.y, box.w, box.h) == (80, 160, 240, 480)


def test_pose_box_is_clipped_to_the_frame():
    lms = [Landmark(10, 20, 0, 0.9, 0.9), Landmark(310, 1070, 0, 0.9, 0.9)]
    box = _pose_box(BodyResult(landmarks=lms), 1920, 1080)
    assert (box.x, box.y, box.x + box.w, box.y + box.h) == (0, 0, 340, 1080)


def test_follow_is_on_unless_no_follow():
    from src.config import parse_args
    with patch("sys.argv", ["poser.py"]):
        assert parse_args().follow is True
    with patch("sys.argv", ["poser.py", "--no-follow"]):
        assert parse_args().follow is False
