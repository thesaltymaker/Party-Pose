from src.tracker import PersonTracker
from src.types import BoundingBox


def _box(cx, cy=500, w=200, h=500, conf=0.8):
    return BoundingBox(x=cx - w / 2, y=cy - h / 2, w=w, h=h, confidence=conf)


def test_ids_stay_fixed_when_detector_order_swaps():
    # Issue #6: the detector orders by score, so the two people swap index from frame to frame.
    t = PersonTracker()
    a, b = t.update([_box(400), _box(1400)])
    for f in range(30):
        boxes = [_box(400 + f), _box(1400 - f)]
        if f % 2:
            boxes.reverse()
            assert t.update(boxes) == [b, a]
        else:
            assert t.update(boxes) == [a, b]
    assert a != b


def test_one_missed_frame_keeps_the_id():
    t = PersonTracker()
    [a] = t.update([_box(500)])
    t.update([_box(505)])
    assert t.update([]) == []
    assert t.update([_box(515)]) == [a]


def test_track_expires_after_max_missed_frames():
    t = PersonTracker(max_missed=5)
    [a] = t.update([_box(500)])
    for _ in range(6):
        t.update([])
    assert t.update([_box(500)]) != [a]


def test_leaving_and_re_entering_gets_a_new_id_and_does_not_steal_others():
    t = PersonTracker(max_missed=5)
    a, b = t.update([_box(400), _box(1400)])
    for f in range(10):  # a walks out of frame, b stays
        ids = t.update([_box(1400)])
        assert ids == [b]
    ids = t.update([_box(100), _box(1400)])  # a comes back at the left edge
    assert ids[1] == b
    assert ids[0] not in (a, b)


def test_two_people_crossing_keep_their_ids():
    # a walks right, b walks left; they pass each other, and at the crossing only one box is detected.
    t = PersonTracker()
    a, b = t.update([_box(700), _box(1300)])
    ax, bx = 700, 1300
    for _ in range(60):
        ax += 10
        bx -= 10
        if abs(ax - bx) < 60:
            ids = t.update([_box(ax)])  # b hidden behind a
            assert ids == [a]
            continue
        boxes = [_box(bx), _box(ax)]  # detector order independent of identity
        assert t.update(boxes) == [b, a], (ax, bx)
    assert ax > bx  # they did cross


def test_fast_movement_matches_by_centre_distance():
    t = PersonTracker()
    [a] = t.update([_box(500, w=100)])
    # 150 px step with 100 px wide boxes: no overlap, but the centre moved < 0.6 of the box diagonal.
    assert t.update([_box(650, w=100)]) == [a]


def test_far_new_box_is_a_new_person():
    t = PersonTracker()
    [a] = t.update([_box(300)])
    ids = t.update([_box(300), _box(1600)])
    assert ids[0] == a and ids[1] != a


def test_history_records_centres():
    t = PersonTracker()
    t.update([_box(500)])
    t.update([_box(510)])
    [track] = t.tracks
    assert list(track.history) == [(500, 500), (510, 500)]


def test_track_is_confirmed_after_min_hits():
    t = PersonTracker(min_hits=3)
    assert [tr.confirmed for tr in t.update_tracks([_box(500)])] == [False]
    assert [tr.confirmed for tr in t.update_tracks([_box(505)])] == [False]
    assert [tr.confirmed for tr in t.update_tracks([_box(510)])] == [True]
    t.update_tracks([])  # a missed frame doesn't unconfirm
    assert [tr.confirmed for tr in t.update_tracks([_box(520)])] == [True]


def test_flickering_false_positive_is_never_confirmed():
    # A box that shows up every few frames in a new-ish place (e.g. detector noise) never reaches min_hits.
    t = PersonTracker(min_hits=3, max_missed=1)
    for f in range(30):
        boxes = [_box(400)]
        if f % 4 == 0:
            boxes.append(_box(1500))
        tracks = t.update_tracks(boxes)
        assert all(not tr.confirmed for tr in tracks[1:])
    assert tracks[0].confirmed
