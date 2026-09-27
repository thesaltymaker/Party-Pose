from src.tracker import HEAD_WINDOW, PersonTracker
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


def _confirmed(t, box, frames=5):
    for _ in range(frames):
        [tr] = t.update_tracks([box])
    assert tr.confirmed
    return tr


def test_lost_person_back_in_the_same_spot_keeps_their_id():
    # Orin run, 3 people: a far person (score 0.33-0.40) was dropped by the detector for more than
    # max_missed frames again and again and got 5 IDs in 60 s (#9, #16, #24, #32, #34).
    t = PersonTracker(max_missed=5, max_lost=150)
    tr = _confirmed(t, _box(1600, w=120, h=240))
    for _ in range(40):
        t.update([])
    [back] = t.update_tracks([_box(1605, w=120, h=240)])
    assert back.id == tr.id and back.confirmed


def test_lost_person_whose_box_changes_size_keeps_their_id():
    # Orin run: the far person's box flips between 117x239 and 220x510 (IoU 0.25, small box inside the big one).
    t = PersonTracker(max_missed=5, max_lost=150)
    tr = _confirmed(t, BoundingBox(1540, 433, 117, 239, 0.4))
    for _ in range(20):
        t.update([])
    assert t.update([BoundingBox(1512, 421, 223, 507, 0.4)]) == [tr.id]


def test_unconfirmed_flicker_is_not_revived():
    t = PersonTracker(max_missed=2, max_lost=150, min_hits=3)
    [a] = t.update([_box(500)])
    for _ in range(5):
        t.update([])
    assert t.update([_box(500)]) != [a]


def test_lost_track_is_forgotten_after_max_lost():
    t = PersonTracker(max_missed=2, max_lost=10)
    tr = _confirmed(t, _box(500))
    for _ in range(20):
        t.update([])
    assert t.update([_box(500)]) != [tr.id]


def test_lost_track_is_not_revived_somewhere_else():
    t = PersonTracker(max_missed=2, max_lost=150)
    tr = _confirmed(t, _box(500))
    for _ in range(5):
        t.update([])
    assert t.update([_box(1500)]) != [tr.id]


def test_lost_track_revives_for_only_one_box():
    t = PersonTracker(max_missed=2, max_lost=150)
    tr = _confirmed(t, _box(500))
    for _ in range(5):
        t.update([])
    ids = t.update([_box(500), _box(510)])
    assert ids.count(tr.id) == 1


def test_head_filter_shows_person_and_hides_headless_box():
    # Orin runs: people had a head in 48-100% of frames, false positives in 8-33%.
    t = PersonTracker()
    person, prop = t.update_tracks([_box(400), _box(1400)])
    for f in range(HEAD_WINDOW):
        person.note_head(f % 2 == 0)   # 50%
        prop.note_head(f % 3 == 0)     # 33%
    assert person.head_ok and not prop.head_ok


def test_head_filter_waits_for_enough_frames():
    t = PersonTracker()
    [tr] = t.update_tracks([_box(400)])
    for _ in range(5):
        tr.note_head(True)
    assert not tr.head_ok


def test_head_filter_does_not_flicker_near_the_threshold():
    # Once shown, a track is hidden only when its head rate falls clearly below the show threshold.
    t = PersonTracker()
    [tr] = t.update_tracks([_box(400)])
    for _ in range(HEAD_WINDOW):
        tr.note_head(True)
    assert tr.head_ok
    for f in range(HEAD_WINDOW):
        tr.note_head(f % 10 < 4)       # 40%: below show (45%), above hide (35%)
    assert tr.head_ok
    for f in range(HEAD_WINDOW):
        tr.note_head(f % 10 < 1)       # 10%
    assert not tr.head_ok


def test_box_flipping_between_small_and_big_keeps_the_id():
    # Orin run: a still, far person's box flips between 117x239 and ~220x510 every few frames. The centre
    # jumps ~125 px each flip, which the velocity estimate took as movement, so the prediction missed the
    # next box and a new ID started while the old track was still alive.
    t = PersonTracker()
    small = BoundingBox(1540, 433, 117, 239, 0.4)
    big = BoundingBox(1512, 421, 223, 507, 0.4)
    [first] = t.update([small])
    for f in range(90):
        box = big if (f // 3) % 2 == 0 else small
        assert t.update([box] if f % 7 else []) == ([first] if f % 7 else [])
