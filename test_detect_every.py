"""--detect-every N: the person detector runs every Nth frame while every live person is followed by the pose."""
from types import SimpleNamespace

from poser import _detect_now


def _tr(tid, still=False):
    return SimpleNamespace(id=tid, is_still=lambda: still)


def test_every_frame_without_the_option():
    assert all(_detect_now(n, 1, [_tr(1)], {1}, True) for n in range(1, 10))


def test_everyone_followed_runs_on_every_nth_frame_only():
    runs = [n for n in range(1, 13) if _detect_now(n, 3, [_tr(1), _tr(2)], {1, 2}, True)]
    assert runs == [3, 6, 9, 12]


def test_a_person_not_followed_runs_it_every_frame():
    # Lost by the pose, or new and not confirmed yet: only a detector box finds them again.
    assert all(_detect_now(n, 3, [_tr(1), _tr(2)], {1}, True) for n in range(1, 10))


def test_nobody_tracked_runs_it_every_frame():
    assert all(_detect_now(n, 3, [], set(), True) for n in range(1, 10))


def test_a_still_track_dropped_by_drop_still_does_not_count():
    # A prop (the lamp) is never followed; with --drop-still it must not keep the detector on every frame.
    assert not _detect_now(1, 3, [_tr(1), _tr(9, still=True)], {1}, True)
    assert _detect_now(1, 3, [_tr(1), _tr(9, still=True)], {1}, False)
