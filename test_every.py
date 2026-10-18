from src.ghosts import Ghosts, skip_turn


def test_off_or_nothing_to_reuse_never_skips():
    assert not any(skip_turn(1, f, k, True) for f in range(6) for k in range(3))
    assert not any(skip_turn(0, f, k, True) for f in range(6) for k in range(3))
    assert not any(skip_turn(2, f, k, False) for f in range(6) for k in range(3))  # nothing found last time


def test_people_alternate_so_every_frame_has_the_same_load():
    for f in range(6):
        skipped = [skip_turn(2, f, k, True) for k in range(4)]
        assert sum(skipped) == 2
    assert [skip_turn(2, f, 0, True) for f in range(4)] == [False, True, False, True]  # each person runs every 2nd


def test_every_3_runs_one_frame_in_3():
    runs = [not skip_turn(3, f, 0, True) for f in range(9)]
    assert sum(runs) == 3


def test_skipped_part_is_redrawn_at_full_brightness_and_not_aged():
    g = Ghosts(frames=5)
    g.update({(1, 'hand0'): 'h'})
    assert g.fresh((1, 'hand0'))
    for _ in range(4):
        assert g.update({}, skipped={(1, 'hand0')}) == [((1, 'hand0'), 'h', 1.0)]
        assert g.fresh((1, 'hand0'))
    assert g.update({(1, 'hand0'): 'h2'}) == []  # run again: drawn fresh


def test_a_part_lost_after_a_run_dims_as_before():
    g = Ghosts(frames=5)
    g.update({(1, 'hand0'): 'h'})
    g.update({}, skipped={(1, 'hand0')})
    assert g.update({}) == [((1, 'hand0'), 'h', 0.8)]
    assert not g.fresh((1, 'hand0'))


def test_skip_does_nothing_with_ghosts_off_or_for_an_unknown_part():
    g = Ghosts(frames=0)
    g.update({(1, 'face'): 'f'})
    assert g.update({}, skipped={(1, 'face')}) == []
    g = Ghosts(frames=5)
    assert g.update({}, skipped={(2, 'face')}) == []
