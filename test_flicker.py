from src.flicker import FlickerCounter


def _run(frames):
    """frames: one set of drawn (track id, part) keys per frame."""
    fc = FlickerCounter()
    for drawn in frames:
        fc.update(drawn)
    return fc


def test_short_gap_between_drawn_frames_counts_as_a_blink():
    a = (1, 'body')
    fc = _run([{a}, set(), set(), {a}])
    assert fc.blinks['body'] == {2: 1}
    assert fc.drawn['body'] == 2


def test_parts_and_tracks_are_counted_on_their_own():
    fc = _run([{(1, 'body'), (1, 'face'), (2, 'face')}, {(1, 'body')}, {(1, 'body'), (1, 'face'), (2, 'face')}])
    assert fc.blinks['face'] == {1: 2}
    assert 'body' not in fc.blinks


def test_long_absence_is_not_a_blink():
    a = (1, 'hand0')
    fc = _run([{a}] + [set()] * 6 + [{a}])
    assert fc.blinks == {}


def test_report_line():
    fc = _run([{(1, 'body')}, set(), {(1, 'body')}])
    assert fc.report() == '[FLICKER] body: drawn=2 blinks=1 (1f:1)'
