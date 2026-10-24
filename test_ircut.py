from src.video_capture import DAY_GAIN_MAX, IRCutFilter, NightModeDetector, SWITCH_CHECKS


def _detector(readings, changes):
    it = iter(readings)
    return NightModeDetector(check_every_n_frames=1, query=lambda: next(it), on_change=changes.append)


def _run(det, n):
    return [det.update() for _ in range(n)]


def test_dark_for_three_checks_switches_to_night():
    changes = []
    det = _detector([(170, 30000)] * SWITCH_CHECKS, changes)
    assert _run(det, SWITCH_CHECKS) == [False] * (SWITCH_CHECKS - 1) + [True]
    assert changes == [True]


def test_one_bright_reading_restarts_the_count():
    changes = []
    det = _detector([(170, 30000), (170, 30000), (100, 30000)] + [(170, 30000)] * SWITCH_CHECKS, changes)
    assert not any(_run(det, 3))
    assert _run(det, SWITCH_CHECKS)[-1] is True


def test_short_exposure_is_not_night():
    changes = []
    det = _detector([(170, 10000)] * 5, changes)
    assert not any(_run(det, 5)) and changes == []


def test_night_stays_until_gain_is_low_then_day():
    changes = []
    readings = [(170, 30000)] * SWITCH_CHECKS + [(120, 30000)] * 5 + [(DAY_GAIN_MAX, 20000)] * SWITCH_CHECKS
    det = _detector(readings, changes)
    _run(det, SWITCH_CHECKS)
    assert all(_run(det, 5))  # IR light lowers the gain a little: stays night
    assert _run(det, SWITCH_CHECKS)[-1] is False
    assert changes == [True, False]


def test_unreadable_camera_keeps_the_mode():
    changes = []
    det = _detector([(None, None)] * 5, changes)
    assert not any(_run(det, 5)) and changes == []


class _Proc:
    def __init__(self, args):
        self.args, self.stopped = args, False

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0


def test_ircut_holds_the_pin_high_at_night_and_low_by_day():
    procs = []
    f = IRCutFilter(popen=lambda args: procs.append(_Proc(args)) or procs[-1])
    f.set(True)
    f.set(False)
    f.close()
    assert [p.args[-1] for p in procs] == ['144=0', '144=1', '144=0']
    assert all(p.stopped for p in procs)


def test_ircut_without_gpioset_does_not_crash():
    def missing(args):
        raise FileNotFoundError('gpioset')
    f = IRCutFilter(popen=missing)
    f.set(True)
    f.close()
