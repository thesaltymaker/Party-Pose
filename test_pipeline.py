import threading
import time

import pytest

from poser import _run_pipelined


def test_results_are_consumed_in_order_until_consume_stops():
    n = iter(range(100))
    seen = []
    _run_pipelined(lambda: next(n), lambda x: seen.append(x) or len(seen) < 10)
    assert seen == list(range(10))


def test_produce_runs_in_another_thread_while_consume_works():
    # The point of --pipeline: the next frame's models run while this frame is drawn.
    produced_during_consume = []
    busy = threading.Event()

    def produce():
        produced_during_consume.append(busy.is_set())
        return 0

    def consume(_):
        busy.set()
        time.sleep(0.02)
        busy.clear()
        return len(produced_during_consume) < 6

    _run_pipelined(produce, consume)
    assert any(produced_during_consume)


def test_error_in_produce_is_raised_in_the_caller():
    def produce():
        raise RuntimeError('Camera read failed')
    with pytest.raises(RuntimeError, match='Camera read failed'):
        _run_pipelined(produce, lambda x: True)


def test_worker_stops_after_consume_stops():
    calls = []
    _run_pipelined(lambda: calls.append(1) or len(calls), lambda x: x < 3)
    time.sleep(0.3)
    n = len(calls)
    time.sleep(0.3)
    assert len(calls) == n          # nothing produced after the stop
    assert not [t for t in threading.enumerate() if t.name == 'poser-infer']
