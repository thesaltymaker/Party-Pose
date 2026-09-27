import os
import signal

import pytest

from poser import _is_quit_key, _raise_terminated, _Terminated
from src.config import parse_args


def test_q_and_escape_quit():
    assert _is_quit_key(ord('q'))
    assert _is_quit_key(27)


def test_no_key_and_other_keys_do_not_quit():
    assert not _is_quit_key(-1)
    assert not _is_quit_key(ord('a'))
    assert not _is_quit_key(32)


def test_quit_key_ignores_high_bits():
    # Some GTK builds return modifier bits above the low byte.
    assert _is_quit_key(0x100000 | 27)


def test_sigterm_raises_terminated():
    old = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        with pytest.raises(_Terminated):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, old)


def test_fullscreen_flag(monkeypatch):
    monkeypatch.setattr('sys.argv', ['poser.py', '--platform', 'laptop'])
    assert parse_args().fullscreen is False
    monkeypatch.setattr('sys.argv', ['poser.py', '--platform', 'laptop', '--fullscreen'])
    assert parse_args().fullscreen is True


def test_require_face_flag(monkeypatch):
    monkeypatch.setattr('sys.argv', ['poser.py', '--platform', 'laptop'])
    assert parse_args().require_face is True
    monkeypatch.setattr('sys.argv', ['poser.py', '--platform', 'laptop', '--no-require-face'])
    assert parse_args().require_face is False
