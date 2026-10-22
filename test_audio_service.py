import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools" / "orin_setup" / "audio"))

from party_pose_audio import (  # noqa: E402
    BUTTON_COMMANDS, find_button_device, find_usb_sink, list_tracks,
    parse_key_presses, rotate_from)


def touch(folder, *names):
    for n in names:
        (folder / n).write_text("x")


def names(paths):
    return [p.name for p in paths]


def test_sorted_is_filename_order_and_filters_extensions(tmp_path):
    touch(tmp_path, "b.m4a", "A.webm", "c.txt", "d.mp3")
    (tmp_path / "sub").mkdir()
    touch(tmp_path / "sub", "e.m4a")
    out = list_tracks(tmp_path, "sorted", None, lambda m: None)
    assert names(out) == ["A.webm", "b.m4a", "d.mp3"]


def test_random_has_same_files_and_is_seeded(tmp_path):
    touch(tmp_path, *[f"t{i}.mp3" for i in range(8)])
    base = list_tracks(tmp_path, "sorted", None, lambda m: None)
    a = list_tracks(tmp_path, "random", None, lambda m: None, random.Random(1))
    b = list_tracks(tmp_path, "random", None, lambda m: None, random.Random(1))
    assert sorted(a) == base
    assert a == b


def test_playlist_order_comments_blanks_and_spaces(tmp_path):
    touch(tmp_path, "my song.m4a", "b.m4a", "é☃.mp3", "z.mp3")
    pl = tmp_path / "playlist.m3u"
    pl.write_text("# c\n\n  my song.m4a  \nb.m4a\n", encoding="utf-8")
    out = list_tracks(tmp_path, "playlist", pl, lambda m: None)
    assert names(out) == ["my song.m4a", "b.m4a"]
    pl.write_text("é☃.mp3\n", encoding="utf-8")
    out = list_tracks(tmp_path, "playlist", pl, lambda m: None)
    assert names(out) == ["é☃.mp3"]


def test_playlist_missing_file_is_skipped_with_warning(tmp_path):
    touch(tmp_path, "a.mp3")
    pl = tmp_path / "playlist.m3u"
    pl.write_text("a.mp3\nghost.mp3\n")
    logs = []
    out = list_tracks(tmp_path, "playlist", pl, logs.append)
    assert names(out) == ["a.mp3"]
    assert len(logs) == 1 and "ghost.mp3" in logs[0]


def test_playlist_missing_or_all_missing_falls_back_to_sorted(tmp_path):
    touch(tmp_path, "b.mp3", "a.mp3")
    logs = []
    out = list_tracks(tmp_path, "playlist", tmp_path / "none.m3u", logs.append)
    assert names(out) == ["a.mp3", "b.mp3"]
    assert len(logs) == 1
    pl = tmp_path / "playlist.m3u"
    pl.write_text("ghost.mp3\n")
    logs = []
    out = list_tracks(tmp_path, "playlist", pl, logs.append)
    assert names(out) == ["a.mp3", "b.mp3"]
    assert len(logs) == 1


def test_unknown_mode_falls_back_to_sorted_with_warning(tmp_path):
    touch(tmp_path, "b.mp3", "a.mp3")
    logs = []
    out = list_tracks(tmp_path, "shuffle", None, logs.append)
    assert names(out) == ["a.mp3", "b.mp3"]
    assert len(logs) == 1


def test_empty_or_missing_folder_returns_empty_list(tmp_path):
    assert list_tracks(tmp_path / "nope", "sorted", None, lambda m: None) == []
    assert list_tracks(tmp_path, "sorted", None, lambda m: None) == []


def test_rotate_from_wraps():
    assert rotate_from(["a", "b", "c"], 1) == ["b", "c", "a"]
    assert rotate_from(["a", "b", "c"], 0) == ["a", "b", "c"]
    assert rotate_from(["a", "b", "c"], 9) == ["a", "b", "c"]


SINKS = (
    "0\talsa_output.platform-sound.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
    "1\talsa_output.usb-Logitech_Logitech_USB_Speaker-00.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
)


def test_find_usb_sink_picks_usb_not_builtin():
    assert (find_usb_sink(SINKS, "usb")
            == "alsa_output.usb-Logitech_Logitech_USB_Speaker-00.analog-stereo")


def test_find_usb_sink_none_cases():
    builtin = "0\talsa_output.platform-sound.analog-stereo\tm\ts16le\tSUSPENDED\n"
    assert find_usb_sink(builtin, "usb") is None
    assert find_usb_sink("", "usb") is None
    assert find_usb_sink(None, "usb") is None


def test_find_usb_sink_first_of_two_and_case_insensitive():
    two = ("0\talsa_output.usb-A.analog-stereo\tm\ts\tS\n"
           "1\talsa_output.usb-B.analog-stereo\tm\ts\tS\n")
    assert find_usb_sink(two, "USB") == "alsa_output.usb-A.analog-stereo"


def test_find_button_device():
    names_ = ["usb-Logitech_Logitech_USB_Speaker-event-if02", "usb-Foo_Keyboard-event-kbd"]
    assert find_button_device(names_, "speaker") == names_[0]
    assert find_button_device(names_, "nomatch") is None
    assert find_button_device([], "speaker") is None


def rec(type_, code, value):
    return struct.pack("llHHi", 0, 0, type_, code, value)


def test_parse_key_presses_press_only():
    data = rec(1, 163, 1) + rec(1, 163, 0) + rec(0, 0, 0) + rec(1, 30, 1)
    assert parse_key_presses(data) == ([163, 30], b"")


def test_parse_key_presses_keeps_partial_record():
    data = rec(1, 164, 1) + b"abcdef"
    codes, rest = parse_key_presses(data)
    assert codes == [164]
    assert len(rest) == 6


def test_button_commands_map():
    assert BUTTON_COMMANDS == {163: ["playlist-next"], 165: ["playlist-prev"],
                               164: ["cycle", "pause"]}
    for code in (166, 113, 114, 115):
        assert code not in BUTTON_COMMANDS
