import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools" / "orin_setup" / "audio"))

from party_pose_audio import (  # noqa: E402
    BUTTON_COMMANDS, Config, Controller, find_button_device, find_usb_sink,
    list_tracks, mpv_args, parse_key_presses, rotate_from)


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


USB = "alsa_output.usb-Logitech_Logitech_USB_Speaker-00.analog-stereo"


class FakePlayer:
    def __init__(self, sink, tracks):
        self.sink, self.tracks = sink, tracks
        self.alive, self.pos, self.sent, self.stopped = True, 0, [], False

    def running(self):
        return self.alive

    def position(self):
        return self.pos

    def command(self, cmd):
        self.sent.append(cmd)

    def stop(self):
        self.stopped = True
        self.alive = False


class Rig:
    def __init__(self, tmp_path, mode="sorted", names_=("a.mp3", "b.mp3", "c.mp3"),
                 seed=None):
        touch(tmp_path, *names_)
        self.cfg = Config(tmp_path, mode, tmp_path / "playlist.m3u", 50, "usb")
        self.sinks, self.players, self.volumes, self.logs = SINKS, [], [], []
        self.t = 100.0
        self.ctl = Controller(
            self.cfg, lambda: self.sinks, self.make, self.set_volume,
            self.logs.append, lambda: self.t, random.Random(seed))

    def make(self, sink, tracks):
        assert self.volumes, "volume must be set before the player starts"
        self.players.append(FakePlayer(sink, tracks))
        return self.players[-1]

    def set_volume(self, sink, vol):
        self.volumes.append((sink, vol))


def test_no_speakers_no_player(tmp_path):
    r = Rig(tmp_path)
    r.sinks = "0\talsa_output.platform-sound.analog-stereo\tm\ts\tS\n"
    for _ in range(3):
        r.ctl.tick()
    assert r.players == []


def test_speakers_present_starts_player_and_sets_volume(tmp_path):
    r = Rig(tmp_path)
    r.ctl.tick()
    r.ctl.tick()
    assert len(r.players) == 1
    assert r.volumes == [(USB, 50)]
    assert r.players[0].sink == USB
    assert names(r.players[0].tracks) == ["a.mp3", "b.mp3", "c.mp3"]


def test_unplug_stops_player_and_replug_resumes_at_same_track(tmp_path):
    r = Rig(tmp_path)
    r.ctl.tick()
    r.players[0].pos = 2
    r.sinks = None
    r.ctl.tick()
    assert r.players[0].stopped
    r.sinks = SINKS
    r.ctl.tick()
    assert names(r.players[1].tracks) == ["c.mp3", "a.mp3", "b.mp3"]


def test_random_mode_reshuffles_on_replug(tmp_path):
    r = Rig(tmp_path, "random", [f"t{i}.mp3" for i in range(8)], seed=3)
    r.ctl.tick()
    r.players[0].pos = 5
    r.sinks = None
    r.ctl.tick()
    r.sinks = SINKS
    r.ctl.tick()
    first, second = r.players[0].tracks, r.players[1].tracks
    assert sorted(second) == sorted(first)
    assert second != first


def test_pactl_failure_is_no_speakers(tmp_path):
    r = Rig(tmp_path)
    r.sinks = None
    r.ctl.tick()
    r.sinks = ""
    r.ctl.tick()
    assert r.players == []


def test_empty_music_folder_warns_once_and_never_starts(tmp_path):
    r = Rig(tmp_path, names_=())
    for _ in range(10):
        r.ctl.tick()
    assert r.players == []
    assert len(r.logs) == 1


def test_player_exit_restarts_after_five_seconds_only(tmp_path):
    r = Rig(tmp_path)
    r.ctl.tick()
    r.players[0].alive = False
    r.t += 1
    r.ctl.tick()
    assert len(r.players) == 1
    r.t += 4
    r.ctl.tick()
    assert len(r.players) == 2


def test_replug_starts_unpaused(tmp_path):
    r = Rig(tmp_path)
    r.ctl.tick()
    r.ctl.press(164)
    r.sinks = None
    r.ctl.tick()
    r.sinks = SINKS
    r.ctl.tick()
    assert r.players[0].sent == [["cycle", "pause"]]
    assert r.players[1] is not r.players[0]
    assert r.players[1].sent == []


def test_press_maps_buttons_when_running_only(tmp_path):
    r = Rig(tmp_path)
    r.ctl.press(163)
    r.ctl.tick()
    for code in (163, 165, 164, 166, 113):
        r.ctl.press(code)
    assert r.players[0].sent == [["playlist-next"], ["playlist-prev"], ["cycle", "pause"]]


def test_config_from_env_defaults_and_clamp():
    c = Config.from_env({})
    home = Path("~/Projects/Video_Projects/Music").expanduser()
    assert (c.music_dir, c.mode, c.volume, c.sink_match) == (home, "sorted", 50, "usb")
    assert c.playlist == home / "playlist.m3u"
    assert Config.from_env({"VOLUME": "abc"}).volume == 50
    assert Config.from_env({"VOLUME": "250"}).volume == 100
    c = Config.from_env({"MODE": "random", "SINK_MATCH": "foo", "MUSIC_DIR": "/m"})
    assert (c.mode, c.sink_match, c.music_dir) == ("random", "foo", Path("/m"))
    assert c.playlist == Path("/m/playlist.m3u")


def test_mpv_command_line_uses_double_dash_before_files():
    weird = Path("/m/-weird [x] \u00e9.m4a")
    args = mpv_args("sink1", [weird], "/run/u")
    for flag in ("--audio-device=pulse/sink1", "--no-video", "--vid=no",
                 "--loop-playlist=inf", "--input-ipc-server=/run/u/party-pose-audio.sock"):
        assert flag in args
    assert args.index("--") < args.index(str(weird))
    assert args.count(str(weird)) == 1
