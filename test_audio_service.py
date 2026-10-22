import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools" / "orin_setup" / "audio"))

from party_pose_audio import list_tracks, rotate_from  # noqa: E402


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
