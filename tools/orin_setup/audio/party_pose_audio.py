import random
import struct
from pathlib import Path

AUDIO_EXTS = (".webm", ".m4a", ".mp3", ".flac", ".ogg", ".opus", ".wav")


def _sorted_tracks(music_dir):
    if not music_dir.is_dir():
        return []
    files = [p for p in music_dir.iterdir()
             if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files, key=lambda p: p.name.casefold())


def list_tracks(music_dir, mode, playlist, log, rng=None):
    tracks = _sorted_tracks(music_dir)
    if mode == "sorted":
        return tracks
    if mode == "random":
        (rng or random).shuffle(tracks)
        return tracks
    if mode == "playlist":
        if playlist is None or not playlist.is_file():
            log(f"playlist {playlist} not found, using sorted order")
            return tracks
        listed, missing = [], []
        for line in playlist.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = music_dir / line
            (listed if path.is_file() else missing).append(path)
        if not listed:
            log(f"playlist {playlist} has no existing files, using sorted order")
            return tracks
        for path in missing:
            log(f"playlist entry not found: {path.name}")
        return listed
    log(f"unknown mode {mode!r}, using sorted order")
    return tracks


def rotate_from(tracks, index):
    if not 0 <= index < len(tracks):
        return tracks
    return tracks[index:] + tracks[:index]


BUTTON_COMMANDS = {
    163: ["playlist-next"],
    165: ["playlist-prev"],
    164: ["cycle", "pause"],
}

_EVENT = struct.Struct("llHHi")


def find_usb_sink(pactl_short_sinks, match):
    for line in (pactl_short_sinks or "").splitlines():
        cols = line.split("\t")
        if len(cols) > 1 and match.casefold() in cols[1].casefold():
            return cols[1]
    return None


def find_button_device(by_id_names, match):
    for name in by_id_names:
        if match.casefold() in name.casefold() and "-event" in name:
            return name
    return None


def parse_key_presses(data):
    codes = []
    end = len(data) - len(data) % _EVENT.size
    for off in range(0, end, _EVENT.size):
        _, _, etype, code, value = _EVENT.unpack_from(data, off)
        if etype == 1 and value == 1:
            codes.append(code)
    return codes, data[end:]
