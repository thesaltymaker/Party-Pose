import random
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
