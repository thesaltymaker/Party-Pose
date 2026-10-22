import json
import os
import random
import signal
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
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


@dataclass
class Config:
    music_dir: Path
    mode: str
    playlist: Path
    volume: int
    sink_match: str

    @classmethod
    def from_env(cls, environ):
        music = Path(environ.get("MUSIC_DIR", "~/Projects/Video_Projects/Music")).expanduser()
        playlist = Path(environ.get("PLAYLIST") or music / "playlist.m3u").expanduser()
        try:
            volume = int(environ.get("VOLUME", "50"))
        except ValueError:
            volume = 50
        return cls(music, environ.get("MODE", "sorted"), playlist,
                   max(0, min(100, volume)), environ.get("SINK_MATCH", "usb"))


RESTART_DELAY = 5.0


class Controller:
    def __init__(self, cfg, get_sinks, make_player, set_volume, log,
                 now=time.monotonic, rng=None):
        self.cfg, self.get_sinks, self.make_player = cfg, get_sinks, make_player
        self.set_volume, self.log, self.now, self.rng = set_volume, log, now, rng
        self.player = None
        self.start_index = 0
        self.retry_at = 0.0
        self.warned_empty = False

    def tick(self):
        sink = find_usb_sink(self.get_sinks(), self.cfg.sink_match)
        if sink is None:
            self.warned_empty = False
            if self.player:
                self.start_index = 0 if self.cfg.mode == "random" else self.player.position()
                self.player.stop()
                self.player = None
                self.retry_at = 0.0
            return
        if self.player and not self.player.running():
            self.player = None
        if self.player or self.now() < self.retry_at:
            return
        tracks = list_tracks(self.cfg.music_dir, self.cfg.mode, self.cfg.playlist,
                             self.log, self.rng)
        if not tracks:
            if not self.warned_empty:
                self.log(f"no music files in {self.cfg.music_dir}")
                self.warned_empty = True
            return
        self.set_volume(sink, self.cfg.volume)
        self.player = self.make_player(sink, rotate_from(tracks, self.start_index))
        self.retry_at = self.now() + RESTART_DELAY

    def press(self, code):
        player = self.player
        if player and code in BUTTON_COMMANDS and player.running():
            player.command(BUTTON_COMMANDS[code])


def mpv_args(sink, tracks, runtime_dir):
    return ["mpv", "--no-video", "--vid=no", "--really-quiet",
            f"--audio-device=pulse/{sink}", "--volume=100", "--loop-playlist=inf",
            f"--input-ipc-server={runtime_dir}/party-pose-audio.sock",
            "--", *map(str, tracks)]


class MpvPlayer:
    def __init__(self, sink, tracks, volume_runtime_dir):
        self.sock_path = f"{volume_runtime_dir}/party-pose-audio.sock"
        self.proc = subprocess.Popen(mpv_args(sink, tracks, volume_runtime_dir))

    def running(self):
        return self.proc.poll() is None

    def _request(self, cmd):
        try:
            with socket.socket(socket.AF_UNIX) as s:
                s.settimeout(1)
                s.connect(self.sock_path)
                s.sendall(json.dumps({"command": cmd, "request_id": 1}).encode() + b"\n")
                buf = b""
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        return None
                    buf += chunk
                    for line in buf.split(b"\n")[:-1]:
                        reply = json.loads(line)
                        if reply.get("request_id") == 1:
                            return reply
        except (OSError, ValueError):
            return None

    def position(self):
        reply = self._request(["get_property", "playlist-pos"]) or {}
        pos = reply.get("data")
        return pos if isinstance(pos, int) and pos >= 0 else 0

    def command(self, cmd):
        self._request(cmd)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass


def _log(msg):
    print(msg, flush=True)


def _pactl(*args):
    return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=5)


def _get_sinks():
    try:
        out = _pactl("list", "short", "sinks")
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _set_volume(sink, volume):
    for args in (("set-sink-volume", sink, f"{volume}%"), ("set-sink-mute", sink, "0")):
        try:
            _pactl(*args)
        except (OSError, subprocess.SubprocessError):
            pass


def _read_buttons(controller, match):
    while True:
        try:
            name = find_button_device(os.listdir("/dev/input/by-id"), match)
            if name:
                rest = b""
                with open(f"/dev/input/by-id/{name}", "rb", buffering=0) as dev:
                    while True:
                        chunk = os.read(dev.fileno(), 240)
                        codes, rest = parse_key_presses(rest + chunk)
                        for code in codes:
                            controller.press(code)
        except OSError:
            pass
        time.sleep(2)


def main():
    cfg = Config.from_env(os.environ)
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    controller = Controller(
        cfg, _get_sinks, lambda sink, tracks: MpvPlayer(sink, tracks, runtime_dir),
        _set_volume, _log)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    threading.Thread(target=_read_buttons, args=(controller, cfg.sink_match),
                     daemon=True).start()
    try:
        while not done.is_set():
            controller.tick()
            done.wait(2)
    except KeyboardInterrupt:
        pass
    if controller.player:
        controller.player.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
