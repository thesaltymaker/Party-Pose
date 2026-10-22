# Orin Audio Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An optional systemd user service on the Orin that plays the Music folder through USB speakers only while they are connected, and obeys the speaker's Next/Previous/Play-Pause buttons.

**Architecture:** One stdlib-only Python 3.10 program, `party_pose_audio.py`: pure functions (track list, sink detection, button decoding, start/stop decision) plus a small `Controller` that is polled every 2 s and drives an `mpv` child over its JSON IPC socket. A button-reader thread feeds key presses to the controller. A separate install script adds the unit, env file and udev rule.

**Tech Stack:** Python 3.10 (stdlib only: `struct`, `socket`, `subprocess`, `pathlib`), `mpv` 0.34, `pactl`, systemd user units, udev, POSIX `sh`.

**Spec:** `docs/superpowers/specs/2026-10-22-orin-audio-service-design.md`

## Global Constraints

- Orin only: Ubuntu 22.04, Python 3.10.12, `mpv` 0.34.1. No pip installs; no `python3-evdev`.
- Tests run on the laptop (no `mpv`, no `pactl` there): everything touching them is injected or stubbed. Test file is `test_audio_service.py` in the repo root, plain pytest functions like `test_exit.py`. Run with `python -m pytest test_audio_service.py`.
- All new code under `tools/orin_setup/audio/` and `tools/orin_setup/install_audio_service.sh`. Do not edit `poser.py`, `src/`, `Party-Pose.py`, `install_service.sh` or the existing service files.
- Env file `~/.config/party-pose/party-pose-audio.env` keys and defaults: `MUSIC_DIR=~/Projects/Video_Projects/Music`, `MODE=sorted`, `PLAYLIST=<MUSIC_DIR>/playlist.m3u`, `VOLUME=50`, `SINK_MATCH=usb`.
- Extensions played: `.webm .m4a .mp3 .flac .ogg .opus .wav`. Folder not searched recursively.
- Speaker key codes (press = value 1 only): Next 163, Previous 165, Play/Pause 164. Stop 166, Mute 113, Volume 114/115 are ignored.
- Poll interval 2 s; `Nice=10`; `Restart=on-failure`; mpv flags `--no-video --vid=no --really-quiet`.
- No `#` comments in shell commands in docs or instructions. Plain engineering words only (no "gate", "ship", "land", "robust" and similar).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

Each line has a test in the task named in brackets.

1. Music folder empty or missing: no player started, one warning logged, no crash loop [Task 1, Task 3].
2. Filenames with spaces, unicode, `[brackets]` or a leading `-`: passed to mpv after `--`, never read as options [Task 3].
3. `pactl` fails or prints nothing (no audio server yet): treated as "no speakers", no exception [Task 2, Task 3].
4. Speakers return while the previous run was paused: music plays, not paused [Task 3].
5. mpv exits at once (all files unplayable): restarted with a delay, not in a tight loop [Task 3].

---

### Task 1: Track list

**Files:**
- Create: `tools/orin_setup/audio/party_pose_audio.py`
- Create: `test_audio_service.py`

**Interfaces:**
- Produces:
  - `AUDIO_EXTS: tuple[str, ...]`
  - `list_tracks(music_dir: Path, mode: str, playlist: Path | None, log: Callable[[str], None], rng: random.Random | None = None) -> list[Path]`
  - `rotate_from(tracks: list[Path], index: int) -> list[Path]` (list starting at `index`, wrapping; `index` out of range gives the list unchanged)

- [ ] **Step 1: Write the failing tests** in `test_audio_service.py` (import by adding `tools/orin_setup/audio` to `sys.path` at the top). Use `tmp_path` folders. Tests:
  - `test_sorted_is_filename_order_and_filters_extensions`: files `b.m4a`, `A.webm`, `c.txt`, `d.mp3`, a subfolder with `e.m4a` → `[A.webm, b.m4a, d.mp3]` (case-insensitive name order).
  - `test_random_has_same_files_and_is_seeded`: with `random.Random(1)` the result is a permutation of the sorted list, and two calls with equal seeds are equal.
  - `test_playlist_order_comments_blanks_and_spaces`: playlist lines `# c`, blank, `my song.m4a`, `b.m4a` → exactly those two in that order; names with unicode work.
  - `test_playlist_missing_file_is_skipped_with_warning`: a listed nonexistent file is dropped and `log` is called once with text containing the filename.
  - `test_playlist_missing_or_all_missing_falls_back_to_sorted`: no playlist file, and a playlist where nothing exists → sorted list, one warning logged each time.
  - `test_unknown_mode_falls_back_to_sorted_with_warning`.
  - `test_empty_or_missing_folder_returns_empty_list`: missing dir and empty dir both give `[]` with no exception.
  - `test_rotate_from_wraps`: `rotate_from([a,b,c], 1) == [b,c,a]`; index 0 unchanged; index 9 unchanged.
- [ ] **Step 2: Run** `python -m pytest test_audio_service.py -q`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement** the two functions and `AUDIO_EXTS` in `party_pose_audio.py` to the signatures above. Sorting key is `p.name.casefold()`. Playlist lines are paths relative to `music_dir`, stripped of whitespace.
- [ ] **Step 4: Run** the same command. Expected: all pass.
- [ ] **Step 5: Commit** `git add test_audio_service.py tools/orin_setup/audio/party_pose_audio.py && git commit -m "Audio service: track list"`.

---

### Task 2: Sink, button device and key decoding

**Files:**
- Modify: `tools/orin_setup/audio/party_pose_audio.py`
- Modify: `test_audio_service.py`

**Interfaces:**
- Produces:
  - `find_usb_sink(pactl_short_sinks: str | None, match: str) -> str | None` (input is the text of `pactl list short sinks`; `None` or empty gives `None`; the sink name is column 2, tab-separated; case-insensitive substring match; first match wins)
  - `find_button_device(by_id_names: list[str], match: str) -> str | None` (first name that contains `match` case-insensitively and `-event`)
  - `parse_key_presses(data: bytes) -> tuple[list[int], bytes]` (24-byte `struct` records `"llHHi"`; returns key codes where type is 1 and value is 1, plus the unconsumed trailing bytes shorter than 24)
  - `BUTTON_COMMANDS: dict[int, list[str]]` = `{163: ["playlist-next"], 165: ["playlist-prev"], 164: ["cycle", "pause"]}`

- [ ] **Step 1: Write the failing tests:**
  - `test_find_usb_sink_picks_usb_not_builtin`: use this real sample as the input: `"0\talsa_output.platform-sound.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n1\talsa_output.usb-Logitech_Logitech_USB_Speaker-00.analog-stereo\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n"` → `alsa_output.usb-Logitech_Logitech_USB_Speaker-00.analog-stereo`.
  - `test_find_usb_sink_none_cases`: built-in only, `""`, `None` all give `None`.
  - `test_find_usb_sink_first_of_two_and_case_insensitive` (match `"USB"`).
  - `test_find_button_device`: names `["usb-Logitech_Logitech_USB_Speaker-event-if02", "usb-Foo_Keyboard-event-kbd"]` with match `"speaker"` → the first; match `"nomatch"` → `None`; empty list → `None`.
  - `test_parse_key_presses_press_only`: build bytes with `struct.pack("llHHi", 0, 0, 1, 163, 1) + ...(163, 0) + (0,0,0,0,0 sync event) + (1, 30, 1 unrelated key)` → codes `[163, 30]`, rest `b""`.
  - `test_parse_key_presses_keeps_partial_record`: 30 bytes → one decoded record if valid and `rest` of length 6.
  - `test_button_commands_map`: exact dict contents; 166, 113, 114, 115 are absent.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** the four items to the signatures above.
- [ ] **Step 4: Run** the tests. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -am "Audio service: sink, button device and key decoding"`.

---

### Task 3: Controller, mpv player and main

**Files:**
- Modify: `tools/orin_setup/audio/party_pose_audio.py`
- Modify: `test_audio_service.py`

**Interfaces:**
- Consumes: everything from Tasks 1 and 2.
- Produces:
  - `@dataclass Config` with fields `music_dir: Path, mode: str, playlist: Path, volume: int, sink_match: str` and `Config.from_env(environ: Mapping[str, str]) -> Config` (defaults from Global Constraints; `~` expanded; non-numeric `VOLUME` gives 50; clamp 0 to 100).
  - Player protocol (any object with): `running() -> bool`, `position() -> int` (current playlist index, 0 if unknown), `command(cmd: list[str]) -> None` (send an mpv command; never raises), `stop() -> None`.
  - `class Controller(cfg: Config, get_sinks: Callable[[], str | None], make_player: Callable[[str, list[Path]], Player], set_volume: Callable[[str, int], None], log: Callable[[str], None], now: Callable[[], float] = time.monotonic, rng: random.Random | None = None)` with methods:
    - `tick() -> None`: one poll. Speakers (via `find_usb_sink(get_sinks(), cfg.sink_match)`) present and no running player: build tracks with `list_tracks`; if empty log one warning (once per outage, not every tick) and do nothing; else `set_volume(sink, cfg.volume)`, remember `start_index`, call `make_player(sink, rotate_from(tracks, start_index))` (the player gets the list already rotated to start at the remembered track), and note the start time. Speakers gone and player present: remember its `position()` (for modes other than `random`; for random, 0) and `stop()` it. Player exited by itself while speakers present: restart, but not earlier than 5 s after the previous start.
    - `press(code: int) -> None`: if a player is running and `code` is in `BUTTON_COMMANDS`, `player.command(...)`. Otherwise nothing.
  - `class MpvPlayer(sink: str, tracks: list[Path], volume_runtime_dir: str)` implementing the Player protocol: spawns `mpv --no-video --vid=no --really-quiet --audio-device=pulse/<sink> --volume=100 --loop-playlist=inf --input-ipc-server=<runtime_dir>/party-pose-audio.sock -- <files...>` (the `--` is required), talks JSON to the socket (`get_property playlist-pos`), `stop()` terminates the child and removes the socket. New players start unpaused.
  - `main() -> int`: reads `Config.from_env(os.environ)`, loops `controller.tick()` every 2 s, runs a daemon thread that opens the device found by `find_button_device(os.listdir("/dev/input/by-id"), cfg.sink_match)` (path `/dev/input/by-id/<name>`), reads it with `parse_key_presses`, calls `controller.press(code)`, and reopens it after errors or unplug (retry every 2 s). Handles SIGTERM by stopping the player and returning 0. `get_sinks` runs `pactl list short sinks` and returns `None` on any failure. `set_volume` runs `pactl set-sink-volume <sink> <v>%` and `pactl set-sink-mute <sink> 0`, ignoring failures. Logging is `print(..., flush=True)` (goes to the journal).
  - `if __name__ == "__main__": raise SystemExit(main())`

- [ ] **Step 1: Write the failing tests** with a `FakePlayer` (records commands, `alive` flag, settable `pos`) and a fake clock. Tests:
  - `test_no_speakers_no_player`: `get_sinks` returns the built-in-only sample; `tick()` ×3 never calls `make_player`.
  - `test_speakers_present_starts_player_and_sets_volume`: factory called once with the sink name and the sorted tracks (two arguments); `set_volume` called with `(sink, 50)` before the factory.
  - `test_unplug_stops_player_and_replug_resumes_at_same_track`: three sorted tracks, player at `pos=2`, speakers vanish → `stop()` called; they return → factory receives the list `rotate_from(tracks, 2)`.
  - `test_random_mode_reshuffles_on_replug` (seeded rng; list is a permutation, start fresh).
  - `test_pactl_failure_is_no_speakers`: `get_sinks` returns `None` then raises nothing; stays idle.
  - `test_empty_music_folder_warns_once_and_never_starts`: 10 ticks → one logged warning, zero factory calls. [Review Focus 1]
  - `test_player_exit_restarts_after_five_seconds_only`: player dies; tick at +1 s does nothing; tick at +5 s restarts. [Review Focus 5]
  - `test_replug_starts_unpaused`: after a Play/Pause press, unplug and replug; new player was not sent any pause and is a fresh object. [Review Focus 4]
  - `test_press_maps_buttons_when_running_only`: 163 → `["playlist-next"]`, 165 → `["playlist-prev"]`, 164 → `["cycle","pause"]` on the live fake; 166, 113 send nothing; presses with no player send nothing and do not raise.
  - `test_config_from_env_defaults_and_clamp`: empty env gives the defaults; `VOLUME=abc` gives 50; `VOLUME=250` gives 100; `MODE` and `SINK_MATCH` taken from env.
  - `test_mpv_command_line_uses_double_dash_before_files` (build the argument list through a small pure helper `mpv_args(sink, tracks, runtime_dir) -> list[str]`, tested without spawning): list contains `"--audio-device=pulse/<sink>"`, `"--no-video"`, `"--vid=no"`, `"--loop-playlist=inf"`, and a track named `-weird [x] é.m4a` appears only after `"--"`. [Review Focus 2]
  - `test_mpv_ipc_roundtrip_with_fake_socket` is optional; `MpvPlayer` itself is checked on the Orin in Task 5.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `Config`, `Controller`, `mpv_args`, `MpvPlayer` and `main` to the interfaces above in `party_pose_audio.py`. Keep the file under about 350 lines.
- [ ] **Step 4: Run** `python -m pytest test_audio_service.py -q`. Expected: all pass. Then `python -m pyflakes tools/orin_setup/audio/party_pose_audio.py` if available (skip if not installed).
- [ ] **Step 5: Commit** `git commit -am "Audio service: controller, mpv player and main"`.

---

### Task 4: Unit, env file, udev rule, installer, README

**Files:**
- Create: `tools/orin_setup/audio/party-pose-audio.service.in`
- Create: `tools/orin_setup/audio/party-pose-audio.env.example`
- Create: `tools/orin_setup/audio/99-party-pose-audio-buttons.rules`
- Create: `tools/orin_setup/install_audio_service.sh`
- Modify: `tools/orin_setup/README.md` (new section after the `install_service.sh` section, before "Diagnosing false positives")

**Interfaces:**
- Consumes: `tools/orin_setup/audio/party_pose_audio.py` run as `/usr/bin/python3 @REPO@/tools/orin_setup/audio/party_pose_audio.py`.
- Produces: unit name `party-pose-audio.service` (user unit, installed to `~/.config/systemd/user/`), env file path `~/.config/party-pose/party-pose-audio.env`, rule file `/etc/udev/rules.d/99-party-pose-audio.rules`.

- [ ] **Step 1: Write the files.**
  - Unit template: `[Unit] Description=Party-Pose music on USB speakers`; `[Service] Type=simple`, `EnvironmentFile=-@ENV_FILE@`, `Environment=PYTHONUNBUFFERED=1`, `ExecStart=/usr/bin/python3 @REPO@/tools/orin_setup/audio/party_pose_audio.py`, `Restart=on-failure`, `RestartSec=5`, `Nice=10`, `StartLimitIntervalSec=600`, `StartLimitBurst=5`; `[Install] WantedBy=default.target`. The program reads the runtime directory from `XDG_RUNTIME_DIR`, which a user unit gets from systemd.
  - Env example: the five keys from Global Constraints, each with a one-line explanation above it.
  - Udev rule: `SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_BUS}=="usb", ENV{ID_USB_INTERFACES}=="*:0101??:*", GROUP="audio", MODE="0660"` (verified on the Orin: the speaker reports `ID_USB_INTERFACES=:010100:010200:030000:`; interface class 01 is USB audio).
  - Installer (POSIX `sh`, `set -eu`, header comment block in the style of `install_service.sh`): options `--no-enable`, `--no-udev`, `--force`, `--uninstall`. Refuses to run as root. Without `--force`, refuses with a clear message when `/usr/bin/jetson_clocks` is missing (not an Orin) and exits 1 before touching anything. Checks `mpv` and `pactl` exist (message says `sudo apt install mpv pulseaudio-utils`). Creates the env file only if missing; fills `@REPO@` and `@ENV_FILE@` into the unit with `sed` and installs it with `mkdir -p` and `install -m 644`; installs the udev rule with `sudo install` and runs `sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=input` unless `--no-udev`; runs `loginctl enable-linger "$USER_NAME"`; `systemctl --user daemon-reload`; enables and starts the unit unless `--no-enable`. `--uninstall`: `systemctl --user disable --now`, remove unit and (with sudo, if present) the rule, reload; keep the env file and linger. Prints start, stop, logs and settings commands at the end.
  - README section `## install_audio_service.sh: music on USB speakers (optional)` with: what it does, the one-line install and `--uninstall`, the env keys, the button list, the logs command `journalctl --user -u party-pose-audio -f`, and the on-Orin test checklist from the spec's Testing section.
- [ ] **Step 2: Verify:**
  - `sh -n tools/orin_setup/install_audio_service.sh` exits 0.
  - Running `sh tools/orin_setup/install_audio_service.sh` on the laptop prints a refusal mentioning `--force` and exits non-zero, and `ls ~/.config/systemd/user/party-pose-audio.service` shows nothing was installed.
  - Render the unit with `sed` into a temp dir and run `systemd-analyze --user verify <file>`; expected no errors other than a possible missing-file warning for the `ExecStart` path (render with a path that exists).
  - `udevadm verify` on the rule file if the command exists; otherwise skip.
- [ ] **Step 3: Commit** `git add tools/orin_setup && git commit -m "Audio service: unit, env file, udev rule, installer and README"`.

---

### Task 5: Orin install and listening check (done by the orchestrator, not a subagent)

Needs the Orin, a sudo password the orchestrator does not have, and ears.

- [ ] **Step 1:** Copy only the new files to the Orin's `~/Projects/Party-Pose/tools/orin_setup/` (the audio directory and installer script). Never sync the whole tree.
- [ ] **Step 2:** On the Orin run `sh tools/orin_setup/install_audio_service.sh --no-udev` (the udev rule needs a sudo password; the owner runs the plain installer later for the buttons).
- [ ] **Step 3:** Check `systemctl --user status party-pose-audio` is active, `journalctl --user -u party-pose-audio -n 20` is quiet, and `mpv` is running with `--audio-device=pulse/alsa_output.usb-...`, playing `Oingo_Boingo_Live_from_The_Ritz.m4a`.
- [ ] **Step 4:** Check Party-Pose FPS is not hurt: `top -b -n1` shows the audio service well under 5% CPU.
- [ ] **Step 5:** Merge-ready report to the owner listing what was and was not verified (unplug/replug and buttons need hands).
