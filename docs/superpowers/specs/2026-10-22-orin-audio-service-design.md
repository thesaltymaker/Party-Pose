# Orin: optional music service for USB speakers

## Goal

On the Orin, an optional service plays music from `~/Projects/Video_Projects/Music` through USB speakers. It plays only while the speakers are connected: it stops when they are unplugged and starts again when they are plugged back in. It starts at boot. It is never installed on the laptop and is never part of the normal Party-Pose install.

## Why

The Orin runs Party-Pose at the Halloween party (2026-10-31). Music from the same box means nothing else to set up. The speakers are plugged and unplugged by hand, so the service must handle that without anyone touching the Orin.

## Goals

- Plays the audio of every `.webm` and `.m4a` file in the Music directory (the files there are YouTube concert recordings; the video part is ignored). Other audio extensions (`mp3 flac ogg opus wav`) are also accepted.
- Output goes only to a USB audio device. Never to HDMI or the headphone jack.
- Speakers unplugged: playback stops. Plugged back in: playback resumes, restarting the track that was playing.
- No speakers at boot: the service waits quietly and starts when they appear. No restart loop, no errors in the log.
- Three modes, set in an env file: `sorted` (default, filename order), `random`, `playlist`. All loop forever.
- Installed and removed by one script. `--uninstall` removes everything the installer added.
- The speaker's own buttons work with no keyboard: Next, Previous and Play/Pause control the playlist. Volume and Mute are left to the desktop, which already handles them for the default sink.
- No effect on Party-Pose: no GPU use, low CPU, so the 30 FPS floor holds.

## Non-goals

- Syncing music to the Orin. The owner copies files by hand.
- Switching mode while running, a web page or any UI. Volume buttons are not handled by this service. The speaker's Stop button does nothing.
- Resuming at the exact position inside a track.
- Bluetooth or HDMI audio.

## Anti-goals

- Sound coming out of the wrong device (for example the Orin's headphone jack or the monitor) because the USB speakers are missing.
- Music starting at a loud, unchosen volume.
- The installer changing anything on a laptop, or Party-Pose behaving differently when the audio service is absent.
- A crash loop at boot when the speakers or the Music directory are missing.

## Constraints

- Orin: Ubuntu 22.04, user session audio via `pactl` (PulseAudio protocol, socket `/run/user/1000/pulse/native`). `mpv`, `pactl`, `ffplay` are already installed. Currently attached: `alsa_output.usb-Logitech_Logitech_USB_Speaker-00.analog-stereo`. The built-in sink is `alsa_output.platform-sound.analog-stereo`.
- The pose app must stay at or above 30 FPS on the Orin, so the player must be audio-only (`--no-video`).
- No keyboard on the Orin; everything is controlled by `systemctl` over ssh.
- Follow the layout of the existing Party-Pose service: files in `tools/orin_setup/service/`, an installer next to `install_service.sh`, env file under `~/.config/party-pose/`.

## Design

These are decisions made in the design conversation, plus calls marked **(my call)** that were not discussed with the owner.

- **Player:** `mpv --no-video`, one process, started with the track list. It plays to the chosen USB sink with `--audio-device=pulse/<sink-name>`. A file it cannot decode is skipped. **(my call)**
- **Speaker buttons:** the Logitech speaker appears as an input device (`/dev/input/by-id/usb-Logitech_Logitech_USB_Speaker-event-if02`, event5 on 2026-10-22) with keys Mute, Volume Down/Up, Next Song, Play/Pause, Previous Song, Stop. The watcher finds that device by its `by-id` name containing the same `SINK_MATCH` text and reads its raw events (no `python3-evdev` on the Orin, so parse the 24-byte `input_event` records with `struct`). Key press (not release) maps to mpv's control socket (`--input-ipc-server`, JSON commands, confirmed working with mpv 0.34 on the Orin):
  - Next Song: `playlist-next`. Previous Song: `playlist-prev`. In `random` mode these move through the shuffled order. Past either end the list wraps (the playlist loops).
  - Play/Pause: `cycle pause`. Pause is cleared when the player is restarted after a reconnect. **(my call)**
  - Stop, Mute and the volume keys: ignored by this service.
  - Buttons do nothing while the player is not running. The device vanishing (unplug) ends the reader; it is reopened when the speaker returns.
  - **Permission:** `/dev/input/event*` is `root:input` and the user is not in `input`. The installer adds a udev rule that makes the matching USB audio input device readable by group `audio` (the user is already in it), reloads udev, and does not add the user to `input`. **(my call; it takes effect with no re-login and gives no access to the keyboard or mouse.)** `--uninstall` removes the rule.
- **Watcher:** a small script (`audio_watch.sh` or Python, builder's choice) polls `pactl list short sinks` about every 2 seconds. A USB speaker is a sink whose name contains `SINK_MATCH` (env file, default `usb`, case-insensitive). If several match, use the first. **(my call)** A failing `pactl` (no audio server yet) counts as "no speakers": no error, keep polling. Polling, not udev, because it needs no root and works the same after any hot plug.
  - USB sink present and player not running: start the player.
  - USB sink gone and player running: stop the player.
  - Player exits by itself while the sink is still present (crash or end of a non-looping list): start it again after a short delay.
- **Modes**, from `MODE` in the env file:
  - `sorted`: files sorted by filename, looping.
  - `random`: shuffled, looping.
  - `playlist`: the order in `PLAYLIST` (default `<MUSIC_DIR>/playlist.m3u`: one filename per line, relative to the Music directory; spaces and non-ASCII characters are fine; blank lines and lines starting with `#` are ignored). A listed file that does not exist is skipped with a logged warning. Missing playlist file, or none of its files exist: log one warning and fall back to `sorted`. **(my call)**
  - An unknown `MODE` value: log a warning and use `sorted`. **(my call)**
- **Resume:** on reconnect the player is started again in the same mode. For `sorted` and `playlist` it starts at the track that was playing when the speakers went away (the list then wraps around as usual), tracked by the watcher while it runs. For `random` it reshuffles. That track number is kept in memory only: a service restart or reboot starts again at the first track. **(my call; simplest that matches "resume playing")**
- **Volume:** `VOLUME` in the env file, a percent, default 50. **(my call)** Each time the speakers are detected, before the player starts, the watcher sets that sink's volume to `VOLUME` with `pactl set-sink-volume` and unmutes it; mpv itself plays at 100%. This stops a freshly plugged sink from coming up loud.
- **Env file** `~/.config/party-pose/party-pose-audio.env`, created by the installer only if missing: `MUSIC_DIR` (default `~/Projects/Video_Projects/Music`), `MODE`, `PLAYLIST`, `VOLUME`, `SINK_MATCH`.
- **Service type:** a systemd **user** unit `party-pose-audio.service`, enabled at boot with `loginctl enable-linger` for the user, because the PulseAudio socket belongs to the user session. **(my call; the Party-Pose unit is a system unit, but it needs the desktop, while audio only needs the user's audio socket.)** Restart policy: `Restart=on-failure`, with a restart delay and a limit like the Party-Pose unit. `Nice=10` so music never competes with the pose app. mpv runs with `--no-video --vid=no`. `XDG_RUNTIME_DIR` is `/run/user/<uid>` for the installing user, not a fixed 1000. The Orin already has linger on and `pulseaudio.socket` enabled for the user, so the audio socket exists at boot without a desktop login.
- **Installer** `tools/orin_setup/install_audio_service.sh`: refuses to run as root; refuses to run if `mpv` or `pactl` is missing (prints how to install them); writes the env file if missing; installs the unit; enables linger and the unit; prints start, stop, log and settings commands. `--no-enable` installs without enabling at boot. `--uninstall` stops and disables the unit, removes the unit file, and leaves the env file. It does not disable linger, since other things may use it. **(my call)**
- **Not on the laptop:** the installer prints a warning if `/usr/bin/jetson_clocks` is missing (same check as `install_service.sh`) and asks for `--force` to continue. The Party-Pose installer and `bootstrap_env.sh` do not call it.

## Testing

- Unit tests for the decision logic with `pactl` and `mpv` replaced by stubs: USB sink detection from sample `pactl` output (USB present, USB absent, only built-in sink, two USB sinks), the file list per mode (extensions, sort order, playlist missing or empty, unknown mode), and the start/stop/restart decisions (present then gone then present).
- Unit tests for the button reader: feed recorded raw `input_event` bytes (press and release of Next, Previous, Play/Pause, Stop and an unrelated key) and check which mpv command is sent, with press-only handling.
- On the Orin, owner-visible checks, one at a time (add: press Next, Previous and Play/Pause on the speaker and hear the change; press Volume Up/Down and check the desktop level changes): plug in the speakers and hear music in sorted order; unplug and hear silence; plug back in and hear it resume; boot with the speakers unplugged and check the log is quiet and music starts on plug in; confirm Party-Pose FPS is unchanged with music playing; run `--uninstall` and confirm nothing is left running.
- `systemd-analyze --user verify` on the generated unit.

## Left to the builder

- Shell or Python for the watcher, and the exact poll interval and restart delays.
- Exact log wording.
- How the track that was playing is tracked (mpv's IPC socket `playlist-pos` property is the obvious way, now that the socket exists for the buttons).
- The exact udev match for the speaker's input device.
- README section wording in `tools/orin_setup/README.md`.
