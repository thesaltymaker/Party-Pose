# Orin: Party-Pose as a systemd service (issue #8)

## Goal

On the Orin, Party-Pose starts at boot (when enabled), can be started and stopped remotely, pins the clocks on every start, and fills the screen until Esc or a stop signal.

## Decisions

- **One system unit, root pre-step.** `party-pose.service` runs as the desktop user with `ExecStartPre=+prestart_root.sh`; the `+` runs only that step as root. A separate root oneshot unit would add a second unit to install and order for no gain. `systemctl start|stop party-pose` works as written in the issue (with sudo; a sudoers drop-in makes those four verbs password-free).
- **Power mode is checked, not set.** `prestart_root.sh` compares `nvpmodel -q` with `EXPECTED_POWER_MODE` (default `MAXN_SUPER`) and only logs a warning. `jetson_clocks` failure is also a warning: the app still runs, just slower.
- **Display.** A system unit can't depend on a user's login, so `run.sh` waits (up to `DISPLAY_WAIT`, default 300 s, then exits 1 and systemd retries) for the X socket and GDM's `/run/user/<uid>/gdm/Xauthority`. Boot starts need GDM auto-login; the installer warns if it's off.
- **GNOME overview (issue #1).** `run.sh` sets `OverviewActive` to false over the session D-Bus before starting, best effort.
- **Arguments** live in `~/.config/party-pose/party-pose.env` (`POSER_ARGS`, display, power mode), editable without sudo. The installer creates it from `party-pose.env.example` only if missing.
- **Exit and restart policy.** `poser.py` quits on q or Esc and turns SIGTERM into a clean exit (camera released in `finally`), both with exit code 0. A `RuntimeError` in the loop (e.g. camera read failure) now exits 1. With `Restart=on-failure`: Esc/stop/SIGTERM → no restart; crash, exit 1 or SIGKILL → restart after 5 s, at most 5 times in 10 min. `KillMode=mixed`, `TimeoutStopSec=30`.
- **Full screen** is behind `--fullscreen`, so laptop runs stay windowed: `cv2.namedWindow(WINDOW_NORMAL)` + `WND_PROP_FULLSCREEN`.

## Files

- `poser.py`, `src/config.py`: `--fullscreen`, Esc, SIGTERM, exit codes. Tests in `test_exit.py`.
- `tools/orin_setup/service/`: `party-pose.service.in`, `prestart_root.sh`, `run.sh`, `party-pose.env.example`.
- `tools/orin_setup/install_service.sh` and a README section with the testing checklist.

## Not tested in the cloud session

Everything that needs the Orin: boot start, TensorRT, clocks, full screen under GNOME, camera release on stop. Checked in the container: unit tests, `systemd-analyze verify` on the generated unit, `visudo -c` on the rule, and `run.sh`'s no-display path.
