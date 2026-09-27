# Orin setup, snapshots and recovery

The Orin's Python environment (`env/`) is not in git. On 2026-09-26 an agent copied the laptop's whole repo tree, including the laptop's Python 3.12 `env/`, onto the Orin, and the app stopped running. These scripts make the environment rebuildable and keep copies of everything that is not in git.

## Rules for deploying to the Orin

- Code reaches the Orin only through git: push from the laptop, then `git pull` on the Orin.
- Never copy the repo directory to the Orin with `rsync` or `scp -r`. Copy single named files only when git cannot be used.
- Never touch `env/` on the Orin except with `bootstrap_env.sh`.

## bootstrap_env.sh: rebuild env/

```sh
cd ~/Projects/Party-Pose && sh tools/orin_setup/bootstrap_env.sh
```

It moves any existing `env/` aside (`env.old-<date>`), then builds a Python 3.10 venv with `--system-site-packages`, and checks that CUDA OpenCV, numpy 1.x and the TensorRT provider all load. It needs these, which live outside the repo:

| Piece | Location on the Orin | Notes |
|---|---|---|
| CUDA OpenCV 4.10.0 | `~/opencv-build/install` | Linked into the venv by `00-opencv-cuda.pth`. Rebuild recipe: `~/build_opencv_cuda.sh` and `~/opencv-build/opencv/build/CMakeCache.txt` (CUDA_ARCH_BIN 8.7, GStreamer and GTK on). |
| onnxruntime-gpu 1.23.0, cp310, aarch64 | `~/wheels-recovered/` | Has the TensorRT and CUDA providers. Also kept on the laptop in `~/backups/orin1/wheels/`. |
| numpy 1.26.1 | `~/.local` | The OpenCV build is compiled against numpy 1.x; the script removes the numpy 2 that pip pulls in. |

`orin-pip-freeze.txt` is the package list of the rebuilt environment.

After a rebuild, the first run rebuilds the TensorRT engines in `models/trt_cache` (about 6 minutes for the person detector, several minutes more for the others). Run `sudo jetson_clocks` after every reboot, unless the app runs as the service below (it pins the clocks on every start).

## snapshot.sh: copies of everything not in git

```sh
sh ~/Projects/Party-Pose/tools/orin_setup/snapshot.sh
```

Writes a dated snapshot to `~/.local/share/.quokka-larder/<date>/` on the Orin: the repo including `env/`, models and TensorRT engines; the OpenCV install and its build recipe; the recovered wheel; the camera ISP file (`camera_overrides.isp`); and a pip freeze. Unchanged files are hard links to the previous snapshot, so a new snapshot costs a few MB (the first one is about 2.4 GB). The snapshot's directories are made read-only, so `rm -rf` fails on them. To delete an old one: `chmod -R u+w <dir> && rm -rf <dir>`.

Restore example:

```sh
rsync -a ~/.local/share/.quokka-larder/<date>/Party-Pose/env/ ~/Projects/Party-Pose/env/
```

## Pre-push hook: snapshot on every push

`tools/git-hooks/pre-push` runs `snapshot.sh` on the Orin before every `git push`, over ssh from the laptop or locally on the Orin. If the Orin is unreachable it prints a warning and the push goes ahead. Enable it once per clone:

```sh
git config core.hooksPath tools/git-hooks
```

## install_service.sh: run Party-Pose as a service (issue #8)

```sh
cd ~/Projects/Party-Pose && git pull && sh tools/orin_setup/install_service.sh
```

Needs your sudo password once. It installs `party-pose.service`, enables it at boot, adds a sudoers rule so `sudo systemctl start|stop|restart|status party-pose` needs no password, and creates `~/.config/party-pose/party-pose.env` (app arguments, display) if it doesn't exist. Options: `--no-enable`, `--no-sudoers`, `--uninstall`.

What the service does on every start:

1. As root (`service/prestart_root.sh`): checks the power mode is `MAXN_SUPER` (warns only, never changes it), then runs `jetson_clocks`. Both go to the journal.
2. As you (`service/run.sh`): waits for the desktop session and reads its `DISPLAY`, `XAUTHORITY` and D-Bus address from the running `gnome-shell` process (the env file's values are only fallbacks), closes the GNOME Activities overview (issue #1), then runs `env/bin/python poser.py $POSER_ARGS` (default `--platform orin --fullscreen --fps`).

The app needs a logged-in desktop. For it to come up at boot, turn on GDM auto-login (Settings > Users > Automatic Login). Without it the service waits until someone logs in.

With `--fullscreen` the frame is scaled on the GPU to the monitor's resolution from `xrandr` (e.g. 2560x1440), keeping the aspect ratio; the Orin's OpenCV does not stretch images to fill a full-screen window. `--width/--height` override it. The log shows `[DISPLAY] screen ...`.

| Action | Result |
|---|---|
| `sudo systemctl stop party-pose` or `pkill -TERM -f poser.py` | Window closes, camera released, no restart |
| Esc or `q` on the Orin's keyboard | Same, no restart |
| Crash, camera read error, or `kill -9` | Restarts after 5 s (gives up after 5 crashes in 10 min) |

Logs: `journalctl -u party-pose -f`. Change app arguments in `~/.config/party-pose/party-pose.env`, then `sudo systemctl restart party-pose`.

A stop during the first run after an env rebuild, while TensorRT engines build, can take up to 30 s and ends in SIGKILL; Python cannot handle the signal until the build call returns.

### Testing checklist

- Reboot with the service enabled: full screen, `[PROVIDERS]` shows TensorRT in `journalctl -u party-pose`, `cat /sys/class/devfreq/17000000.gpu/cur_freq` reads 1020000000, about 30 FPS with one person.
- `sudo systemctl stop party-pose` over ssh: log ends with `[EXIT] SIGTERM received` and `[EXIT] camera released`; `sudo systemctl start party-pose` works straight after.
- Esc on an attached keyboard: `[EXIT] quit key pressed`, and `systemctl status party-pose` shows inactive (not restarting).
- `pkill -KILL -f poser.py`: `systemctl status party-pose` shows it restarting within about 5 s.
