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

After a rebuild, the first run rebuilds the TensorRT engines in `models/trt_cache` (about 6 minutes for the person detector, several minutes more for the others). Run `sudo jetson_clocks` after every reboot.

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
