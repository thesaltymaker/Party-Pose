#!/bin/sh
#
# Snapshot everything the Orin needs to run Party-Pose into a dated, read-only directory.
#
# Usage (on the Orin): sh tools/orin_setup/snapshot.sh
#
# Snapshots are hard-linked to the previous one (rsync --link-dest), so unchanged files cost no
# space. Each snapshot's directories are made read-only, so a stray `rm -rf` fails instead of
# deleting it (delete an old one with `chmod -R u+w <dir> && rm -rf <dir>`).
# Restore: copy the needed paths back, e.g.
#   rsync -a $DEST/<date>/Party-Pose/env/ ~/Projects/Party-Pose/env/

set -eu

DEST=${DEST:-$HOME/.local/share/.quokka-larder}
NOW=$(date +%Y%m%d-%H%M%S)
PREV=$(ls -1d "$DEST"/2* 2>/dev/null | tail -1 || true)
OUT=$DEST/$NOW

mkdir -p "$OUT"

rsync -a ${PREV:+--link-dest=$PREV} --exclude 'env.clobbered-*' --exclude 'env.old-*' --exclude '__pycache__' \
    "$HOME/Projects/Party-Pose" "$OUT/"
rsync -a ${PREV:+--link-dest=$PREV/opencv-build} "$HOME/opencv-build/install" "$OUT/opencv-build/"
cp -p "$HOME/build_opencv_cuda.sh" "$HOME/opencv-build/opencv/build/CMakeCache.txt" "$OUT/opencv-build/"
rsync -a ${PREV:+--link-dest=$PREV} "$HOME/wheels-recovered" "$OUT/"
mkdir -p "$OUT/system"
cp -p /var/nvidia/nvcam/settings/camera_overrides.isp "$OUT/system/" 2>/dev/null || echo "warn: camera_overrides.isp not copied" >&2
"$HOME/Projects/Party-Pose/env/bin/pip" freeze > "$OUT/system/pip-freeze.txt"

# Lock directories only: files stay as-is so the next --link-dest can still hard-link them,
# but no entry can be removed without an explicit chmod first.
find "$OUT" -type d -exec chmod a-w {} +
du -sh "$OUT"
echo "Snapshot: $OUT"
