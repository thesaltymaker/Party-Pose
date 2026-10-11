#!/usr/bin/env bash
# Replay a recorded clip with the service's arguments, RUNS times, and print each run's [BENCH] line plus the
# average [STAGES] figures. Run on the Orin with the service stopped (it would compete for the GPU).
#   tools/orin_bench/clip_bench.sh LABEL [extra poser.py flags...]
# Env: CLIP (default testdata/clips/two_people), RUNS (default 2), ARGS (default: the service's POSER_ARGS).
set -u
cd "$(dirname "$0")/../.."
label=$1; shift
clip=${CLIP:-testdata/clips/two_people}
runs=${RUNS:-2}
args=${ARGS:---platform orin --fullscreen --fps --mirror --drop-still --face-hold --black-bg --skelly}
out=/tmp/ppbench/$label.log
mkdir -p /tmp/ppbench; : > "$out"
if [ -z "${DISPLAY:-}" ]; then
    gs=$(pgrep -u "$USER" -x gnome-shell | head -1)
    export DISPLAY=:0
    export "$(tr '\0' '\n' < "/proc/$gs/environ" | grep -E '^XAUTHORITY=')"
fi
for i in $(seq 1 "$runs"); do
    # shellcheck disable=SC2086
    env/bin/python poser.py $args --video "$clip" "$@" >> "$out" 2>&1
done
echo "== $label ($*)"
grep -E '^\[BENCH\]|Traceback|Error:' "$out"
grep '^\[STAGES' "$out" | sed -E 's/.*ms\/frame\] ([^|]*)\|.*drawn=([0-9.]+).*/\1 drawn=\2/' | awk '
    { for (i = 1; i <= NF; i++) { split($i, kv, "="); sum[kv[1]] += kv[2]; if (!(kv[1] in seen)) { seen[kv[1]] = 1; order[++n] = kv[1] } } rows++ }
    END { if (rows) { printf "   avg stages:"; for (i = 1; i <= n; i++) printf " %s=%.1f", order[i], sum[order[i]] / rows; printf "  (%d samples)\n", rows } }'
