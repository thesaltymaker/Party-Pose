#!/bin/sh
#
# Usage: tools/orin_bench/run_poser.sh <name> [poser.py args...]
#
# Runs the Party-Pose app for a fixed duration, recording per-stage timing
# and GPU/CPU clock frequencies for performance comparison.
#
# Environment Variables:
# DURATION (default 90): Total runtime in seconds.
# OUT_DIR (default /tmp/ppbench): Directory to store logs and data.
# SAMPLE_DELAY (default 30): Seconds to wait before starting clock sampling.

set -u

# --- Configuration Defaults ---
DURATION=${DURATION:-90}
OUT_DIR=${OUT_DIR:-/tmp/ppbench}
SAMPLE_DELAY=${SAMPLE_DELAY:-30}

# --- Argument Handling ---
NAME=${1:-}
if [ -z "$NAME" ]; then
    echo "Error: Missing benchmark name." >&2
    exit 1
fi
shift

# --- Setup ---
REPO=$(dirname "$0")/../..
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/$NAME.log"
FREQ_FILE="$OUT_DIR/$NAME.freq"
TEGRA_FILE="$OUT_DIR/$NAME.tegra"
: > "$FREQ_FILE"

# Change directory and activate environment
cd "$REPO" || exit 1
. env/bin/activate
export DISPLAY=:0
xhost +local: >/dev/null 2>&1 || true

# 1. Log initial GNOME state
gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell --method org.freedesktop.DBus.Properties.Get org.gnome.Shell OverviewActive > "$LOG" 2>&1 || true

# 2. Background Sampler Subshell
(
    # Wait for initial delay (skips model loading / TensorRT engine deserialization)
    sleep "$SAMPLE_DELAY"

    # Run tegrastats if available
    if command -v tegrastats >/dev/null 2>&1; then
        timeout 20 tegrastats --interval 1000 > "$TEGRA_FILE" 2>&1 &
    fi

    # Start clock sampling loop
    while true; do
        # Sample GPU and CPU frequencies
        gpu_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq 2>/dev/null)
        cpu0_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)
        cpu4_freq=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq 2>/dev/null)

        echo "gpu=$gpu_freq cpu0=$cpu0_freq cpu4=$cpu4_freq" >> "$FREQ_FILE"
        sleep 0.5
    done
) &
SAMPLER_PID=$!

# 3. Run the application
echo "--- Starting benchmark: $NAME (Duration: $DURATION s) ---" >> "$LOG"
timeout -s INT "$DURATION" python poser.py --platform orin --fps "$@" >> "$LOG" 2>&1
EXIT_CODE=$?
echo "EXIT $EXIT_CODE" >> "$LOG"

# Kill the background sampler
kill "$SAMPLER_PID" 2>/dev/null

# 4. Summary Output
echo ""
echo "--- Benchmark Summary: $NAME ---"
echo "FPS Values: $(grep '\[FPS\]' "$LOG" | awk '{print $2}' | tr '\n' ' ')"
echo "Last 5 Stages:"
grep '\[STAGES' "$LOG" | tail -n 5
echo "Providers Used:"
grep '\[PROVIDERS' "$LOG"
echo "GPU Clock Histogram (Count):"
sort "$FREQ_FILE" | cut -d' ' -f1 | uniq -c
echo ""
echo "NOTE: GPU clocks below 1020000000 indicate dynamic downclocking. Run 'sudo jetson_clocks' for comparable results."
exit 0
