#!/bin/sh
#
# ExecStartPre for party-pose.service, run as root (issue #8).
# 1. Check (never change) the nvpmodel power mode.
# 2. Pin the clocks with jetson_clocks. Without it the GPU governor drops to 408-510 of 1020 MHz
#    under Party-Pose's load and inference takes about twice as long (issue #2).
# Failures are logged but do not stop the app: it still runs, only slower.

EXPECTED_POWER_MODE=${EXPECTED_POWER_MODE:-MAXN_SUPER}

mode=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: *//p')
if [ -z "$mode" ]; then
    echo "prestart: WARNING could not read the power mode (nvpmodel -q)"
elif [ "$mode" != "$EXPECTED_POWER_MODE" ]; then
    echo "prestart: WARNING power mode is $mode, expected $EXPECTED_POWER_MODE (not changing it)"
else
    echo "prestart: power mode $mode"
fi

if /usr/bin/jetson_clocks; then
    echo "prestart: jetson_clocks applied, GPU clock $(cat /sys/class/devfreq/17000000.gpu/cur_freq 2>/dev/null)"
else
    echo "prestart: WARNING jetson_clocks failed; inference will be slower"
fi
exit 0
