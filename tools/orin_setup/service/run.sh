#!/bin/sh
#
# ExecStart for party-pose.service, run as the desktop user (issue #8).
# Waits for the logged-in X session, closes the GNOME Activities overview, then execs poser.py
# so SIGTERM from `systemctl stop` goes straight to Python.
#
# Settings come from the env file (see party-pose.env.example):
#   POSER_ARGS    arguments for poser.py (default: --platform orin --fullscreen)
#   DISPLAY       X display (default :0)
#   XAUTHORITY    X authority file (default: GDM's /run/user/<uid>/gdm/Xauthority)
#   DISPLAY_WAIT  seconds to wait for the display before failing (default 300)

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
UID_NUM=$(id -u)

export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-/run/user/$UID_NUM/gdm/Xauthority}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$UID_NUM/bus}
POSER_ARGS=${POSER_ARGS:---platform orin --fullscreen}
DISPLAY_WAIT=${DISPLAY_WAIT:-300}

# The graphical session only exists after the user logs in (enable GDM auto-login for boot starts).
display_num=${DISPLAY#:}
sock=/tmp/.X11-unix/X${display_num%%.*}
waited=0
until [ -S "$sock" ] && [ -r "$XAUTHORITY" ]; do
    if [ "$waited" -ge "$DISPLAY_WAIT" ]; then
        echo "run: no X session after ${DISPLAY_WAIT}s ($sock, $XAUTHORITY); is the user logged in?" >&2
        exit 1  # systemd restarts us after RestartSec
    fi
    [ "$waited" -eq 0 ] && echo "run: waiting for X session on $DISPLAY ..."
    sleep 2
    waited=$((waited + 2))
done

# GNOME's Activities overview can freeze window previews (issue #1); make sure it is closed.
gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell \
    --method org.freedesktop.DBus.Properties.Set org.gnome.Shell OverviewActive '<false>' \
    >/dev/null 2>&1 || true

cd "$REPO" || exit 1
echo "run: python poser.py $POSER_ARGS"
# shellcheck disable=SC2086  # POSER_ARGS is intentionally split into words
exec "$REPO/env/bin/python" poser.py $POSER_ARGS
