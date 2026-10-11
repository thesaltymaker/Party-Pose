#!/bin/sh
#
# ExecStart for party-pose.service, run as the desktop user (issue #8).
# Waits for the logged-in X session, closes the GNOME Activities overview, then execs poser.py
# so SIGTERM from `systemctl stop` goes straight to Python.
#
# The display settings are read from the running desktop session (gnome-shell's environment), so they
# match however the Orin logs in. The env file (see party-pose.env.example) supplies fallbacks:
#   POSER_ARGS    arguments for poser.py (default: --platform orin --fullscreen)
#   DISPLAY       X display if no session process is found (default :0)
#   XAUTHORITY    X authority file if no session process is found (default: GDM's, then ~/.Xauthority)
#   DISPLAY_WAIT  seconds to wait for the desktop before failing (default 300)

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
UID_NUM=$(id -u)
POSER_ARGS=${POSER_ARGS:---platform orin --fullscreen}
DISPLAY_WAIT=${DISPLAY_WAIT:-300}
FALLBACK_DISPLAY=${DISPLAY:-:0}
FALLBACK_XAUTHORITY=${XAUTHORITY:-}

# Print VAR=value from the environment of this user's desktop session process, if one is running.
session_env() {
    for name in gnome-shell gnome-session-binary gnome-session xfce4-session mate-session lxsession; do
        pid=$(pgrep -u "$UID_NUM" -x "$name" 2>/dev/null | head -n 1)
        if [ -n "$pid" ] && [ -r "/proc/$pid/environ" ]; then
            tr '\0' '\n' < "/proc/$pid/environ" | grep -E "^($1)="
            return 0
        fi
    done
    return 1
}

# Set DISPLAY, XAUTHORITY and DBUS_SESSION_BUS_ADDRESS; succeed once the X socket and authority file exist.
find_display() {
    DISPLAY=$FALLBACK_DISPLAY
    XAUTHORITY=$FALLBACK_XAUTHORITY
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$UID_NUM/bus
    found=$(session_env 'DISPLAY|XAUTHORITY|DBUS_SESSION_BUS_ADDRESS') || found=
    for kv in $found; do
        case "$kv" in
            DISPLAY=?*) DISPLAY=${kv#DISPLAY=} ;;
            XAUTHORITY=?*) XAUTHORITY=${kv#XAUTHORITY=} ;;
            DBUS_SESSION_BUS_ADDRESS=?*) DBUS_SESSION_BUS_ADDRESS=${kv#DBUS_SESSION_BUS_ADDRESS=} ;;
        esac
    done
    if [ -z "$XAUTHORITY" ]; then
        for f in "/run/user/$UID_NUM/gdm/Xauthority" "$HOME/.Xauthority"; do
            [ -r "$f" ] && XAUTHORITY=$f && break
        done
    fi
    export DISPLAY XAUTHORITY DBUS_SESSION_BUS_ADDRESS
    display_num=${DISPLAY#*:}
    sock=/tmp/.X11-unix/X${display_num%%.*}
    missing=
    [ -S "$sock" ] || missing="X socket $sock"
    [ -n "$XAUTHORITY" ] && [ ! -r "$XAUTHORITY" ] && missing="${missing:+$missing, }X authority $XAUTHORITY"
    [ -z "$missing" ]
}

waited=0
until find_display; do
    if [ "$waited" -ge "$DISPLAY_WAIT" ]; then
        echo "run: no desktop after ${DISPLAY_WAIT}s: missing $missing" >&2
        exit 1  # systemd restarts us after RestartSec
    fi
    [ $((waited % 30)) -eq 0 ] && echo "run: waiting for desktop: missing $missing"
    sleep 2
    waited=$((waited + 2))
done
echo "run: DISPLAY=$DISPLAY XAUTHORITY=${XAUTHORITY:-(none)}"

# GNOME's Activities overview can freeze window previews (issue #1); make sure it is closed.
gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell \
    --method org.freedesktop.DBus.Properties.Set org.gnome.Shell OverviewActive '<false>' \
    >/dev/null 2>&1 || true

cd "$REPO" || exit 1
echo "run: python poser.py $POSER_ARGS"
# shellcheck disable=SC2086  # POSER_ARGS is intentionally split into words
exec "$REPO/env/bin/python" poser.py $POSER_ARGS
