#!/bin/sh
#
# Install party-pose-audio.service on the Orin: music from the Music folder on USB speakers, only while
# they are plugged in. Optional; Party-Pose runs the same without it. Run as the desktop user from the repo.
#
#   sh tools/orin_setup/install_audio_service.sh              install, enable at boot, start now
#   sh tools/orin_setup/install_audio_service.sh --no-enable  install only; start by hand
#   sh tools/orin_setup/install_audio_service.sh --no-udev    skip the udev rule (it needs a sudo password)
#   sh tools/orin_setup/install_audio_service.sh --force      install on a machine that is not the Orin
#   sh tools/orin_setup/install_audio_service.sh --uninstall  stop, disable and remove everything it installed
#
# Installs:
#   ~/.config/systemd/user/party-pose-audio.service   from audio/party-pose-audio.service.in
#   ~/.config/party-pose/party-pose-audio.env         music folder, mode, volume (only if missing)
#   /etc/udev/rules.d/99-party-pose-audio.rules       lets group audio read the speaker's buttons
# Also turns on lingering for your user so the service starts at boot without a login.

set -eu

UNIT_NAME=party-pose-audio.service
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/$UNIT_NAME"
RULE=/etc/udev/rules.d/99-party-pose-audio.rules
REPO=$(cd "$(dirname "$0")/../.." && pwd)
AUDIO_DIR="$REPO/tools/orin_setup/audio"
USER_NAME=$(id -un)
ENV_FILE="$HOME/.config/party-pose/party-pose-audio.env"

ENABLE=1
UDEV_RULE=1
FORCE=0
UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --no-enable) ENABLE=0 ;;
        --no-udev) UDEV_RULE=0 ;;
        --force) FORCE=1 ;;
        --uninstall) UNINSTALL=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as the desktop user, not root (it uses sudo where needed)." >&2
    exit 1
fi

if [ "$UNINSTALL" -eq 1 ]; then
    systemctl --user disable --now party-pose-audio 2>/dev/null || true
    rm -f "$UNIT"
    if [ -f "$RULE" ]; then
        sudo rm -f "$RULE"
        sudo udevadm control --reload-rules
    fi
    systemctl --user daemon-reload
    echo "Removed $UNIT and $RULE. Left $ENV_FILE and lingering in place."
    exit 0
fi

if [ ! -x /usr/bin/jetson_clocks ] && [ "$FORCE" -eq 0 ]; then
    echo "/usr/bin/jetson_clocks not found; this does not look like the Orin. Nothing installed." >&2
    echo "Use --force to install here anyway." >&2
    exit 1
fi
for tool in mpv pactl; do
    command -v "$tool" >/dev/null 2>&1 || { echo "$tool not found; run: sudo apt install mpv pulseaudio-utils" >&2; exit 1; }
done

# 1. Env file: never overwrite the user's edits.
if [ ! -f "$ENV_FILE" ]; then
    mkdir -p "$(dirname "$ENV_FILE")"
    cp "$AUDIO_DIR/party-pose-audio.env.example" "$ENV_FILE"
    echo "Created $ENV_FILE"
else
    echo "Keeping existing $ENV_FILE"
fi

# 2. Unit file.
mkdir -p "$UNIT_DIR"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
sed -e "s|@REPO@|$REPO|g" -e "s|@ENV_FILE@|$ENV_FILE|g" \
    "$AUDIO_DIR/party-pose-audio.service.in" > "$tmp"
install -m 644 "$tmp" "$UNIT"
echo "Installed $UNIT"

# 3. udev rule: group audio may read the speaker's input device (the buttons).
if [ "$UDEV_RULE" -eq 1 ]; then
    sudo install -m 644 "$AUDIO_DIR/99-party-pose-audio-buttons.rules" "$RULE"
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=input
    echo "Installed $RULE"
else
    echo "Skipped the udev rule; the speaker buttons will not work until it is installed."
fi

# 4. Start at boot without a login (the audio socket belongs to the user's session).
loginctl enable-linger "$USER_NAME"

systemctl --user daemon-reload
if [ "$ENABLE" -eq 1 ]; then
    systemctl --user enable --now party-pose-audio
    echo "Enabled at boot and started."
fi

cat <<MSG

Start now:   systemctl --user start party-pose-audio
Stop:        systemctl --user stop party-pose-audio
Logs:        journalctl --user -u party-pose-audio -f
Settings:    $ENV_FILE  (then systemctl --user restart party-pose-audio)
MSG
