#!/bin/sh
#
# Install party-pose.service on the Orin (issue #8). Run as the desktop user from the repo;
# it asks for your sudo password once.
#
#   sh tools/orin_setup/install_service.sh              install, enable at boot
#   sh tools/orin_setup/install_service.sh --no-enable  install, start by hand only
#   sh tools/orin_setup/install_service.sh --no-sudoers skip the password-free systemctl rule
#   sh tools/orin_setup/install_service.sh --uninstall  stop, disable and remove everything it installed
#
# Installs:
#   /etc/systemd/system/party-pose.service   from service/party-pose.service.in
#   /etc/sudoers.d/party-pose                lets you run `sudo systemctl start|stop|restart party-pose`
#                                            without a password (e.g. over ssh)
#   ~/.config/party-pose/party-pose.env      app arguments and display settings (only if missing)

set -eu

UNIT=/etc/systemd/system/party-pose.service
SUDOERS=/etc/sudoers.d/party-pose
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SERVICE_DIR="$REPO/tools/orin_setup/service"
USER_NAME=$(id -un)
ENV_FILE="$HOME/.config/party-pose/party-pose.env"

ENABLE=1
SUDOERS_RULE=1
for arg in "$@"; do
    case "$arg" in
        --no-enable) ENABLE=0 ;;
        --no-sudoers) SUDOERS_RULE=0 ;;
        --uninstall)
            sudo systemctl disable --now party-pose 2>/dev/null || true
            sudo rm -f "$UNIT" "$SUDOERS"
            sudo systemctl daemon-reload
            echo "Removed $UNIT and $SUDOERS. Left $ENV_FILE in place."
            exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as the desktop user, not root (it uses sudo where needed)." >&2
    exit 1
fi
[ -x "$REPO/env/bin/python" ] || { echo "No $REPO/env; run tools/orin_setup/bootstrap_env.sh first." >&2; exit 1; }
[ -x /usr/bin/jetson_clocks ] || echo "WARNING: /usr/bin/jetson_clocks not found; is this the Orin?" >&2

# 1. Env file: never overwrite the user's edits.
if [ ! -f "$ENV_FILE" ]; then
    mkdir -p "$(dirname "$ENV_FILE")"
    cp "$SERVICE_DIR/party-pose.env.example" "$ENV_FILE"
    echo "Created $ENV_FILE"
else
    echo "Keeping existing $ENV_FILE"
fi

# 2. Unit file.
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
sed -e "s|@USER@|$USER_NAME|g" -e "s|@REPO@|$REPO|g" -e "s|@ENV_FILE@|$ENV_FILE|g" \
    "$SERVICE_DIR/party-pose.service.in" > "$tmp"
sudo install -m 644 "$tmp" "$UNIT"
echo "Installed $UNIT"

# 3. Password-free start/stop/restart/status for this one unit only.
if [ "$SUDOERS_RULE" -eq 1 ]; then
    : > "$tmp"
    for sc in /usr/bin/systemctl /bin/systemctl; do
        for verb in start stop restart status; do
            echo "$USER_NAME ALL=(root) NOPASSWD: $sc $verb party-pose, $sc $verb party-pose.service" >> "$tmp"
        done
    done
    if sudo visudo -cf "$tmp" >/dev/null; then
        sudo install -m 440 "$tmp" "$SUDOERS"
        echo "Installed $SUDOERS"
    else
        echo "WARNING: sudoers rule failed visudo check; not installed" >&2
    fi
fi

sudo systemctl daemon-reload
if [ "$ENABLE" -eq 1 ]; then
    sudo systemctl enable party-pose
    echo "Enabled at boot."
fi

# The service needs a logged-in desktop; at boot that means GDM auto-login.
if ! grep -Eqs "^[[:space:]]*AutomaticLoginEnable[[:space:]]*=[[:space:]]*[Tt]rue" /etc/gdm3/custom.conf; then
    echo "NOTE: GDM auto-login looks off. At boot the service waits for someone to log in."
    echo "      Turn it on in Settings > Users > Automatic Login, or in /etc/gdm3/custom.conf."
fi

cat <<MSG

Start now:   sudo systemctl start party-pose
Stop:        sudo systemctl stop party-pose      (or Esc on the Orin's keyboard)
Logs:        journalctl -u party-pose -f
Settings:    $ENV_FILE  (then sudo systemctl restart party-pose)
MSG
