#!/bin/bash
# update the remote infrastructure
# only runnable by Mihai with vpn
# spam him with messages if you need to change things
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR_LOCAL=$(cd "$SCRIPT_DIR/.." && pwd)

read_cfg() {
  python3 - <<EOF
import sys
sys.path.insert(0, "$ROOT_DIR_LOCAL")
import config as c
s = c.SERVERS[c.ACTIVE_SERVER]
print(c.ACTIVE_SERVER)
print(s["ssh"])
print(s["root_dir"])
print("1" if s["user_systemd"] else "0")
print(s["listen_host"])
EOF
}

mapfile -t CFG < <(read_cfg)
ACTIVE="${CFG[0]}"
REMOTE="${CFG[1]}"
ROOT_DIR="${CFG[2]}"
USER_SYSTEMD="${CFG[3]}"
LISTEN_HOST="${CFG[4]}"

echo "==> Deploying to '${ACTIVE}' (${REMOTE}), root_dir=${ROOT_DIR}, user_systemd=${USER_SYSTEMD}"

UNIT_TMPL="$SCRIPT_DIR/aos-api.service"
UNIT_OUT="$(mktemp)"
trap 'rm -f "$UNIT_OUT"' EXIT

if [[ "$USER_SYSTEMD" == "1" ]]; then
  REMOTE_USER="${REMOTE%@*}"
  HOME_DIR="/home/${REMOTE_USER}"
  USER_LINE=""
  WANTED_BY="default.target"
else
  HOME_DIR="/root"
  USER_LINE="User=root"
  WANTED_BY="multi-user.target"
fi

sed \
  -e "s|{{ROOT_DIR}}|${ROOT_DIR}|g" \
  -e "s|{{HOME}}|${HOME_DIR}|g" \
  -e "s|{{LISTEN_HOST}}|${LISTEN_HOST}|g" \
  -e "s|{{WANTED_BY}}|${WANTED_BY}|g" \
  -e "s|# {{USER_LINE}}.*|${USER_LINE}|" \
  "$UNIT_TMPL" > "$UNIT_OUT"

ssh "$REMOTE" "mkdir -p /tmp/aos_setup/server"
rsync -avz --exclude 'deploy.sh' --exclude 'aos-api.service' "$SCRIPT_DIR/" "${REMOTE}:/tmp/aos_setup/server/"
rsync -avz "$ROOT_DIR_LOCAL/pyproject.toml" "$ROOT_DIR_LOCAL/uv.lock" "$ROOT_DIR_LOCAL/config.py" "${REMOTE}:/tmp/aos_setup/"
scp "$UNIT_OUT" "${REMOTE}:/tmp/aos_setup/aos-api.service"

if [[ "$USER_SYSTEMD" == "1" ]]; then
  ssh "$REMOTE" "ROOT_DIR='${ROOT_DIR}' bash -s" <<'REMOTE_EOF'
    set -euo pipefail
    # Non-interactive SSH skips .bashrc / .profile, so PATH lacks ~/.local/bin where uv lives.
    export PATH="$HOME/.local/bin:$PATH"
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

    if ! loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes'; then
      echo "ERR: linger is not enabled for $USER."
      echo "Run once on the box:  sudo loginctl enable-linger $USER"
      exit 1
    fi

    if ! command -v uv >/dev/null 2>&1; then
      echo "ERR: uv not found in PATH on remote. Install with:"
      echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
      exit 1
    fi

    mkdir -p "$ROOT_DIR"/{api,scripts,workspaces,artifacts}

    cp /tmp/aos_setup/server/main.py        "$ROOT_DIR/api/"
    cp /tmp/aos_setup/pyproject.toml        "$ROOT_DIR/api/"
    cp /tmp/aos_setup/uv.lock               "$ROOT_DIR/api/"
    cp /tmp/aos_setup/config.py             "$ROOT_DIR/api/"

    cp /tmp/aos_setup/server/worker.py      "$ROOT_DIR/scripts/"
    chmod +x "$ROOT_DIR/scripts/worker.py"

    mkdir -p "$HOME/.config/systemd/user"
    cp /tmp/aos_setup/aos-api.service       "$HOME/.config/systemd/user/aos-api.service"

    cd "$ROOT_DIR/api"
    uv sync --group server --no-install-project

    systemctl --user daemon-reload
    systemctl --user enable --now aos-api.service
    systemctl --user restart aos-api.service
    systemctl --user --no-pager status aos-api.service | head -20 || true

    rm -rf /tmp/aos_setup
    echo "Done."
REMOTE_EOF
else
  # legacy mihai n100 server
  ssh "$REMOTE" "ROOT_DIR='${ROOT_DIR}' bash -s" <<'REMOTE_EOF'
    set -euo pipefail
    echo "Placing files under $ROOT_DIR..."
    mkdir -p "$ROOT_DIR"/{api,scripts,workspaces,artifacts}

    cp /tmp/aos_setup/server/main.py        "$ROOT_DIR/api/"
    cp /tmp/aos_setup/pyproject.toml        "$ROOT_DIR/api/"
    cp /tmp/aos_setup/uv.lock               "$ROOT_DIR/api/"
    cp /tmp/aos_setup/config.py             "$ROOT_DIR/api/"

    cp /tmp/aos_setup/server/worker.py      "$ROOT_DIR/scripts/"
    chmod +x "$ROOT_DIR/scripts/worker.py"
    rm -f "$ROOT_DIR/scripts/worker_reference.py"

    cp /tmp/aos_setup/aos-api.service       /etc/systemd/system/aos-api.service

    echo "Syncing Python environment (uv)..."
    cd "$ROOT_DIR/api"
    uv sync --group server --no-install-project

    echo "Restarting aos-api (system unit)..."
    systemctl daemon-reload
    systemctl restart aos-api.service

    rm -rf /tmp/aos_setup
    echo "Done."
REMOTE_EOF
fi
