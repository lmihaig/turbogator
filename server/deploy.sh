#!/bin/bash
# update the remote infrastructure
# only runnable by Mihai with vpn
# spam him with messages if you need to change things

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

REMOTE="root@192.168.1.130"
echo "Deploying Infrastructure..."

rsync -avz --exclude 'deploy.sh' "$SCRIPT_DIR/" ${REMOTE}:/tmp/aos_setup/server/
rsync -avz "$ROOT_DIR/pyproject.toml" "$ROOT_DIR/uv.lock" ${REMOTE}:/tmp/aos_setup/

ssh ${REMOTE} <<'EOF'
    echo "Placing files in system directories..."
    mkdir -p /opt/aos/{api,scripts,workspaces,artifacts}
    
    # Move Python API files
    cp /tmp/aos_setup/server/main.py /opt/aos/api/
    cp /tmp/aos_setup/pyproject.toml /opt/aos/api/
    cp /tmp/aos_setup/uv.lock /opt/aos/api/
    
    # Move and restrict worker script
    cp /tmp/aos_setup/server/worker.py /opt/aos/scripts/
    chmod +x /opt/aos/scripts/worker.py
    cp /tmp/aos_setup/server/worker_reference.py /opt/aos/scripts/
    chmod +x /opt/aos/scripts/worker_reference.py
    
    # Move Systemd Service
    cp /tmp/aos_setup/server/aos-api.service /etc/systemd/system/
    
    echo "Syncing Python environment (uv)..."
    cd /opt/aos/api
    uv sync --group api
    
    echo "Restarting AOS Daemon..."
    systemctl daemon-reload
    systemctl restart aos-api.service
    
    # Cleanup staging
    rm -rf /tmp/aos_setup
    echo "AOS node infrastructure updated successfully."
EOF
