#!/bin/bash
# update the remote infrastructure
# only runnable by Mihai with vpn
# spam him with messages if you need to change things

REMOTE="root@192.168.1.130"
echo "Deploying Infrastructure..."

rsync -avz --exclude 'deploy.sh' ./server/ ${REMOTE}:/tmp/aos_setup/

ssh ${REMOTE} <<'EOF'
    echo "Placing files in system directories..."
    mkdir -p /opt/aos/{api,scripts,workspaces,artifacts}
    
    # Move Python API files
    cp /tmp/aos_setup/main.py /opt/aos/api/
    cp /tmp/aos_setup/pyproject.toml /opt/aos/api/
    
    # Move and restrict worker script
    cp /tmp/aos_setup/worker.py /opt/aos/scripts/
    chmod +x /opt/aos/scripts/worker.py
    cp /tmp/aos_setup/worker_baseline.py /opt/aos/scripts/
    chmod +x /opt/aos/scripts/worker_baseline.py
    
    # Move Systemd Service
    cp /tmp/aos_setup/aos-api.service /etc/systemd/system/
    
    echo "Syncing Python environment (uv)..."
    cd /opt/aos/api
    uv sync
    
    echo "Restarting AOS Daemon..."
    systemctl daemon-reload
    systemctl restart aos-api.service
    
    # Cleanup staging
    rm -rf /tmp/aos_setup
    echo "AOS node infrastructure updated successfully."
EOF
