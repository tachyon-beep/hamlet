#!/bin/bash
set -e

# Installation script for hamlet-demo systemd service

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
USER="${USER:-$USER}"
WORKDIR="${WORKDIR:-$PROJECT_DIR}"
VENV_PYTHON="${VENV_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/townlet/sparse_adaptive.yaml}"
DB_PATH="${DB_PATH:-$PROJECT_DIR/demo_state.db}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints}"

echo "Installing hamlet-demo.service with:"
echo "  User: $USER"
echo "  WorkingDirectory: $WORKDIR"
echo "  Python: $VENV_PYTHON"
echo "  Config: $CONFIG"
echo "  Database: $DB_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"

# Create service file with substitutions
SERVICE_FILE="/tmp/hamlet-demo.service"
sed -e "s|%USER%|$USER|g" \
    -e "s|%WORKDIR%|$WORKDIR|g" \
    -e "s|%VENV_PYTHON%|$VENV_PYTHON|g" \
    -e "s|%CONFIG%|$CONFIG|g" \
    -e "s|%DB_PATH%|$DB_PATH|g" \
    -e "s|%CHECKPOINT_DIR%|$CHECKPOINT_DIR|g" \
    "$SCRIPT_DIR/hamlet-demo.service" > "$SERVICE_FILE"

# Install service
sudo cp "$SERVICE_FILE" /etc/systemd/system/hamlet-demo.service
sudo systemctl daemon-reload

echo ""
echo "Service installed! Commands:"
echo "  sudo systemctl start hamlet-demo    # Start training"
echo "  sudo systemctl stop hamlet-demo     # Stop training"
echo "  sudo systemctl status hamlet-demo   # Check status"
echo "  sudo journalctl -u hamlet-demo -f   # View logs"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo systemctl enable hamlet-demo"
