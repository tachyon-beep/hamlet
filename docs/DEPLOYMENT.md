# Multi-Day Demo Deployment Guide

## Prerequisites

- Ubuntu 24.04 (or compatible Linux with systemd)
- Python 3.11+
- uv package manager
- GPU (optional but recommended)

## Installation

### 1. Install Dependencies

```bash
cd /path/to/hamlet
uv sync
```

### 2. Install systemd Service

```bash
chmod +x deploy/install-service.sh
./deploy/install-service.sh
```

This installs the `hamlet-demo` service with auto-restart on failure.

### 3. Start Training

```bash
sudo systemctl start hamlet-demo
```

### 4. Monitor Progress

```bash
# View live logs
sudo journalctl -u hamlet-demo -f

# Check status
sudo systemctl status hamlet-demo

# Query database
sqlite3 demo_state.db "SELECT COUNT(*) FROM episodes"
```

## Starting Visualization

In a separate terminal:

```bash
# Terminal 1: Visualization server (TODO: implement in Task 4)
python -m hamlet.demo.viz_server

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open browser to `http://localhost:5173`

## Stopping the Demo

```bash
# Graceful shutdown (saves checkpoint)
sudo systemctl stop hamlet-demo

# Check final status
sqlite3 demo_state.db "SELECT episode_id, survival_time FROM episodes ORDER BY episode_id DESC LIMIT 10"
```

## Troubleshooting

**Training not starting:**
```bash
sudo journalctl -u hamlet-demo --since "5 minutes ago"
```

**Database locked:**
- Ensure only one training process is running
- Check for stale lock files: `ls -la demo_state.db*`

**Out of disk space:**
```bash
du -sh checkpoints/  # Check checkpoint size
df -h                # Check disk usage
```
