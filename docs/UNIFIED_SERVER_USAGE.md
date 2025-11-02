# Unified Server Usage Guide

## Overview

The unified server runs **training + inference** in a single command, while the frontend runs separately for better stability and Vue.js Hot Module Replacement (HMR) support.

## Quick Start

### Terminal 1: Training + Inference Server

```bash
# Activate environment
source .venv/bin/activate

# Start training + inference (will run until episodes complete)
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 10000

# For L3 debugging:
python run_demo.py --config configs/level_3_temporal.yaml --episodes 10000
```

### Terminal 2: Frontend (once checkpoints exist)

```bash
# Start Vue dev server
cd frontend && npm run dev

# Open browser to http://localhost:5173
```

## Architecture

```
run_demo.py (Main Process)
  ├─> Training Thread
  │   └─> DemoRunner (saves checkpoints every 100 episodes)
  │
  └─> Inference Thread
      └─> LiveInferenceServer (WebSocket on port 8766)
          ├─> Watches for new checkpoints
          ├─> Runs inference at 5 steps/sec
          └─> Broadcasts state to frontend
```

## Command Line Options

```bash
python run_demo.py --help
```

**Required:**

- `--config PATH` - Training configuration YAML file

**Optional:**

- `--episodes N` - Number of training episodes (default: 10000)
- `--checkpoint-dir PATH` - Custom checkpoint directory (auto-generated if not provided)
- `--inference-port PORT` - WebSocket port (default: 8766)
- `--debug` - Enable debug logging with full tracebacks

## Examples

### Resume from Checkpoint

```bash
python run_demo.py \
  --config configs/level_1_full_observability.yaml \
  --checkpoint-dir runs/L1_full_observability/2025-11-02_123456/checkpoints \
  --episodes 20000
```

### Custom Inference Port

```bash
python run_demo.py \
  --config configs/level_1_full_observability.yaml \
  --episodes 5000 \
  --inference-port 8800

# Frontend will auto-connect to http://localhost:8800
```

### Short Training Run for Testing

```bash
python run_demo.py \
  --config configs/level_1_full_observability.yaml \
  --episodes 100
```

## Directory Structure

Training creates organized runs directory:

```
runs/
└── L1_full_observability/          # Level name from config
    └── 2025-11-02_123456/          # Timestamp
        ├── config.yaml              # Copy of training config
        ├── checkpoints/             # Model checkpoints
        │   ├── checkpoint_ep00000.pt
        │   ├── checkpoint_ep00100.pt
        │   └── ...
        ├── tensorboard/             # TensorBoard logs
        │   └── events.out.tfevents...
        └── metrics.db               # SQLite episode metrics
```

## TensorBoard

View training metrics:

```bash
tensorboard --logdir runs/L1_full_observability/2025-11-02_123456/tensorboard
# Open http://localhost:6006
```

## Graceful Shutdown

Press `Ctrl+C` once to trigger graceful shutdown:

1. Inference server stops immediately
2. Training finishes current episode
3. Final checkpoint saved
4. All threads cleaned up

## Benefits of Split Architecture

**Why Training + Inference Unified:**

- Single command to start both
- Training automatically updates checkpoints
- Inference watches for new checkpoints
- No port conflicts or coordination issues

**Why Frontend Separate:**

- Vue HMR works without restart
- Frontend can be restarted independently
- Easier frontend development workflow
- No subprocess management complexity

## Troubleshooting

**"Port 8766 already in use"**

```bash
# Find and kill existing process
lsof -ti:8766 | xargs kill -9

# Or use custom port
python run_demo.py --config ... --inference-port 8800
```

**"No checkpoints found" in frontend**

```bash
# Wait for first checkpoint (saved at episode 0)
# Or point frontend to existing checkpoint directory
```

**Inference server not connecting**

```bash
# Check inference server is running
curl http://localhost:8766/health  # Should return 200 OK

# Check frontend config (frontend/.env or frontend/vite.config.ts)
# VITE_WS_URL should match inference port
```

## For L3 Debugging

Level 3 uses temporal mechanics - you'll want to watch:

- Time-of-day cycles (24-tick days)
- Multi-tick interactions (jobs take 5 ticks to complete)
- Operating hours (Bar open 6pm-4am, Job 8am-6pm, etc.)
- Interaction progress bars in visualization

```bash
# Start L3 training + inference
python run_demo.py --config configs/level_3_temporal.yaml --episodes 10000

# In another terminal: frontend
cd frontend && npm run dev
```

## Implementation Details

**Phase 2 Complete (Current):**

- ✅ Training runs in background thread
- ✅ Inference runs in separate thread with proper async shutdown
- ✅ Auto-generated directory structure
- ✅ Graceful shutdown on Ctrl+C
- ✅ Training completion detection

**Phase 3 Deferred:**

- ⏳ Frontend subprocess integration (manual start preferred for HMR)

## See Also

- `DEMO_README.md` - Legacy 3-terminal workflow documentation
- `docs/UNIFIED_SERVER_PLAN.md` - Architecture and design decisions
- `TRAINING_UI_GUIDE.md` - Frontend visualization guide
