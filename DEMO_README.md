# Multi-Day Demo Guide

## Architecture Overview

The multi-day demo has **two independent components**:

1. **Demo Runner** (`hamlet.demo.runner`) - Autonomous training process
2. **Viz Server** (`hamlet.demo.viz_server`) - WebSocket server streaming from SQLite

**Key difference from old training server:** The demo runner starts training immediately and runs independently. The viz server just monitors the database - it doesn't control training.

## Running the Demo

### Step 1: Start the Demo Runner

```bash
python -m hamlet.demo.runner configs/townlet/sparse_adaptive.yaml demo_state.db checkpoints
```

This will:
- Train for up to 10,000 episodes
- Save checkpoints every 100 episodes
- Log all metrics to `demo_state.db`
- Randomize affordances at episode 5000 (generalization test)
- Auto-resume from last checkpoint if restarted

### Step 2: Start the Viz Server

```bash
python -m hamlet.demo.viz_server demo_state.db frontend/dist 8765
```

This streams episode data from SQLite to browsers via WebSocket on port 8765.

### Step 3: View in Browser

You have **two options**:

#### Option A: Standalone HTML Page (Recommended)
```
http://localhost:8765/
```

**Pros:**
- No build required
- Simple, focused demo metrics
- Real-time charts
- Works immediately

**Cons:**
- Basic visualization only
- No grid/agent display

#### Option B: Full Vue Frontend
```bash
cd frontend && npm run dev
# Then open http://localhost:5173
```

**Pros:**
- All Phase 3 visualization components
- Novelty heatmap, curriculum tracker, survival trends
- AffordanceGraph component
- Full UI controls

**Cons:**
- Requires npm build
- "Start Training" button doesn't work (training runs independently)
- Grid won't show agents (demo doesn't stream live positions)

## Important Notes

### "Start Training" Button Doesn't Work

**Why:** The demo runner starts training automatically when launched. The viz server only monitors the database - it doesn't control training.

**What happens:** If you click "Start Training" in the Vue UI, the command is logged but ignored.

**Solution:** Don't use the button. Just start the demo runner in a separate terminal.

### UI Rendering

The viz server sends these messages to make the frontend work:

1. **On connection:**
   - `connected` - Tells frontend server is ready
   - `training_status` - Shows current episode and total
   - `training_started` - Confirms training is running (if demo is active)

2. **Every second:**
   - `episode_start` - When a new episode begins
   - `episode_complete` - With survival time, reward, epsilon, etc.

3. **Grid visualization:** Empty (demo doesn't stream live agent positions)

### Monitoring Progress

**In terminal (demo runner):**
```
Episode 10/10000 | Survival: 87 steps | Reward: 12.34 | Intrinsic Weight: 0.956 | Stage: 2/5
```

**In browser:**
- Current episode number
- Survival time (steps)
- Total reward
- Curriculum stage (1-5)
- Intrinsic weight (1.0 → 0.0)
- Exploration epsilon

**In database:**
```bash
sqlite3 demo_state.db "SELECT episode_id, survival_time, intrinsic_weight FROM episodes ORDER BY episode_id DESC LIMIT 10"
```

## Systemd Deployment (Production)

For unattended multi-day runs:

```bash
# Install service
./deploy/install-service.sh

# Start training
sudo systemctl start hamlet-demo

# View logs
sudo journalctl -u hamlet-demo -f

# Stop
sudo systemctl stop hamlet-demo
```

The service auto-restarts on failure and resumes from last checkpoint.

## Troubleshooting

### "Training" button is disabled
- Check if viz server is running: `curl http://localhost:8765`
- Viz server needs `/ws/training` endpoint (now included)

### No data in UI
- Check if demo runner is writing to database: `ls -lh demo_state.db*`
- Check runner logs for errors
- Restart viz server after starting runner

### "Start Training" does nothing
- **Expected behavior** - demo runs independently
- Just ignore the button or use standalone HTML page instead

### Grid is empty
- **Expected behavior** - demo doesn't stream live positions
- Use StatsPanel and charts instead of grid

## Phase 3.5 Success Criteria

- [ ] Training runs for 48+ hours without crashes
- [ ] Intrinsic weight anneals from 1.0 → <0.1
- [ ] Agent progresses through all 5 curriculum stages
- [ ] Final survival time > 200 steps (vs ~115 baseline)
- [ ] Visualization updates in real-time
- [ ] No memory leaks or performance degradation
- [ ] Generalization test executes at episode 5000

## Next Phase

After Phase 3.5 validation, move to **Phase 4: POMDP Extension** (partial observability + LSTM).
