# Multi-Day Demo Guide

## Architecture Overview

The multi-day demo has **three components** working together:

1. **Demo Runner** (`hamlet.demo.runner`) - Fast background training (burn mode)
2. **Live Inference Server** (`hamlet.demo.live_inference`) - Human-speed inference with latest checkpoint
3. **Vue Frontend** - Real-time visualization of agent learning

**The Secret:** Training runs at full speed in the background. The inference server loads the latest checkpoint every ~100 episodes and runs inference at human-watchable speed (5 steps/sec). Viewers see the agent progressively improve without waiting for slow training!

## Running the Demo

### Step 1: Start the Background Training (Burn Mode)

```bash
python -m hamlet.demo.runner configs/townlet/sparse_adaptive.yaml demo_state.db checkpoints
```

This will:
- Train at **full speed** (~190 episodes/hour)
- Save checkpoints every 100 episodes
- Log all metrics to `demo_state.db`
- Randomize affordances at episode 5000 (generalization test)
- Auto-resume from last checkpoint if restarted

**Leave this running in the background!**

### Step 2: Start the Live Inference Server

```bash
python -m hamlet.demo.live_inference checkpoints 8766 0.2
```

Arguments:
- `checkpoints` - Directory to watch for new checkpoints
- `8766` - WebSocket port (Vue frontend will connect here)
- `0.2` - Step delay in seconds (0.2 = 5 steps/sec, human-watchable)

This will:
- Load the latest checkpoint
- Run inference episodes at **human speed**
- Stream step-by-step updates via WebSocket
- Hot-swap to newer checkpoints automatically
- Show the agent getting progressively smarter

### Step 3: Open the Vue Frontend

```bash
cd frontend && npm run dev
```

Then open: `http://localhost:5173`

**What you'll see:**
- Agent moving on grid in real-time
- Meters updating each step
- Survival time increasing as training progresses
- Model checkpoint indicator (e.g., "checkpoint_ep01500")
- Full Phase 3 visualizations

**Controls:**
- **Play/Pause**: Start/stop inference episodes
- **Step**: Run single step
- **Reset**: Reset episode counter
- ~~Training/Inference toggle~~: Removed - always runs inference on latest checkpoint

## How It Works (The "Cheat")

**Training Process:**
```
Episode 0 → 100 → 200 → 300 → ... → 10,000
  ↓        ↓      ↓      ↓              ↓
Save    Save   Save   Save    ...    Save
checkpoint checkpoint checkpoint
(Fast, no visualization)
```

**Inference Process:**
```
Every 30 seconds:
1. Check checkpoints/ directory
2. If new checkpoint found, load Q-network weights
3. Run inference episode at 0.2s/step (5 steps/sec)
4. Stream grid updates via WebSocket
5. Viewers see agent behavior at this checkpoint
6. Repeat
```

**Result:** Viewers see the agent getting progressively better over time, but training runs at full speed in the background. This is standard practice for ML demos!

## What You'll See

**Early training (Episodes 0-500):**
- Agent wanders randomly
- Dies quickly (~80-100 steps)
- Poor affordance utilization

**Mid training (Episodes 500-2000):**
- Agent starts finding resources
- Basic routines emerge
- Survival time ~150 steps

**Late training (Episodes 2000-5000):**
- Clear job→bed→shower→fridge patterns
- Survival time ~250+ steps
- Intrinsic weight → 0 (pure exploitation)

**Post-randomization (Episodes 5000+):**
- Affordance positions randomized
- Agent adapts to new layout
- Tests generalization capability

## Monitoring Progress

**Terminal 1 (Training):**
```
Episode 1523/10000 | Survival: 234 steps | Reward: 45.67 | Intrinsic Weight: 0.123 | Stage: 4/5
```

**Terminal 2 (Inference):**
```
Loading checkpoint: checkpoint_ep01500 (episode 1500)
Episode 42 complete: 187 steps, reward: 38.45
```

**Browser (Vue Frontend):**
- Grid shows agent moving in real-time
- Meters update each step (energy, hygiene, satiation, money)
- Checkpoint indicator: "Running checkpoint_ep01500"
- Survival trends chart shows improvement
- Intrinsic weight chart shows annealing

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

### No checkpoints found
```
No checkpoints found in checkpoints/
```
**Solution:** Start the demo runner first. It will create the first checkpoint after episode 0.

### Agent isn't moving in UI
- Check that live_inference server is running
- Click "Play" button in Vue frontend
- Check browser console for WebSocket connection errors

### Agent behavior looks random/bad
- Check which checkpoint is loaded (shown in UI)
- Early checkpoints (ep < 500) will be untrained
- Wait for training to progress or manually load a later checkpoint

### Model not updating
- Checkpoints saved every 100 episodes
- Hot-swap happens automatically when new checkpoint appears
- Check training terminal for checkpoint save messages

### Training too slow
- Training should run at ~0.5s per episode (~190 episodes/hour)
- If slower, check CPU/GPU usage
- Consider reducing `max_steps_per_episode` in config

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
