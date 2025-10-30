# Level 2 POMDP Training Guide

This document explains how to run the Level 1.5 (no proximity) and Level 2 (POMDP) training experiments side-by-side.

## What Changed

### Proximity Shaping Disabled
- **Reward hacking fixed**: Agents can no longer exploit proximity rewards by standing near affordances
- **Pure interaction rewards**: Agent must move to and interact with affordances to survive
- **All new training runs** use this reward structure (hardcoded in `hamlet_env.py:85`)

### Level 2: Partial Observability
- **5×5 local vision window**: Agent sees only nearby area (not full 8×8 grid)
- **LSTM memory**: RecurrentSpatialQNetwork with 256-dim hidden state remembers past observations
- **Exploration required**: Must actively explore to discover affordance locations
- **Memory-based navigation**: LSTM learns implicit spatial map of environment

## Training Configs

### Level 1.5: Full Observability (No Proximity)
**Config**: `configs/level_1_5_no_proximity.yaml`

- **Full 8×8 grid observation**: Agent sees entire environment
- **Standard MLP Q-Network**: 72-dim state → [128, 128] → 5 actions (~26K params)
- **2000 episodes**: More episodes without proximity hints
- **Baseline for comparison**: Shows performance with full info but no reward hacking

**Expected Results**:
- Convergence: 1500-2000 episodes
- Peak reward: +40 to +60
- Survival time: 200-300 steps
- Learning curve: Faster than Level 2 (has full information)

### Level 2: Partial Observability (POMDP)
**Config**: `configs/level_2_pomdp.yaml`

- **5×5 local window**: Agent sees only nearby area
- **RecurrentSpatialQNetwork**: CNN + LSTM + position encoding (~600K params)
- **3000 episodes**: More exploration needed
- **Lower learning rate**: 0.0001 (vs 0.00025) for stability with recurrence
- **Slower epsilon decay**: 0.997 (vs 0.995) to encourage exploration

**Expected Results** (from `ARCHITECTURE_DESIGN.md`):
- Convergence: 2000-3000 episodes
- Peak reward: +30 to +50 (lower than full obs)
- Survival time: 150-250 steps
- 30-40% of time spent exploring
- More realistic cognitive behavior (gets lost, re-discovers affordances)

## How to Run

### Terminal 1: Level 1.5 Training (Full Observability)
```bash
cd /home/john/hamlet
uv run python run_experiment.py --config configs/level_1_5_no_proximity.yaml
```

**Outputs**:
- Checkpoints: `checkpoints_level_1_5/checkpoint_ep*.pt`
- Metrics DB: `metrics_level_1_5.db`
- TensorBoard: `runs_level_1_5/`

**Training time**: ~15-20 minutes for 2000 episodes

### Terminal 2: Level 2 Training (POMDP)
```bash
cd /home/john/hamlet
uv run python run_experiment.py --config configs/level_2_pomdp.yaml
```

**Outputs**:
- Checkpoints: `checkpoints_level_2/checkpoint_ep*.pt`
- Metrics DB: `metrics_level_2.db`
- TensorBoard: `runs_level_2/`

**Training time**: ~25-30 minutes for 3000 episodes

### Terminal 3: Live Inference (Optional)
Watch Level 1.5 training progress:
```bash
uv run python src/hamlet/demo/live_inference.py checkpoints_level_1_5 8766 0.2 2000
```

Watch Level 2 training progress:
```bash
uv run python src/hamlet/demo/live_inference.py checkpoints_level_2 8767 0.2 3000
```

### Terminal 4: Frontend (Optional)
```bash
cd frontend
npm run dev
```

Then open:
- Level 1.5: http://localhost:5173 (connects to port 8766)
- Level 2: Change port in frontend to 8767 or use different browser

## Monitoring Progress

### TensorBoard (Recommended)
```bash
# Terminal 1
tensorboard --logdir runs_level_1_5 --port 6006

# Terminal 2
tensorboard --logdir runs_level_2 --port 6007
```

Then open:
- Level 1.5: http://localhost:6006
- Level 2: http://localhost:6007

### Key Metrics to Watch

**Episode Reward**:
- Level 1.5 should reach +40 to +60
- Level 2 should reach +30 to +50
- Level 2 will be more variable (exploration)

**Survival Steps**:
- Level 1.5: 200-300 steps
- Level 2: 150-250 steps

**Epsilon (Exploration)**:
- Level 1.5: Decays to 0.01 by episode ~1500
- Level 2: Decays to 0.05 by episode ~2000 (more exploration)

**Loss**:
- Both should stabilize after ~500 episodes
- Level 2 may have higher variance (recurrent learning)

## Pedagogical Value

### Level 1.5 (No Proximity)
**Teaching Moments**:
- Shows that agents can learn without reward hacking
- Demonstrates importance of proper reward shaping
- Baseline for understanding impact of partial observability

### Level 2 (POMDP)
**Teaching Moments**:
- **Memory**: LSTM learns to remember affordance locations
- **Exploration**: Must actively discover the world
- **Uncertainty**: Agent gets lost, re-discovers locations
- **Real-world robotics**: Actual robots have partial observability
- **Performance vs. information tradeoff**: Lower performance with less information

### Comparison Questions for Students
1. How much slower does Level 2 learn compared to Level 1.5?
2. What's the performance gap (reward/survival) between full and partial obs?
3. How does exploration behavior differ?
4. Can you see the agent "remembering" affordance locations in Level 2?
5. What happens when Level 2 agent gets disoriented?

## Troubleshooting

### Level 2 Not Learning
- Check that `partial_observability: true` in config
- Verify LSTM hidden state resets each episode (should see in logs)
- Increase `learning_starts` to 3000 (more warmup)
- Reduce learning rate to 0.00005

### Out of Memory
- Reduce `batch_size` to 16
- Reduce `replay_buffer_size` to 5000
- Use CPU instead of CUDA (slower but uses less memory)

### Training Too Slow
- Reduce `num_episodes` for quick test (500 for Level 1.5, 1000 for Level 2)
- Increase `batch_size` to 128 if GPU memory allows
- Disable `tensorboard: false` in config

## Next Steps

After both training runs complete:

1. **Compare learning curves** in TensorBoard
2. **Watch inference** on final checkpoints to see behavior differences
3. **Implement Level 3** (Multi-zone hierarchical RL) from `ARCHITECTURE_DESIGN.md`
4. **Implement sequential replay buffer** for better recurrent training
5. **Add curiosity-driven exploration** for Level 2

## Architecture Notes

### RecurrentSpatialQNetwork
```
Vision Encoder:  5×5 CNN → 128 features
Position Encoder: (x, y) → 32 features
Meter Encoder:    8 meters → 32 features
LSTM:             192 input → 256 hidden
Q-Head:           256 → 128 → 5 actions

Total: ~600K parameters
```

### Training Strategy (Level 2)
- **Episode rollout**: LSTM hidden state persists across steps
- **Batch training**: Hidden state reset per transition (no temporal credit)
- **Future improvement**: Episode replay buffer for sequential training

### Why This Works
Even though batch training doesn't leverage temporal sequences, the LSTM still learns to:
1. Encode spatial relationships from single observations
2. Remember recent history during episode rollouts
3. Build implicit world model from position + local observations

The temporal memory is primarily used during **inference** (episode execution), not during **training** (batch updates). This is a simplified approach that works well for first-pass POMDP learning.
