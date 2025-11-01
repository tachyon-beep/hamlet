# Live Inference Updates: Curriculum & Baseline Display

**Date:** November 2, 2025  
**Status:** ✅ IMPLEMENTED

---

## Changes to Live Inference Server

### Problem

Live inference wasn't passing curriculum difficulty to the environment, so:

- Rewards shown were incorrect (always calculated with baseline R=100)
- Viewers couldn't see learning progress properly
- Stage information wasn't displayed

### Solution

Updated `live_inference.py` to:

1. Get curriculum decisions at episode start
2. Pass `depletion_multiplier` to `env.step()`
3. Broadcast baseline and stage info to frontend

---

## New WebSocket Messages

### 1. Episode Start (Enhanced)

**Message Type:** `episode_start`

```json
{
  "type": "episode_start",
  "episode": 42,
  "checkpoint": "checkpoint_ep00500",
  "checkpoint_episode": 500,
  "total_episodes": 1000,
  "epsilon": 0.525,
  "curriculum_stage": 1,              // NEW: Current stage (1-5)
  "curriculum_multiplier": 0.2,        // NEW: Difficulty multiplier
  "baseline_survival": 166.7           // NEW: R (expected random walk survival)
}
```

**Frontend Can Display:**

- "Stage 1/5 (20% difficulty)"
- "Baseline: 167 steps"
- "Beat baseline by X steps to get positive reward!"

### 2. State Update (Enhanced)

**Message Type:** `state_update`

```json
{
  "type": "state_update",
  "step": 150,
  "cumulative_reward": -17.5,
  "grid": { ... },
  "agent_meters": { ... },
  "q_values": [0.5, 0.3, 0.2, 0.8, 0.1],
  "affordance_stats": [ ... ],
  "baseline_survival": 166.7           // NEW: For progress bar
}
```

**Frontend Can Display:**

- Progress bar: "150 / 167 steps (90% of baseline)"
- Real-time comparison: "On track for +33 reward!" or "Below baseline (-17)"

### 3. Episode End (Enhanced)

**Message Type:** `episode_end`

```json
{
  "type": "episode_end",
  "episode": 42,
  "steps": 200,
  "total_reward": 33.3,
  "reason": "done",
  "checkpoint": "checkpoint_ep00500",
  "checkpoint_episode": 500,
  "total_episodes": 1000,
  "epsilon": 0.525,
  "baseline_survival": 166.7,          // NEW: For context
  "performance_vs_baseline": +33.3,    // NEW: steps - baseline
  "curriculum_stage": 1                // NEW: Current stage
}
```

**Frontend Can Display:**

- "Episode 42: 200 steps"
- "Reward: +33.3 (beat baseline by 33 steps!)" ✅
- "Stage 1/5" with progress indicator

---

## Frontend Display Suggestions

### During Episode (state_update)

```
┌─────────────────────────────────────────────────────┐
│ Episode 42 • Stage 1/5 (20% difficulty)             │
├─────────────────────────────────────────────────────┤
│ Step: 150                                            │
│ Progress: [███████████████░░░░] 150/167 (90%)       │
│ Baseline: 167 steps                                  │
│ Projected: +33 reward (on track!)                    │
│ Cumulative Reward: -17.5                             │
└─────────────────────────────────────────────────────┘
```

### After Episode (episode_end)

```
┌─────────────────────────────────────────────────────┐
│ ✅ Episode 42 Complete!                              │
├─────────────────────────────────────────────────────┤
│ Survival: 200 steps                                  │
│ Baseline: 167 steps                                  │
│ Performance: +33 steps above baseline                │
│ Reward: +33.3 🎉                                     │
│ Stage: 1/5                                           │
│ Checkpoint: checkpoint_ep00500                       │
└─────────────────────────────────────────────────────┘
```

### Learning Progress Indicator

Show color-coded performance:

- 🔴 Red: < 50% of baseline (struggling)
- 🟡 Yellow: 50-100% of baseline (learning)
- 🟢 Green: > 100% of baseline (mastered!)

Example:

```
Episode Performance Trend:
Ep 1:  [🔴 30 steps]  -137 reward
Ep 10: [🔴 80 steps]  -87 reward
Ep 25: [🟡 130 steps] -37 reward
Ep 50: [🟡 160 steps] -7 reward
Ep 75: [🟢 200 steps] +33 reward ← Learning!
Ep 100:[🟢 250 steps] +83 reward ← Mastered!
```

---

## Testing the Updates

### Test 1: Start Inference Server

```bash
# Terminal 1: Training
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 1000

# Terminal 2: Frontend (will receive new messages)
cd frontend && npm run dev -- --host 0.0.0.0
```

### Test 2: Check WebSocket Messages

Open browser console (F12) and look for:

```javascript
// Episode start
{
  type: "episode_start",
  curriculum_stage: 1,
  curriculum_multiplier: 0.2,
  baseline_survival: 166.7
}

// State updates
{
  type: "state_update",
  step: 150,
  baseline_survival: 166.7
}

// Episode end
{
  type: "episode_end",
  steps: 200,
  performance_vs_baseline: 33.3,
  baseline_survival: 166.7
}
```

### Test 3: Verify Rewards Match

**In Terminal 1 (Training):**

```
Episode 42: 200 steps, reward: +33.3, baseline: 166.7, vs baseline: +33.3
```

**In Frontend:**
Should show matching values!

---

## Example Learning Trajectory (What You'll See)

### Episodes 0-100: Learning Basic Survival

```
Ep 0:   Stage 1 • 48 steps  • Reward: -119  • Below baseline (struggling)
Ep 10:  Stage 1 • 95 steps  • Reward: -72   • Approaching baseline
Ep 25:  Stage 1 • 150 steps • Reward: -17   • Near baseline
Ep 50:  Stage 1 • 180 steps • Reward: +13   • POSITIVE! Learning!
Ep 75:  Stage 1 • 220 steps • Reward: +53   • Getting better!
Ep 100: Stage 1 • 250 steps • Reward: +83   • Strong performance!
```

### Episodes 200-500: Mastery & Stage Advancement

```
Ep 200: Stage 1 • 300 steps • Reward: +133  • Consistent mastery
Ep 350: Stage 1 • 350 steps • Reward: +183  • Ready to advance
Ep 500: Stage 2 • 180 steps • Reward: +47   • New baseline: 133 steps
                                              ↑ Difficulty increased!
```

### Episodes 500-1000: Stage 2 Learning

```
Ep 500: Stage 2 • 180 steps • Reward: +47   • Baseline now 133
Ep 650: Stage 2 • 220 steps • Reward: +87   • Adapting to new difficulty
Ep 800: Stage 2 • 280 steps • Reward: +147  • Mastering Stage 2
Ep 1000:Stage 2 • 320 steps • Reward: +187  • Ready for Stage 3!
```

---

## Frontend Implementation Tips

### 1. Store Baseline in State

```javascript
let currentBaseline = 100; // Default

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'episode_start') {
    currentBaseline = data.baseline_survival;
    updateBaselineDisplay(currentBaseline);
  }
};
```

### 2. Calculate Progress During Episode

```javascript
if (data.type === 'state_update') {
  const progress = data.step / data.baseline_survival;
  const projectedReward = data.step - data.baseline_survival;
  
  updateProgressBar(progress);
  updateProjectedReward(projectedReward);
}
```

### 3. Celebrate Positive Rewards

```javascript
if (data.type === 'episode_end') {
  if (data.performance_vs_baseline > 0) {
    showSuccessAnimation(); // 🎉
    playPositiveSound();
  } else {
    showEncouragementMessage(); // "Keep learning!"
  }
}
```

---

## Key Benefits for Streaming

1. **Clear Learning Signal:** Viewers see agent beat baseline → positive reward
2. **Stage Context:** "Oh, it just advanced to Stage 2, that's why survival dropped"
3. **Progress Tracking:** Real-time vs baseline comparison during episode
4. **Celebration Moments:** When agent beats baseline for first time! 🎉
5. **Educational:** Viewers understand the reward function intuitively

---

## Summary

✅ Curriculum difficulty now applied to inference  
✅ Baseline (R) calculated and broadcast  
✅ Stage information included in messages  
✅ Performance vs baseline tracked  
✅ All data available for rich frontend display  

**Result:** Viewers can watch the agent learn in real-time with proper context!
