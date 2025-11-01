# Real-Time Projected Reward Feature

**Date:** November 2, 2025  
**Status:** ✅ IMPLEMENTED

## What Changed

Added `projected_reward` to state updates so the frontend can show real-time learning progress during episodes.

### WebSocket State Update (Enhanced)

```json
{
  "type": "state_update",
  "step": 150,
  "projected_reward": -16.7,           // NEW: current_step - baseline
  "baseline_survival": 166.7,
  "cumulative_reward": 0.0,            // Legacy (always 0 in new system)
  "grid": { ... },
  "agent_meters": { ... },
  "q_values": [...]
}
```

## Calculation

```python
projected_reward = current_step - baseline_survival

# Examples (Stage 1, baseline=167):
# Step 50:  projected = 50 - 167 = -117  (way below baseline)
# Step 150: projected = 150 - 167 = -17  (approaching baseline)
# Step 167: projected = 167 - 167 = 0    (at baseline)
# Step 200: projected = 200 - 167 = +33  (beating baseline!)
```

## Frontend Display Ideas

### 1. Real-Time Reward Counter

```
Current Projected Reward: -16.7
[===================>    ] 150/167 steps
```

Updates every step, showing if agent is on track.

### 2. Color-Coded Progress

```css
.reward-positive { color: #4CAF50; }  /* Green: beating baseline */
.reward-neutral { color: #FFC107; }   /* Yellow: near baseline */
.reward-negative { color: #F44336; }  /* Red: below baseline */
```

### 3. Live Chart

Plot `projected_reward` over time during episode:
- Starts negative (e.g., -167 at step 0)
- Climbs toward zero
- Goes positive if agent beats baseline
- Shows learning gradient visually

### 4. Milestone Markers

```
Step 150: Projected reward -17
└─ 17 steps away from breaking even!
└─ If episode ended now: reward = -17

Step 200: Projected reward +33 ✅
└─ 33 steps above baseline!
└─ If episode ended now: reward = +33
```

### 5. Performance Indicator

```javascript
const progress = currentStep / baselineSurvival;
const status = progress < 0.5 ? "struggling" :
               progress < 1.0 ? "learning" :
               "mastered";

// Display:
// 🔴 Struggling (0-50% of baseline)
// 🟡 Learning (50-100% of baseline)
// 🟢 Mastered! (>100% of baseline)
```

## Why This Matters

**Before:** Viewers only saw reward at episode end. No sense of progress during episode.

**After:** Viewers see:
1. **Real-time learning signal** - Is agent beating baseline?
2. **Progress tracking** - How close to positive reward?
3. **Celebration moments** - When projected_reward crosses 0!
4. **Learning gradient** - Visual representation of improvement

## Example Episode Visualization

```
Episode 42 • Stage 1/5 • Baseline: 167 steps

Step 0:   Projected: -167 [▱▱▱▱▱▱▱▱▱▱] 0%
Step 50:  Projected: -117 [██▱▱▱▱▱▱▱▱] 30%
Step 100: Projected: -67  [████▱▱▱▱▱▱] 60%
Step 150: Projected: -17  [████████▱▱] 90%
Step 167: Projected: 0    [██████████] 100% 🎯 BASELINE!
Step 200: Projected: +33  [████████████] 120% 🎉 MASTERED!

Final: 200 steps, reward: +33
```

## Integration Example

```javascript
// In frontend state update handler:
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'state_update') {
    const projected = data.projected_reward;
    const baseline = data.baseline_survival;
    const step = data.step;
    
    // Update UI
    updateRewardCounter(projected);
    updateProgressBar(step, baseline);
    
    // Celebrate milestones
    if (projected >= 0 && !hasReachedBaseline) {
      showCelebration("Baseline reached! 🎯");
      hasReachedBaseline = true;
    }
    
    // Plot on chart
    rewardChart.addPoint(step, projected);
  }
};
```

## Testing

Start live inference and watch the console:

```bash
# Terminal 1: Training
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 100

# Terminal 2: Check WebSocket messages
# Open browser console and filter for "projected_reward"
```

Expected behavior:
- `projected_reward` starts at `-baseline` (e.g., -167)
- Increases by 1 each step
- Goes positive when agent survives past baseline
- Matches `total_reward` at episode end

## Summary

✅ Real-time learning signal available every step  
✅ Frontend can show progress toward positive reward  
✅ Viewers see agent learning as it happens  
✅ Creates engaging "will it beat baseline?" moments  

**Result:** Much more engaging streaming experience! 🚀
