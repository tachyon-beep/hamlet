# Survival Reward Fix

## Problem Discovered

**Date**: 2025-10-30
**Episode**: 1270-1570
**Issue**: Agents surviving longer were getting MORE NEGATIVE rewards

### Example from Training Logs
```
Episode 1300: 39 steps  | Reward: -64.65   (better!)
Episode 1400: 219 steps | Reward: -621.71  (worse!)
Episode 1570: 161 steps | Reward: -443.43  (worse!)
```

**Backwards incentive**: Agent is literally punished for surviving longer.

### Root Cause

Complex meter-based reward shaping accumulated penalties:
- Every meter below threshold gave negative reward EVERY STEP
- Worst case: -10.5 per step
- Over 200 steps: -2100 cumulative penalty
- Result: **Survival is punished, not rewarded**

---

## Solution: Simple Survival Reward

### New Reward Structure
```python
rewards = +1.0  # Per step survived
rewards = -100  # For dying
```

### Examples

**Survives 200 steps, then dies**:
- 199 steps × +1.0 = +199
- 1 death × -100 = -100
- **Total = +99** ✓

**Survives 200 steps without dying**:
- 200 steps × +1.0 = +200
- **Total = +200** ✓

**Dies quickly (20 steps)**:
- 19 steps × +1.0 = +19
- 1 death × -100 = -100
- **Total = -81** ✗

### Direct Survival Incentive

| Survival Time | Old Reward (worst case) | New Reward      |
|---------------|-------------------------|-----------------|
| 20 steps      | -310                    | -81             |
| 100 steps     | -1150                   | 0 (or +100)     |
| 200 steps     | -2200                   | +100 (or +200)  |
| 300 steps     | -3250                   | +200 (or +300)  |

**Result**: Longer survival → Better reward (as it should be!)

---

## Implementation

**File**: `src/townlet/environment/vectorized_env.py`

**Changed**: `_calculate_shaped_rewards()` method

```python
def _calculate_shaped_rewards(self) -> torch.Tensor:
    """
    SIMPLE SURVIVAL REWARD: Directly reward staying alive.

    Problem with old complex rewards: Longer survival → more accumulated penalties → negative rewards.
    Solution: +1.0 per step survived, -100 for dying.
    """
    # Base reward: +1.0 for surviving this step
    rewards = torch.ones(self.num_agents, device=self.device)

    # Death penalty: -100.0 for dying
    rewards = torch.where(self.dones, -100.0, rewards)

    return rewards
```

**Disabled**: Complex meter-based shaping (renamed to `_calculate_shaped_rewards_COMPLEX_DISABLED` for reference)

---

## Expected Training Behavior

### With Simple Survival Rewards

**Early episodes (random policy)**:
- Survival: 5-20 steps
- Reward: -95 to -80
- Behavior: Random movement, quick death

**Learning phase (1000-2000 episodes)**:
- Survival: 50-150 steps
- Reward: -50 to +50
- Behavior: Learning to interact with affordances

**Converged policy (3000+ episodes)**:
- Survival: 200-400 steps
- Reward: +100 to +300
- Behavior: Systematic cycles, long survival

### Key Metrics to Watch

- **Survival time**: Should steadily increase
- **Reward**: Should become less negative, then positive
- **Correlation**: Reward should track survival time (not inverse!)

---

## Pedagogical Value

This demonstrates a critical RL lesson: **Reward shaping can backfire**

**What went wrong**:
- Intended: Guide agent with meter-based feedback
- Reality: Penalties accumulated, punished survival
- Lesson: Simple rewards often work better than complex shaping

**Teaching moment**:
- Show students the logs (survival up, reward down)
- Explain accumulating penalties
- Demonstrate simple survival reward fix
- Connect to real-world ML: "measure what you want, not proxies"

---

## Next Steps

1. **Restart training** with new reward structure:
   ```bash
   # Delete old checkpoints (incompatible reward scale)
   rm -rf checkpoints_level2/*
   rm demo_level2.db

   # Start fresh training
   python -m hamlet.demo.runner configs/townlet_level_2_pomdp.yaml demo_level2.db checkpoints_level2 10000
   ```

2. **Monitor correlation**:
   - Plot survival vs reward (should be positive correlation)
   - Verify rewards increase as agent learns
   - Confirm long survival = positive rewards

3. **Compare to old system**:
   - Keep logs from old complex rewards
   - Show students the before/after
   - Highlight the backwards incentive problem

---

## Design Philosophy

> "Low and slow" - Simple rewards, patient training

**Principles**:
- **Simple rewards**: Directly measure what you want (survival)
- **Patient training**: Let the agent figure out HOW to survive
- **Sparse feedback**: Agent must explore strategies
- **Clear objective**: Survive as long as possible

**Why this works**:
- Agent discovers affordance interactions through exploration
- Intrinsic rewards (RND) encourage discovery
- Survival reward provides clear objective
- No confusing meter penalties to navigate

---

## Code Reference

**Active reward function**: `src/townlet/environment/vectorized_env.py:684-700`

**Disabled complex rewards**: `src/townlet/environment/vectorized_env.py:702-859` (kept for reference)

**Config**: `configs/townlet_level_2_pomdp.yaml` (no changes needed)

**Training logs**: Monitor for positive correlation between survival and reward
