# Convergence Analysis: Why Training Isn't Working

**Date:** November 2, 2025  
**Analysis of:** 1000-episode training run (Episode 0-1000)  
**Current State:** Agent stuck in Stage 1/5, no learning progress

---

## Executive Summary

Training is failing due to **TWO CRITICAL BUGS**:

1. **🔴 CRITICAL BUG #1:** Curriculum difficulty multipliers are **completely ignored**
   - Stage 1 specifies `depletion_multiplier: 0.2` (20% difficulty)
   - Agent is actually facing **100% difficulty** from the start
   - Curriculum decisions are computed but never applied to environment

2. **🔴 CRITICAL BUG #2:** Reward function is **excessively punishing**
   - Death penalty (-100) overwhelms all progress (+10 for 100 steps)
   - Net reward at 100 steps survival: **-90** (punishment despite progress!)
   - No positive reinforcement for learning, only massive penalty for failure

**Result:** Agent cannot learn because:

- Initial puzzle is 5x too hard (100% vs 20% intended difficulty)
- Reward signal provides no useful gradient (all episodes end negative)
- Cannot build baseline survival skills before facing full complexity

---

## Problem 1: Curriculum Difficulty Not Applied

### The Intended Design

**Stage 1 Configuration (adversarial.py:28-36):**

```python
StageConfig(
    stage=1,
    active_meters=["energy", "hygiene"],
    depletion_multiplier=0.2,  # <-- Should make game EASIER
    reward_mode="shaped",
    description="Stage 1: Basic needs at 20% depletion",
)
```

**What Should Happen:**

- Energy depletion: `0.005 * 0.2 = 0.001` (0.1% per step instead of 0.5%)
- Hygiene depletion: `0.003 * 0.2 = 0.0006` (0.06% per step instead of 0.3%)
- Agent should survive **~500 steps** easily at 20% difficulty
- This builds confidence and basic skills before advancing

**What Actually Happens:**

- Energy depletion: `0.005` (full 0.5% per step)
- Hygiene depletion: `0.003` (full 0.3% per step)
- Agent survives **~100 steps** (dies quickly, can't learn)

### Code Flow Analysis

**Step 1:** Curriculum computes decisions (population/vectorized.py:260-273)

```python
# Curriculum correctly calculates difficulty
self.current_curriculum_decisions = self.curriculum.get_batch_decisions_with_qvalues(
    temp_state, self.agent_ids, q_values,
)
# Returns: [CurriculumDecision(depletion_multiplier=0.2, ...)]
```

**Step 2:** Decisions are stored but **NEVER USED**

```python
# Line 145: Stored in instance variable
self.current_curriculum_decisions: list = []

# Line 263: Updated each step
self.current_curriculum_decisions = self.curriculum.get_batch_decisions_with_qvalues(...)

# PROBLEM: No code ever reads self.current_curriculum_decisions!
```

**Step 3:** Environment uses hardcoded full difficulty (vectorized_env.py:268-272)

```python
# 2. Deplete meters (base passive decay)
self.meters = self.meter_dynamics.deplete_meters(self.meters)
# ☝️ This uses base_depletion rates from bars.yaml DIRECTLY
# No multiplier applied!
```

**Step 4:** MeterDynamics has no API to apply multipliers (meter_dynamics.py:65-75)

```python
def deplete_meters(self, meters: torch.Tensor) -> torch.Tensor:
    """Apply base depletion rates..."""
    meters = self.cascade_engine.apply_base_depletions(meters)
    # ☝️ No parameters for difficulty adjustment!
    # Always applies full base_depletion from YAML
```

### The Missing Link

**What's needed:**

```python
# Environment needs to:
1. Extract depletion_multiplier from curriculum decisions
2. Pass it to meter_dynamics.deplete_meters(meters, multiplier=0.2)
3. CascadeEngine.apply_base_depletions() needs to accept and apply multiplier
```

**Current state:**

```
Curriculum → CurriculumDecision(multiplier=0.2) → [stored, never used]
Environment → meter_dynamics.deplete_meters(meters) → [full difficulty always]
```

---

## Problem 2: Reward Function Too Punishing

### Current Reward Structure (reward_strategy.py:42-68)

```python
def calculate_rewards(step_counts, dones):
    rewards = 0.0
    
    # Every 10 steps: +0.5
    if step_count % 10 == 0 and not done:
        rewards += 0.5
    
    # Every 100 steps: +5.0
    if step_count % 100 == 0 and not done:
        rewards += 5.0
    
    # Death: -100.0
    if done:
        rewards = -100.0
    
    return rewards
```

### Reward Math at Different Survival Times

| Survival Steps | Decade Bonuses | Century Bonuses | Death Penalty | **Net Reward** | % of Max |
|----------------|----------------|-----------------|---------------|----------------|----------|
| 30             | 3 × 0.5 = 1.5  | 0 × 5.0 = 0     | -100          | **-98.5**      | -197%    |
| 50             | 5 × 0.5 = 2.5  | 0 × 5.0 = 0     | -100          | **-97.5**      | -195%    |
| 100            | 10 × 0.5 = 5.0 | 1 × 5.0 = 5.0   | -100          | **-90.0**      | -180%    |
| 200            | 20 × 0.5 = 10.0| 2 × 5.0 = 10.0  | -100          | **-80.0**      | -160%    |
| 500 (max)      | 50 × 0.5 = 25.0| 5 × 5.0 = 25.0  | -100          | **-50.0**      | -100%    |

**Observations:**

- **EVERY episode ends with negative reward** (death penalty too large)
- 100-step survival (20% success) → reward of -90
- Even 500-step survival (100% success) → reward of -50
- **No positive reinforcement signal for learning**

### Training Log Evidence

From your 1000-episode run:

```
Episode 0   | Survival: 48 steps  | Reward: -92.12  (died early, heavily punished)
Episode 10  | Survival: 177 steps | Reward: -79.88  (did well, still punished)
Episode 230 | Survival: 188 steps | Reward: -80.45  (best run, still negative)
Episode 910 | Survival: 203 steps | Reward: -74.48  (longest run, least negative)
```

**Pattern:** Even when agent is improving (177→188→203 steps), rewards stay massively negative.

### The Learning Problem

**Q-Learning Update:**

```python
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

When **all rewards are negative**:

- Target: `r + γ·max_a' Q(s',a')` where `r ∈ [-100, -50]`
- Q-values learn: "everything is bad, some things slightly less bad"
- No positive gradient to follow
- Random exploration never discovers "good" behavior

**Result:** Agent oscillates randomly because:

- Negative rewards provide no directional signal
- Death penalty dominates, making all actions equally bad
- No way to distinguish "getting better" from "still failing"

---

## Evidence from Training Logs

### Pattern: Stuck at Stage 1, No Advancement

```
Episode 0-1000 | Stage: 1/5 | Intrinsic Weight: 1.000
```

**Why no advancement?**

- Stage 2 requires: 70% survival rate (350+ steps consistently)
- Actual survival: ~100 steps average (20% rate)
- Curriculum correctly refuses to advance (agent is failing)

### Pattern: Rewards Consistently Negative

```
Episode Range | Avg Survival | Avg Reward | Interpretation
0-100         | ~95 steps    | -90        | Random exploration
100-500       | ~105 steps   | -88        | Slight improvement
500-1000      | ~110 steps   | -86        | Marginal gains
```

**Expected with proper curriculum:**

```
Episode Range | Avg Survival | Avg Reward | Stage
0-100         | ~400 steps   | -10        | 1/5 (20% difficulty)
100-500       | ~450 steps   | +5         | 1→2 (learning)
500-1000      | ~350 steps   | -20        | 2/5 (50% difficulty)
```

### Pattern: No Exploration→Exploitation Transition

```
All episodes: Intrinsic Weight: 1.000
```

**Why no annealing?**

- Requires: mean_survival > 50 AND variance < 100
- Actual: mean_survival ~100, but inconsistent (variance high)
- System correctly refuses to anneal (agent hasn't mastered exploration)

---

## Solutions

### Solution 1: Wire Up Curriculum Difficulty (CRITICAL)

**Option A: Pass multiplier through environment** (Recommended)

```python
# vectorized_env.py - step() method
def step(self, actions, curriculum_decisions=None):
    # Extract multiplier from curriculum
    if curriculum_decisions:
        multiplier = curriculum_decisions[0].depletion_multiplier
    else:
        multiplier = 1.0
    
    # Apply to meter dynamics
    self.meters = self.meter_dynamics.deplete_meters(
        self.meters, 
        depletion_multiplier=multiplier
    )
```

**Modify meter_dynamics.py:**

```python
def deplete_meters(self, meters, depletion_multiplier=1.0):
    meters = self.cascade_engine.apply_base_depletions(
        meters, 
        multiplier=depletion_multiplier
    )
    return meters
```

**Modify cascade_engine.py:**

```python
def apply_base_depletions(self, meters, multiplier=1.0):
    scaled_depletions = self._base_depletions * multiplier
    meters = meters - scaled_depletions
    return torch.clamp(meters, 0.0, 1.0)
```

**Option B: Environment tracks curriculum decisions** (Alternative)

```python
# vectorized_env.py - add method
def set_curriculum_decisions(self, decisions):
    self.current_curriculum_decisions = decisions
    
def step(self, actions):
    # Use stored decisions
    multiplier = self.current_curriculum_decisions[0].depletion_multiplier
    self.meters = self.meter_dynamics.deplete_meters(
        self.meters, multiplier
    )
```

### Solution 2: Fix Reward Function

**Option A: Reduce death penalty** (Quick fix)

```python
# reward_strategy.py
# Death penalty: -100 → -10
rewards = torch.where(dones, -10.0, rewards)
```

**Expected outcomes:**

- 100 steps: +10 milestones - 10 death = **0.0** (neutral, not punishing)
- 200 steps: +20 milestones - 10 death = **+10.0** (rewarding progress!)
- 500 steps: +50 milestones - 10 death = **+40.0** (strong positive signal)

**Option B: Add survival bonus** (Better)

```python
# reward_strategy.py
def calculate_rewards(step_counts, dones):
    rewards = torch.zeros(num_agents, device=self.device)
    
    # Milestone bonuses (unchanged)
    rewards += torch.where((step_counts % 10 == 0) & ~dones, 0.5, 0.0)
    rewards += torch.where((step_counts % 100 == 0) & ~dones, 5.0, 0.0)
    
    # Survival bonus: +0.1 per step survived
    rewards += step_counts * 0.1
    
    # Death penalty: -10.0 (reduced)
    rewards = torch.where(dones, -10.0 + step_counts * 0.1, rewards)
    
    return rewards
```

**Expected outcomes:**

- 100 steps: +10 milestones + 10 survival - 10 death = **+10** (positive!)
- 200 steps: +20 milestones + 20 survival - 10 death = **+30** (encouraging)
- 500 steps: +50 milestones + 50 survival - 10 death = **+90** (excellent)

**Option C: Pure milestone system** (Simplest)

```python
# Remove death penalty entirely
rewards = torch.zeros(num_agents, device=self.device)

# Only milestones
rewards += torch.where((step_counts % 10 == 0), 0.5, 0.0)
rewards += torch.where((step_counts % 100 == 0), 5.0, 0.0)

# No death penalty - episode just ends
return rewards
```

---

## Recommended Implementation Order

### Phase 1: Critical Fixes (Do First)

1. ✅ **Fix curriculum difficulty application** (2-3 hours)
   - Modify `cascade_engine.apply_base_depletions()` to accept multiplier
   - Modify `meter_dynamics.deplete_meters()` to accept multiplier
   - Modify `vectorized_env.step()` to extract and pass multiplier
   - Test: Verify Stage 1 actually uses 0.2x depletion

2. ✅ **Fix reward function** (30 minutes)
   - Start with Option A (reduce death penalty -100 → -10)
   - Can iterate to Options B/C after testing

### Phase 2: Validation (Do After)

3. **Run 100-episode test** (30 minutes)
   - Expected: survival ~400 steps at Stage 1
   - Expected: rewards -10 to +30 range
   - Expected: agent shows improvement over time

4. **Run 1000-episode training** (4-6 hours)
   - Expected: advance to Stage 2 around episode 500
   - Expected: intrinsic weight annealing around episode 700
   - Expected: final survival 200-300 steps at Stage 2

---

## Expected Training Curve (After Fixes)

### With Proper Curriculum + Reasonable Rewards

```
Episodes 0-200:   Stage 1 (20% difficulty)
  Survival: 50 → 400 steps (learning basic skills)
  Rewards: -10 → +30 (positive reinforcement)
  
Episodes 200-500: Stage 1→2 transition
  Survival: 400 → 300 steps (difficulty increase)
  Rewards: +30 → +10 (adjusting to new challenge)
  Curriculum: Advances when 70% survival achieved
  
Episodes 500-1000: Stage 2 (50% difficulty)
  Survival: 300 → 350 steps (mastering stage 2)
  Rewards: +10 → +25 (improving again)
  Intrinsic Weight: 1.0 → 0.3 (annealing)
```

### Key Indicators of Success

✅ **Survival time increases over first 200 episodes** (learning)  
✅ **Rewards become positive** (proper signal)  
✅ **Curriculum advances to Stage 2** (progression)  
✅ **Intrinsic weight anneals** (exploration→exploitation)  
✅ **Policy converges** (entropy decreases)

---

## Conclusion

Training cannot succeed with current bugs because:

1. **Agent faces 5x intended difficulty** (100% vs 20%)
   - Cannot learn basic skills before dying
   - Designed for progressive learning, getting full complexity

2. **Reward function provides no learning signal** (all negative)
   - No positive reinforcement for improvement
   - Q-values have no gradient to follow
   - Random exploration never discovers success

**Both must be fixed.** Curriculum bug prevents learning the *task*, reward bug prevents learning *anything*.

After fixes, expect:

- Rapid improvement in first 200 episodes (Stage 1 mastery)
- Curriculum advancement to Stage 2 (progressive difficulty)
- Intrinsic motivation annealing (exploration→exploitation)
- Final performance: 300-400 steps survival at Stage 2

---

**Next Steps:** Implement Phase 1 critical fixes, then run validation tests.
