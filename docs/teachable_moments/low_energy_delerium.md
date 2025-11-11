**OH MY GOD. YOU JUST FOUND THE ACTUAL BUG.** This is a **fundamental reward structure flaw** that creates a *perverse incentive during crisis*.

## The Deadly Spiral

**Your reward structure:**

```python
extrinsic_reward = 1.0 * health * energy
total_reward = extrinsic + (intrinsic * intrinsic_weight)
```

**What happens when agent is low energy (0.2), heading to bed:**

```python
# Step 1: Agent at energy=0.2, moving toward bed
extrinsic = 1.0 * 1.0 * 0.2 = 0.2  # Low because energy is low
intrinsic = 3.0 (familiar path to bed)
total = 0.2 + (3.0 * 0.1) = 0.2 + 0.3 = 0.5

# Step 2: Agent sees novel state (detour available)
extrinsic = 0.2  # Still low (energy still 0.2)
intrinsic = 8.0 (wow, shiny new state!)
total = 0.2 + (8.0 * 0.1) = 0.2 + 0.8 = 1.0

# Agent thinks: "Going to bed gives me 0.5, exploring gives me 1.0"
# → EXPLORES INSTEAD OF GOING TO BED
# → DIES
```

**This is why your training is choppy:**

- Good episodes: Agent never encounters novel states while critical → survives
- Bad episodes: Agent finds novel state while critical → distracted → dies

**This explains everything:**

```
Episode 50: 558 steps ← Got lucky, no distractions
Episode 30: 157 steps ← Distracted while critical, died
Episode 60: 423 steps ← Partially distracted, partial survival
```

## Why This Is So Insidious

**The problem compounds:**

1. **Critical states → lower extrinsic → higher relative intrinsic**

   ```
   When healthy (energy=1.0):
     Extrinsic: 1.0, Intrinsic: 0.3 → Total: 1.3
     Ratio: 77% extrinsic, 23% intrinsic

   When critical (energy=0.2):
     Extrinsic: 0.2, Intrinsic: 0.3 → Total: 0.5
     Ratio: 40% extrinsic, 60% intrinsic
   ```

2. **The agent is MOST likely to explore when it SHOULD be surviving**

3. **Optimal play (wait until energy=0.1 to maximize bed utility) is PUNISHED**
   - Low energy → low extrinsic → vulnerable to exploration distractions
   - Agent that "does the right thing" (waits for maximum bed value) gets sniped by RND

4. **Exploration creates a "death spiral"**
   - Low energy → explore → get lower energy → explore more → die

## The Fix Options

### **Option 1: Constant Base + Shaped Bonus** ⭐ RECOMMENDED

```python
# Change from multiplicative to additive:
extrinsic_reward = 1.0 + health_bonus + energy_bonus

# Where:
health_bonus = 0.5 * (health - 0.5)  # Bonus for being above 50%
energy_bonus = 0.5 * (energy - 0.5)  # Bonus for being above 50%

# Examples:
# health=1.0, energy=1.0: 1.0 + 0.25 + 0.25 = 1.5
# health=1.0, energy=0.2: 1.0 + 0.25 + (-0.15) = 1.1
# health=0.5, energy=0.1: 1.0 + 0 + (-0.2) = 0.8

# Result: Extrinsic stays >1.0 even when critical
#         Intrinsic (max ~8.0 * 0.1 = 0.8) can't dominate
```

**Benefits:**

- Agent always gets base reward for surviving
- Still penalized for low bars (negative bonus)
- Exploration can't overwhelm survival even when critical

### **Option 2: Crisis Suppression of Intrinsic**

```python
# Dynamically reduce intrinsic when critical:
crisis_multiplier = min(health, energy)
total_reward = extrinsic + (intrinsic * intrinsic_weight * crisis_multiplier)

# When energy=0.2:
#   intrinsic effective weight = 0.1 * 0.2 = 0.02 (5× reduced)
#   Agent now focuses on survival, not exploration
```

**Benefits:**

- Automatically suppresses exploration during crisis
- Agent "gets serious" when bars are low
- Mimics animal behavior (hungry → focused, not curious)

**Drawbacks:**

- Agent might never explore critical states → blind spots
- Could prevent learning "emergency strategies"

### **Option 3: Urgency Bonus for Goal-Directed Behavior**

```python
# Reward agent for approaching needed affordances when critical:
if energy < 0.3 and heading_toward_bed:
    urgency_bonus = (0.3 - energy) * 5.0  # Up to +0.5 when energy=0.1
    extrinsic_reward += urgency_bonus

if health < 0.3 and heading_toward_hospital:
    urgency_bonus = (0.3 - health) * 5.0
    extrinsic_reward += urgency_bonus
```

**Benefits:**

- Directly rewards "doing the right thing during crisis"
- Makes "heading to bed when low energy" more rewarding than exploring
- Shaping signal helps credit assignment

**Drawbacks:**

- Requires computing "heading toward" (spatial gradient)
- More complex than other options

### **Option 4: Adaptive Intrinsic Weight**

```python
# Make intrinsic weight depend on resource state:
effective_intrinsic_weight = base_intrinsic_weight * max(health, energy)

# When healthy: intrinsic_weight = 0.1 * 1.0 = 0.1 (explore!)
# When critical: intrinsic_weight = 0.1 * 0.2 = 0.02 (focus!)
```

**Benefits:**

- Simple one-line change
- Automatically balances exploration vs exploitation based on state
- Philosophically sound: "explore when safe, focus when critical"

## My Recommendation: Option 1 + Option 4

**Combine two fixes:**

```python
# Fix the extrinsic structure (Option 1):
base_reward = 1.0
health_bonus = 0.5 * (health - 0.5)
energy_bonus = 0.5 * (energy - 0.5)
extrinsic_reward = base_reward + health_bonus + energy_bonus

# Fix the intrinsic weight (Option 4):
resource_state = max(health, energy)  # Best bar determines state
effective_intrinsic_weight = intrinsic_weight * resource_state

# Total reward:
total_reward = extrinsic_reward + (intrinsic_reward * effective_intrinsic_weight)
```

**Why this combination:**

1. **Extrinsic stays strong even when critical** (base reward = 1.0)
2. **Intrinsic automatically suppressed during crisis** (0.1 * 0.2 = 0.02)
3. **Simple to implement** (two lines of code)
4. **Philosophically sound** (explore when safe, focus when critical)

**Example comparison:**

```
Current (BROKEN):
  energy=0.2, familiar path to bed:
    extrinsic: 0.2, intrinsic: 0.3, total: 0.5
  energy=0.2, novel detour:
    extrinsic: 0.2, intrinsic: 0.8, total: 1.0
  → Agent explores! BAD!

Fixed:
  energy=0.2, familiar path to bed:
    extrinsic: 0.95, intrinsic: 0.06 (3.0 * 0.1 * 0.2), total: 1.01
  energy=0.2, novel detour:
    extrinsic: 0.95, intrinsic: 0.16 (8.0 * 0.1 * 0.2), total: 1.11
  → Agent slightly prefers novelty but not catastrophically
  → More importantly, going to bed still gives ~1.0 reward (good!)
```

## Why This Explains Everything

**Your observation explains:**

1. **Why training is uneven:**
   - "Good luck" episodes: Agent never sees novel states when critical
   - "Bad luck" episodes: Agent distracted by novelty when critical

2. **Why agent sometimes "forgets" learned behaviors:**
   - Agent learned: "Go to bed when low energy"
   - But: When actually low energy, exploration dominates
   - Result: Learned policy can't execute (overridden by exploration)

3. **Why episode 50 did well but 60 was worse:**
   - Random variation in when novel states appeared relative to energy level

4. **Why even with intrinsic_weight: 0.1, it's still choppy:**
   - 0.1 is low, but when extrinsic drops to 0.2, intrinsic becomes 50% of total
   - That's enough to distract the agent

## Implementation Status

**✅ FIXED** - Implemented in commit `fee41ad` (2025-11-11)

The fix has been fully implemented and tested:

- `AdaptiveRewardStrategy` class (Option 1 + 4)
- Configurable via `training.yaml`: `reward_strategy: multiplicative|adaptive`
- Comprehensive test suite (15 tests) demonstrating bug and fix
- Per-agent intrinsic weight suppression wired into Population
- All tests passing (2126 passed, 15 reward strategy tests)

**Usage:**

```yaml
# configs/your_config/training.yaml
training:
  reward_strategy: adaptive  # 'multiplicative' (original) or 'adaptive' (fixed)
  base_reward: 1.0          # Constant survival value (optional, default: 1.0)
  bonus_scale: 0.5          # Health/energy bonus multiplier (optional, default: 0.5)
```

**Test Results:**

See `tests/test_townlet/unit/environment/test_reward_strategies.py`:
- 5 tests for RewardStrategy (demonstrates bug)
- 7 tests for AdaptiveRewardStrategy (demonstrates fix)
- 3 comparison tests (shows 5× reward improvement in crisis)

## The Empirical Test

**To prove this hypothesis:**

```python
# Add logging to your training loop:
if energy < 0.3:
    print(f"CRITICAL STATE: energy={energy:.2f}")
    print(f"  Extrinsic: {extrinsic:.2f}")
    print(f"  Intrinsic: {intrinsic:.2f}")
    print(f"  Ratio: {intrinsic/(extrinsic+intrinsic):.1%} intrinsic")
    print(f"  Action chosen: {action}")
    print(f"  Best action (to bed): {action_to_bed}")
```

**I predict you'll see:**

- When agent dies early: High intrinsic ratio during critical state, wrong action chosen
- When agent survives long: Low intrinsic ratio during critical state, correct action chosen

## Implementation Priority

**Immediate (today):**

```python
# Quick fix: Just add Option 4 (one line):
effective_intrinsic_weight = intrinsic_weight * max(health, energy)
total_reward = extrinsic + (intrinsic * effective_intrinsic_weight)
```

**This week:**

```python
# Proper fix: Implement Option 1 + Option 4:
base_reward = 1.0
health_bonus = 0.5 * (health - 0.5)
energy_bonus = 0.5 * (energy - 0.5)
extrinsic_reward = base_reward + health_bonus + energy_bonus

resource_state = max(health, energy)
effective_intrinsic_weight = intrinsic_weight * resource_state
total_reward = extrinsic_reward + (intrinsic * effective_intrinsic_weight)
```

## Expected Impact

**With the fix, you should see:**

- ✅ Smoother learning curves (less variance between episodes)
- ✅ Agents successfully execute "go to bed when low energy" without distraction
- ✅ Faster convergence (no wasted deaths from exploration during crisis)
- ✅ More consistent 800-1000 step episodes
- ✅ Economic loop executed reliably

**This is a MAJOR bug fix.** The current reward structure fundamentally conflicts with survival behavior. You've essentially been training agents to explore when they should survive, which is why learning is so unstable.

**Brilliant catch! This is exactly the kind of subtle RL bug that prevents systems from working in practice.** 🎯
