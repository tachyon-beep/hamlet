# Interoception-Based Reward Design: Modeling Internal Awareness

**Date**: 2025-11-03
**Status**: Proposed Design
**Related**: `milestone_rewards_design.md`, `reward_hacking_interact_spam.md`

---

## Executive Summary

Moving from flat per-step rewards (`+1.0 alive, 0.0 dead`) to **health-energy modulated rewards** (`health × energy × 1.0`) solves a critical temporal credit assignment problem while modeling a real biological capability: **interoception** (awareness of internal body states).

**Key Insight**: Humans don't need to die to learn "low energy is bad" - we *feel* tired. This reward structure gives agents the same immediate feedback about their internal state.

---

## The Problem: Wasteful Resource Usage

### Current Flat Reward Behavior

```python
# Current reward: +1.0 per step alive, 0.0 when dead
reward = 1.0 if alive else 0.0
```

**What the agent learns:**
```
Episode Early On (before convergence):
Step 1-50:   Energy 100% → 95%  [reward = 1.0 per step = 50.0 total]
Step 51:     Sleep (cost $1)     [energy 95% → 100%]
Step 52-100: Energy 100% → 95%  [reward = 1.0 per step = 50.0 total]
Step 101:    Sleep (cost $1)     [energy 95% → 100%]
Step 102-150: Energy 100% → 95% [reward = 1.0 per step = 50.0 total]
...
Step 500:    Out of money, can't sleep anymore
Step 501-600: Energy depletes to 0%
Step 600:    DIES [reward = 0.0]

Agent learns: "Bed good! Sleep often!"
Agent DOESN'T learn: "Only sleep when ACTUALLY TIRED!"
```

### The Core Issue: No Gradient for Timing

Both scenarios produce identical reward streams:

**Wasteful Sleep (energy 95% → 100%):**
```python
before_sleep = [1.0, 1.0, 1.0, ...]  # 50 steps
after_sleep  = [1.0, 1.0, 1.0, ...]  # 50 steps
Net gain: 0.0 reward difference
Cost: $1
ROI: Negative!
```

**Optimal Sleep (energy 10% → 100%):**
```python
before_sleep = [1.0, 1.0, 1.0, ...]  # 50 steps (but would die soon!)
after_sleep  = [1.0, 1.0, 1.0, ...]  # 50 steps (life saved!)
Net gain: 0.0 reward difference (until death prevented much later)
Cost: $1
ROI: Positive, but only clear in hindsight!
```

**Problem**: The gradient for "when to sleep" only appears hundreds of steps later when the agent dies. Classic **sparse reward / temporal credit assignment** issue.

---

## The Solution: Health-Energy Modulated Rewards

```python
# Proposed reward: interoception-aware
reward = 1.0 * (health / 100.0) * (energy / 100.0) if alive else 0.0
```

### Why This Works: Immediate ROI Signal

**Wasteful Sleep (energy 95%):**
```python
# Before sleep:
reward = 1.0 * 1.0 * 0.95 = 0.95 per step  # "Feeling pretty good!"
# After $1 sleep:
reward = 1.0 * 1.0 * 1.00 = 1.00 per step  # "Feeling slightly better"

Net gain: +0.05 reward/step
Cost: $1
ROI: LOW (need 20 steps to break even)
Q-network learns: "Not worth it at 95% energy"
```

**Optimal Sleep (energy 10%):**
```python
# Before sleep:
reward = 1.0 * 1.0 * 0.10 = 0.10 per step  # "EXHAUSTED!"
# After $1 sleep:
reward = 1.0 * 1.0 * 1.00 = 1.00 per step  # "Fully rested!"

Net gain: +0.90 reward/step
Cost: $1
ROI: HIGH (breaks even in ~1 step!)
Q-network learns: "Totally worth it at 10% energy"
```

### Natural Behavior Emerges

The agent discovers optimal timing through **immediate feedback**:

```
Episode Timeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1:   Energy 100%, reward = 1.00  ✅ "Peak performance"
          Q(SLEEP) = low (waste of money)
          Q(WAIT)  = high (no benefit to sleep)
          Action: WAIT

Step 50:  Energy 95%, reward = 0.95   ⚠️ "Slightly tired"
          Q(SLEEP) = medium-low (marginal benefit)
          Q(WAIT)  = medium-high (still mostly okay)
          Action: WAIT

Step 150: Energy 50%, reward = 0.50   🟡 "Getting tired..."
          Q(SLEEP) = medium (decent ROI emerging)
          Q(WAIT)  = medium (losing 0.5/step hurts)
          Action: Borderline (learns tradeoff)

Step 200: Energy 20%, reward = 0.20   🔴 "CRITICAL!"
          Q(SLEEP) = HIGH (huge ROI)
          Q(WAIT)  = LOW (hemorrhaging reward)
          Action: SLEEP NOW!

Step 201: Energy 100%, reward = 1.00  ✅ "Refreshed!"
          Immediate feedback: "That was worth it!"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Biological Justification: Interoception

### What is Interoception?

**Interoception**: The sense of the internal state of one's body. Humans are constantly aware of:
- Fatigue (energy depletion)
- Pain/sickness (health depletion)
- Hunger (satiation depletion)
- Thirst (hydration depletion)

This is **not cheating** - it's modeling a real biological capability!

### Why Humans Don't Need Death to Learn

```
Human Learning (with interoception):
"I feel tired → I should rest"
  ↑              ↑
  Immediate     Immediate
  sensation     response

Timeline: ~1 second

Agent Learning (flat reward):
"I ran out of energy 100 steps ago → maybe I should have slept earlier?"
  ↑                                    ↑
  Death                               Delayed inference

Timeline: 100+ steps (sparse reward problem)
```

### Other Biological Systems Use This

- **Pain**: Immediate negative feedback prevents injury
- **Hunger**: Immediate discomfort drives food-seeking
- **Thirst**: Immediate sensation prevents dehydration

Agents *should* have access to meter state in their reward function - it models sensory feedback!

---

## Pedagogical Value: Multiple Teaching Moments

### 1. Reward Shaping vs. Feature Access

**Question**: "Isn't this reward shaping?"

**Answer**: It depends on your definition:
- ❌ **Arbitrary shaping**: Adding proximity bonuses, path bonuses, etc.
- ✅ **Modeling capabilities**: Giving agent access to information *humans have*

Humans know their energy level. Agents should too!

### 2. Temporal Credit Assignment

**Problem**: Q-learning struggles when rewards are delayed.

**Example**:
```python
# Sparse reward (episode-level):
rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]  # Only at end!
# Which action mattered? Hard to tell!

# Dense reward (per-step):
rewards = [1, 1, 1, 1, 0.5, 0.5, 0.8, 0.9, 1, 1]  # Every step!
# Clear gradient: action at step 5 caused drop!
```

Energy-modulated rewards provide **immediate credit** for good/bad actions.

### 3. Resource Management & ROI

Agents naturally learn **return on investment**:

```python
# Agent's internal Q-value reasoning:
def should_sleep(energy, money):
    current_reward_rate = energy / 100.0
    sleep_cost = 1.0  # dollars
    sleep_benefit = (1.0 - current_reward_rate) * 100  # future reward gain

    roi = sleep_benefit - sleep_cost

    if energy > 80%:
        return False  # roi < 0 (wasteful)
    elif energy < 30%:
        return True   # roi >> 0 (critical!)
    else:
        return coin_flip()  # roi ≈ 0 (borderline)
```

This is **learned**, not programmed!

### 4. Comparing Learning Curves

**Perfect teaching demo**:

```python
# Train two agents side-by-side:
Agent A: Flat reward (r = 1.0)
Agent B: Interoception reward (r = health × energy)

# Plot metrics:
- Average survival time
- Money wasted on unnecessary sleeps
- Energy at time of sleep actions
- Sample efficiency (episodes to convergence)

Expected result: Agent B learns faster and uses resources optimally
```

Students see that **modeling biological capabilities beats arbitrary shaping**.

---

## Implementation Details

### Proposed Change: reward_strategy.py

```python
class RewardStrategy:
    """
    Calculates interoception-aware per-step survival rewards.

    Reward Formula:
    - Alive: health × energy (both normalized to [0,1])
    - Dead: 0.0

    This models human interoception - we're aware of tiredness/sickness
    and use that immediate feedback to guide behavior.
    """

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
        baseline_steps: torch.Tensor | float | list[float],
        meters: torch.Tensor,  # NEW: [num_agents, 8] meter values
    ) -> torch.Tensor:
        """
        Calculate health-energy aware per-step survival rewards.

        Args:
            step_counts: [num_agents] current step count
            dones: [num_agents] whether each agent is dead
            baseline_steps: Retained for API compatibility (unused)
            meters: [num_agents, 8] meter values (energy=0, health=6)

        Returns:
            rewards: [num_agents] calculated rewards
        """
        # Extract and normalize meters
        energy = meters[:, 0].clamp(min=0.0, max=100.0) / 100.0  # [0, 1]
        health = meters[:, 6].clamp(min=0.0, max=100.0) / 100.0  # [0, 1]

        # Interoception-aware reward
        rewards = torch.where(
            dones,
            0.0,                    # Dead: no reward
            health * energy * 1.0,  # Alive: modulated by internal state
        )

        return rewards
```

### Expected Behavior Changes

**Before (flat reward):**
```
Episode 1: 15 sleeps (many wasteful), survives 300 steps
Episode 10: 12 sleeps (some wasteful), survives 350 steps
Episode 50: 8 sleeps (few wasteful), survives 400 steps
```

**After (interoception reward):**
```
Episode 1: 15 sleeps (many wasteful), survives 250 steps (lower reward/step)
Episode 10: 8 sleeps (fewer wasteful), survives 350 steps
Episode 50: 4 sleeps (optimal timing), survives 500 steps
```

Faster convergence to optimal resource management!

---

## Visualization: Reward Landscape

```
Per-Step Reward by Energy Level:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.0 │████████████████  ← Energy 100%: "Peak performance!"
    │                    ROI of sleep: NEGATIVE (don't waste money)
0.9 │██████████████
0.8 │████████████
0.7 │██████████
0.6 │████████
0.5 │██████        ← Energy 50%: "Getting tired..."
    │                ROI of sleep: NEUTRAL (breakeven)
0.4 │████
0.3 │██
0.2 │█             ← Energy 20%: "CRITICAL!"
    │                ROI of sleep: POSITIVE (sleep NOW!)
0.1 │
0.0 │              ← Dead (game over)
    └─────────────────────────────────────────────────►
                Energy Level (%)

Gradient is clear! Agent has immediate feedback.
```

---

## Testing Strategy

### Unit Tests (TDD)

```python
class TestInteroceptionRewards:
    def test_full_health_and_energy_gives_max_reward(self):
        """Both meters at 100% should give reward = 1.0"""
        # health=100, energy=100 → 1.0 * 1.0 = 1.0

    def test_half_health_and_energy_gives_quarter_reward(self):
        """Both meters at 50% should give reward = 0.25"""
        # health=50, energy=50 → 0.5 * 0.5 = 0.25

    def test_critical_energy_gives_low_reward(self):
        """Energy at 10% should give reward = 0.1"""
        # health=100, energy=10 → 1.0 * 0.1 = 0.1

    def test_dead_gives_zero_reward(self):
        """Dead agents get 0.0 regardless of meters"""
        # dones=True → 0.0

    def test_reward_gradient_exists(self):
        """Reward should decrease smoothly as energy depletes"""
        # energy=[100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        # rewards should be monotonically decreasing
```

### Integration Tests

```python
def test_agent_learns_optimal_sleep_timing():
    """Agent should learn to sleep at low energy, not high energy"""
    # Train L0 minimal for 500 episodes
    # Measure: average energy level at time of sleep action
    # Expected: early episodes ~80%, late episodes ~30%

def test_fewer_wasted_sleeps():
    """Agent should reduce unnecessary sleeps over time"""
    # Track: number of sleeps when energy > 70%
    # Expected: decreases from ~10/episode to ~1/episode
```

---

## Potential Concerns & Responses

### Concern 1: "Is this reward hacking?"

**Response**: No - it's modeling a biological capability (interoception). Humans have access to their internal state. If anything, *not* giving agents this information is the artificial constraint.

**Analogy**: Vision-based RL gives agents pixel observations. We don't call that "cheating" - it's modeling sensory input. Meter-modulated rewards model *proprioceptive* input.

### Concern 2: "Will this make learning harder?"

**Response**: No - it provides *more information*, not less. The reward signal is:
- Still continuous (dense feedback)
- Still differentiable (smooth gradient)
- Now has richer structure (timing information)

Learning should be **faster**, not harder.

### Concern 3: "What about other meters?"

**Response**: Start with energy × health (most critical for survival). Can extend later:

```python
# Phase 1 (current proposal):
reward = health * energy

# Phase 2 (future):
reward = health * energy * satiation  # Add hunger awareness

# Phase 3 (full interoception):
# Multiplicative may be too harsh (0.5 * 0.5 * 0.5 = 0.125)
# Consider weighted sum instead:
reward = 0.4*health + 0.4*energy + 0.2*satiation
```

Start simple, iterate based on results!

---

## Next Steps

1. **[TDD]** Implement interoception rewards with full test coverage
2. **[Experiment]** Run L0 minimal with both reward types side-by-side
3. **[Metrics]** Track:
   - Convergence speed (episodes to stable survival)
   - Resource efficiency (sleeps per episode, energy at sleep time)
   - Sample efficiency (total steps to convergence)
4. **[Documentation]** Create teaching materials comparing both approaches
5. **[Visualization]** Add TensorBoard plots showing:
   - Reward per step over time
   - Energy level at sleep actions
   - Money spent on sleeps

---

## Related Teachable Moments

- **`milestone_rewards_design.md`**: Original sparse reward problem
- **`reward_hacking_interact_spam.md`**: What happens with bad reward design
- **`temporal_credit_assignment.md`**: (future) Deep dive into credit assignment
- **`three_stages_of_learning.md`**: How this enables faster progression

---

## Conclusion

**Interoception-based rewards solve a real problem** (temporal credit assignment) **by modeling a real capability** (internal state awareness).

This is not arbitrary reward shaping - it's giving agents the same sensory feedback humans use to survive. The result is:
- Faster learning
- More efficient resource usage
- Better pedagogical value (shows importance of modeling biological reality)

**Status**: Ready for TDD implementation and experimental validation.

**Key Quote**: *"Humans don't need to die to learn that being tired is bad - we feel it immediately. Agents should too."*
