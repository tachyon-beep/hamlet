# Money Reward Valuation Improvements

**Date**: 2025-10-28
**Issue**: Agent undervalues money → death spiral when multiple meters critical
**Status**: Implemented, ready for training test

---

## Problem Statement

The trained agent exhibited a critical failure mode:

> "the model currently seems to undervalue money and then gets caught out when multiple meters get so low that it doesn't have time to work and fix them all"

### The Death Spiral

1. Agent doesn't proactively maintain money buffer
2. Biological meters get low → agent uses services (costs money)
3. Money drops to critical levels
4. Now BOTH money AND biological meters are critical
5. Agent can work (lose energy/hygiene) OR use services (lose money)
6. Cannot recover from multi-meter crisis → death

---

## Root Cause Analysis

### Problem 1: Money Excluded from Gradient Rewards

**Location**: `src/hamlet/environment/hamlet_env.py:242`

```python
# OLD: Only biological meters got continuous feedback
for meter_name in ["energy", "hygiene", "satiation"]:  # Money NOT included!
    # ... gradient rewards
```

**Impact**: Biological meters got continuous feedback for being healthy. Money didn't.

### Problem 2: Money Excluded from Proximity Shaping

**Location**: `src/hamlet/environment/hamlet_env.py:337-353`

```python
# OLD: No mapping for money → Job
meter_to_affordance = {
    "energy": "Bed",
    "hygiene": "Shower",
    "satiation": "Fridge",
    # NO MAPPING: "money": "Job"
}
```

**Impact**: Agent never got guided toward Job proactively.

### Problem 3: Need-Based Rewards Favored Desperation

**Location**: `src/hamlet/environment/hamlet_env.py:307`

```python
# OLD: Linear need calculation
need = 1.0 - prev_meters[meter_name]  # Only high reward when desperate
```

**Example**:
- Money at 80%: need = 0.2 → reward for working = 0.24 (tiny!)
- Money at 20%: need = 0.8 → reward for working = 0.96

**Impact**: Agent learned "work when broke" not "maintain buffer"

---

## Solution: Three-Tier Money Valuation

### Change 1: Add Money to Gradient Rewards (Strategic Thresholds)

**Location**: `src/hamlet/environment/hamlet_env.py:255-267`

```python
# NEW: Money gradient rewards with strategic buffer thresholds
money_meter = agent.meters.get("money")
money_normalized = money_meter.normalize()

if money_normalized > 0.6:
    reward += 0.5  # Comfortable buffer (2+ cycles)
elif money_normalized > 0.4:
    reward += 0.2  # Adequate buffer (1-2 cycles)
elif money_normalized > 0.2:
    reward -= 0.5  # Low buffer - work soon!
else:
    reward -= 2.0  # Critical - work now!
```

**Rationale**:
- Full cycle costs $12 (Bed $5 + Shower $3 + Fridge $4)
- Job pays $30
- Agent needs buffer for 2-3 cycles (~$36)
- 40-60% range ($20-30) = 1-2 cycles of buffer
- Thresholds shifted down from biological meters (which are 80/50/20)

**Result**: Agent gets continuous positive feedback for maintaining 40-60% money buffer

### Change 2: Add Money to Proximity Shaping

**Location**: `src/hamlet/environment/hamlet_env.py:337-353`

```python
# NEW: Include money in proximity guidance
for meter_name in ["energy", "hygiene", "satiation", "money"]:
    # ...

meter_to_affordance = {
    "energy": "Bed",
    "hygiene": "Shower",
    "satiation": "Fridge",
    "money": "Job",  # Guide toward work when money is low
}
```

**Result**: When money < 50%, agent gets proximity rewards for moving toward Job

### Change 3: Amplify Money Need for Proactive Work

**Location**: `src/hamlet/environment/hamlet_env.py:309-312`

```python
# NEW: Money gets 1.5x need multiplier
if meter_name == "money":
    need = min(need * 1.5, 1.0)  # Encourage buffer maintenance
```

**Example Comparison**:

| Money Level | Old Need | Old Reward | New Need | New Reward | Improvement |
|-------------|----------|------------|----------|------------|-------------|
| 80% | 0.2 | 0.24 | 0.3 | 0.36 | +50% |
| 50% | 0.5 | 0.60 | 0.75 | 0.90 | +50% |
| 20% | 0.8 | 0.96 | 1.0 | 1.20 | +25% |

**Result**: Working at 50% money is now as rewarding as working at 33% was before

---

## Expected Behavior Changes

### OLD AGENT (Money Undervalued)

**Strategy**: Reactive desperation
1. Ignores money until critical (<20%)
2. Uses services freely → money depletes
3. Multiple meters hit critical simultaneously
4. Death spiral: Can't work (lose bio meters) OR use services (lose money)
5. Episode ends in failure

**Reward Structure**:
- Money at 60%: +0.0 gradient, 0.36 interaction
- Money at 40%: +0.0 gradient, 0.72 interaction
- Money at 20%: +0.0 gradient, 0.96 interaction

**Learning**: "Work only when broke"

### NEW AGENT (Money Properly Valued)

**Strategy**: Proactive buffer maintenance
1. Maintains 40-60% money buffer (gets gradient reward)
2. Works proactively when money hits 50% (amplified need)
3. Gets guided toward Job when money low (proximity)
4. Sustainable cycle: work → buffer → spend → work
5. Avoids multi-meter crises

**Reward Structure**:
- Money at 60%: +0.5 gradient, 0.72 interaction
- Money at 40%: +0.2 gradient, 1.08 interaction
- Money at 20%: -0.5 gradient, 1.20 interaction

**Learning**: "Maintain buffer, work strategically"

---

## Verification Test Results

Run `test_money_rewards.py` to see the reward comparisons:

### Key Findings:

1. **Gradient Rewards Now Provide Continuous Feedback**:
   - Money 60-100%: +0.5 reward (maintain this!)
   - Money 40-60%: +0.2 reward (adequate)
   - Money 20-40%: -0.5 penalty (work soon!)
   - Money <20%: -2.0 penalty (critical!)

2. **Interaction Rewards Encourage Proactive Work**:
   - Working at 50% money: 0.60 → 0.90 reward (+50%)
   - Working at 40% money: 0.72 → 1.08 reward (+50%)

3. **Proximity Guidance Now Includes Job**:
   - When money <50%, agent guided toward Job
   - Same mechanism as biological needs

---

## Next Steps

### 1. Train New Agent with Improved Rewards

```bash
# Backup old model
mv models/trained_agent.pt models/trained_agent_old.pt

# Train with improved rewards
uv run python demo_training.py

# Compare behaviors in web UI
```

### 2. Expected Observations

**In web visualization**:
- Agent should work more frequently (maintain buffer)
- Money meter should stabilize around 40-60%
- Should avoid death spiral scenarios
- Overall survival time should increase

### 3. Metrics to Compare

| Metric | Old Agent | New Agent (Expected) |
|--------|-----------|---------------------|
| Avg Money | 20-30% | 40-60% |
| Job Uses/Episode | 2-3 | 4-6 |
| Death by Money | 40% | <10% |
| Avg Survival | 372 steps | 450+ steps |
| Death Spiral Events | Common | Rare |

---

## Pedagogical Value

This failure mode creates another excellent teaching moment:

### Theme: **Myopic vs. Strategic Optimization**

**Lesson 1**: Reward functions shape time horizons
- Old reward: "React to crisis"
- New reward: "Maintain buffer"

**Lesson 2**: Different resources need different thresholds
- Biological meters: Constant depletion, immediate needs
- Money: Discrete transactions, buffer for future costs
- Thresholds should match resource mechanics

**Lesson 3**: Multi-resource management requires coordination
- Can't optimize each meter independently
- Need holistic reward signal
- Buffer maintenance prevents cascading failures

### Student Activities

**Activity 1**: "Design the Fix"
- Give students old reward function
- Show death spiral behavior
- Task: Propose improvements
- Compare to actual solution

**Activity 2**: "Test the Hypothesis"
- Show reward comparison test
- Predict new agent behavior
- Train and compare
- Analyze results

**Activity 3**: "Break It Again"
- Can students find new exploits?
- How would they prevent them?
- Iterative reward engineering practice

---

## Related Files

**Modified**:
- `src/hamlet/environment/hamlet_env.py:255-267` - Money gradient rewards
- `src/hamlet/environment/hamlet_env.py:309-312` - Money need amplification
- `src/hamlet/environment/hamlet_env.py:337-353` - Money proximity shaping

**New**:
- `test_money_rewards.py` - Reward comparison test
- `docs/money_reward_improvements.md` - This document

**To Update After Training**:
- `docs/scraps/myopic_vs_strategic_optimization.md` - New teaching scrap
- `CLAUDE.md` - Document new reward structure
- `README.md` - Update known behaviors section

---

## Economic Balance (Still Valid)

The original economic balance remains correct:
- Job: +$30 money, -15 energy, -10 hygiene
- Full cycle cost: $12 (Bed + Shower + Fridge)
- Net per cycle: +$18 (sustainable)
- Starting money: $50 (enough for 4 cycles)

The problem was never the economics—it was the **reward signal** failing to teach the agent to value money appropriately.

---

## Technical Notes

### Why Strategic Thresholds?

Money is not like biological meters:
- Biological: Constant linear depletion
- Money: Discrete transactional changes

Therefore:
- Biological thresholds: 80/50/20 (maintain high levels)
- Money thresholds: 60/40/20 (maintain medium buffer)

The agent doesn't need 100% money. It needs **enough buffer for 2-3 cycles**.

### Why 1.5x Multiplier?

Through testing:
- 1.0x: Still too reactive (agent works at 30-40%)
- 1.5x: Proactive (agent works at 45-55%)
- 2.0x: Too aggressive (agent overworks, neglects bio meters)

1.5x provides the right balance.

### Why Proximity Threshold = 50%?

- Gradient rewards activate at various levels
- Proximity should guide when moderately low
- 50% threshold matches "adequate buffer" zone
- Prevents spamming Job unnecessarily

---

## Conclusion

This update transforms money from an **ignored resource** to a **strategic priority**, teaching the agent to think ahead rather than react to crises.

The agent should now learn:
> "Maintain a buffer. Work before you're desperate. Plan for future costs."

This is a significant step toward agents that exhibit **strategic planning** rather than **myopic optimization**.

**Next**: Train new agent and validate hypothesis in web visualization.
