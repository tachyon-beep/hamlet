# Discovered Insights - Testing Findings & Traps

**Created:** October 31, 2025  
**Purpose:** Document unexpected behaviors, design traps, and bugs discovered during systematic test implementation  
**Status:** ACTIVE - Updated as tests are written and run

---

## 🎯 Testing Philosophy

"Don't assume the code is right. We've had trouble getting the agent to converge, so there are likely insidious bugs. Test everything, question everything."

---

## ⚡ Energy System Insights

### INSIGHT #1: Dual Energy Depletion (NOT A BUG - CORRECT DESIGN)

**Date:** October 31, 2025  
**Discovered While:** Writing base depletion tests  
**Severity:** CRITICAL UNDERSTANDING

#### The Discovery

Energy depletes at **1.0% per movement step**, not the expected 0.5%. Initial assumption was this was a bug.

#### The Reality

**This is CORRECT design!** Energy has TWO sources of depletion:

1. **Movement Cost** (0.5%): Charged when actions 0-3 (UP/DOWN/LEFT/RIGHT) are executed
   - Applied in `_execute_actions()` line 463
   - `self.meters[movement_mask] -= movement_costs`

2. **Passive Depletion** (0.5%): Time passing regardless of action
   - Applied in `_deplete_meters()` line 779
   - Called every step from `step()` line 379

**Total per movement step: 0.5% + 0.5% = 1.0%**

#### Why This Matters

- Agent reaches 0 energy in **100 steps** with continuous movement
- Agent reaches 0 energy in **200 steps** with no movement (INTERACT on empty tiles)
- This dual cost is INTENTIONAL: movement is expensive + time passes
- Tests must account for both costs or use INTERACT action to isolate passive depletion

#### Code Locations

```python
# Movement costs (vectorized_env.py:448-463)
movement_costs = torch.tensor([
    0.005,  # energy: -0.5%
    0.003,  # hygiene: -0.3%
    0.004,  # satiation: -0.4%
    ...
])
self.meters[movement_mask] -= movement_costs.unsqueeze(0)

# Passive depletion (vectorized_env.py:770-798)
depletions = torch.tensor([
    0.005,  # energy: 0.5% per step
    0.003,  # hygiene: 0.3%
    0.004,  # satiation: 0.4%
    ...
])
self.meters = torch.clamp(self.meters - depletions, 0.0, 1.0)
```

#### Testing Strategy

**To test PASSIVE depletion only:**

```python
# Use INTERACT (action 4) on empty tiles - no movement cost
env.positions[0] = torch.tensor([4, 4])  # Empty tile
env.step(torch.tensor([4]))  # INTERACT - fails but time passes
```

**To test COMBINED movement + passive:**

```python
# Use movement actions normally
env.step(torch.tensor([0]))  # UP - incurs both costs
```

#### Pedagogical Value

This is a GREAT teaching moment about:

- Reading code carefully before assuming bugs
- Understanding the difference between action costs and time costs
- Why agents that move constantly die faster than stationary agents
- The importance of distinguishing "what action did I take" from "time passing"

---

## 🔬 Methodology Notes

### Test Design Patterns

#### Pattern 1: Isolating Passive Depletion

When testing base depletion rates without action costs, use INTERACT on empty tiles:

```python
env.positions[0] = torch.tensor([4, 4])  # Move to empty tile
for _ in range(100):
    env.step(torch.tensor([4]))  # INTERACT - no movement cost
```

#### Pattern 2: Testing Combined Costs

When testing real gameplay scenarios, use actual movement:

```python
# Agent at boundary (y=0) trying to move UP
# UP action fails (boundary) but still charges costs
for _ in range(100):
    env.step(torch.tensor([0]))  # Movement cost + passive depletion
```

#### Pattern 3: Debugging Depletion Step-by-Step

Track meter changes per step to understand cascading effects:

```python
for step in range(10):
    before = env.meters[0, 0].item()
    env.step(action)
    after = env.meters[0, 0].item()
    print(f"Step {step}: {before*100:.2f}% → {after*100:.2f}% "
          f"(loss: {(before-after)*100:.3f}%)")
```

---

## 🚨 Bug Suspects (To Be Investigated)

### SUSPECT #1: Cascading Penalties Too Aggressive?

**Status:** NOT YET TESTED  
**Location:** `_apply_secondary_to_primary_effects()` lines 820-860

#### The Concern

Low satiation triggers BOTH health AND energy penalties:

- Health penalty: `0.004 * deficit` per step
- Energy penalty: `0.005 * deficit` per step

**Combined with mood → energy penalty**, agent may have:

- Base energy depletion: 0.5%
- Movement cost: 0.5%
- Low satiation penalty: ~0.167% (at 20% satiation)
- Low mood penalty: ~0.167% (at 20% mood)
- **TOTAL: ~1.33% per step**

This means energy reaches 0 in **~75 steps** instead of 100-200.

#### Why This Matters

If penalties are too aggressive, agent may not have time to:

1. Explore to find affordances (needs ~20-30 steps)
2. Reach affordances (needs ~10-20 steps)
3. Complete interactions (needs 4-5 ticks)

**Total needed: ~50-80 steps minimum**

If death spiral starts at 75 steps, game may be unwinnable from low satiation states.

#### Test Priority

**HIGH** - This directly impacts convergence difficulty.

---

### SUSPECT #2: Fitness Death Spiral

**Status:** NOT YET TESTED  
**Location:** `_deplete_meters()` lines 800-815

#### The Concern

Low fitness (<30%) causes **3x health depletion multiplier**:

- Normal: 0.1% per step
- Low fitness: 0.3% per step

Combined with:

- Fitness itself decaying at 0.2% per step
- Low satiation also damaging health
- No way to rapidly recover fitness (Gym gives +30% but costs energy)

This creates potential unrecoverable death spiral.

#### Calculation

Starting from 30% fitness (threshold):

- Fitness decays to 20% in 50 steps
- Health depletes at 0.3% per step = 15% in 50 steps
- If low satiation also active: additional 0.133% per step = 6.65% in 50 steps
- **Total health loss: ~22% in 50 steps**

Agent starting with 50% health would die in ~115 steps.

#### Test Priority

**HIGH** - Another convergence blocker.

---

### SUSPECT #3: Action Masking Not Working?

**Status:** NOT YET TESTED  
**Location:** `get_action_masks()` lines 300-350

#### The Concern

If masked actions are still being selected during exploration, agent wastes steps on invalid actions:

- Trying to move through walls
- Trying to interact without money
- Trying to use closed affordances

This would manifest as:

- Random exploration taking longer than expected
- Agent "stuck" at boundaries
- Poor sample efficiency

#### Test Priority

**MEDIUM** - Affects learning speed, not fundamental winnability.

---

### SUSPECT #4: LSTM Not Using History?

**Status:** NOT YET TESTED  
**Location:** `agent/networks.py` RecurrentSpatialQNetwork

#### The Concern

If LSTM hidden state is not carrying information across steps:

- Network behaves like MLP despite LSTM architecture
- Partial observability provides no benefit over full observability
- Agent can't remember locations of affordances outside view

This would explain poor POMDP performance.

#### Test Priority

**MEDIUM** - Only affects Level 2+ (POMDP), not current Level 1.5.

---

### SUSPECT #5: Reward Signal Too Weak?

**Status:** NOT YET TESTED  
**Location:** `_calculate_shaped_rewards()` lines 950-1000

#### The Concern

Death penalty (-100) dominates all positive rewards:

- Decade milestone: +0.5 (need 200 decades to offset one death)
- Century milestone: +5.0 (need 20 centuries to offset one death)

If agent dies frequently during early training:

- Average reward is dominated by -100
- Positive signals are drowned out
- Gradient updates push toward "avoid death" but don't guide toward "use affordances"

#### Calculation

Agent surviving 50 steps:

- Gets: 5 decade bonuses = +2.5
- Dies: -100
- **Net: -97.5**

Agent surviving 150 steps:

- Gets: 15 decade bonuses + 1 century = +12.5
- Dies: -100
- **Net: -87.5**

Only 5% improvement despite 3x longer survival!

#### Test Priority

**MEDIUM-HIGH** - Could explain why agent improves but doesn't converge.

---

## 📊 Confirmed Behaviors (Not Bugs)

### CONFIRMED #1: Money Doesn't Deplete Passively

**Status:** CONFIRMED CORRECT  
**Date:** Code review October 31, 2025

Money is a resource, not a need. It only changes via:

- Gaining: Job (+$22.50), Labor (+$30)
- Spending: Using affordances (varies by affordance)

This is correct design - money is a constraint, not a meter.

---

### CONFIRMED #2: Secondary Meters Don't Cause Direct Death

**Status:** CONFIRMED CORRECT  
**Date:** Code review October 31, 2025

Only PRIMARY meters (health, energy) cause death when reaching 0.

SECONDARY/TERTIARY meters at 0 don't kill directly, but cause death via cascading to primaries:

- Low satiation → damages health & energy
- Low fitness → accelerates health depletion
- Low mood → damages energy
- Low hygiene → weak effects on everything
- Low social → damages mood (which damages energy)

This creates the desired coupled differential equation behavior.

---

## 🎓 Pedagogical Insights

### Teaching Moment #1: Specification Gaming

The disabled proximity reward system (lines 1150-1230) is a PERFECT example of specification gaming:

- Original intent: Reward agent for approaching affordances
- Actual behavior: Agent learned to stand near affordances without using them
- Lesson: "Be careful what you reward - you'll get exactly that"

**Keep this in codebase as commented example for students!**

---

### Teaching Moment #2: Coupled Differential Equations

The cascading meter system demonstrates:

- PRIMARY depends on SECONDARY
- SECONDARY depends on TERTIARY
- Changes propagate through the system
- Small initial conditions (low satiation) → large outcomes (death)

This is a simplified version of real physiological systems.

---

### Teaching Moment #3: Exploration-Exploitation Tradeoff

The dual energy cost creates a natural exploration-exploitation tension:

- Moving to explore: expensive (1.0% per step)
- Standing still: cheaper (0.5% per step)
- Must balance searching vs. conserving

Agent must learn: "explore early, exploit once you know where things are"

---

## 🎨 Design Insights (For Future Refactoring)

### DESIGN INSIGHT #1: Hard Threshold Creates Cliff Effect

**Date:** October 31, 2025  
**Status:** DESIGN FLAW - To be fixed in refactor  
**Severity:** MEDIUM - Affects gameplay feel and learning curve

#### The Problem

Current cascade penalties use a **hard 30% threshold**:

```python
threshold = 0.3  # below this, aggressive penalties apply
low_satiation = satiation < threshold
if low_satiation.any():
    deficit = (threshold - satiation[low_satiation]) / threshold
    energy_penalty = 0.005 * deficit
```

**Behavior:**

- Satiation 31%: No cascade penalty ✓
- Satiation 29%: CASCADE ACTIVATED ⚡ penalty = 0.167% per step
- Satiation 0%: MAXIMUM CASCADE ⚡⚡⚡ penalty = 0.500% per step

**This creates a "cliff effect"** where dropping from 31% → 29% causes sudden dramatic change.

#### Why This Is Bad

1. **Unrealistic**: Real hunger doesn't suddenly "turn on" at 30%
2. **Harsh Learning Signal**: Agent gets no warning gradient as satiation decreases
3. **All-or-Nothing**: Either you're fine (>30%) or you're in trouble (<30%)
4. **Poor Exploration Incentive**: No reason to maintain meters above 30% until crisis

#### Better Design (For Refactor)

**Smooth gradient from 100% → 0%:**

```python
# Proposed: Linear scaling from 100% (no penalty) to 0% (max penalty)
satiation_factor = satiation  # 1.0 = healthy, 0.0 = critical
penalty_strength = 1.0 - satiation_factor  # 0.0 = healthy, 1.0 = critical

# Very gradual penalties starting immediately
energy_penalty = 0.005 * penalty_strength

# Examples:
# satiation=100%: penalty_strength=0.0, penalty=0.000% (none)
# satiation=80%:  penalty_strength=0.2, penalty=0.100% (tiny)
# satiation=50%:  penalty_strength=0.5, penalty=0.250% (moderate)
# satiation=20%:  penalty_strength=0.8, penalty=0.400% (severe)
# satiation=0%:   penalty_strength=1.0, penalty=0.500% (maximum)
```

**Benefits:**

1. **Smooth gradient**: Agent feels effects gradually
2. **Early warning**: Small penalties at 80% encourage maintenance
3. **Realistic**: Mimics real physiological decline
4. **Better learning**: Clear gradient for RL to follow
5. **No cliffs**: No sudden activation threshold

#### Alternative: Exponential Curve

Could also use exponential for more dramatic low-meter effects:

```python
# Exponential: penalties accelerate as meter drops
penalty_strength = (1.0 - satiation) ** 2  # Squared for acceleration

# Examples:
# satiation=80%: penalty_strength=0.04, penalty=0.020% (barely noticeable)
# satiation=50%: penalty_strength=0.25, penalty=0.125% (mild)
# satiation=20%: penalty_strength=0.64, penalty=0.320% (urgent!)
# satiation=0%:  penalty_strength=1.00, penalty=0.500% (critical!)
```

#### When to Fix

**NOT NOW** - This is a larger design change affecting:

- All cascade calculations (satiation, mood, hygiene, social)
- Balance tuning (max penalty values may need adjustment)
- Test expectations (all cascade tests)
- Agent training (different learning dynamics)

**During refactor:**

1. Extract `MeterDynamics` class
2. Implement both threshold and gradient strategies
3. Make it configurable (A/B test different curves)
4. Retrain and compare convergence rates

#### Pedagogical Value

**Keep both implementations!** Great teaching moment:

- Hard thresholds vs smooth gradients
- How reward shaping affects learning
- "Cliff effects" in game design
- Why biological systems use gradients, not switches

---

## 🔧 Testing Utilities

### Debug Script Template

```python
import torch
from src.townlet.environment.vectorized_env import VectorizedHamletEnv

env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
env.reset()

# Set specific meter states
env.meters[0, 2] = 0.2  # Low satiation

# Track evolution
for step in range(50):
    print(f"Step {step}: Energy={env.meters[0,0].item()*100:.1f}%, "
          f"Satiation={env.meters[0,2].item()*100:.1f}%")
    env.step(torch.tensor([4]))  # INTERACT on empty tile
    if env.dones[0]:
        print(f"DIED at step {step}")
        break
```

---

## 📝 Notes for Documentation Phase

When documenting this codebase later:

1. **Energy System**: Explain dual depletion clearly in docstrings
2. **Cascading Effects**: Diagram the PRIMARY → SECONDARY → TERTIARY hierarchy
3. **Action Costs**: Clarify movement costs vs passive depletion
4. **Reward Hacking**: Keep commented-out proximity rewards as teaching example
5. **Death Spirals**: Document intended difficulty curve (thresholds, multipliers)
6. **Testing Patterns**: Include INTERACT-on-empty-tile pattern in examples

---

## � CRITICAL BUGS FOUND

### BUG #1: Social and Fitness Start at 50%, Not 100%

**Date:** October 31, 2025  
**Discovered While:** Running base depletion tests  
**Severity:** CRITICAL - Requirements→Implementation Flaw  
**Status:** CONFIRMED BUG

#### The Discovery

Two tests failed with unexpected depletion rates:

- **Mood depleted 26.9% instead of 10%** (2.7x expected rate)
- **Social depleted 50% instead of 60%** (started at 50%, not 100%)

Debug investigation revealed:

```python
# vectorized_env.py line 128
self.meters = torch.tensor([
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5]  # Social=0.5, Fitness=0.5 ❌
    # ^energy ^hygiene ^satiation ^money ^mood ^SOCIAL ^health ^FITNESS
], device=self.device).repeat(self.num_agents, 1)
```

**Social and Fitness start at 50%, not 100%!**

#### The Impact

Starting at 50% social means:

1. **Agent immediately in cascade zone** - Social crosses 30% threshold in just 33 steps
2. **Mood starts cascading early** - Low social (below 30%) damages mood at 0.004 * deficit per step
3. **Tests fail** - Expected 60% depletion but got 50% because started at 50%
4. **Agent handicapped from birth** - Fighting uphill battle from step 1

#### Why This Is Wrong

**Requirements:** All meters should start at 100% (healthy agent)  
**Implementation:** Social and Fitness start at 50% (struggling agent)

This creates:

- Unrealistic difficulty curve (agent born sick)
- Cascading penalties activate immediately
- Mood depletes 2.7x faster due to low social cascade
- Agent has ~83 steps before social reaches 0 (not the intended ~166 steps)

#### The Math

**Expected behavior (starting at 100%):**

- Social depletes 0.6% per step
- Reaches 30% threshold in 116 steps
- Reaches 0% in 166 steps
- Mood only gets base 0.1% depletion until step 116

**Actual behavior (starting at 50%):**

- Social depletes 0.6% per step  
- Reaches 30% threshold in 33 steps ❌
- Reaches 0% in 83 steps ❌
- Mood gets cascading penalty starting at step 33 ❌

#### Cascading Penalty Calculation

After social drops below 30% (at step 33):

- Social = 29.6% → deficit = (30 - 29.6) / 30 = 0.013
- Mood penalty = 0.004 * 0.013 = 0.000053 per step (TINY at first)

After social reaches 0% (at step 83):

- Social = 0% → deficit = (30 - 0) / 30 = 1.0
- Mood penalty = 0.004 * 1.0 = 0.4% per step (MASSIVE!)
- Base mood depletion = 0.1% per step
- **Total mood loss = 0.5% per step** (5x the base rate!)

Over 100 steps with social starting at 50%:

- Steps 1-33: Mood loses 0.1% per step = 3.3% total
- Steps 34-83: Mood loses ~0.1-0.4% per step (escalating) ≈ 15% total
- Steps 84-100: Mood loses 0.5% per step = 8.5% total
- **Total mood loss ≈ 26.8%** ✓ (matches test failure!)

#### Code Location

`src/townlet/environment/vectorized_env.py` line 128

```python
# BUG: Should be [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
self.meters = torch.tensor([
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5]  # ❌ Social and Fitness at 50%
], device=self.device).repeat(self.num_agents, 1)
```

#### Fix Required

```python
# FIXED: All meters start at 100%
self.meters = torch.tensor([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # ✅ All at 100%
], device=self.device).repeat(self.num_agents, 1)

# Or with semantic clarity:
self.meters = torch.ones((self.num_agents, 8), device=self.device)
# Set money to $50 (0.5) as intended starting amount
self.meters[:, 3] = 0.5  # Money index = 3
```

#### Why This Bug Matters for Convergence

**This could be why agents aren't converging!**

Starting with social=50% means:

1. Agent has only ~83 steps to learn basic survival
2. Mood cascading starts at step 33 (very early)
3. By step 100, mood is at 73% and dropping fast
4. Agent never experiences "healthy state" to learn optimal policy
5. Exploration is rushed - must find affordances before cascades kill them

**Hypothesis:** If we fix this, agents will have 2x more time to explore and learn before cascading becomes critical.

#### Test Adjustments

The tests were CORRECT to fail! They caught this bug. After fixing:

- Social should deplete 60% over 100 steps ✓
- Mood should deplete 10% over 100 steps (no cascade) ✓
- Fitness starting at 100% will also affect health depletion tests ✓

#### Pedagogical Value

**Keep this bug in teaching materials!** Perfect example of:

- How tests catch requirements→implementation flaws
- Cascade amplification effects
- "Born sick" vs "born healthy" game balance
- Why initial conditions matter in differential equations

#### Meta-Lesson: Debugging with Incorrect Information

**CRITICAL INSIGHT:** The UI was showing 100% for all bars at start, but the data was 50%.

This created an **insidious debugging trap**:

1. UI showed "everything is 100%" ✓
2. We assumed the UI was correct ✓
3. We debugged based on that false assumption ✗
4. Only systematic testing caught the discrepancy ✓

**Guiding Principle for Future Testing:**

> **If something seems inconsistent or doesn't "feel" right (≥30% confidence), PAUSE and ask:**
>
> - "Should this be consistent with something else?"
> - "Does this match the stated requirements?"
> - "Am I debugging based on UI/logs that might be wrong?"

**Why This Matters:**

- UI bugs can mask data bugs
- Visual confirmation ≠ data verification
- Test the data, not the display
- Trust tests over intuition, but trust intuition enough to write tests

**In this case:**

- Test: "Social should deplete 60% over 100 steps"
- Actual: "Social depleted 50%"
- Initial reaction: "Test is wrong, social started at 100%"
- Correct reaction: "Wait, did social actually start at 100%?"
- **Investigation proved: Social started at 50% despite UI showing 100%**

---

### BUG #2: Fitness → Health Uses Hard Thresholds (Inconsistent Pattern)

**Date:** October 31, 2025  
**Discovered While:** Auditing cascade consistency  
**Severity:** MEDIUM - Design inconsistency  
**Status:** ✅ FIXED - Now uses smooth gradient (0.5x to 3.0x)

#### The Problem

**All cascades use gradient calculation based on current level:**

```python
# Pattern used by satiation, mood, hygiene, social
threshold = 0.3
low_meter = meter < threshold
deficit = (threshold - meter[low_meter]) / threshold  # Gradient!
penalty = base_rate * deficit
```

**Except fitness → health, which uses hard thresholds:**

```python
# Fitness-modulated health depletion (lines 792-807)
fitness = self.meters[:, 7]
health_depletion = torch.where(
    fitness < 0.3,
    0.003,  # Low fitness: 3x depletion (HARD VALUE!)
    torch.where(
        fitness > 0.7,
        0.0005,  # High fitness: 0.5x depletion (HARD VALUE!)
        0.001  # Medium fitness: baseline (HARD VALUE!)
    )
)
```

**This creates stepped behavior:**

- fitness=31%: 0.001/step (baseline)
- fitness=29%: 0.003/step (3x jump!)
- fitness=71%: 0.001/step (baseline)
- fitness=69%: 0.0005/step (0.5x drop!)

#### Why This Is Wrong

**Principle:** All meters should work the same way!

- User expectation: Consistent gradient behavior
- Code maintainability: Same pattern everywhere
- Agent learning: Smooth gradients, no cliffs
- Testing: Predictable cascade calculations

#### The Fix

**Change to gradient-based calculation:**

```python
# Fitness-modulated health depletion (gradient approach)
fitness = self.meters[:, 7]
baseline_health_depletion = 0.001

# Calculate multiplier based on current fitness
# fitness=100%: multiplier=0.5x (very healthy)
# fitness=50%: multiplier=1.0x (normal)
# fitness=0%: multiplier=3.0x (very unhealthy)
fitness_penalty_strength = 1.0 - fitness  # 0.0 to 1.0
multiplier = 0.5 + (2.5 * fitness_penalty_strength)  # 0.5 to 3.0

health_depletion = baseline_health_depletion * multiplier
self.meters[:, 6] = torch.clamp(
    self.meters[:, 6] - health_depletion, 0.0, 1.0
)
```

**Examples with gradient:**

- fitness=100%: multiplier=0.5, depletion=0.0005/step (same as old "high")
- fitness=70%: multiplier=1.25, depletion=0.00125/step (smooth transition)
- fitness=50%: multiplier=1.75, depletion=0.00175/step (smooth transition)
- fitness=30%: multiplier=2.25, depletion=0.00225/step (smooth transition)
- fitness=0%: multiplier=3.0, depletion=0.003/step (same as old "low")

#### When to Fix

**NOW** - This should be fixed immediately because:

1. It's inconsistent with all other cascades
2. Simple change (20 lines)
3. Doesn't require rebalancing (maintains same endpoints)
4. Tests need updating anyway
5. Part of "make them all work the same" principle

#### Impact Assessment

**Low risk:**

- Endpoints unchanged (0.0005 at 100%, 0.003 at 0%)
- Only changes intermediate values (30-70% range)
- Makes behavior more predictable
- Aligns with all other cascade patterns

---

## 🚨 BALANCE ISSUES (Not Bugs - Design Concerns)

### BALANCE ISSUE #1: Combined Low Meters Create Unrecoverable Death Spiral

**Date:** October 31, 2025  
**Discovered While:** Running combined cascade tests  
**Severity:** HIGH - Affects learnability  
**Status:** DOCUMENTED - Needs game balance evaluation

#### The Discovery

**CRITICAL UPDATE:** The 81.5% loss was caused by **movement costs compounding with cascades**, NOT multiplicative cascade math!

**Test Results: Low Satiation (20%) + Low Mood (20%) → Energy**

- **WITHOUT MOVEMENT** (INTERACT action): 52.3% loss over 50 steps (1.05x amplification - ACCEPTABLE)
- **WITH MOVEMENT** (UP/DOWN/LEFT/RIGHT): 70-85% loss over 50 steps (1.4-1.7x amplification)
- Base cascades are nearly additive: Expected 49.6%, got 52.3% (only 1.05x compound effect)
- Movement adds 25% energy cost (0.5% per step × 50 steps)
- **Movement + Cascades = Multiplicative Death Spiral**

#### Why This Matters

**CORRECTED: Cascades themselves are nearly additive (1.05x) - the problem is MOVEMENT!**

**Energy drain breakdown (50 steps):**

1. Base energy depletion: 25.0% (0.5% per step)
2. Low satiation cascade: ~16.7% (escalating from 0.167% to 0.5% per step)
3. Low mood cascade: ~7.9% (semi-stable at ~0.158% per step)
4. **Total passive + cascades: 52.3% (1.05% per step) - Agent survives!**
5. **Movement cost if exploring: +25.0% (0.5% per step) - Agent dies!**

**With random exploration (moving every step):**

- Total: 52.3% + 25% = **77.3% loss over 50 steps**
- Actual observed: 70-85% depending on meter levels
- Depletion rate: 1.4-1.7% per step
- **Death in 60-80 steps = barely enough time to learn**

**This creates tight time pressure:**

- Agent needs ~20-30 steps to explore and find affordances
- Agent needs ~10-20 steps to reach affordances  
- Agent needs 4-5 ticks to complete interactions
- **Minimum survival time needed: 50-80 steps**
- **Actual survival with combined low meters + movement: 60-80 steps**

**Agent has almost no margin for error - must learn efficiently on first try!**

#### Maximum Expected Life Calculations

**Best Case (All Meters 100%, No Affordances, No Movement):**

- Limiting factor: Social (depletes fastest at 0.6% per step)
- Social reaches 30% threshold: 116 steps
- Social reaches 0%: 166 steps
- Once social hits 0%, mood cascades at 0.4% per step
- Mood reaches 30%: Additional 175 steps (mood starts 100%)
- Mood reaches 0% with cascade: ~140 steps
- Once mood hits 30%, energy cascades at 0.5% per step
- **Maximum theoretical life (passive only): ~400-450 steps**

**Best Case with Movement (No Affordances):**

- Energy depletes at 1.0% per step (0.5% base + 0.5% movement)
- BUT social cascade → mood → energy kicks in at step 116
- Energy with cascades: ~1.5% per step after social drops
- **Maximum with continuous movement: ~200-250 steps**

**Worst Case (Multiple Meters Low):**

- Starting with satiation=20%, mood=20%
- Energy depletes at 1.5-2.0% per step
- **Death in 50-70 steps**

#### Impact on Reward Function

**Current milestone rewards:**

- Decade (10 steps): +0.5
- Century (100 steps): +5.0
- Death penalty: -100

**Problem:**

- Agent surviving 60 steps: 6 decades = +3.0, death = -100, **net = -97.0**
- Agent surviving 120 steps: 12 decades + 1 century = +11.0, death = -100, **net = -89.0**
- Only 8% improvement despite 2x survival time!

**Proposed Fix (After Refactor):**

```python
# Reward every step beyond minimum expected survival
MINIMUM_EXPECTED_LIFE = 50  # Steps needed to find and use ONE affordance
survival_bonus = max(0, steps_survived - MINIMUM_EXPECTED_LIFE) * 0.1
# Agent surviving 120 steps: 70 × 0.1 = +7.0 bonus
# Makes gradient steeper: longer survival = significantly more reward
```

#### Recommendations

**Option A: Reduce Cascade Penalties (Easier Difficulty)**

- Change base penalty rates from 0.004-0.005 to 0.002-0.003
- Gives agents more time to learn (120+ steps before death)
- Reduces "born sick" feeling

**Option B: Remove 30% Threshold Gates (Smoother Difficulty)**

- Apply penalties from 100% → 0% (smooth gradient)
- Small penalties at 80% (early warning)
- Full penalties at 0% (critical)
- More forgiving learning curve

**Option C: Add WAIT Action (Tool for Agents) - MOST CRITICAL**

- Separate from INTERACT (which requires affordance)
- Costs no energy, only base passive depletion
- Allows agents to "rest" and plan
- Teaches resource management
- **CRITICAL DESIGN FLAW DISCOVERED**: Currently INTERACT is masked unless on affordance, forcing agents to move every step
- This causes oscillation near affordances (agent wants to wait but can't)
- Adds 25% energy drain (0.5% per step × 50 steps) on top of cascades
- Prevents "stand still and recover" strategy

**Option D: Adjust Reward Function (Better Signal)**

- Reward steps beyond minimum survival threshold
- De-emphasize death penalty (reduce from -100 to -50)
- Increase milestone bonuses for 150+ step survival
- Create clearer gradient for learning

#### Testing Priority

- [x] Document combined cascade behavior
- [ ] Measure actual survival times with random exploration
- [ ] Test if agents can learn to survive >100 steps consistently
- [ ] If not → implement Option A or C (balance changes)

---

### ~~BUG #2~~ RESOLVED: Cascade Calculation Produces 36% More Damage Than Expected

**Date:** October 31, 2025  
**Discovered While:** Running cascading meter tests  
**Severity:** ~~HIGH~~ FALSE ALARM - My math was wrong  
**Status:** RESOLVED - Environment is correct

#### The Discovery

Test `test_low_satiation_damages_energy` shows energy loss of **41.8%** over 50 steps with satiation starting at 20%.

**Expected mathematically:**

- Passive depletion: 0.5% × 50 = 25.0%
- Cascade penalties: ~5.6% (escalating from 0.167% to 0.500% per step)
- **Total: 30.6%**

**Actual from environment:**

- **Total: 41.8%**
- **Discrepancy: 11.2% (36% MORE damage than expected!)**

#### The Investigation

**Desktop analysis of cascade code (lines 820-860):**

```python
threshold = 0.3
satiation = self.meters[:, 2]
low_satiation = satiation < threshold
if low_satiation.any():
    deficit = (threshold - satiation[low_satiation]) / threshold
    energy_penalty = 0.005 * deficit
    self.meters[low_satiation, 0] = torch.clamp(
        self.meters[low_satiation, 0] - energy_penalty, 0.0, 1.0
    )
```

**Step-by-step trace:**

- Step 0: satiation=20% → deficit=0.333 → penalty=0.167%
- Step 10: satiation=16% → deficit=0.467 → penalty=0.233%
- Step 20: satiation=12% → deficit=0.600 → penalty=0.300%
- Step 30: satiation=8% → deficit=0.733 → penalty=0.367%
- Step 40: satiation=4% → deficit=0.867 → penalty=0.433%
- Step 50: satiation=0% → deficit=1.000 → penalty=0.500%

**This should accumulate to ~16.7% cascade damage** (average 0.334% × 50 steps)

**But we're seeing 41.8% - 25.0% = 16.8% cascade damage** ✓

Wait... that actually MATCHES! Let me recalculate...

#### The Resolution

**My initial mathematical model was WRONG!**

I calculated cascade damage as ~5.1%, but I made an error. The correct calculation:

- Average cascade penalty = (0.167% + 0.500%) / 2 = 0.334% per step
- Total cascade over 50 steps = 0.334% × 50 = 16.7%
- Total loss = 25.0% (passive) + 16.7% (cascade) = **41.7%** ✓

**The environment is CORRECT!** The 41.8% actual vs 41.7% expected is just rounding error.

#### Why This Confused Me

My `analyze_cascade_math.py` script had a bug in how I calculated the accumulating deficit. The script showed 30.6% expected, but the actual mathematical model (when done correctly) shows 41.7%.

**Lesson:** When debugging, verify your test expectations are correct! The environment was right all along.

#### Impact on Testing

The cascade calculations are **MATHEMATICALLY CORRECT**. The test expectation of <50% energy loss was appropriate (actual 41.8% passes).

However, this reveals the cascades ARE very aggressive:

- 16.7% additional damage from low satiation
- That's 67% MORE damage than passive alone (16.7 / 25.0 = 0.67)
- Agent energy depleting at 0.84% per step average (vs 0.5% baseline)

This may still be too punishing for learning, but it's **not a bug** - it's working as designed.

#### Next Steps

- [ ] Update test assertions based on correct math
- [ ] Evaluate if cascade penalties are too harsh for gameplay (design decision)
- [ ] Consider implementing smooth gradient penalties (DESIGN INSIGHT #1)

---

## 🔍 Investigation Queue

As tests are written and run, add findings here:

- [x] **BUG #1 FIXED** - Changed social/fitness initialization to 1.0
- [x] **BUG #2 FIXED** - Changed fitness → health to use smooth gradient (consistency!)
- [x] Re-run all base depletion tests after fix (ALL PASSING ✓)
- [x] Validate cascading penalty math (CORRECT - my initial math was wrong)
- [x] Confirm all cascades use same pattern (FITNESS was inconsistent, NOW FIXED)
- [x] **FITNESS TESTS PASSING** - Gradient calculation validated (3/3 tests ✓)
  - Smooth 0.5x to 3.0x multiplier confirmed
  - Death spiral documented (agents die ~250 steps with 50% starting fitness)
  - Escalating depletion rate verified (9.4% → 29.3% per 50 steps)
- [x] **BALANCE ISSUE #1 RESOLVED** - Movement costs compound with cascades!
  - Original: 81.5% energy loss seemed like multiplicative cascade bug
  - Actual: 52.3% from cascades (1.05x - nearly additive!) + 25% from movement
  - Cascades themselves are well-behaved, problem is forced movement
  - Maximum expected life: 200-450 steps depending on strategy
  - Minimum survival needed: 50-80 steps (find + reach + use affordance)
  - **Margin is too thin for learning!**
- [x] **ACTION #8 ADDED** - WAIT action to give agents recovery mechanism
- [x] **CRITICAL BEHAVIORAL INSIGHT** - INTERACT masking creates multiple strategic failures
  - INTERACT is masked unless agent is on affordance
  - This FORCES agents to move every step (can't wait)
  - **Problem #1: Oscillation** - Causes observable oscillation near affordances (wants to wait, can't)
  - **Problem #2: Energy waste** - Adds unnecessary 25% energy drain from forced movement
  - **Problem #3: Recovery prevention** - Prevents "stand still and recover" strategy from emerging
  - **Problem #4: Timing optimization impossible** - Can't wait to maximize affordance benefit
    - Example: Bed restores 75% energy linearly + 12.5% completion bonus
    - Optimal: Wait until energy=12.5%, get full 87.5% benefit
    - Current: Must keep moving, wastes restoration on already-high meters
    - Burns 0.5% energy per step while waiting for optimal timing
    - This punishes strategic play and rewards random immediate usage
  - **WAIT action is HIGH PRIORITY** - Enables strategic timing and resource optimization
- [ ] Fix health cascade test (use INTERACT not UP)
- [ ] Fix combined cascade test (update expectations for escalating penalties)
- [ ] Measure actual survival times with random exploration
- [ ] Test if agents can learn >100 step survival with current balance
- [ ] Run death condition tests
- [x] **NETWORK ARCHITECTURE TESTING STARTED** (2025-10-31)
  - Created 19 comprehensive tests for neural networks
  - 12/19 passing on first run (63%)
  - Revealed fundamental architecture issues:
    - Inconsistent observation handling between network types
    - Manual hidden state management (error-prone)
    - Hardcoded observation parsing (not extensible)
    - No abstraction layer between network architectures
  - **ACTION #9 CREATED**: Root-and-branch network redesign (HIGH priority)
  - Need unified interfaces, observation abstraction, automatic state management

---

## 📋 Next Steps

**For Refactoring Actions:** See [`REFACTORING_ACTIONS.md`](./REFACTORING_ACTIONS.md)

Major refactoring items identified:

1. **🔴 HIGH PRIORITY:** Extract configurable "Bar Management Engine" - move cascade logic to config
2. **🟡 MEDIUM:** Extract RewardStrategy, MeterDynamics, ObservationBuilder classes
3. **🟢 LOW:** Target network DQN, GPU optimizations, sequential replay buffer

**DO NOT START** refactoring until test coverage reaches 70%+ on affected modules.

- [ ] Check if action masking prevents invalid Q-value selection
- [ ] Verify LSTM actually uses hidden state history
- [ ] Analyze reward signal strength vs death penalty dominance
- [ ] Test if 30% threshold is optimal (too high? too low?)
- [ ] Measure exploration time needed to find all affordances
- [ ] Calculate minimum survival time needed to complete one affordance cycle
- [ ] **After fix:** Run multi-day training to see if convergence improves

---

**End of Document** - Updated as testing progresses
