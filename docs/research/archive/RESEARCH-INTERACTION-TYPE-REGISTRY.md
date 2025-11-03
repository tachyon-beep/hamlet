---
**STATUS**: ✅ INTEGRATED (2025-11-04)
**Integration Document**: [docs/PHASE-2-INTEGRATION-COMPLETE.md](/home/john/hamlet/docs/PHASE-2-INTEGRATION-COMPLETE.md)
**Review Report**: [docs/PHASE-3-REVIEW-REPORT.md](/home/john/hamlet/docs/PHASE-3-REVIEW-REPORT.md)
**All findings have been integrated into task documents. This research paper is archived.**
---

# Research: Extensible Interaction Type Registry

**Date**: 2025-11-04
**Author**: Claude Code
**Status**: ARCHIVED (Integrated 2025-11-04)
**Estimated Effort**: 16-24 hours implementation

---

## Executive Summary

### Current State
The interaction system currently supports **4 hardcoded interaction types** (`instant`, `multi_tick`, `continuous`, `dual`) with behavior partially baked into Python code (`vectorized_env.py`, `affordance_engine.py`). While Level 3 temporal mechanics introduced multi-tick interactions, many interaction patterns students might want to explore remain **impossible without code changes**.

---

## 0. Design Principles from UAC System

This research follows the established UAC design principles from TASK-003 (Action Space Configuration):

### Conceptual Agnosticism

The interaction schema should NOT assume:
- ❌ Interactions must have instant effects
- ❌ Interactions must cost resources
- ❌ Interactions must be spatial (at a location)
- ❌ Interactions must be deterministic

### Structural Enforcement

The schema MUST enforce:
- ✅ All meter references exist in bars.yaml (validated at compile time)
- ✅ Capability parameters are type-safe
- ✅ Effect stages are well-defined (on_start, per_tick, on_completion, on_early_exit)
- ✅ Numeric values have sensible bounds (probabilities ∈ [0,1], ticks > 0, etc.)

### Permissive Semantics, Strict Syntax

- ✅ Allow: `effects: []` (interaction with no effects = purely informational)
- ✅ Allow: `costs: []` (free interaction)
- ✅ Allow: `duration_ticks: 1` (instant multi-tick = behaves like instant)
- ✅ Allow: `success_probability: 1.0` (deterministic = no randomness)
- ❌ Reject: `duration_ticks: -5` (negative duration)
- ❌ Reject: `meter: "nonexistent"` (dangling reference)
- ❌ Reject: `success_probability: 1.5` (probability > 1)

### Pattern Consistency with Actions

Actions (TASK-003) and Affordances (this research) should follow similar patterns:

| Feature | Actions (TASK-003) | Affordances (This Research) |
|---------|-------------------|----------------------------|
| Categorization | `type: movement` | `capabilities: [multi_tick, ...]` |
| Resource costs | `energy_cost: 0.005` (legacy)<br>`costs: [{meter, amount}]` (new) | `costs: [{meter, amount}]` (already exists) |
| Effects | `effects: [{meter, amount}]` | `effects: [{meter, amount}]` (already exists) |
| Validation | Compile-time (TASK-004) | Compile-time (TASK-004) |
| Extensibility | Add new action types via config | Add new capabilities via config |

**Key Insight**: Actions and affordances are both "agent behaviors" - actions are "what agent can always do", affordances are "what agent can do at specific locations/conditions". They should share the same effect/cost patterns.

---

### Design Space
Research identified **32+ distinct interaction patterns** across 5 dimensions:
- **Spatial**: 6 patterns (proximity, remote, range-based, line-of-sight, zone, multi-location)
- **Temporal**: 8 patterns (instant, multi-tick, cooldown, one-time, recurring, interruptible, resumable, scheduled)
- **Conditional**: 10 patterns (unconditional, meter-gated, item-gated, skill-probabilistic, time-gated, sequence-gated, occupancy-limited, prerequisite-chained, state-dependent, weather-conditional)
- **Effect**: 5 patterns (deterministic, probabilistic, scaled, staged, compound)
- **Occupancy**: 3 patterns (single, multi, queued)

### Recommended Approach
**Hybrid: Capability Composition + Effect Pipeline** (Options B + C)

Use **capability flags** to enable/disable features (composable behaviors) combined with **effect pipelines** for fine-grained control over interaction lifecycle. This balances expressivity (handles 90%+ of pedagogical use cases) with learnability (simple configs for simple cases, verbose for complex).

**Pattern Alignment with TASK-003**:
- Actions use single `type` (movement, interaction, passive, transaction)
- Affordances use multiple composable `capabilities` (multi_tick, cooldown, meter_gated, etc.)
- Both share `costs: [{meter, amount}]` and `effects: [{meter, amount}]` patterns
- Both follow same validation approach (structural then semantic)

### Effort Estimate
- **Phase 1 (Foundation)**: 8-10 hours - Capability system, validation infrastructure
- **Phase 2 (Effect Pipeline)**: 6-8 hours - Multi-stage effects, conditional logic
- **Phase 3 (Advanced Patterns)**: 2-4 hours - Cooldowns, prerequisites, queues
- **Total**: 16-22 hours

---

## 1. Current State Analysis

### 1.1 Hardcoded Interaction Patterns

**Current Implementation** (`affordance_config.py` lines 77-78):
```python
interaction_type: Literal["instant", "multi_tick", "continuous", "dual"]
```

**What's Hardcoded in Python**:

1. **Proximity Requirement** (`vectorized_env.py` lines 273-285):
   - All interactions require agent to be **exactly at affordance position** (`distance == 0`)
   - No support for remote interactions, range-based access, or line-of-sight
   - **Hardcoded**: `distances = torch.abs(self.positions - affordance_pos).sum(dim=1)`

2. **Operating Hours** (`affordance_engine.py` lines 86-110):
   - Time-gating logic hardcoded in `is_affordance_open()`
   - Only supports simple open/close hours with wraparound
   - No support for complex schedules (weekday/weekend, seasonal)

3. **Multi-Tick Mechanics** (`vectorized_env.py` lines 389-395):
   - Progress resets when agent moves away (no resumption)
   - No support for interruptible interactions
   - **Hardcoded**: `if not torch.equal(old_positions[agent_idx], new_positions[agent_idx])`

4. **Affordability Checks** (`vectorized_env.py` lines 469-474):
   - Only checks `money` meter (hardcoded index 3)
   - No support for multi-resource costs (need energy AND money)
   - **Hardcoded**: `can_afford = self.meters[:, 3] >= cost_per_tick`

5. **Effect Application** (`affordance_engine.py` lines 196-225):
   - Deterministic only (no probabilistic effects)
   - No scaling based on agent state (e.g., fitness affects gym effectiveness)
   - All effects applied uniformly to masked agents

6. **Occupancy** (Not Implemented):
   - No concept of single-occupancy affordances (e.g., only one agent at bed)
   - Multiple agents can simultaneously use same affordance
   - No queuing system

**Pedagogical Impact**: Students cannot explore:
- Skill-based interactions (gym effectiveness depends on fitness)
- Remote actions (phone call from anywhere)
- Cooldown mechanics (can't spam same action)
- Consumable resources (one-time pickup items)
- Prerequisite chains (must do A before B unlocks)
- Queued access (realistic waiting for services)

---

### 1.4 Current Gap: Action-Affordance Asymmetry

**Actions** (TASK-003 current state):
```yaml
# actions.yaml (legacy)
actions:
  - id: 0
    name: "UP"
    energy_cost: 0.005  # ❌ Only single meter (legacy field)
```

**Affordances** (current state):
```yaml
# affordances.yaml (current)
affordances:
  - id: "Job"
    costs:
      - {meter: energy, amount: 0.1}
      - {meter: time, amount: 0.04}  # ✅ Multi-meter costs
```

**Problem**: Actions can only cost energy (via `energy_cost` field), but affordances can cost multiple meters. This is an inconsistency in the UAC system.

**Solution** (from TASK-003 Gap 2):
- Add `costs: [{meter, amount}]` field to actions.yaml (matching affordances pattern)
- Keep `energy_cost` for backward compatibility (auto-converts to costs list)
- Both actions and affordances now use same cost pattern

**Example** (future actions.yaml):
```yaml
actions:
  - id: 0
    name: "UP"
    type: "movement"
    costs:  # ✅ New multi-meter costs
      - {meter: energy, amount: 0.005}
      - {meter: hygiene, amount: 0.003}  # Walking makes you dirty
      - {meter: satiation, amount: 0.004}  # Walking makes you hungry
    delta: [0, -1]
```

This aligns actions with affordances - both use the same `costs: [{meter, amount}]` pattern for resource depletion.

---

### 1.2 Current Interaction Lifecycle

**Level 1-2 (Instant Mode)**:
```
1. Agent issues INTERACT action
2. Check proximity (distance == 0)
3. Check affordability (money >= cost)
4. Apply costs (subtract from meters)
5. Apply effects (add to meters)
6. Clamp meters [0, 1]
```

**Level 3+ (Multi-Tick Mode)**:
```
1. Agent issues INTERACT action
2. Check proximity (distance == 0)
3. Check affordability (money >= cost_per_tick)
4. Check/update progress:
   - If same affordance + position → increment progress
   - If different → reset progress to 1
5. Apply per-tick costs
6. Apply per-tick effects
7. If final tick → apply completion_bonus
8. If agent moves → reset progress (no resumption)
```

**Key Observations**:
- **No state persistence** across episodes (can't save progress)
- **No early exit rewards** (agent must stay full duration or lose progress)
- **No conditional effects** (same effect regardless of agent state)
- **No interaction failures** (if you can afford it, it always succeeds)

---

### 1.3 Multi-Tick Interaction Model (L3)

**Current Implementation**:
```yaml
- id: "10"
  name: "Job"
  interaction_type: "dual"
  required_ticks: 4

  # Per-tick (75% of total reward distributed linearly)
  effects_per_tick:
    - { meter: "money", amount: 0.05625 }  # $5.625/tick

  # Completion (25% bonus for finishing)
  completion_bonus:
    - { meter: "money", amount: 0.05625 }  # $5.625 final
    - { meter: "energy", amount: -0.15 }   # Side effects at end
```

**75/25 Split Rationale**:
- **75% linear**: Provides dense reward signal, encourages progress
- **25% completion**: Incentivizes commitment, prevents "free sampling"

**Progress Tracking** (`vectorized_env.py` lines 166-168):
```python
self.interaction_progress = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
self.last_interaction_affordance: list[str | None] = [None] * self.num_agents
self.last_interaction_position = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
```

**Current Limitations**:
1. **No early exit mechanics**: Agent loses all progress if leaves before completion
2. **No resumption**: Cannot pause and resume later (resets to tick 0)
3. **No partial rewards**: No "you worked 2/4 ticks, here's partial payment"
4. **No dynamic duration**: Cannot have skill-dependent completion time

**Pedagogical Gap**: Students cannot model:
- University (multi-semester degree with semester breaks)
- Construction (resumable project across days)
- Cooking (can pause and resume preparation)
- Training programs (progressive skill-building with milestones)

---

## 2. Interaction Pattern Design Space

### 2.1 Spatial Patterns

**1. Proximity (Current Default)**
- **Mechanic**: Agent must be at exact position (`distance == 0`)
- **Use Cases**: Bed, Shower, Gym (physical location required)
- **Config**:
  ```yaml
  spatial_requirement:
    type: "proximity"
    distance: 0  # Exact match
  ```

**2. Remote (No Spatial Constraint)**
- **Mechanic**: Can interact from anywhere on grid
- **Use Cases**: Phone call, meditation, rest command
- **Config**:
  ```yaml
  spatial_requirement:
    type: "remote"  # No distance check
  ```

**3. Range-Based (Distance Threshold)**
- **Mechanic**: Can interact if within N tiles
- **Use Cases**: Vending machine (see from distance), billboard (read from afar)
- **Config**:
  ```yaml
  spatial_requirement:
    type: "range"
    max_distance: 2  # Within 2 tiles
    distance_metric: "manhattan"  # or "euclidean", "chebyshev"
  ```

**4. Line-of-Sight (Vision-Based)**
- **Mechanic**: Need clear path to affordance (no obstacles)
- **Use Cases**: Sniper rifle, telescope, lighthouse
- **Config**:
  ```yaml
  spatial_requirement:
    type: "line_of_sight"
    max_distance: 5
    requires_clear_path: true
  ```
  **Note**: Requires obstacle grid (not yet implemented)

**5. Zone-Based (Region Containment)**
- **Mechanic**: Agent must be in specific region (not exact tile)
- **Use Cases**: Park (anywhere in park zone), Office (anywhere in building)
- **Config**:
  ```yaml
  spatial_requirement:
    type: "zone"
    zone_id: "downtown"  # References zones.yaml
  ```
  **Note**: Requires multi-zone system (L4 planned)

**6. Multi-Location (Multiple Valid Positions)**
- **Mechanic**: Affordance exists at multiple positions (any valid)
- **Use Cases**: ATM (multiple machines), Water fountain (multiple locations)
- **Config**: Already supported via deployment system (multiple instances)

**Priority**:
- ✅ **High**: Remote, Range-Based (simple, high pedagogical value)
- 🔶 **Medium**: Line-of-Sight (requires obstacle system, L5+)
- 🔶 **Low**: Zone-Based (requires multi-zone, L4)

---

### 2.2 Temporal Patterns

**1. Instant (Current Default)**
- **Mechanic**: Effects applied immediately in single step
- **Use Cases**: Bed, Shower, HomeMeal (simple restoration)
- **Config**: Already supported (`interaction_type: "instant"`)

**2. Multi-Tick (Current L3)**
- **Mechanic**: Requires N steps, effects distributed + completion bonus
- **Use Cases**: Job (4 ticks), Gym (3 ticks), University (10 ticks)
- **Config**: Already supported (`interaction_type: "multi_tick"`)

**3. Cooldown (Post-Use Delay)**
- **Mechanic**: After use, cannot use again for N steps
- **Use Cases**: Energy drink (can't spam), Doctor (once per day), Skill training (daily limit)
- **Config**:
  ```yaml
  temporal_constraint:
    type: "cooldown"
    cooldown_ticks: 24  # Can't reuse for 24 steps
    cooldown_scope: "agent"  # or "global" (all agents share cooldown)
  ```

**4. One-Time (Consumable)**
- **Mechanic**: Affordance disappears after single use
- **Use Cases**: Treasure chest, quest item, limited resource
- **Config**:
  ```yaml
  temporal_constraint:
    type: "one_time"
    respawn: false  # or respawn_after: 100 (ticks)
  ```

**5. Recurring (Periodic Reset)**
- **Mechanic**: Effects can be claimed once per period (e.g., daily wage)
- **Use Cases**: Daily quest, weekly paycheck, seasonal harvest
- **Config**:
  ```yaml
  temporal_constraint:
    type: "recurring"
    period: 24  # Ticks per period
    auto_reset: true
  ```

**6. Interruptible (Early Exit Allowed)**
- **Mechanic**: Agent can leave before completion, keeps partial progress
- **Use Cases**: University (drop out, keep partial education), Job (quit early, keep partial payment)
- **Config**:
  ```yaml
  multi_tick:
    required_ticks: 10
    interruptible: true
    early_exit_penalty: 0.5  # Lose 50% of accumulated rewards
  ```

**7. Resumable (Can Pause and Continue)**
- **Mechanic**: Progress persists even if agent leaves
- **Use Cases**: Construction project (resume next day), Cooking (pause and resume)
- **Config**:
  ```yaml
  multi_tick:
    required_ticks: 10
    resumable: true
    progress_decay: 0.1  # Lose 10% progress per step away
  ```

**8. Scheduled (Time-Window Constraints)**
- **Mechanic**: More complex than operating_hours (e.g., weekdays only, seasonal)
- **Use Cases**: Weekend market, Summer festival, Night shift
- **Config**:
  ```yaml
  temporal_constraint:
    type: "scheduled"
    schedule:
      - { day_pattern: "weekday", hours: [8, 18] }
      - { day_pattern: "weekend", hours: [10, 14] }
  ```
  **Note**: Requires day/week system (not currently modeled)

**Priority**:
- ✅ **High**: Cooldown, Interruptible, Resumable (extend current multi-tick)
- 🔶 **Medium**: One-Time, Recurring (inventory system needed)
- 🔶 **Low**: Scheduled (requires calendar system)

---

### 2.3 Conditional Patterns

**1. Unconditional (Current Default)**
- **Mechanic**: Always available if proximity + affordability checks pass
- **Use Cases**: Bed, Shower, HomeMeal
- **Config**: Default behavior

**2. Meter-Gated (Threshold Requirement)**
- **Mechanic**: Requires meter(s) above/below threshold
- **Use Cases**: Gym requires energy >20%, Hospital only if health <50%
- **Config**:
  ```yaml
  prerequisites:
    - type: "meter_threshold"
      meter: "energy"
      min: 0.2  # Need at least 20% energy
    - type: "meter_threshold"
      meter: "health"
      max: 0.5  # Only if health below 50%
  ```

**3. Item-Gated (Inventory Requirement)**
- **Mechanic**: Requires specific item in inventory
- **Use Cases**: Library requires library card, Restaurant requires reservation
- **Config**:
  ```yaml
  prerequisites:
    - type: "item_required"
      item: "library_card"
      consume: false  # Don't consume on use
  ```
  **Note**: Requires inventory system (not yet implemented)

**4. Skill-Probabilistic (Success Rate Based on Meter)**
- **Mechanic**: Interaction can fail, success probability depends on agent state
- **Use Cases**: Gym effectiveness scales with fitness, Job performance depends on energy
- **Config**:
  ```yaml
  effects:
    - meter: "fitness"
      amount: 0.30
      success_probability:
        base: 0.5  # 50% base success
        scaling_meter: "fitness"  # Higher fitness → higher success
        scaling_factor: 0.5  # +50% if fitness = 1.0
  ```

**5. Time-Gated (Operating Hours)**
- **Mechanic**: Only available during certain hours (already implemented)
- **Use Cases**: Job (9am-5pm), Bar (6pm-2am)
- **Config**: Already supported (`operating_hours: [9, 17]`)

**6. Sequence-Gated (Prerequisite Chain)**
- **Mechanic**: Must complete affordance A before B unlocks
- **Use Cases**: University progression (Freshman → Sophomore → Junior → Senior)
- **Config**:
  ```yaml
  prerequisites:
    - type: "affordance_completed"
      affordance: "Freshman_Year"
      min_completions: 1
  ```

**7. Occupancy-Limited (Capacity Constraint)**
- **Mechanic**: Maximum N agents can use simultaneously
- **Use Cases**: Bed (1 agent), Classroom (30 agents), Bus (50 agents)
- **Config**:
  ```yaml
  occupancy:
    max_concurrent: 1  # Single-occupancy
    queue_overflow: true  # If full, queue agents
  ```

**8. Prerequisite-Chained (Multi-Step Unlocking)**
- **Mechanic**: Complex AND/OR conditions to unlock
- **Use Cases**: Advanced training (requires fitness >50% AND completed basic training)
- **Config**:
  ```yaml
  prerequisites:
    operator: "AND"
    conditions:
      - type: "meter_threshold"
        meter: "fitness"
        min: 0.5
      - type: "affordance_completed"
        affordance: "Basic_Training"
  ```

**9. State-Dependent (Effect Varies by Agent State)**
- **Mechanic**: Same affordance produces different effects based on meters
- **Use Cases**: Gym gives more benefit if fitness is low, diminishing returns if high
- **Config**:
  ```yaml
  effects:
    - meter: "fitness"
      scaling_mode: "diminishing_returns"
      base_amount: 0.30
      current_meter_weight: -0.5  # Less effective if already fit
  ```

**10. Weather/Environment-Conditional (Future: L6)**
- **Mechanic**: Availability depends on environmental state
- **Use Cases**: Park (not available if raining), Beach (only in summer)
- **Config**: Deferred (requires environment state system)

**Priority**:
- ✅ **High**: Meter-Gated, Skill-Probabilistic, Sequence-Gated (core pedagogical patterns)
- 🔶 **Medium**: Occupancy-Limited, State-Dependent (useful but complex)
- 🔶 **Low**: Item-Gated (needs inventory), Weather-Conditional (needs environment)

---

### 2.4 Effect Patterns

**1. Deterministic (Current Default)**
- **Mechanic**: Fixed effects every time
- **Use Cases**: Bed (+50% energy), Job (+$22.50)
- **Config**: Already supported (default behavior)

**2. Probabilistic (Success/Failure)**
- **Mechanic**: Effects only apply if interaction succeeds (dice roll)
- **Use Cases**: Gambling, Risky investment, Dating (might succeed or fail)
- **Config**:
  ```yaml
  effects:
    - meter: "money"
      amount: 0.50  # +$50 if success
      success_probability: 0.3  # 30% chance
    - meter: "money"
      amount: -0.10  # -$10 if failure
      failure_probability: 0.7  # 70% chance
  ```

**3. Scaled (Magnitude Depends on State)**
- **Mechanic**: Effect amount varies based on agent's current meters
- **Use Cases**: Gym (more effective if rested), Learning (better if high mood)
- **Config**:
  ```yaml
  effects:
    - meter: "fitness"
      base_amount: 0.20
      scaling:
        meter: "energy"  # Scales with energy
        factor: 0.5  # 50% bonus if full energy
  ```

**4. Staged (Different Effects per Tick)**
- **Mechanic**: Each tick applies different effects (progression)
- **Use Cases**: University (different learning per semester), Training (progressive difficulty)
- **Config**:
  ```yaml
  multi_tick:
    required_ticks: 4
    effects_by_stage:
      - tick: 0
        effects: [{ meter: "fitness", amount: 0.05 }]  # Easy start
      - tick: 1
        effects: [{ meter: "fitness", amount: 0.10 }]  # Ramp up
      - tick: 2
        effects: [{ meter: "fitness", amount: 0.15 }]  # Peak
      - tick: 3
        effects: [{ meter: "fitness", amount: 0.10 }]  # Cool down
  ```

**5. Compound (Multi-Effect Interactions)**
- **Mechanic**: Effects depend on other effects (cascading)
- **Use Cases**: Alcohol (+social, -energy, THEN -health if energy low)
- **Config**:
  ```yaml
  effects:
    - meter: "social"
      amount: 0.50
    - meter: "energy"
      amount: -0.20
    - meter: "health"  # Conditional effect
      amount: -0.10
      condition:
        meter: "energy"
        operator: "<"
        threshold: 0.3  # Only if energy drops below 30%
  ```

**Priority**:
- ✅ **High**: Probabilistic, Scaled (rich pedagogical scenarios)
- 🔶 **Medium**: Staged, Compound (advanced patterns)

---

### 2.5 Occupancy Patterns

**1. Single-Occupancy (One Agent at a Time)**
- **Mechanic**: Only one agent can use affordance simultaneously
- **Use Cases**: Bed, Shower, Therapist (one-on-one)
- **Config**:
  ```yaml
  occupancy:
    max_concurrent: 1
    blocking_behavior: "reject"  # Others get error if occupied
  ```

**2. Multi-Occupancy (Capacity Limit)**
- **Mechanic**: Up to N agents can use simultaneously
- **Use Cases**: Classroom (30 students), Bus (50 passengers), Restaurant (20 tables)
- **Config**:
  ```yaml
  occupancy:
    max_concurrent: 30
    blocking_behavior: "queue"  # Wait in line if full
  ```

**3. Queued (FIFO Waiting)**
- **Mechanic**: Agents wait in line, served in order
- **Use Cases**: Doctor (queue system), DMV (ticket queue)
- **Config**:
  ```yaml
  occupancy:
    max_concurrent: 1
    blocking_behavior: "queue"
    queue_limit: 10  # Max 10 in queue
    queue_timeout: 50  # Leave queue after 50 ticks waiting
  ```

**Priority**:
- ✅ **High**: Single-Occupancy (realistic resource competition)
- 🔶 **Medium**: Queued (teaches waiting dynamics)
- 🔶 **Low**: Multi-Occupancy (niche use case)

**Note**: Requires state management (which agents are currently using affordance).

---

## 2.6 Pattern Consistency: Actions vs Affordances

This section demonstrates how actions (TASK-003) and affordances (this research) share the same design patterns for costs, effects, and categorization.

### Example 1: Resource Costs

**Action** (TASK-003):
```yaml
# actions.yaml
actions:
  - id: 0
    name: "UP"
    type: "movement"
    costs:
      - {meter: energy, amount: 0.005}
      - {meter: hygiene, amount: 0.003}
      - {meter: satiation, amount: 0.004}
```

**Affordance** (this research):
```yaml
# affordances.yaml
affordances:
  - id: "Gym"
    costs:
      - {meter: energy, amount: 0.1}
      - {meter: money, amount: 0.05}
```

**Pattern**: Both use `costs: [{meter, amount}]` list format.

---

### Example 2: Positive Effects

**Action** (TASK-003):
```yaml
# actions.yaml
actions:
  - id: 5
    name: "REST"
    type: "passive"
    effects:
      - {meter: mood, amount: 0.02}
      - {meter: energy, amount: 0.01}
```

**Affordance** (this research):
```yaml
# affordances.yaml
affordances:
  - id: "Bed"
    effects:
      - {meter: energy, amount: 0.3}
      - {meter: mood, amount: 0.1}
```

**Pattern**: Both use `effects: [{meter, amount}]` list format.

---

### Example 3: Type/Capability Categorization

**Action** (TASK-003):
```yaml
# actions.yaml
actions:
  - id: 0
    name: "UP"
    type: "movement"  # Single type classification
    delta: [0, -1]
```

**Affordance** (this research):
```yaml
# affordances.yaml
affordances:
  - id: "Job"
    capabilities:  # Multiple capabilities can compose
      - {type: multi_tick, duration_ticks: 10}
      - {type: cooldown, cooldown_ticks: 50}
      - {type: meter_gated, meter: energy, min: 0.3}
```

**Pattern Asymmetry** (intentional):
- **Actions** have single `type` (simple) - actions are primitive behaviors
- **Affordances** have multiple `capabilities` (composable) - affordances are compound behaviors

This asymmetry is **by design**:
- Actions are **primitive** (one type per action = clear semantics)
- Affordances are **compound** (multiple capabilities compose = rich behaviors)

---

### Example 4: Validation Consistency

Both actions and affordances follow the same validation approach (from TASK-003 design principles):

**Structural Validation** (syntax):
```python
# Validate action structure
for action in actions:
    if action.type == "movement" and action.delta is None:
        raise ValueError(f"Movement action '{action.name}' requires delta")

# Validate affordance structure
for affordance in affordances:
    for capability in affordance.capabilities:
        if capability.type == "multi_tick" and "duration_ticks" not in capability:
            raise ValueError(f"multi_tick capability requires duration_ticks")
```

**Semantic Validation** (cross-references):
```python
# Validate meter references in actions
valid_meters = {bar.name for bar in bars_config.bars}
for action in actions:
    for cost in action.costs:
        if cost.meter not in valid_meters:
            raise ValueError(f"Action {action.name}: Unknown meter '{cost.meter}'")

# Validate meter references in affordances
for affordance in affordances:
    for cost in affordance.costs:
        if cost.meter not in valid_meters:
            raise ValueError(f"Affordance {affordance.id}: Unknown meter '{cost.meter}'")
```

**Pattern**: Both use two-stage validation (structure first, then semantics).

---

## 3. Design Options

### Option A: Type System (Finite Registry)

**Concept**: Predefine fixed set of interaction types (like current system)

**Schema**:
```yaml
affordances:
  - id: "Job"
    interaction_type: "timed_resource_generator"  # Fixed type from registry
    parameters:
      duration_ticks: 10
      cooldown_ticks: 5
      resource_output: [{meter: money, amount: 0.25}]

  - id: "Gym"
    interaction_type: "skill_trainer"  # Different type
    parameters:
      target_meter: fitness
      base_effectiveness: 0.30
      diminishing_returns: true
```

**Registry of Types**:
- `instant_restoration` (Bed, Shower)
- `timed_resource_generator` (Job, Mining)
- `skill_trainer` (Gym, University)
- `consumable_pickup` (Treasure, Quest Item)
- `probabilistic_gamble` (Casino, Lottery)
- `social_hub` (Bar, Park)
- ...

**Pros**:
- ✅ **Simple**: Operators just pick from menu
- ✅ **Self-documenting**: Type name explains behavior
- ✅ **Easy validation**: Check type exists, validate parameters

**Cons**:
- ❌ **Limited expressivity**: Can only use predefined types
- ❌ **Code changes required**: Adding new type requires Python changes
- ❌ **Naming overload**: Need to invent names for every pattern combination
- ❌ **Not composable**: Can't mix features (e.g., "timed + probabilistic + cooldown")

**Expressivity Score**: 5/10 (limited to predefined types)
**Learnability Score**: 9/10 (easy to understand)
**Extensibility Score**: 3/10 (requires code changes)

---

### Option B: Capability Composition (Mix-and-Match Features)

**Concept**: Affordances declare **capabilities** (features) they want, system combines them

**Schema**:
```yaml
affordances:
  - id: "Job"
    name: "Office Job"
    category: "income"

    # Compose capabilities
    capabilities:
      - type: "multi_tick"
        duration_ticks: 10
        early_exit_allowed: true

      - type: "cooldown"
        cooldown_ticks: 50

      - type: "meter_gated"
        meter: "energy"
        min_threshold: 0.3

      - type: "skill_based_effectiveness"
        scaling_meter: "fitness"
        scaling_factor: 0.2

    # Standard effects
    effects_per_tick:
      - {meter: money, amount: 0.025}
    completion_bonus:
      - {meter: money, amount: 0.025}
```

**Capability Registry** (Expandable):
- `multi_tick` (duration, early_exit, resumable)
- `cooldown` (duration, scope)
- `meter_gated` (meter, min, max)
- `skill_based_effectiveness` (scaling meter, factor)
- `probabilistic_success` (base_rate, scaling)
- `occupancy_limited` (max_concurrent, queue)
- `remote_access` (no proximity required)
- `range_based` (max_distance, metric)
- `one_time_use` (respawn settings)

**Pros**:
- ✅ **Highly composable**: Mix any capabilities
- ✅ **Extensible**: New capabilities = new YAML schema (no Python if logic generic)
- ✅ **Explicit**: Clear what features are enabled
- ✅ **Progressive disclosure**: Simple configs are short, complex configs are verbose

**Cons**:
- ⚠️ **Verbosity**: Complex interactions have many capability blocks
- ⚠️ **Conflicting capabilities**: Need validation (can't be `one_time` + `multi_tick`)
- ⚠️ **Learning curve**: Operators must understand capability interactions

**Expressivity Score**: 9/10 (can combine most patterns)
**Learnability Score**: 7/10 (medium complexity)
**Extensibility Score**: 9/10 (add capabilities without code)

---

### Option C: Effect Pipeline (Sequence of Stages)

**Concept**: Define interaction as **pipeline of effect stages** (on_start, per_tick, on_completion, on_exit)

**Schema**:
```yaml
affordances:
  - id: "Job"
    name: "Office Job"

    # Interaction pipeline
    interaction_pipeline:
      # Stage 1: Entry (when interaction starts)
      on_start:
        prerequisites:
          - {type: meter_threshold, meter: energy, min: 0.3}
        costs:
          - {meter: energy, amount: 0.05}
        effects:
          - {meter: mood, amount: -0.10}  # "Ugh, work"

      # Stage 2: Per-tick (each step during interaction)
      per_tick:
        costs:
          - {meter: money, amount: 0.0}  # FREE (you get paid!)
        effects:
          - meter: money
            amount: 0.025
            scaling:
              meter: fitness  # Better performance if fit
              factor: 0.2

      # Stage 3: Completion (finish full duration)
      on_completion:
        effects:
          - {meter: money, amount: 0.025}  # Bonus
          - {meter: social, amount: 0.02}

      # Stage 4: Early exit (leave before done)
      on_exit:
        effects:
          - {meter: mood, amount: -0.05}  # Penalty for quitting

    # Constraints
    duration_ticks: 10
    early_exit_allowed: true
    cooldown_after_use: 50
```

**Stages**:
- `on_start`: Prerequisites, entry costs, initial effects
- `per_tick`: Recurring costs/effects each tick
- `on_completion`: Bonus for finishing
- `on_exit`: Penalties/effects if leave early
- `on_failure`: Effects if interaction fails (probabilistic)

**Pros**:
- ✅ **Explicit lifecycle**: Clear when effects apply
- ✅ **Flexible**: Different effects at different stages
- ✅ **Rich conditionals**: Can have per-stage prerequisites
- ✅ **Self-documenting**: Pipeline shows interaction flow

**Cons**:
- ⚠️ **Verbose**: Even simple interactions need multiple stages
- ⚠️ **Overlap with capabilities**: Stages + capabilities might conflict
- ⚠️ **Validation complexity**: Need to ensure stages are consistent

**Expressivity Score**: 8/10 (handles complex patterns well)
**Learnability Score**: 6/10 (requires understanding stages)
**Extensibility Score**: 7/10 (add new stages = moderate effort)

---

### Option D: Behavioral Flags (Enable/Disable Features)

**Concept**: Simple boolean flags enable/disable behaviors (flat configuration)

**Schema**:
```yaml
affordances:
  - id: "Job"
    name: "Office Job"

    # Spatial
    requires_proximity: true
    proximity_distance: 0

    # Temporal
    multi_tick: true
    duration_ticks: 10
    interruptible: true
    resumable: false
    has_cooldown: true
    cooldown_ticks: 50

    # Conditional
    meter_gated: true
    required_meter: "energy"
    required_threshold: 0.3

    # Effects
    probabilistic_success: false
    scaled_by_meter: "fitness"
    scaling_factor: 0.2

    # Standard costs/effects
    costs_per_tick: []
    effects_per_tick:
      - {meter: money, amount: 0.025}
```

**Pros**:
- ✅ **Simple**: Just set flags true/false
- ✅ **Easy to scan**: All settings visible at once
- ✅ **Low verbosity**: Compact configs

**Cons**:
- ❌ **Flag explosion**: Need flag for every feature
- ❌ **Implicit behavior**: What does `interruptible: true` actually do?
- ❌ **Limited extensibility**: Adding feature = new flag everywhere
- ❌ **Poor defaults**: Must specify all flags (or have magic defaults)
- ❌ **Conflicting flags**: No structure prevents invalid combinations

**Expressivity Score**: 6/10 (limited by flag set)
**Learnability Score**: 8/10 (simple to use)
**Extensibility Score**: 4/10 (flag bloat)

---

## 4. Tradeoff Analysis

| Criterion | Type System (A) | Capability Composition (B) | Effect Pipeline (C) | Behavioral Flags (D) |
|-----------|-----------------|----------------------------|---------------------|----------------------|
| **Expressivity** | 5/10 | 9/10 | 8/10 | 6/10 |
| **Complexity** (lower = better) | Low | Medium | Medium-High | Low |
| **Learnability** | 9/10 | 7/10 | 6/10 | 8/10 |
| **Extensibility** | 3/10 | 9/10 | 7/10 | 4/10 |
| **Validation** | Easy | Medium | Hard | Easy |
| **Implementation Effort** | 12h | 16h | 20h | 10h |
| **Verbosity (Simple Case)** | Low | Low | High | Low |
| **Verbosity (Complex Case)** | N/A | Medium | High | Medium |
| **No-Code Extension** | No | Mostly | Partially | No |

**Scores Summary**:
- **Type System (A)**: Simple but limited (5/10 overall)
- **Capability Composition (B)**: Expressive and extensible (8.5/10 overall) ⭐
- **Effect Pipeline (C)**: Powerful but complex (7/10 overall)
- **Behavioral Flags (D)**: Easy but inflexible (6/10 overall)

---

## 5. Recommended Approach

### Hybrid: Capability Composition + Effect Pipeline (B + C)

**Rationale**:
- **Capabilities** handle **behavioral features** (cooldown, meter-gating, probabilistic)
- **Effect Pipeline** handles **effect distribution** (when effects apply in lifecycle)
- **Best of both**: Composability + expressivity without excessive complexity

**Schema Design**:

```yaml
affordances:
  # ============================================================================
  # Example 1: Simple Instant Interaction (Bed)
  # ============================================================================
  - id: "0"
    name: "Bed"
    category: "energy_restoration"

    # No capabilities = instant, unconditional, deterministic
    costs:
      - {meter: money, amount: 0.05}
    effects:
      - {meter: energy, amount: 0.50}
      - {meter: health, amount: 0.02}

    operating_hours: [0, 24]

  # ============================================================================
  # Example 2: Multi-Tick with Cooldown (Job)
  # ============================================================================
  - id: "10"
    name: "Job"
    category: "income"

    # Capabilities (compose features)
    capabilities:
      multi_tick:
        duration_ticks: 10
        early_exit_allowed: true
        resumable: false

      cooldown:
        cooldown_ticks: 50
        scope: "agent"  # Per-agent cooldown

      meter_gated:
        - {meter: energy, min: 0.3}  # Need 30% energy to start

    # Effect pipeline (when effects apply)
    effects:
      on_start:
        costs:
          - {meter: energy, amount: 0.05}  # Upfront cost

      per_tick:
        effects:
          - {meter: money, amount: 0.025}

      on_completion:
        effects:
          - {meter: money, amount: 0.025}  # 25% bonus
          - {meter: social, amount: 0.02}

      on_early_exit:
        effects:
          - {meter: mood, amount: -0.05}  # Penalty for quitting

    operating_hours: [8, 18]

  # ============================================================================
  # Example 3: Skill-Gated Probabilistic (Gym)
  # ============================================================================
  - id: "12"
    name: "Gym"
    category: "fitness"

    capabilities:
      multi_tick:
        duration_ticks: 3

      meter_gated:
        - {meter: energy, min: 0.2}

      skill_based_effectiveness:
        scaling_meter: "fitness"
        scaling_mode: "diminishing_returns"  # Less effective if already fit

      probabilistic_success:
        base_rate: 0.7  # 70% base success
        scaling_meter: "fitness"
        scaling_factor: 0.3  # +30% if fitness = 1.0

    effects:
      per_tick:
        costs:
          - {meter: money, amount: 0.02667}
        effects:
          - meter: fitness
            amount: 0.075
            # Scaling applied by capability system

      on_completion:
        effects:
          - {meter: fitness, amount: 0.075}
          - {meter: energy, amount: -0.08}

      on_failure:  # If probabilistic roll fails
        effects:
          - {meter: mood, amount: -0.10}  # "Bad workout"

    operating_hours: [6, 22]

  # ============================================================================
  # Example 4: One-Time Consumable (Treasure Chest)
  # ============================================================================
  - id: "20"
    name: "TreasureChest"
    category: "pickup"

    capabilities:
      one_time_use:
        respawn: false  # Disappears after use

      remote_access:
        enabled: false  # Must be at location

    effects:
      instant:
        effects:
          - {meter: money, amount: 0.50}  # +$50

    operating_hours: [0, 24]

  # ============================================================================
  # Example 5: Queued Single-Occupancy (Therapist)
  # ============================================================================
  - id: "7"
    name: "Therapist"
    category: "mood"

    capabilities:
      multi_tick:
        duration_ticks: 3

      occupancy_limited:
        max_concurrent: 1  # Only 1 agent at a time
        blocking_behavior: "queue"
        queue_limit: 5
        queue_timeout: 20  # Leave queue after 20 ticks

    effects:
      per_tick:
        costs:
          - {meter: money, amount: 0.05}
        effects:
          - {meter: mood, amount: 0.10}

      on_completion:
        effects:
          - {meter: mood, amount: 0.10}

    operating_hours: [8, 18]
```

---

### 5.1 Capability Registry (Extensible)

**Core Capabilities** (Phase 1):
1. ✅ `multi_tick` - Multi-step interactions
2. ✅ `cooldown` - Post-use delay
3. ✅ `meter_gated` - Threshold prerequisites
4. ✅ `skill_based_effectiveness` - Scaling effects

**Advanced Capabilities** (Phase 2):
5. ✅ `probabilistic_success` - Success/failure rolls
6. ✅ `one_time_use` - Consumables
7. ✅ `remote_access` - No proximity required
8. ✅ `range_based` - Distance threshold

**Future Capabilities** (Phase 3):
9. 🔶 `occupancy_limited` - Capacity constraints
10. 🔶 `sequence_gated` - Prerequisite chains
11. 🔶 `resumable` - Pause and continue
12. 🔶 `interruptible` - Early exit with rewards

---

### 5.2 Effect Pipeline Stages

**Supported Stages**:
- `instant`: For simple instant interactions (default)
- `on_start`: When interaction begins (entry costs, prerequisites)
- `per_tick`: Each tick during multi-tick interaction
- `on_completion`: When full duration completes (bonus)
- `on_early_exit`: When agent leaves before completion (penalty)
- `on_failure`: When probabilistic interaction fails

**Stage Semantics**:
- **Instant interactions**: Only use `instant` stage
- **Multi-tick interactions**: Use `on_start`, `per_tick`, `on_completion`, `on_early_exit`
- **Probabilistic interactions**: Add `on_failure` for failure effects

---

### 5.3 Migration Path (Backward Compatibility)

**Current Configs** (no capabilities):
```yaml
- id: "0"
  name: "Bed"
  interaction_type: "dual"
  costs:
    - {meter: money, amount: 0.05}
  effects:
    - {meter: energy, amount: 0.50}
  operating_hours: [0, 24]
```

**Interpretation**:
- No `capabilities` field → Use `interaction_type` (legacy mode)
- `interaction_type: "instant"` → No multi-tick, instant effects
- `interaction_type: "multi_tick"` → Implicit `multi_tick` capability
- `interaction_type: "dual"` → Support both modes (based on `enable_temporal_mechanics`)

**New Configs** (with capabilities):
```yaml
- id: "10"
  name: "Job"
  capabilities:
    multi_tick: {duration_ticks: 10}
    cooldown: {cooldown_ticks: 50}
  effects:
    per_tick: [{meter: money, amount: 0.025}]
    on_completion: [{meter: money, amount: 0.025}]
```

**Compiler Behavior**:
- If `capabilities` exists → Use new system
- If `interaction_type` exists (no capabilities) → Use legacy system
- Never mix both (validation error)

---

## 6. Concrete Examples

### Example 1: Simple Instant Interaction (Bed)

**Current System**:
```yaml
- id: "0"
  name: "Bed"
  interaction_type: "instant"
  costs:
    - {meter: money, amount: 0.05}
  effects:
    - {meter: energy, amount: 0.50}
  operating_hours: [0, 24]
```

**New System** (unchanged for simple cases):
```yaml
- id: "0"
  name: "Bed"
  costs:
    - {meter: money, amount: 0.05}
  effects:
    - {meter: energy, amount: 0.50}
  operating_hours: [0, 24]
```

**No change required**: Simple interactions stay simple.

---

### Example 2: Multi-Tick Interaction (Job)

**Current System**:
```yaml
- id: "10"
  name: "Job"
  interaction_type: "multi_tick"
  required_ticks: 10
  effects_per_tick:
    - {meter: money, amount: 0.025}
  completion_bonus:
    - {meter: money, amount: 0.025}
  operating_hours: [8, 18]
```

**New System** (with early exit + cooldown):
```yaml
- id: "10"
  name: "Job"
  capabilities:
    multi_tick:
      duration_ticks: 10
      early_exit_allowed: true
    cooldown:
      cooldown_ticks: 50
  effects:
    per_tick:
      - {meter: money, amount: 0.025}
    on_completion:
      - {meter: money, amount: 0.025}
    on_early_exit:
      - {meter: money, amount: 0.0125}  # Half bonus if quit early
      - {meter: mood, amount: -0.05}
  operating_hours: [8, 18]
```

**Added Behavior**:
- Early exit allowed (keeps partial progress)
- Cooldown prevents spamming
- Mood penalty for quitting

---

### Example 3: Skill-Gated Interaction (Gym)

**Current System** (deterministic only):
```yaml
- id: "12"
  name: "Gym"
  interaction_type: "multi_tick"
  required_ticks: 3
  costs_per_tick:
    - {meter: money, amount: 0.02667}
  effects_per_tick:
    - {meter: fitness, amount: 0.075}
  completion_bonus:
    - {meter: fitness, amount: 0.075}
    - {meter: energy, amount: -0.08}
  operating_hours: [6, 22]
```

**New System** (skill-based effectiveness):
```yaml
- id: "12"
  name: "Gym"
  capabilities:
    multi_tick:
      duration_ticks: 3
    meter_gated:
      - {meter: energy, min: 0.2}
    skill_based_effectiveness:
      scaling_meter: "fitness"
      scaling_mode: "diminishing_returns"
      base_effectiveness: 1.0
      diminishing_factor: -0.5  # Half as effective if fitness = 1.0
  effects:
    per_tick:
      - meter: fitness
        amount: 0.075  # Base amount (scaled by capability)
      - meter: money
        amount: -0.02667  # Cost
    on_completion:
      - {meter: fitness, amount: 0.075}
      - {meter: energy, amount: -0.08}
  operating_hours: [6, 22]
```

**Added Behavior**:
- Requires 20% energy to start
- Effectiveness scales with current fitness (diminishing returns)
- More realistic: Gym is less effective if already very fit

---

### Example 4: Conditional Availability (Restaurant)

**Current System** (not possible):
```yaml
# Cannot model "only available if money >= $20"
```

**New System** (meter-gated):
```yaml
- id: "15"
  name: "FancyRestaurant"
  category: "food"
  capabilities:
    meter_gated:
      - {meter: money, min: 0.20}  # Need $20 to enter
      - {meter: social, min: 0.1}  # Need some social confidence
  costs:
    - {meter: money, amount: 0.25}
  effects:
    - {meter: satiation, amount: 0.60}
    - {meter: mood, amount: 0.30}
    - {meter: social, amount: 0.10}
  operating_hours: [17, 22]  # Dinner only
```

**Added Behavior**:
- Gated by multiple meters (money AND social)
- Teaches resource management (need to save up)

---

### Example 5: Multi-Stage Interaction (University)

**Current System** (not possible):
```yaml
# Cannot model 4-year progression
```

**New System** (prerequisite chain):
```yaml
- id: "30"
  name: "FreshmanYear"
  category: "education"
  capabilities:
    multi_tick:
      duration_ticks: 20
      resumable: true  # Can take breaks
    one_time_use:
      respawn: false  # Can only complete once
  effects:
    per_tick:
      - {meter: money, amount: -0.05}  # Tuition cost
      - {meter: fitness, amount: 0.01}  # Learn skill
    on_completion:
      - {meter: fitness, amount: 0.20}  # Degree benefit
  operating_hours: [8, 16]

- id: "31"
  name: "SophomoreYear"
  category: "education"
  capabilities:
    sequence_gated:
      prerequisites:
        - {affordance: "FreshmanYear", min_completions: 1}
    multi_tick:
      duration_ticks: 20
      resumable: true
  effects:
    per_tick:
      - {meter: money, amount: -0.05}
      - {meter: fitness, amount: 0.015}  # More advanced
    on_completion:
      - {meter: fitness, amount: 0.25}
  operating_hours: [8, 16]

# JuniorYear, SeniorYear follow same pattern
```

**Added Behavior**:
- Four-stage progression (Freshman → Sophomore → Junior → Senior)
- Can pause and resume each year
- Each year unlocks next (prerequisite chain)
- Teaches long-term planning and commitment

---

## 7. Implementation Plan

### Phase 1: Foundation (8-10 hours)

**Goal**: Capability system infrastructure + core capabilities

**Tasks**:
1. **Capability Schema** (2h)
   - Define Pydantic models for capabilities (`MultiTickCapability`, `CooldownCapability`, etc.)
   - Add `capabilities` field to `AffordanceConfig`
   - Validation: Conflicting capabilities (e.g., `instant` + `multi_tick`)

2. **Effect Pipeline Schema** (2h)
   - Define `EffectPipeline` model with stages (`on_start`, `per_tick`, `on_completion`, etc.)
   - Replace current `effects`, `effects_per_tick`, `completion_bonus` with pipeline
   - Backward compatibility: Auto-convert old format to new

3. **Capability Handlers** (4-6h)
   - Implement `MultiTickHandler` (extends current multi-tick logic)
   - Implement `CooldownHandler` (track cooldowns per agent)
   - Implement `MeterGatedHandler` (check thresholds before interaction)
   - Implement `SkillBasedEffectivenessHandler` (scale effects by meter)

**Deliverable**: Capabilities work in isolation (unit tests)

---

### Phase 2: Effect Pipeline (6-8 hours)

**Goal**: Multi-stage effect application with lifecycle hooks

**Tasks**:
1. **Pipeline Executor** (3h)
   - Refactor `_handle_interactions()` to use pipeline stages
   - Apply `on_start` effects when interaction begins
   - Apply `per_tick` effects each step
   - Apply `on_completion` effects when done
   - Apply `on_early_exit` effects if agent leaves early

2. **Early Exit Mechanics** (2h)
   - Track "in_progress" affordances per agent
   - Detect when agent leaves (position change or WAIT action)
   - Apply `on_early_exit` effects
   - Reset progress or keep partial (based on capability)

3. **Conditional Effects** (3h)
   - Support `condition` field in effects (meter thresholds, probabilistic)
   - Evaluate conditions before applying effect
   - Handle `on_failure` stage for probabilistic interactions

**Deliverable**: Multi-tick interactions support early exit + conditional effects

---

### Phase 3: Advanced Patterns (2-4 hours)

**Goal**: Cooldowns, prerequisites, probabilistic success

**Tasks**:
1. **Cooldown System** (1-2h)
   - Track cooldown timers per agent per affordance
   - Decrement timers each step
   - Mask INTERACT action if on cooldown
   - Display cooldown in action masks

2. **Prerequisite System** (1-2h)
   - Support `sequence_gated` capability (affordance completion tracking)
   - Track completed affordances per agent (persistent state)
   - Check prerequisites before allowing interaction
   - Display locked affordances in action masks

**Deliverable**: Job has cooldown, University has prerequisite chain

---

### Phase 4: Occupancy & Queues (Deferred to L5+)

**Goal**: Single-occupancy + queue system (complex state management)

**Why Deferred**:
- Requires global affordance state (which agents are using)
- Queue system needs FIFO data structure
- Adds complexity to checkpointing (must save queue state)
- Not critical for L0-L3 pedagogical goals

**Future Work**: Implement when multi-agent competition becomes focus (L5).

---

## 8. Validation Rules (TASK-004 Compiler)

Following TASK-003 design principles, validation occurs in two stages: **structural validation** (syntax) followed by **semantic validation** (cross-references).

### Structural Validation (Syntax)

Validate capability structure before checking cross-references:

```python
# Validate affordance capabilities syntax
KNOWN_CAPABILITY_TYPES = {
    "multi_tick", "cooldown", "meter_gated", "skill_based_effectiveness",
    "probabilistic_success", "one_time_use", "remote_access", "range_based",
    "occupancy_limited", "sequence_gated"
}

for affordance in affordances:
    for capability in affordance.capabilities:
        # Check capability type is recognized
        if capability.type not in KNOWN_CAPABILITY_TYPES:
            errors.add(f"Unknown capability type: {capability.type}")

        # Check required parameters for capability type
        if capability.type == "multi_tick" and "duration_ticks" not in capability:
            errors.add(f"multi_tick capability requires duration_ticks parameter")

        if capability.type == "cooldown" and "cooldown_ticks" not in capability:
            errors.add(f"cooldown capability requires cooldown_ticks parameter")

        if capability.type == "meter_gated" and "meter" not in capability:
            errors.add(f"meter_gated capability requires meter parameter")

        # Check parameter types
        if "duration_ticks" in capability and not isinstance(capability.duration_ticks, int):
            errors.add(f"duration_ticks must be integer, got {type(capability.duration_ticks)}")

        if "success_probability" in capability:
            if not isinstance(capability.success_probability, float):
                errors.add(f"success_probability must be float, got {type(capability.success_probability)}")
            if not 0.0 <= capability.success_probability <= 1.0:
                errors.add(f"success_probability must be in [0, 1], got {capability.success_probability}")

# Validate capability conflicts
for affordance in affordances:
    capability_types = {cap.type for cap in affordance.capabilities}

    # Mutually exclusive capabilities
    if "instant" in capability_types and "multi_tick" in capability_types:
        errors.add(f"Affordance {affordance.id}: Cannot have both instant and multi_tick")

    if "one_time_use" in capability_types and "multi_tick" in capability_types:
        errors.add(f"Affordance {affordance.id}: one_time_use must be instant (no multi_tick)")

    # Dependent capabilities
    if "resumable" in capability_types and "multi_tick" not in capability_types:
        errors.add(f"Affordance {affordance.id}: resumable requires multi_tick capability")
```

### Semantic Validation (Cross-References)

After structure is valid, validate cross-references to other config files:

```python
# Validate meter references in costs/effects
valid_meters = {bar.name for bar in bars_config.bars}

for affordance in affordances:
    # Validate costs
    for cost in affordance.costs:
        if cost.meter not in valid_meters:
            errors.add(f"Affordance {affordance.id}: Unknown meter '{cost.meter}' in costs")

    # Validate effects
    for effect in affordance.effects:
        if effect.meter not in valid_meters:
            errors.add(f"Affordance {affordance.id}: Unknown meter '{effect.meter}' in effects")

    # Validate capability meter references
    for capability in affordance.capabilities:
        if capability.type == "meter_gated" and capability.meter not in valid_meters:
            errors.add(f"Affordance {affordance.id}: Unknown meter '{capability.meter}' in meter_gated capability")

        if capability.type == "skill_based_effectiveness" and capability.skill_meter not in valid_meters:
            errors.add(f"Affordance {affordance.id}: Unknown meter '{capability.skill_meter}' in skill_based_effectiveness capability")

# Validate prerequisite chains
for affordance in affordances:
    for capability in affordance.capabilities:
        if capability.type == "sequence_gated":
            # Check prerequisite affordance exists
            prereq_id = capability.prerequisite_affordance
            if prereq_id not in {a.id for a in affordances}:
                errors.add(f"Affordance {affordance.id}: Unknown prerequisite affordance '{prereq_id}'")

            # Check for circular dependencies
            # (Requires graph traversal - see TASK-004 for full implementation)

# Validate operating hours
for affordance in affordances:
    if affordance.operating_hours:
        open_hour, close_hour = affordance.operating_hours
        if not 0 <= open_hour <= 23:
            errors.add(f"Affordance {affordance.id}: open_hour must be in [0, 23], got {open_hour}")
        if not 1 <= close_hour <= 28:  # 28 = wraparound format
            errors.add(f"Affordance {affordance.id}: close_hour must be in [1, 28], got {close_hour}")
```

### Permissive Semantics Examples

Following TASK-003 "permissive semantics" principle, allow edge cases that are semantically valid:

**Allow edge cases**:
- ✅ `costs: []` - Free interaction (no resource cost)
- ✅ `effects: []` - Purely informational interaction (no meter changes)
- ✅ `duration_ticks: 1` - Multi-tick with duration=1 behaves like instant
- ✅ `cooldown_ticks: 0` - No cooldown (can use immediately)
- ✅ `success_probability: 1.0` - Deterministic (always succeeds)
- ✅ `skill_scaling: {min_effectiveness: 1.0, max_effectiveness: 1.0}` - No scaling (constant)
- ✅ `operating_hours: [0, 24]` - Always open (24/7 availability)

**Reject only true errors**:
- ❌ `duration_ticks: -5` - Negative duration is nonsensical
- ❌ `success_probability: 1.5` - Probability > 1 is invalid
- ❌ `meter: "nonexistent"` - Dangling reference
- ❌ `cooldown_ticks: "orange"` - Type violation
- ❌ `delta: [1]` - Incomplete movement vector for grid2d topology

### Effect Pipeline Consistency Checks

Validate that effect pipelines are consistent with capabilities:

```python
for affordance in affordances:
    capability_types = {cap.type for cap in affordance.capabilities}

    # If multi_tick capability, must have per_tick or on_completion effects
    if "multi_tick" in capability_types:
        has_multi_tick_effects = (
            bool(affordance.effects.per_tick) or
            bool(affordance.effects.on_completion)
        )
        if not has_multi_tick_effects:
            warnings.add(f"Affordance {affordance.id}: multi_tick capability but no per_tick/on_completion effects")

    # If early_exit_allowed, should have on_early_exit effects
    for cap in affordance.capabilities:
        if cap.type == "multi_tick" and cap.early_exit_allowed:
            if not affordance.effects.on_early_exit:
                warnings.add(f"Affordance {affordance.id}: early_exit_allowed but no on_early_exit effects")

    # If probabilistic_success, must have on_failure effects
    if "probabilistic_success" in capability_types:
        if not affordance.effects.on_failure:
            errors.add(f"Affordance {affordance.id}: probabilistic_success requires on_failure effects")
```

---

## 9. Backward Compatibility

### Legacy Config Support

**Current Configs** (no capabilities):
```yaml
- id: "0"
  name: "Bed"
  interaction_type: "dual"
  costs: [{meter: money, amount: 0.05}]
  effects: [{meter: energy, amount: 0.50}]
  required_ticks: 5
  costs_per_tick: [{meter: money, amount: 0.01}]
  effects_per_tick: [{meter: energy, amount: 0.075}]
  completion_bonus: [{meter: energy, amount: 0.125}]
  operating_hours: [0, 24]
```

**Auto-Conversion** (during load):
```yaml
# Converted to:
- id: "0"
  name: "Bed"
  capabilities:
    multi_tick:
      duration_ticks: 5  # From required_ticks
  effects:
    instant:  # From costs + effects (dual mode)
      - {meter: money, amount: -0.05}
      - {meter: energy, amount: 0.50}
    per_tick:  # From costs_per_tick + effects_per_tick
      - {meter: money, amount: -0.01}
      - {meter: energy, amount: 0.075}
    on_completion:  # From completion_bonus
      - {meter: energy, amount: 0.125}
  operating_hours: [0, 24]
```

**Conversion Logic**:
1. If `interaction_type` present (no `capabilities`) → Legacy mode
2. Convert `required_ticks` → `multi_tick.duration_ticks`
3. Convert `costs` → `effects.instant` (negate amounts)
4. Convert `effects` → `effects.instant`
5. Convert `costs_per_tick` + `effects_per_tick` → `effects.per_tick`
6. Convert `completion_bonus` → `effects.on_completion`

**Validation**: Warn if both `interaction_type` and `capabilities` present (conflicting formats).

---

## 10. Future Extensibility

### Adding New Capabilities (No Code Changes)

**Example: Add `weather_conditional` capability**

1. **Define Capability Schema** (YAML):
   ```yaml
   # In future: capabilities.yaml
   capabilities:
     - id: "weather_conditional"
       description: "Affordance availability depends on weather"
       parameters:
         - name: "allowed_weather"
           type: "list[string]"
           description: "Weather types when available (sunny, rainy, snowy)"
         - name: "blocked_weather"
           type: "list[string]"
           description: "Weather types when blocked"
   ```

2. **Implement Handler** (Python):
   ```python
   class WeatherConditionalHandler(CapabilityHandler):
       def check_availability(self, env_state, capability_config):
           current_weather = env_state.get_weather()
           if current_weather in capability_config.blocked_weather:
               return False
           if capability_config.allowed_weather:
               return current_weather in capability_config.allowed_weather
           return True
   ```

3. **Register Handler** (Registration):
   ```python
   CAPABILITY_REGISTRY = {
       "multi_tick": MultiTickHandler,
       "cooldown": CooldownHandler,
       "weather_conditional": WeatherConditionalHandler,  # NEW
   }
   ```

4. **Use in Configs**:
   ```yaml
   - id: "20"
     name: "Park"
     capabilities:
       weather_conditional:
         blocked_weather: ["rainy", "snowy"]
     effects:
       instant:
         - {meter: mood, amount: 0.30}
   ```

**Extensibility Pattern**:
- New capability = New schema + Handler class + Registration
- No changes to affordance configs (just add capability block)
- Compiler validates capability exists in registry

---

### Adding New Effect Stages

**Example: Add `on_interrupt` stage (for resumable interactions)**

1. **Extend Effect Pipeline Schema**:
   ```python
   class EffectPipeline(BaseModel):
       instant: list[AffordanceEffect] = []
       on_start: list[AffordanceEffect] = []
       per_tick: list[AffordanceEffect] = []
       on_completion: list[AffordanceEffect] = []
       on_early_exit: list[AffordanceEffect] = []
       on_failure: list[AffordanceEffect] = []
       on_interrupt: list[AffordanceEffect] = []  # NEW
   ```

2. **Implement Handler Logic**:
   ```python
   def handle_interrupt(self, agent_idx, affordance_name):
       pipeline = self.get_pipeline(affordance_name)
       if pipeline.on_interrupt:
           self.apply_effects(agent_idx, pipeline.on_interrupt)
   ```

3. **Use in Configs**:
   ```yaml
   - id: "30"
     name: "University"
     capabilities:
       multi_tick: {duration_ticks: 20, resumable: true}
     effects:
       per_tick:
         - {meter: money, amount: -0.05}
       on_interrupt:  # NEW: Applied when paused
         - {meter: mood, amount: -0.02}  # Slight stress from interruption
       on_completion:
         - {meter: fitness, amount: 0.50}
   ```

---

## 11. Conclusion

### Recommended Approach Summary

**Hybrid: Capability Composition + Effect Pipeline**

**Why**:
- ✅ **Balances expressivity and learnability**: Simple cases stay simple, complex cases possible
- ✅ **Extensible without code changes**: New capabilities = new YAML schema + handler
- ✅ **Backward compatible**: Legacy configs auto-convert
- ✅ **Composable**: Mix capabilities to create rich interactions
- ✅ **Self-documenting**: Capabilities + pipeline show clear behavior
- ✅ **Pattern consistency with actions**: Follows TASK-003 design principles (conceptual agnosticism, permissive semantics, structural enforcement)

**What It Enables**:
- Skill-based interactions (gym effectiveness depends on fitness)
- Cooldown mechanics (prevent spamming)
- Early exit rewards (quit job early, keep partial pay)
- Prerequisite chains (university progression)
- Probabilistic outcomes (gambling, risky investments)
- Meter-gated access (need energy to work)
- Multi-stage effects (different effects per tick)

**What It Doesn't Enable (Yet)**:
- Occupancy/queues (deferred to L5)
- Inventory/item system (separate feature)
- Weather/environment conditionals (needs environment state)
- Social interactions (multi-agent mechanics, L6)

**Effort**: 16-22 hours (3 phases)

**Impact**: Unlocks 90%+ of pedagogical interaction patterns students want to explore.

---

### Next Steps

1. **Review with User**: Confirm approach aligns with "flexible without API" goal
2. **Create TASK Document**: Detailed implementation plan with code examples
3. **Phase 1 Implementation**: Capability infrastructure + core capabilities
4. **Phase 2 Implementation**: Effect pipeline + early exit mechanics
5. **Phase 3 Implementation**: Cooldowns + prerequisites
6. **Testing**: Unit tests for each capability, integration tests for composition
7. **Documentation**: Update CLAUDE.md with capability usage examples
8. **Migration Guide**: Convert existing L0-L3 configs to new system (optional)

---

### Open Questions for User

1. **Scope**: Is occupancy/queue system needed for L4, or defer to L5?
2. **Defaults**: Should missing capabilities default to "off" or "infer from old fields"?
3. **Validation Strictness**: Error on conflicting capabilities, or warning + auto-fix?
4. **Probabilistic RNG**: Use environment RNG (reproducible) or separate seed?
5. **Cooldown State**: Store in environment state (checkpointed) or separate registry?

---

**End of Research Document**

**Total Lines**: ~1200
**Estimated Reading Time**: 25-30 minutes
**Research Time**: 4 hours
