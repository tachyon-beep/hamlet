# Research: Environment Configuration Gaps - True Gap Analysis

**Status**: Draft
**Created**: 2025-11-04
**Last Updated**: 2025-11-04
**Effort Estimate**: 10-14 hours

## Executive Summary

This document identifies **configuration gaps** - features that should be expressible in YAML configs but currently require code changes.

**Finding**: 2 high-priority gaps + 1 low-priority gap identified.

**Corrected Understanding**:
- ❌ Agent lifecycle is NOT a gap (bars with base_depletion handle this)
- ❌ Meter bounds is NOT a gap (TASK-001 adds range: [min, max])
- ❌ Temporal constants is NOT a gap (time is just a bar)
- ✅ Affordance masking based on bar values IS a gap (operating hours, mode switching)
- ✅ Multi-meter action costs IS a gap (actions need affordance-like costs pattern)
- ✅ RND architecture IS a gap (low priority)

---

## 0. What's Already Covered

Before identifying gaps, here's what existing UAC systems handle:

| Feature | Covered By | How It Works |
|---------|------------|--------------|
| Agent lifespan/aging | bars.yaml | Define bar with positive base_depletion (increments per step) |
| Meter ranges (debt, overbuffing) | TASK-001 | bars.yaml supports `range: [-100, 100]` per meter |
| Time of day | bars.yaml | Time is a bar with special semantics |
| Action space | TASK-003 | actions.yaml defines movement, costs |
| Spatial topology | TASK-000 | substrate.yaml defines 2D/3D/hex/graph/aspatial |
| Observation normalization | TASK-000/substrate.yaml | Substrate handles position encoding |

**Key Insight**: The UAC system already handles most configuration needs. What remains are **behavioral constraints** and **multi-meter effects**.

---

## 1. Real Gaps (Configuration Needs Code Changes)

### Gap 1: Affordance Masking Based on Bar Values (HIGH PRIORITY)

**Problem**: Cannot conditionally enable/disable affordances based on bar state without code changes.

**Use Cases**:
1. **Operating hours**: Job open only when time ∈ [0.375, 0.708] (9am-5pm in 24h cycle)
2. **Mode switching**: Location is "Coffee Shop" when time ∈ [0.25, 0.75], "Bar" otherwise
3. **Stamina gates**: Gym only usable when fitness > 0.3
4. **Wealth gates**: Restaurant only affordable when money > 0.5

**Current Workaround**: Hardcode temporal logic in vectorized_env.py (lines 465-485 for operating hours)

**Proposed Solution**:

```yaml
# affordances.yaml
affordances:
  - id: "Job"
    name: "Job"
    position: [2, 3]
    # NEW: Availability conditions
    availability:
      - meter: "time_of_day"
        min: 0.375  # 9am in [0, 1] normalized
        max: 0.708  # 5pm
        description: "Open 9am-5pm"

    # NEW: Mode switching
    modes:
      default:
        effects: [{meter: "money", amount: 0.25}]

      overtime:  # Active when time > 0.75 (after 6pm)
        condition:
          meter: "time_of_day"
          min: 0.75
        effects: [{meter: "money", amount: 0.375}, {meter: "energy", amount: -0.05}]
```

**Implementation Approach**:
1. Add `availability` field to AffordanceConfig (list of bar constraints)
2. Add `modes` field for state-dependent behavior
3. AffordanceEngine evaluates constraints before allowing interaction
4. Observation encoding includes "available affordances" mask

**Effort**: 6-8 hours
- Schema updates: 1h
- AffordanceEngine logic: 3-4h
- Observation encoding: 1-2h
- Testing: 1-2h

**Pedagogical Value**: HIGH
- Teaches temporal planning (agents learn when to visit Job)
- Demonstrates emergent scheduling behavior
- Enables realistic operating hours without hardcoding

---

### Gap 2: Multi-Meter Action Costs (HIGH PRIORITY)

**Problem**: Actions can only cost energy (via `energy_cost`), but should support multi-meter costs/effects like affordances do.

**Current State** (TASK-003 lines 226-244):
```yaml
# actions.yaml (current schema)
actions:
  - id: 0
    name: "UP"
    type: "movement"
    delta: [0, -1]
    energy_cost: 0.005  # ❌ Only energy supported
    effects: [{meter: "cash", amount: -1.0}]  # ✅ Multi-meter effects supported
```

**Pattern Mismatch**:
- **Affordances**: `costs: [{meter, amount}, ...]` + `effects: [{meter, amount}, ...]`
- **Actions**: `energy_cost: float` (singular) + `effects: [{meter, amount}, ...]`

**Use Cases**:
1. **Movement costs hygiene**: Walking makes you sweaty
2. **Movement costs satiation**: Physical activity burns calories
3. **Wait action restores mood**: Resting is mentally restorative
4. **Interact action costs social**: Interacting with strangers is draining

**Proposed Solution**:
```yaml
# actions.yaml (NEW pattern - matches affordances)
actions:
  - id: 0
    name: "UP"
    type: "movement"
    delta: [0, -1]
    # NEW: Multi-meter costs (like affordances)
    costs:
      - {meter: "energy", amount: 0.005}
      - {meter: "hygiene", amount: 0.003}  # Movement makes you sweaty
      - {meter: "satiation", amount: 0.004}  # Activity burns calories

  - id: 5
    name: "WAIT"
    type: "passive"
    # NEW: Positive effects (restoration)
    effects:
      - {meter: "mood", amount: 0.02}  # Resting restores mood
      - {meter: "energy", amount: 0.01}  # Passive energy recovery
```

**Implementation Approach**:
1. Add `ActionCost` schema (same pattern as `AffordanceCost`)
2. Replace `energy_cost: float` with `costs: list[ActionCost]` in ActionConfig
3. Update VectorizedHamletEnv.step() to apply multi-meter costs
4. Backward compatibility: Support legacy `energy_cost` field (convert to costs list)

**Effort**: 4-6 hours
- Schema updates: 1h
- VectorizedHamletEnv.step() refactor: 2-3h
- Backward compatibility: 1h
- Testing: 1h

**Pedagogical Value**: HIGH
- Teaches opportunity costs (movement isn't free)
- Demonstrates resource tradeoffs
- More realistic simulation (activity affects multiple systems)

**Key Insight**: Actions and affordances should follow the **same pattern** for costs/effects. Current asymmetry is technical debt.

---

### Gap 3: RND Network Architecture (LOW PRIORITY - DEFER)

**Problem**: RND (Random Network Distillation) network architecture is hardcoded, preventing experimentation.

**Current State**:
```python
# src/townlet/exploration/rnd.py:25,28
self.predictor = nn.Sequential(
    nn.Linear(obs_dim, 256),  # ❌ Hardcoded
    nn.ReLU(),
    nn.Linear(256, 256),  # ❌ Hardcoded
    nn.ReLU(),
    nn.Linear(256, embed_dim)
)
```

**Proposed Solution**:
```yaml
# training.yaml
exploration:
  rnd:
    hidden_layers: [256, 256]  # NEW: Configurable architecture
    activation: "relu"
```

**Effort**: 1-2 hours

**Pedagogical Value**: LOW
- Limited teaching value (students don't need to tune RND)
- More relevant for researchers than students

**Recommendation**: DEFER until after Gaps 1-2 complete.

---

## 2. Why Previous "Gaps" Weren't Actually Gaps

### Not a Gap: Agent Lifecycle/Aging

**Why Not**: bars.yaml already supports this pattern.

```yaml
# bars.yaml
bars:
  - name: "age"
    index: 8
    initial: 0.0
    base_depletion: 0.001  # Increments 0.001 per step (age increases)
    range: [0.0, 100.0]
    description: "Agent age in arbitrary units"

terminal_conditions:
  - meter: "age"
    operator: ">="
    value: 1.0
    description: "Death by old age at 1000 steps"
```

**Pattern**: Any bar with `base_depletion > 0` and no external modulation = incremental counter (age, time, etc.)

**Insight**: "Age is just a bar that increments per step rather than responding to sensor data."

---

### Not a Gap: Meter Bounds (Debt/Overbuffing)

**Why Not**: TASK-001 adds `range` field to bars.yaml.

```yaml
# bars.yaml (from TASK-001:223)
bars:
  - name: "money"
    range: [-100.0, 100.0]  # ✅ Allows debt
    initial: 0.0

  - name: "energy"
    range: [0.0, 1.5]  # ✅ Allows overbuffing
    initial: 1.0
```

**Already Planned**: TASK-001 implements variable meter ranges.

---

### Not a Gap: Temporal Constants

**Why Not**: Time is just a bar with special semantics.

```yaml
# bars.yaml
bars:
  - name: "time_of_day"
    index: 9
    initial: 0.0
    base_depletion: 0.04167  # Increments to 1.0 over 24 "hours"
    range: [0.0, 1.0]
    modulations:
      - type: "wrap"  # Resets to 0.0 when reaching 1.0
```

**Insight**: "Time is just a bar with a special purpose. The hours-per-day scaling is a visualization concern, not a config concern."

**Note**: The REAL gap is affordance masking based on time (Gap 1), not the time bar itself.

---

### Not a Gap: Observation Normalization

**Why Not**: Substrate type determines position encoding.

From TASK-000 (substrate.yaml), position encoding is substrate-specific:
- 2D grid → coordinate encoding `[x_norm, y_norm]`
- 3D grid → coordinate encoding `[x_norm, y_norm, z_norm]`
- Graph → one-hot node encoding
- Aspatial → no position encoding

**Already Planned**: TASK-000 handles this in substrate.yaml.

---

## 3. Implementation Priority

| Gap | Priority | Effort | Pedagogical Value | Blocks |
|-----|----------|--------|-------------------|--------|
| **Gap 1**: Affordance Masking | HIGH | 6-8h | HIGH - temporal planning | None |
| **Gap 2**: Multi-Meter Actions | HIGH | 4-6h | HIGH - opportunity costs | None |
| **Gap 3**: RND Architecture | LOW | 1-2h | LOW - research tool | None |

**Total Effort**: 10-14 hours (HIGH priority only)

**Recommendation**: Implement Gap 1 and Gap 2 in parallel (independent). Defer Gap 3.

---

## 4. Coordination with Existing Tasks

| Gap | Related Task | Relationship |
|-----|--------------|--------------|
| Affordance Masking | TASK-002 (UAC Contracts) | Add availability field to AffordanceConfig schema |
| Affordance Masking | TASK-004 (Compiler) | Add validation for bar references in availability conditions |
| Multi-Meter Actions | TASK-003 (Action Space) | Extend actions.yaml schema with costs field |
| Multi-Meter Actions | TASK-004 (Compiler) | Validate meter references in action costs |

**Critical Path**: Both gaps should be implemented **during** their related tasks to avoid rework.

---

## 5. Success Criteria

### Gap 1: Affordance Masking

- [ ] Affordance can specify `availability` conditions (bar constraints)
- [ ] Affordance can define `modes` with state-dependent behavior
- [ ] AffordanceEngine evaluates conditions before allowing interaction
- [ ] Observation includes "available affordances" mask (for policy learning)
- [ ] Example: Job open only during time ∈ [0.375, 0.708]

### Gap 2: Multi-Meter Actions

- [ ] Actions can specify `costs: list[{meter, amount}]`
- [ ] Actions can specify `effects: list[{meter, amount}]` (already supported)
- [ ] VectorizedHamletEnv.step() applies multi-meter costs
- [ ] Backward compatibility: Legacy `energy_cost` field still works
- [ ] Example: Movement costs energy + hygiene + satiation

---

## 6. Validation Rules

### Affordance Masking (TASK-004)

```python
# Validate availability conditions reference valid meters
for aff in affordances:
    if aff.availability:
        for condition in aff.availability:
            if condition.meter not in valid_meters:
                errors.add(f"Affordance {aff.id}: Unknown meter in availability: {condition.meter}")
```

### Multi-Meter Actions (TASK-004)

```python
# Validate action costs/effects reference valid meters
for action in actions:
    for cost in action.costs:
        if cost.meter not in valid_meters:
            errors.add(f"Action {action.name}: Unknown meter in costs: {cost.meter}")

    for effect in action.effects:
        if effect.meter not in valid_meters:
            errors.add(f"Action {action.name}: Unknown meter in effects: {effect.meter}")
```

---

## 7. Backward Compatibility

### Affordance Masking

**New Field** (`availability` is optional):
```yaml
# Old configs (still work)
affordances:
  - id: "Job"
    # No availability field → always available

# New configs (with masking)
affordances:
  - id: "Job"
    availability:
      - {meter: "time_of_day", min: 0.375, max: 0.708}
```

### Multi-Meter Actions

**Backward Compatible** (support both formats):
```yaml
# Old format (legacy)
actions:
  - id: 0
    name: "UP"
    energy_cost: 0.005  # Converted to costs: [{meter: energy, amount: 0.005}]

# New format (preferred)
actions:
  - id: 0
    name: "UP"
    costs:
      - {meter: "energy", amount: 0.005}
      - {meter: "hygiene", amount: 0.003}
```

**Migration Strategy**: Schema accepts both `energy_cost` (deprecated) and `costs` (preferred). If both present, `costs` takes precedence.

---

## 8. Design Rationale: Why These Are True Gaps

### Gap 1: Affordance Masking

**Test**: "Can operators experiment with operating hours without code changes?"

**Current Answer**: NO
- Operating hours hardcoded in vectorized_env.py (lines 465-485)
- Adding new time-based mechanics requires Python changes
- Mode switching not possible without code

**Future Answer** (with Gap 1 fixed): YES
- Operating hours defined in affordances.yaml
- Mode switching via `modes` field
- Experimentation enabled

**Conclusion**: True gap - UAC principle violated.

---

### Gap 2: Multi-Meter Actions

**Test**: "Can operators make movement cost hygiene without code changes?"

**Current Answer**: NO
- Actions only support `energy_cost` (singular)
- Adding hygiene cost requires Python changes
- Pattern asymmetry: affordances support multi-meter costs, actions don't

**Future Answer** (with Gap 2 fixed): YES
- Actions support `costs: list[{meter, amount}]` like affordances
- Pattern consistency: costs/effects work the same everywhere
- Experimentation enabled

**Conclusion**: True gap - pattern asymmetry is technical debt.

---

### Gap 3: RND Architecture

**Test**: "Do students need to configure RND network architecture?"

**Current Answer**: NO
- RND is an exploration implementation detail
- Students learn RL concepts, not RND tuning
- Low pedagogical value

**Future Answer** (with Gap 3 fixed): MAYBE
- Researchers might want to experiment
- But not critical for teaching

**Conclusion**: True gap, but low priority. Defer.

---

## 9. Appendix: Full Constant Audit Summary

**Original Audit**: 47 constants found
**After UAC Understanding**: 45 already covered or will be covered by existing tasks
**True Gaps**: 2 configuration features + 1 low-priority feature

**Not Gaps** (covered by existing UAC):
- Agent lifespan → bars with positive base_depletion
- Meter bounds → TASK-001 (range field)
- Time constants → time is a bar
- Network architecture → TASK-005 (brain.yaml - future)
- Training hyperparameters → training.yaml (already exposed)
- Curriculum → training.yaml (already exposed)
- Observation normalization → TASK-000 (substrate.yaml)

**True Gaps** (need new config features):
1. Affordance masking based on bar values (operating hours, mode switching)
2. Multi-meter action costs (like affordances)
3. RND network architecture (low priority)

---

## 10. Example: Operating Hours Implementation

**Before** (hardcoded in Python):
```python
# src/townlet/environment/vectorized_env.py:465-485
if affordance.name == "Job":
    time_normalized = self.meters[:, 9]  # time_of_day
    hour = (time_normalized * 24) % 24
    if not (9 <= hour < 17):  # 9am-5pm
        return False  # Job closed
```

**After** (config-driven):
```yaml
# configs/L3_temporal_mechanics/affordances.yaml
affordances:
  - id: "Job"
    name: "Job"
    position: [2, 3]
    availability:
      - meter: "time_of_day"
        min: 0.375  # 9am
        max: 0.708  # 5pm
    effects: [{meter: "money", amount: 0.25}]
```

**Benefits**:
- Operators can change hours via YAML
- No Python code changes needed
- Multiple affordances with different hours
- Mode switching possible (overtime pay after 6pm)

---

## 11. Example: Multi-Meter Action Costs

**Before** (energy only):
```yaml
# configs/L1_full_observability/actions.yaml (current)
actions:
  - id: 0
    name: "UP"
    type: "movement"
    delta: [0, -1]
    energy_cost: 0.005  # Only energy
```

**After** (multi-meter costs):
```yaml
# configs/L1_full_observability/actions.yaml (NEW)
actions:
  - id: 0
    name: "UP"
    type: "movement"
    delta: [0, -1]
    costs:
      - {meter: "energy", amount: 0.005}
      - {meter: "hygiene", amount: 0.003}  # Movement makes you sweaty
      - {meter: "satiation", amount: 0.004}  # Activity burns calories
```

**Benefits**:
- Realistic resource tradeoffs
- Same pattern as affordances (consistency)
- Operators can experiment with costs

---

## Conclusion

**Original Gap Analysis**: Incorrectly identified 10 "gaps" (most were already covered by UAC)

**Corrected Gap Analysis**: 2 high-priority configuration features + 1 low-priority feature

**Impact**:
- Affordance masking enables temporal planning and operating hours
- Multi-meter actions enables realistic resource tradeoffs
- Both are natural extensions of existing UAC patterns

**Effort**: 10-14 hours (HIGH priority only)

**Priority**: Implement during TASK-002 (masking) and TASK-003 (actions) to avoid rework

**Pedagogical Value**: HIGH for both gaps (temporal planning + opportunity costs)

**Key Insight**: Most "constants" are already configurable via UAC patterns (bars with base_depletion, substrate.yaml, etc.). The true gaps are **behavioral constraints** (availability conditions) and **pattern consistency** (actions should work like affordances).

---

**End of Reoriented Gap Analysis**
