# TASK-004B: UAC Contracts - Capability System

**Status**: Planned
**Priority**: MEDIUM (can defer until L4+)
**Estimated Effort**: 8-12 hours
**Dependencies**: TASK-003 (Core DTOs)
**Enables**: Advanced curriculum levels (L4-L6)

---

## Problem Statement

Current affordance system is rigid:

- All interactions are instant (no multi-tick mechanics)
- No operating hours (Job always available, not just 9am-5pm)
- No cooldowns (can spam same affordance)
- No skill-based effectiveness (Gym works same at fitness=0.1 and fitness=0.9)
- No prerequisite chains (can't require A before B)
- Simple `effects` dict can't model staged rewards (entry cost, per-tick pay, completion bonus)

**Solution**: Extend AffordanceConfig from TASK-003 with a capability composition system that supports rich interaction patterns without requiring code changes.

---

## Guiding Principles (from TASK-003)

**Universe Compiler**: DTOs enforce schema at load time, catching errors before training starts.

**No-Defaults Principle**: All behavioral parameters must be explicit. See TASK-003 for full details on why this matters.

**Permissive Semantics, Strict Syntax**: Allow semantically valid edge cases (empty effects, duration=1) but reject syntax errors (negative duration, invalid meter references). See TASK-003 for full philosophy.

**Pattern Consistency with Actions**: Affordances follow same cost/effect patterns as Actions (TASK-002B). Both use `costs: [{meter, amount}]` and `effects: [{meter, amount}]` lists.

**Conceptual Agnosticism**: The universe compiler should be domain-agnostic. It validates structure (types, references), not semantics (whether a design is "sensible"). This system should work for villages, factories, trading bots, or any other universe.

---

## Capability System Overview

**Design Decision**: Use **capability composition** instead of rigid type system. Affordances can mix-and-match capabilities (multi_tick + cooldown + meter_gated) instead of being locked into predefined types.

### Quick Reference: Core Capabilities

| Capability | Key Parameters | Use Case Example |
|------------|----------------|------------------|
| `multi_tick` | `duration_ticks`, `early_exit_allowed`, `resumable` | Job (10 ticks), University (20 ticks) |
| `cooldown` | `cooldown_ticks`, `scope` | Prevent spamming (Job cooldown 50 ticks) |
| `meter_gated` | `meter`, `min`, `max` | Gym requires energy >0.2, Hospital if health <0.5 |
| `skill_scaling` | `skill`, `base_multiplier`, `max_multiplier` | Gym effectiveness scales with fitness |
| `probabilistic` | `success_probability` | Gambling (30% success), Dating (50% success) |
| `prerequisite` | `required_affordances` | University Sophomore requires Freshman completion |

---

## Extension 1: Affordance Masking Schema

**Gap Identified**: Current affordances are always available. No support for:

- Operating hours (Job 9am-5pm, Bar 6pm-2am)
- Resource gates (Gym requires energy > 0.3)
- Mode switching (Coffee Shop vs Bar, same location)

**Solution**: Add `availability` and `modes` fields to AffordanceConfig.

### DTOs for Availability Masking

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal

class BarConstraint(BaseModel):
    """Meter-based availability constraint."""
    meter: str  # Required: which meter to check
    min: float | None = None  # Optional: minimum value (inclusive)
    max: float | None = None  # Optional: maximum value (inclusive)

    @model_validator(mode="after")
    def validate_min_less_than_max(self) -> "BarConstraint":
        """Ensure min < max if both specified."""
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError(
                    f"min ({self.min}) must be < max ({self.max})"
                )
        return self

    @model_validator(mode="after")
    def validate_at_least_one_bound(self) -> "BarConstraint":
        """Ensure at least one bound is specified."""
        if self.min is None and self.max is None:
            raise ValueError(
                "At least one of 'min' or 'max' must be specified"
            )
        return self

class ModeConfig(BaseModel):
    """Mode-specific configuration (e.g., operating hours)."""
    hours: tuple[int, int] | None = None  # (start_hour, end_hour), e.g., (9, 17) for 9am-5pm
    effects: dict[str, float] | None = None  # Mode-specific effect overrides

    @field_validator("hours")
    @classmethod
    def validate_hour_range(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        """Validate hours are in valid range (0-23) and handle midnight wrap."""
        if v is None:
            return v

        start, end = v
        if not (0 <= start <= 23):
            raise ValueError(f"Start hour must be 0-23, got {start}")
        if not (0 <= end <= 23):
            raise ValueError(f"End hour must be 0-23, got {end}")

        # Note: start > end is ALLOWED (e.g., 18-2 for 6pm-2am, wraps midnight)
        return v
```

### Example Configs: Operating Hours & Resource Gates

```yaml
# Job affordance with operating hours (9am-5pm)
affordances:
  - id: "Job"
    name: "Office Job"
    effects: {money: 22.5}

    # ONLY available 9am-5pm
    modes:
      office_hours:
        hours: [9, 17]  # 9am-5pm

# Bar with resource gate + operating hours
affordances:
  - id: "Bar"
    name: "Night Bar"
    effects: {mood: 0.2, money: -10.0}

    # Requires mood > 0.2 (can't drink while depressed)
    availability:
      - {meter: mood, min: 0.2}

    # ONLY available 6pm-2am (wraps midnight)
    modes:
      night_hours:
        hours: [18, 2]  # 6pm-2am

# Gym with energy gate (requires energy > 0.3 to enter)
affordances:
  - id: "Gym"
    name: "Fitness Center"
    effects: {fitness: 0.1}

    availability:
      - {meter: energy, min: 0.3}  # Too tired → can't work out

# Hospital with health gate (only if sick)
affordances:
  - id: "Hospital"
    name: "Emergency Room"
    effects: {health: 0.3, money: -15.0}

    availability:
      - {meter: health, max: 0.5}  # Only available when health < 0.5
```

**Validation Rules** (implemented in TASK-004A compiler):

- Meter references in `availability` must exist in bars.yaml
- `min < max` enforced (already done in DTO)
- Hour ranges validated (0-23, wrapping allowed)

---

## Extension 2: Capability Composition System

**Gap Identified**: Interaction patterns are hardcoded (instant, multi-tick, cooldown). Cannot combine patterns (e.g., multi-tick + cooldown + meter-gated).

**Solution**: Replace single `type` field with composable `capabilities` list.

### DTOs for Capabilities

```python
from typing import Literal
from pydantic import BaseModel, Field

class MultiTickCapability(BaseModel):
    """Multi-tick interaction (takes N ticks to complete)."""
    type: Literal["multi_tick"]
    duration_ticks: int = Field(gt=0)  # Required: how many ticks to complete
    early_exit_allowed: bool = False  # Can agent leave before completion?
    resumable: bool = False  # Can agent resume if interrupted?

class CooldownCapability(BaseModel):
    """Cooldown period after interaction."""
    type: Literal["cooldown"]
    cooldown_ticks: int = Field(gt=0)  # Required: ticks before can interact again
    scope: Literal["agent", "global"] = "agent"  # Per-agent or global cooldown

class MeterGatedCapability(BaseModel):
    """Requires meter within range to interact."""
    type: Literal["meter_gated"]
    meter: str  # Required: which meter to check
    min: float | None = None  # Optional: minimum value
    max: float | None = None  # Optional: maximum value

    @model_validator(mode="after")
    def validate_at_least_one_bound(self) -> "MeterGatedCapability":
        """Ensure at least one bound is specified."""
        if self.min is None and self.max is None:
            raise ValueError(
                "At least one of 'min' or 'max' must be specified"
            )
        return self

class SkillScalingCapability(BaseModel):
    """Effect scales with skill level."""
    type: Literal["skill_scaling"]
    skill: str  # Required: which meter represents skill
    base_multiplier: float = 1.0  # Multiplier at skill=0
    max_multiplier: float = 2.0   # Multiplier at skill=1

    @model_validator(mode="after")
    def validate_multiplier_order(self) -> "SkillScalingCapability":
        """Ensure base_multiplier <= max_multiplier."""
        if self.base_multiplier > self.max_multiplier:
            raise ValueError(
                f"base_multiplier ({self.base_multiplier}) must be <= "
                f"max_multiplier ({self.max_multiplier})"
            )
        return self

class ProbabilisticCapability(BaseModel):
    """Probabilistic success/failure."""
    type: Literal["probabilistic"]
    success_probability: float = Field(ge=0.0, le=1.0)  # Required: 0.0-1.0

class PrerequisiteCapability(BaseModel):
    """Requires prior interaction completion."""
    type: Literal["prerequisite"]
    required_affordances: list[str]  # Required: affordance IDs that must be completed

    @field_validator("required_affordances")
    @classmethod
    def validate_nonempty(cls, v: list[str]) -> list[str]:
        """Ensure at least one prerequisite specified."""
        if not v:
            raise ValueError("required_affordances cannot be empty")
        return v

# Union type for capability composition
CapabilityConfig = (
    MultiTickCapability |
    CooldownCapability |
    MeterGatedCapability |
    SkillScalingCapability |
    ProbabilisticCapability |
    PrerequisiteCapability
)
```

### Example Configs: Capability Composition

```yaml
# Job: Multi-tick + Cooldown + Meter-gated
affordances:
  - id: "Job"
    name: "Office Job"

    capabilities:
      # Takes 10 ticks to complete
      - type: multi_tick
        duration_ticks: 10
        early_exit_allowed: true

      # 50-tick cooldown (can't work again immediately)
      - type: cooldown
        cooldown_ticks: 50
        scope: agent

      # Requires energy > 0.3 to start
      - type: meter_gated
        meter: energy
        min: 0.3

    # Effects defined in effect pipeline (see next section)

# Gym: Meter-gated + Skill Scaling
affordances:
  - id: "Gym"
    name: "Fitness Center"

    capabilities:
      # Requires energy > 0.2
      - type: meter_gated
        meter: energy
        min: 0.2

      # Effectiveness scales with fitness (1.0x → 2.0x)
      - type: skill_scaling
        skill: fitness
        base_multiplier: 1.0
        max_multiplier: 2.0

    effect_pipeline:
      on_completion:
        - {meter: fitness, amount: 0.1}  # Scaled by skill_scaling

# University: Prerequisite Chain
affordances:
  - id: "UniversityFreshman"
    name: "University (Year 1)"
    effect_pipeline:
      on_completion:
        - {meter: education, amount: 0.25}

  - id: "UniversitySophomore"
    name: "University (Year 2)"

    capabilities:
      # Requires completion of Year 1
      - type: prerequisite
        required_affordances: ["UniversityFreshman"]

    effect_pipeline:
      on_completion:
        - {meter: education, amount: 0.25}

# Gambling: Probabilistic + Cooldown
affordances:
  - id: "Casino"
    name: "Slot Machine"

    capabilities:
      # 30% success rate
      - type: probabilistic
        success_probability: 0.3

      # 20-tick cooldown (prevent spamming)
      - type: cooldown
        cooldown_ticks: 20
        scope: agent

    effect_pipeline:
      on_start:
        - {meter: money, amount: -5.0}  # Entry cost (always paid)

      on_completion:  # SUCCESS path
        - {meter: money, amount: 20.0}  # Jackpot!
        - {meter: mood, amount: 0.2}    # Happy

      on_failure:  # FAILURE path (70% of the time)
        - {meter: mood, amount: -0.1}   # Disappointed
```

**Validation Rules** (implemented in TASK-004A compiler):

- **Capability conflicts**: Certain capabilities are mutually exclusive (e.g., instant + multi_tick)
- **Dependent capabilities**: `resumable` requires `multi_tick`
- **Meter references**: All meter names must exist in bars.yaml
- **Prerequisite references**: All affordance IDs must exist in affordances.yaml

---

## Extension 3: Effect Pipeline System

**Gap Identified**: Current affordances have simple `effects` dict. Cannot model:

- On-start costs (Job entry fee)
- Per-tick incremental rewards (Job pays per hour)
- Completion bonuses (Job completion bonus)
- Early-exit penalties (quit Job early → mood penalty)
- Failure effects (probabilistic failure → different outcome)

**Solution**: Replace `effects` dict with multi-stage `effect_pipeline`.

### DTOs for Effect Pipeline

```python
from pydantic import BaseModel, Field

class AffordanceEffect(BaseModel):
    """Single effect on a meter."""
    meter: str  # Required: which meter to affect
    amount: float  # Required: how much to change (can be negative)

class EffectPipeline(BaseModel):
    """Multi-stage effect application."""
    on_start: list[AffordanceEffect] = Field(default_factory=list)  # When interaction begins
    per_tick: list[AffordanceEffect] = Field(default_factory=list)  # Every tick during interaction
    on_completion: list[AffordanceEffect] = Field(default_factory=list)  # When interaction completes
    on_early_exit: list[AffordanceEffect] = Field(default_factory=list)  # When agent exits early
    on_failure: list[AffordanceEffect] = Field(default_factory=list)  # When probabilistic interaction fails
```

### Extended AffordanceConfig Schema

**Base Schema (from TASK-003)**:

```python
class AffordanceConfig(BaseModel):
    id: str
    name: str
    effects: dict[str, float]  # Simple effects
```

**Extended Schema (TASK-004B)**:

```python
class AffordanceConfig(BaseModel):
    # Base fields (from TASK-003)
    id: str
    name: str

    # DEPRECATED: Simple effects (auto-migrated)
    effects: dict[str, float] | None = None

    # NEW: Capability composition
    capabilities: list[CapabilityConfig] = Field(default_factory=list)

    # NEW: Availability masking
    availability: list[BarConstraint] = Field(default_factory=list)

    # NEW: Mode switching
    modes: dict[str, ModeConfig] = Field(default_factory=dict)

    # NEW: Effect pipeline
    effect_pipeline: EffectPipeline | None = None

    @model_validator(mode="after")
    def migrate_effects_to_pipeline(self) -> "AffordanceConfig":
        """Auto-migrate legacy 'effects' to 'effect_pipeline.on_completion'."""
        if self.effects and not self.effect_pipeline:
            logger.warning(
                f"Affordance {self.id}: 'effects' field deprecated. "
                f"Use 'effect_pipeline.on_completion' instead."
            )
            # Auto-migrate for backward compatibility
            self.effect_pipeline = EffectPipeline(
                on_completion=[
                    AffordanceEffect(meter=meter, amount=amount)
                    for meter, amount in self.effects.items()
                ]
            )
        return self
```

### Example Configs: Complete Effect Pipelines

```yaml
# Job: Complete multi-stage interaction
affordances:
  - id: "Job"
    name: "Office Job"

    capabilities:
      - type: multi_tick
        duration_ticks: 10
        early_exit_allowed: true

      - type: cooldown
        cooldown_ticks: 50
        scope: agent

      - type: meter_gated
        meter: energy
        min: 0.3

    effect_pipeline:
      # On interaction start (immediate costs)
      on_start:
        - {meter: energy, amount: -0.05}  # Entry energy cost

      # Every tick during interaction (linear rewards)
      per_tick:
        - {meter: money, amount: 2.25}  # $2.25/tick × 10 ticks = $22.50
        - {meter: energy, amount: -0.01}  # Fatigue per tick

      # On successful completion (bonuses)
      on_completion:
        - {meter: money, amount: 5.0}   # Completion bonus ($5)
        - {meter: social, amount: 0.02}  # Coworker interaction

      # On early exit (penalties)
      on_early_exit:
        - {meter: mood, amount: -0.05}  # Quitting early feels bad

# University: Long multi-tick with resumability
affordances:
  - id: "UniversitySemester"
    name: "University Semester"

    capabilities:
      - type: multi_tick
        duration_ticks: 50  # Long interaction
        early_exit_allowed: true
        resumable: true  # Can resume later

      - type: meter_gated
        meter: energy
        min: 0.2

    effect_pipeline:
      on_start:
        - {meter: money, amount: -30.0}  # Tuition (upfront)

      per_tick:
        - {meter: energy, amount: -0.005}  # Study fatigue
        - {meter: education, amount: 0.002}  # Incremental learning

      on_completion:
        - {meter: education, amount: 0.1}  # Degree progress
        - {meter: social, amount: 0.05}     # Campus friends

      on_early_exit:
        - {meter: mood, amount: -0.1}  # Dropout guilt
        # Note: Keeps accumulated education (resumable)

# Gym: Skill-scaled effectiveness
affordances:
  - id: "Gym"
    name: "Fitness Center"

    capabilities:
      - type: meter_gated
        meter: energy
        min: 0.2

      - type: skill_scaling
        skill: fitness
        base_multiplier: 1.0  # Beginner: full effect
        max_multiplier: 2.0   # Expert: 2x effect

    effect_pipeline:
      on_start:
        - {meter: energy, amount: -0.1}  # Workout cost

      on_completion:
        - {meter: fitness, amount: 0.05}  # Scaled by skill_scaling (0.05 → 0.1)
        - {meter: health, amount: 0.02}   # Also scaled

# Restaurant: Mode-specific effects (breakfast vs dinner)
affordances:
  - id: "Restaurant"
    name: "Local Restaurant"

    modes:
      breakfast:
        hours: [6, 11]  # 6am-11am
        effects:
          satiation: 0.3
          energy: 0.1  # Morning boost
          money: -8.0

      dinner:
        hours: [17, 22]  # 5pm-10pm
        effects:
          satiation: 0.4
          mood: 0.05  # Nice dinner
          money: -15.0
```

**Validation Rules** (implemented in TASK-004A compiler):

- **Pipeline consistency**: `multi_tick` capability should have `per_tick` or `on_completion` effects
- **Meter references**: All effect meters must exist in bars.yaml
- **Mutual exclusivity**: If using `effect_pipeline`, `effects` field should be empty (or auto-migrated)
- **Probabilistic completeness**: If `probabilistic` capability, should have both `on_completion` and `on_failure` effects

---

## Backward Compatibility Strategy

To avoid breaking existing configs, implement auto-migration:

```python
class AffordanceConfig(BaseModel):
    # ... fields ...

    @model_validator(mode="after")
    def ensure_backward_compatibility(self) -> "AffordanceConfig":
        """Auto-migrate legacy configs to new schema."""

        # Migrate 'effects' → 'effect_pipeline.on_completion'
        if self.effects and not self.effect_pipeline:
            self.effect_pipeline = EffectPipeline(
                on_completion=[
                    AffordanceEffect(meter=m, amount=a)
                    for m, a in self.effects.items()
                ]
            )
            logger.warning(
                f"{self.id}: Auto-migrated 'effects' to 'effect_pipeline.on_completion'. "
                f"Consider updating config to use effect_pipeline directly."
            )

        return self
```

**Example auto-migration**:

```yaml
# Old config (still works)
affordances:
  - id: "Bed"
    effects: {energy: 0.2}

# Auto-migrated to:
affordances:
  - id: "Bed"
    effect_pipeline:
      on_completion: [{meter: energy, amount: 0.2}]
```

---

## Implementation Plan

### Phase 1: Capability DTOs (3-4 hours)

**Deliverable**: `src/townlet/config/capability_config.py`

**Tasks**:

1. Define 6 capability DTOs (MultiTickCapability, CooldownCapability, etc.)
2. Add Pydantic validation for each (ranges, required fields, constraints)
3. Create CapabilityConfig union type
4. Write unit tests for each capability DTO

**Example Test**:

```python
def test_multi_tick_capability_validation():
    # Valid
    cap = MultiTickCapability(
        type="multi_tick",
        duration_ticks=10,
        early_exit_allowed=True
    )
    assert cap.duration_ticks == 10

    # Invalid: negative duration
    with pytest.raises(ValidationError):
        MultiTickCapability(
            type="multi_tick",
            duration_ticks=-5
        )
```

**Validation Scope Note**: TASK-004B DTOs validate structure (types, ranges). Cross-file validation (meter references, capability conflicts) happens in TASK-004A. See "Validation" section below (lines 763-778) for scope boundaries.

---

### Phase 2: Effect Pipeline DTOs (2-3 hours)

**Deliverable**: `src/townlet/config/effect_pipeline.py`

**Tasks**:

1. Define AffordanceEffect DTO (meter, amount)
2. Define EffectPipeline DTO (5 lifecycle stages)
3. Add validation for meter references (deferred to TASK-004A)
4. Write unit tests for effect pipeline construction

**Example Test**:

```python
def test_effect_pipeline_construction():
    pipeline = EffectPipeline(
        on_start=[AffordanceEffect(meter="energy", amount=-0.05)],
        per_tick=[AffordanceEffect(meter="money", amount=2.25)],
        on_completion=[AffordanceEffect(meter="money", amount=5.0)]
    )

    assert len(pipeline.on_start) == 1
    assert len(pipeline.per_tick) == 1
    assert len(pipeline.on_completion) == 1
    assert pipeline.on_early_exit == []  # Default empty
```

---

### Phase 3: Availability/Mode DTOs (2-3 hours)

**Deliverable**: `src/townlet/config/affordance_masking.py`

**Tasks**:

1. Define BarConstraint DTO (meter, min, max)
2. Define ModeConfig DTO (hours, effects)
3. Add validation for hour ranges (0-23, wrapping allowed)
4. Write unit tests for availability masking

**Example Test**:

```python
def test_bar_constraint_validation():
    # Valid: min only
    BarConstraint(meter="energy", min=0.3)

    # Valid: max only
    BarConstraint(meter="health", max=0.5)

    # Valid: both
    BarConstraint(meter="mood", min=0.2, max=0.8)

    # Invalid: min >= max
    with pytest.raises(ValidationError):
        BarConstraint(meter="energy", min=0.8, max=0.3)

    # Invalid: neither min nor max
    with pytest.raises(ValidationError):
        BarConstraint(meter="energy")

def test_mode_config_hour_wrapping():
    # Normal hours (9am-5pm)
    mode = ModeConfig(hours=(9, 17))
    assert mode.hours == (9, 17)

    # Midnight wrap (6pm-2am)
    mode = ModeConfig(hours=(18, 2))
    assert mode.hours == (18, 2)  # Allowed!

    # Invalid: out of range
    with pytest.raises(ValidationError):
        ModeConfig(hours=(25, 30))
```

---

### Phase 4: Extended AffordanceConfig (1-2 hours)

**Deliverable**: Update `src/townlet/config/affordance_config.py`

**Tasks**:

1. Extend AffordanceConfig from TASK-003 with new fields
2. Add auto-migration from `effects` dict to `effect_pipeline`
3. Update all L0-L3 configs to use extended schema
4. Write integration tests loading full configs

**Example Test**:

```python
def test_affordance_config_auto_migration():
    # Old-style config
    config = AffordanceConfig(
        id="Bed",
        name="Bed",
        effects={"energy": 0.2}
    )

    # Auto-migrated
    assert config.effect_pipeline is not None
    assert len(config.effect_pipeline.on_completion) == 1
    assert config.effect_pipeline.on_completion[0].meter == "energy"
    assert config.effect_pipeline.on_completion[0].amount == 0.2

def test_affordance_config_full_capabilities():
    config = AffordanceConfig(
        id="Job",
        name="Office Job",
        capabilities=[
            MultiTickCapability(type="multi_tick", duration_ticks=10, early_exit_allowed=True),
            CooldownCapability(type="cooldown", cooldown_ticks=50, scope="agent"),
            MeterGatedCapability(type="meter_gated", meter="energy", min=0.3)
        ],
        effect_pipeline=EffectPipeline(
            on_start=[AffordanceEffect(meter="energy", amount=-0.05)],
            per_tick=[AffordanceEffect(meter="money", amount=2.25)],
            on_completion=[AffordanceEffect(meter="money", amount=5.0)]
        )
    )

    assert len(config.capabilities) == 3
    assert config.effect_pipeline.on_start[0].amount == -0.05
```

---

## Validation

**NOTE**: Structural validation (field types, ranges) happens in DTOs (this task). Cross-file validation (meter references, capability conflicts) happens in TASK-004A (Compiler).

**TASK-004B Validation Scope**:

- ✅ Field types correct (duration_ticks is int, not string)
- ✅ Numeric bounds (duration_ticks > 0, probability ∈ [0,1])
- ✅ Required fields present
- ✅ Constraint validation (min < max, at least one bound)

**TASK-004A Validation Scope**:

- Meter references valid (meter exists in bars.yaml)
- Capability conflicts (can't have instant + multi_tick)
- Effect pipeline consistency (multi_tick should have per_tick or on_completion)
- Prerequisite references (affordance IDs exist)

---

## Schema Validation Summary

**TASK-004B Deliverables** (DTOs):

- [x] `BarConstraint` DTO (meter-based availability)
- [x] `ModeConfig` DTO (operating hours, mode switching)
- [x] `MultiTickCapability`, `CooldownCapability`, `MeterGatedCapability`, `SkillScalingCapability`, `ProbabilisticCapability`, `PrerequisiteCapability` (6 capability DTOs)
- [x] `EffectPipeline` DTO (multi-stage effects)
- [x] Extended `AffordanceConfig` with `availability`, `modes`, `capabilities`, `effect_pipeline`
- [x] Auto-migration from legacy `effects` dict

**TASK-004A Deliverables** (Validation):

- [ ] Validate `availability` meter references (Stage 4)
- [ ] Validate capability conflicts (Stage 4)
- [ ] Validate capability meter references (Stage 3)
- [ ] Validate effect pipeline consistency (Stage 4)
- [ ] Validate prerequisite affordance references (Stage 3)

---

## Success Criteria

- [ ] All 6 capability DTOs defined with Pydantic validation
- [ ] EffectPipeline DTO supports 5 lifecycle stages (on_start, per_tick, on_completion, on_early_exit, on_failure)
- [ ] BarConstraint DTO supports meter-based availability
- [ ] ModeConfig DTO supports operating hours and mode-specific effects
- [ ] Extended AffordanceConfig includes capabilities, availability, modes, effect_pipeline
- [ ] Backward compatibility maintained (auto-migration from `effects` dict)
- [ ] Advanced L3+ configs (operating hours, multi-tick) load successfully
- [ ] Capability composition validated (no structural errors)
- [ ] Unit tests pass for all DTOs
- [ ] Integration tests load example configs with capabilities

---

## Estimated Effort: 8-12 hours

**Breakdown**:

- Phase 1 (Capability DTOs): 3-4h
- Phase 2 (Effect Pipeline): 2-3h
- Phase 3 (Availability/Modes): 2-3h
- Phase 4 (Extended AffordanceConfig): 1-2h

**Confidence**: Medium (complex system, but clear design from research)

---

## Dependencies

**Depends on**:

- TASK-003 (Core DTOs) - Extends AffordanceConfig from TASK-003

**Enables**:

- Advanced curriculum levels (L4-L6)
- TASK-004A (Compiler) - Optional capability validation (cross-file checks)

---

## Examples

### Example 1: Job with Multi-Tick + Cooldown + Meter-Gated

```yaml
affordances:
  - id: "Job"
    name: "Office Job"

    capabilities:
      # Takes 10 ticks to complete
      - type: multi_tick
        duration_ticks: 10
        early_exit_allowed: true

      # 50-tick cooldown
      - type: cooldown
        cooldown_ticks: 50
        scope: agent

      # Requires energy > 0.3
      - type: meter_gated
        meter: energy
        min: 0.3

    effect_pipeline:
      on_start:
        - {meter: energy, amount: -0.05}

      per_tick:
        - {meter: money, amount: 2.25}
        - {meter: energy, amount: -0.01}

      on_completion:
        - {meter: money, amount: 5.0}
        - {meter: social, amount: 0.02}

      on_early_exit:
        - {meter: mood, amount: -0.05}
```

**Benefits**:

- Multi-stage rewards (entry cost, per-tick pay, completion bonus)
- Can quit early (with mood penalty, but keep earned money)
- Can't spam (50-tick cooldown)
- Can't work when exhausted (energy gate)

---

### Example 2: Gym with Skill Scaling

```yaml
affordances:
  - id: "Gym"
    name: "Fitness Center"

    capabilities:
      # Requires energy > 0.2
      - type: meter_gated
        meter: energy
        min: 0.2

      # Effectiveness scales with fitness (1.0x → 2.0x)
      - type: skill_scaling
        skill: fitness
        base_multiplier: 1.0
        max_multiplier: 2.0

    effect_pipeline:
      on_start:
        - {meter: energy, amount: -0.1}

      on_completion:
        - {meter: fitness, amount: 0.05}  # 0.05 → 0.1 as fitness improves
        - {meter: health, amount: 0.02}   # Also scaled
```

**Benefits**:

- Beginner gains (fitness=0.1): 0.05 × 1.0 = 0.05
- Intermediate gains (fitness=0.5): 0.05 × 1.5 = 0.075
- Expert gains (fitness=0.9): 0.05 × 1.9 = 0.095
- Realistic skill learning curves

---

### Example 3: Restaurant with Operating Hours + Mode Switching

```yaml
affordances:
  - id: "Restaurant"
    name: "Local Restaurant"

    modes:
      breakfast:
        hours: [6, 11]  # 6am-11am
        effects:
          satiation: 0.3
          energy: 0.1  # Morning boost
          money: -8.0

      dinner:
        hours: [17, 22]  # 5pm-10pm
        effects:
          satiation: 0.4
          mood: 0.05  # Nice dinner
          money: -15.0
```

**Benefits**:

- Same location, different effects by time of day
- Breakfast: cheaper, energy boost
- Dinner: more filling, mood boost, more expensive

---

### Example 4: University with Prerequisite Chain

```yaml
affordances:
  - id: "UniversityFreshman"
    name: "University (Year 1)"
    effect_pipeline:
      on_completion:
        - {meter: education, amount: 0.25}

  - id: "UniversitySophomore"
    name: "University (Year 2)"

    capabilities:
      # Requires completion of Year 1
      - type: prerequisite
        required_affordances: ["UniversityFreshman"]

    effect_pipeline:
      on_completion:
        - {meter: education, amount: 0.25}

  - id: "UniversityJunior"
    name: "University (Year 3)"

    capabilities:
      # Requires completion of Year 2
      - type: prerequisite
        required_affordances: ["UniversitySophomore"]

    effect_pipeline:
      on_completion:
        - {meter: education, amount: 0.25}

  - id: "UniversitySenior"
    name: "University (Year 4)"

    capabilities:
      # Requires completion of Year 3
      - type: prerequisite
        required_affordances: ["UniversityJunior"]

    effect_pipeline:
      on_completion:
        - {meter: education, amount: 0.25}
```

**Benefits**:

- Enforces progression: must complete Year 1 before Year 2
- Prevents skipping (can't do Senior without Junior)
- Tracks completion state per agent

---

## Recommendation

Implement TASK-004B **after** TASK-003 and **after** core UAC system is stable (L0-L3 working with basic DTOs). Capability system enables advanced curriculum levels (L4+) but is not required for basic operation.

**Incremental Rollout**:

1. Implement DTOs (Phase 1-4)
2. Test with simple examples (Job with multi_tick only)
3. Add complexity gradually (multi_tick + cooldown, then + meter_gated)
4. Validate with advanced configs (operating hours, skill scaling)
5. Deploy to L4+ curriculum levels

---

**End of TASK-004B**
