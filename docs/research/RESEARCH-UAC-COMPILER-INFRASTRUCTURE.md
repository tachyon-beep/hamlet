# Research: UNIVERSE_AS_CODE Compiler Infrastructure

## Executive Summary

This research focuses on the **bones of the UAC system**: ensuring we expose maximum configurability, compile universes following best practices, and address architectural/performance considerations for scalability.

**KEY FINDINGS**:

1. **8-Bar Constraint** - Critical bottleneck limiting universe expressivity
2. **No Compilation Pipeline** - Configs loaded independently, no cross-validation
3. **Hardcoded Type Systems** - Interaction types, operator literals, meter names baked into code
4. **Performance Optimization Gaps** - Some pre-computation exists, but many opportunities missed
5. **Missing Knobs** - Many gameplay-critical values hardcoded in Python

**RECOMMENDATIONS**:

1. **Implement Variable-Size Meter System** (HIGH PRIORITY)
2. **Build Universe Compiler with 7-Stage Pipeline** (TASK-003)
3. **Make Type System Configurable** (MEDIUM PRIORITY)
4. **Add Compile-Time Optimization Pass** (PERFORMANCE)
5. **Expose All Hidden Knobs** (UAC COMPLETENESS)

---

## Part 1: Hardcoded Constraints (Missing Knobs)

### CRITICAL: 8-Bar Constraint

**Current State**:

```python
# cascade_config.py:70
if len(v) != 8:
    raise ValueError(f"Expected 8 bars, got {len(v)}")

# cascade_config.py:75
if indices != {0, 1, 2, 3, 4, 5, 6, 7}:
    raise ValueError(f"Bar indices must be 0-7, got {sorted(indices)}")
```

**Impact**:

- Cannot model universes with more/fewer meters
- Cannot add "spiritual_alignment", "reputation", "skill", etc.
- Cannot create simplified 4-meter tutorial universes
- Cannot scale to complex 16-meter sociological simulations

**Why It Exists**:
From UAC.md: "Those indices are wired everywhere (policy nets, replay buffers, cascade maths, affordance effects). Changing them casually will break everything. So we treat them as stable ABI."

**The Problem**: This is treating a **technical debt** as a **design constraint**.

**Research Questions**:

1. Can we make meter count configurable while preserving network compatibility?
2. Should observation dim be computed from meter count?
3. How do we handle checkpoints with different meter counts?
4. Can networks learn to handle variable-size meter spaces?

**Proposed Solution**: **Variable-Size Meter System**

```yaml
# bars.yaml
version: "2.0"
description: "Variable-size meter system"

metadata:
  meter_count: 12  # CONFIGURABLE! Not hardcoded to 8

bars:
  # Standard 8 meters
  - {name: "energy", index: 0, ...}
  - {name: "hygiene", index: 1, ...}
  ...
  - {name: "fitness", index: 7, ...}

  # NEW meters for complex universes
  - {name: "reputation", index: 8, tier: "secondary", ...}
  - {name: "skill", index: 9, tier: "secondary", ...}
  - {name: "spirituality", index: 10, tier: "secondary", ...}
  - {name: "technology", index: 11, tier: "resource", ...}
```

**Implementation Approach**:

1. **Config Layer**: Remove hardcoded 8-bar validation, use `len(bars)`

   ```python
   @field_validator("bars")
   def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
       meter_count = len(v)
       # Check indices are contiguous from 0
       indices = {bar.index for bar in v}
       if indices != set(range(meter_count)):
           raise ValueError(f"Indices must be 0 to {meter_count-1}, got {sorted(indices)}")
       return v
   ```

2. **Engine Layer**: Dynamically size tensors based on meter count

   ```python
   # BEFORE: Hardcoded
   meters = torch.zeros((num_agents, 8), device=device)

   # AFTER: Dynamic
   meter_count = len(config.bars.bars)
   meters = torch.zeros((num_agents, meter_count), device=device)
   ```

3. **Network Layer**: Compute observation dim from config

   ```python
   # BEFORE: Assumes 8 meters
   obs_dim = grid_size * grid_size + 8 + affordance_vocab + extras

   # AFTER: Read from config
   obs_dim = grid_size * grid_size + meter_count + affordance_vocab + extras
   ```

4. **Checkpoint Compatibility**: Store meter_count in checkpoint metadata

   ```python
   checkpoint = {
       "meter_count": 12,
       "network_state": ...,
       "meters": meters,  # [num_agents, 12]
   }
   ```

**Benefits**:

- ✅ Enable 4-meter tutorial universes (L0: just energy + health)
- ✅ Enable 16-meter complex universes (reputation, skill, spirituality)
- ✅ Pedagogical: Students learn meter systems are designable, not fixed
- ✅ Domain flexibility: Different universes need different meters

**Risks**:

- Network architecture depends on obs_dim (must rebuild on meter_count change)
- Checkpoints incompatible across different meter counts (expected)
- Cascade configurations must reference valid meter indices (already validated)

**Estimated Effort**: 12-16 hours (refactoring engine, validation, tests)

**Priority**: HIGH (unblocks entire design space)

---

### Hardcoded Meter Name-to-Index Mapping

**Current State**:

```python
# affordance_config.py:27
METER_NAME_TO_IDX: dict[str, int] = {
    "energy": 0,
    "hygiene": 1,
    "satiation": 2,
    "money": 3,
    "mood": 4,
    "social": 5,
    "health": 6,
    "fitness": 7,
}
```

**Impact**:

- Cannot add new meters without editing Python
- Cannot rename meters (e.g., "satiation" → "hunger")
- Meter names hardcoded in affordance validation

**Solution**: **Dynamic Meter Name Resolution**

```python
# Build from bars.yaml at load time
meter_name_to_idx = {bar.name: bar.index for bar in config.bars.bars}

# Use in validation
@model_validator(mode="after")
def validate_meter_name(self) -> "AffordanceEffect":
    if self.meter not in self.model_config['meter_names']:
        raise ValueError(f"Invalid meter: {self.meter}")
    return self
```

**Benefit**: Meter vocabulary defined by bars.yaml, not hardcoded in Python

---

### Hardcoded Range [0.0, 1.0]

**Current State**:

```python
# cascade_config.py:43
if v != (0.0, 1.0):
    raise ValueError(f"Range must be (0.0, 1.0), got {v}")
```

**Impact**:

- Cannot model debt (negative money)
- Cannot model overbuffing (health >1.0 temporarily)
- Forces all meters into same scale (pedagogically limiting)

**Rationale** (from UAC.md):
> "Ranges are fixed at [0.0, 1.0] and enforced by validation. Concepts such as debt should be modelled through affordance outcomes and cascade penalties rather than extending meter bounds. Normalisation keeps the observation space stable for policy learning."

**Counter-Argument**: This is **overly restrictive**. We can normalize for policy learning while allowing flexible ranges:

```yaml
bars:
  - name: "money"
    range: [-1.0, 2.0]  # Allow debt and wealth accumulation
    normalization: [0.0, 1.0]  # For policy network observation
```

**Solution**: **Configurable Ranges with Normalization**

```python
class BarConfig(BaseModel):
    range: tuple[float, float]  # Physical range (can be negative, >1.0)
    normalization: tuple[float, float] = (0.0, 1.0)  # For policy observation

    @field_validator("range")
    def validate_range(cls, v):
        min_val, max_val = v
        if min_val >= max_val:
            raise ValueError("range[0] must be < range[1]")
        return v
```

**Benefit**: Enables debt mechanics, overbuffing, accumulation without breaking policy learning

---

### Hardcoded Interaction Types

**Current State**:

```python
# affordance_config.py:77
interaction_type: Literal["instant", "multi_tick", "continuous", "dual"]
```

**Impact**:

- Cannot add new interaction types without Python changes
- Cannot experiment with "staged", "conditional", "reactive" interactions
- Type system is closed, not extensible

**Solution**: **Extensible Interaction Type Registry**

```yaml
# interaction_types.yaml (NEW)
version: "1.0"
interaction_types:
  - id: "instant"
    description: "Single-tick completion"
    requires_progress: false

  - id: "multi_tick"
    description: "Sustained commitment over N ticks"
    requires_progress: true
    requires_field: "required_ticks"

  - id: "continuous"
    description: "Active while agent remains"
    requires_progress: true

  - id: "dual"
    description: "Instant trigger + multi-tick continuation"
    requires_progress: true
    requires_field: "required_ticks"

  # NEW: Custom interaction types
  - id: "staged"
    description: "Multi-stage interaction with checkpoints"
    requires_progress: true
    requires_field: "stages"
```

**Benefit**: Operators can define custom interaction types for advanced gameplay

---

### Hardcoded Operating Hours Format

**Current State**:

```yaml
operating_hours: [9, 18]  # 9am-6pm
operating_hours: [18, 28]  # 6pm-4am (wraps midnight)
```

**Limitation**:

- Cannot specify multiple time windows (9-12, 14-18)
- Cannot specify days of week (weekends vs weekdays)
- Cannot specify seasonal changes

**Solution**: **Flexible Operating Hours Schema**

```yaml
# Simple (backward compatible)
operating_hours: [9, 18]

# Multiple windows
operating_hours:
  windows:
    - [9, 12]   # Morning shift
    - [14, 18]  # Afternoon shift

# Days of week
operating_hours:
  windows: [9, 18]
  days: [0, 1, 2, 3, 4]  # Monday-Friday

# Seasonal
operating_hours:
  summer: [6, 22]
  winter: [8, 18]
```

**Benefit**: Enables complex temporal mechanics (shift work, seasonal variations)

---

## Part 2: Compilation Pipeline Architecture

### CRITICAL: No Universe Compiler

**Current State**: Configs are loaded **independently** without cross-validation.

From code analysis:

- `load_bars_config()` loads bars.yaml
- `load_cascade_config()` loads cascades.yaml
- `load_affordance_config()` loads affordances.yaml
- No orchestration layer validates they're compatible

**Missing Validations**:

1. **Dangling References**: Cascade references non-existent meter
2. **Spatial Impossibility**: Too many affordances for grid size
3. **Economic Impossibility**: Costs exceed possible income
4. **Temporal Conflicts**: Affordance open hours exceed 28
5. **Cascade Circularity**: A → B → A death spiral

**Solution**: Implement TASK-003 (Universe Compilation Pipeline)

```python
class UniverseCompiler:
    """7-stage compilation pipeline with cross-validation."""

    def compile(self, config_dir: Path) -> CompiledUniverse:
        # Stage 1: Load and validate bars.yaml
        bars = self._load_bars(config_dir / "bars.yaml")

        # Stage 2: Load and validate substrate.yaml (TASK-000)
        substrate = self._load_substrate(config_dir / "substrate.yaml")

        # Stage 3: Load and validate actions.yaml (TASK-002)
        actions = self._load_actions(config_dir / "actions.yaml", substrate)

        # Stage 4: Load and validate cascades.yaml
        cascades = self._load_cascades(config_dir / "cascades.yaml", bars)

        # Stage 5: Load and validate affordances.yaml
        affordances = self._load_affordances(config_dir / "affordances.yaml", bars, actions)

        # Stage 6: Load training.yaml
        training = self._load_training(config_dir / "training.yaml", affordances)

        # Stage 7: Cross-validate entire universe
        self._validate_universe(bars, substrate, actions, cascades, affordances, training)

        return CompiledUniverse(
            bars=bars,
            substrate=substrate,
            actions=actions,
            cascades=cascades,
            affordances=affordances,
            training=training,
            metadata=self._compute_metadata(...)
        )
```

**Cross-Validation Examples**:

```python
def _validate_universe(self, ...):
    # 1. Spatial feasibility
    grid_cells = substrate.dimensions[0] * substrate.dimensions[1]
    required_cells = len(training.environment.enabled_affordances) + 1  # +1 for agent
    if required_cells > grid_cells:
        raise CompilationError(f"Grid too small: {grid_cells} cells, need {required_cells}")

    # 2. Dangling meter references
    meter_names = {bar.name for bar in bars.bars}
    for cascade in cascades.cascades:
        if cascade.source not in meter_names:
            raise CompilationError(f"Cascade references non-existent meter: {cascade.source}")

    # 3. Action compatibility with substrate
    if substrate.type == "aspatial" and any(a.type == "movement" for a in actions.actions):
        raise CompilationError("Aspatial substrate cannot have movement actions")

    # 4. Economic feasibility
    total_income = sum(aff.income_rate for aff in affordances if aff.category == "income")
    total_costs = sum(aff.cost_rate for aff in affordances)
    if total_income < total_costs:
        logger.warning("Economic imbalance: costs exceed income (poverty trap)")

    # 5. Operating hours validity
    for aff in affordances.affordances:
        open_hour, close_hour = aff.operating_hours
        if open_hour < 0 or open_hour > 23:
            raise CompilationError(f"{aff.name}: open_hour must be 0-23, got {open_hour}")
        if close_hour < 1 or close_hour > 28:
            raise CompilationError(f"{aff.name}: close_hour must be 1-28, got {close_hour}")
```

**Benefits**:

- ✅ Fail fast: Catch errors at load time, not during training
- ✅ Clear errors: "Grid has 9 cells but need 11 (10 affordances + 1 agent)"
- ✅ Cross-file safety: Validate meter references, spatial feasibility, economic balance
- ✅ Dependency ordering: Load configs in correct order (bars → actions → cascades → affordances)

**Estimated Effort**: 11-16 hours (TASK-003)

**Priority**: HIGH (foundational for UAC system integrity)

---

## Part 3: Performance and Scalability

### Current Performance Optimizations

**Good**: CascadeEngine does pre-computation

- Pre-builds lookup maps: `_bar_name_to_idx`, `_bar_idx_to_name`
- Pre-computes base depletion tensor: `_base_depletions`
- Pre-builds cascade data by category: `_cascade_data`
- Pre-builds modulation data: `_modulation_data`

**From code**:

```python
# cascade_engine.py:49-63
# Pre-build lookup maps for performance
self._bar_name_to_idx = {bar.name: bar.index for bar in config.bars.bars}
self._bar_idx_to_name = {bar.index: bar.name for bar in config.bars.bars}

# Pre-compute base depletion tensor [8]
self._base_depletions = self._build_base_depletion_tensor()

# Pre-build cascade tensors for efficient batch application
self._cascade_data = self._build_cascade_data()
```

**Impact**: Avoids per-step lookups and string comparisons during tick execution.

### Missing Performance Optimizations

**1. No Compiled Universe Artifact**

Currently, every training run:

1. Loads YAML files from disk
2. Parses YAML into Python dicts
3. Validates with Pydantic
4. Builds lookup tables
5. Pre-computes tensors

**Problem**: Steps 1-3 are **pure overhead** repeated every run.

**Solution**: **Compiled Universe Cache**

```python
# First run: Compile and cache
universe = UniverseCompiler.compile("configs/L1_full_observability")
universe.save_compiled("configs/L1_full_observability/.compiled")

# Subsequent runs: Load cached compiled universe
universe = CompiledUniverse.load("configs/L1_full_observability/.compiled")

# File format: MessagePack or Pickle for fast deserialization
```

**Benefit**: 10-50x faster startup for repeated training runs

---

**2. No Action Masking Pre-Computation**

Currently, action masks are computed **every tick** for every agent:

- Check operating hours for each affordance
- Check agent position matches affordance position
- Compute valid actions dynamically

**Problem**: Most of this is **static** (operating hours don't change mid-episode).

**Solution**: **Pre-Compute Static Action Masks**

```python
class AffordanceMaskTable:
    """Pre-computed action availability by hour."""

    def __init__(self, affordances: list[AffordanceConfig]):
        # Build 24-hour mask table [24, num_affordances]
        self.masks = torch.zeros((24, len(affordances)), dtype=torch.bool)

        for hour in range(24):
            for aff_idx, aff in enumerate(affordances):
                open_hour, close_hour = aff.operating_hours
                self.masks[hour, aff_idx] = self._is_open(hour, open_hour, close_hour)

    def get_mask(self, hour: int) -> torch.Tensor:
        """O(1) lookup instead of O(num_affordances) computation."""
        return self.masks[hour]
```

**Benefit**: O(1) mask lookup vs O(num_affordances) computation per tick

---

**3. No Cascade Pre-Sorting**

Cascades are applied in category order, but within each category they're unordered.

**Problem**: Might apply cascades in suboptimal order (cache misses, branch misprediction).

**Solution**: **Sort Cascades by Target Index**

```python
def _build_cascade_data(self):
    cascade_data = {}

    for cascade in self.config.cascades.cascades:
        category = cascade.category
        if category not in cascade_data:
            cascade_data[category] = []
        cascade_data[category].append({...})

    # OPTIMIZATION: Sort by target index for cache locality
    for category in cascade_data:
        cascade_data[category].sort(key=lambda c: c["target_idx"])

    return cascade_data
```

**Benefit**: Better CPU cache locality, fewer cache misses

---

**4. No Tensor Fusion**

Multiple small tensor operations instead of fused operations.

**Example**:

```python
# CURRENT: Multiple operations
meters -= base_depletions
meters -= cascade_penalties
meters = torch.clamp(meters, 0.0, 1.0)

# OPTIMIZED: Fused operation
meters = torch.clamp(meters - base_depletions - cascade_penalties, 0.0, 1.0)
```

**Benefit**: Fewer kernel launches on GPU, better memory bandwidth utilization

---

### Scalability Analysis

**Question**: Can UAC scale to larger universes?

**Current Bottlenecks**:

1. **Cascade Computation**: O(num_cascades × num_agents) per tick
   - With 20 cascades, 1000 agents: 20k operations per tick
   - GPU-friendly (fully parallel), scales well

2. **Action Masking**: O(num_affordances × num_agents) per tick
   - With 50 affordances, 1000 agents: 50k operations per tick
   - Can pre-compute (see optimization above)

3. **Affordance Interaction**: O(interacting_agents) per tick
   - Only agents on affordance tiles pay cost
   - Scales sublinearly (most agents not interacting)

4. **Terminal Condition Checking**: O(num_agents) per tick
   - Check pivotal meters for each agent
   - GPU-friendly, scales well

**Scaling Targets**:

| Metric | Current | Target | Feasible? |
|--------|---------|--------|-----------|
| Meters | 8 | 16 | ✅ (with variable-size refactor) |
| Affordances | 14 | 100 | ✅ (with action mask pre-compute) |
| Cascades | 11 | 50 | ✅ (GPU-parallel, already fast) |
| Agents | 1-16 | 1000 | ✅ (vectorized env, GPU-native) |
| Grid Size | 8×8 | 32×32 | ✅ (memory scales O(grid²), acceptable) |

**Conclusion**: UAC architecture is **fundamentally scalable**. Main blockers are hardcoded constraints (8-bar), not performance.

---

## Part 4: Exposed vs Hidden Knobs

### Currently Exposed (✅)

**bars.yaml**:

- Meter names, indices, tiers
- Initial values
- Base depletion rates
- Terminal conditions

**cascades.yaml**:

- Modulation parameters (base_multiplier, range)
- Cascade sources, targets, thresholds, strengths
- Execution order

**affordances.yaml**:

- Interaction types
- Required ticks
- Costs, effects, completion bonuses
- Operating hours

**training.yaml**:

- Grid size
- Device (CPU/CUDA)
- Enabled affordances
- Network type

### Currently Hidden in Python (❌)

**Environment Constants** (`vectorized_env.py`):

```python
# Hardcoded movement costs
move_energy_cost: float = 0.005
wait_energy_cost: float = 0.001
interact_energy_cost: float = 0.0

# Hardcoded grid boundaries
boundary_behavior = "clamp"  # Not configurable!

# Hardcoded distance metric
distance_metric = "manhattan"  # Not configurable!
```

**Cascade Engine Constants** (`cascade_engine.py`):

```python
# Curriculum difficulty multiplier (not in YAML)
depletion_multiplier: float = 1.0  # Should be in training.yaml

# Clamp ranges (hardcoded)
min_val = 0.0
max_val = 1.0
```

**Affordance Engine Constants** (`affordance_engine.py`):

```python
# Interaction progress thresholds
# Completion detection logic
# Early exit penalties
```

**Solution**: **Move ALL gameplay constants to YAML**

```yaml
# environment.yaml (NEW)
version: "1.0"

physics:
  movement_energy_cost: 0.005
  wait_energy_cost: 0.001
  interact_energy_cost: 0.0
  boundary_behavior: "clamp"  # clamp, wrap, bounce, fail
  distance_metric: "manhattan"  # manhattan, euclidean, chebyshev

constraints:
  meter_clamp_min: 0.0
  meter_clamp_max: 1.0

curriculum:
  depletion_multiplier: 1.0  # Curriculum difficulty scaling

interactions:
  allow_early_exit: true
  early_exit_penalty: 0.5  # Forfeit 50% of accumulated rewards
  completion_bonus_multiplier: 1.0
```

**Benefit**: Operators can tune **every** gameplay parameter without touching Python

---

## Part 5: Research Recommendations

### HIGH PRIORITY (Do Now)

**1. Variable-Size Meter System** (12-16 hours)

- Remove 8-bar hardcode constraint
- Enable 4-meter tutorials, 16-meter complex universes
- Compute observation dim dynamically
- Handle checkpoints with metadata

**2. Universe Compilation Pipeline** (TASK-003) (11-16 hours)

- Implement 7-stage compiler with cross-validation
- Catch dangling references, spatial impossibility, economic imbalance
- Fail fast at load time, not during training

**3. Expose All Hidden Knobs** (4-6 hours)

- Move environment constants to YAML
- Move cascade engine constants to YAML
- Move affordance engine constants to YAML
- Enable complete configuration without Python

**Total High Priority Effort**: 27-38 hours (3-5 days)

---

### MEDIUM PRIORITY (Do During Refactoring)

**4. Extensible Type System** (6-8 hours)

- Make interaction types configurable (not hardcoded literals)
- Make operator types configurable
- Enable custom interaction types

**5. Configurable Ranges** (4-6 hours)

- Remove [0.0, 1.0] hardcoding
- Support debt (negative money), overbuffing (health >1.0)
- Separate physical range from normalization range

**6. Flexible Operating Hours** (3-4 hours)

- Support multiple time windows
- Support days of week
- Support seasonal variations

**Total Medium Priority Effort**: 13-18 hours (2 days)

---

### PERFORMANCE (Optional Optimization)

**7. Compiled Universe Cache** (4-6 hours)

- Serialize compiled universe to disk
- 10-50x faster startup for repeated runs

**8. Action Mask Pre-Computation** (3-4 hours)

- Pre-compute 24-hour mask table
- O(1) lookup vs O(num_affordances) computation

**9. Cascade Pre-Sorting** (1-2 hours)

- Sort cascades by target index
- Better cache locality

**10. Tensor Fusion** (2-3 hours)

- Fuse multi-step tensor operations
- Fewer kernel launches on GPU

**Total Performance Effort**: 10-15 hours (1-2 days)

---

## Part 6: Architectural Best Practices

### Compiler Design Patterns

**1. Builder Pattern for Universe Construction**

```python
class UniverseBuilder:
    """Incremental universe construction with validation at each stage."""

    def __init__(self):
        self.bars = None
        self.substrate = None
        self.actions = None
        self.cascades = None
        self.affordances = None
        self.training = None

    def with_bars(self, bars: BarsConfig) -> "UniverseBuilder":
        self.bars = bars
        return self

    def with_substrate(self, substrate: SubstrateConfig) -> "UniverseBuilder":
        if self.bars is None:
            raise BuilderError("Must set bars before substrate")
        self.substrate = substrate
        return self

    def build(self) -> CompiledUniverse:
        if not all([self.bars, self.substrate, ...]):
            raise BuilderError("Incomplete universe")
        return CompiledUniverse(...)
```

**Benefit**: Enforces dependency ordering, makes construction explicit

---

**2. Validation Stratification**

Separate validation into layers:

```python
# Layer 1: Syntactic validation (Pydantic)
class BarConfig(BaseModel):
    name: str
    index: int = Field(ge=0)  # Syntax: must be non-negative

# Layer 2: Semantic validation (model_validator)
@model_validator(mode="after")
def validate_indices_contiguous(self):
    # Semantics: indices must be contiguous

# Layer 3: Cross-file validation (UniverseCompiler)
def validate_cascade_meter_references(cascades, bars):
    # Cross-file: cascades reference existing meters
```

**Benefit**: Clear separation of concerns, easier debugging

---

**3. Immutable Compiled Universe**

Once compiled, universe should be immutable:

```python
@dataclass(frozen=True)
class CompiledUniverse:
    bars: BarsConfig
    substrate: SubstrateConfig
    actions: ActionSpaceConfig
    cascades: CascadeConfig
    affordances: AffordanceConfig
    training: TrainingConfig
    metadata: UniverseMetadata

    def __post_init__(self):
        # Validate universe is complete and consistent
        self._validate()
```

**Benefit**: Prevents accidental mutation during training, enables caching

---

**4. Schema Versioning**

```yaml
# bars.yaml
version: "2.0"  # Breaking change: variable meter count
compatibility: "v2+"

# Old config (v1.0)
bars: [8 meters]

# New config (v2.0)
metadata:
  meter_count: 12
bars: [12 meters]
```

**Benefit**: Graceful migration, backward compatibility detection

---

## Part 7: Critical Questions

**1. Should we prioritize variable-size meters over compilation pipeline?**

**Answer**: YES. Variable-size meters unblock entire design space, while compilation pipeline "just" catches errors earlier. Do meters first, then compiler.

**2. Can we maintain checkpoint compatibility across meter counts?**

**Answer**: NO, but that's expected. Checkpoints should store meter_count in metadata and reject incompatible loads.

**3. Should we expose ALL constants or keep some hardcoded?**

**Answer**: Expose ALL constants that affect gameplay. Keep only true engine internals hidden (e.g., tensor dtypes, device management).

**4. What's the performance target for compilation?**

**Answer**: <100ms for typical universe. Current YAML loading is ~10-50ms, validation adds ~10-20ms. With caching, <1ms for subsequent loads.

**5. Should compilation be incremental or all-at-once?**

**Answer**: All-at-once for simplicity. Incremental compilation adds complexity without clear benefit (universes are small).

---

## Conclusion

**The UAC compiler infrastructure needs foundational work before we worry about reward models or economic tuning.**

**Key Priorities**:

1. **Variable-Size Meter System** - Unblocks expressivity
2. **Universe Compilation Pipeline** - Ensures correctness
3. **Expose All Hidden Knobs** - Achieves full UAC vision

**Once these are done, UAC will be**:

- ✅ Maximally configurable (all knobs exposed)
- ✅ Robust (compiler catches errors early)
- ✅ Scalable (supports 16 meters, 100 affordances, 1000 agents)
- ✅ Best-practice (immutable compiled universes, validation stratification)

**Estimated Total Effort**: 40-56 hours (5-7 days)

**Slogan**: "Build the platform right, then build the content."
