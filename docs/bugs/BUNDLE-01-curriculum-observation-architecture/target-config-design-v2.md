# Target Configuration Design v2
## BUNDLE-01: Experiment-Level Hierarchy (Revised)

**Status**: Design v2 - Addressing Code Review Feedback

**Core Principle**: WHAT exists (vocabulary, breaks checkpoints) vs HOW it behaves (parameters, curriculum-safe)

**Key Pattern**: **Support vs Active** - Experiment declares which fields CAN exist (support), curriculum declares which ARE active vs masked (active).

---

## Changes from v1

### Critical Fixes

1. **Vision Support/Active Pattern** (fixes Review Issue #1, #2)
   - Experiment: `vision_support: both|global|partial|none` (what fields exist)
   - Curriculum: `active_vision: global|partial` (which is active, which is masked)
   - Preserves BUG-43 masking while enabling power user optimization

2. **Temporal Support/Active Pattern** (fixes Review Issue #3)
   - Experiment: `temporal_support: enabled|disabled` (do temporal fields exist)
   - Curriculum: `active_temporal: true|false` (is temporal active this level)
   - L0/L1/L2 can have `active_temporal: false`, L3 has `active_temporal: true`

3. **Global Actions Vocabulary** (fixes Review Issue #4)
   - Added `actions.yaml` at experiment level
   - Defines action vocabulary (movement, custom actions, labels)

4. **Cascade Structure Split** (addresses Review Issue #5)
   - Experiment: Cascade graph structure (which→which relationships)
   - Curriculum: Cascade parameters (thresholds, strengths)

5. **Meter Ranges Split** (new insight from review)
   - Experiment: Meter vocabulary + semantic ranges (0-1 vs 0-100)
   - Curriculum: Initial values, depletion rates, terminal conditions

---

## Architecture Layers

### Layer 1: Experiment Metadata
**File**: `experiment.yaml`

```yaml
name: "Default Curriculum"
description: "Standard L0→L3 pedagogical progression"
author: "HAMLET Team"
version: "1.0"
```

**Breaking**: No (metadata only)

---

### Layer 2: Stratum (World Shape)
**File**: `stratum.yaml`

```yaml
version: "1.0"

substrate:
  type: grid  # grid, grid3d, gridnd, continuous, continuousnd, aspatial

  grid:
    topology: square
    width: 8
    height: 8

# Support vs Active Pattern: Which observation fields CAN exist
vision_support: both  # both, global, partial, none
  # both:    grid_encoding + local_window (one active, one masked per level)
  # global:  grid_encoding only (power user: no transfer)
  # partial: local_window only (power user: no transfer)
  # none:    neither (aspatial substrates)

temporal_support: enabled  # enabled, disabled
  # enabled:  time_sin, time_cos fields exist (can be masked per level)
  # disabled: no temporal fields at all
```

**Breaking**: YES
- Substrate type changes position_dim
- Grid dimensions change grid_encoding dims
- `vision_support` determines which spatial fields exist (obs_dim)
- `temporal_support` determines if temporal fields exist (obs_dim)

**Rationale**:
- Grid size must be consistent for transfer learning
- `vision_support: both` enables BUG-43 masking pattern
- `vision_support: global` optimizes obs_dim for single-level full-obs experiments

---

### Layer 3: Environment (World Vocabulary)
**File**: `environment.yaml`

```yaml
version: "1.0"

# Meter vocabulary (defines which obs_meter_* fields exist)
meters:
  - name: energy
    terminal: true
    range_type: normalized  # normalized (0-1), integer (0-100), unbounded
    description: "Can you move? Death if depleted."

  - name: health
    terminal: true
    range_type: normalized
    description: "Are you alive? Death if depleted."

  - name: satiation
    range_type: normalized
    description: "Hunger level. FUNDAMENTAL need."

  # ... all 8 meters

# Cascade graph structure (which meters cascade to which)
# Parameters (threshold, strength) defined per curriculum level
cascade_graph:
  - source: satiation
    target: health
    description: "Low hunger damages health"

  - source: satiation
    target: energy
    description: "Low hunger drains energy"

  - source: mood
    target: energy
    description: "Depression causes exhaustion"

  - source: hygiene
    target: satiation
    description: "Poor hygiene affects appetite"

  # ... all cascade relationships

# Modulation graph structure
modulation_graph:
  - source: fitness
    target: health
    type: depletion_multiplier
    description: "Low fitness increases health depletion"

# Affordance vocabulary (defines affordance_at_position encoding size)
affordances:
  - name: Bed
    category: energy_restoration

  - name: LuxuryBed
    category: energy_restoration

  # ... all 14 affordances

# VFS variable vocabulary (defines custom observation fields)
variables:
  - name: deficit_energy
    type: scalar
    dims: 1
    description: "How far below target energy (for shaping bonuses)"

  - name: rush_hour
    type: scalar
    dims: 1
    description: "True during commute hours (affects affordance availability)"

  # Note: Computation logic lives in VFS compiler, not here
  # This just declares "these fields exist in observations"
```

**Breaking**: YES
- Changing meter count/names changes obs field count
- Changing affordance count/names changes one-hot encoding size
- Changing VFS variables changes obs fields
- Cascade/modulation graph structure changes environment dynamics

**Compiler Validation**:
- All curriculum levels must have parameters for all meters in vocabulary
- All curriculum levels must have parameters for all affordances in vocabulary
- All curriculum levels must have parameters for all cascades in graph

---

### Layer 4: Actions (Action Vocabulary)
**File**: `actions.yaml`

```yaml
version: "1.0"

# Global action vocabulary shared across all curriculum levels
# Defines action space dimension (breaking change)

substrate_actions:
  # Automatically provided by substrate type
  # grid: [UP, DOWN, LEFT, RIGHT]
  # grid3d: [UP, DOWN, LEFT, RIGHT, ASCEND, DESCEND]
  # aspatial: []
  inherit: true

custom_actions:
  - name: INTERACT
    description: "Interact with affordance at current position"
    enabled_by_default: true

  - name: WAIT
    description: "Do nothing (only pay base_depletion)"
    enabled_by_default: true

  - name: REST
    description: "Recover energy (custom action for energy restoration)"
    enabled_by_default: false  # Must be explicitly enabled in curriculum

  - name: MEDITATE
    description: "Improve mood (custom action for mood boost)"
    enabled_by_default: false

# Action labels (UI/terminology)
labels:
  preset: gaming  # gaming, 6dof, cardinal, math
  # gaming:   UP, DOWN, LEFT, RIGHT
  # cardinal: NORTH, SOUTH, EAST, WEST
  # 6dof:     FORWARD, BACK, LEFT, RIGHT, UP, DOWN
  # math:     +X, -X, +Y, -Y
```

**Breaking**: YES
- Changing custom action count changes action_dim
- Action space dimension affects network output layer

**Curriculum Control**:
- `training.yaml: enabled_actions` can DISABLE actions (subset of vocabulary)
- Cannot ADD actions not in vocabulary (compiler error)

---

### Layer 5: Agent (Perception + Drive + Brain)
**File**: `agent.yaml`

```yaml
version: "1.0"

# ============================================================================
# PERCEPTION (How I observe - now just encoding, not fields)
# ============================================================================
perception:
  observation_encoding: relative  # relative, scaled, absolute
  # Coordinate normalization only - field inclusion controlled by stratum

# ============================================================================
# DRIVE (What I want - reward function)
# ============================================================================
drive:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]

  intrinsic:
    strategy: rnd
    base_weight: 1.0
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true

# ============================================================================
# BRAIN (How I learn)
# ============================================================================
brain:
  architecture:
    type: feedforward

    feedforward:
      hidden_layers: [256, 128]
      activation: relu
      dropout: 0.0
      layer_norm: true

  optimizer:
    type: adam
    learning_rate: 0.00025
    schedule:
      type: constant

  q_learning:
    gamma: 0.99
    target_update_frequency: 100
    use_double_dqn: true

  replay:
    capacity: 10000
    prioritized: false
```

**Breaking**:
- `drive.*`: NO (reward function doesn't affect obs_dim)
- `brain.architecture`: YES (network structure)
- `brain.q_learning.gamma`: MAYBE (changes Q-value meaning, but network structure unchanged)

**Note**: `partial_observability` and `vision_range` REMOVED from agent - now controlled by curriculum-level active_vision.

---

### Layer 6: Curriculum Levels (Behavioral Parameters + Active Fields)
**Files**: `levels/L*/`

#### `levels/L1/curriculum.yaml` (NEW - Curriculum-Level Active Field Control)

```yaml
version: "1.0"

# Support vs Active Pattern: Which fields are ACTIVE vs MASKED
active_vision: global  # global, partial
  # global:  grid_encoding active, local_window masked (if vision_support: both)
  # partial: local_window active, grid_encoding masked (if vision_support: both)
  # If vision_support: global, only 'global' is valid
  # If vision_support: partial, only 'partial' is valid

active_temporal: false  # true, false
  # false: temporal fields masked (if temporal_support: enabled)
  # true:  temporal fields active (if temporal_support: enabled)
  # If temporal_support: disabled, this must be false

# Curriculum-specific perception (was in agent.yaml v1)
vision_range: 8  # Only used if active_vision: partial
  # Determines local_window size (vision_range=2 → 5×5 window)

# Temporal mechanics parameters (was in training.yaml)
day_length: 24  # Only used if active_temporal: true
```

**Breaking**: NO (just controls masking, not obs_dim)

---

#### `levels/L1/bars.yaml` (Parameters)

```yaml
version: '1.0'

# Bar parameters (MUST match environment.yaml vocabulary)
meters:
  - name: energy  # Must exist in environment.yaml
    initial: 1.0  # ← Curriculum parameter
    base_depletion: 0.003  # ← Curriculum parameter
    base_move_depletion: 0.005
    base_interaction_cost: 0.005

  - name: health
    initial: 1.0
    base_depletion: 0.0
    base_move_depletion: 0.0
    base_interaction_cost: 0.0

  # ... all 8 meters

# Terminal conditions (behavioral)
terminal_conditions:
  - meter: health
    operator: <=
    threshold: 0.0  # ← Curriculum parameter (could be 0.1 in harder level)

  - meter: energy
    operator: <=
    threshold: 0.0

# Cascade parameters (graph structure from environment.yaml)
cascades:
  - source: satiation  # Must exist in environment.yaml cascade_graph
    target: health
    threshold: 0.3  # ← Curriculum parameter
    strength: 0.004  # ← Curriculum parameter

  - source: satiation
    target: energy
    threshold: 0.3
    strength: 0.002

  # ... all cascades from environment.yaml

# Modulation parameters
modulations:
  - source: fitness  # Must exist in environment.yaml modulation_graph
    target: health
    base_multiplier: 0.5  # ← Curriculum parameter
    range: 2.5
    baseline_depletion: 0.001
```

**Breaking**: NO - all parameters are behavioral

**Compiler Validation**:
- All meters from environment.yaml must be present
- All cascades from environment.yaml cascade_graph must be present
- Cannot add meters/cascades not in vocabulary

---

#### `levels/L1/affordances.yaml` (Parameters)

```yaml
version: '2.0'

affordances:
  - name: Bed  # Must exist in environment.yaml
    costs:
      - meter: money
        amount: 0.05  # ← Curriculum parameter

    costs_per_tick:
      - meter: money
        amount: 0.01

    operating_hours: [0, 24]  # ← Curriculum parameter

    effect_pipeline:
      per_tick:
        - meter: energy
          amount: 0.075  # ← Curriculum parameter
      on_completion:
        - meter: energy
          amount: 0.125
        - meter: health
          amount: 0.02

    duration_ticks: 5  # ← Curriculum parameter

  # ... all 14 affordances from environment.yaml
```

**Breaking**: NO - costs, effects, hours are behavioral

**Compiler Validation**:
- All affordances from environment.yaml must be present
- Cannot add affordances not in vocabulary

---

#### `levels/L1/training.yaml` (Runtime Orchestration)

```yaml
version: "1.0"

run_metadata:
  output_subdir: L1_full_observability

# Environment runtime settings
environment:
  enabled_affordances: null  # null = all, or subset like [Bed, Shower, Job]
    # Subset of actions.yaml vocabulary - cannot add new affordances
  randomize_affordances: true

  enabled_actions: null  # null = all enabled_by_default, or explicit subset
    # Subset of actions.yaml vocabulary - cannot add new actions

# Population settings
population:
  num_agents: 1

# Curriculum progression
curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000

# Exploration annealing
exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  min_survival_fraction: 0.4
  survival_window: 100

# Training loop
training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  batch_size: 64
  sequence_length: 8
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
```

**Breaking**: NO - all runtime/orchestration

---

## Complete Directory Structure

```
configs/
└── default_curriculum/
    ├── experiment.yaml        # Metadata
    ├── stratum.yaml           # World shape + vision/temporal support
    ├── environment.yaml       # Vocabulary (meters, affordances, cascades, VFS)
    ├── actions.yaml           # Action vocabulary
    ├── agent.yaml             # Perception encoding + Drive + Brain
    └── levels/
        ├── L0_0_minimal/
        │   ├── curriculum.yaml     # Active vision/temporal + vision_range
        │   ├── bars.yaml           # Meter parameters + cascade params
        │   ├── affordances.yaml    # Affordance parameters
        │   └── training.yaml       # Runtime orchestration
        ├── L1_full_observability/
        │   ├── curriculum.yaml     # active_vision: global
        │   ├── bars.yaml
        │   ├── affordances.yaml
        │   └── training.yaml
        └── L2_partial_observability/
            ├── curriculum.yaml     # active_vision: partial, vision_range: 2
            ├── bars.yaml
            ├── affordances.yaml
            └── training.yaml
```

---

## Observation Spec Generation (Revised)

```python
# Stage 5: Build observation spec from vocabulary + support + active

observation_fields = []

# 1. Position (from stratum substrate)
if config.stratum.substrate.type in ["grid", "grid3d"]:
    observation_fields.append(ObservationField(
        name="obs_position",
        dims=config.stratum.substrate.position_dim,
        curriculum_active=True  # Always active
    ))

# 2. Spatial views (from stratum vision_support + curriculum active_vision)
if config.stratum.vision_support in ["both", "global"]:
    # grid_encoding field exists
    grid_dims = config.stratum.substrate.grid.width * config.stratum.substrate.grid.height
    observation_fields.append(ObservationField(
        name="obs_grid_encoding",
        dims=grid_dims,
        curriculum_active=(curriculum.active_vision == "global")  # Active only if selected
    ))

if config.stratum.vision_support in ["both", "partial"]:
    # local_window field exists
    window_size = curriculum.vision_range * 2 + 1
    observation_fields.append(ObservationField(
        name="obs_local_window",
        dims=window_size ** 2,
        curriculum_active=(curriculum.active_vision == "partial")  # Active only if selected
    ))

# 3. Meters (from environment vocabulary)
for meter in config.environment.meters:
    observation_fields.append(ObservationField(
        name=f"obs_{meter.name}",
        dims=1,
        curriculum_active=True  # Always active
    ))

# 4. Affordances (from environment vocabulary)
affordance_count = len(config.environment.affordances)
observation_fields.append(ObservationField(
    name="obs_affordance_at_position",
    dims=affordance_count + 1,  # +1 for "none"
    curriculum_active=True  # Always active
))

# 5. Temporal (from stratum temporal_support + curriculum active_temporal)
if config.stratum.temporal_support == "enabled":
    observation_fields.extend([
        ObservationField(
            name="obs_time_sin",
            dims=1,
            curriculum_active=curriculum.active_temporal  # Masked if not active
        ),
        ObservationField(
            name="obs_time_cos",
            dims=1,
            curriculum_active=curriculum.active_temporal
        ),
    ])

# 6. VFS variables (from environment vocabulary)
for variable in config.environment.variables:
    observation_fields.append(ObservationField(
        name=f"obs_{variable.name}",
        dims=variable.dims,
        curriculum_active=True  # Always active (for now - future: per-level VFS)
    ))

observation_spec = ObservationSpec(fields=observation_fields)

# Observation dim is CONSTANT across curriculum (all fields included, some masked)
# Example: vision_support: both → L1 and L2 both have grid_encoding + local_window
# L1: grid_encoding active, local_window masked
# L2: grid_encoding masked, local_window active
# Same obs_dim → checkpoint transfer works!
```

---

## Example Configurations

### Default Curriculum (BUG-43 Masking Pattern)

**stratum.yaml**:
```yaml
substrate:
  type: grid
  grid:
    width: 8
    height: 8
vision_support: both  # Enable both spatial views for transfer learning
temporal_support: enabled  # Enable temporal for L3
```

**L1/curriculum.yaml**:
```yaml
active_vision: global  # Use grid_encoding, mask local_window
active_temporal: false  # Mask temporal fields
vision_range: 8  # Not used (active_vision: global)
```

**L2/curriculum.yaml**:
```yaml
active_vision: partial  # Use local_window, mask grid_encoding
active_temporal: false
vision_range: 2  # 5×5 local window
```

**L3/curriculum.yaml**:
```yaml
active_vision: global
active_temporal: true  # Activate temporal fields
day_length: 24
```

**Result**: L1, L2, L3 all have **same obs_dim** → checkpoint transfer works!

---

### Power User: Single-Level Full-Obs Optimization

**stratum.yaml**:
```yaml
substrate:
  type: grid
  grid:
    width: 8
    height: 8
vision_support: global  # Only grid_encoding, skip local_window entirely
temporal_support: disabled  # No temporal overhead
```

**L1/curriculum.yaml**:
```yaml
active_vision: global  # Only valid option (vision_support: global)
active_temporal: false  # Only valid option (temporal_support: disabled)
```

**Result**: Minimized obs_dim, no wasted dimensions, no transfer capability.

---

## Removed Files

- ❌ `brain.yaml` → merged into `agent.yaml`
- ❌ `drive_as_code.yaml` → merged into `agent.yaml`
- ❌ `substrate.yaml` → split into `stratum.yaml` (shape) + removed
- ❌ `cascades.yaml` → structure in `environment.yaml`, params in `bars.yaml`
- ❌ `variables_reference.yaml` → vocabulary in `environment.yaml`
- ❌ `cues.yaml` → future multi-agent (not in critical path)

---

## New Files

- ✅ `stratum.yaml` (world shape + support declarations)
- ✅ `environment.yaml` (vocabulary: meters, affordances, cascades, VFS)
- ✅ `actions.yaml` (action vocabulary)
- ✅ `curriculum.yaml` (per-level active field control)

---

## Code Review Issues Addressed

### Issue #1: Observation Structure Contradiction ✅ FIXED
- **v1 Problem**: Mutual exclusion (grid_encoding OR local_window)
- **v2 Solution**: Support/Active pattern preserves masking
- `vision_support: both` creates both fields, curriculum selects which is active

### Issue #2: Perception Can't Vary Across Curriculum ✅ FIXED
- **v1 Problem**: Single agent.yaml can't handle L1 full obs + L2 partial obs
- **v2 Solution**: Moved active_vision and vision_range to curriculum.yaml
- Each level declares its own perception mode

### Issue #3: Temporal Mechanics All-or-Nothing ✅ FIXED
- **v1 Problem**: Stratum forces all levels to have temporal
- **v2 Solution**: temporal_support + active_temporal pattern
- L0/L1/L2: `active_temporal: false` (masked), L3: `active_temporal: true` (active)

### Issue #4: Global Actions Missing ✅ FIXED
- **v1 Problem**: No home for global_actions.yaml
- **v2 Solution**: Added actions.yaml at experiment level
- Defines action vocabulary (substrate + custom actions)

### Issue #5: Cascade Semantics ✅ ADDRESSED
- **v1 Problem**: Cascade structure vs parameters not separated
- **v2 Solution**: Cascade graph in environment.yaml, parameters in bars.yaml
- All curriculum levels must implement all cascades from graph (compiler validation)

### Issue #6: VFS Implementation ✅ CLARIFIED
- **v1 Problem**: Variables defined but computation logic unclear
- **v2 Solution**: Vocabulary in environment.yaml, computation in VFS compiler
- Variables declare "this field exists", VFS system handles computation

### Issue #7: Action Labels and Custom Actions ✅ FIXED
- **v1 Problem**: No clear home for action labels, custom actions
- **v2 Solution**: All in actions.yaml (labels, substrate actions, custom actions)

### Issue #8: Migration Complexity ✅ ACKNOWLEDGED
- **v1 Problem**: substrate.yaml settings span experiment + curriculum
- **v2 Solution**: Document split clearly (boundary, distance_metric → where?)
- **Open**: Some substrate settings (boundary, distance_metric) need home

---

## Open Questions

1. **Where do substrate behavioral parameters live?**
   - `boundary: clamp` (behavioral, doesn't break checkpoints)
   - `distance_metric: manhattan` (behavioral)
   - Options: curriculum.yaml, stratum.yaml with "support" suffix, new substrate_params.yaml?

2. **Can meters be activated/deactivated per curriculum?**
   - Pattern suggests: `active_meters: [energy, health, satiation]` in curriculum.yaml
   - Use case: L0 starts with fewer meters, L1 adds more complexity
   - Risk: Changes active_mask, might affect policy if meter index matters

3. **Cascade graph enforcement strictness**
   - STRICT: All curriculum levels must have ALL cascades from graph
   - LENIENT: Curriculum can set cascade strength=0 to disable
   - Current design: STRICT (more explicit, follows no-defaults)

4. **VFS variable curriculum-level computation?**
   - Can L0 compute deficit_energy differently than L3?
   - Or is VFS computation uniform across curriculum?
   - Current design: Uniform (variables just declare existence)

---

## Benefits

1. **Checkpoint Portability**: Support/Active pattern preserves BUG-43 masking
2. **Power User Optimization**: `vision_support: global` minimizes obs_dim
3. **Clear Boundaries**: Support (experiment) vs Active (curriculum) is unambiguous
4. **Curriculum Flexibility**: Each level controls perception mode independently
5. **Compiler Validation**: Vocabulary consistency enforced across levels
6. **Reduced Complexity**: Fewer experiment-level files (4 vs 7+ in v1)
7. **Migration Clarity**: Support/Active pattern provides clear upgrade path

---

## Implementation Priority

1. **Phase 1**: Schema design (StratumConfig, EnvironmentConfig with support fields)
2. **Phase 2**: Curriculum.yaml schema (active_vision, active_temporal, vision_range)
3. **Phase 3**: Compiler Stage 5 observation spec with curriculum_active masking
4. **Phase 4**: Cross-curriculum validation (vocabulary consistency)
5. **Phase 5**: Migration script (split substrate.yaml → stratum.yaml)
6. **Phase 6**: Update all test configs to new structure
7. **Phase 7**: Remove legacy support (pre-release = breaking changes OK)

---

## Related Tickets

- **BUG-43**: Curriculum masking (enabled Support/Active pattern)
- **ENH-28**: Experiment-level hierarchy (this is the design)
- **Code Review**: feature-dev:code-reviewer identified critical issues fixed here
