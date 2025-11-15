# Target Configuration Design
## BUNDLE-01: Experiment-Level Hierarchy

**Status**: Design Complete, Implementation Pending

**Core Principle**: Separate **WHAT exists** (vocabulary, breaks checkpoints) from **HOW it behaves** (parameters, curriculum-safe).

---

## Architecture Layers

### Layer 1: Experiment Metadata
**File**: `experiment.yaml`
**Purpose**: High-level experiment information and future experiment-level parameters

```yaml
name: "Default Curriculum"
description: "Standard L0→L3 pedagogical progression"
author: "HAMLET Team"
version: "1.0"

# Future: curriculum sequencing, experiment-level flags, etc.
```

**Breaking**: No (metadata only)

---

### Layer 2: Stratum (World Shape)
**File**: `stratum.yaml`
**Purpose**: Physical reality - substrate type, dimensions, temporal mechanics

```yaml
version: "1.0"

substrate:
  type: grid  # grid, grid3d, gridnd, continuous, continuousnd, aspatial

  grid:
    topology: square
    width: 8
    height: 8
    boundary: clamp  # clamp, wrap, bounce, sticky
    distance_metric: manhattan  # manhattan, euclidean, chebyshev
    observation_encoding: relative  # relative, scaled, absolute

temporal_mechanics:
  enabled: true
  # OR
  # enabled: false  # No-defaults rule: must explicitly disable
```

**Breaking**: YES
- Substrate type changes position_dim (Grid2D=2, Grid3D=3)
- Grid dimensions change grid_encoding dims (3×3=9, 8×8=64)
- Temporal mechanics adds time_sin, time_cos observation fields

**Rationale**: Grid size must be consistent across curriculum for checkpoint transfer.

---

### Layer 3: Environment (World Vocabulary)
**File**: `environment.yaml`
**Purpose**: WHAT exists - bars, affordances, VFS variables (defines obs_dim)

```yaml
version: "1.0"

# Bar vocabulary (defines which obs_meter_* fields exist)
bars:
  - name: energy
    terminal: true  # Death condition
    description: "Can you move? Death if depleted."

  - name: health
    terminal: true
    description: "Are you alive? Death if depleted."

  - name: satiation
    description: "Hunger level. FUNDAMENTAL need."

  - name: fitness
    description: "Physical fitness. Modulates health."

  - name: mood
    description: "Mental well-being."

  - name: hygiene
    description: "Cleanliness."

  - name: social
    description: "Social connection."

  - name: money
    description: "Currency for purchasing affordances."

# Affordance vocabulary (defines affordance_at_position encoding size)
affordances:
  - name: Bed
    category: energy_restoration

  - name: LuxuryBed
    category: energy_restoration

  - name: Shower
    category: hygiene

  - name: HomeMeal
    category: food

  - name: Restaurant
    category: food

  - name: Gym
    category: fitness

  - name: Bar
    category: social

  - name: Job
    category: income

  - name: Cafe
    category: mood

  - name: Park
    category: mood

  - name: Library
    category: mood

  - name: Concert
    category: social

  - name: SpaDay
    category: hygiene

  - name: Therapist
    category: mood

# VFS variable vocabulary (defines custom observation fields)
variables:
  - name: deficit_energy
    type: scalar
    description: "How far below target energy"

  - name: rush_hour
    type: scalar
    description: "True during commute hours"

  # ... additional VFS variables
```

**Breaking**: YES
- Changing bar count/names changes observation field count
- Changing affordance count/names changes one-hot encoding size
- Changing VFS variables changes observation fields

**Compiler Validation**:
- All curriculum levels must have bars/affordances matching environment.yaml vocabulary
- Adding a bar in L1 but not L0 = compiler error

---

### Layer 4: Agent (Complete Agent Specification)
**File**: `agent.yaml`
**Purpose**: Perception + Drive + Brain - everything about the agent

```yaml
version: "1.0"

# ============================================================================
# PERCEPTION (How I observe the world)
# ============================================================================
perception:
  partial_observability: false  # true = local window, false = full grid
  vision_range: 2  # Local window size (5×5 grid when range=2)
  observation_encoding: relative  # relative, scaled, absolute (inherited from stratum if not specified)

# ============================================================================
# DRIVE (What I want - reward function)
# ============================================================================
drive:
  version: "1.0"

  modifiers: {}  # Context-sensitive adjustment (crisis suppression, etc.)

  extrinsic:
    type: multiplicative  # or constant_base_with_shaped_bonus, weighted_sum, etc.
    base: 1.0
    bars: [energy, health]

  intrinsic:
    strategy: rnd  # rnd, icm, count_based, adaptive_rnd, none
    base_weight: 1.0
    apply_modifiers: []

  shaping: []  # Behavioral incentives (approach_reward, completion_bonus, etc.)

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true

# ============================================================================
# BRAIN (How I learn)
# ============================================================================
brain:
  architecture:
    type: feedforward  # feedforward → SimpleQNetwork, recurrent → RecurrentSpatialQNetwork

    feedforward:
      hidden_layers: [256, 128]
      activation: relu
      dropout: 0.0
      layer_norm: true

  optimizer:
    type: adam
    learning_rate: 0.00025  # Atari DQN standard
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-8
    weight_decay: 0.0
    schedule:
      type: constant  # constant, exponential_decay, cosine_annealing

  loss:
    type: mse  # mse, huber
    huber_delta: 1.0

  q_learning:
    gamma: 0.99
    target_update_frequency: 100
    use_double_dqn: true

  replay:
    capacity: 10000
    prioritized: false
```

**Breaking**:
- `partial_observability` changes local_window inclusion (YES)
- `vision_range` changes local_window dims (YES)
- `architecture.type` changes network completely (YES)
- `hidden_layers` changes network structure (YES)
- `drive.*` does NOT break (reward function doesn't affect obs_dim)

**Rationale**: All agents in experiment share same perception + drive + brain.

---

### Layer 5: Curriculum Levels (Behavioral Parameters)
**Files**: `levels/L0_0_minimal/*.yaml`
**Purpose**: HOW things behave - difficulty tuning without breaking checkpoints

#### `levels/L0/bars.yaml` (Parameters)
```yaml
version: '1.0'
description: "L0 bars configuration (easy difficulty)"

# Bar parameters (MUST match environment.yaml vocabulary)
bars:
  - name: energy  # Must exist in environment.yaml
    index: 0
    tier: pivotal
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.003  # ← Curriculum parameter (can vary per level)
    base_move_depletion: 0.005  # ← Curriculum parameter
    base_interaction_cost: 0.005  # ← Curriculum parameter

  - name: health
    index: 6
    tier: pivotal
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.0
    base_move_depletion: 0.0
    base_interaction_cost: 0.0

  # ... all 8 bars (must match environment.yaml)

# Terminal conditions (behavioral)
terminal_conditions:
  - meter: health
    operator: <=
    value: 0.0  # ← Curriculum parameter (could be 0.1 in harder level)

  - meter: energy
    operator: <=
    value: 0.0

# Cascades (moved from separate file - part of bar behavior)
cascades:
  - name: "satiation_to_health"
    source: satiation
    target: health
    threshold: 0.3  # ← Curriculum parameter
    strength: 0.004  # ← Curriculum parameter

  - name: "satiation_to_energy"
    source: satiation
    target: energy
    threshold: 0.3
    strength: 0.002

  # ... all cascades

# Modulations (fitness affects health)
modulations:
  - name: "fitness_health_modulation"
    source: fitness
    target: health
    type: depletion_multiplier
    base_multiplier: 0.5
    range: 2.5  # ← Curriculum parameter
    baseline_depletion: 0.001
```

**Breaking**: NO - all parameters are behavioral

---

#### `levels/L0/affordances.yaml` (Parameters)
```yaml
version: '2.0'
description: "L0 affordances configuration (easy difficulty)"

affordances:
  - name: Bed  # Must exist in environment.yaml
    id: '0'
    category: energy_restoration
    interaction_type: dual

    # Curriculum parameters (can vary per level)
    costs:
      - meter: money
        amount: 0.05  # ← Cheaper in L0 than L3

    costs_per_tick:
      - meter: money
        amount: 0.01

    operating_hours: [0, 24]  # ← Could be [20, 8] in L3 (night only)

    effect_pipeline:
      per_tick:
        - meter: energy
          amount: 0.075  # ← Stronger effect in L0
      on_completion:
        - meter: energy
          amount: 0.125
        - meter: health
          amount: 0.02

    duration_ticks: 5

  # ... all 14 affordances (must match environment.yaml)
```

**Breaking**: NO - costs, effects, hours are behavioral

---

#### `levels/L0/training.yaml` (Runtime Orchestration)
```yaml
version: "1.0"

run_metadata:
  output_subdir: L0_0_minimal

# Environment runtime settings
environment:
  enable_temporal_mechanics: false  # Behavioral: day/night cycle on/off
  day_length: 24  # Only used if temporal_mechanics enabled
  enabled_affordances: null  # null = all, or subset like [0, 1, 2, 3]
  randomize_affordances: true

# Population settings
population:
  num_agents: 1
  mask_unused_obs: false

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
  initial_intrinsic_weight: 1.0  # Should match agent.yaml drive.intrinsic.base_weight
  variance_threshold: 100.0
  min_survival_fraction: 0.4
  survival_window: 100

# Training loop
training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  batch_size: 64
  sequence_length: 8  # LSTM only
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
```

**Breaking**: NO - all runtime/orchestration settings

---

## Complete Directory Structure

```
configs/
├── default_curriculum/              # Experiment directory
│   ├── experiment.yaml              # Metadata
│   ├── stratum.yaml                 # World shape (grid 8×8, temporal)
│   ├── environment.yaml             # World vocabulary (8 bars, 14 affordances)
│   ├── agent.yaml                   # Complete agent (perception + drive + brain)
│   └── levels/
│       ├── L0_0_minimal/
│       │   ├── bars.yaml            # Bar parameters + cascades
│       │   ├── affordances.yaml     # Affordance parameters
│       │   └── training.yaml        # Runtime orchestration
│       ├── L0_5_dual_resource/
│       │   ├── bars.yaml
│       │   ├── affordances.yaml
│       │   └── training.yaml
│       ├── L1_full_observability/
│       │   ├── bars.yaml
│       │   ├── affordances.yaml
│       │   └── training.yaml
│       ├── L2_partial_observability/
│       │   ├── bars.yaml
│       │   ├── affordances.yaml
│       │   └── training.yaml
│       └── L3_temporal_mechanics/
│           ├── bars.yaml
│           ├── affordances.yaml
│           └── training.yaml
│
├── single_level_experiment/         # Another experiment
│   ├── experiment.yaml
│   ├── stratum.yaml
│   ├── environment.yaml
│   ├── agent.yaml
│   └── levels/
│       └── L1_full_observability/
│           ├── bars.yaml
│           ├── affordances.yaml
│           └── training.yaml
```

---

## Removed Files

The following files are consolidated or replaced:

- ❌ `brain.yaml` → merged into `agent.yaml`
- ❌ `drive_as_code.yaml` → merged into `agent.yaml` (drive section)
- ❌ `cascades.yaml` → merged into `bars.yaml`
- ❌ `cues.yaml` → remains (future multi-agent), but not in critical path
- ❌ `substrate.yaml` → split into `stratum.yaml` (shape) and curriculum-level params
- ❌ `variables_reference.yaml` → merged into `environment.yaml`

---

## Compiler Changes

### Stage 1: Load Experiment Structure
```python
# New loader understands experiment hierarchy
config = load_experiment(config_dir)

# Returns:
# config.experiment       # Metadata
# config.stratum          # World shape
# config.environment      # Vocabulary (bars, affordances, VFS)
# config.agent            # Perception + drive + brain
# config.levels           # Dict[str, CurriculumLevel]
```

### Stage 2: Cross-Curriculum Validation
```python
# Validate vocabulary consistency across all levels
for level_name, level in config.levels.items():
    # All levels must have same bars
    assert set(level.bars.keys()) == set(config.environment.bars.keys()), \
        f"Level {level_name} has different bar vocabulary"

    # All levels must have same affordances
    assert set(level.affordances.keys()) == set(config.environment.affordances.keys()), \
        f"Level {level_name} has different affordance vocabulary"
```

### Stage 5: Observation Spec Generation
```python
# Build observation spec from vocabulary (environment.yaml)
# Agent perception (agent.yaml) determines which fields are active

observation_fields = []

# Position (from stratum)
if config.stratum.substrate.type in ["grid", "grid3d"]:
    observation_fields.append(ObservationField(
        name="obs_position",
        dims=config.stratum.substrate.position_dim
    ))

# Spatial views (from agent perception)
if config.agent.perception.partial_observability:
    # Include local_window
    window_size = config.agent.perception.vision_range * 2 + 1
    observation_fields.append(ObservationField(
        name="obs_local_window",
        dims=window_size ** 2
    ))
else:
    # Include grid_encoding
    grid_dims = config.stratum.substrate.grid.width * config.stratum.substrate.grid.height
    observation_fields.append(ObservationField(
        name="obs_grid_encoding",
        dims=grid_dims
    ))

# Meters (from environment vocabulary)
for bar in config.environment.bars:
    observation_fields.append(ObservationField(
        name=f"obs_{bar.name}",
        dims=1
    ))

# Affordances (from environment vocabulary)
affordance_count = len(config.environment.affordances)
observation_fields.append(ObservationField(
    name="obs_affordance_at_position",
    dims=affordance_count + 1  # +1 for "none"
))

# Temporal (from stratum)
if config.stratum.temporal_mechanics.enabled:
    observation_fields.extend([
        ObservationField(name="obs_time_sin", dims=1),
        ObservationField(name="obs_time_cos", dims=1),
    ])

# VFS variables (from environment vocabulary)
for variable in config.environment.variables:
    observation_fields.append(ObservationField(
        name=f"obs_{variable.name}",
        dims=variable.dims
    ))

observation_spec = ObservationSpec(fields=observation_fields)
```

---

## Migration Path

### Phase 1: Backwards Compatibility (Legacy Mode)
```python
# If no experiment.yaml, infer structure from flat config
if not (config_dir / "experiment.yaml").exists():
    # Legacy flat structure detected
    config = migrate_legacy_config(config_dir)
```

### Phase 2: Migration Script
```bash
# Migrate existing config to new structure
python -m townlet.tools.migrate_config configs/L1_full_observability configs/default_curriculum/levels/L1_full_observability

# Auto-generates:
# - experiment.yaml (from metadata)
# - stratum.yaml (from substrate.yaml)
# - environment.yaml (from bars/affordances vocabulary)
# - agent.yaml (from brain.yaml + drive_as_code.yaml + training.partial_observability)
```

### Phase 3: Deprecation
- Version 2.0: New structure is default, legacy supported
- Version 3.0: Legacy structure removed (pre-release = no users, can break freely)

---

## Benefits

1. **Clear Separation of Concerns**: What vs How, Vocabulary vs Parameters
2. **Checkpoint Portability**: Environment vocabulary enforces consistent obs_dim
3. **Reduced File Count**: 4 experiment files + 3 per level (vs 10+ per level)
4. **Compiler Validation**: Vocabulary mismatches caught at compile time
5. **Power User Control**: Can optimize obs_dim by changing agent perception
6. **Curriculum Flexibility**: Easy to A/B test different reward functions, depletion rates
7. **Future-Proof**: Clean structure for multi-agent, heterogeneous agents

---

## Open Questions

1. **cues.yaml**: Keep for future multi-agent, or remove entirely?
   - Decision: Keep in curriculum level, not critical path

2. **VFS expression language**: How to handle custom VFS variables?
   - Decision: environment.yaml defines vocabulary, implementation deferred

3. **Observation policy modes**: Still need curriculum_superset, minimal, explicit?
   - Decision: NOT NEEDED - agent.perception handles this directly
   - `partial_observability: false` = full grid (no local_window)
   - `partial_observability: true` = local window (no grid_encoding)
   - Curriculum masking (both views present) was intermediate solution

4. **Global actions**: Where does `global_actions.yaml` live?
   - Decision: Experiment-level (defines action vocabulary)

---

## Implementation Priority

1. **Phase 1**: Create new file schemas (ExperimentConfig, StratumConfig, EnvironmentConfig, AgentConfig)
2. **Phase 2**: Update compiler to load new structure
3. **Phase 3**: Implement cross-curriculum validation
4. **Phase 4**: Migration script for existing configs
5. **Phase 5**: Update all tests
6. **Phase 6**: Remove legacy support (pre-release = breaking changes OK)

---

## Related Tickets

- **BUG-43**: Curriculum masking (resolved) - enabled this design
- **ENH-28**: Experiment-level hierarchy (this document is the design)
- **Future ENH-XX**: Multi-experiment workspaces
- **Future ENH-XX**: Heterogeneous agent populations
