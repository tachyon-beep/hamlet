# drive_as_code.yaml Configuration

---
## AI-Friendly Frontmatter

**Purpose**: Declarative reward function specification for HAMLET RL environments

**When to Read**: Working with reward structures, A/B testing reward strategies, or configuring training runs

**AI-Friendly Summary**:
Drive As Code (DAC) is a declarative reward function compiler that extracts all reward logic from Python into composable YAML configurations. It enables researchers to A/B test reward structures without code changes. DAC compiles YAML specs into GPU-native computation graphs with three components: (1) **Extrinsic** strategies (9 types for base rewards), (2) **Intrinsic** exploration drives (RND, ICM, count-based), and (3) **Shaping** bonuses (11 types for behavioral incentives). Modifiers enable context-sensitive reward adjustment (e.g., crisis suppression). Final composition: `total_reward = extrinsic + (intrinsic × modifiers) + shaping`.

**Reading Strategy**:
- **Quick Reference**: Jump to "Field Reference" section for specific field documentation
- **Examples**: See "Curriculum Examples" section for real configs from L0-L3
- **Strategy Details**: Read "Extrinsic Strategies" or "Shaping Bonuses" for implementation details
- **First-Time Users**: Read "Overview" → "File Structure" → "Curriculum Examples"

**Related Documents**:
- `docs/plans/2025-11-12-drive-as-code-implementation.md` - Implementation plan
- `docs/guides/dac-migration.md` - Migration guide from old reward_strategy
- `docs/config-schemas/training.md` - Training configuration
- `docs/config-schemas/bars.md` - Meter definitions

---

**Location**: `<config_pack>/drive_as_code.yaml`

**Status**: PRODUCTION (TASK-004C Complete)

**Pattern**: All reward parameters must be explicitly specified (no-defaults principle). This ensures reproducibility and prevents silent behavioral changes.

---

## Overview

Drive As Code (DAC) is a declarative reward function system that separates reward specification from implementation. Instead of hardcoding reward logic in Python, operators define reward structures in YAML configuration files. The DACEngine compiles these specs into optimized GPU-native computation graphs.

### Key Benefits

1. **A/B Testing**: Change reward structures without touching code
2. **Reproducibility**: Checkpoints include drive_hash for provenance
3. **Composability**: Mix extrinsic strategies, intrinsic drives, and shaping bonuses
4. **Pedagogical Value**: Expose "interesting failures" (like reward hacking) as teaching moments
5. **GPU Optimization**: All operations vectorized across agents

### Architecture

```
YAML Config → Pydantic DTOs → Compiler Validation → Runtime Execution
     ↓              ↓                    ↓                    ↓
drive_as_code → DriveAsCodeConfig → DACEngine → GPU Tensors
```

**Components**:
- **Modifiers**: Range-based multipliers for context-sensitive adjustment
- **Extrinsic**: Base reward strategies (9 types)
- **Intrinsic**: Exploration drives (RND, ICM, count-based, adaptive, none)
- **Shaping**: Behavioral incentives (11 types)
- **Composition**: Normalization, clipping, logging

**Formula**:
```python
total_reward = extrinsic + (intrinsic × effective_intrinsic_weight) + shaping

where:
    effective_intrinsic_weight = base_weight × modifier₁ × modifier₂ × ...
```

---

## File Structure

```yaml
drive_as_code:
  version: "1.0"

  modifiers:
    <modifier_name>:
      bar: <bar_id>              # OR variable: <vfs_variable>
      ranges:
        - name: <range_name>
          min: <float>
          max: <float>
          multiplier: <float>

  extrinsic:
    type: <strategy_type>        # multiplicative | constant_base_with_shaped_bonus | ...
    # ... strategy-specific fields ...
    apply_modifiers: [<mod1>, <mod2>]

  intrinsic:
    strategy: <strategy_name>    # rnd | icm | count_based | adaptive_rnd | none
    base_weight: <float>
    apply_modifiers: [<mod1>, <mod2>]

  shaping:
    - type: <bonus_type>         # approach_reward | completion_bonus | ...
      # ... bonus-specific fields ...

  composition:
    normalize: <bool>
    clip:
      min: <float>
      max: <float>
    log_components: <bool>
    log_modifiers: <bool>
```

---

## Field Reference

### Top-Level Fields

#### `version` (string, REQUIRED)

**Type**: `str`
**Required**: Yes
**Example**: `version: "1.0"`

DAC schema version. Always "1.0" for current implementation.

**Validation**: Must be "1.0"

---

#### `modifiers` (dict, REQUIRED)

**Type**: `dict[str, ModifierConfig]`
**Required**: Yes (can be empty dict)
**Example**:
```yaml
modifiers:
  energy_crisis:
    bar: energy
    ranges:
      - name: crisis
        min: 0.0
        max: 0.3
        multiplier: 0.0
      - name: normal
        min: 0.3
        max: 1.0
        multiplier: 1.0
```

Named modifier definitions. Modifiers apply range-based multipliers to intrinsic weight or extrinsic rewards based on bar/variable values.

**Use Cases**:
- **Crisis suppression**: Disable intrinsic curiosity when resources critically low
- **Temporal decay**: Reduce rewards over time
- **Boredom boost**: Increase exploration when performance plateaus

**Validation**:
- Keys are unique modifier names
- All referenced modifiers must be defined before use

---

#### `extrinsic` (ExtrinsicStrategyConfig, REQUIRED)

**Type**: `ExtrinsicStrategyConfig`
**Required**: Yes
**Example**:
```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy, health]
```

Base reward strategy configuration. Defines how to compute extrinsic rewards from meter values.

**See**: "Extrinsic Strategies" section for all 9 strategy types

**Validation**:
- `type` must be one of 9 supported strategies
- Referenced bars must exist in bars.yaml
- Referenced variables must exist in variables_reference.yaml

---

#### `intrinsic` (IntrinsicStrategyConfig, REQUIRED)

**Type**: `IntrinsicStrategyConfig`
**Required**: Yes
**Example**:
```yaml
intrinsic:
  strategy: rnd
  base_weight: 0.1
  apply_modifiers: [energy_crisis]
```

Intrinsic curiosity/exploration configuration. Defines how to compute novelty-seeking rewards.

**See**: "Intrinsic Strategies" section for all 5 strategy types

**Validation**:
- `strategy` must be one of 5 supported strategies
- `base_weight` must be in [0.0, 1.0]
- `apply_modifiers` must reference defined modifiers

---

#### `shaping` (list[ShapingBonusConfig], REQUIRED)

**Type**: `list[ShapingBonusConfig]`
**Required**: Yes (can be empty list)
**Example**:
```yaml
shaping:
  - type: approach_reward
    weight: 0.5
    target_affordance: Bed
    max_distance: 10.0
  - type: completion_bonus
    weight: 1.0
    affordance: Job
```

Behavioral shaping bonuses. Encourages specific agent behaviors without modifying base reward structure.

**See**: "Shaping Bonuses" section for all 11 bonus types

**Validation**:
- Each bonus must have valid `type` field
- Referenced affordances must exist in affordances.yaml
- Referenced bars/variables must exist

---

#### `composition` (CompositionConfig, REQUIRED)

**Type**: `CompositionConfig`
**Required**: Yes
**Example**:
```yaml
composition:
  normalize: false
  clip: null
  log_components: true
  log_modifiers: true
```

Reward composition settings. Controls how components are combined and logged.

**Fields**:
- `normalize` (bool): Apply tanh normalization to total reward
- `clip` (dict | null): Clip total reward to {min, max} range
- `log_components` (bool): Log extrinsic/intrinsic/shaping separately to TensorBoard
- `log_modifiers` (bool): Log modifier values each step

**Validation**:
- All fields required (no defaults)

---

## Modifiers

### ModifierConfig

Range-based multipliers for contextual reward adjustment.

**Fields**:
- `bar` (string | null): Bar name to monitor (mutually exclusive with `variable`)
- `variable` (string | null): VFS variable name to monitor (mutually exclusive with `bar`)
- `ranges` (list[RangeConfig]): Range definitions with multipliers

**Range Coverage**: Ranges must cover [0.0, 1.0] without gaps or overlaps.

### RangeConfig

Single range in a modifier.

**Fields**:
- `name` (string): Human-readable range name
- `min` (float): Range minimum (inclusive)
- `max` (float): Range maximum (exclusive for all but last)
- `multiplier` (float): Multiplier to apply when value in this range

**Example**:
```yaml
modifiers:
  energy_crisis:
    bar: energy
    ranges:
      - name: crisis
        min: 0.0
        max: 0.2
        multiplier: 0.0      # Suppress intrinsic in crisis
      - name: low
        min: 0.2
        max: 0.4
        multiplier: 0.3      # Partial suppression
      - name: normal
        min: 0.4
        max: 1.0
        multiplier: 1.0      # Full intrinsic weight
```

**Pedagogical Note**: Crisis suppression prevents "Low Energy Delirium" bug where agents exploit low extrinsic rewards to maximize intrinsic exploration.

---

## Extrinsic Strategies

DAC supports 9 extrinsic reward strategies. Each strategy defines how to compute base rewards from meter values.

### 1. multiplicative

**Formula**: `reward = base × bar₁ × bar₂ × ...`

**Use Case**: Compound survival incentive (all bars must be high)

**Fields**:
- `base` (float): Multiplicative base
- `bars` (list[str]): Bar names to multiply
- `apply_modifiers` (list[str], optional): Modifiers to apply

**Example**:
```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy, health]
  apply_modifiers: []
```

**Pedagogical Note**: Creates "Low Energy Delirium" bug when intrinsic weight is high - agents learn to exploit low bars for exploration.

---

### 2. constant_base_with_shaped_bonus

**Formula**: `reward = base_reward + Σ(bar_bonuses) + Σ(variable_bonuses)`

**Bar Bonus Formula**: `bonus = scale × (bar_value - center)`

**Use Case**: Fixes "Low Energy Delirium" - constant base prevents reward hacking

**Fields**:
- `base_reward` (float): Constant survival reward
- `bar_bonuses` (list[BarBonusConfig]): Bar-based bonuses
- `variable_bonuses` (list[VariableBonusConfig]): VFS variable bonuses
- `apply_modifiers` (list[str], optional): Modifiers to apply

**BarBonusConfig**:
- `bar` (string): Bar name
- `center` (float): Neutral point (no bonus/penalty)
- `scale` (float): Bonus magnitude

**VariableBonusConfig**:
- `variable` (string): VFS variable name
- `weight` (float): Weight (can be negative)

**Example**:
```yaml
extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  bar_bonuses:
    - bar: energy
      center: 0.5
      scale: 0.5
    - bar: health
      center: 0.5
      scale: 0.5
  variable_bonuses: []
```

**Pedagogical Note**: This is the "fixed" strategy compared to multiplicative - demonstrates importance of reward structure design.

---

### 3. additive_unweighted

**Formula**: `reward = base + Σ(bars)`

**Use Case**: Encourage high total across all resources

**Fields**:
- `base` (float): Additive base
- `bars` (list[str]): Bar names to sum
- `apply_modifiers` (list[str], optional): Modifiers to apply

**Example**:
```yaml
extrinsic:
  type: additive_unweighted
  base: 0.0
  bars: [energy, health, satiation]
```

---

### 4. weighted_sum

**Formula**: `reward = Σ(weight_i × bar_i)`

**Use Case**: Different importance for different resources

**Fields**:
- `base` (float): Base value
- `bar_bonuses` (list[BarBonusConfig]): Uses `scale` as weight, ignores `center`
- `apply_modifiers` (list[str], optional): Modifiers to apply

**Example**:
```yaml
extrinsic:
  type: weighted_sum
  base: 0.0
  bar_bonuses:
    - bar: health
      center: 0.0    # Ignored
      scale: 2.0     # Weight for health
    - bar: energy
      center: 0.0
      scale: 1.0     # Weight for energy
```

---

### 5. polynomial

**Formula**: `reward = Σ(weight_i × bar_i^exponent_i)`

**Use Case**: Non-linear reward scaling

**Fields**:
- `base` (float): Base value
- `bar_bonuses` (list[BarBonusConfig]): Uses `scale` as weight, `center` as exponent
- `apply_modifiers` (list[str], optional): Modifiers to apply

**Example**:
```yaml
extrinsic:
  type: polynomial
  base: 0.0
  bar_bonuses:
    - bar: energy
      center: 2.0    # Exponent (quadratic)
      scale: 1.0     # Weight
```

---

### 6. threshold_based

**Formula**: `reward = base + Σ(bonus_i if bar_i ≥ threshold_i else 0)`

**Use Case**: Binary milestones (bonus when bar crosses threshold)

**Fields**:
- `base` (float): Base value
- `bar_bonuses` (list[BarBonusConfig]): Uses `center` as threshold, `scale` as bonus
- `apply_modifiers` (list[str], optional): Modifiers to apply

**Example**:
```yaml
extrinsic:
  type: threshold_based
  base: 0.0
  bar_bonuses:
    - bar: energy
      center: 0.8    # Threshold
      scale: 5.0     # Bonus when energy ≥ 0.8
```

---

### 7. aggregation

**Formula**: `reward = base + min(bars)` (simplified - always uses min)

**Use Case**: Reward worst bar (bottleneck incentive)

**Fields**:
- `base` (float): Base value
- `bars` (list[str]): Bars to aggregate
- `apply_modifiers` (list[str], optional): Modifiers to apply

**Note**: Current implementation hardcodes `min` operation. Future versions will support `max`, `mean`, `product`.

**Example**:
```yaml
extrinsic:
  type: aggregation
  base: 0.0
  bars: [energy, health, satiation]
```

---

### 8. vfs_variable

**Formula**: `reward = variable_value`

**Use Case**: Delegate reward computation to VFS (escape hatch for custom logic)

**Fields**:
- `variable` (string): VFS variable name
- `apply_modifiers` (list[str], optional): Modifiers to apply

**Example**:
```yaml
extrinsic:
  type: vfs_variable
  variable: custom_reward_function
```

**Note**: Requires VFS variable with `readable_by: ["engine"]` and `writable_by: ["bac"]` or custom computation.

---

### 9. hybrid

**Formula**: `reward = Σ(weight_i × strategy_i)`

**Use Case**: Weighted combination of multiple strategies

**Status**: NOT YET IMPLEMENTED

**Future Fields**:
- `strategies` (list[ExtrinsicStrategyConfig]): Sub-strategies to combine
- `weights` (list[float]): Weights for each sub-strategy

---

## Intrinsic Strategies

DAC supports 5 intrinsic exploration strategies.

### 1. rnd (Random Network Distillation)

**Method**: Novelty = prediction error between fixed random network and trained predictor

**Use Case**: State-space exploration, novelty-seeking

**Fields**:
- `strategy: rnd`
- `base_weight` (float): Base intrinsic weight in [0.0, 1.0]
- `apply_modifiers` (list[str], optional): Modifiers for contextual adjustment
- `rnd_config` (dict, optional): RND-specific configuration

**Example**:
```yaml
intrinsic:
  strategy: rnd
  base_weight: 0.1
  apply_modifiers: [energy_crisis]
  rnd_config:
    feature_dim: 128
    learning_rate: 0.001
```

**Pedagogical Note**: RND is the standard intrinsic strategy used in all curriculum levels.

---

### 2. icm (Intrinsic Curiosity Module)

**Method**: Novelty = forward model prediction error

**Use Case**: Action-conditioned exploration

**Status**: Supported by schema but not yet implemented in DACEngine

**Fields**:
- `strategy: icm`
- `base_weight` (float): Base intrinsic weight
- `apply_modifiers` (list[str], optional): Modifiers
- `icm_config` (dict, optional): ICM-specific configuration

---

### 3. count_based

**Method**: Novelty = 1/√count(state)

**Use Case**: Tabular exploration, discrete state spaces

**Status**: Supported by schema but not yet implemented in DACEngine

**Fields**:
- `strategy: count_based`
- `base_weight` (float): Base intrinsic weight
- `apply_modifiers` (list[str], optional): Modifiers
- `count_config` (dict, optional): Count-based configuration

---

### 4. adaptive_rnd

**Method**: RND with performance-based weight decay

**Use Case**: Anneal exploration as agent improves

**Fields**:
- `strategy: adaptive_rnd`
- `base_weight` (float): Initial intrinsic weight
- `apply_modifiers` (list[str], optional): Modifiers
- `adaptive_config` (dict, optional): Adaptive annealing configuration

**Annealing Logic**: Weight decays when mean episode survival exceeds threshold (configured in training.yaml).

**Example**:
```yaml
intrinsic:
  strategy: adaptive_rnd
  base_weight: 0.5
  apply_modifiers: []
  adaptive_config:
    threshold: 100.0  # Mean survival steps
```

---

### 5. none

**Method**: No intrinsic rewards (pure extrinsic)

**Use Case**: Ablation studies, curriculum levels without exploration

**Fields**:
- `strategy: none`
- `base_weight: 0.0`
- `apply_modifiers: []`

**Example**:
```yaml
intrinsic:
  strategy: none
  base_weight: 0.0
  apply_modifiers: []
```

---

## Shaping Bonuses

DAC supports 11 shaping bonus types for behavioral incentives.

### 1. approach_reward

**Purpose**: Reward moving closer to target affordance

**Use Case**: Guide agents toward needed resources

**Fields**:
- `type: approach_reward`
- `weight` (float): Bonus magnitude
- `target_affordance` (string): Affordance to approach
- `max_distance` (float): Distance beyond which bonus = 0

**Formula**: `bonus = weight × (1 - distance / max_distance)`

**Example**:
```yaml
shaping:
  - type: approach_reward
    weight: 0.5
    target_affordance: Bed
    max_distance: 10.0
```

---

### 2. completion_bonus

**Purpose**: Fixed bonus when agent completes affordance interaction

**Use Case**: Encourage completing activities

**Fields**:
- `type: completion_bonus`
- `weight` (float): Bonus magnitude
- `affordance` (string): Affordance to reward

**Example**:
```yaml
shaping:
  - type: completion_bonus
    weight: 1.0
    affordance: Job
```

---

### 3. efficiency_bonus

**Purpose**: Bonus for maintaining bar above threshold

**Use Case**: Encourage resource efficiency

**Fields**:
- `type: efficiency_bonus`
- `weight` (float): Bonus magnitude
- `bar` (string): Bar to monitor
- `threshold` (float): Minimum value for bonus

**Example**:
```yaml
shaping:
  - type: efficiency_bonus
    weight: 0.5
    bar: energy
    threshold: 0.7
```

---

### 4. state_achievement

**Purpose**: Bonus when ALL bar conditions met simultaneously

**Use Case**: Reward achieving target state

**Fields**:
- `type: state_achievement`
- `weight` (float): Bonus magnitude
- `conditions` (list[BarCondition]): All must be satisfied

**BarCondition**:
- `bar` (string): Bar name
- `min_value` (float): Minimum required value

**Example**:
```yaml
shaping:
  - type: state_achievement
    weight: 2.0
    conditions:
      - bar: energy
        min_value: 0.8
      - bar: health
        min_value: 0.8
```

---

### 5. streak_bonus

**Purpose**: Bonus for consecutive uses of same affordance

**Use Case**: Reward building habits/routines

**Fields**:
- `type: streak_bonus`
- `weight` (float): Bonus magnitude
- `affordance` (string): Target affordance
- `min_streak` (int): Minimum streak length for bonus

**Example**:
```yaml
shaping:
  - type: streak_bonus
    weight: 5.0
    affordance: Bed
    min_streak: 3
```

---

### 6. diversity_bonus

**Purpose**: Bonus for using many different affordances

**Use Case**: Encourage exploration of all activities

**Fields**:
- `type: diversity_bonus`
- `weight` (float): Bonus magnitude
- `min_unique_affordances` (int): Minimum unique affordances for bonus

**Example**:
```yaml
shaping:
  - type: diversity_bonus
    weight: 3.0
    min_unique_affordances: 4
```

---

### 7. timing_bonus

**Purpose**: Bonus for using affordance during specific time windows

**Use Case**: Contextually appropriate timing (e.g., sleeping at night)

**Fields**:
- `type: timing_bonus`
- `weight` (float): Base bonus weight
- `time_ranges` (list[TimeRange]): Time windows with multipliers

**TimeRange**:
- `start_hour` (int): Start hour [0-23]
- `end_hour` (int): End hour [0-23]
- `affordance` (string): Affordance to reward
- `multiplier` (float): Bonus multiplier for this window

**Example**:
```yaml
shaping:
  - type: timing_bonus
    weight: 1.0
    time_ranges:
      - start_hour: 22
        end_hour: 6
        affordance: Bed
        multiplier: 2.0
```

---

### 8. economic_efficiency

**Purpose**: Bonus for maintaining money above threshold

**Use Case**: Financial responsibility

**Fields**:
- `type: economic_efficiency`
- `weight` (float): Bonus magnitude
- `money_bar` (string): Money bar name
- `min_balance` (float): Minimum balance for bonus

**Example**:
```yaml
shaping:
  - type: economic_efficiency
    weight: 2.0
    money_bar: money
    min_balance: 0.6
```

---

### 9. balance_bonus

**Purpose**: Bonus when all specified bars are in balanced range

**Use Case**: Encourage holistic resource management

**Fields**:
- `type: balance_bonus`
- `weight` (float): Bonus magnitude
- `bars` (list[string]): Bars to balance
- `min_value` (float): Minimum for each bar
- `max_value` (float): Maximum for each bar

**Example**:
```yaml
shaping:
  - type: balance_bonus
    weight: 3.0
    bars: [energy, health, satiation]
    min_value: 0.6
    max_value: 1.0
```

---

### 10. crisis_avoidance

**Purpose**: Penalty when any bar drops below crisis threshold

**Use Case**: Discourage near-death states

**Fields**:
- `type: crisis_avoidance`
- `weight` (float): Penalty magnitude (applied as negative)
- `bars` (list[string]): Bars to monitor
- `crisis_threshold` (float): Threshold below which penalty applies

**Example**:
```yaml
shaping:
  - type: crisis_avoidance
    weight: 5.0
    bars: [energy, health]
    crisis_threshold: 0.2
```

---

### 11. vfs_variable

**Purpose**: Custom shaping logic via VFS variable

**Use Case**: Escape hatch for arbitrary shaping signals

**Fields**:
- `type: vfs_variable`
- `variable` (string): VFS variable name
- `weight` (float): Weight to apply

**Example**:
```yaml
shaping:
  - type: vfs_variable
    variable: custom_shaping_signal
    weight: 1.0
```

---

## Curriculum Examples

### L0_0_minimal (Temporal Credit Assignment)

**Goal**: Single-resource temporal credit assignment

**Strategy**: Multiplicative with energy only

```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy]

  intrinsic:
    strategy: rnd
    base_weight: 0.1
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Pedagogical Note**: "Low Energy Delirium" bug present - agents exploit low energy for exploration.

---

### L0_5_dual_resource (Multiple Resources)

**Goal**: Fix "Low Energy Delirium" bug with constant base

**Strategy**: Constant base with shaped bonuses

```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: constant_base_with_shaped_bonus
    base_reward: 1.0
    bar_bonuses:
      - bar: energy
        center: 0.5
        scale: 0.5
      - bar: health
        center: 0.5
        scale: 0.5
      - bar: satiation
        center: 0.5
        scale: 0.5
      - bar: hygiene
        center: 0.5
        scale: 0.5
    variable_bonuses: []

  intrinsic:
    strategy: rnd
    base_weight: 0.1
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Pedagogical Note**: Constant base prevents reward hacking - compare against L0_0 to demonstrate importance of reward structure.

---

### L1_full_observability (Full Observability Baseline)

**Goal**: Full observability baseline with multiplicative reward

**Strategy**: Multiplicative (energy × health)

```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]

  intrinsic:
    strategy: rnd
    base_weight: 1.0  # Higher intrinsic weight
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Pedagogical Note**: Higher intrinsic weight (1.0) encourages exploration in full observability.

---

### L2_partial_observability (POMDP with LSTM)

**Goal**: POMDP learning with partial observability

**Strategy**: Multiplicative (energy × health)

```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]

  intrinsic:
    strategy: rnd
    base_weight: 0.1  # Lower intrinsic weight
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Pedagogical Note**: Lower intrinsic weight (0.1) - let extrinsic milestones drive learning in partial observability where exploration is harder.

---

### L3_temporal_mechanics (Time-Based Dynamics)

**Goal**: Time-based dynamics with day/night cycle

**Strategy**: Multiplicative (energy × health)

```yaml
drive_as_code:
  version: "1.0"

  modifiers: {}

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]

  intrinsic:
    strategy: rnd
    base_weight: 1.0  # Higher for temporal exploration
    apply_modifiers: []

  shaping: []

  composition:
    normalize: false
    clip: null
    log_components: true
    log_modifiers: true
```

**Pedagogical Note**: Higher intrinsic weight (1.0) encourages temporal exploration patterns.

---

## Advanced Patterns

### Crisis Suppression

**Pattern**: Disable intrinsic curiosity when resources critically low

```yaml
drive_as_code:
  modifiers:
    energy_crisis:
      bar: energy
      ranges:
        - name: crisis
          min: 0.0
          max: 0.2
          multiplier: 0.0  # Suppress intrinsic
        - name: normal
          min: 0.2
          max: 1.0
          multiplier: 1.0  # Full intrinsic

  extrinsic:
    type: multiplicative
    base: 1.0
    bars: [energy, health]

  intrinsic:
    strategy: rnd
    base_weight: 0.5
    apply_modifiers: [energy_crisis]  # Apply suppression
```

---

### Multi-Modifier Chaining

**Pattern**: Chain multiple modifiers for compound effects

```yaml
drive_as_code:
  modifiers:
    energy_crisis:
      bar: energy
      ranges:
        - name: crisis
          min: 0.0
          max: 0.2
          multiplier: 0.0
        - name: normal
          min: 0.2
          max: 1.0
          multiplier: 1.0

    temporal_decay:
      variable: time_normalized
      ranges:
        - name: early
          min: 0.0
          max: 0.5
          multiplier: 1.0
        - name: late
          min: 0.5
          max: 1.0
          multiplier: 0.5  # Reduce intrinsic late in episode

  intrinsic:
    apply_modifiers: [energy_crisis, temporal_decay]  # Both applied
```

**Formula**: `effective_weight = base_weight × energy_crisis_mult × temporal_decay_mult`

---

### VFS Integration

**Pattern**: Use VFS variables for custom logic

```yaml
drive_as_code:
  extrinsic:
    type: constant_base_with_shaped_bonus
    base_reward: 1.0
    bar_bonuses: []
    variable_bonuses:
      - variable: energy_urgency  # VFS-computed urgency
        weight: 2.0
      - variable: health_urgency
        weight: 2.0

  intrinsic:
    strategy: rnd
    base_weight: 0.1
    apply_modifiers: []

  shaping:
    - type: vfs_variable
      variable: custom_proximity_bonus
      weight: 1.0
```

**Requirement**: VFS variables must have `readable_by: ["engine"]`

---

## Validation

### Compile-Time Validation

**Location**: `townlet.universe.compiler.UniverseCompiler`

**Checks**:
1. **Modifier references**: All bars/variables exist
2. **Extrinsic references**: All bars/variables exist
3. **Shaping references**: All affordances exist
4. **Modifier coverage**: Ranges cover [0.0, 1.0] without gaps
5. **Modifier self-references**: modifiers dict contains all referenced names

**Error Codes**:
- `DAC-REF-001`: Modifier references undefined bar
- `DAC-REF-002`: Modifier references undefined VFS variable
- `DAC-REF-003`: Extrinsic references undefined bar
- `DAC-REF-004`: Extrinsic bar bonus references undefined bar
- `DAC-REF-005`: Extrinsic variable bonus references undefined VFS variable
- `DAC-REF-006`: Shaping bonus references undefined affordance

### Runtime Validation

**Location**: `townlet.environment.dac_engine.DACEngine`

**Checks**:
1. **VFS access control**: Engine can read variables with `readable_by: ["engine"]`
2. **Bar index mapping**: Bar names map to valid meter indices
3. **Tensor shapes**: All operations broadcast correctly across agents

---

## Best Practices

### 1. Start Simple

Begin with basic strategies (multiplicative, constant_base) before adding modifiers and shaping.

**Good First Config**:
```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy]

intrinsic:
  strategy: rnd
  base_weight: 0.1
  apply_modifiers: []

shaping: []
```

---

### 2. Use Modifiers for Crisis Suppression

Always suppress intrinsic curiosity during resource crises to prevent "Low Energy Delirium".

**Anti-Pattern**: High intrinsic weight + multiplicative reward without crisis suppression

**Correct Pattern**: Apply crisis modifier to intrinsic strategy

---

### 3. Document Pedagogical Intent

Add comments explaining what behavior the config is intended to teach/demonstrate.

```yaml
# Pedagogical Goal: Demonstrate "Low Energy Delirium" bug
# Teaching Moment: Compare against L0_5 to show importance of reward structure
drive_as_code:
  # ... config ...
```

---

### 4. A/B Test Strategy Changes

Use drive_hash for provenance when comparing different reward structures.

```bash
# Train with multiplicative
uv run scripts/run_demo.py --config configs/L1_full_observability

# Change to constant_base in drive_as_code.yaml
# Drive hash will change, enabling comparison in TensorBoard
```

---

### 5. Log Components Separately

Enable `log_components: true` to track extrinsic/intrinsic/shaping breakdown in TensorBoard.

```yaml
composition:
  log_components: true  # Enables tensorboard logging
  log_modifiers: true   # Log modifier multipliers
```

---

### 6. Explicit Over Implicit

Always specify all fields explicitly - no relying on defaults.

**Anti-Pattern**:
```yaml
extrinsic:
  type: multiplicative
  # Missing base and bars!
```

**Correct**:
```yaml
extrinsic:
  type: multiplicative
  base: 1.0
  bars: [energy, health]
  apply_modifiers: []  # Explicit empty list
```

---

## Troubleshooting

### CompilationError: "undefined bar"

**Cause**: DAC references bar not defined in bars.yaml

**Fix**: Add bar to bars.yaml or fix typo in drive_as_code.yaml

```yaml
# bars.yaml
bars:
  - id: energy  # Must match reference in DAC
    # ...
```

---

### CompilationError: "undefined VFS variable"

**Cause**: DAC references variable not in variables_reference.yaml

**Fix**: Add variable definition with `readable_by: ["engine"]`

```yaml
# variables_reference.yaml
variables:
  - id: energy_urgency
    scope: agent
    type: scalar
    readable_by: [agent, engine]  # Must include "engine"
    # ...
```

---

### RuntimeError: "Bar index not found"

**Cause**: Bar name doesn't map to meter index

**Fix**: Ensure bar exists in universe metadata and matches bars.yaml

---

### Unexpected Behavior: "Low Energy Delirium"

**Cause**: Multiplicative reward + high intrinsic weight without crisis suppression

**Fix**: Add crisis modifier or switch to constant_base_with_shaped_bonus

```yaml
# Option 1: Add crisis suppression
modifiers:
  energy_crisis:
    bar: energy
    ranges:
      - {name: crisis, min: 0.0, max: 0.2, multiplier: 0.0}
      - {name: normal, min: 0.2, max: 1.0, multiplier: 1.0}

intrinsic:
  apply_modifiers: [energy_crisis]

# Option 2: Switch to constant_base
extrinsic:
  type: constant_base_with_shaped_bonus
  base_reward: 1.0
  # ...
```

---

## Related Documentation

- **Implementation Plan**: `docs/plans/2025-11-12-drive-as-code-implementation.md`
- **Migration Guide**: `docs/guides/dac-migration.md`
- **Training Config**: `docs/config-schemas/training.md`
- **Bar Config**: `docs/config-schemas/bars.md`
- **VFS Config**: `docs/config-schemas/variables.md`
- **Compiler Architecture**: `docs/architecture/COMPILER_ARCHITECTURE.md`

---

## Implementation Notes

**Files**:
- DTOs: `src/townlet/config/drive_as_code.py`
- Engine: `src/townlet/environment/dac_engine.py`
- Compiler: `src/townlet/universe/compiler.py` (DAC validation)
- Compiled: `src/townlet/universe/compiled.py` (dac_config, drive_hash fields)

**Tests**:
- Unit: `tests/test_townlet/unit/config/test_drive_as_code_dto.py`
- Engine: `tests/test_townlet/unit/environment/test_dac_engine.py`
- Integration: `tests/test_townlet/integration/test_dac_integration.py`

**CLI**:
```bash
# Compile with DAC validation
python -m townlet.compiler compile configs/L0_0_minimal

# Inspect compiled universe (includes drive_hash)
python -m townlet.compiler inspect configs/L0_0_minimal
```

---

**Last Updated**: 2025-11-12
**Status**: PRODUCTION (TASK-004C Complete)
**Version**: 1.0
