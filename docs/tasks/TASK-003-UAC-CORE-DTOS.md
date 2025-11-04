# TASK-003: UAC Contracts - Core DTOs

**Status**: Planned
**Priority**: HIGH (foundational)
**Estimated Effort**: 7-12 hours
**Dependencies**: None
**Enables**: TASK-002A (SubstrateConfig), TASK-002B (ActionConfig), TASK-004A (core schemas), TASK-004B (extends AffordanceConfig)

---

## Guiding Principle: The Universe Compiler

**Core Philosophy**: Every aspect of the universe should be configurable via config files (UNIVERSE_AS_CODE), **BUT** that schema must be enforced mercilessly with versioning and compile-time checks.

**Mental Model**: Configuration files are not just data - they are **programs that define the universe**. Before running a training session, we must **compile the universe** and validate it against the universe's laws (the schema). This is not testing *values* (whether epsilon=0.99 is "good"), but testing **structural compliance** ("does this config speak the universe's language?").

Think of it as:

1. **Write universe program** (edit YAML configs)
2. **Compile universe** (load into Pydantic DTOs)
3. **Compilation errors** = schema violations, invalid references, constraint violations
4. **Runtime** = only if universe compiles successfully

Just like you wouldn't run Python code with syntax errors, you shouldn't run training with schema-invalid configs.

---

## The No-Defaults Principle: Full Operator Accountability

**Critical Design Rule**: Pydantic DTOs must **require ALL behavioral parameters**. No implicit defaults allowed.

**Rationale**:

- **Accountability**: Operator must explicitly specify every parameter that affects universe behavior
- **Transparency**: Universe behavior fully determined by visible config files (no hidden state)
- **Reproducibility**: Config is complete specification - no need to read source code
- **Prevents Drift**: Changing code defaults doesn't silently change existing universes
- **Better Mental Model**: Operator must understand each parameter to set it

**Allowed Defaults** (rare exceptions):

1. **Truly optional features**: `cues: CuesConfig | None = None` (visualization is optional for headless training)
2. **Metadata only**: `description: str | None = None` (doesn't affect simulation)
3. **Computed values**: `observation_dim: int` (calculated at compile time)

**Rule of thumb**: If omitting the field changes simulation behavior, it's REQUIRED.

---

## Problem: Schema Drift in UNIVERSE_AS_CODE System

### Current State

HAMLET implements "UNIVERSE_AS_CODE" where all game mechanics (bars, cascades, affordances, training hyperparameters) are defined in YAML configuration files. This is a **strength** - the universe is data-driven, not hardcoded.

However, we have **no enforcement** of config schema correctness, leading to:

1. **Silent Schema Drift**: Config files evolve during agile development, creating inconsistencies:
   - L1 missing entire `training` section (epsilon params) → fell back to defaults
   - L0 comments claiming "5×5 grid" while actual `grid_size: 3`
   - No validation that required fields exist

2. **Runtime-Only Errors**: Invalid configs fail during training, not at load time:
   - Typos in field names (e.g., `epsilon_deccay`) → silently ignored, uses defaults
   - Missing required fields → crashes deep in training loop
   - Type errors (string where float expected) → cryptic PyTorch errors

3. **No Contract Between Config & Code**:
   - `runner.py` uses `.get("epsilon_start", 1.0)` everywhere → hides missing fields
   - No guarantee config files match what code expects
   - Refactoring breaks configs silently (rename field in code, configs still use old name)

4. **Documentation Decay**:
   - Comments in YAML become stale (grid size mismatch)
   - No single source of truth for "what fields are valid?"
   - New developers don't know what parameters exist

### Impact

- **Wasted debugging time**: L0 not learning due to slow epsilon decay (0.998 vs 0.99)
- **Silent failures**: L1 using wrong defaults because section missing
- **Maintenance burden**: Must manually audit all configs when adding new parameters
- **Cognitive load**: No autocomplete, no validation, "edit and pray"

---

## Solution: DTO-Based Config Schema Enforcement

### Proposed Architecture

Use **Pydantic data transfer objects (DTOs)** as the single source of truth for config schemas, similar to existing `affordance_config.py` and `cascade_config.py` patterns.

### Core DTOs to Implement

This task creates the foundational DTOs for HAMLET's configuration system:

1. **TrainingConfig** - Training hyperparameters (epsilon-greedy, Q-learning, etc.)
2. **EnvironmentConfig** - Environment setup (grid size, vision range, enabled affordances)
3. **CurriculumConfig** - Curriculum progression (survival thresholds, stage gates)
4. **PopulationConfig** - Population parameters (learning rate, gamma, network type)
5. **SubstrateConfig** - Spatial substrate configuration (TASK-002A integration)
6. **BarConfig** - Meter definitions (energy, health, etc.) - BASIC version
7. **CascadeConfig** - Meter relationships - BASIC version
8. **ActionConfig** - Action space definitions - BASIC version (TASK-002B integration)
9. **AffordanceConfig** - Interaction definitions - BASIC version (WITHOUT capabilities)
10. **HamletConfig** - Master config composing all sub-configs

**Note**: AffordanceConfig is implemented in BASIC form here (core fields only). TASK-004B extends it with capabilities, effect pipelines, and availability masking.

---

## Implementation Plan

### Phase 1: Training Configuration DTOs (2-4 hours)

**Deliverable**: `src/townlet/config/training_config.py`

**Example Implementation**:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import math
import logging

logger = logging.getLogger(__name__)

class TrainingConfig(BaseModel):
    """
    Training hyperparameters configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.
    Operator must explicitly specify all parameters that affect training.
    """

    # Compute device (REQUIRED)
    device: Literal["cpu", "cuda", "mps"]

    # Training duration (REQUIRED)
    max_episodes: int = Field(gt=0)

    # Q-learning hyperparameters (ALL REQUIRED)
    train_frequency: int = Field(gt=0)  # No default!
    target_update_frequency: int = Field(gt=0)  # No default!
    batch_size: int = Field(gt=0)  # No default!
    max_grad_norm: float = Field(gt=0)  # No default!

    # Epsilon-greedy exploration (ALL REQUIRED)
    epsilon_start: float = Field(ge=0.0, le=1.0)  # No default!
    epsilon_decay: float = Field(gt=0.0, lt=1.0)  # No default!
    epsilon_min: float = Field(ge=0.0, le=1.0)  # No default!

    @model_validator(mode="after")
    def validate_epsilon_order(self) -> "TrainingConfig":
        """Ensure epsilon_start >= epsilon_min."""
        if self.epsilon_start < self.epsilon_min:
            raise ValueError(
                f"epsilon_start ({self.epsilon_start}) must be >= "
                f"epsilon_min ({self.epsilon_min})"
            )
        return self

    @model_validator(mode="after")
    def validate_epsilon_decay(self) -> "TrainingConfig":
        """
        Warn (not error) if epsilon decay seems unreasonable.

        NOTE: This is a HINT, not enforcement. Operator may intentionally
        set slow decay for their experiment. We validate structure, not semantics.
        """
        episodes_to_01 = math.log(0.1) / math.log(self.epsilon_decay)
        if self.epsilon_decay > 0.999:
            logger.warning(
                f"epsilon_decay={self.epsilon_decay} is very slow. "
                f"Will take {episodes_to_01:.0f} episodes to reach ε=0.1. "
                f"Typical values: 0.99 (L0), 0.995 (L0.5/L1), 0.998 (L2)."
            )
        return self

def load_training_config(config_path: Path) -> TrainingConfig:
    """
    Load and validate training configuration.

    Raises:
        ValidationError: If required fields missing or invalid values
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    try:
        return TrainingConfig(**data["training"])
    except ValidationError as e:
        # Re-raise with helpful context
        raise ValueError(
            f"training.yaml validation failed:\n{e}\n\n"
            f"All training parameters must be explicitly specified.\n"
            f"See configs/templates/training.yaml.template for reference."
        ) from e
```

**Update `runner.py` to use DTOs**:

```python
# BEFORE: Manual dict access with defaults scattered everywhere
epsilon_start = training_cfg.get("epsilon_start", 1.0)
epsilon_decay = training_cfg.get("epsilon_decay", 0.995)
epsilon_min = training_cfg.get("epsilon_min", 0.01)

# AFTER: Load validated DTO
training_config = load_training_config(config_file)
epsilon_start = training_config.epsilon_start
epsilon_decay = training_config.epsilon_decay
epsilon_min = training_config.epsilon_min
```

---

### Phase 2: Environment Configuration DTOs (2-3 hours)

**Deliverable**: `src/townlet/config/environment_config.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class EnvironmentConfig(BaseModel):
    """Environment configuration."""

    # Grid parameters (REQUIRED)
    grid_size: int = Field(gt=0, le=64)  # Reasonable upper bound

    # Observability (REQUIRED)
    partial_observability: bool
    vision_range: int = Field(gt=0)  # Required even if not used (explicit)

    # Enabled affordances (REQUIRED)
    enabled_affordances: list[str]  # List of affordance IDs to deploy

    @field_validator("vision_range")
    @classmethod
    def validate_vision_range_usage(cls, v: int, info) -> int:
        """Warn if vision_range specified but partial_observability=False."""
        partial_obs = info.data.get("partial_observability", False)
        if not partial_obs and v != info.data.get("grid_size", 0):
            logger.warning(
                f"vision_range={v} specified but partial_observability=False. "
                f"Agent will see full grid regardless."
            )
        return v

    @field_validator("enabled_affordances")
    @classmethod
    def validate_affordances_fit(cls, v: list[str], info) -> list[str]:
        """Ensure enough grid cells for affordances + agent."""
        grid_size = info.data.get("grid_size", 0)
        total_cells = grid_size * grid_size
        required_cells = len(v) + 1  # affordances + agent

        if required_cells > total_cells:
            raise ValueError(
                f"Grid too small: {grid_size}×{grid_size} = {total_cells} cells, "
                f"but need {required_cells} cells ({len(v)} affordances + 1 agent)"
            )

        return v
```

---

### Phase 3: Curriculum & Population Configuration DTOs (2-3 hours)

**Deliverable**: `src/townlet/config/curriculum_config.py` and `src/townlet/config/population_config.py`

```python
# curriculum_config.py
class CurriculumConfig(BaseModel):
    """Adversarial curriculum configuration."""

    # Stage parameters (ALL REQUIRED)
    max_steps_per_episode: int = Field(gt=0)
    survival_advance_threshold: float = Field(ge=0.0, le=1.0)
    survival_retreat_threshold: float = Field(ge=0.0, le=1.0)
    entropy_gate: float = Field(ge=0.0, le=1.0)
    min_steps_at_stage: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "CurriculumConfig":
        """Ensure retreat < advance thresholds."""
        if self.survival_retreat_threshold >= self.survival_advance_threshold:
            raise ValueError(
                f"survival_retreat_threshold ({self.survival_retreat_threshold}) "
                f"must be < survival_advance_threshold ({self.survival_advance_threshold})"
            )
        return self

# population_config.py
class PopulationConfig(BaseModel):
    """Population/agent configuration."""

    # Agent count (REQUIRED)
    num_agents: int = Field(gt=0)

    # Q-learning parameters (ALL REQUIRED)
    learning_rate: float = Field(gt=0.0)
    gamma: float = Field(ge=0.0, le=1.0)  # Discount factor
    replay_buffer_capacity: int = Field(gt=0)

    # Network architecture (REQUIRED)
    network_type: Literal["simple", "recurrent"]

    @model_validator(mode="after")
    def validate_network_for_observability(self) -> "PopulationConfig":
        """Warn if network type seems mismatched for observability."""
        # Note: This requires environment config context (handled by HamletConfig)
        return self
```

---

### Phase 4: Master Configuration DTO (1-2 hours)

**Deliverable**: `src/townlet/config/hamlet_config.py`

```python
from pathlib import Path
import yaml
from pydantic import BaseModel, model_validator

class HamletConfig(BaseModel):
    """Master configuration composing all sub-configs."""

    # Sub-configurations (ALL REQUIRED)
    training: TrainingConfig
    environment: EnvironmentConfig
    curriculum: CurriculumConfig
    population: PopulationConfig

    # Optional (can be None for headless training)
    cues: CuesConfig | None = None

    @model_validator(mode="after")
    def validate_cross_config_constraints(self) -> "HamletConfig":
        """Validate constraints across configs."""

        # Network type should match observability
        if self.environment.partial_observability and self.population.network_type != "recurrent":
            logger.warning(
                f"partial_observability=True but network_type={self.population.network_type}. "
                f"Consider network_type='recurrent' for POMDP."
            )

        # Batch size should be <= replay buffer capacity
        if self.training.batch_size > self.population.replay_buffer_capacity:
            raise ValueError(
                f"batch_size ({self.training.batch_size}) cannot exceed "
                f"replay_buffer_capacity ({self.population.replay_buffer_capacity})"
            )

        return self

def load_hamlet_config(config_dir: Path) -> HamletConfig:
    """
    Load and compile universe configuration from directory.

    Args:
        config_dir: Directory containing training.yaml, environment.yaml, etc.

    Returns:
        Validated HamletConfig

    Raises:
        ValidationError: If universe compilation fails
    """
    # Load all YAML files
    with open(config_dir / "training.yaml") as f:
        training_data = yaml.safe_load(f)

    with open(config_dir / "environment.yaml") as f:
        env_data = yaml.safe_load(f)

    # ... load other configs ...

    try:
        config = HamletConfig(
            training=TrainingConfig(**training_data["training"]),
            environment=EnvironmentConfig(**env_data["environment"]),
            # ... other configs ...
        )
        print("✅ Universe compiled successfully")
        return config
    except ValidationError as e:
        print("❌ UNIVERSE COMPILATION FAILED")
        print(e)
        raise
```

---

## Extension: TASK-002A Integration (SubstrateConfig)

**Context**: TASK-002A (Configurable Spatial Substrates) introduces new spatial configuration that needs DTO validation.

**Deliverable**: `src/townlet/environment/substrate_config.py`

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal

class GridConfig(BaseModel):
    """Grid substrate configuration."""
    topology: Literal["square", "cubic", "hexagonal"]  # Required
    dimensions: list[int]  # Required: [width, height] or [width, height, depth]
    boundary: Literal["clamp", "wrap", "bounce"]  # Required
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"]  # Required
    position_encoding: Literal["auto", "onehot", "coords", "fourier"] = "auto"  # Optional (default: auto)

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: list[int]) -> list[int]:
        if len(v) not in [2, 3]:
            raise ValueError(f"Grid dimensions must be 2D or 3D, got {len(v)}D")

        if any(dim < 1 for dim in v):
            raise ValueError(f"All dimensions must be >= 1, got {v}")

        if any(dim > 64 for dim in v):
            raise ValueError(f"Dimensions too large (max 64), got {v}")

        return v

    @model_validator(mode="after")
    def validate_encoding_for_3d(self) -> "GridConfig":
        """Forbid one-hot encoding for 3D grids (would be 512+ dimensions)."""
        if len(self.dimensions) == 3 and self.position_encoding == "onehot":
            raise ValueError(
                "One-hot encoding not supported for 3D grids (512+ dimensions). "
                "Use 'coords' or 'fourier' instead."
            )
        return self

class SubstrateConfig(BaseModel):
    """Spatial substrate configuration."""
    type: Literal["grid", "graph", "continuous", "aspatial"]  # Required
    grid: GridConfig | None = None  # Required if type="grid"

    @field_validator("grid")
    @classmethod
    def validate_grid_required(cls, v, info):
        substrate_type = info.data.get("type")
        if substrate_type == "grid" and v is None:
            raise ValueError("substrate.grid is required when type='grid'")
        if substrate_type != "grid" and v is not None:
            raise ValueError(f"substrate.grid not allowed for type='{substrate_type}'")
        return v
```

**Example valid config**:

```yaml
# configs/L1_full_observability/substrate.yaml
substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [8, 8]
    boundary: "clamp"
    distance_metric: "manhattan"
    position_encoding: "auto"  # Optional - defaults to "auto"
```

**Example invalid config (caught at compile time)**:

```yaml
# configs/my_3d_house/substrate.yaml
substrate:
  type: "grid"
  grid:
    topology: "cubic"
    dimensions: [8, 8, 3]
    boundary: "clamp"
    distance_metric: "manhattan"
    position_encoding: "onehot"  # ❌ ERROR: One-hot not supported for 3D
```

**Error message**:

```
❌ SUBSTRATE COMPILATION FAILED
ValidationError: substrate.grid.position_encoding
  One-hot encoding not supported for 3D grids (512+ dimensions).
  Use 'coords' or 'fourier' instead.

  Fix:
  position_encoding: "coords"  # 512 dims → 3 dims
```

---

## Extension: Position Field Validation for AffordanceConfig

**Context**: TASK-002A requires affordances to support flexible position formats (2D list, 3D list, hex dict, graph int).

**File to extend**: `src/townlet/environment/affordance_config.py`

```python
from pydantic import BaseModel, field_validator
from typing import Any

class AffordanceConfig(BaseModel):
    """Affordance configuration with optional positioning."""
    id: str
    name: str
    type: str
    effects: dict[str, float]

    # Position field - format depends on substrate type
    position: list[int] | dict[str, int] | int | None = None  # Optional

    @field_validator("position")
    @classmethod
    def validate_position_format(cls, v):
        """Validate position format.

        Valid formats:
        - list[int]: [x, y] for 2D, [x, y, z] for 3D
        - dict: {q: 3, r: 4} for hexagonal grids (axial coordinates)
        - int: node_id for graph substrates
        - None: randomize position (default)
        """
        if v is None:
            return v  # Randomize (default)

        if isinstance(v, list):
            if not all(isinstance(x, int) for x in v):
                raise ValueError(f"List position must contain only integers, got {v}")
            if len(v) not in [2, 3]:
                raise ValueError(f"List position must be 2D or 3D, got {len(v)}D: {v}")
            return v

        if isinstance(v, dict):
            # Hexagonal axial coordinates
            if set(v.keys()) != {"q", "r"}:
                raise ValueError(f"Dict position must have 'q' and 'r' keys, got {list(v.keys())}")
            if not all(isinstance(val, int) for val in v.values()):
                raise ValueError(f"Dict position values must be integers, got {v}")
            return v

        if isinstance(v, int):
            # Graph node ID
            if v < 0:
                raise ValueError(f"Graph node ID must be >= 0, got {v}")
            return v

        raise ValueError(f"Invalid position format: {type(v)}. Expected list, dict, int, or None.")
```

**Example configs**:

```yaml
# 2D square grid
affordances:
  - id: "Bed"
    position: [2, 5]  # Explicit 2D position
    effects: {energy: 0.2}

  - id: "Hospital"
    position: null  # Randomize (default)
    effects: {health: 0.2}

# 3D cubic grid
affordances:
  - id: "Bed"
    position: [2, 5, 0]  # Floor 0
    effects: {energy: 0.2}

  - id: "Job"
    position: [3, 1, 1]  # Floor 1
    effects: {money: 22.5}

# Hexagonal grid
affordances:
  - id: "Bed"
    position: {q: 3, r: 4}  # Axial coordinates
    effects: {energy: 0.2}

# Graph substrate
affordances:
  - id: "Bed"
    position: 0  # Node 0
    effects: {energy: 0.2}

  - id: "Hospital"
    position: 7  # Node 7
    effects: {health: 0.2}
```

**Benefits**:

1. ✅ **Flexible positioning**: Supports all substrate types (2D, 3D, hex, graph)
2. ✅ **Backward compatible**: `position: null` (or omit field) = randomize
3. ✅ **Type-safe**: Pydantic validates format at compile time
4. ✅ **Clear errors**: Invalid formats caught before training starts

---

## Design Philosophy: Permissive Semantics, Strict Syntax

The universe compiler follows this principle: **Enforce structure mercilessly, but remain conceptually agnostic about semantics.**

**❌ REJECT (syntax/structure violations)**:

```yaml
energy_move_depletion: "orange"  # Type error: expected float, got string
```

**✅ ALLOW (semantically unusual but structurally valid)**:

```yaml
# Current action energy costs (typical):
energy_move_depletion: 0.005   # Move costs 0.5% energy
energy_wait_depletion: 0.004   # Wait costs 0.4% energy
energy_interact_depletion: 0.003  # Interact costs 0.3% energy

# Operator adds "Rest" action that RESTORES energy:
energy_rest_depletion: -0.002  # Rest ADDS 0.2% energy (negative cost = gain)
```

**Why allow negative energy costs?**

An overzealous compiler might reject `energy_rest_depletion: -0.002` with "energy costs cannot be negative!" But this is **semantic overreach**. The compiler should be **conceptually agnostic**:

- The operator might be testing a "rest" action (light recovery without affordances)
- Or studying whether agents exploit free energy restoration
- Or running inverted universe experiments
- Or testing RL robustness to unusual reward structures

**The concept of "resting" is not something the engine should have an opinion on.** Everything comes from UNIVERSE_AS_CODE. The compiler validates the type (float), not the concept (whether rest makes sense).

**The boundary**:

- Compiler enforces: "`energy_rest_depletion` must be a float"
- Operator decides: "I set it to -0.002 (restoration) for my experiment"
- Compiler allows: Structurally valid float, regardless of sign
- Operator owns: Understanding what negative depletion means behaviorally

**Conceptual Agnosticism Principle**: The universe compiler should not encode assumptions about what makes a "reasonable" universe. It validates that the universe is **well-formed** (correct types, valid references), not **well-behaved** (sensible game design).

**Generalization Beyond HAMLET**: The universe compiler should be domain-agnostic. Today it models agents in a village (move, wait, interact with affordances). Tomorrow it might model:

- A sentient box jumping between conveyor belts in a factory (actions: `jump_left`, `jump_right`, `process_item`)
- A trading bot in financial markets (actions: `buy`, `sell`, `hold`)
- A cellular automaton (actions: `split`, `merge`, `signal`)

The compiler shouldn't hardcode concepts like "move" or "energy" or "affordances". Instead:

```python
# ❌ BAD: Hardcoded domain concepts
class UniverseCompiler:
    def validate(self, config):
        if "energy_move_depletion" not in config:
            raise Error("All universes must have move energy cost!")
        # ^ Assumes all universes have "moving" and "energy"

# ✅ GOOD: Domain-agnostic schema validation
class UniverseCompiler:
    def validate(self, config):
        for action in config.actions:
            if not isinstance(action.cost, float):
                raise Error(f"Action '{action.name}' cost must be float, got {type(action.cost)}")
        # ^ Validates structure without assuming what actions exist
```

For a sentient box factory simulation, the operator might define:

```yaml
actions:
  - name: "jump_left"
    cost: 0.1  # Processing cycles
  - name: "jump_right"
    cost: 0.1
  - name: "process_item"
    cost: 0.5

meters:
  - name: "processing_cycles"
    initial: 100.0
  - name: "items_processed"
    initial: 0.0

affordances:  # Now "conveyor belts" instead of "beds"
  - name: "ConveyorA"
    effects:
      - { meter: "items_processed", amount: 1.0 }
```

The universe compiler validates this identically to HAMLET configs:

- ✅ Are action costs floats?
- ✅ Do meter references exist?
- ✅ Are affordance effects well-formed?
- ❌ NOT: "Does 'jump_left' make sense?" (domain semantics)
- ❌ NOT: "Should processing cycles work this way?" (domain logic)

**The compiler is a universal validator for UNIVERSE_AS_CODE systems, not a HAMLET-specific validator.**

**Where structural constraints ARE appropriate**:

```yaml
# ✅ ENFORCE: Type constraints
energy_move_depletion: float  # Must be float, not string/list/dict

# ✅ ENFORCE: Reference validity
enabled_affordances: ["Bed", "Hospital"]  # Must exist in affordances.yaml

# ✅ ENFORCE: Physical impossibility
grid_size: 3
enabled_affordances: ["Bed", "Hospital", "Job", ...]  # 10 affordances
# ❌ REJECT: 3×3 = 9 cells, need 10 affordances + 1 agent = 11 cells (impossible)

# ❌ DON'T ENFORCE: Semantic "reasonableness"
energy_move_depletion: -0.5  # ✅ Allowed (might be testing inverted physics)
epsilon_decay: 0.5           # ✅ Allowed (might be testing rapid exploitation)
grid_size: 100               # ✅ Allowed (might be testing large spaces)
```

The line is: **Enforce what makes the universe mathematically/structurally coherent, not what makes it pedagogically sensible.**

**Another example**:

```yaml
# REJECT: Structure violation
enabled_affordances: ["Bed", "Zorblax"]  # ❌ "Zorblax" not in affordances.yaml

# ALLOW: Semantically unusual but valid
enabled_affordances: []  # ✅ Zero affordances (agent can only wait/move)
```

Zero affordances seems "broken" but might be testing pure navigation, or studying death from starvation with no recourse. The compiler validates the reference is structurally sound (list of strings), not whether it's a "sensible" experiment.

**Summary**:

- **Strict syntax**: `energy_cost: "orange"` → compilation error
- **Permissive semantics**: `energy_cost: -0.5` → allowed (operator's experiment)
- **Operator responsibility**: Understanding that negative costs = restoration mechanics

---

## No-Defaults Enforcement: Explicit Over Implicit

**Core Principle**: All behavioral parameters must be **explicitly specified** in config files. No implicit defaults allowed.

### Why No Defaults?

**Problem with defaults**:

```python
# ❌ BAD: Pydantic model with defaults
class TrainingConfig(BaseModel):
    epsilon_start: float = 1.0      # Default hidden in code
    epsilon_decay: float = 0.995    # Operator may not know this exists
    epsilon_min: float = 0.01       # Configs incomplete without reading source
```

**Consequences**:

1. **Hidden State**: Operator doesn't know what values are actually being used
2. **Drift Risk**: Code defaults change, old configs behave differently
3. **Non-Reproducible**: Two configs that look different may behave identically
4. **Incomplete Mental Model**: Can't reason about universe without reading source code
5. **Silent Breaking Changes**: Default changes break existing configs invisibly

### The Fix: Required Fields

```python
# ✅ GOOD: All parameters required
class TrainingConfig(BaseModel):
    """
    ALL FIELDS REQUIRED - enforces operator accountability.
    No hidden defaults. Config file is complete specification.
    """
    epsilon_start: float = Field(ge=0.0, le=1.0)  # No default!
    epsilon_decay: float = Field(gt=0.0, lt=1.0)  # No default!
    epsilon_min: float = Field(ge=0.0, le=1.0)     # No default!
```

**If operator omits a field**:

```
❌ UNIVERSE COMPILATION FAILED
training.yaml missing required field: 'epsilon_decay'

All training parameters must be explicitly specified.
This ensures you understand and control universe behavior.

Add to training.yaml:
  epsilon_decay: 0.995  # Or your preferred value (0.99 for fast, 0.995 for moderate)

Tip: See configs/templates/training.yaml.template for reference.
```

### Exemptions: When Defaults Are Allowed

Only three categories can have defaults:

**1. Truly Optional Features** (doesn't affect core simulation):

```python
cues: CuesConfig | None = None  # ✅ Visualization optional for headless training
description: str | None = None   # ✅ Metadata only
```

**2. Computed Values** (calculated from other configs):

```python
class UniverseMetadata(BaseModel):
    observation_dim: int  # ✅ Computed from grid_size + affordances + meters
    action_dim: int       # ✅ Computed from len(actions)
```

**3. Deprecated Fields** (temporary backwards compatibility):

```python
@deprecated("Use 'energy_cost' instead")
energy_depletion: float | None = None  # ✅ With loud warning
```

**Rule of thumb**: If omitting the field changes simulation behavior, it's REQUIRED.

### Config Templates

Provide annotated templates showing all required fields:

```yaml
# configs/templates/training.yaml.template
# ALL parameters REQUIRED - copy and customize for your universe

training:
  # Compute device
  device: cuda  # 'cuda' for GPU, 'cpu' for CPU-only

  # Training duration
  max_episodes: 5000  # Total episodes to train

  # Q-learning hyperparameters
  train_frequency: 4             # Train Q-network every N steps
  target_update_frequency: 100   # Update target network every N training steps
  batch_size: 64                 # Experience replay batch size
  max_grad_norm: 10.0            # Gradient clipping (prevents exploding gradients)

  # Epsilon-greedy exploration
  epsilon_start: 1.0   # Initial exploration rate (1.0 = 100% random)
  epsilon_decay: 0.995 # Decay per episode (0.995 → ε=0.1 at ep 460)
  epsilon_min: 0.01    # Minimum exploration (1% random floor)

# Decay formula: ε(n) = max(epsilon_start * epsilon_decay^n, epsilon_min)
# Fast learning (L0):     epsilon_decay: 0.99  (reaches ε=0.1 at ep 229)
# Moderate (L0.5/L1):     epsilon_decay: 0.995 (reaches ε=0.1 at ep 460)
# Slow (L2 POMDP):        epsilon_decay: 0.998 (reaches ε=0.1 at ep 1150)
```

### Benefits of No-Defaults

1. **Full Accountability**: Operator explicitly chooses every parameter
2. **Transparency**: Universe fully specified by visible config files (no hidden state)
3. **Reproducibility**: Config is complete specification (no need to read source code)
4. **Better Mental Model**: Operator must understand parameters to set them
5. **Prevents Drift**: Code changes don't silently alter existing universes
6. **Self-Documenting**: Config files show all available parameters
7. **Debugging Aid**: "What's my epsilon_decay?" → Just look at config file

**Slogan**: "If it affects the universe, it's in the config. No exceptions."

---

## Universe Compiler Design Pattern

The DTO-based approach implements a **universe compiler** pattern:

```python
# Phase 1: Parse (YAML → Python dicts)
raw_config = yaml.safe_load(config_file)

# Phase 2: Compile (validate schema, resolve references, check constraints)
try:
    universe = HamletConfig(**raw_config)  # DTO validation
except ValidationError as e:
    print("❌ UNIVERSE COMPILATION FAILED")
    print(e)
    sys.exit(1)

# Phase 3: Runtime (only if compilation succeeded)
print("✅ Universe compiled successfully")
env = VectorizedHamletEnv(universe.environment)
trainer = DemoRunner(universe.training)
```

**Compilation checks**:

- ✅ Structural: All required fields present
- ✅ Type safety: `epsilon_decay` is float, not string
- ✅ Range constraints: `epsilon_decay ∈ (0.0, 1.0)`
- ✅ Cross-references: `enabled_affordances` all exist in `affordances.yaml`
- ✅ Domain rules: `grid_size² > len(enabled_affordances) + 1` (room for agent)
- ✅ Performance warnings: `epsilon_decay > 0.999` (too slow)

**Not compilation checks** (those are runtime/empirical):

- ❌ Whether epsilon=0.99 converges faster than 0.995 (empirical)
- ❌ Whether 128 hidden units is "enough" (depends on task)
- ❌ Whether agent learns optimal policy (that's the experiment!)

The compiler validates **structure and constraints**, not **effectiveness**.

---

## Benefits

1. **Fail Fast**: Invalid configs rejected at load time, not runtime
   - Missing required field → clear error: "Field 'max_episodes' is required"
   - Type mismatch → clear error: "Expected float, got string for 'epsilon_decay'"
   - Out of range → clear error: "epsilon_decay=1.5 must be in range (0.0, 1.0)"

2. **Self-Documenting**: DTO is the schema
   - Field types, defaults, constraints in one place
   - IDE autocomplete works (`.epsilon_start` not `["epsilon_start"]`)
   - New developers read DTO to see available parameters

3. **Validation Rules**: Domain-specific constraints
   - Epsilon decay > 0.999? Warn about slow convergence
   - Batch size > replay buffer capacity? Error
   - Grid size < num_affordances? Error (no room on grid)

4. **Refactoring Safety**: Rename field in DTO → configs break loudly
   - Better than silent fallback to defaults
   - Type checker catches mismatches

5. **Version Control**: DTO changes tracked in git
   - PR shows "added new field, set default=X"
   - Config schema evolution is explicit

---

## Success Criteria

- [ ] TrainingConfig DTO created with no-defaults enforcement
- [ ] EnvironmentConfig DTO created with no-defaults enforcement
- [ ] CurriculumConfig DTO created with no-defaults enforcement
- [ ] PopulationConfig DTO created with no-defaults enforcement
- [ ] SubstrateConfig DTO created (TASK-002A integration)
- [ ] AffordanceConfig DTO created (BASIC - without capabilities)
- [ ] BarConfig DTO created
- [ ] CascadeConfig DTO created
- [ ] ActionConfig DTO created (TASK-002B integration)
- [ ] HamletConfig DTO created (master config)
- [ ] All L0-L3 configs load through DTOs
- [ ] Invalid configs rejected at load with clear error messages
- [ ] IDE autocomplete works for config field access
- [ ] Config templates created showing all required fields
- [ ] Position validation supports 2D/3D/hex/graph/aspatial
- [ ] All configs load through DTOs (no manual dict access)
- [ ] Adding new parameter requires updating DTO (forces documentation)
- [ ] CI can validate all configs in repo without running training
- [ ] **Validation Scope**: Core DTOs validate structural integrity (types, ranges, constraints)
- [ ] **Cross-file validation** (meter references, affordance IDs) deferred to TASK-004A

---

## Estimated Effort: 7-12 hours

**Breakdown**:

- Phase 1 (TrainingConfig): 2-4h
- Phase 2 (EnvironmentConfig): 2-3h
- Phase 3 (Curriculum/Population): 2-3h
- Phase 4 (Master Config): 1-2h
- SubstrateConfig (TASK-002A): Included in Phase 2
- Position validation: Included in Phase 2
- Testing/Documentation: Included in phases

**Confidence**: High (straightforward DTO creation following established patterns)

---

## Dependencies

**None** - This is a foundational task.

**Enables**:

- TASK-002A (Spatial Substrates) - Needs SubstrateConfig
- TASK-002B (Action Space) - Needs ActionConfig
- TASK-004A (Compiler) - Needs core schemas
- TASK-004B (Capabilities) - Extends AffordanceConfig

---

## Recommendation

Implement TASK-003 **before** TASK-004B. Core DTOs are foundational and enable L0-L3 configs. Capability system (TASK-004B) can be deferred until after core UAC is stable.

Start with Phase 1 (TrainingConfig) - it's where schema drift is most painful (epsilon params, Q-learning hyperparameters). Validate against existing L0/L0.5/L1 configs to ensure backwards compatibility.

---

## Alternatives Considered

1. **JSON Schema**: More verbose, less Pythonic, no runtime types
2. **dataclasses + manual validation**: Reinventing Pydantic
3. **Cerberus/Marshmallow**: Less type-safe, no IDE support
4. **Keep current approach**: Continue manual audits, runtime errors

---

## Appendix: Real Examples of Schema Drift Found

### Example 1: Missing Epsilon Config (L1)

**File**: `configs/L1_full_observability/training.yaml`

**Problem**: Entire epsilon-greedy section missing, fell back to hardcoded defaults in `runner.py`:

```python
epsilon_start = training_cfg.get("epsilon_start", 1.0)  # Silent default!
epsilon_decay = training_cfg.get("epsilon_decay", 0.995)
```

**Impact**: L1 used 0.995 decay by luck, but no way to know from config file what values were actually used.

**With DTO**: Would have either:

- Rejected config: "Missing required field 'epsilon_decay'"
- Or been explicit: "Using default epsilon_decay=0.995"

### Example 2: Misleading Comments (L0)

**File**: `configs/L0_0_minimal/training.yaml`

**Problem**:

```yaml
# Key features:
# - Tiny 5×5 grid (25 cells total)
environment:
  grid_size: 3  # 5×5 grid world (25 cells)  ❌ WRONG! Actually 3×3
```

**Impact**: Developer reads comments, expects 25 cells, actual grid has 9 cells.

**With DTO**: Comments can't lie about actual values. Schema documents what's valid.

### Example 3: Slow Epsilon Decay (L0)

**File**: `configs/L0_0_minimal/training.yaml`

**Problem**:

```yaml
epsilon_decay: 0.998  # Decay per episode
```

At episode 100 (target learning point), ε still 0.82 (82% exploration!). Agent can't exploit learned policy.

**Impact**: Agent takes 1150 episodes to reach ε=0.1, way beyond "should learn in <100 episodes" comment.

**With DTO**: Custom validator catches this:

```python
@model_validator(mode="after")
def validate_epsilon_decay(self) -> "TrainingConfig":
    if self.epsilon_decay > 0.999:
        raise ValueError(
            f"epsilon_decay={self.epsilon_decay} is too slow. "
            f"Will take {math.log(0.1)/math.log(self.epsilon_decay):.0f} episodes to reach ε=0.1"
        )
```

Error at load time: "epsilon_decay=0.998 is too slow. Will take 1150 episodes to reach ε=0.1. Consider 0.99 (L0), 0.995 (L0.5/L1)."

---

**End of TASK-003**
