# Tasking Statement: Config Schema Enforcement with DTOs

## Guiding Principle: The Universe Compiler

**Core Philosophy**: Every aspect of the universe should be configurable via config files (UNIVERSE_AS_CODE), **BUT** that schema must be enforced mercilessly with versioning and compile-time checks.

**Mental Model**: Configuration files are not just data - they are **programs that define the universe**. Before running a training session, we must **compile the universe** and validate it against the universe's laws (the schema). This is not testing *values* (whether epsilon=0.99 is "good"), but testing **structural compliance** ("does this config speak the universe's language?").

Think of it as:
1. **Write universe program** (edit YAML configs)
2. **Compile universe** (load into Pydantic DTOs)
3. **Compilation errors** = schema violations, invalid references, constraint violations
4. **Runtime** = only if universe compiles successfully

Just like you wouldn't run Python code with syntax errors, you shouldn't run training with schema-invalid configs.

### The No-Defaults Principle: Full Operator Accountability

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

## Solution: DTO-Based Config Schema Enforcement

### Proposed Architecture

Use **Pydantic data transfer objects (DTOs)** as the single source of truth for config schemas, similar to existing `affordance_config.py` and `cascade_config.py` patterns.

### Example Implementation

**Create `src/townlet/training/training_config.py`:**

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import math

class TrainingConfig(BaseModel):
    """
    Training hyperparameters configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.
    Operator must explicitly specify all parameters that affect training.
    """

    # Compute device (REQUIRED)
    device: Literal["cpu", "cuda"]

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

**Update `runner.py` to use DTOs:**

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

### Benefits

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

### Implementation Plan

1. **Phase 1: Training Config**
   - Create `training_config.py` DTO
   - Update `runner.py` to use it
   - Validate all level configs (L0, L0.5, L1, L2)

2. **Phase 2: Environment Config**
   - Create `environment_config.py` DTO
   - Validate grid_size, vision_range, enabled_affordances, etc.
   - Add constraint: `grid_size² > len(enabled_affordances) + 1`

3. **Phase 3: Curriculum/Population Config**
   - Create DTOs for curriculum and population sections
   - Validate learning_rate, gamma, network_type, etc.

4. **Phase 4: Master Config**
   - Create `HamletConfig` that composes all sub-configs
   - Single entry point: `load_hamlet_config(dir) -> HamletConfig`
   - All validation happens at load time

### Alternatives Considered

1. **JSON Schema**: More verbose, less Pythonic, no runtime types
2. **dataclasses + manual validation**: Reinventing Pydantic
3. **Cerberus/Marshmallow**: Less type-safe, no IDE support
4. **Keep current approach**: Continue manual audits, runtime errors

### Success Criteria

- [ ] All configs load through DTOs (no manual dict access)
- [ ] Invalid config rejected at load with clear error message
- [ ] IDE autocomplete works for config field access
- [ ] Adding new parameter requires updating DTO (forces documentation)
- [ ] CI can validate all configs in repo without running training
- [ ] **No-Defaults Enforcement**: All behavioral parameters marked as required (no default values)
- [ ] **Explicit Configs**: All existing config files specify all required parameters explicitly
- [ ] **Config Templates**: Reference templates created showing all required fields with documentation
- [ ] **Clear Error Messages**: Missing required fields produce helpful error messages with examples

### Estimated Effort

**Core DTOs (Original Scope)**:
- **Phase 1** (training config): 2-4 hours
- **Phase 2** (environment config): 2-3 hours
- **Phase 3** (curriculum/population): 2-3 hours
- **Phase 4** (master config): 1-2 hours
- **Subtotal (Core)**: 7-12 hours

**Extended Schema (Capability System Integration)**:
- **Phase 5** (affordance masking + capability DTOs): 4-6 hours
- **Phase 6** (effect pipeline DTOs): 2-3 hours
- **Phase 7** (validation rules): 2-3 hours
- **Subtotal (Extensions)**: 8-12 hours

**Total**: 15-24 hours

**Note**: Due to significant scope expansion (+8-12h, +114-200% increase), consider splitting into:
- **TASK-002A** (Core UAC Contracts): 7-12h - Core DTOs for training, environment, curriculum
- **TASK-002B** (Capability System): 8-12h - Affordance masking, capabilities, effect pipeline

This allows incremental progress and clearer milestones.

### Risks

- **Migration**: Must update all config files if DTO adds required field (breaking change)
- **Strictness**: May catch old configs that "worked" due to defaults
- **Learning curve**: Team must learn Pydantic patterns

### Recommendation

**Implement incrementally starting with training config** - it's where schema drift is most painful (epsilon params, Q-learning hyperparameters). Validate against existing L0/L0.5/L1 configs to ensure backwards compatibility. Add new fields as optional with sensible defaults to avoid breaking existing configs.

This follows HAMLET's core principle:

> **Everything configurable. Schema enforced mercilessly.**
>
> The universe is compiled, not interpreted. Schema violations are compilation errors, not runtime surprises.

### Universe Compiler Design Pattern

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

### Design Philosophy: Permissive Semantics, Strict Syntax

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

### No-Defaults Enforcement: Explicit Over Implicit

**Core Principle**: All behavioral parameters must be **explicitly specified** in config files. No implicit defaults allowed.

#### Why No Defaults?

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

#### The Fix: Required Fields

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

#### Exemptions: When Defaults Are Allowed

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

#### Config Templates

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

#### Benefits of No-Defaults

1. **Full Accountability**: Operator explicitly chooses every parameter
2. **Transparency**: Universe fully specified by visible config files (no hidden state)
3. **Reproducibility**: Config is complete specification (no need to read source code)
4. **Better Mental Model**: Operator must understand parameters to set them
5. **Prevents Drift**: Code changes don't silently alter existing universes
6. **Self-Documenting**: Config files show all available parameters
7. **Debugging Aid**: "What's my epsilon_decay?" → Just look at config file

**Slogan**: "If it affects the universe, it's in the config. No exceptions."

---

## Extension: Validation Schemas for TASK-000 (Spatial Substrates)

**Added**: 2025-11-04 (from research findings)
**Related Research**: `docs/research/RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md`

TASK-000 introduces new configuration files that require DTO validation:

### substrate.yaml Validation Schema

**File to create**: `src/townlet/environment/substrate_config.py`

```python
from pydantic import BaseModel, field_validator
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

    @field_validator("position_encoding")
    @classmethod
    def validate_encoding_for_3d(cls, v: str, info) -> str:
        """Forbid one-hot encoding for 3D grids (would be 512+ dimensions)."""
        dimensions = info.data.get("dimensions", [])
        if len(dimensions) == 3 and v == "onehot":
            raise ValueError(
                "One-hot encoding not supported for 3D grids (512+ dimensions). "
                "Use 'coords' or 'fourier' instead."
            )
        return v

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

### affordances.yaml Position Field Validation

**File to extend**: `src/townlet/environment/affordance_config.py`

```python
from pydantic import BaseModel, field_validator
from typing import Literal, Any

class AffordanceConfig(BaseModel):
    """Affordance configuration with optional positioning."""
    id: str
    name: str
    type: str
    effects: dict[str, float]

    # Position field - format depends on substrate type
    position: list[int] | dict[str, int] | int | None = None  # Optional

    # ... other fields ...

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

## Extension: Affordance Masking and Capability System

**Added**: 2025-11-04 (from research findings)
**Related Research**: `docs/research/RESEARCH-ENVIRONMENT-CONSTANTS-EXPOSURE.md`, `docs/research/RESEARCH-INTERACTION-TYPE-REGISTRY.md`

The core DTO structure must be extended to support:
1. **Affordance masking** based on meter values (operating hours, resource gates)
2. **Capability composition** for rich interaction patterns (multi-tick, cooldown, etc.)
3. **Effect pipelines** with lifecycle stages (on_start, per_tick, on_completion, etc.)

### Quick Navigation (This Section)

- [Affordance Masking Schema](#affordance-masking-schema) - Operating hours, resource gates (lines 776-855)
- [Capability Composition System](#capability-composition-system) - Multi-tick, cooldown, skill scaling (lines 856-919)
- [Effect Pipeline](#effect-pipeline) - Lifecycle stages and multi-stage effects (lines 920-1065)
- [Backward Compatibility](#backward-compatibility) - Migration from legacy configs (lines 1065-1092)

### Quick Reference: Core Capabilities

| Capability | Key Parameters | Use Case Example |
|------------|----------------|------------------|
| `multi_tick` | `duration_ticks`, `early_exit_allowed`, `resumable` | Job (10 ticks), University (20 ticks) |
| `cooldown` | `cooldown_ticks`, `scope` | Prevent spamming (Job cooldown 50 ticks) |
| `meter_gated` | `meter`, `min`, `max` | Gym requires energy >0.2, Hospital if health <0.5 |
| `skill_scaling` | `skill`, `base_multiplier`, `max_multiplier` | Gym effectiveness scales with fitness |
| `probabilistic` | `success_probability` | Gambling (30% success), Dating (50% success) |
| `prerequisite` | `required_affordances` | University Sophomore requires Freshman completion |

### Affordance Masking Schema

**Gap Identified**: Current affordances are always available. No support for:
- Operating hours (Job 9am-5pm, Bar 6pm-2am)
- Resource gates (Gym requires energy > 0.3)
- Mode switching (Coffee Shop vs Bar, same location)

**Solution**: Add `availability` and `modes` fields to AffordanceConfig.

```python
from pydantic import BaseModel, Field, model_validator

class BarConstraint(BaseModel):
    """Meter-based availability constraint."""
    meter: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_min_less_than_max(self) -> "BarConstraint":
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self

class ModeConfig(BaseModel):
    """Mode-specific configuration (e.g., operating hours)."""
    hours: tuple[int, int] | None = None  # (start_hour, end_hour), e.g., (9, 17) for 9am-5pm
    effects: dict[str, float] | None = None  # Mode-specific effect overrides

class AffordanceConfig(BaseModel):
    """Affordance configuration with availability masking and capabilities."""
    id: str
    name: str
    type: str
    effects: dict[str, float]

    # NEW: Availability conditions
    availability: list[BarConstraint] | None = None

    # NEW: Mode switching
    modes: dict[str, ModeConfig] | None = None

    # ... existing fields ...
```

**Example Config (Operating Hours)**:
```yaml
# Job affordance with operating hours
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

    # ONLY available 6pm-2am
    modes:
      night_hours:
        hours: [18, 2]  # 6pm-2am (wraps midnight)
```

**Validation Rules** (implemented in TASK-004):
- Meter references in `availability` must exist
- `min < max` enforced
- Hour ranges validated (0-23, wrapping allowed)

### Capability Composition System

**Gap Identified**: Interaction patterns are hardcoded (instant, multi-tick, cooldown). Cannot combine patterns (e.g., multi-tick + cooldown + meter-gated).

**Solution**: Replace single `type` field with composable `capabilities` list.

```python
from typing import Literal

class MultiTickCapability(BaseModel):
    """Multi-tick interaction (takes N ticks to complete)."""
    type: Literal["multi_tick"]
    duration_ticks: int = Field(gt=0)
    early_exit_allowed: bool = False
    resumable: bool = False

class CooldownCapability(BaseModel):
    """Cooldown period after interaction."""
    type: Literal["cooldown"]
    cooldown_ticks: int = Field(gt=0)
    scope: Literal["agent", "global"] = "agent"

class MeterGatedCapability(BaseModel):
    """Requires meter within range to interact."""
    type: Literal["meter_gated"]
    meter: str
    min: float | None = None
    max: float | None = None

class SkillScalingCapability(BaseModel):
    """Effect scales with skill level."""
    type: Literal["skill_scaling"]
    skill: str
    base_multiplier: float = 1.0
    max_multiplier: float = 2.0

class ProbabilisticCapability(BaseModel):
    """Probabilistic success/failure."""
    type: Literal["probabilistic"]
    success_probability: float = Field(ge=0.0, le=1.0)

class PrerequisiteCapability(BaseModel):
    """Requires prior interaction completion."""
    type: Literal["prerequisite"]
    required_affordances: list[str]

CapabilityConfig = (
    MultiTickCapability |
    CooldownCapability |
    MeterGatedCapability |
    SkillScalingCapability |
    ProbabilisticCapability |
    PrerequisiteCapability
)

class AffordanceConfig(BaseModel):
    """Affordance with capability composition."""
    id: str
    name: str

    # NEW: Capability composition (replaces single 'type' field)
    capabilities: list[CapabilityConfig] | None = None

    # ... other fields ...
```

**Example Config (Capability Composition)**:
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
```

**Validation Rules** (implemented in TASK-004):
- **Capability conflicts**: `instant` and `multi_tick` are mutually exclusive
- **Dependent capabilities**: `resumable` requires `multi_tick`
- **Meter references**: `meter_gated` meters must exist
- **Prerequisite references**: `prerequisite` affordances must exist

### Effect Pipeline System

**Gap Identified**: Current affordances have simple `effects` dict. Cannot model:
- On-start costs (Job entry fee)
- Per-tick incremental rewards (Job pays per hour)
- Completion bonuses (Job completion bonus)
- Early-exit penalties (quit Job early → mood penalty)
- Failure effects (probabilistic failure → different outcome)

**Solution**: Replace `effects` dict with multi-stage `effect_pipeline`.

```python
class AffordanceEffect(BaseModel):
    """Single effect on a meter."""
    meter: str
    amount: float

class EffectPipeline(BaseModel):
    """Multi-stage effect application."""
    on_start: list[AffordanceEffect] = Field(default_factory=list)
    per_tick: list[AffordanceEffect] = Field(default_factory=list)
    on_completion: list[AffordanceEffect] = Field(default_factory=list)
    on_early_exit: list[AffordanceEffect] = Field(default_factory=list)
    on_failure: list[AffordanceEffect] = Field(default_factory=list)

class AffordanceConfig(BaseModel):
    """Affordance with effect pipeline."""
    id: str
    name: str
    capabilities: list[CapabilityConfig] | None = None

    # NEW: Effect pipeline (replaces simple 'effects' dict)
    effect_pipeline: EffectPipeline | None = None

    # DEPRECATED: Keep for backward compatibility (emit warning if used)
    effects: dict[str, float] | None = None

    @model_validator(mode="after")
    def migrate_effects_to_pipeline(self) -> "AffordanceConfig":
        """Migrate legacy 'effects' to 'effect_pipeline.on_completion'."""
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

**Example Config (Complete Job with Effect Pipeline)**:
```yaml
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
        - {meter: money, amount: 5.0}   # Completion bonus
        - {meter: social, amount: 0.02}  # Coworker interaction

      # On early exit (penalties)
      on_early_exit:
        - {meter: mood, amount: -0.05}  # Quitting early feels bad
```

**Validation Rules** (implemented in TASK-004):
- **Pipeline consistency**: `multi_tick` capability requires `per_tick` or `on_completion` effects
- **Meter references**: All effect meters must exist
- **Mutual exclusivity**: If using `effect_pipeline`, `effects` field should be empty (or auto-migrated)

### Backward Compatibility Strategy

To avoid breaking existing configs:

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
            logger.warning(f"{self.id}: Auto-migrated 'effects' to 'effect_pipeline.on_completion'")

        # Migrate 'type' → 'capabilities' (if type field existed)
        # (Not needed - current system doesn't have 'type' field)

        return self
```

### Schema Validation Summary

**TASK-002 Deliverables** (DTOs):
- [x] `BarConstraint` DTO (meter-based availability)
- [x] `ModeConfig` DTO (operating hours, mode switching)
- [x] `MultiTickCapability`, `CooldownCapability`, etc. (6 capability DTOs)
- [x] `EffectPipeline` DTO (multi-stage effects)
- [x] Extended `AffordanceConfig` with `availability`, `modes`, `capabilities`, `effect_pipeline`

**TASK-004 Deliverables** (Validation):
- [ ] Validate `availability` meter references (Stage 4)
- [ ] Validate capability conflicts (Stage 4)
- [ ] Validate capability meter references (Stage 3)
- [ ] Validate effect pipeline consistency (Stage 4)
- [ ] Validate prerequisite affordance references (Stage 3)

**Total Effort**: +8-12 hours for TASK-002 schema extensions

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

**File**: `configs/L0_minimal/training.yaml`

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

**File**: `configs/L0_minimal/training.yaml`

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
