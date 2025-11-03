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
    """Training hyperparameters configuration."""

    device: Literal["cpu", "cuda"] = "cuda"
    max_episodes: int = Field(gt=0)

    # Q-learning
    train_frequency: int = Field(default=4, gt=0)
    target_update_frequency: int = Field(default=100, gt=0)
    batch_size: int = Field(default=64, gt=0)
    max_grad_norm: float = Field(default=10.0, gt=0)

    # Epsilon-greedy exploration
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, gt=0.0, lt=1.0)
    epsilon_min: float = Field(default=0.01, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_epsilon_decay(self) -> "TrainingConfig":
        """Ensure epsilon decay is reasonable."""
        if self.epsilon_decay > 0.999:
            raise ValueError(
                f"epsilon_decay={self.epsilon_decay} is too slow. "
                f"Will take {math.log(0.1)/math.log(self.epsilon_decay):.0f} episodes to reach ε=0.1. "
                f"Consider 0.99 (L0), 0.995 (L0.5/L1), or 0.998 (L2)."
            )
        return self

def load_training_config(config_path: Path) -> TrainingConfig:
    """Load and validate training configuration."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**data["training"])
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

### Estimated Effort

- **Phase 1** (training config): 2-4 hours
- **Phase 2** (environment config): 2-3 hours
- **Phase 3** (curriculum/population): 2-3 hours
- **Phase 4** (master config): 1-2 hours
- **Total**: 7-12 hours

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
