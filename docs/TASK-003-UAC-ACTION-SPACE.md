# TASK-003: Move Action Space to UNIVERSE_AS_CODE

## Problem: Hardcoded Action Space Violates UAC Principle

### Current State

The action space (UP, DOWN, LEFT, RIGHT, WAIT, INTERACT) is **hardcoded in Python**, violating the core UNIVERSE_AS_CODE principle that "everything configurable."

**Current Implementation** (`src/townlet/environment/vectorized_env.py`):

```python
# Hardcoded action space
self.action_dim = 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT

# Hardcoded action handling in step()
if action == 0:  # UP
    new_positions[:, 1] = torch.clamp(positions[:, 1] - 1, 0, self.grid_size - 1)
elif action == 1:  # DOWN
    new_positions[:, 1] = torch.clamp(positions[:, 1] + 1, 0, self.grid_size - 1)
elif action == 2:  # LEFT
    new_positions[:, 0] = torch.clamp(positions[:, 0] - 1, 0, self.grid_size - 1)
elif action == 3:  # RIGHT
    new_positions[:, 0] = torch.clamp(positions[:, 0] + 1, 0, self.grid_size - 1)
elif action == 4:  # INTERACT
    # Interaction logic...
elif action == 5:  # WAIT
    # Wait logic...
```

### Why This Violates UAC

1. **No Domain Flexibility**: Cannot model universes without directional movement:
   - Sentient box in factory: `jump_left`, `jump_right` (no UP/DOWN)
   - Trading bot: `buy`, `sell`, `hold` (no spatial movement)
   - Cellular automaton: `split`, `merge`, `signal` (different topology)

2. **Hardcoded Assumptions**:
   - Movement must be cardinal directions
   - Grid must be 2D
   - Actions must have fixed indices (0=UP, 1=DOWN, etc.)
   - Energy costs hardcoded per action type

3. **Cannot Experiment**: Operator cannot:
   - Add diagonal movement (NE, NW, SE, SW)
   - Add "rest" action (negative energy cost)
   - Remove movement entirely (pure interaction game)
   - Change action costs per-level without code changes

4. **Action Costs Split**: Some in training.yaml, some hardcoded:
   ```yaml
   # training.yaml (partial)
   energy_move_depletion: 0.005
   energy_wait_depletion: 0.004
   energy_interact_depletion: 0.003
   ```
   But action space itself is Python code!

## Solution: Config-Driven Action Space

### Proposed Architecture

Define the complete action space in YAML configuration, making it a first-class citizen of UNIVERSE_AS_CODE.

**Create `actions.yaml` in config packs:**

```yaml
version: "1.0"
description: "Action space definition for HAMLET village simulation"

# Action space defines what the agent can DO in this universe
actions:
  # MOVEMENT ACTIONS
  # Note: Multi-meter costs pattern (matches affordances.yaml effects)
  - id: 0
    name: "UP"
    type: "movement"
    delta: [0, -1]  # (dx, dy) on grid
    costs:
      - {meter: energy, amount: 0.005}    # 0.5% energy per step
      - {meter: hygiene, amount: 0.003}   # Movement dirties agent
      - {meter: satiation, amount: 0.004} # Movement burns calories

  - id: 1
    name: "DOWN"
    type: "movement"
    delta: [0, 1]
    costs:
      - {meter: energy, amount: 0.005}
      - {meter: hygiene, amount: 0.003}
      - {meter: satiation, amount: 0.004}

  - id: 2
    name: "LEFT"
    type: "movement"
    delta: [-1, 0]
    costs:
      - {meter: energy, amount: 0.005}
      - {meter: hygiene, amount: 0.003}
      - {meter: satiation, amount: 0.004}

  - id: 3
    name: "RIGHT"
    type: "movement"
    delta: [1, 0]
    costs:
      - {meter: energy, amount: 0.005}
      - {meter: hygiene, amount: 0.003}
      - {meter: satiation, amount: 0.004}

  # INTERACTION ACTION
  - id: 4
    name: "INTERACT"
    type: "interaction"
    costs:
      - {meter: energy, amount: 0.003}  # Minimal interaction cost
    description: "Interact with affordance at current position"

  # PASSIVE ACTION
  - id: 5
    name: "WAIT"
    type: "passive"
    costs:
      - {meter: energy, amount: 0.004}  # Idle metabolic cost
    description: "Wait in place (low energy recovery)"

# Metadata
topology: "grid2d"  # Could be: grid2d, graph, continuous, discrete
boundary: "clamp"   # How to handle out-of-bounds: clamp, wrap, bounce, fail
```

### Alternative Universe Examples

**Example 1: Sentient Box Factory**

```yaml
# configs/factory_box/actions.yaml
version: "1.0"
description: "Sentient box jumping between conveyor belts"

actions:
  - id: 0
    name: "JUMP_LEFT"
    type: "movement"
    delta: [-1, 0]  # Jump to left belt
    energy_cost: 0.1  # Processing cycles

  - id: 1
    name: "JUMP_RIGHT"
    type: "movement"
    delta: [1, 0]  # Jump to right belt
    energy_cost: 0.1

  - id: 2
    name: "PROCESS_ITEM"
    type: "interaction"
    energy_cost: 0.5  # Higher cost for processing
    description: "Process item on current conveyor"

  - id: 3
    name: "INSPECT"
    type: "passive"
    energy_cost: 0.05
    description: "Observe current belt state"

topology: "grid1d"  # One-dimensional conveyor line
boundary: "clamp"
```

**Example 2: Trading Bot**

```yaml
# configs/trading_bot/actions.yaml
version: "1.0"
description: "Financial trading agent"

actions:
  - id: 0
    name: "BUY"
    type: "transaction"
    effects:
      - { meter: "cash", amount: -1.0 }  # Variable, depends on price
      - { meter: "holdings", amount: 1.0 }
    energy_cost: 0.01  # Transaction fees

  - id: 1
    name: "SELL"
    type: "transaction"
    effects:
      - { meter: "cash", amount: 1.0 }
      - { meter: "holdings", amount: -1.0 }
    energy_cost: 0.01

  - id: 2
    name: "HOLD"
    type: "passive"
    energy_cost: 0.0  # No cost to wait
    description: "Do nothing this timestep"

topology: "discrete"  # No spatial movement
boundary: "none"
```

**Example 3: HAMLET with Diagonal Movement**

```yaml
# configs/hamlet_8way/actions.yaml
version: "1.0"
description: "HAMLET with 8-directional movement"

actions:
  # Cardinal directions
  - { id: 0, name: "UP", type: "movement", delta: [0, -1], energy_cost: 0.005 }
  - { id: 1, name: "DOWN", type: "movement", delta: [0, 1], energy_cost: 0.005 }
  - { id: 2, name: "LEFT", type: "movement", delta: [-1, 0], energy_cost: 0.005 }
  - { id: 3, name: "RIGHT", type: "movement", delta: [1, 0], energy_cost: 0.005 }

  # Diagonal directions (longer distance = higher cost)
  - { id: 4, name: "UP_LEFT", type: "movement", delta: [-1, -1], energy_cost: 0.007 }
  - { id: 5, name: "UP_RIGHT", type: "movement", delta: [1, -1], energy_cost: 0.007 }
  - { id: 6, name: "DOWN_LEFT", type: "movement", delta: [-1, 1], energy_cost: 0.007 }
  - { id: 7, name: "DOWN_RIGHT", type: "movement", delta: [1, 1], energy_cost: 0.007 }

  - { id: 8, name: "INTERACT", type: "interaction", costs: [{meter: energy, amount: 0.003}] }
  - { id: 9, name: "WAIT", type: "passive", costs: [{meter: energy, amount: 0.004}] }
  - id: 10
    name: "REST"
    type: "passive"
    costs:
      - {meter: energy, amount: -0.002}  # RESTORES energy (negative cost)
      - {meter: mood, amount: -0.01}     # RESTORES mood
    description: "Rest in place (slow recovery without affordances)"

topology: "grid2d"
boundary: "clamp"
```

### Implementation Plan

#### Phase 1: Create Action Config Schema

**File**: `src/townlet/environment/action_config.py`

```python
from pydantic import BaseModel, Field
from typing import Literal

class ActionEffect(BaseModel):
    """Effect of an action on a meter."""
    meter: str
    amount: float

class ActionCost(BaseModel):
    """Cost applied to a meter when action is taken."""
    meter: str
    amount: float  # Can be negative (restoration)

class ActionConfig(BaseModel):
    """Single action definition."""
    id: int = Field(ge=0)
    name: str
    type: Literal["movement", "interaction", "passive", "transaction"]

    # Movement-specific
    delta: list[int] | None = None  # [dx, dy] for grid movement

    # Multi-meter costs (replaces single energy_cost)
    # Pattern consistency: matches affordances.yaml effects structure
    costs: list[ActionCost] = Field(default_factory=list)

    # DEPRECATED: Single-meter energy cost (auto-migrated to costs)
    # Kept for backward compatibility with existing configs
    energy_cost: float | None = None

    # Additional effects (e.g., side effects beyond costs)
    effects: list[ActionEffect] = Field(default_factory=list)

    # Metadata
    description: str | None = None

    @model_validator(mode="after")
    def migrate_energy_cost(self) -> "ActionConfig":
        """Auto-migrate legacy energy_cost to costs list."""
        if self.energy_cost is not None and not self.costs:
            logger.warning(
                f"Action '{self.name}': energy_cost field deprecated. "
                f"Use 'costs: [{{'meter': 'energy', 'amount': {self.energy_cost}}}]' instead."
            )
            # Auto-migrate
            self.costs = [ActionCost(meter="energy", amount=self.energy_cost)]
        return self

class ActionSpaceConfig(BaseModel):
    """Complete action space definition."""
    version: str
    description: str
    actions: list[ActionConfig]

    # Topology metadata
    topology: Literal["grid2d", "grid1d", "graph", "continuous", "discrete"]
    boundary: Literal["clamp", "wrap", "bounce", "fail", "none"]

    @model_validator(mode="after")
    def validate_action_ids(self) -> "ActionSpaceConfig":
        """Ensure action IDs are contiguous from 0."""
        ids = [a.id for a in self.actions]
        if ids != list(range(len(ids))):
            raise ValueError(f"Action IDs must be contiguous from 0, got {ids}")
        return self

    @model_validator(mode="after")
    def validate_movement_actions(self) -> "ActionSpaceConfig":
        """Ensure movement actions have delta defined."""
        for action in self.actions:
            if action.type == "movement" and action.delta is None:
                raise ValueError(f"Movement action '{action.name}' must define delta")
            if action.type != "movement" and action.delta is not None:
                raise ValueError(f"Non-movement action '{action.name}' cannot have delta")
        return self

def load_action_config(config_path: Path) -> ActionSpaceConfig:
    """Load and validate action space configuration."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return ActionSpaceConfig(**data)
```

#### Phase 2: Update VectorizedEnv to Use Action Config

**File**: `src/townlet/environment/vectorized_env.py`

```python
# BEFORE: Hardcoded
self.action_dim = 6

# AFTER: Load from config
action_config_path = config_pack_path / "actions.yaml"
self.action_config = load_action_config(action_config_path)
self.action_dim = len(self.action_config.actions)

# Build action lookup
self.actions_by_id = {a.id: a for a in self.action_config.actions}
```

**Update step() logic**:

```python
# BEFORE: Hardcoded if/elif chain
if action == 0:  # UP
    new_positions[:, 1] = torch.clamp(positions[:, 1] - 1, 0, self.grid_size - 1)
# ... etc

# AFTER: Config-driven dispatch
for agent_idx in range(self.num_agents):
    action_id = actions[agent_idx].item()
    action_config = self.actions_by_id[action_id]

    if action_config.type == "movement":
        # Apply delta from config
        dx, dy = action_config.delta
        new_x = positions[agent_idx, 0] + dx
        new_y = positions[agent_idx, 1] + dy

        # Apply boundary rules from config
        if self.action_config.boundary == "clamp":
            new_x = torch.clamp(new_x, 0, self.grid_size - 1)
            new_y = torch.clamp(new_y, 0, self.grid_size - 1)
        elif self.action_config.boundary == "wrap":
            new_x = new_x % self.grid_size
            new_y = new_y % self.grid_size

        new_positions[agent_idx] = torch.tensor([new_x, new_y])

    elif action_config.type == "interaction":
        # Interaction logic (unchanged)
        pass

    elif action_config.type == "passive":
        # Passive action (wait/rest)
        pass

    # Apply action costs (multi-meter)
    for cost in action_config.costs:
        meter_idx = self.meter_name_to_idx[cost.meter]
        self.meters[agent_idx, meter_idx] -= cost.amount  # Note: negative amounts = restoration
```

#### Phase 3: Migrate Existing Configs

Update all existing config packs to include `actions.yaml`:

1. **L0_minimal**: 4-way movement + interact + wait
2. **L0_5_dual_resource**: 4-way movement + interact + wait
3. **L1_full_observability**: 4-way movement + interact + wait

Initial configs should replicate current behavior exactly (6 actions with current costs).

#### Phase 4: Remove Hardcoded Action Logic

Delete hardcoded action handling from:
- `vectorized_env.py` (if/elif chains)
- `training.yaml` (move energy costs to actions.yaml)

### Benefits

1. **Full UAC Compliance**: Action space now defined in config, not code
2. **Domain Flexibility**: Can model any universe (villages, factories, markets, etc.)
3. **Experimentation**: Operators can:
   - Add/remove actions without code changes
   - Modify energy costs per-config
   - Test diagonal movement, rest actions, etc.
4. **Single Source of Truth**: Action space and costs in one file
5. **Compile-Time Validation**: Invalid action configs rejected at load time

### Migration Strategy

**Backwards Compatibility**: Initially, actions.yaml is optional. If missing, fall back to hardcoded 6-action space with warning:

```python
try:
    action_config = load_action_config(config_pack_path / "actions.yaml")
except FileNotFoundError:
    logger.warning("No actions.yaml found, using legacy hardcoded action space")
    action_config = get_legacy_action_space()  # Returns current 6 actions
```

This allows gradual migration without breaking existing configs.

### Success Criteria

- [ ] `actions.yaml` schema defined with Pydantic DTOs
- [ ] All existing configs have `actions.yaml` (replicating current behavior)
- [ ] `vectorized_env.py` dispatches actions via config, not hardcoded logic
- [ ] Can add "REST" action via config without code changes
- [ ] Can create factory box config with different action space
- [ ] Action space compilation errors caught at load time
- [ ] All tests pass with new action system

### Estimated Effort

**Core Action Space (Original Scope)**:
- **Phase 1** (schema): 2-3 hours
- **Phase 2** (env integration): 4-6 hours
- **Phase 3** (migrate configs): 1-2 hours
- **Phase 4** (cleanup): 1-2 hours
- **Subtotal (Original)**: 8-13 hours

**Multi-Meter Costs Extension** (from research findings):
- **Schema update** (ActionCost DTO, costs field): +1 hour
- **Environment refactor** (apply multi-meter costs): +2-3 hours
- **Subtotal (Extension)**: +3-4 hours

**Total**: 11-17 hours (+38-31% from original estimate)

### Risks

- **Network Architecture**: Q-network output dim must match action_dim
  - Mitigation: Load actions.yaml before network creation
- **Breaking Changes**: Existing code expects action IDs (0=UP, etc.)
  - Mitigation: Backwards compatibility layer during migration
- **Performance**: Config dispatch might be slower than hardcoded if/elif
  - Mitigation: Pre-build action lookup tables, benchmark before/after

### Pattern Consistency: Actions vs Affordances

**Added**: 2025-11-04 (from research findings)

Actions and affordances share similar cost/effect patterns to maintain consistency:

**Actions (TASK-003)**:
- **Categorization**: Single `type` field (movement, interaction, passive, transaction)
- **Costs**: `costs: [{meter, amount}]` (multi-meter pattern)
- **Trigger**: Agent chooses action each timestep
- **Examples**: UP, DOWN, INTERACT, WAIT, REST

**Affordances (TASK-002)**:
- **Categorization**: Multiple composable `capabilities` (multi_tick, cooldown, meter_gated, etc.)
- **Effects**: Multi-stage `effect_pipeline` (on_start, per_tick, on_completion, etc.)
- **Trigger**: Agent must be at affordance position and choose INTERACT
- **Examples**: Bed, Hospital, Job, Bar

**Why Different?**
- Actions are **primitive** (one type = clear semantics: movement vs interaction vs passive)
- Affordances are **compound** (multiple capabilities = rich behaviors: multi-tick + cooldown + meter-gated)
- Actions are **always available** (can always move/wait)
- Affordances are **conditionally available** (operating hours, resource gates)

**Pattern Alignment**:
Both use `{meter, amount}` structure for costs/effects:
```yaml
# Action cost (depletes meters)
costs:
  - {meter: energy, amount: 0.005}
  - {meter: hygiene, amount: 0.003}

# Affordance effect (restores meters)
effect_pipeline:
  on_completion:
    - {meter: energy, amount: 0.2}
    - {meter: hygiene, amount: 0.15}
```

This consistency enables:
- Shared validation logic (TASK-004 validates meter references identically)
- Easier mental model (operators learn pattern once, apply twice)
- Future unification (actions could gain effect_pipeline if needed)

**Permissive Semantics Example** (REST action):
```yaml
# REST action with negative costs = restoration
- id: 10
  name: "REST"
  type: "passive"
  costs:
    - {meter: energy, amount: -0.002}  # RESTORES energy
    - {meter: mood, amount: -0.01}     # RESTORES mood
```

Negative costs are structurally valid (float type) even if semantically unusual. The compiler validates structure, not "reasonableness" (see TASK-002 Design Philosophy).

### Design Principles (from TASK-001)

**Conceptual Agnosticism**: The action config schema should NOT assume:
- ❌ Actions must include movement
- ❌ Movement must be on a 2D grid
- ❌ Energy costs must be positive
- ❌ There must be an INTERACT action

**Structural Enforcement**: The schema MUST enforce:
- ✅ Action IDs are contiguous from 0
- ✅ Energy costs are floats (can be negative)
- ✅ Movement actions define delta
- ✅ All meter references exist in bars.yaml

**Permissive Semantics, Strict Syntax**:
- ✅ Allow: `energy_cost: -0.002` (rest action restores energy)
- ✅ Allow: `actions: []` (no actions = pure observation task)
- ❌ Reject: `energy_cost: "orange"` (type violation)
- ❌ Reject: `delta: [1]` (movement needs 2D delta for grid2d)

### Relationship to TASK-001

TASK-000 is a **dependency** for TASK-001. The action space is currently the largest violation of UAC principle. By moving actions to config first, we:

1. **Prove the concept**: Show that config-driven systems work at scale
2. **Build momentum**: Demonstrate value before tackling training/environment configs
3. **Reduce scope**: TASK-001 becomes smaller (fewer hardcoded things to migrate)

**Recommended order**: TASK-000 → TASK-001 (actions first, then training/env configs).

## Appendix: Current Action Space Hardcoding

**Hardcoded locations in codebase**:

```python
# src/townlet/environment/vectorized_env.py:171
self.action_dim = 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT

# src/townlet/environment/vectorized_env.py:300-340
# Hardcoded action dispatch (50 lines of if/elif)

# src/townlet/agent/networks.py:45
self.q_head = nn.Linear(hidden_dim, action_dim)  # Must match action space

# configs/L0_minimal/training.yaml:28-30
# Energy costs defined but action space is Python
energy_move_depletion: 0.005
energy_wait_depletion: 0.003
energy_interact_depletion: 0.0029
```

All of these would be replaced with config-driven action space from `actions.yaml`.
