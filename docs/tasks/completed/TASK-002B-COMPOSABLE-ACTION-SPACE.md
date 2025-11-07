# TASK-002B: Composable Action Space (Substrate + Custom Actions)

**STATUS**: 📝 **READY FOR IMPLEMENTATION** (2025-11-06)

**Replaces**: TASK-002B-UAC-ACTION-SPACE (retired - conflicted with substrate abstraction)

---

## Problem: Action Space Missing Custom Actions and Extensibility

### Current State (Post-TASK-002A)

**Substrate-defined actions work correctly**, but operators cannot:
- Add custom actions (REST, MEDITATE - substrate-agnostic actions)
- Override default action costs per-config
- Define domain-specific actions (inventory, social, combat)

**Current Implementation** (`src/townlet/substrate/base.py:59`):

```python
@property
def action_space_size(self) -> int:
    """Action space determined by substrate dimensionality."""
    if self.position_dim == 0:
        return 2  # Aspatial: INTERACT + WAIT
    return 2 * self.position_dim + 2  # Spatial: ±movement per dim + INTERACT + WAIT
```

**This works** but is **not extensible**:
- ✅ Grid2D → 6 actions (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT)
- ✅ Grid3D → 8 actions (add UP_Z/DOWN_Z)
- ✅ GridND(7D) → 16 actions (14 movement + INTERACT + WAIT)
- ❌ Cannot add REST action (passive energy recovery)
- ❌ Cannot add MEDITATE action (mood boost)
- ❌ Cannot override movement costs per-config

---

## Solution: Composable Action Space

### Key Insight: Actions Come from Three Sources

**Action Space = Substrate Actions + Custom Actions + (Future) Affordance Actions**

```
┌─────────────────────────────────────────────────────────────┐
│                   COMPOSED ACTION SPACE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐   ┌─────────────────┐   ┌──────────┐ │
│  │ SUBSTRATE        │   │ CUSTOM          │   │ AFFORD.  │ │
│  │ (REQUIRED)       │ + │ (OPTIONAL)      │ + │ (FUTURE) │ │
│  ├──────────────────┤   ├─────────────────┤   ├──────────┤ │
│  │ UP, DOWN         │   │ REST            │   │ (deferred│ │
│  │ LEFT, RIGHT      │   │ MEDITATE        │   │ to       │ │
│  │ INTERACT, WAIT   │   │ (substrate-     │   │ TASK-003)│ │
│  │                  │   │  agnostic only) │   │          │ │
│  └──────────────────┘   └─────────────────┘   └──────────┘ │
│                                                              │
│  Grid2D: 6 substrate + 2 custom = 8 total actions           │
│  Grid3D: 8 substrate + 2 custom = 10 total actions          │
│  GridND(7D): 14 substrate + 2 custom = 16 total actions     │
└─────────────────────────────────────────────────────────────┘
```

---

### 1. Substrate Actions (REQUIRED - Already Implemented)

**What**: Movement + core interactions derived from substrate topology

**Authority**: Substrate owns this. Automatically generated based on dimensionality.

**Examples**:

```python
# Grid2D contributes 6 actions:
[UP, DOWN, LEFT, RIGHT, INTERACT, WAIT]

# Grid3D contributes 8 actions:
[UP, DOWN, LEFT, RIGHT, UP_Z, DOWN_Z, INTERACT, WAIT]

# GridND(7D) contributes 16 actions:
[DIM0_NEG, DIM0_POS, DIM1_NEG, DIM1_POS, ..., DIM6_POS, INTERACT, WAIT]

# Aspatial contributes 2 actions:
[INTERACT, WAIT]  # No movement (position_dim=0)
```

**Substrate method** (to be added):
```python
class SpatialSubstrate(ABC):
    @abstractmethod
    def get_default_actions(self) -> list[ActionConfig]:
        """Return substrate's default action space with default costs."""
        pass
```

---

### 2. Custom Actions (OPTIONAL - This Task)

**What**: Operator-defined domain-specific actions

**Authority**: Config file (`custom_actions.yaml`)

**Examples**:

```yaml
# configs/L1_full_observability/custom_actions.yaml
version: "1.0"
description: "Custom actions for L1 curriculum level"

actions:
  # Passive recovery (weak version of Bed)
  - name: "REST"
    type: "passive"
    costs: {energy: -0.002, mood: -0.01}  # Negative = restoration
    description: "Rest in place (slower than Bed, but available anywhere)"

  # Mental health action
  - name: "MEDITATE"
    type: "passive"
    costs: {energy: 0.001}  # Small cost
    effects: {mood: 0.02}   # Mood boost
    description: "Meditate to improve mood"

  # Movement shortcut (expensive)
  - name: "TELEPORT_HOME"
    type: "movement"
    teleport_to: [0, 0]  # Warp to origin
    costs: {energy: 0.5, money: 10.0}
    description: "Instantly return to home position (expensive)"
```

**Supported Custom Action Types**:

1. **Passive actions**: Manipulate bars without movement
   - REST (restore energy/mood)
   - MEDITATE (restore mood)
   - FOCUS (restore mental_clarity, if added to bars)

2. **Movement actions**: Non-standard movement
   - TELEPORT_HOME (warp to fixed position)
   - JUMP (move 2 cells in one action)
   - DASH (fast movement with higher cost)

3. **Future action types** (extensibility):
   - **Inventory actions**: DROP_ITEM, USE_ITEM (requires inventory system)
   - **Social actions**: GIFT_MONEY, CHAT (requires multi-agent)
   - **Combat actions**: ATTACK, DEFEND (requires combat system)

---

### 3. Affordance Actions (FUTURE - Deferred to TASK-003)

**What**: Affordances optionally provide "remote" actions

**Problem to solve first**: Action space stability across curriculum levels
- L0: Bed only → provides REST action → action_dim = 7
- L1: All 14 affordances → 14 remote actions → action_dim = 20
- **Network incompatibility**: Can't transfer 7-action checkpoint to 20-action space

**Deferred**: Affordances only work via INTERACT for now. Remote actions added after solving curriculum transfer.

---

## Proposed Architecture

### File Structure

```
configs/
├── global_actions.yaml     # SHARED: Complete action vocabulary for curriculum
│
├── L0_0_minimal/
│   ├── substrate.yaml      # Grid2D (contributes 6 substrate actions)
│   ├── bars.yaml
│   ├── cascades.yaml
│   ├── affordances.yaml
│   ├── cues.yaml
│   └── training.yaml       # Includes enabled_actions list
│
└── L1_full_observability/
    ├── substrate.yaml      # Grid2D (same substrate actions)
    ├── bars.yaml
    ├── cascades.yaml
    ├── affordances.yaml
    ├── cues.yaml
    └── training.yaml       # Includes enabled_actions list (more enabled than L0)
```

**Key decisions**:
- **`global_actions.yaml`**: Defines complete action vocabulary (substrate + custom) shared across **all configs**
- **`training.yaml`**: Each config specifies which actions from the global vocabulary are **enabled**
- **Fixed action_dim**: All configs have same action space size, enabling checkpoint transfer

---

### ActionSpaceBuilder

**File**: `src/townlet/environment/action_builder.py` (new)

```python
"""Composes action space from substrate + custom actions."""

from pathlib import Path
from typing import Optional

import torch

from townlet.substrate.base import SpatialSubstrate
from townlet.environment.action_config import ActionConfig, ActionSpaceConfig


class ActionSpaceBuilder:
    """Composes action space from global vocabulary.

    Action Space = Substrate Actions + Custom Actions (from global_actions.yaml)
    Enabled Actions = Subset specified in training.yaml

    CRITICAL: All curriculum levels share the SAME action vocabulary.
    This enables checkpoint transfer (same action_dim across configs).

    Examples:
        Global: 6 substrate + 4 custom = 10 total actions (all configs)
        L0: 7 enabled (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT/REST)
        L1: 10 enabled (all actions available)
    """

    def __init__(
        self,
        substrate: SpatialSubstrate,
        global_actions_path: Path,
        enabled_action_names: Optional[list[str]] = None,
    ):
        """Initialize action space builder.

        Args:
            substrate: Spatial substrate (provides substrate actions)
            global_actions_path: Path to configs/global_actions.yaml
            enabled_action_names: List of action names to enable (from training.yaml)
                                 If None, all actions are enabled.
        """
        self.substrate = substrate
        self.global_actions_path = global_actions_path
        self.enabled_action_names = set(enabled_action_names) if enabled_action_names else None

    def build(self) -> "ComposedActionSpace":
        """Build complete action space from global vocabulary."""
        actions = []
        action_id = 0

        # === 1. SUBSTRATE ACTIONS (REQUIRED) ===
        substrate_actions = self.substrate.get_default_actions()
        for action in substrate_actions:
            action.id = action_id
            action.source = "substrate"
            action.enabled = self._is_enabled(action.name)
            actions.append(action)
            action_id += 1

        # === 2. CUSTOM ACTIONS (from global_actions.yaml) ===
        custom_action_count = 0
        if self.global_actions_path.exists():
            custom_actions = self._load_global_custom_actions()
            for action in custom_actions:
                action.id = action_id
                action.source = "custom"
                action.enabled = self._is_enabled(action.name)
                actions.append(action)
                action_id += 1
            custom_action_count = len(custom_actions)

        # === 3. AFFORDANCE ACTIONS (FUTURE - Deferred to TASK-003) ===
        # Will be added to global_actions.yaml when implemented

        return ComposedActionSpace(
            actions=actions,
            substrate_action_count=len(substrate_actions),
            custom_action_count=custom_action_count,
            affordance_action_count=0,  # Future
            enabled_action_names=self.enabled_action_names,
        )

    def _is_enabled(self, action_name: str) -> bool:
        """Check if action is enabled in this config."""
        if self.enabled_action_names is None:
            return True  # All actions enabled if not specified
        return action_name in self.enabled_action_names

    def _load_global_custom_actions(self) -> list[ActionConfig]:
        """Load custom actions from global_actions.yaml."""
        import yaml

        with open(self.global_actions_path) as f:
            data = yaml.safe_load(f)

        # Global file contains custom actions only (substrate provides its own)
        return [ActionConfig(**action_data) for action_data in data.get("custom_actions", [])]


class ComposedActionSpace:
    """Composed action space with metadata about sources.

    CRITICAL: All curriculum levels use the SAME action space (same action_dim).
    Disabled actions are masked out but still occupy action IDs.

    Attributes:
        actions: Complete action list (substrate + custom + affordance)
        substrate_action_count: Number from substrate (6 for Grid2D, 8 for Grid3D)
        custom_action_count: Number from global_actions.yaml
        affordance_action_count: Number from affordances (future)
        enabled_action_names: Set of action names enabled in this config
    """

    def __init__(
        self,
        actions: list[ActionConfig],
        substrate_action_count: int,
        custom_action_count: int,
        affordance_action_count: int,
        enabled_action_names: Optional[set[str]] = None,
    ):
        self.actions = actions
        self.substrate_action_count = substrate_action_count
        self.custom_action_count = custom_action_count
        self.affordance_action_count = affordance_action_count
        self.enabled_action_names = enabled_action_names

    @property
    def action_dim(self) -> int:
        """Total number of actions (including disabled ones).

        CRITICAL: This is the SAME across all curriculum levels.
        """
        return len(self.actions)

    @property
    def enabled_action_count(self) -> int:
        """Number of enabled actions in this config."""
        return sum(1 for a in self.actions if a.enabled)

    def get_action_by_id(self, action_id: int) -> ActionConfig:
        """Get action by ID."""
        return self.actions[action_id]

    def get_action_by_name(self, name: str) -> ActionConfig:
        """Get action by name."""
        for action in self.actions:
            if action.name == name:
                return action
        raise ValueError(f"Action '{name}' not found")

    def get_enabled_actions(self) -> list[ActionConfig]:
        """Get only enabled actions."""
        return [a for a in self.actions if a.enabled]

    def get_disabled_actions(self) -> list[ActionConfig]:
        """Get only disabled actions."""
        return [a for a in self.actions if not a.enabled]

    def get_substrate_actions(self) -> list[ActionConfig]:
        """Get only substrate-provided actions."""
        return [a for a in self.actions if a.source == "substrate"]

    def get_custom_actions(self) -> list[ActionConfig]:
        """Get only custom actions."""
        return [a for a in self.actions if a.source == "custom"]

    def get_base_action_mask(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Get base action mask (disabled actions masked out).

        Args:
            num_agents: Number of agents
            device: PyTorch device

        Returns:
            [num_agents, action_dim] bool tensor
            False = action disabled, True = action available
        """
        mask = torch.ones(num_agents, self.action_dim, dtype=torch.bool, device=device)

        # Mask out disabled actions
        for action_id, action in enumerate(self.actions):
            if not action.enabled:
                mask[:, action_id] = False

        return mask
```

---

### ActionConfig Schema Updates

**File**: `src/townlet/environment/action_config.py`

```python
"""Action configuration schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ActionConfig(BaseModel):
    """Single action definition.

    Can come from:
    - Substrate (movement, INTERACT, WAIT)
    - Custom config (REST, MEDITATE, TELEPORT_HOME)
    - Affordances (future)
    """

    id: int = Field(ge=0, description="Action ID (assigned by builder)")
    name: str = Field(min_length=1, description="Action name (UP, DOWN, REST, etc.)")
    type: Literal["movement", "interaction", "passive", "transaction"]

    # Costs: Multi-meter pattern (matches affordances.yaml effects structure)
    costs: dict[str, float] = Field(
        default_factory=dict,
        description="Meter costs: {meter_name: amount}. Negative = restoration.",
    )

    # Effects: Additional meter changes beyond costs
    effects: dict[str, float] = Field(
        default_factory=dict,
        description="Meter effects: {meter_name: amount}. For actions with benefits.",
    )

    # Movement-specific
    delta: Optional[list[int]] = Field(
        None,
        description="Movement delta [dx, dy] or [dx, dy, dz] for standard movement",
    )
    teleport_to: Optional[list[int]] = Field(
        None,
        description="Teleport destination [x, y] or [x, y, z]. Overrides delta.",
    )

    # Enabled/disabled state (for curriculum progression)
    enabled: bool = Field(
        True,
        description="Whether this action is enabled in current config (for masking)",
    )

    # Metadata
    description: Optional[str] = Field(None, description="Human-readable description")
    icon: Optional[str] = Field(None, max_length=10, description="Emoji for UI")
    source: Literal["substrate", "custom", "affordance"] = Field(
        "custom",
        description="Where this action came from",
    )
    source_affordance: Optional[str] = Field(
        None,
        description="If source='affordance', which affordance provided it",
    )
```

---

## Implementation Plan

### Phase 1: Substrate Default Actions (4-6 hours)

**Add `get_default_actions()` to all substrates**

**Files to modify**:
- `src/townlet/substrate/grid2d.py`
- `src/townlet/substrate/grid3d.py`
- `src/townlet/substrate/gridnd.py`
- `src/townlet/substrate/continuous.py`
- `src/townlet/substrate/continuousnd.py`
- `src/townlet/substrate/aspatial.py`

**Example implementation (Grid2D)**:

```python
# src/townlet/substrate/grid2d.py

from townlet.environment.action_config import ActionConfig

class Grid2DSubstrate(SpatialSubstrate):
    def get_default_actions(self) -> list[ActionConfig]:
        """Return Grid2D's 6 default actions with default costs."""
        return [
            ActionConfig(
                id=0,  # Temporary ID, will be reassigned by builder
                name="UP",
                type="movement",
                delta=[0, -1],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell upward (north)",
            ),
            ActionConfig(
                id=1,
                name="DOWN",
                type="movement",
                delta=[0, 1],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell downward (south)",
            ),
            ActionConfig(
                id=2,
                name="LEFT",
                type="movement",
                delta=[-1, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell left (west)",
            ),
            ActionConfig(
                id=3,
                name="RIGHT",
                type="movement",
                delta=[1, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell right (east)",
            ),
            ActionConfig(
                id=4,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
            ),
            ActionConfig(
                id=5,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
            ),
        ]
```

**Testing**: Verify each substrate returns correct action count and structure.

---

### Phase 2: ActionSpaceBuilder (4-5 hours)

**Create builder and composed action space**

**Files to create**:
- `src/townlet/environment/action_builder.py` (new)
- `src/townlet/environment/action_config.py` (update with new fields)

**Testing**:
```python
def test_action_space_builder_substrate_only():
    """Builder with no custom actions should return substrate actions only."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")
    builder = ActionSpaceBuilder(substrate=substrate)

    action_space = builder.build()

    assert action_space.action_dim == 6  # Grid2D default
    assert action_space.substrate_action_count == 6
    assert action_space.custom_action_count == 0


def test_action_space_builder_with_custom():
    """Builder with custom actions should compose substrate + custom."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")
    custom_config = Path("configs/test_custom_actions/custom_actions.yaml")

    builder = ActionSpaceBuilder(substrate=substrate, custom_config_path=custom_config)

    action_space = builder.build()

    assert action_space.action_dim == 8  # 6 substrate + 2 custom
    assert action_space.substrate_action_count == 6
    assert action_space.custom_action_count == 2

    # Substrate actions come first (IDs 0-5)
    assert action_space.get_action_by_id(0).name == "UP"
    assert action_space.get_action_by_id(0).source == "substrate"

    # Custom actions come after (IDs 6-7)
    assert action_space.get_action_by_id(6).name == "REST"
    assert action_space.get_action_by_id(6).source == "custom"
```

---

### Phase 3: Environment Integration (6-8 hours)

**Update VectorizedHamletEnv to use ActionSpaceBuilder**

**Files to modify**:
- `src/townlet/environment/vectorized_env.py`

**Changes**:

```python
# In __init__:
def __init__(self, config_pack_path: Path, ...):
    # ... load substrate ...

    # Build composed action space
    custom_actions_path = config_pack_path / "custom_actions.yaml"
    builder = ActionSpaceBuilder(
        substrate=self.substrate,
        custom_config_path=custom_actions_path if custom_actions_path.exists() else None,
    )
    self.action_space = builder.build()
    self.action_dim = self.action_space.action_dim

    # Pre-build action cost tensor (from addendum)
    self._build_action_cost_tensor()
```

**Add custom action dispatch**:

```python
def _apply_custom_action(self, agent_idx: int, action: ActionConfig):
    """Apply custom action effects.

    Args:
        agent_idx: Agent index
        action: Custom action config
    """
    # Apply costs
    for meter_name, cost in action.costs.items():
        meter_idx = self._get_meter_index(meter_name)
        if meter_idx is not None:
            self.meters[agent_idx, meter_idx] -= cost  # Negative costs = restoration

    # Apply effects
    for meter_name, effect in action.effects.items():
        meter_idx = self._get_meter_index(meter_name)
        if meter_idx is not None:
            self.meters[agent_idx, meter_idx] += effect

    # Handle teleportation
    if action.teleport_to is not None:
        target_pos = torch.tensor(action.teleport_to, device=self.device)
        self.positions[agent_idx] = target_pos
```

**Update step() to dispatch custom actions**:

```python
def step(self, actions: torch.Tensor, ...):
    # ...existing logic...

    # Identify custom actions
    custom_action_start_id = self.action_space.substrate_action_count
    custom_mask = actions >= custom_action_start_id

    if custom_mask.any():
        custom_agent_indices = torch.where(custom_mask)[0]
        for agent_idx in custom_agent_indices:
            action_id = actions[agent_idx].item()
            action = self.action_space.get_action_by_id(action_id)
            self._apply_custom_action(agent_idx, action)
```

---

### Phase 4: Create Example Configs (2-3 hours)

**Create custom_actions.yaml for all curriculum levels**

**Files to create**:
- `configs/L0_0_minimal/custom_actions.yaml`
- `configs/L0_5_dual_resource/custom_actions.yaml`
- `configs/L1_full_observability/custom_actions.yaml`
- `configs/L2_partial_observability/custom_actions.yaml`
- `configs/L3_temporal_mechanics/custom_actions.yaml`

**Example (L1)**:

```yaml
# configs/L1_full_observability/custom_actions.yaml
version: "1.0"
description: "Custom actions for Level 1 - Full Observability"

actions:
  - name: "REST"
    type: "passive"
    costs:
      energy: -0.002  # Restores 0.2% energy per step
      mood: -0.01     # Restores 1% mood per step
    description: "Rest in place (weaker than Bed, but available anywhere)"
    icon: "😴"

  - name: "MEDITATE"
    type: "passive"
    costs:
      energy: 0.001   # Costs 0.1% energy
    effects:
      mood: 0.02      # Restores 2% mood
    description: "Meditate to improve mood without sleeping"
    icon: "🧘"
```

**Example (L0_0_minimal - no custom actions)**:

```yaml
# configs/L0_0_minimal/custom_actions.yaml
version: "1.0"
description: "No custom actions for L0 (pure substrate actions)"

actions: []  # Empty - only substrate actions (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT)
```

---

### Phase 5: Testing (4-6 hours)

**Test matrix**: Substrate × Custom Actions

| Substrate | Substrate Actions | Custom Actions | Total |
|-----------|-------------------|----------------|-------|
| Grid2D    | 6                 | 0              | 6     |
| Grid2D    | 6                 | 2              | 8     |
| Grid3D    | 8                 | 2              | 10    |
| GridND(7D)| 16                | 2              | 18    |
| Aspatial  | 2                 | 2              | 4     |

**Integration tests**:

```python
def test_rest_action_restores_energy():
    """REST custom action should restore energy (negative cost)."""
    env = create_env_with_custom_actions(["REST"])
    env.reset()

    # Drain energy
    env.meters[0, env.energy_idx] = 0.5
    initial_energy = env.meters[0, env.energy_idx].item()

    # Execute REST action (ID = 6 for Grid2D with 6 substrate actions)
    rest_action_id = env.action_space.substrate_action_count
    actions = torch.tensor([rest_action_id], device=env.device)
    env.step(actions)

    final_energy = env.meters[0, env.energy_idx].item()

    assert final_energy > initial_energy  # Energy restored
    assert abs((final_energy - initial_energy) - 0.002) < 1e-6  # Exact amount


def test_teleport_home_warps_position():
    """TELEPORT_HOME custom action should warp agent to origin."""
    env = create_env_with_custom_actions(["TELEPORT_HOME"])
    env.reset()

    # Move agent away from origin
    env.positions[0] = torch.tensor([5, 7], device=env.device)

    # Execute TELEPORT_HOME (ID = 6)
    teleport_action_id = env.action_space.substrate_action_count
    actions = torch.tensor([teleport_action_id], device=env.device)
    env.step(actions)

    # Agent should be at [0, 0]
    assert torch.equal(env.positions[0], torch.tensor([0, 0], device=env.device))
```

---

## Benefits

### 1. Extensibility
- ✅ Operators can add custom actions without code changes
- ✅ Different curriculum levels can have different action sets
- ✅ Future-proof for inventory, social, combat systems

### 2. UAC Compliance
- ✅ Custom actions defined in YAML (UNIVERSE_AS_CODE)
- ✅ Substrate actions auto-generated (no manual config for GridND)
- ✅ Single source of truth per action source

### 3. Backward Compatibility
- ✅ `custom_actions.yaml` is optional (defaults to substrate only)
- ✅ Existing configs work without modification
- ✅ No breaking changes to VectorizedHamletEnv API

### 4. Composability
- ✅ Clear separation: substrate geometry vs domain logic
- ✅ Action IDs assigned deterministically (substrate first, then custom)
- ✅ Easy to debug (know which source an action came from)

---

## Design Decisions

### 0. Two Substrate-Agnostic Custom Actions (Not Four) - IMPLEMENTATION DECISION

**Original Plan**: 4 custom actions (REST, MEDITATE, TELEPORT_HOME, SPRINT)
**Final Implementation**: 2 custom actions (REST, MEDITATE)

**Rationale**: Global vocabulary must work on ALL substrates for curriculum transfer

**Why TELEPORT_HOME and SPRINT were removed**:

1. **TELEPORT_HOME breaks on non-Grid2D substrates**:
   - Original design: `teleport_to: [0, 0]` (2D coordinates)
   - Grid3D requires: `teleport_to: [0, 0, 0]` (3D coordinates)
   - GridND(7D) requires: `teleport_to: [0, 0, 0, 0, 0, 0, 0]` (7D coordinates)
   - Continuous substrates: float coordinates instead of integers
   - Aspatial: No concept of position at all
   - **Problem**: Fixed 2D coordinate in global vocabulary breaks universal curriculum

2. **SPRINT breaks on non-Grid2D substrates**:
   - Original design: `delta: [0, -2]` (move 2 cells north on Grid2D)
   - Grid3D: Which axis is "north"? [0, -2, 0]?
   - GridND(7D): Meaningless without axis specification
   - Continuous substrates: Movement uses `movement_delta` scaling, not integer deltas
   - Aspatial: No movement possible
   - **Problem**: Grid2D-specific movement delta breaks universal curriculum

**Why REST and MEDITATE work universally**:
- Pure meter manipulation (costs/effects on energy, mood)
- No spatial assumptions (no position, delta, or teleportation)
- Work on ALL substrate types:
  - ✅ Grid2D/3D/ND: Meter effects apply regardless of position dimensionality
  - ✅ Continuous substrates: Meter effects apply regardless of coordinate system
  - ✅ Aspatial: Meter effects apply even without position

**Benefits of substrate-agnostic global vocabulary**:
- ✅ Same action_dim across ALL configs (checkpoint transfer works)
- ✅ No conditional logic in ActionSpaceBuilder (simpler, cleaner)
- ✅ Operator can still add substrate-specific actions in per-config files (future extensibility)

**Action Counts (Implementation)**:
- Grid2D: 6 substrate + 2 custom = **8 actions total** (not 10)
- Grid3D: 8 substrate + 2 custom = **10 actions total** (not 12)
- GridND(7D): 14 substrate + 2 custom = **16 actions total** (not 18)
- Aspatial: 2 substrate + 2 custom = **4 actions total** (not 6)

**Future Work**: Substrate-specific actions (TELEPORT_HOME, SPRINT) can be added in per-config `custom_actions_grid2d.yaml` files for operators who want them, but they won't be in the global vocabulary.

---

### 1. Fixed Action Space with Masking (CRITICAL FOR CURRICULUM TRANSFER)

**Decision**: All curriculum levels share the **same action vocabulary**. Unavailable actions are **masked out** but still take up action IDs.

**Rationale**: Enables checkpoint transfer across curriculum levels
- L0_0_minimal: action_dim = 8 (6 substrate + 2 custom)
- L1_full_observability: action_dim = 8 (same vocabulary)
- **Checkpoints transferable**: Q-network has same output dimension

**How it works**:

```yaml
# global_actions.yaml (SHARED across all configs)
version: "1.0"
description: "Global action vocabulary for entire curriculum"

custom_actions:
  # Substrate actions provided by substrate.get_default_actions()
  # Custom actions (substrate-agnostic only)
  - name: "REST"
    type: "passive"
    costs: {energy: -0.002, mood: -0.01}

  - name: "MEDITATE"
    type: "passive"
    costs: {energy: 0.001}
    effects: {mood: 0.02}
```

```yaml
# configs/L0_0_minimal/training.yaml (future - enabled_actions feature)
enabled_actions:
  # Substrate actions (always enabled)
  - "UP"
  - "DOWN"
  - "LEFT"
  - "RIGHT"
  - "INTERACT"
  - "WAIT"

  # Custom actions (only REST enabled for L0)
  - "REST"
  # MEDITATE disabled (but ID reserved for curriculum transfer)
```

```yaml
# configs/L1_full_observability/training.yaml (future - enabled_actions feature)
enabled_actions:
  # All actions enabled for L1
  - "UP"
  - "DOWN"
  - "LEFT"
  - "RIGHT"
  - "INTERACT"
  - "WAIT"
  - "REST"
  - "MEDITATE"
```

**Action masking**:

```python
def get_action_masks(self) -> torch.Tensor:
    """Return action masks [num_agents, action_dim].

    Disabled actions return False (cannot be selected).
    """
    masks = torch.ones(self.num_agents, self.action_dim, dtype=torch.bool, device=self.device)

    # Mask out disabled actions
    for action_idx, action in enumerate(self.action_space.actions):
        if action.name not in self.enabled_action_names:
            masks[:, action_idx] = False  # Disable this action for all agents

    # ... existing masking logic (dead agents, temporal mechanics, etc.)

    return masks
```

**Benefits**:
- ✅ **Checkpoint transfer**: L0 model can transfer to L1 (same action_dim)
- ✅ **Curriculum progression**: Enable more actions as agent improves
- ✅ **Pedagogical value**: Students see action space grow with capabilities
- ✅ **Stable network architecture**: Q-network output dim never changes

**Trade-off**:
- ❌ Unused action IDs (L0 has 8 actions defined, 7 enabled if MEDITATE disabled)
- **Acceptable**: Curriculum transfer > memory efficiency

---

### 2. Why Separate Files? (substrate.yaml vs custom_actions.yaml)

**Rationale**: Separation of concerns
- **substrate.yaml**: Defines geometry → implies movement actions
- **custom_actions.yaml**: Defines domain logic → custom behaviors

**Alternative considered**: Single `actions.yaml` with composition spec
- Rejected: Mixing substrate geometry and custom actions in one file is confusing

---

### 3. Why Action IDs Assigned by Builder? (Not in YAML)

**Rationale**: Avoid ID conflicts across sources

**Bad** (manual IDs):
```yaml
# custom_actions.yaml
actions:
  - id: 6  # What if substrate changes to 8 actions? Conflict!
    name: "REST"
```

**Good** (builder assigns):
```yaml
# custom_actions.yaml
actions:
  - name: "REST"  # Builder assigns ID = substrate_action_count
```

---

### 4. Why Defer Affordance Actions?

**Rationale**: Need to solve fixed vocabulary pattern first

**Problem**:
- Affordances could contribute 0-N actions depending on which are enabled
- With fixed vocabulary approach, we need to decide upfront which affordances contribute

**Decision**: Implement substrate + custom actions first (this task), add affordance actions in TASK-003 after validating the masking pattern works.

---

### 5. Operator Accountability (Permissive Semantics)

**Decision**: The operator is **fully accountable** for their action space design choices.

**What we validate** (structural):
- ✅ YAML syntax is valid
- ✅ Meter references exist in bars.yaml
- ✅ Action IDs are assigned correctly
- ✅ Required fields are present

**What we DON'T validate** (semantic):
- ❌ "This action space is too large" (100D with 200 actions? That's your choice)
- ❌ "These costs seem unbalanced" (energy: 0.00001? That's your experiment)
- ❌ "You shouldn't allow teleportation" (Your universe, your rules)
- ❌ "Fixed vocabulary won't transfer well" (You chose it, you own the consequences)

**Rationale**: **Permissive Semantics, Strict Syntax** (from TASK-001)
- Operators learn by experiencing the consequences of their choices
- "Interesting failures" are pedagogically valuable
- Framework should enable experiments, not prevent them

**Examples of permissive behaviors**:

```yaml
# ✅ ALLOWED: Massive action space (operator's problem if training fails)
actions:
  - name: "TELEPORT_TO_EVERY_CELL"  # 64 actions for 8×8 grid
    # ... 64 teleport actions defined ...

# ✅ ALLOWED: Negative costs on movement (makes movement profitable)
- name: "UP"
  costs: {energy: -0.01}  # Walking RESTORES energy? Weird but allowed

# ✅ ALLOWED: Custom action that references future meter
- name: "FOCUS"
  effects: {mental_clarity: 0.05}  # Meter doesn't exist yet
  # Will fail at runtime, but that's how operator discovers the dependency

# ❌ REJECTED: Structural violation
- name: "UP"
  delta: "north"  # Type error: delta must be list[int]
```

**Philosophy**: The framework is a **tool**, not a **teacher**. Operators discover best practices through experimentation and failure.

---

### 6. Why Custom Actions Can Manipulate Any Bar?

**Rationale**: Extensibility for future bar additions

**Example**: Operator adds `mental_clarity` meter in bars.yaml

**Custom action can immediately use it**:
```yaml
- name: "FOCUS"
  type: "passive"
  costs: {energy: 0.01}
  effects: {mental_clarity: 0.05}  # New meter, no code changes needed
```

---

## Success Criteria

- [ ] All substrates implement `get_default_actions()`
- [ ] `ActionSpaceBuilder` composes substrate + custom actions
- [ ] `VectorizedHamletEnv` uses `ComposedActionSpace`
- [ ] Custom actions can manipulate bars (costs/effects)
- [ ] Custom actions can teleport (non-standard movement)
- [ ] All curriculum levels have `custom_actions.yaml` (even if empty)
- [ ] REST action works (negative costs restore meters)
- [ ] Grid2D + custom → 8 actions (6 substrate + 2 custom)
- [ ] Grid3D + custom → 10 actions (8 substrate + 2 custom)
- [ ] Aspatial + custom → 4 actions (2 substrate + 2 custom)
- [ ] All tests pass (unit + integration)

---

## Estimated Effort

| Phase | Description | Hours |
|-------|-------------|-------|
| Phase 1 | Substrate default actions | 4-6 |
| Phase 2 | ActionSpaceBuilder | 4-5 |
| Phase 3 | Environment integration | 6-8 |
| Phase 4 | Example configs | 2-3 |
| Phase 5 | Testing | 4-6 |
| **Total** | | **20-28 hours** |

**Comparison**:
- Original TASK-002B estimate: 13-20 hours (incompatible with substrate abstraction)
- Revised estimate: 20-28 hours (+30-40% realistic adjustment)

---

## Risks

| Risk | Level | Mitigation |
|------|-------|------------|
| Action ID conflicts | 🟡 Low | Builder assigns IDs deterministically |
| Custom action dispatch bugs | 🟡 Low | Comprehensive integration tests |
| Substrate changes break custom configs | 🟡 Low | Validation at load time |
| Performance degradation | 🟢 Very Low | Action cost tensor pre-built (from addendum) |

---

## Future Extensions (Post-TASK-002B)

### Affordance Actions (TASK-003)
```yaml
# affordances.yaml
- name: "Bed"
  effects: [...]
  provides_action:
    name: "REST"
    type: "passive"
    costs: {energy: -0.002}
```

### Action Modifiers (TASK-004)
```yaml
# Custom action with conditions
- name: "SPRINT"
  type: "movement"
  delta: [0, -2]  # Move 2 cells
  costs: {energy: 0.02, satiation: 0.01}
  requires: {fitness: 0.7}  # Only available if fit
```

### Multi-Target Actions (TASK-005)
```yaml
# Social action targeting other agents
- name: "GIFT_MONEY"
  type: "social"
  target: "nearest_agent"
  costs: {money: 5.0}
  effects_on_target: {social: 0.1}
```

---

## Related Tasks

- **TASK-002A**: Configurable Spatial Substrates (✅ COMPLETED - provides substrate actions)
- **TASK-003**: Multi-Tick Affordances (affordance actions deferred to this)
- **TASK-004**: Cross-Config Validation (validates meter references in custom actions)
- **TASK-001**: No-Defaults Principle (ensures substrate defaults are explicit)

---

## Appendix: Example Composed Action Spaces

### Grid2D + REST/MEDITATE

```
Action Space (8 total):
├─ [0] UP (substrate, movement)
├─ [1] DOWN (substrate, movement)
├─ [2] LEFT (substrate, movement)
├─ [3] RIGHT (substrate, movement)
├─ [4] INTERACT (substrate, interaction)
├─ [5] WAIT (substrate, passive)
├─ [6] REST (custom, passive)
└─ [7] MEDITATE (custom, passive)
```

### Grid3D + REST/MEDITATE

```
Action Space (10 total):
├─ [0] UP (substrate, movement)
├─ [1] DOWN (substrate, movement)
├─ [2] LEFT (substrate, movement)
├─ [3] RIGHT (substrate, movement)
├─ [4] UP_Z (substrate, movement)
├─ [5] DOWN_Z (substrate, movement)
├─ [6] INTERACT (substrate, interaction)
├─ [7] WAIT (substrate, passive)
├─ [8] REST (custom, passive)
└─ [9] MEDITATE (custom, passive)
```

### Aspatial + REST/MEDITATE

```
Action Space (4 total):
├─ [0] INTERACT (substrate, interaction)
├─ [1] WAIT (substrate, passive)
├─ [2] REST (custom, passive)
└─ [3] MEDITATE (custom, passive)
```

Note: Aspatial has no movement actions (position_dim=0).
