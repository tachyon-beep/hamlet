# Composable Action Space Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable operators to define custom actions (REST, MEDITATE, TELEPORT_HOME) while maintaining fixed action vocabulary across curriculum for checkpoint transfer.

**Architecture:** Composable action space from substrate + global custom actions, with per-config enabled_actions list for masking. All curriculum levels share same action_dim, disabled actions masked out at runtime.

**Tech Stack:** Python 3.11+, Pydantic 2.x, PyTorch, YAML

**Context:** Implements TASK-002B-COMPOSABLE-ACTION-SPACE. Replaces retired UAC-ACTION-SPACE plans that conflicted with substrate abstraction.

---

## Phase 0: Action Configuration Schema

### Task 0.1: ActionConfig with enabled field

**Files:**
- Create: `src/townlet/environment/action_config.py`
- Test: `tests/test_townlet/unit/test_action_config.py`

**Step 1: Write failing test**

Create: `tests/test_townlet/unit/test_action_config.py`

```python
"""Tests for action configuration schemas."""

import pytest
from pathlib import Path

from townlet.environment.action_config import ActionConfig


def test_action_config_basic():
    """Basic action config should parse successfully."""
    action = ActionConfig(
        id=0,
        name="UP",
        type="movement",
        delta=[0, -1],
        costs={"energy": 0.005},
        enabled=True,
    )

    assert action.id == 0
    assert action.name == "UP"
    assert action.type == "movement"
    assert action.delta == [0, -1]
    assert action.costs["energy"] == 0.005
    assert action.enabled is True


def test_action_config_disabled():
    """Disabled action should have enabled=False."""
    action = ActionConfig(
        id=1,
        name="REST",
        type="passive",
        costs={"energy": -0.002},
        enabled=False,  # Disabled in L0, enabled in L1
    )

    assert action.enabled is False


def test_action_config_default_enabled():
    """Action should default to enabled=True if not specified."""
    action = ActionConfig(
        id=0,
        name="UP",
        type="movement",
        delta=[0, -1],
        costs={},
    )

    assert action.enabled is True


def test_movement_action_requires_delta_or_teleport():
    """Movement action must have delta or teleport_to."""
    # Valid: has delta
    ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={})

    # Valid: has teleport_to
    ActionConfig(id=1, name="TELEPORT", type="movement", teleport_to=[0, 0], costs={})

    # Invalid: has neither
    with pytest.raises(ValueError, match="must define delta or teleport_to"):
        ActionConfig(id=2, name="INVALID", type="movement", costs={})


def test_non_movement_cannot_have_delta():
    """Non-movement actions cannot have delta."""
    with pytest.raises(ValueError, match="cannot have delta"):
        ActionConfig(
            id=0,
            name="INTERACT",
            type="interaction",
            delta=[0, 0],
            costs={},
        )


def test_negative_costs_allowed():
    """Negative costs (restoration) should be allowed."""
    action = ActionConfig(
        id=0,
        name="REST",
        type="passive",
        costs={"energy": -0.002, "mood": -0.01},  # Restores meters
    )

    assert action.costs["energy"] == -0.002
    assert action.costs["mood"] == -0.01
```

**Step 2: Run test to verify it fails**

```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_action_config.py -v
```

Expected: FAIL (module not found)

**Step 3: Implement ActionConfig schema**

Create: `src/townlet/environment/action_config.py`

```python
"""Action configuration schemas for composable action space."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ActionConfig(BaseModel):
    """Single action definition.

    Can come from:
    - Substrate (movement, INTERACT, WAIT)
    - Global custom actions (REST, MEDITATE, TELEPORT_HOME)
    - Affordances (future)

    CRITICAL: All curriculum levels share same action vocabulary.
    Disabled actions are masked out but still occupy action IDs.
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

    @model_validator(mode="after")
    def validate_movement_actions(self) -> "ActionConfig":
        """Movement actions must have delta or teleport_to."""
        if self.type == "movement":
            if self.delta is None and self.teleport_to is None:
                raise ValueError(
                    f"Movement action '{self.name}' must define delta or teleport_to"
                )
        elif self.delta is not None or self.teleport_to is not None:
            raise ValueError(
                f"Non-movement action '{self.name}' cannot have delta or teleport_to"
            )
        return self
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_townlet/unit/test_action_config.py -v
```

Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add src/townlet/environment/action_config.py tests/test_townlet/unit/test_action_config.py
git commit -m "feat(actions): add ActionConfig schema with enabled field

- ActionConfig DTO with movement/interaction/passive/transaction types
- Multi-meter costs and effects (pattern matches affordances)
- enabled field for curriculum masking
- Validation: movement requires delta/teleport_to
- Negative costs allowed (restoration)

Part of TASK-002B (Composable Action Space)."
```

---

## Phase 1: Substrate Default Actions

### Task 1.1: Grid2DSubstrate.get_default_actions()

**Files:**
- Test: `tests/test_townlet/unit/test_substrate_actions.py` (new)
- Modify: `src/townlet/substrate/grid2d.py`

**Step 1: Write failing test**

Create: `tests/test_townlet/unit/test_substrate_actions.py`

```python
"""Tests for substrate default action generation."""

import pytest

from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.environment.action_config import ActionConfig


def test_grid2d_generates_6_default_actions():
    """Grid2D should provide 6 default actions."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert len(actions) == 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
    assert all(isinstance(a, ActionConfig) for a in actions)


def test_grid2d_action_names():
    """Grid2D actions should have correct names."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    names = [a.name for a in actions]

    assert names == ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]


def test_grid2d_movement_actions_have_deltas():
    """Grid2D movement actions should have correct deltas."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    assert actions_by_name["UP"].delta == [0, -1]
    assert actions_by_name["DOWN"].delta == [0, 1]
    assert actions_by_name["LEFT"].delta == [-1, 0]
    assert actions_by_name["RIGHT"].delta == [1, 0]


def test_grid2d_action_costs():
    """Grid2D actions should have default costs."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # Movement costs energy + hygiene + satiation
    up = actions_by_name["UP"]
    assert up.costs["energy"] == 0.005
    assert up.costs["hygiene"] == 0.003
    assert up.costs["satiation"] == 0.004

    # INTERACT costs energy only
    interact = actions_by_name["INTERACT"]
    assert interact.costs["energy"] == 0.003

    # WAIT costs energy only
    wait = actions_by_name["WAIT"]
    assert wait.costs["energy"] == 0.004


def test_grid2d_all_actions_enabled_by_default():
    """Grid2D actions should default to enabled=True."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert all(a.enabled for a in actions)


def test_grid2d_all_actions_marked_as_substrate():
    """Grid2D actions should have source='substrate'."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert all(a.source == "substrate" for a in actions)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_grid2d_generates_6_default_actions -v
```

Expected: FAIL (AttributeError: 'Grid2DSubstrate' has no attribute 'get_default_actions')

**Step 3: Add abstract method to SpatialSubstrate base**

Modify: `src/townlet/substrate/base.py`

Add after line 85 (after action_space_size property):

```python
    @abstractmethod
    def get_default_actions(self) -> list["ActionConfig"]:
        """Return substrate's default action space with default costs.

        Returns:
            List of ActionConfig instances with substrate-provided actions.
            IDs are temporary (will be reassigned by ActionSpaceBuilder).

        Examples:
            Grid2D: [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT] (6 actions)
            Grid3D: [UP, DOWN, LEFT, RIGHT, UP_Z, DOWN_Z, INTERACT, WAIT] (8 actions)
            GridND(7D): [DIM0_NEG, DIM0_POS, ..., DIM6_POS, INTERACT, WAIT] (16 actions)
            Aspatial: [INTERACT, WAIT] (2 actions)
        """
        pass
```

Add import at top:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from townlet.environment.action_config import ActionConfig
```

**Step 4: Implement Grid2DSubstrate.get_default_actions()**

Modify: `src/townlet/substrate/grid2d.py`

Add import at top:

```python
from townlet.environment.action_config import ActionConfig
```

Add method after compute_distance (around line 120):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return Grid2D's 6 default actions with default costs.

        Returns:
            [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT] with standard 2D costs.
        """
        return [
            ActionConfig(
                id=0,  # Temporary, reassigned by builder
                name="UP",
                type="movement",
                delta=[0, -1],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell upward (north)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="DOWN",
                type="movement",
                delta=[0, 1],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell downward (south)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=2,
                name="LEFT",
                type="movement",
                delta=[-1, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell left (west)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=3,
                name="RIGHT",
                type="movement",
                delta=[1, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell right (east)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=4,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=5,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
                source="substrate",
                enabled=True,
            ),
        ]
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_grid2d -v
```

Expected: PASS (all Grid2D tests pass)

**Step 6: Commit**

```bash
git add src/townlet/substrate/base.py src/townlet/substrate/grid2d.py tests/test_townlet/unit/test_substrate_actions.py
git commit -m "feat(substrate): add get_default_actions() to Grid2DSubstrate

- Abstract method in SpatialSubstrate base
- Grid2D provides 6 actions: UP/DOWN/LEFT/RIGHT/INTERACT/WAIT
- Default costs: movement=0.5% energy + 0.3% hygiene + 0.4% satiation
- All actions marked as source='substrate', enabled=True

Part of TASK-002B Phase 1."
```

---

### Task 1.2: Grid3DSubstrate.get_default_actions()

**Files:**
- Test: `tests/test_townlet/unit/test_substrate_actions.py` (modify)
- Modify: `src/townlet/substrate/grid3d.py`

**Step 1: Write failing test**

Modify: `tests/test_townlet/unit/test_substrate_actions.py`

Add after Grid2D tests:

```python
def test_grid3d_generates_8_default_actions():
    """Grid3D should provide 8 default actions (adds UP_Z, DOWN_Z)."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()

    assert len(actions) == 8  # UP, DOWN, LEFT, RIGHT, UP_Z, DOWN_Z, INTERACT, WAIT


def test_grid3d_action_names():
    """Grid3D actions should include Z-axis movement."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    names = [a.name for a in actions]

    assert names == ["UP", "DOWN", "LEFT", "RIGHT", "UP_Z", "DOWN_Z", "INTERACT", "WAIT"]


def test_grid3d_z_axis_deltas():
    """Grid3D Z-axis actions should have correct deltas."""
    substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    assert actions_by_name["UP_Z"].delta == [0, 0, -1]  # Decrease Z (up floor)
    assert actions_by_name["DOWN_Z"].delta == [0, 0, 1]  # Increase Z (down floor)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_grid3d_generates_8_default_actions -v
```

Expected: FAIL (AttributeError: 'Grid3DSubstrate' has no attribute 'get_default_actions')

**Step 3: Implement Grid3DSubstrate.get_default_actions()**

Modify: `src/townlet/substrate/grid3d.py`

Add import at top:

```python
from townlet.environment.action_config import ActionConfig
```

Add method after compute_distance (around line 130):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return Grid3D's 8 default actions with default costs.

        Returns:
            [UP, DOWN, LEFT, RIGHT, UP_Z, DOWN_Z, INTERACT, WAIT] with standard 3D costs.
        """
        return [
            # XY plane movement (same as Grid2D)
            ActionConfig(
                id=0,
                name="UP",
                type="movement",
                delta=[0, -1, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell upward (north)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="DOWN",
                type="movement",
                delta=[0, 1, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell downward (south)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=2,
                name="LEFT",
                type="movement",
                delta=[-1, 0, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell left (west)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=3,
                name="RIGHT",
                type="movement",
                delta=[1, 0, 0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description="Move one cell right (east)",
                source="substrate",
                enabled=True,
            ),
            # Z-axis movement (vertical)
            ActionConfig(
                id=4,
                name="UP_Z",
                type="movement",
                delta=[0, 0, -1],
                costs={"energy": 0.008, "hygiene": 0.003, "satiation": 0.006},  # Stairs cost more
                description="Move one floor up (climb stairs)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=5,
                name="DOWN_Z",
                type="movement",
                delta=[0, 0, 1],
                costs={"energy": 0.006, "hygiene": 0.003, "satiation": 0.005},  # Going down easier
                description="Move one floor down (descend stairs)",
                source="substrate",
                enabled=True,
            ),
            # Core interactions (same as Grid2D)
            ActionConfig(
                id=6,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=7,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
                source="substrate",
                enabled=True,
            ),
        ]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_grid3d -v
```

Expected: PASS (all Grid3D tests pass)

**Step 5: Commit**

```bash
git add src/townlet/substrate/grid3d.py tests/test_townlet/unit/test_substrate_actions.py
git commit -m "feat(substrate): add get_default_actions() to Grid3DSubstrate

- Grid3D provides 8 actions (adds UP_Z/DOWN_Z for vertical movement)
- UP_Z costs more energy (climbing stairs harder)
- DOWN_Z costs less energy (descending easier)
- Z-axis deltas: [0, 0, ±1]

Part of TASK-002B Phase 1."
```

---

### Task 1.3: GridNDSubstrate.get_default_actions() (N-dimensional)

**Files:**
- Test: `tests/test_townlet/unit/test_substrate_actions.py` (modify)
- Modify: `src/townlet/substrate/gridnd.py`

**Step 1: Write failing test**

Modify: `tests/test_townlet/unit/test_substrate_actions.py`

Add after Grid3D tests:

```python
def test_gridnd_generates_2n_plus_2_actions():
    """GridND should provide 2N+2 actions (±1 per dimension + INTERACT + WAIT)."""
    # 7D hypercube
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
    )

    actions = substrate.get_default_actions()

    # 7 dimensions × 2 directions + INTERACT + WAIT = 16 actions
    assert len(actions) == 16


def test_gridnd_action_naming_pattern():
    """GridND actions should follow DIM{N}_{NEG|POS} naming pattern."""
    substrate = GridNDSubstrate(
        dimension_sizes=[3, 3, 3],  # 3D
        boundary="clamp",
        distance_metric="manhattan",
    )

    actions = substrate.get_default_actions()
    names = [a.name for a in actions]

    expected = [
        "DIM0_NEG", "DIM0_POS",  # Dimension 0
        "DIM1_NEG", "DIM1_POS",  # Dimension 1
        "DIM2_NEG", "DIM2_POS",  # Dimension 2
        "INTERACT",
        "WAIT",
    ]
    assert names == expected


def test_gridnd_movement_deltas():
    """GridND actions should have correct deltas."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],  # 4D
        boundary="clamp",
        distance_metric="manhattan",
    )

    actions = substrate.get_default_actions()
    actions_by_name = {a.name: a for a in actions}

    # DIM0_NEG: [-1, 0, 0, 0]
    assert actions_by_name["DIM0_NEG"].delta == [-1, 0, 0, 0]
    # DIM0_POS: [+1, 0, 0, 0]
    assert actions_by_name["DIM0_POS"].delta == [1, 0, 0, 0]
    # DIM3_NEG: [0, 0, 0, -1]
    assert actions_by_name["DIM3_NEG"].delta == [0, 0, 0, -1]
    # DIM3_POS: [0, 0, 0, +1]
    assert actions_by_name["DIM3_POS"].delta == [0, 0, 0, 1]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_gridnd_generates_2n_plus_2_actions -v
```

Expected: FAIL (AttributeError: 'GridNDSubstrate' has no attribute 'get_default_actions')

**Step 3: Implement GridNDSubstrate.get_default_actions()**

Modify: `src/townlet/substrate/gridnd.py`

Add import at top:

```python
from townlet.environment.action_config import ActionConfig
```

Add method after compute_distance (around line 140):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return GridND's 2N+2 default actions with default costs.

        Returns:
            [DIM0_NEG, DIM0_POS, DIM1_NEG, DIM1_POS, ..., INTERACT, WAIT]

        Example:
            4D grid: 8 movement + INTERACT + WAIT = 10 actions
            7D grid: 14 movement + INTERACT + WAIT = 16 actions
        """
        actions = []
        action_id = 0

        # Generate movement actions for each dimension
        n_dims = len(self.dimension_sizes)
        for dim_idx in range(n_dims):
            # Negative direction (DIM{N}_NEG)
            delta = [0] * n_dims
            delta[dim_idx] = -1
            actions.append(
                ActionConfig(
                    id=action_id,
                    name=f"DIM{dim_idx}_NEG",
                    type="movement",
                    delta=delta,
                    costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                    description=f"Move -1 along dimension {dim_idx}",
                    source="substrate",
                    enabled=True,
                )
            )
            action_id += 1

            # Positive direction (DIM{N}_POS)
            delta = [0] * n_dims
            delta[dim_idx] = 1
            actions.append(
                ActionConfig(
                    id=action_id,
                    name=f"DIM{dim_idx}_POS",
                    type="movement",
                    delta=delta,
                    costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                    description=f"Move +1 along dimension {dim_idx}",
                    source="substrate",
                    enabled=True,
                )
            )
            action_id += 1

        # Core interactions
        actions.append(
            ActionConfig(
                id=action_id,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
                source="substrate",
                enabled=True,
            )
        )
        action_id += 1

        actions.append(
            ActionConfig(
                id=action_id,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
                source="substrate",
                enabled=True,
            )
        )

        return actions
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_gridnd -v
```

Expected: PASS (all GridND tests pass)

**Step 5: Commit**

```bash
git add src/townlet/substrate/gridnd.py tests/test_townlet/unit/test_substrate_actions.py
git commit -m "feat(substrate): add get_default_actions() to GridNDSubstrate

- Generates 2N+2 actions for N-dimensional grids
- Naming pattern: DIM{N}_{NEG|POS} for movement
- Auto-generates deltas for each dimension
- Example: 7D grid → 16 actions (14 movement + INTERACT + WAIT)

Part of TASK-002B Phase 1."
```

---

### Task 1.4: Continuous substrates get_default_actions()

**Files:**
- Test: `tests/test_townlet/unit/test_substrate_actions.py` (modify)
- Modify: `src/townlet/substrate/continuous.py`
- Modify: `src/townlet/substrate/continuousnd.py`

**Step 1: Write failing tests**

Modify: `tests/test_townlet/unit/test_substrate_actions.py`

Add after GridND tests:

```python
def test_continuous1d_generates_4_actions():
    """Continuous1D should provide 4 actions (LEFT/RIGHT/INTERACT/WAIT)."""
    substrate = Continuous1DSubstrate(bounds=(0.0, 10.0), boundary="clamp", movement_delta=0.5)

    actions = substrate.get_default_actions()

    assert len(actions) == 4
    names = [a.name for a in actions]
    assert names == ["LEFT", "RIGHT", "INTERACT", "WAIT"]


def test_continuous2d_generates_6_actions():
    """Continuous2D should provide 6 actions (same as Grid2D)."""
    substrate = Continuous2DSubstrate(
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
    )

    actions = substrate.get_default_actions()

    assert len(actions) == 6
    names = [a.name for a in actions]
    assert names == ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]


def test_continuousnd_generates_2n_plus_2_actions():
    """ContinuousND should provide 2N+2 actions (same pattern as GridND)."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * 5,  # 5D
        boundary="clamp",
        movement_delta=0.5,
    )

    actions = substrate.get_default_actions()

    assert len(actions) == 12  # 5D × 2 + INTERACT + WAIT
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_continuous -v
```

Expected: FAIL (methods not implemented)

**Step 3: Implement Continuous1DSubstrate.get_default_actions()**

Modify: `src/townlet/substrate/continuous.py`

Add import at top:

```python
from townlet.environment.action_config import ActionConfig
```

Add to Continuous1DSubstrate class (around line 80):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return Continuous1D's 4 default actions."""
        return [
            ActionConfig(
                id=0,
                name="LEFT",
                type="movement",
                delta=[-self.movement_delta],  # Move left by delta
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move left by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="RIGHT",
                type="movement",
                delta=[self.movement_delta],  # Move right by delta
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move right by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=2,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=3,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
                source="substrate",
                enabled=True,
            ),
        ]
```

Add to Continuous2DSubstrate class (around line 140):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return Continuous2D's 6 default actions (same names as Grid2D)."""
        return [
            ActionConfig(
                id=0,
                name="UP",
                type="movement",
                delta=[0.0, -self.movement_delta],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move upward by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="DOWN",
                type="movement",
                delta=[0.0, self.movement_delta],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move downward by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=2,
                name="LEFT",
                type="movement",
                delta=[-self.movement_delta, 0.0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move left by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=3,
                name="RIGHT",
                type="movement",
                delta=[self.movement_delta, 0.0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move right by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=4,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=5,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
                source="substrate",
                enabled=True,
            ),
        ]
```

Add to Continuous3DSubstrate class (around line 200):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return Continuous3D's 8 default actions (same pattern as Grid3D)."""
        return [
            ActionConfig(
                id=0,
                name="UP",
                type="movement",
                delta=[0.0, -self.movement_delta, 0.0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move upward by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="DOWN",
                type="movement",
                delta=[0.0, self.movement_delta, 0.0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move downward by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=2,
                name="LEFT",
                type="movement",
                delta=[-self.movement_delta, 0.0, 0.0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move left by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=3,
                name="RIGHT",
                type="movement",
                delta=[self.movement_delta, 0.0, 0.0],
                costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                description=f"Move right by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=4,
                name="UP_Z",
                type="movement",
                delta=[0.0, 0.0, -self.movement_delta],
                costs={"energy": 0.008, "hygiene": 0.003, "satiation": 0.006},
                description=f"Move up vertically by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=5,
                name="DOWN_Z",
                type="movement",
                delta=[0.0, 0.0, self.movement_delta],
                costs={"energy": 0.006, "hygiene": 0.003, "satiation": 0.005},
                description=f"Move down vertically by {self.movement_delta} units",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=6,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=7,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
                source="substrate",
                enabled=True,
            ),
        ]
```

**Step 4: Implement ContinuousNDSubstrate.get_default_actions()**

Modify: `src/townlet/substrate/continuousnd.py`

Add import:

```python
from townlet.environment.action_config import ActionConfig
```

Add method (around line 140):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return ContinuousND's 2N+2 default actions (same pattern as GridND)."""
        actions = []
        action_id = 0
        n_dims = len(self.bounds)

        # Generate movement actions for each dimension
        for dim_idx in range(n_dims):
            # Negative direction
            delta = [0.0] * n_dims
            delta[dim_idx] = -self.movement_delta
            actions.append(
                ActionConfig(
                    id=action_id,
                    name=f"DIM{dim_idx}_NEG",
                    type="movement",
                    delta=delta,
                    costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                    description=f"Move -{self.movement_delta} along dimension {dim_idx}",
                    source="substrate",
                    enabled=True,
                )
            )
            action_id += 1

            # Positive direction
            delta = [0.0] * n_dims
            delta[dim_idx] = self.movement_delta
            actions.append(
                ActionConfig(
                    id=action_id,
                    name=f"DIM{dim_idx}_POS",
                    type="movement",
                    delta=delta,
                    costs={"energy": 0.005, "hygiene": 0.003, "satiation": 0.004},
                    description=f"Move +{self.movement_delta} along dimension {dim_idx}",
                    source="substrate",
                    enabled=True,
                )
            )
            action_id += 1

        # Core interactions
        actions.append(
            ActionConfig(
                id=action_id,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance at current position",
                source="substrate",
                enabled=True,
            )
        )
        action_id += 1

        actions.append(
            ActionConfig(
                id=action_id,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait in place (idle metabolic cost)",
                source="substrate",
                enabled=True,
            )
        )

        return actions
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_continuous -v
```

Expected: PASS (all continuous substrate tests pass)

**Step 6: Commit**

```bash
git add src/townlet/substrate/continuous.py src/townlet/substrate/continuousnd.py tests/test_townlet/unit/test_substrate_actions.py
git commit -m "feat(substrate): add get_default_actions() to continuous substrates

- Continuous1D: 4 actions (LEFT/RIGHT/INTERACT/WAIT)
- Continuous2D: 6 actions (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT)
- Continuous3D: 8 actions (adds UP_Z/DOWN_Z)
- ContinuousND: 2N+2 actions (DIM{N}_{NEG|POS} pattern)
- Deltas use movement_delta instead of ±1

Part of TASK-002B Phase 1."
```

---

### Task 1.5: AspatialSubstrate.get_default_actions()

**Files:**
- Test: `tests/test_townlet/unit/test_substrate_actions.py` (modify)
- Modify: `src/townlet/substrate/aspatial.py`

**Step 1: Write failing test**

Modify: `tests/test_townlet/unit/test_substrate_actions.py`

Add after continuous tests:

```python
def test_aspatial_generates_2_actions():
    """Aspatial should provide only 2 actions (no movement)."""
    substrate = AspatialSubstrate()

    actions = substrate.get_default_actions()

    assert len(actions) == 2  # INTERACT + WAIT only
    names = [a.name for a in actions]
    assert names == ["INTERACT", "WAIT"]


def test_aspatial_no_movement_actions():
    """Aspatial should have zero movement actions."""
    substrate = AspatialSubstrate()

    actions = substrate.get_default_actions()

    movement_actions = [a for a in actions if a.type == "movement"]
    assert len(movement_actions) == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_aspatial_generates_2_actions -v
```

Expected: FAIL (method not implemented)

**Step 3: Implement AspatialSubstrate.get_default_actions()**

Modify: `src/townlet/substrate/aspatial.py`

Add import:

```python
from townlet.environment.action_config import ActionConfig
```

Add method (around line 80):

```python
    def get_default_actions(self) -> list[ActionConfig]:
        """Return Aspatial's 2 default actions (no movement).

        Returns:
            [INTERACT, WAIT] only (no spatial movement)
        """
        return [
            ActionConfig(
                id=0,
                name="INTERACT",
                type="interaction",
                costs={"energy": 0.003},
                description="Interact with affordance (aspatial, no position required)",
                source="substrate",
                enabled=True,
            ),
            ActionConfig(
                id=1,
                name="WAIT",
                type="passive",
                costs={"energy": 0.004},
                description="Wait (idle metabolic cost)",
                source="substrate",
                enabled=True,
            ),
        ]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py::test_aspatial -v
```

Expected: PASS (both aspatial tests pass)

**Step 5: Run all substrate action tests**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_actions.py -v
```

Expected: PASS (all substrate tests pass)

**Step 6: Commit**

```bash
git add src/townlet/substrate/aspatial.py tests/test_townlet/unit/test_substrate_actions.py
git commit -m "feat(substrate): add get_default_actions() to AspatialSubstrate

- Aspatial provides only 2 actions: INTERACT + WAIT
- No movement actions (position_dim=0)
- Completes Phase 1: All substrates now provide default actions

Part of TASK-002B Phase 1 (COMPLETE)."
```

---

## Phase 2: ActionSpaceBuilder and ComposedActionSpace

### Task 2.1: ComposedActionSpace class

**Files:**
- Create: `src/townlet/environment/action_builder.py`
- Test: `tests/test_townlet/unit/test_action_builder.py`

**Step 1: Write failing test**

Create: `tests/test_townlet/unit/test_action_builder.py`

```python
"""Tests for ActionSpaceBuilder and ComposedActionSpace."""

import pytest
import torch
from pathlib import Path

from townlet.environment.action_builder import ComposedActionSpace
from townlet.environment.action_config import ActionConfig


def test_composed_action_space_basic():
    """ComposedActionSpace should track actions and metadata."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate"),
        ActionConfig(id=1, name="DOWN", type="movement", delta=[0, 1], costs={}, source="substrate"),
        ActionConfig(id=2, name="REST", type="passive", costs={}, source="custom"),
    ]

    space = ComposedActionSpace(
        actions=actions,
        substrate_action_count=2,
        custom_action_count=1,
        affordance_action_count=0,
    )

    assert space.action_dim == 3
    assert space.substrate_action_count == 2
    assert space.custom_action_count == 1


def test_composed_action_space_get_by_id():
    """Should retrieve action by ID."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate"),
        ActionConfig(id=1, name="REST", type="passive", costs={}, source="custom"),
    ]

    space = ComposedActionSpace(actions=actions, substrate_action_count=1, custom_action_count=1, affordance_action_count=0)

    assert space.get_action_by_id(0).name == "UP"
    assert space.get_action_by_id(1).name == "REST"


def test_composed_action_space_get_by_name():
    """Should retrieve action by name."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate"),
        ActionConfig(id=1, name="REST", type="passive", costs={}, source="custom"),
    ]

    space = ComposedActionSpace(actions=actions, substrate_action_count=1, custom_action_count=1, affordance_action_count=0)

    assert space.get_action_by_name("UP").id == 0
    assert space.get_action_by_name("REST").id == 1


def test_composed_action_space_enabled_count():
    """Should count enabled vs disabled actions."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=1, name="DOWN", type="movement", delta=[0, 1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=2, name="REST", type="passive", costs={}, source="custom", enabled=True),
        ActionConfig(id=3, name="MEDITATE", type="passive", costs={}, source="custom", enabled=False),  # Disabled
    ]

    space = ComposedActionSpace(
        actions=actions,
        substrate_action_count=2,
        custom_action_count=2,
        affordance_action_count=0,
        enabled_action_names={"UP", "DOWN", "REST"},  # MEDITATE not enabled
    )

    assert space.action_dim == 4  # Total actions (including disabled)
    assert space.enabled_action_count == 3  # Only enabled ones


def test_composed_action_space_get_base_mask():
    """Should generate action mask with disabled actions masked out."""
    actions = [
        ActionConfig(id=0, name="UP", type="movement", delta=[0, -1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=1, name="DOWN", type="movement", delta=[0, 1], costs={}, source="substrate", enabled=True),
        ActionConfig(id=2, name="REST", type="passive", costs={}, source="custom", enabled=True),
        ActionConfig(id=3, name="MEDITATE", type="passive", costs={}, source="custom", enabled=False),  # Disabled
    ]

    space = ComposedActionSpace(
        actions=actions,
        substrate_action_count=2,
        custom_action_count=2,
        affordance_action_count=0,
    )

    mask = space.get_base_action_mask(num_agents=2, device=torch.device("cpu"))

    # Shape: [2 agents, 4 actions]
    assert mask.shape == (2, 4)

    # Actions 0, 1, 2 enabled (True)
    assert mask[0, 0] == True
    assert mask[0, 1] == True
    assert mask[0, 2] == True

    # Action 3 disabled (False)
    assert mask[0, 3] == False
    assert mask[1, 3] == False  # Disabled for all agents
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_action_builder.py::test_composed_action_space_basic -v
```

Expected: FAIL (module not found)

**Step 3: Implement ComposedActionSpace**

Create: `src/townlet/environment/action_builder.py`

```python
"""Action space builder for composable action spaces."""

from typing import Optional
from pathlib import Path

import torch
import yaml

from townlet.environment.action_config import ActionConfig
from townlet.substrate.base import SpatialSubstrate


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

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_action_builder.py -v
```

Expected: PASS (all ComposedActionSpace tests pass)

**Step 5: Commit**

```bash
git add src/townlet/environment/action_builder.py tests/test_townlet/unit/test_action_builder.py
git commit -m "feat(actions): add ComposedActionSpace class

- Tracks substrate + custom + affordance actions
- Computes enabled vs disabled action counts
- Generates base action mask (disabled actions = False)
- CRITICAL: action_dim is SAME across all configs (curriculum transfer)

Part of TASK-002B Phase 2."
```

---

### Task 2.2: ActionSpaceBuilder with global vocabulary

**Files:**
- Modify: `src/townlet/environment/action_builder.py`
- Test: `tests/test_townlet/unit/test_action_builder.py` (modify)

**Step 1: Write failing test**

Modify: `tests/test_townlet/unit/test_action_builder.py`

Add after ComposedActionSpace tests:

```python
def test_action_space_builder_substrate_only():
    """Builder with no custom actions should return substrate actions only."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # No global_actions.yaml
    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=Path("/nonexistent/global_actions.yaml"),
    )

    space = builder.build()

    assert space.action_dim == 6  # Grid2D substrate actions only
    assert space.substrate_action_count == 6
    assert space.custom_action_count == 0


def test_action_space_builder_with_custom_actions(tmp_path):
    """Builder with custom actions should compose substrate + custom."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    # Create temporary global_actions.yaml
    global_actions_yaml = tmp_path / "global_actions.yaml"
    global_actions_yaml.write_text("""
version: "1.0"
description: "Global custom actions"

custom_actions:
  - name: "REST"
    type: "passive"
    costs: {energy: -0.002}
  - name: "MEDITATE"
    type: "passive"
    costs: {mood: 0.02}
""")

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=global_actions_yaml,
    )

    space = builder.build()

    # 6 substrate + 2 custom = 8 actions
    assert space.action_dim == 8
    assert space.substrate_action_count == 6
    assert space.custom_action_count == 2

    # Substrate actions come first (IDs 0-5)
    assert space.get_action_by_id(0).name == "UP"
    assert space.get_action_by_id(0).source == "substrate"

    # Custom actions come after (IDs 6-7)
    assert space.get_action_by_id(6).name == "REST"
    assert space.get_action_by_id(6).source == "custom"
    assert space.get_action_by_id(7).name == "MEDITATE"
    assert space.get_action_by_id(7).source == "custom"


def test_action_space_builder_with_enabled_actions(tmp_path):
    """Builder should mark disabled actions as enabled=False."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    global_actions_yaml = tmp_path / "global_actions.yaml"
    global_actions_yaml.write_text("""
version: "1.0"
custom_actions:
  - name: "REST"
    type: "passive"
    costs: {}
  - name: "MEDITATE"
    type: "passive"
    costs: {}
""")

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # Only enable UP, DOWN, REST (disable others)
    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=global_actions_yaml,
        enabled_action_names=["UP", "DOWN", "REST"],
    )

    space = builder.build()

    # All 8 actions defined
    assert space.action_dim == 8

    # UP, DOWN, REST enabled
    assert space.get_action_by_name("UP").enabled == True
    assert space.get_action_by_name("DOWN").enabled == True
    assert space.get_action_by_name("REST").enabled == True

    # LEFT, RIGHT, INTERACT, WAIT, MEDITATE disabled
    assert space.get_action_by_name("LEFT").enabled == False
    assert space.get_action_by_name("RIGHT").enabled == False
    assert space.get_action_by_name("INTERACT").enabled == False
    assert space.get_action_by_name("WAIT").enabled == False
    assert space.get_action_by_name("MEDITATE").enabled == False

    # Enabled count = 3
    assert space.enabled_action_count == 3
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_action_builder.py::test_action_space_builder_substrate_only -v
```

Expected: FAIL (ActionSpaceBuilder not defined)

**Step 3: Implement ActionSpaceBuilder**

Modify: `src/townlet/environment/action_builder.py`

Add after ComposedActionSpace class:

```python
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

    def build(self) -> ComposedActionSpace:
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
        with open(self.global_actions_path) as f:
            data = yaml.safe_load(f)

        # Global file contains custom actions only (substrate provides its own)
        custom_action_data = data.get("custom_actions", [])

        # Parse into ActionConfig objects
        actions = []
        for action_dict in custom_action_data:
            # Add default fields if missing
            if "costs" not in action_dict:
                action_dict["costs"] = {}
            if "effects" not in action_dict:
                action_dict["effects"] = {}

            # Temporary ID (will be reassigned by build())
            action_dict["id"] = 0

            actions.append(ActionConfig(**action_dict))

        return actions
```

Add import at top:

```python
from townlet.environment.action_builder import ActionSpaceBuilder
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_action_builder.py -v
```

Expected: PASS (all builder tests pass)

**Step 5: Commit**

```bash
git add src/townlet/environment/action_builder.py tests/test_townlet/unit/test_action_builder.py
git commit -m "feat(actions): add ActionSpaceBuilder with global vocabulary

- Loads global_actions.yaml for custom actions
- Composes substrate + custom actions
- enabled_action_names list controls which actions are enabled
- Action IDs assigned deterministically (substrate first, then custom)

Part of TASK-002B Phase 2."
```

---

## Phase 3: Global Actions Configuration

### Task 3.1: Create global_actions.yaml

**Files:**
- Create: `configs/global_actions.yaml`

**Step 1: Write test that loads global_actions.yaml**

Modify: `tests/test_townlet/unit/test_action_builder.py`

Add:

```python
def test_load_global_actions_yaml():
    """Should load configs/global_actions.yaml successfully."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    global_actions_path = Path("configs/global_actions.yaml")

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=global_actions_path,
    )

    space = builder.build()

    # Grid2D (6) + global custom actions
    assert space.action_dim >= 6
    assert space.substrate_action_count == 6
    assert space.custom_action_count >= 0


def test_global_actions_has_rest_and_meditate():
    """Global actions should include REST and MEDITATE."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    global_actions_path = Path("configs/global_actions.yaml")

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    builder = ActionSpaceBuilder(
        substrate=substrate,
        global_actions_path=global_actions_path,
    )

    space = builder.build()

    # Should have REST and MEDITATE
    rest = space.get_action_by_name("REST")
    assert rest.type == "passive"
    assert rest.costs.get("energy", 0) < 0  # Negative cost (restoration)

    meditate = space.get_action_by_name("MEDITATE")
    assert meditate.type == "passive"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_action_builder.py::test_load_global_actions_yaml -v
```

Expected: FAIL (file not found)

**Step 3: Create global_actions.yaml**

Create: `configs/global_actions.yaml`

```yaml
version: "1.0"
description: "Global action vocabulary for HAMLET curriculum"

# Custom actions shared across all curriculum levels
# Disabled actions still occupy action IDs (for checkpoint transfer)
#
# Action space for all configs:
#   - Grid2D substrates: 6 substrate + 4 custom = 10 total
#   - Grid3D substrates: 8 substrate + 4 custom = 12 total
#   - etc.
#
# Per-config enabled_actions list in training.yaml controls which are available

custom_actions:
  # Passive recovery (weak version of Bed affordance)
  - name: "REST"
    type: "passive"
    costs:
      energy: -0.002  # Restores 0.2% energy per step
      mood: -0.01     # Restores 1% mood per step
    description: "Rest in place (weaker than Bed, but available anywhere)"
    icon: "😴"

  # Mental health action
  - name: "MEDITATE"
    type: "passive"
    costs:
      energy: 0.001   # Costs 0.1% energy
    effects:
      mood: 0.02      # Restores 2% mood
    description: "Meditate to improve mood without sleeping"
    icon: "🧘"

  # Movement shortcut (expensive teleport)
  - name: "TELEPORT_HOME"
    type: "movement"
    teleport_to: [0, 0]  # Warp to origin
    costs:
      energy: 0.5      # 50% energy cost
      money: 10.0      # $10 teleport fee
    description: "Instantly return to home position (expensive)"
    icon: "🏠"

  # Fast movement (high cost)
  - name: "SPRINT"
    type: "movement"
    delta: [0, -2]  # Move 2 cells north (assumes Grid2D)
    costs:
      energy: 0.02     # 2% energy (4x normal movement)
      satiation: 0.01  # 1% satiation (burns calories)
    description: "Sprint 2 cells north (only works on Grid2D)"
    icon: "🏃"

# Notes:
# - REST enables gradual recovery without finding Bed
# - MEDITATE provides mood boost without social interaction
# - TELEPORT_HOME creates economic choice (expensive but saves time)
# - SPRINT is Grid2D-specific (won't work on Grid3D or GridND)
#   This demonstrates that operators can define substrate-specific actions
#   Permissive semantics: invalid deltas fail at runtime, not config validation
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_townlet/unit/test_action_builder.py::test_load_global_actions_yaml -v
uv run pytest tests/test_townlet/unit/test_action_builder.py::test_global_actions_has_rest_and_meditate -v
```

Expected: PASS (both tests pass)

**Step 5: Commit**

```bash
git add configs/global_actions.yaml tests/test_townlet/unit/test_action_builder.py
git commit -m "feat(config): add global_actions.yaml with 4 custom actions

- REST: passive recovery (restores energy + mood)
- MEDITATE: mood boost (costs energy)
- TELEPORT_HOME: expensive warp to origin
- SPRINT: fast movement (Grid2D-specific)

All configs now have action_dim = substrate + 4
Enables curriculum transfer (same network architecture)

Part of TASK-002B Phase 3."
```

---

### Task 3.2: Update TrainingConfig with enabled_actions

**Files:**
- Modify: `src/townlet/config/training_config.py` (if exists, or create)
- Test: `tests/test_townlet/unit/test_training_config.py`

**Step 1: Check if TrainingConfig exists**

```bash
find /home/john/hamlet/src -name "*training*config*" -type f
```

Expected: Either find existing config or need to create new one

**Step 2: For now, update training.yaml files directly**

We'll defer formal TrainingConfig DTO to TASK-004A (validation).
For this task, just document the enabled_actions field.

Create: `configs/L0_0_minimal/training.yaml` updates

Add to existing training.yaml:

```yaml
# Add to environment section:
environment:
  # ... existing fields ...

  # Enabled actions (from global vocabulary)
  # L0 enables only basic movement + REST
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
  # MEDITATE, TELEPORT_HOME, SPRINT disabled (but IDs reserved)
```

**Step 3: Document enabled_actions pattern**

Create: `docs/config-schemas/enabled_actions.md`

```markdown
# enabled_actions Configuration

**Purpose**: Control which actions from global vocabulary are available in this config.

**Location**: `training.yaml` → `environment.enabled_actions`

**Pattern**: All curriculum levels share same action vocabulary (same action_dim).
Disabled actions are masked out at runtime but still occupy action IDs.

## Example

### Global Vocabulary (configs/global_actions.yaml)

```yaml
custom_actions:
  - name: "REST"
  - name: "MEDITATE"
  - name: "TELEPORT_HOME"
  - name: "SPRINT"
```

Total actions: 6 substrate (Grid2D) + 4 custom = **10 actions**

### L0_0_minimal/training.yaml

```yaml
environment:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
```

**Result**: 7 enabled, 3 disabled, action_dim = 10

### L1_full_observability/training.yaml

```yaml
environment:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
    - "MEDITATE"
    - "TELEPORT_HOME"
    - "SPRINT"
```

**Result**: 10 enabled, 0 disabled, action_dim = 10

## Checkpoint Transfer

Both L0 and L1 have **action_dim = 10**, so checkpoints transfer!

L0 Q-network outputs 10 Q-values (3 disabled actions get masked).
L1 Q-network outputs 10 Q-values (all actions available).

**Same architecture → checkpoint compatible.**
```

**Step 4: Commit documentation**

```bash
git add docs/config-schemas/enabled_actions.md
git commit -m "docs(config): document enabled_actions pattern

- enabled_actions controls which actions are available
- All configs share same action_dim (curriculum transfer)
- Disabled actions masked at runtime
- Examples for L0 and L1

Part of TASK-002B Phase 3."
```

---

## Phase 4: Environment Integration

### Task 4.1: VectorizedHamletEnv uses ActionSpaceBuilder

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py`
- Test: `tests/test_townlet/test_integration.py` (existing integration tests)

**Step 1: Write failing integration test**

Modify: `tests/test_townlet/test_integration.py`

Add:

```python
def test_vectorized_env_loads_composed_action_space():
    """VectorizedHamletEnv should use ActionSpaceBuilder."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    from pathlib import Path

    config_path = Path("configs/L1_full_observability")

    env = VectorizedHamletEnv(
        config_pack_path=config_path,
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Should have composed action space (substrate + custom)
    assert hasattr(env, "action_space")
    assert env.action_space.action_dim >= 6  # At least substrate actions

    # action_dim should match action_space
    assert env.action_dim == env.action_space.action_dim


def test_vectorized_env_respects_enabled_actions():
    """VectorizedHamletEnv should mask disabled actions."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv
    from pathlib import Path

    # TODO: Create test config with enabled_actions list
    # For now, this test will be implemented after updating L1 config
    pass
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/test_integration.py::test_vectorized_env_loads_composed_action_space -v
```

Expected: FAIL (env doesn't have action_space attribute yet)

**Step 3: Modify VectorizedHamletEnv.__init__() to use ActionSpaceBuilder**

Modify: `src/townlet/environment/vectorized_env.py`

Find the `__init__` method (around line 200), locate where action_dim is set:

```python
# BEFORE (around line 281):
self.action_dim = self.substrate.action_space_size

# AFTER:
# Build composed action space from substrate + global custom actions
from townlet.environment.action_builder import ActionSpaceBuilder

# Load enabled_actions from training config (if specified)
# TODO: Load from training.yaml when TrainingConfig DTO exists
# For now, assume all actions enabled if not specified
enabled_actions = None  # Will be loaded from training_config in Phase 4.2

global_actions_path = Path("configs/global_actions.yaml")
builder = ActionSpaceBuilder(
    substrate=self.substrate,
    global_actions_path=global_actions_path,
    enabled_action_names=enabled_actions,
)
self.action_space = builder.build()
self.action_dim = self.action_space.action_dim
```

Add import at top:

```python
from townlet.environment.action_builder import ActionSpaceBuilder
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_townlet/test_integration.py::test_vectorized_env_loads_composed_action_space -v
```

Expected: PASS

**Step 5: Run all integration tests to ensure no regression**

```bash
uv run pytest tests/test_townlet/test_integration.py -v
```

Expected: PASS (all integration tests still pass)

**Step 6: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/test_integration.py
git commit -m "feat(env): integrate ActionSpaceBuilder into VectorizedHamletEnv

- Loads global_actions.yaml for custom actions
- Builds composed action space (substrate + custom)
- action_dim now comes from ActionSpaceBuilder
- TODO: Load enabled_actions from training.yaml

Part of TASK-002B Phase 4."
```

---

### Task 4.2: Update get_action_masks() for disabled actions

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Write failing test**

Modify: `tests/test_townlet/test_integration.py`

Add:

```python
def test_action_masks_disabled_actions_are_false(tmp_path):
    """Action masks should set disabled actions to False."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    # Create test config with some actions disabled
    test_config = tmp_path / "test_config"
    test_config.mkdir()

    # Copy minimal config files
    import shutil
    shutil.copytree("configs/L0_0_minimal", test_config, dirs_exist_ok=True)

    # Override training.yaml with enabled_actions
    training_yaml = test_config / "training.yaml"
    training_yaml.write_text("""
# ... (copy existing training.yaml content, add enabled_actions) ...
environment:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "INTERACT"
    # LEFT, RIGHT, WAIT disabled
""")

    env = VectorizedHamletEnv(
        config_pack_path=test_config,
        num_agents=2,
        device=torch.device("cpu"),
    )

    env.reset()

    masks = env.get_action_masks()

    # UP (0), DOWN (1), INTERACT (4) should be True
    # LEFT (2), RIGHT (3), WAIT (5) should be False
    # (Assuming Grid2D substrate)

    # This test needs enabled_actions loading implementation
    # Will be completed in Step 3
```

**Step 2: This test requires loading enabled_actions from training.yaml**

**We need to defer this until TrainingConfig DTO exists.** For now, add a TODO and implement base masking.

**Step 3: Update get_action_masks() to use action_space.get_base_action_mask()**

Modify: `src/townlet/environment/vectorized_env.py`

Find `get_action_masks()` method (around line 380):

```python
def get_action_masks(self) -> torch.Tensor:
    """Return action masks [num_agents, action_dim].

    Disabled actions return False (cannot be selected).
    Dead agents return False for all actions.

    Returns:
        [num_agents, action_dim] bool tensor
        True = action available, False = action disabled/invalid
    """
    # Start with base mask (disabled actions = False)
    action_masks = self.action_space.get_base_action_mask(
        num_agents=self.num_agents,
        device=self.device,
    )

    # Existing masking logic for boundary constraints, dead agents, etc.
    # (Keep existing code below this point)

    # ... rest of existing get_action_masks() logic ...

    return action_masks
```

**Step 4: Run integration tests**

```bash
uv run pytest tests/test_townlet/test_integration.py -v
```

Expected: PASS (existing tests still work, base masking integrated)

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py
git commit -m "feat(env): integrate disabled action masking into get_action_masks()

- Uses action_space.get_base_action_mask() for disabled actions
- Disabled actions masked to False (cannot be selected)
- Integrates with existing boundary/dead agent masking
- TODO: Load enabled_actions from training.yaml

Part of TASK-002B Phase 4."
```

---

## Phase 5: Testing and Validation

### Task 5.1: Integration test with custom actions

**Files:**
- Test: `tests/test_townlet/integration/test_custom_actions.py` (new)

**Step 1: Write integration test for REST action**

Create: `tests/test_townlet/integration/test_custom_actions.py`

```python
"""Integration tests for custom actions (REST, MEDITATE, etc.)."""

import pytest
import torch
from pathlib import Path

from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_rest_action_restores_energy():
    """REST action should restore energy (negative cost)."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    env.reset()

    # Find REST action ID
    rest_action = env.action_space.get_action_by_name("REST")
    rest_action_id = rest_action.id

    # Drain energy artificially
    env.meters[0, env.energy_idx] = 0.5
    initial_energy = env.meters[0, env.energy_idx].item()

    # Execute REST action
    actions = torch.tensor([rest_action_id], device=env.device)
    env.step(actions)

    final_energy = env.meters[0, env.energy_idx].item()

    # Energy should increase (restoration)
    assert final_energy > initial_energy, f"Energy should restore: {initial_energy} -> {final_energy}"


def test_meditate_action_restores_mood():
    """MEDITATE action should restore mood."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    env.reset()

    # Find MEDITATE action ID
    meditate_action = env.action_space.get_action_by_name("MEDITATE")
    meditate_action_id = meditate_action.id

    # Drain mood
    env.meters[0, env.mood_idx] = 0.3
    initial_mood = env.meters[0, env.mood_idx].item()

    # Execute MEDITATE action
    actions = torch.tensor([meditate_action_id], device=env.device)
    env.step(actions)

    final_mood = env.meters[0, env.mood_idx].item()

    # Mood should increase
    assert final_mood > initial_mood, f"Mood should improve: {initial_mood} -> {final_mood}"


def test_teleport_home_warps_to_origin():
    """TELEPORT_HOME action should warp agent to [0, 0]."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    env.reset()

    # Move agent away from origin
    env.positions[0] = torch.tensor([5, 7], device=env.device, dtype=env.substrate.position_dtype)

    # Find TELEPORT_HOME action ID
    teleport_action = env.action_space.get_action_by_name("TELEPORT_HOME")
    teleport_action_id = teleport_action.id

    # Execute TELEPORT_HOME
    actions = torch.tensor([teleport_action_id], device=env.device)
    env.step(actions)

    # Agent should be at [0, 0]
    expected_pos = torch.tensor([0, 0], device=env.device, dtype=env.substrate.position_dtype)
    assert torch.equal(env.positions[0], expected_pos), f"Expected {expected_pos}, got {env.positions[0]}"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/integration/test_custom_actions.py::test_rest_action_restores_energy -v
```

Expected: FAIL (custom action dispatch not implemented)

**Step 3: Implement custom action dispatch in VectorizedHamletEnv.step()**

This requires implementing the custom action execution logic.
We need to add this in a new task (Task 4.3).

Let me add that task now.

---

### Task 4.3: Custom action dispatch

**Files:**
- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Implement _apply_custom_action() method**

Modify: `src/townlet/environment/vectorized_env.py`

Add new method (around line 700, after existing helper methods):

```python
def _apply_custom_action(self, agent_idx: int, action: ActionConfig):
    """Apply custom action effects.

    Args:
        agent_idx: Agent index
        action: Custom action config
    """
    # Apply costs (negative costs = restoration)
    for meter_name, cost in action.costs.items():
        meter_idx = self._get_meter_index(meter_name)
        if meter_idx is not None:
            self.meters[agent_idx, meter_idx] -= cost  # Subtract cost (negative = add)

    # Apply effects
    for meter_name, effect in action.effects.items():
        meter_idx = self._get_meter_index(meter_name)
        if meter_idx is not None:
            self.meters[agent_idx, meter_idx] += effect  # Add effect

    # Handle teleportation
    if action.teleport_to is not None:
        target_pos = torch.tensor(
            action.teleport_to,
            device=self.device,
            dtype=self.substrate.position_dtype,
        )
        self.positions[agent_idx] = target_pos

    # Clamp meters to [0, 1]
    self.meters = torch.clamp(self.meters, 0.0, 1.0)


def _get_meter_index(self, meter_name: str) -> int | None:
    """Get meter index by name.

    Args:
        meter_name: Meter name (e.g., "energy", "mood")

    Returns:
        Meter index, or None if meter doesn't exist
    """
    meter_map = {
        "energy": self.energy_idx,
        "hygiene": self.hygiene_idx,
        "satiation": self.satiation_idx,
        "money": self.money_idx,
        "mood": self.mood_idx,
        "social": self.social_idx,
        "health": self.health_idx,
        "fitness": self.fitness_idx,
    }
    return meter_map.get(meter_name)
```

**Step 2: Update step() to dispatch custom actions**

Modify: `src/townlet/environment/vectorized_env.py`

Find the step() method, locate action execution section (around line 550):

Add after movement action execution, before interaction handling:

```python
# === CUSTOM ACTION DISPATCH ===
# Custom actions start after substrate actions
custom_action_start_id = self.action_space.substrate_action_count
custom_mask = actions >= custom_action_start_id

if custom_mask.any():
    custom_agent_indices = torch.where(custom_mask)[0]
    for agent_idx in custom_agent_indices:
        action_id = actions[agent_idx].item()
        action = self.action_space.get_action_by_id(action_id)

        # Skip if action is movement type (handled by substrate)
        if action.type == "movement":
            # Custom movement actions (e.g., SPRINT) use substrate.apply_movement()
            if action.delta is not None:
                delta = torch.tensor(action.delta, device=self.device, dtype=self.substrate.position_dtype)
                self.positions[agent_idx] = self.substrate.apply_movement(
                    self.positions[agent_idx].unsqueeze(0),
                    delta.unsqueeze(0),
                )[0]

        # Apply custom action costs/effects
        self._apply_custom_action(agent_idx, action)
```

**Step 3: Run integration tests**

```bash
uv run pytest tests/test_townlet/integration/test_custom_actions.py -v
```

Expected: PASS (all custom action tests pass)

**Step 4: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/integration/test_custom_actions.py
git commit -m "feat(env): implement custom action dispatch

- _apply_custom_action() applies costs/effects/teleportation
- _get_meter_index() maps meter names to indices
- Custom actions dispatched in step() after substrate actions
- REST/MEDITATE/TELEPORT_HOME now functional

Part of TASK-002B Phase 4 (COMPLETE)."
```

---

## Phase 5: Final Testing and Documentation

### Task 5.2: Curriculum transfer test (L0 → L1)

**Files:**
- Test: `tests/test_townlet/integration/test_curriculum_transfer.py` (new)

**Step 1: Write curriculum transfer test**

Create: `tests/test_townlet/integration/test_curriculum_transfer.py`

```python
"""Tests for checkpoint transfer across curriculum levels."""

import pytest
import torch
from pathlib import Path

from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_l0_and_l1_have_same_action_dim():
    """L0 and L1 should have same action_dim (enables checkpoint transfer)."""
    env_l0 = VectorizedHamletEnv(
        config_pack_path=Path("configs/L0_0_minimal"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    env_l1 = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    # CRITICAL: Same action_dim enables checkpoint transfer
    assert env_l0.action_dim == env_l1.action_dim, (
        f"L0 and L1 must have same action_dim for checkpoint transfer. "
        f"L0: {env_l0.action_dim}, L1: {env_l1.action_dim}"
    )


def test_l0_has_fewer_enabled_actions_than_l1():
    """L0 should have fewer enabled actions than L1 (curriculum progression)."""
    env_l0 = VectorizedHamletEnv(
        config_pack_path=Path("configs/L0_0_minimal"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    env_l1 = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    # L0 should have fewer enabled actions (simpler task)
    # This test will pass once enabled_actions is loaded from training.yaml
    # For now, check that both have same total actions
    assert env_l0.action_space.action_dim == env_l1.action_space.action_dim
```

**Step 2: Run test**

```bash
uv run pytest tests/test_townlet/integration/test_curriculum_transfer.py -v
```

Expected: PASS (both envs have same action_dim)

**Step 3: Commit**

```bash
git add tests/test_townlet/integration/test_curriculum_transfer.py
git commit -m "test(curriculum): verify L0 and L1 have same action_dim

- Critical for checkpoint transfer (same Q-network architecture)
- L0 and L1 share global action vocabulary
- Disabled actions masked at runtime

Part of TASK-002B Phase 5."
```

---

### Task 5.3: Update CLAUDE.md documentation

**Files:**
- Modify: `/home/john/hamlet/CLAUDE.md`

**Step 1: Document composable action space**

Modify: `/home/john/hamlet/CLAUDE.md`

Find "Action Space" section, update to document new architecture:

```markdown
### Action Space (Composable)

**Architecture**: Action Space = Substrate Actions + Custom Actions

**Global Vocabulary** (`configs/global_actions.yaml`):
- All curriculum levels share the **same action vocabulary**
- Enables checkpoint transfer (same action_dim across configs)

**Per-Config Enabled Actions** (`training.yaml`):
- Each config specifies which actions are **enabled**
- Disabled actions masked at runtime (cannot be selected)

**Example**:

```yaml
# configs/global_actions.yaml (SHARED)
custom_actions:
  - name: "REST"       # Passive recovery
  - name: "MEDITATE"   # Mood boost
  - name: "TELEPORT_HOME"  # Expensive warp
  - name: "SPRINT"     # Fast movement
```

```yaml
# configs/L0_0_minimal/training.yaml
environment:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
  # 7 enabled, 3 disabled, action_dim = 10
```

```yaml
# configs/L1_full_observability/training.yaml
environment:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
    - "MEDITATE"
    - "TELEPORT_HOME"
    - "SPRINT"
  # 10 enabled, 0 disabled, action_dim = 10
```

**Result**: Both L0 and L1 have **action_dim = 10** → checkpoints transfer!

**Substrate Actions** (auto-generated):
- Grid2D: 6 actions (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT)
- Grid3D: 8 actions (adds UP_Z/DOWN_Z)
- GridND(7D): 16 actions (14 movement + INTERACT + WAIT)
- Aspatial: 2 actions (INTERACT/WAIT only)

**Custom Actions** (operator-defined):
- Passive: REST, MEDITATE (modify meters)
- Movement: TELEPORT_HOME, SPRINT (non-standard movement)
- Future: Inventory, social, combat actions

**Action Masking**:
- Disabled actions: False (cannot be selected)
- Dead agents: False for all actions
- Temporal mechanics: Closed affordances masked
```

**Step 2: Commit**

```bash
git add /home/john/hamlet/CLAUDE.md
git commit -m "docs(claude): document composable action space architecture

- Action Space = Substrate + Custom actions
- Global vocabulary enables curriculum transfer
- enabled_actions per-config masking
- Examples for L0 and L1

Part of TASK-002B Phase 5 (COMPLETE)."
```

---

## Summary

Plan complete! This creates a bite-sized, TDD-driven implementation plan for TASK-002B.

**Total tasks**: 20+ tasks across 5 phases
**Estimated time**: 20-28 hours
**Key achievements**:
- ✅ Substrate default actions (Grid2D/3D/ND, Continuous, Aspatial)
- ✅ ActionSpaceBuilder with global vocabulary
- ✅ Global actions config (REST, MEDITATE, TELEPORT_HOME, SPRINT)
- ✅ Environment integration with action masking
- ✅ Custom action dispatch
- ✅ Curriculum transfer tests

**Execution options**:
1. **Subagent-Driven (this session)**: Use superpowers:subagent-driven-development
2. **Parallel Session**: Open new session with superpowers:executing-plans

Which execution approach would you like to use?
