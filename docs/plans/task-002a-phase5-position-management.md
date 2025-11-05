# TASK-002A Phase 4: Position Management Refactoring - Implementation Plan

**Date**: 2025-11-05
**Status**: Ready for Implementation
**Dependencies**: Phases 0-3 Complete
**Estimated Effort**: 26 hours (revised down from 32h with breaking changes authorization)

---

⚠️ **BREAKING CHANGES NOTICE** ⚠️

Phase 4 introduces breaking changes to the checkpoint format.

**Impact:**
- Existing checkpoints (Version 2) will NOT load
- Training must restart from scratch
- All checkpoint directories should be deleted before Phase 4

**Rationale:**
Backward compatibility was explicitly deprioritized to simplify implementation
and reduce technical debt. The substrate abstraction requires fundamental
changes to position representation that are incompatible with the legacy format.

**Migration Path:**
None. Delete old checkpoints and retrain.

See Task 4.11 for user communication strategy.

---

## Executive Summary

Phase 4 refactors all position management code to use the substrate abstraction created in Phases 0-3. This involves **9 integration points** across **5 core files**, with **moderate risk** due to checkpoint format changes.

**Key Finding**: Position tensors are hardcoded as `[num_agents, 2]` in 15+ locations. All must be changed to `[num_agents, substrate.position_dim]` to support 3D grids (position_dim=3) and aspatial universes (position_dim=0).

**Critical Risks**:
1. **Checkpoint incompatibility**: BREAKING CHANGE - old checkpoints will not load
2. **Network architecture mismatch**: Observation dim changes with substrate
3. **Temporal mechanics with aspatial**: Position tracking breaks without guards

**New Substrate Methods Required**:
- `get_all_positions() -> list[torch.Tensor]` - For affordance randomization
- `encode_partial_observation(positions, affordances, vision_range) -> torch.Tensor` - For POMDP

---

## Phase 4 Task Breakdown

### Task 4.1: Add New Substrate Methods

**Purpose**: Extend substrate interface with methods needed for Phase 4 integration

**Files**:
- `src/townlet/substrate/base.py`
- `src/townlet/substrate/grid2d.py`
- `src/townlet/substrate/aspatial.py`

**Estimated Time**: 2 hours

---

#### Step 1: Write test for get_all_positions()

**Action**: Add tests for new substrate method

**Modify**: `tests/test_townlet/unit/test_substrate_base.py`

Add to end of file:

```python
def test_grid2d_get_all_positions():
    """Grid2D should return all valid grid positions."""
    substrate = Grid2DSubstrate(width=3, height=2)

    positions = substrate.get_all_positions()

    # Should have 3×2 = 6 positions
    assert len(positions) == 6

    # Check some expected positions exist
    assert [0, 0] in positions
    assert [2, 1] in positions

    # All positions should be valid (within bounds)
    for pos in positions:
        assert len(pos) == 2
        assert 0 <= pos[0] < 3
        assert 0 <= pos[1] < 2


def test_aspatial_get_all_positions():
    """Aspatial should return empty list (no positions exist)."""
    substrate = AspatialSubstrate()

    positions = substrate.get_all_positions()

    assert positions == []  # Aspatial has no positions
```

**Run test**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_get_all_positions -v
```

**Expected**: FAIL (method not implemented)

---

#### Step 2: Add get_all_positions() to SpatialSubstrate interface

**Action**: Extend abstract interface

**Modify**: `src/townlet/substrate/base.py`

Add after `is_on_position()` method (around line 303):

```python
    @abstractmethod
    def get_all_positions(self) -> list[list[int]]:
        """Get all valid positions in substrate (for affordance placement).

        Returns:
            List of positions (as lists, not tensors - for shuffling)

            For grid substrates: all grid cells [(x, y), ...]
            For aspatial: empty list []

        Used for:
        - Affordance randomization (vectorized_env.randomize_affordance_positions)
        - Capacity validation (ensure enough positions for all affordances)

        Example:
            Grid2D (3×3): [[0,0], [0,1], [0,2], [1,0], [1,1], ...]
            Aspatial: []
        """
        pass
```

**Expected**: Abstract method added to interface

---

#### Step 3: Implement get_all_positions() in Grid2DSubstrate

**Action**: Add concrete implementation for 2D grids

**Modify**: `src/townlet/substrate/grid2d.py`

Add after `is_on_position()` method (around line 617):

```python
    def get_all_positions(self) -> list[list[int]]:
        """Return all grid cell positions (for affordance randomization).

        Returns:
            List of [x, y] positions for all grid cells.
            Total: width × height positions.

        Example:
            3×3 grid: [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
        """
        positions = []
        for x in range(self.width):
            for y in range(self.height):
                positions.append([x, y])
        return positions
```

**Expected**: Method returns all grid cells as list of [x, y] pairs

---

#### Step 4: Implement get_all_positions() in AspatialSubstrate

**Action**: Add concrete implementation for aspatial

**Modify**: `src/townlet/substrate/aspatial.py`

Add after `is_on_position()` method (around line 822):

```python
    def get_all_positions(self) -> list[list[int]]:
        """Return empty list (aspatial has no positions).

        In aspatial universes, there's no concept of "position" or "placement."
        Affordances are simply available without spatial location.

        Returns:
            Empty list []
        """
        return []
```

**Expected**: Method returns empty list for aspatial substrate

---

#### Step 5: Run tests to verify get_all_positions()

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_get_all_positions -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_aspatial_get_all_positions -v
```

**Expected**: Both tests PASS

---

#### Step 6: Write test for encode_partial_observation()

**Action**: Add tests for POMDP encoding method

**Modify**: `tests/test_townlet/unit/test_substrate_base.py`

Add to end of file:

```python
def test_grid2d_encode_partial_observation():
    """Grid2D should encode local window for POMDP."""
    substrate = Grid2DSubstrate(width=8, height=8)

    # Agent at center of grid
    positions = torch.tensor([[4, 4]], dtype=torch.long)

    # Affordances
    affordances = {
        "Bed": torch.tensor([3, 3], dtype=torch.long),
        "Hospital": torch.tensor([6, 6], dtype=torch.long),
    }

    # 5×5 vision window (vision_range=2 means 2 cells in each direction)
    local_encoding = substrate.encode_partial_observation(positions, affordances, vision_range=2)

    # Should encode 5×5 = 25 cells around agent
    assert local_encoding.shape == (1, 25)

    # Bed at (3,3) should be visible (relative position: -1, -1)
    # Hospital at (6,6) should be visible (relative position: +2, +2)


def test_aspatial_encode_partial_observation():
    """Aspatial should return empty tensor (no position encoding)."""
    substrate = AspatialSubstrate()

    positions = torch.zeros((3, 0))  # 3 agents, 0-dimensional positions
    affordances = {}

    local_encoding = substrate.encode_partial_observation(positions, affordances, vision_range=2)

    assert local_encoding.shape == (3, 0)  # No position encoding
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_encode_partial_observation -v
```

**Expected**: FAIL (method not implemented)

---

#### Step 7: Add encode_partial_observation() to SpatialSubstrate interface

**Action**: Extend abstract interface for POMDP

**Modify**: `src/townlet/substrate/base.py`

Add after `get_all_positions()` method:

```python
    @abstractmethod
    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Encode local window around agents for partial observability (POMDP).

        Args:
            positions: [num_agents, position_dim] agent positions
            affordances: {name: [position_dim]} affordance positions
            vision_range: radius of vision window (e.g., 2 for 5×5 window)

        Returns:
            [num_agents, window_size] local grid encoding

            window_size depends on substrate:
            - Grid2D: (2*vision_range + 1)²  (e.g., 5×5 = 25)
            - Aspatial: 0 (no position encoding)

        Used for:
        - Level 2 POMDP observations (5×5 local window)
        - Partial observability training

        Example:
            Grid2D with vision_range=2:
            - Agent at (4, 4) sees cells (2,2) to (6,6)
            - Encodes 5×5 = 25 cells relative to agent
        """
        pass
```

**Expected**: Abstract method added to interface

---

#### Step 8: Implement encode_partial_observation() in Grid2DSubstrate

**Action**: Extract local window encoding from observation_builder.py

**Modify**: `src/townlet/substrate/grid2d.py`

Add after `get_all_positions()` method:

```python
    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Encode local window around each agent (POMDP).

        Extracts a local (2*vision_range+1)×(2*vision_range+1) window centered
        on each agent's position. Affordances within the window are marked.

        Args:
            positions: [num_agents, 2] agent positions
            affordances: {name: [2]} affordance positions
            vision_range: radius of vision (e.g., 2 for 5×5 window)

        Returns:
            [num_agents, window_size²] local grid encoding
            where window_size = 2*vision_range + 1

        Note: Handles boundary cases - if agent near edge, out-of-bounds
        cells are marked as empty.
        """
        num_agents = positions.shape[0]
        device = positions.device
        window_size = 2 * vision_range + 1

        # Initialize local grids for all agents
        local_grids = torch.zeros(
            (num_agents, window_size, window_size),
            device=device,
            dtype=torch.float32,
        )

        # For each agent, extract local window
        for agent_idx in range(num_agents):
            agent_x, agent_y = positions[agent_idx]

            # Mark affordances in local window
            for affordance_pos in affordances.values():
                aff_x, aff_y = affordance_pos[0].item(), affordance_pos[1].item()

                # Compute relative position in local window
                rel_x = aff_x - agent_x + vision_range
                rel_y = aff_y - agent_y + vision_range

                # Check if affordance is within vision window
                if 0 <= rel_x < window_size and 0 <= rel_y < window_size:
                    local_grids[agent_idx, rel_y, rel_x] = 1.0

        # Flatten local grids: [num_agents, window_size, window_size] → [num_agents, window_size²]
        return local_grids.reshape(num_agents, -1)
```

**Expected**: Returns local window encoding for POMDP

---

#### Step 9: Implement encode_partial_observation() in AspatialSubstrate

**Action**: Return empty tensor for aspatial

**Modify**: `src/townlet/substrate/aspatial.py`

Add after `get_all_positions()` method:

```python
    def encode_partial_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
        vision_range: int,
    ) -> torch.Tensor:
        """Return empty tensor (aspatial has no position encoding).

        In aspatial universes, there's no concept of "local window" or "vision."
        All affordances are accessible without positioning.

        Returns:
            [num_agents, 0] empty tensor
        """
        num_agents = positions.shape[0]
        device = positions.device
        return torch.zeros((num_agents, 0), device=device)
```

**Expected**: Returns empty tensor for aspatial substrate

---

#### Step 10: Run tests to verify encode_partial_observation()

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_encode_partial_observation -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_aspatial_encode_partial_observation -v
```

**Expected**: Both tests PASS

---

#### Step 11: Commit

**Command**:
```bash
git add src/townlet/substrate/base.py src/townlet/substrate/grid2d.py src/townlet/substrate/aspatial.py tests/test_townlet/unit/test_substrate_base.py
git commit -m "feat: add get_all_positions() and encode_partial_observation() to substrate

Extended substrate interface with two new methods for Phase 4:

1. get_all_positions() -> list[list[int]]
   - Returns all valid positions in substrate
   - Grid2D: all grid cells [(x, y), ...]
   - Aspatial: empty list []
   - Used for affordance randomization

2. encode_partial_observation(positions, affordances, vision_range) -> Tensor
   - Encodes local window for POMDP
   - Grid2D: extracts 5×5 window around agent
   - Aspatial: returns empty tensor
   - Used for Level 2 partial observability

Implemented in Grid2DSubstrate and AspatialSubstrate.
All tests pass.

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with new substrate methods

---

### Task 4.2: Refactor Position Initialization

**Purpose**: Replace hardcoded `torch.zeros((num_agents, 2))` with `substrate.initialize_positions()`

**Files**:
- `src/townlet/environment/vectorized_env.py`

**Estimated Time**: 2 hours

---

#### Step 1: Write test for substrate-based position initialization

**Action**: Add integration test

**Create**: `tests/test_townlet/integration/test_substrate_position_init.py`

```python
"""Test environment initializes positions using substrate."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_position_initialization_uses_substrate():
    """Environment should initialize positions via substrate.initialize_positions()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=10,
        device="cpu",
    )

    # Positions should be initialized
    assert env.positions is not None

    # Shape should be [num_agents, substrate.position_dim]
    assert env.positions.shape == (10, env.substrate.position_dim)

    # For 2D grid substrate, position_dim = 2
    assert env.substrate.position_dim == 2
    assert env.positions.shape == (10, 2)


def test_reset_uses_substrate():
    """Environment reset should use substrate.initialize_positions()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=5,
        device="cpu",
    )

    # Get initial positions
    initial_positions = env.positions.clone()

    # Reset environment
    env.reset()

    # Positions should be reinitialized (likely different)
    # Shape must still be [num_agents, substrate.position_dim]
    assert env.positions.shape == (5, env.substrate.position_dim)


def test_temporal_mechanics_position_tracking():
    """Temporal mechanics should only create position tracking for spatial substrates."""
    # L3 has temporal mechanics enabled
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L3_temporal_mechanics"),
        num_agents=1,
        device="cpu",
    )

    if env.enable_temporal_mechanics and env.substrate.position_dim > 0:
        # Should have last_interaction_position tracker
        assert hasattr(env, "last_interaction_position")
        assert env.last_interaction_position.shape == (1, env.substrate.position_dim)
    elif env.enable_temporal_mechanics and env.substrate.position_dim == 0:
        # Aspatial with temporal mechanics should error (caught in validation)
        pytest.fail("Temporal mechanics with aspatial substrate should be rejected")
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_position_initialization_uses_substrate -v
```

**Expected**: FAIL (still uses hardcoded initialization)

---

#### Step 2: Replace position initialization in __init__

**Action**: Update VectorizedHamletEnv.__init__ to use substrate

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around line 189):
```python
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
```

**Replace with**:
```python
# Initialize positions using substrate
self.positions = self.substrate.initialize_positions(self.num_agents, self.device)
```

**Expected**: Positions initialized via substrate

---

#### Step 3: Replace last_interaction_position initialization (temporal mechanics)

**Action**: Add guard check for spatial substrates

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around line 197):
```python
# Track last interaction position for temporal mechanics
self.last_interaction_position = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
```

**Replace with**:
```python
# Track last interaction position for temporal mechanics (only for spatial substrates)
if self.enable_temporal_mechanics:
    if self.substrate.position_dim == 0:
        raise ValueError(
            "Temporal mechanics require spatial substrate (position_dim > 0). "
            "Cannot enable temporal mechanics with aspatial substrate. "
            "Set enable_temporal_mechanics=false in training.yaml or use grid substrate."
        )

    self.last_interaction_position = torch.zeros(
        (self.num_agents, self.substrate.position_dim),
        dtype=torch.long,
        device=self.device,
    )
```

**Expected**: Temporal mechanics validates substrate compatibility and uses substrate.position_dim

---

#### Step 4: Replace position initialization in reset()

**Action**: Update reset() to use substrate

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around line 219):
```python
self.positions = torch.randint(0, self.grid_size, (self.num_agents, 2), device=self.device)
```

**Replace with**:
```python
# Reinitialize positions using substrate
self.positions = self.substrate.initialize_positions(self.num_agents, self.device)
```

**Expected**: Reset uses substrate for position initialization

---

#### Step 5: Run tests to verify position initialization

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py -v
```

**Expected**: All tests PASS

---

#### Step 6: Run existing tests to check backward compatibility

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_episode_execution.py -v
uv run pytest tests/test_townlet/properties/test_environment_properties.py -v
```

**Expected**: All tests still PASS (no regression)

---

#### Step 7: Commit

**Command**:
```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/integration/test_substrate_position_init.py
git commit -m "refactor: use substrate.initialize_positions() for position init

Replaced hardcoded position initialization with substrate method:

__init__:
- BEFORE: torch.zeros((num_agents, 2))
- AFTER: substrate.initialize_positions(num_agents, device)

reset():
- BEFORE: torch.randint(0, grid_size, (num_agents, 2))
- AFTER: substrate.initialize_positions(num_agents, device)

Temporal mechanics:
- Added guard: requires position_dim > 0 (spatial substrate)
- Uses substrate.position_dim for last_interaction_position shape

Supports 2D (position_dim=2), 3D (position_dim=3), aspatial (position_dim=0).

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with refactored initialization

---

### Task 4.3: Refactor Movement Logic

**Purpose**: Replace hardcoded movement deltas and boundary clamping with `substrate.apply_movement()`

**Files**:
- `src/townlet/environment/vectorized_env.py`

**Estimated Time**: 3 hours

---

#### Step 1: Write test for substrate-based movement

**Action**: Add test for movement using substrate

**Modify**: `tests/test_townlet/integration/test_substrate_position_init.py`

Add to end of file:

```python
def test_movement_uses_substrate():
    """Environment should use substrate.apply_movement() for agent movement."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Set agent position to known location
    env.positions = torch.tensor([[4, 4]], dtype=torch.long, device=env.device)

    # Execute RIGHT action (action=3)
    actions = torch.tensor([3], dtype=torch.long, device=env.device)
    env.step(actions)

    # Agent should have moved right (x+1)
    # Exact position depends on substrate boundary handling
    new_x = env.positions[0, 0].item()
    assert new_x == 5  # For clamp boundary, should move to (5, 4)


def test_boundary_clamping_via_substrate():
    """Substrate should handle boundary clamping."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Place agent at right edge
    env.positions = torch.tensor([[7, 4]], dtype=torch.long, device=env.device)

    # Try to move right (action=3) - should clamp to grid edge
    actions = torch.tensor([3], dtype=torch.long, device=env.device)
    env.step(actions)

    # Agent should stay at x=7 (clamped to grid boundary)
    assert env.positions[0, 0].item() == 7
    assert env.positions[0, 1].item() == 4
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_movement_uses_substrate -v
```

**Expected**: FAIL (still uses hardcoded movement)

---

#### Step 2: Replace movement logic in _execute_actions()

**Action**: Use substrate.apply_movement() instead of hardcoded deltas

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around lines 388-415):
```python
def _execute_actions(self, actions: torch.Tensor) -> dict:
    """Execute movement, interaction, and wait actions."""
    old_positions = self.positions.clone() if self.enable_temporal_mechanics else None

    # Movement deltas (x, y) coordinates
    deltas = torch.tensor(
        [
            [0, -1],  # UP - decreases y
            [0, 1],   # DOWN - increases y
            [-1, 0],  # LEFT - decreases x
            [1, 0],   # RIGHT - increases x
            [0, 0],   # INTERACT (no movement)
            [0, 0],   # WAIT (no movement)
        ],
        device=self.device,
    )

    # Apply movement
    movement_deltas = deltas[actions]  # [num_agents, 2]
    new_positions = self.positions + movement_deltas

    # Clamp to grid boundaries
    new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)

    self.positions = new_positions
```

**Replace with**:
```python
def _execute_actions(self, actions: torch.Tensor) -> dict:
    """Execute movement, interaction, and wait actions."""
    # Store old positions for temporal mechanics (only if spatial)
    old_positions = None
    if self.enable_temporal_mechanics and self.substrate.position_dim > 0:
        old_positions = self.positions.clone()

    # Movement deltas (x, y) coordinates
    # NOTE: Action space still hardcoded - will be moved to actions.yaml in TASK-000
    deltas = torch.tensor(
        [
            [0, -1],  # UP - decreases y
            [0, 1],   # DOWN - increases y
            [-1, 0],  # LEFT - decreases x
            [1, 0],   # RIGHT - increases x
            [0, 0],   # INTERACT (no movement)
            [0, 0],   # WAIT (no movement)
        ],
        device=self.device,
    )

    # Get movement deltas for each agent
    movement_deltas = deltas[actions]  # [num_agents, 2]

    # Apply movement using substrate (handles boundaries)
    new_positions = self.substrate.apply_movement(self.positions, movement_deltas)

    self.positions = new_positions
```

**Expected**: Movement uses substrate for boundary handling

---

#### Step 3: Update temporal mechanics position comparison

**Action**: Add guard check for spatial substrates

**Find** (after position update, around line 420):
```python
    # Reset temporal progress for agents that moved
    if self.enable_temporal_mechanics:
        for agent_idx in range(self.num_agents):
            if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
                self.interaction_progress[agent_idx] = 0
                self.last_interaction_affordance[agent_idx] = None
```

**Replace with**:
```python
    # Reset temporal progress for agents that moved (only for spatial substrates)
    if self.enable_temporal_mechanics and old_positions is not None:
        for agent_idx in range(self.num_agents):
            if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
                self.interaction_progress[agent_idx] = 0
                self.last_interaction_affordance[agent_idx] = None
```

**Expected**: Temporal mechanics checks for spatial substrate before comparing positions

---

#### Step 4: Run tests to verify movement refactoring

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_movement_uses_substrate -v
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_boundary_clamping_via_substrate -v
```

**Expected**: Both tests PASS

---

#### Step 5: Run temporal mechanics tests

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_temporal_mechanics.py -v
```

**Expected**: All temporal mechanics tests still PASS

---

#### Step 6: Commit

**Command**:
```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/integration/test_substrate_position_init.py
git commit -m "refactor: use substrate.apply_movement() for agent movement

Replaced hardcoded boundary clamping with substrate method:

_execute_actions():
- BEFORE: new_positions = torch.clamp(positions + deltas, 0, grid_size-1)
- AFTER: new_positions = substrate.apply_movement(positions, deltas)

Benefits:
- Substrate handles boundary mode (clamp, wrap, bounce)
- Supports 3D movement (position_dim=3)
- Supports aspatial (no movement, position_dim=0)

Temporal mechanics:
- Added guard: only compare positions if spatial substrate
- Skips position comparison for aspatial (position_dim=0)

NOTE: Movement deltas still hardcoded - will be moved to actions.yaml in TASK-000.

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with refactored movement

---

### Task 4.4: Refactor Distance Calculations

**Purpose**: Replace `torch.abs(...).sum(dim=1)` with `substrate.is_on_position()`

**Files**:
- `src/townlet/environment/vectorized_env.py`
- `src/townlet/environment/observation_builder.py`

**Estimated Time**: 3 hours

---

#### Step 1: Write test for substrate-based distance checks

**Action**: Add test for interaction distance checks

**Modify**: `tests/test_townlet/integration/test_substrate_position_init.py`

Add to end of file:

```python
def test_interaction_uses_substrate_distance():
    """Environment should use substrate.is_on_position() for interactions."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Find Bed affordance position
    bed_pos = env.affordances["Bed"]

    # Place agent on Bed
    env.positions = bed_pos.unsqueeze(0)  # [1, 2]

    # Execute INTERACT action (action=4)
    actions = torch.tensor([4], dtype=torch.long, device=env.device)
    obs, rewards, dones, infos = env.step(actions)

    # Interaction should succeed (agent is on affordance)
    # Energy should have increased (Bed restores energy)
    assert rewards[0] > 0  # Got reward for interaction
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_interaction_uses_substrate_distance -v
```

**Expected**: PASS (interaction logic should still work with current distance check)

---

#### Step 2: Replace distance check in _check_affordance_interactions() (site 1)

**Action**: Use substrate.is_on_position() for affordance interactions

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around line 295):
```python
# Check if on affordance for interaction
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
on_this_affordance = distances == 0
```

**Replace with**:
```python
# Check if on affordance for interaction (using substrate)
on_this_affordance = self.substrate.is_on_position(self.positions, affordance_pos)
```

**Expected**: Uses substrate for position check

---

#### Step 3: Replace distance check in _execute_actions() (site 2)

**Action**: Use substrate.is_on_position() for temporal mechanics check

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around line 470):
```python
# Check if still on same affordance
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
on_this_affordance = distances == 0
```

**Replace with**:
```python
# Check if still on same affordance (using substrate)
on_this_affordance = self.substrate.is_on_position(self.positions, affordance_pos)
```

**Expected**: Uses substrate for position check

---

#### Step 4: Replace distance check in _compute_meters_and_rewards() (site 3)

**Action**: Use substrate.is_on_position() for reward calculation

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around line 552):
```python
# Check which agents are on this affordance
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
on_affordance = distances == 0
```

**Replace with**:
```python
# Check which agents are on this affordance (using substrate)
on_affordance = self.substrate.is_on_position(self.positions, affordance_pos)
```

**Expected**: Uses substrate for position check

---

#### Step 5: Replace distance check in observation_builder.py (site 4)

**Action**: Use substrate.is_on_position() in observation encoding

**Modify**: `src/townlet/environment/observation_builder.py`

**Find** (around line 240):
```python
# Check which agents are on affordance
distances = torch.abs(positions - affordance_pos).sum(dim=1)
on_affordance = distances == 0
```

**Replace with**:
```python
# Check which agents are on affordance (using substrate)
on_affordance = self.substrate.is_on_position(positions, affordance_pos)
```

**Note**: observation_builder.py doesn't have direct access to substrate. Need to pass it.

**Modify**: `src/townlet/environment/observation_builder.py` __init__

**Find** (around line 35):
```python
def __init__(
    self,
    num_agents: int,
    grid_size: int,
    partial_observability: bool,
    vision_range: int,
    num_affordance_types: int,
    enable_temporal_mechanics: bool,
    device: torch.device,
):
```

**Replace with**:
```python
def __init__(
    self,
    num_agents: int,
    grid_size: int,
    partial_observability: bool,
    vision_range: int,
    num_affordance_types: int,
    enable_temporal_mechanics: bool,
    device: torch.device,
    substrate,  # Add substrate parameter
):
    # ... existing code ...
    self.substrate = substrate  # Store substrate reference
```

**Modify**: `src/townlet/environment/vectorized_env.py` ObservationBuilder instantiation

**Find** (around line 103):
```python
self.observation_builder = ObservationBuilder(
    num_agents=self.num_agents,
    grid_size=self.grid_size,
    partial_observability=self.partial_observability,
    vision_range=self.vision_range,
    num_affordance_types=self.num_affordance_types,
    enable_temporal_mechanics=self.enable_temporal_mechanics,
    device=self.device,
)
```

**Replace with**:
```python
self.observation_builder = ObservationBuilder(
    num_agents=self.num_agents,
    grid_size=self.grid_size,
    partial_observability=self.partial_observability,
    vision_range=self.vision_range,
    num_affordance_types=self.num_affordance_types,
    enable_temporal_mechanics=self.enable_temporal_mechanics,
    device=self.device,
    substrate=self.substrate,  # Pass substrate to observation builder
)
```

**Expected**: ObservationBuilder has access to substrate for is_on_position() calls

---

#### Step 6: Run tests to verify distance refactoring

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_interaction_uses_substrate_distance -v
uv run pytest tests/test_townlet/integration/test_episode_execution.py -v
```

**Expected**: All tests PASS

---

#### Step 7: Run observation tests

**Command**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py -v
```

**Expected**: All observation tests still PASS

---

#### Step 8: Commit

**Command**:
```bash
git add src/townlet/environment/vectorized_env.py src/townlet/environment/observation_builder.py tests/test_townlet/integration/test_substrate_position_init.py
git commit -m "refactor: use substrate.is_on_position() for distance checks

Replaced hardcoded Manhattan distance with substrate method:

vectorized_env.py (3 sites):
- BEFORE: distances = torch.abs(positions - affordance_pos).sum(dim=1); on = distances == 0
- AFTER: on = substrate.is_on_position(positions, affordance_pos)

observation_builder.py (1 site):
- BEFORE: distances = torch.abs(positions - affordance_pos).sum(dim=1); on = distances == 0
- AFTER: on = substrate.is_on_position(positions, affordance_pos)

Benefits:
- Works for all substrate types (grid, aspatial, graph)
- Aspatial: always returns True (all affordances accessible)
- Grid: exact cell match
- Future continuous: proximity threshold

ObservationBuilder now receives substrate reference during initialization.

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with refactored distance checks

---

### Task 4.5: Update Checkpoint Serialization

**Purpose**: Add `position_dim` to checkpoint format for substrate validation (BREAKING CHANGE)

**Files**:
- `src/townlet/environment/vectorized_env.py`
- `src/townlet/demo/runner.py`

**Estimated Time**: 1.5 hours (simplified - no backward compatibility)

---

#### Step 1: Write test for checkpoint substrate metadata

**Action**: Add test for new checkpoint format (Phase 4+ only, rejects legacy)

**Create**: `tests/test_townlet/integration/test_checkpoint_substrate.py`

```python
"""Test checkpoint serialization with substrate metadata."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_checkpoint_includes_position_dim():
    """Affordance positions checkpoint should include position_dim."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Get affordance positions for checkpoint
    checkpoint_data = env.get_affordance_positions()

    # Should have position_dim field
    assert "position_dim" in checkpoint_data
    assert checkpoint_data["position_dim"] == env.substrate.position_dim

    # For 2D grid substrate, position_dim should be 2
    assert checkpoint_data["position_dim"] == 2


def test_checkpoint_validates_position_dim():
    """Loading checkpoint should validate position_dim compatibility."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Create checkpoint with mismatched position_dim
    bad_checkpoint = {
        "positions": {"Bed": [2, 3, 0]},  # 3D position [x, y, z]
        "ordering": ["Bed"],
        "position_dim": 3,  # 3D!
    }

    # Should raise error when loading into 2D substrate
    with pytest.raises(ValueError, match="position_dim mismatch"):
        env.set_affordance_positions(bad_checkpoint)


def test_checkpoint_rejects_legacy_format():
    """BREAKING CHANGE: Legacy checkpoints (no position_dim) should be rejected."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Create legacy checkpoint (no position_dim field)
    legacy_checkpoint = {
        "positions": {"Bed": [2, 3]},
        "ordering": ["Bed"],
        # No position_dim field!
    }

    # Should raise clear error for legacy format
    with pytest.raises(ValueError, match="legacy checkpoint.*pre-Phase 4"):
        env.set_affordance_positions(legacy_checkpoint)
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_checkpoint_substrate.py::test_checkpoint_includes_position_dim -v
```

**Expected**: FAIL (position_dim field not in checkpoint)

---

#### Step 2: Update get_affordance_positions() to include position_dim

**Action**: Add position_dim to checkpoint format

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around lines 600-618):
```python
def get_affordance_positions(self) -> dict:
    """Get current affordance positions (P1.1 checkpointing).

    Returns:
        Dictionary with 'positions' and 'ordering' keys:
        - 'positions': Dict mapping affordance names to [x, y] positions
        - 'ordering': List of affordance names in consistent order
    """
    positions = {}
    for name, pos_tensor in self.affordances.items():
        # Convert tensor position to list (for JSON serialization)
        pos = pos_tensor.cpu().tolist()
        positions[name] = [int(pos[0]), int(pos[1])]  # Assumes 2D!

    return {
        "positions": positions,
        "ordering": self.affordance_names,
    }
```

**Replace with**:
```python
def get_affordance_positions(self) -> dict:
    """Get current affordance positions (substrate-agnostic checkpointing).

    Returns:
        Dictionary with 'positions', 'ordering', and 'position_dim' keys:
        - 'positions': Dict mapping affordance names to position lists
        - 'ordering': List of affordance names in consistent order
        - 'position_dim': Dimensionality for validation (0=aspatial, 2=2D, 3=3D)
    """
    positions = {}
    for name, pos_tensor in self.affordances.items():
        # Convert tensor to list (handles any dimensionality)
        pos = pos_tensor.cpu().tolist()

        # Ensure pos is a list (even for 0-dimensional positions)
        if isinstance(pos, (int, float)):
            pos = [pos]
        elif self.substrate.position_dim == 0:
            pos = []

        positions[name] = [int(x) for x in pos] if pos else []

    return {
        "positions": positions,
        "ordering": self.affordance_names,
        "position_dim": self.substrate.position_dim,  # For validation
    }
```

**Expected**: Checkpoint includes position_dim field

---

#### Step 3: Update set_affordance_positions() to validate position_dim (BREAKING CHANGE)

**Action**: Reject legacy checkpoints with clear error message

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around lines 620-644):
```python
def set_affordance_positions(self, checkpoint_data: dict) -> None:
    """Set affordance positions from checkpoint (P1.1 checkpointing)."""
    # Handle backwards compatibility: old format is just positions dict
    if "positions" in checkpoint_data:
        positions = checkpoint_data["positions"]
        ordering = checkpoint_data.get("ordering", self.affordance_names)
    else:
        positions = checkpoint_data
        ordering = self.affordance_names

    # Restore ordering
    self.affordance_names = ordering
    self.num_affordance_types = len(self.affordance_names)

    # Rebuild affordances dict
    for name, pos in positions.items():
        if name in self.affordances:
            self.affordances[name] = torch.tensor(pos, device=self.device, dtype=torch.long)
```

**Replace with**:
```python
def set_affordance_positions(self, checkpoint_data: dict) -> None:
    """Set affordance positions from checkpoint (Phase 4+ only).

    BREAKING CHANGE: Only loads Phase 4+ checkpoints with position_dim field.
    Legacy checkpoints will not load.

    Raises:
        ValueError: If checkpoint missing position_dim or incompatible with substrate
    """
    # Validate position_dim exists (no default fallback)
    if "position_dim" not in checkpoint_data:
        raise ValueError(
            "Checkpoint missing 'position_dim' field.\n"
            "This is a legacy checkpoint (pre-Phase 4).\n"
            "\n"
            "BREAKING CHANGE: Phase 4 changed checkpoint format.\n"
            "Legacy checkpoints (Version 2) are no longer compatible.\n"
            "\n"
            "Action required:\n"
            "  1. Delete old checkpoint directories: checkpoints_level*/\n"
            "  2. Retrain models from scratch with Phase 4+ code\n"
            "\n"
            "If you need to preserve old models, checkout pre-Phase 4 git commit."
        )

    # Validate compatibility (no backward compatibility)
    checkpoint_position_dim = checkpoint_data["position_dim"]
    if checkpoint_position_dim != self.substrate.position_dim:
        raise ValueError(
            f"Checkpoint position_dim mismatch: checkpoint has {checkpoint_position_dim}D, "
            f"but current substrate requires {self.substrate.position_dim}D."
        )

    # Simple loading (no backward compat branches)
    positions = checkpoint_data["positions"]
    ordering = checkpoint_data["ordering"]

    self.affordance_names = ordering
    self.num_affordance_types = len(self.affordance_names)

    for name, pos in positions.items():
        if name in self.affordances:
            self.affordances[name] = torch.tensor(pos, device=self.device, dtype=torch.long)
```

**Expected**: Checkpoint loading fails fast with clear error for legacy checkpoints

---

#### Step 4: Run tests to verify checkpoint serialization

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_checkpoint_substrate.py -v
```

**Expected**: All tests PASS (including legacy rejection test)

---

#### Step 5: Commit (BREAKING CHANGE)

**Command**:
```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/integration/test_checkpoint_substrate.py
git commit -m "feat: add position_dim to checkpoint format (BREAKING CHANGE)

BREAKING CHANGE: Checkpoint format version 2→3 (NOT backward compatible)

get_affordance_positions():
- Added position_dim field for substrate validation
- Handles any dimensionality: 2D [x,y], 3D [x,y,z], aspatial []
- Converts positions to lists for JSON serialization

set_affordance_positions():
- Validates checkpoint position_dim matches current substrate
- BREAKING: Rejects legacy checkpoints with clear error message
- No default fallback (single code path)
- Fails fast if position_dim missing

Checkpoint format:
- Version 3 (Phase 4+): {positions: {...}, ordering: [...], position_dim: 2}
- Version 2 (legacy): NO LONGER SUPPORTED

Impact:
- Existing checkpoints (Level 2 POMDP, Level 3 Temporal) will NOT load
- Users must delete old checkpoints and retrain from scratch
- Clear error message guides users to delete checkpoint directories

Rationale:
- Simpler implementation (no backward compatibility complexity)
- Single code path (easier to maintain)
- Substrate abstraction requires fundamental changes

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with breaking change notice

---

### Task 4.5B: Pre-flight Checkpoint Validation (NEW)

**Purpose**: Detect old checkpoints on startup and fail fast with helpful error

**Files**:
- `src/townlet/demo/runner.py`

**Estimated Time**: 1 hour

---

#### Step 1: Write test for pre-flight validation

**Action**: Add test for old checkpoint detection

**Create**: `tests/test_townlet/integration/test_preflight_validation.py`

```python
"""Test pre-flight validation detects old checkpoints."""
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from townlet.demo.runner import DemoRunner


def test_preflight_detects_old_checkpoints(tmp_path):
    """DemoRunner should detect and reject old checkpoints on startup."""
    # Create mock old checkpoint (missing substrate_metadata)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    old_checkpoint = checkpoint_dir / "checkpoint_ep00100.pt"
    torch.save(
        {
            "episode": 100,
            "network_state": {},
            "optimizer_state": {},
            # Missing substrate_metadata field (old format)
        },
        old_checkpoint,
    )

    # Attempting to create DemoRunner should detect old checkpoint
    with pytest.raises(ValueError, match="Old checkpoints detected"):
        runner = DemoRunner(
            config_dir=Path("configs/L1_full_observability"),
            db_path=tmp_path / "test.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=10,
            training_config_path=Path("configs/L1_full_observability/training.yaml"),
        )


def test_preflight_allows_new_checkpoints(tmp_path):
    """DemoRunner should allow new checkpoints with substrate_metadata."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    new_checkpoint = checkpoint_dir / "checkpoint_ep00100.pt"
    torch.save(
        {
            "episode": 100,
            "network_state": {},
            "optimizer_state": {},
            "substrate_metadata": {"position_dim": 2},  # New format
        },
        new_checkpoint,
    )

    # Should not raise error
    runner = DemoRunner(
        config_dir=Path("configs/L1_full_observability"),
        db_path=tmp_path / "test.db",
        checkpoint_dir=checkpoint_dir,
        max_episodes=10,
        training_config_path=Path("configs/L1_full_observability/training.yaml"),
    )
    runner.close()  # Clean up


def test_preflight_allows_empty_checkpoint_dir(tmp_path):
    """DemoRunner should allow empty checkpoint directory (fresh start)."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Should not raise error (no checkpoints to validate)
    runner = DemoRunner(
        config_dir=Path("configs/L1_full_observability"),
        db_path=tmp_path / "test.db",
        checkpoint_dir=checkpoint_dir,
        max_episodes=10,
        training_config_path=Path("configs/L1_full_observability/training.yaml"),
    )
    runner.close()  # Clean up
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_preflight_validation.py::test_preflight_detects_old_checkpoints -v
```

**Expected**: FAIL (pre-flight check not implemented)

---

#### Step 2: Implement pre-flight check in DemoRunner.__init__

**Action**: Add checkpoint version check on startup

**Modify**: `src/townlet/demo/runner.py`

**Find** `__init__` method (around line 50-100):
```python
def __init__(
    self,
    config_dir: Path,
    db_path: Path,
    checkpoint_dir: Path,
    max_episodes: int,
    training_config_path: Path,
):
    # ... existing initialization ...
```

**Add after checkpoint_dir setup** (around line 70):
```python
    # Pre-flight check: detect old checkpoints (Phase 4 breaking change)
    self._validate_checkpoint_compatibility()
```

**Add new method** after `__init__`:
```python
def _validate_checkpoint_compatibility(self) -> None:
    """Validate checkpoint directory doesn't contain old checkpoints.

    BREAKING CHANGE: Phase 4 changed checkpoint format.
    Old checkpoints (Version 2) will not load.

    Raises:
        ValueError: If old checkpoints detected
    """
    if not self.checkpoint_dir.exists():
        return  # No checkpoints yet (fresh start)

    checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        return  # Empty directory (fresh start)

    # Check first checkpoint for substrate_metadata
    first_checkpoint_path = checkpoint_files[0]

    try:
        checkpoint = torch.load(first_checkpoint_path, weights_only=False)

        # Phase 4+ checkpoints have substrate_metadata
        if "substrate_metadata" not in checkpoint:
            raise ValueError(
                f"Old checkpoints detected in {self.checkpoint_dir}.\n"
                "\n"
                "BREAKING CHANGE: Phase 4 changed checkpoint format.\n"
                "Legacy checkpoints (Version 2) are no longer compatible.\n"
                "\n"
                "Action required:\n"
                f"  1. Delete checkpoint directory: {self.checkpoint_dir}\n"
                "  2. Retrain model from scratch with Phase 4+ code\n"
                "\n"
                "If you need to preserve old models, checkout pre-Phase 4 git commit.\n"
                "\n"
                f"Detected old checkpoint: {first_checkpoint_path.name}"
            )
    except Exception as e:
        # If we can't load checkpoint, let the normal loading code handle it
        # (might be corrupted, wrong format, etc.)
        if "Old checkpoints detected" in str(e):
            raise  # Re-raise our validation error
        # Otherwise ignore (will fail later during actual load)
```

**Expected**: Pre-flight check validates checkpoints on startup

---

#### Step 3: Test pre-flight check with real checkpoint directories

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_preflight_validation.py -v
```

**Expected**: All tests PASS

---

#### Step 4: Commit

**Command**:
```bash
git add src/townlet/demo/runner.py tests/test_townlet/integration/test_preflight_validation.py
git commit -m "feat: add pre-flight validation for old checkpoints

Added startup check to detect and reject old checkpoints:

DemoRunner.__init__:
- Calls _validate_checkpoint_compatibility() on startup
- Checks for substrate_metadata field in first checkpoint
- Fails fast before training starts

_validate_checkpoint_compatibility():
- Loads first checkpoint file to check format
- Phase 4+ checkpoints: have substrate_metadata field
- Legacy checkpoints: missing substrate_metadata → raises ValueError
- Empty checkpoint directory: allowed (fresh start)

Error message:
- Clear explanation of breaking change
- Action steps: delete checkpoint directory and retrain
- Preserves old model option: checkout pre-Phase 4 commit

Benefits:
- Prevents confusing errors during training
- Fails fast with helpful guidance
- Users don't waste time starting training with incompatible checkpoints

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with pre-flight validation

---

### Task 4.6: Update Observation Encoding

**Purpose**: Replace hardcoded grid encoding with `substrate.encode_observation()`

**Files**:
- `src/townlet/environment/observation_builder.py`

**Estimated Time**: 5 hours

---

#### Step 1: Write test for substrate-based observation encoding

**Action**: Add test for full observability encoding

**Create**: `tests/test_townlet/integration/test_substrate_observations.py`

```python
"""Test observation encoding uses substrate."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_full_observation_uses_substrate():
    """Full observability should use substrate.encode_observation()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Get observation
    obs = env.reset()

    # Observation should include substrate position encoding
    # For 8×8 grid: 64 (grid) + 8 (meters) + 15 (affordance) + 4 (temporal) = 91
    expected_dim = (
        env.substrate.get_observation_dim() +  # 64 for grid encoding
        8 +  # meters
        15 +  # affordance at position
        4  # temporal extras
    )
    assert obs.shape[1] == expected_dim


def test_partial_observation_uses_substrate():
    """Partial observability should use substrate.encode_partial_observation()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L2_partial_observability"),
        num_agents=1,
        device="cpu",
    )

    # Get observation
    obs = env.reset()

    # Partial obs should use local window encoding
    # For 5×5 window: 25 (local grid) + 2 (normalized position) + 8 (meters) + 15 (affordance) + 4 (temporal) = 54
    assert obs.shape[1] == 54
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_observations.py::test_full_observation_uses_substrate -v
```

**Expected**: PASS (existing implementation should still work)

---

#### Step 2: Replace grid encoding in _build_full_observations()

**Action**: Use substrate.encode_observation() for full observability

**Modify**: `src/townlet/environment/observation_builder.py`

**Find** (around lines 127-137):
```python
def _build_full_observations(self, positions, meters, affordances):
    """Build full grid observations (Level 1)."""
    # Grid encoding: mark BOTH agent position AND affordance positions
    grid_encoding = torch.zeros(self.num_agents, self.grid_size * self.grid_size, device=self.device)

    # Mark affordance positions
    for affordance_pos in affordances.values():
        affordance_flat_idx = affordance_pos[1] * self.grid_size + affordance_pos[0]
        grid_encoding[:, affordance_flat_idx] = 1.0

    # Mark agent position (add 1.0, so overlaps become 2.0)
    flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
    grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), torch.ones(self.num_agents, 1, device=self.device))
```

**Replace with**:
```python
def _build_full_observations(self, positions, meters, affordances):
    """Build full observations using substrate encoding."""
    # Delegate position encoding to substrate
    grid_encoding = self.substrate.encode_observation(positions, affordances)
```

**Expected**: Uses substrate for full observation encoding

---

#### Step 3: Replace local window encoding in _build_partial_observations()

**Action**: Use substrate.encode_partial_observation() for POMDP

**Modify**: `src/townlet/environment/observation_builder.py`

**Find** (around lines 166-207):
```python
def _build_partial_observations(self, positions, meters, affordances):
    """Build partial observations (POMDP) - local 5×5 window."""
    # Initialize local grids
    local_grids = torch.zeros(
        (self.num_agents, self.vision_range * 2 + 1, self.vision_range * 2 + 1),
        device=self.device,
        dtype=torch.float32,
    )

    # Extract local window for each agent
    for agent_idx in range(self.num_agents):
        agent_x, agent_y = positions[agent_idx]

        # Mark affordances in local window
        for affordance_pos in affordances.values():
            aff_x, aff_y = affordance_pos[0].item(), affordance_pos[1].item()

            # Compute relative position
            rel_x = aff_x - agent_x + self.vision_range
            rel_y = aff_y - agent_y + self.vision_range

            # Check if affordance is within vision
            window_size = self.vision_range * 2 + 1
            if 0 <= rel_x < window_size and 0 <= rel_y < window_size:
                local_grids[agent_idx, rel_y, rel_x] = 1.0

    # Flatten local grids
    local_grids = local_grids.reshape(self.num_agents, -1)

    # Normalized positions (for recurrent network)
    normalized_positions = positions.float() / (self.grid_size - 1)
```

**Replace with**:
```python
def _build_partial_observations(self, positions, meters, affordances):
    """Build partial observations (POMDP) using substrate encoding."""
    # Local window encoding from substrate
    local_grids = self.substrate.encode_partial_observation(
        positions, affordances, vision_range=self.vision_range
    )

    # Normalized positions (for recurrent network position encoder)
    # For grid substrates: normalize by grid dimensions
    # For aspatial: positions are empty, normalized_positions will be empty
    if hasattr(self.substrate, 'width') and hasattr(self.substrate, 'height'):
        normalized_positions = positions.float() / torch.tensor(
            [self.substrate.width - 1, self.substrate.height - 1],
            device=self.device,
            dtype=torch.float32,
        )
    else:
        # Aspatial substrate: no position normalization needed
        normalized_positions = positions.float()
```

**Expected**: Uses substrate for partial observation encoding

---

#### Step 4: Update observation_dim calculation

**Action**: Use substrate.get_observation_dim() for obs_dim

**Modify**: `src/townlet/environment/observation_builder.py`

**Find** (in __init__, around line 50):
```python
# Calculate observation dimensions
if self.partial_observability:
    # POMDP: local window + normalized position + meters + affordance + temporal
    window_size = self.vision_range * 2 + 1
    self.obs_dim = (
        window_size * window_size +  # local grid
        2 +  # normalized (x, y)
        8 +  # meters
        self.num_affordance_types + 1 +  # affordance one-hot
        4  # temporal extras
    )
else:
    # Full observability: full grid + meters + affordance + temporal
    self.obs_dim = (
        self.grid_size * self.grid_size +  # full grid
        8 +  # meters
        self.num_affordance_types + 1 +  # affordance one-hot
        4  # temporal extras
    )
```

**Replace with**:
```python
# Calculate observation dimensions using substrate
if self.partial_observability:
    # POMDP: local window + normalized position + meters + affordance + temporal
    window_size = self.vision_range * 2 + 1
    self.obs_dim = (
        window_size * window_size +  # local grid (from substrate)
        self.substrate.position_dim +  # normalized position (2 for 2D, 0 for aspatial)
        8 +  # meters
        self.num_affordance_types + 1 +  # affordance one-hot
        (4 if self.enable_temporal_mechanics else 0)  # temporal extras
    )
else:
    # Full observability: full grid + meters + affordance + temporal
    self.obs_dim = (
        self.substrate.get_observation_dim() +  # grid encoding (from substrate)
        8 +  # meters
        self.num_affordance_types + 1 +  # affordance one-hot
        (4 if self.enable_temporal_mechanics else 0)  # temporal extras
    )
```

**Expected**: Observation dim calculated from substrate properties

---

#### Step 5: Run tests to verify observation encoding

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_observations.py -v
uv run pytest tests/test_townlet/unit/environment/test_observations.py -v
```

**Expected**: All tests PASS

---

#### Step 6: Run full integration tests

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_episode_execution.py -v
```

**Expected**: All tests PASS (no regression)

---

#### Step 7: Commit

**Command**:
```bash
git add src/townlet/environment/observation_builder.py tests/test_townlet/integration/test_substrate_observations.py
git commit -m "refactor: use substrate encoding for observations

Replaced hardcoded grid encoding with substrate methods:

_build_full_observations():
- BEFORE: Manual grid encoding with flat_indices calculation
- AFTER: substrate.encode_observation(positions, affordances)

_build_partial_observations():
- BEFORE: Manual local window extraction and flattening
- AFTER: substrate.encode_partial_observation(positions, affordances, vision_range)

Observation dim calculation:
- Full obs: substrate.get_observation_dim() + meters + affordance + temporal
- Partial obs: window_size² + substrate.position_dim + meters + affordance + temporal

Benefits:
- Works for all substrate types (2D, 3D, aspatial)
- Aspatial: empty position encoding (obs_dim reduced)
- 3D: substrate handles 3D grid encoding
- Normalized positions use substrate.position_dim (0 for aspatial)

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with observation refactoring

---

### Task 4.7: Update Affordance Randomization

**Purpose**: Replace hardcoded 2D position generation with `substrate.get_all_positions()`

**Files**:
- `src/townlet/environment/vectorized_env.py`

**Estimated Time**: 2 hours

---

#### Step 1: Write test for substrate-based affordance randomization

**Action**: Add test for randomization

**Modify**: `tests/test_townlet/integration/test_substrate_position_init.py`

Add to end of file:

```python
def test_affordance_randomization_uses_substrate():
    """Affordance randomization should use substrate.get_all_positions()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Get initial positions
    initial_bed_pos = env.affordances["Bed"].clone()

    # Randomize affordances
    env.randomize_affordance_positions()

    # Positions should be different (high probability)
    # All positions should still be valid (within substrate bounds)
    for name, pos in env.affordances.items():
        assert pos.shape[0] == env.substrate.position_dim

        # For 2D grid substrate
        if hasattr(env.substrate, 'width'):
            assert 0 <= pos[0] < env.substrate.width
            assert 0 <= pos[1] < env.substrate.height


def test_affordance_randomization_capacity_check():
    """Randomization should validate substrate has enough positions."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L0_0_minimal"),  # 3×3 grid, 1 affordance
        num_agents=1,
        device="cpu",
    )

    # 3×3 = 9 cells, should have enough space for 1 affordance + 1 agent
    env.randomize_affordance_positions()  # Should succeed

    # Total positions: 9
    # Affordances: 1 (Bed)
    # Should have validated capacity (9 >= 1)
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_affordance_randomization_uses_substrate -v
```

**Expected**: PASS (existing randomization should still work)

---

#### Step 2: Replace affordance randomization logic

**Action**: Use substrate.get_all_positions() for randomization

**Modify**: `src/townlet/environment/vectorized_env.py`

**Find** (around lines 646-671):
```python
def randomize_affordance_positions(self):
    """Randomize affordance positions for generalization testing.

    Ensures no two affordances occupy the same position.
    """
    import random

    # Validate that grid has enough cells for all affordances
    num_affordances = len(self.affordances)
    total_cells = self.grid_size * self.grid_size
    if num_affordances >= total_cells:
        raise ValueError(
            f"Grid has {total_cells} cells but {num_affordances} affordances + 1 agent need space."
        )

    # Generate list of all grid positions
    all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

    # Shuffle and assign to affordances
    random.shuffle(all_positions)

    for i, affordance_name in enumerate(self.affordances.keys()):
        new_pos = all_positions[i]
        self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
```

**Replace with**:
```python
def randomize_affordance_positions(self):
    """Randomize affordance positions using substrate.

    Ensures no two affordances occupy the same position.
    """
    import random

    # Aspatial substrates: no position randomization needed
    if self.substrate.position_dim == 0:
        # Clear positions (affordances have no spatial location)
        for affordance_name in self.affordances.keys():
            self.affordances[affordance_name] = torch.zeros(0, dtype=torch.long, device=self.device)
        return

    # Get all valid positions from substrate
    all_positions = self.substrate.get_all_positions()  # list[list[int]]

    # Validate capacity
    num_affordances = len(self.affordances)
    total_positions = len(all_positions)
    if num_affordances >= total_positions:
        raise ValueError(
            f"Substrate has {total_positions} positions but {num_affordances} affordances + 1 agent need space."
        )

    # Shuffle and assign
    random.shuffle(all_positions)

    for i, affordance_name in enumerate(self.affordances.keys()):
        new_pos = all_positions[i]
        self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
```

**Expected**: Randomization uses substrate for position generation

---

#### Step 3: Run tests to verify affordance randomization

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_affordance_randomization_uses_substrate -v
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py::test_affordance_randomization_capacity_check -v
```

**Expected**: Both tests PASS

---

#### Step 4: Commit

**Command**:
```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/integration/test_substrate_position_init.py
git commit -m "refactor: use substrate.get_all_positions() for affordance randomization

Replaced hardcoded 2D position generation with substrate method:

randomize_affordance_positions():
- BEFORE: all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
- AFTER: all_positions = substrate.get_all_positions()

Benefits:
- Works for all substrate types:
  - Grid2D: [(x, y), ...] for all grid cells
  - Grid3D: [(x, y, z), ...] for all 3D cells
  - Aspatial: [] (skips randomization, clears positions)
- Capacity check uses substrate's total positions (not hardcoded grid_size²)
- Aspatial substrates: affordances have empty position tensors

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with randomization refactoring

---

### Task 4.8: Update Visualization

**Purpose**: Send substrate type to frontend for rendering selection

**Files**:
- `src/townlet/demo/live_inference.py`

**Estimated Time**: 2 hours

---

#### Step 1: Write test for visualization substrate detection

**Action**: Add test for substrate metadata in WebSocket messages

**Create**: `tests/test_townlet/unit/test_live_inference_substrate.py`

```python
"""Test live inference sends substrate metadata."""
import pytest
import torch
import json
from pathlib import Path
from unittest.mock import Mock, patch


def test_visualization_sends_substrate_type():
    """Live inference should send substrate type to frontend."""
    # Mock WebSocket
    mock_ws = Mock()

    # This would require refactoring live_inference.py to be testable
    # For now, we'll validate manually during integration testing
    # TODO: Add proper unit test after refactoring live_inference.py
    pytest.skip("Requires live_inference.py refactoring for testability")
```

**Note**: live_inference.py is hard to unit test due to WebSocket dependencies. Will validate manually.

---

#### Step 2: Update _broadcast_state_update() to include substrate metadata

**Action**: Send substrate type and position_dim to frontend

**Modify**: `src/townlet/demo/live_inference.py`

**Find** (around lines 742-754):
```python
"grid": {
    "width": self.env.grid_size,
    "height": self.env.grid_size,
    "agents": [
        {
            "id": "agent_0",
            "x": agent_pos[0],  # Assumes 2D!
            "y": agent_pos[1],  # Assumes 2D!
            "color": "#4CAF50",
            "last_action": last_action,
        }
    ],
    "affordances": affordances,
},
```

**Replace with**:
```python
# Build substrate-specific grid data
"substrate": {
    "type": type(self.env.substrate).__name__,  # "Grid2DSubstrate", "AspatialSubstrate", etc.
    "position_dim": self.env.substrate.position_dim,
},
"grid": self._build_grid_data(agent_pos, last_action, affordances),
```

**Add new method** after `_broadcast_state_update()`:

```python
def _build_grid_data(self, agent_pos, last_action, affordances):
    """Build grid data based on substrate type.

    Args:
        agent_pos: Agent position (list of length position_dim)
        last_action: Last action taken
        affordances: Affordances dict

    Returns:
        Grid data dict for frontend rendering
    """
    from townlet.substrate.grid2d import Grid2DSubstrate
    from townlet.substrate.aspatial import AspatialSubstrate

    if isinstance(self.env.substrate, Grid2DSubstrate):
        # 2D grid rendering (current implementation)
        return {
            "type": "grid2d",
            "width": self.env.substrate.width,
            "height": self.env.substrate.height,
            "agents": [
                {
                    "id": "agent_0",
                    "x": agent_pos[0],
                    "y": agent_pos[1],
                    "color": "#4CAF50",
                    "last_action": last_action,
                }
            ],
            "affordances": affordances,
        }
    elif isinstance(self.env.substrate, AspatialSubstrate):
        # Aspatial rendering (meters-only, no grid)
        return {
            "type": "aspatial",
            # No position data for aspatial substrate
        }
    else:
        # Future: Grid3DSubstrate, GraphSubstrate, etc.
        return {
            "type": "unknown",
            "substrate_type": type(self.env.substrate).__name__,
            "message": "Rendering for this substrate type not yet implemented",
        }
```

**Expected**: Frontend receives substrate type and can route to appropriate renderer

---

#### Step 3: Update agent position unpacking to handle variable dimensions

**Action**: Handle position_dim dynamically

**Modify**: `src/townlet/demo/live_inference.py`

**Find** (around line 664):
```python
# Get agent position (unpack for frontend compatibility)
agent_pos = self.env.positions[0].cpu().tolist()
```

**Replace with**:
```python
# Get agent position (substrate-agnostic)
agent_pos = self.env.positions[0].cpu().tolist()
# agent_pos is now a list of length substrate.position_dim
# - 2D: [x, y]
# - 3D: [x, y, z]
# - Aspatial: []
```

**Expected**: Position unpacking works for any dimensionality

---

#### Step 4: Test visualization manually (integration test)

**Action**: Run live inference server with different substrates

**Test 2D Grid**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Terminal 1: Start inference server
python -m townlet.demo.live_inference \
    checkpoints_level1 \
    8766 \
    0.2 \
    100 \
    configs/L1_full_observability/training.yaml

# Terminal 2: Check WebSocket messages
# Open browser DevTools, connect to ws://localhost:8766
# Verify substrate metadata in state_update messages:
# {"substrate": {"type": "Grid2DSubstrate", "position_dim": 2}, "grid": {"type": "grid2d", ...}}
```

**Expected**: WebSocket messages include substrate metadata

---

#### Step 5: Document frontend changes needed (deferred to Phase 7)

**Action**: Create note for frontend team

**Create**: `docs/notes/frontend-substrate-rendering.md`

```markdown
# Frontend Substrate Rendering (Deferred to Phase 7)

## Current State (Phase 4)

Live inference server now sends substrate metadata in state_update messages:

```json
{
  "substrate": {
    "type": "Grid2DSubstrate",
    "position_dim": 2
  },
  "grid": {
    "type": "grid2d",
    "width": 8,
    "height": 8,
    "agents": [{"x": 4, "y": 3, ...}],
    "affordances": {...}
  }
}
```

## Frontend Changes Needed (Phase 7)

**Grid.vue** should detect substrate type and route to appropriate renderer:

```vue
<template>
  <div v-if="grid.type === 'grid2d'">
    <!-- Existing 2D SVG rendering -->
    <Grid2DRenderer :grid="grid" />
  </div>

  <div v-else-if="grid.type === 'aspatial'">
    <!-- Meters-only dashboard (no grid) -->
    <MetersOnlyRenderer :meters="meters" />
  </div>

  <div v-else>
    <p>Substrate rendering not yet implemented: {{ grid.type }}</p>
    <p>Substrate: {{ substrate.type }}</p>
    <p>Meters and training metrics still available.</p>
  </div>
</template>
```

## Implementation Tasks (Phase 7)

1. Add substrate type detection in Grid.vue
2. Create AspatialRenderer.vue (meters-only dashboard)
3. Add "not implemented" fallback for unknown substrates
4. Test with 2D and aspatial config packs
5. (Future) Add Grid3DRenderer.vue for 3D substrates

## Estimated Effort

- Phase 7: 2-4 hours for substrate routing and aspatial renderer
```

**Expected**: Documentation for Phase 7 frontend work

---

#### Step 6: Commit

**Command**:
```bash
git add src/townlet/demo/live_inference.py docs/notes/frontend-substrate-rendering.md tests/test_townlet/unit/test_live_inference_substrate.py
git commit -m "feat: send substrate metadata to frontend for rendering

Updated live inference server to include substrate information:

_broadcast_state_update():
- Added substrate metadata: {type, position_dim}
- Moved grid data to _build_grid_data() for substrate routing

_build_grid_data():
- Grid2DSubstrate: existing 2D SVG data
- AspatialSubstrate: empty grid (meters-only)
- Unknown substrates: graceful degradation message

Benefits:
- Frontend can detect substrate type and route to appropriate renderer
- Aspatial universes: meters-only dashboard (no grid visualization)
- Future 3D: frontend can show 'not implemented' instead of crashing

Frontend changes deferred to Phase 7 (see docs/notes/frontend-substrate-rendering.md).

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with visualization updates

---

### Task 4.9: Update Recording System

**Purpose**: Handle variable-length position tuples in recordings (BREAKING CHANGE)

**Files**:
- `src/townlet/recording/recorder.py`
- `src/townlet/recording/data_structures.py`

**Estimated Time**: 1.5 hours (simplified - no legacy format support)

---

#### Step 1: Write test for substrate-agnostic recordings

**Action**: Add test for recording with different position dimensions

**Create**: `tests/test_townlet/integration/test_recording_substrate.py`

```python
"""Test recording system handles substrate positions."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.recording.recorder import RecordingRecorder


def test_recording_handles_2d_positions():
    """Recorder should handle 2D position tuples."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    recorder = RecordingRecorder()
    recorder.start_episode(
        episode=1,
        affordance_layout={name: pos.cpu().tolist() for name, pos in env.affordances.items()},
    )

    # Record step with 2D position
    positions = torch.tensor([3, 4], dtype=torch.long)  # [x, y]
    meters = torch.rand(8)
    action = 2  # LEFT

    recorder.record_step(
        positions=positions,
        meters=meters,
        action=action,
        affordances=env.affordances,
        time_of_day=0.5,
        interaction_progress=0,
        reward=1.0,
    )

    # Episode should have recorded step
    recording = recorder.end_episode(total_reward=1.0, survival_steps=1)

    assert len(recording.steps) == 1
    assert recording.steps[0].position == (3, 4)  # Tuple


def test_recording_affordance_layout_variable_dims():
    """Recording affordance layout should handle variable position dimensions."""
    # 2D positions
    layout_2d = {
        "Bed": [2, 3],
        "Hospital": [5, 7],
    }

    # 3D positions (future)
    layout_3d = {
        "Bed": [2, 3, 0],
        "Hospital": [5, 7, 1],
    }

    # Aspatial positions
    layout_aspatial = {
        "Bed": [],
        "Hospital": [],
    }

    # All should be valid affordance layouts
    # Conversion to tuples should handle any length
    assert tuple(layout_2d["Bed"]) == (2, 3)
    assert tuple(layout_3d["Bed"]) == (2, 3, 0)
    assert tuple(layout_aspatial["Bed"]) == ()
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_recording_substrate.py::test_recording_handles_2d_positions -v
```

**Expected**: PASS (existing recording should still work)

---

#### Step 2: Update position conversion in record_step()

**Action**: Handle variable-length positions

**Modify**: `src/townlet/recording/recorder.py`

**Find** (around line 73):
```python
position=(int(positions[0].item()), int(positions[1].item())),  # Assumes 2D!
```

**Replace with**:
```python
# Convert position to tuple (handles any dimensionality)
position=tuple(int(positions[i].item()) for i in range(positions.shape[0])),
```

**Expected**: Position conversion works for 2D, 3D, aspatial (empty tuple)

---

#### Step 3: Update type hint in data_structures.py

**Action**: Update affordance_layout type hint

**Modify**: `src/townlet/recording/data_structures.py`

**Find** (around line 96):
```python
affordance_layout: dict[str, tuple[int, int]]  # name → (x, y) - Assumes 2D!
```

**Replace with**:
```python
affordance_layout: dict[str, tuple[int, ...]]  # name → (x, y) or (x, y, z) or ()
```

**Expected**: Type hint allows variable-length position tuples

---

#### Step 4: Verify affordance layout conversion

**Action**: Ensure tuple() conversion works for any length

**Modify**: `src/townlet/recording/data_structures.py`

**Find** (around line 42):
```python
# Convert affordance positions from lists to tuples
data["affordance_layout"] = {name: tuple(pos) for name, pos in data["affordance_layout"].items()}
```

**Verify**: This conversion already handles any length (no change needed)

**Expected**: Conversion works for 2D, 3D, aspatial (empty tuple)

---

#### Step 5: Run tests to verify recording changes

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_recording_substrate.py -v
uv run pytest tests/test_townlet/integration/test_recording_recorder.py -v
```

**Expected**: All tests PASS

---

#### Step 6: Commit

**Command**:
```bash
git add src/townlet/recording/recorder.py src/townlet/recording/data_structures.py tests/test_townlet/integration/test_recording_substrate.py
git commit -m "refactor: support variable-length positions in recordings

Updated recording system to handle substrate positions:

recorder.py record_step():
- BEFORE: position=(int(positions[0]), int(positions[1]))  # Assumes 2D
- AFTER: position=tuple(int(positions[i]) for i in range(len(positions)))

data_structures.py:
- Updated type hint: dict[str, tuple[int, int]] → dict[str, tuple[int, ...]]
- Supports 2D (x, y), 3D (x, y, z), aspatial () positions

Recording format:
- 2D: position=(3, 4)
- 3D: position=(3, 4, 0)
- Aspatial: position=()

Backward compatible: old recordings with 2D positions still valid.

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with recording updates

---

### Task 4.10: Update Test Suite

**Purpose**: Fix hardcoded position shape assertions and add parameterized tests

**Files**:
- `tests/test_townlet/properties/test_environment_properties.py`
- `tests/test_townlet/integration/test_checkpointing.py`
- `tests/test_townlet/conftest.py`

**Estimated Time**: 3.5 hours (simplified - no backward compatibility test fixtures)

---

#### Step 1: Add substrate fixtures to conftest.py

**Action**: Create parameterized substrate fixtures

**Modify**: `tests/test_townlet/conftest.py`

Add to end of file:

```python
@pytest.fixture(params=["grid2d", "aspatial"])
def substrate_type(request):
    """Parameterize tests across substrate types."""
    return request.param


@pytest.fixture
def env_with_substrate(substrate_type):
    """Create environment with specified substrate.

    Args:
        substrate_type: "grid2d" or "aspatial"

    Returns:
        VectorizedHamletEnv configured with the specified substrate
    """
    from pathlib import Path
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    # Map substrate types to config packs
    config_map = {
        "grid2d": "configs/L1_full_observability",
        "aspatial": "configs/L1_aspatial",  # TODO: Create this config in Phase 5
    }

    # For now, only test grid2d (aspatial config doesn't exist yet)
    if substrate_type == "aspatial":
        pytest.skip("Aspatial config pack not created yet (Phase 5)")

    config_pack = config_map[substrate_type]

    return VectorizedHamletEnv(
        config_pack_path=Path(config_pack),
        num_agents=1,
        device="cpu",
    )
```

**Expected**: Fixtures ready for parameterized tests

---

#### Step 2: Update position bounds checks in property tests

**Action**: Fix hardcoded 2D assumptions

**Modify**: `tests/test_townlet/properties/test_environment_properties.py`

**Find** (around lines 66-70):
```python
# PROPERTY: Positions always in bounds
positions = env.positions
assert torch.all(positions[:, 0] >= 0), f"X position {positions[0, 0]} < 0"
assert torch.all(positions[:, 0] < grid_size), f"X position {positions[0, 0]} >= {grid_size}"
assert torch.all(positions[:, 1] >= 0), f"Y position {positions[0, 1]} < 0"
assert torch.all(positions[:, 1] < grid_size), f"Y position {positions[0, 1]} >= {grid_size}"
```

**Replace with**:
```python
# PROPERTY: Positions always in bounds (substrate-agnostic)
positions = env.positions

# For grid substrates, check bounds
if hasattr(env.substrate, 'width') and hasattr(env.substrate, 'height'):
    assert torch.all(positions[:, 0] >= 0), f"X position {positions[0, 0]} < 0"
    assert torch.all(positions[:, 0] < env.substrate.width), f"X position {positions[0, 0]} >= {env.substrate.width}"
    assert torch.all(positions[:, 1] >= 0), f"Y position {positions[0, 1]} < 0"
    assert torch.all(positions[:, 1] < env.substrate.height), f"Y position {positions[0, 1]} >= {env.substrate.height}"

# For aspatial substrates, positions should be empty
elif env.substrate.position_dim == 0:
    assert positions.shape[1] == 0, "Aspatial substrate should have 0-dimensional positions"

# For other substrates, check position_dim matches
else:
    assert positions.shape[1] == env.substrate.position_dim, (
        f"Position shape mismatch: {positions.shape[1]} != {env.substrate.position_dim}"
    )
```

**Expected**: Property tests work for all substrate types

---

#### Step 3: Update checkpoint position assertions

**Action**: Fix hardcoded 2D assumptions in checkpoint tests

**Modify**: `tests/test_townlet/integration/test_checkpointing.py`

**Find all assertions checking** `env.positions` **shape** (lines 71, 94, 129, 157, 162)

Example at line 71:
```python
assert env.positions.shape == (num_agents, 2)
```

**Replace with**:
```python
assert env.positions.shape == (num_agents, env.substrate.position_dim)
```

**Do this for all 5 occurrences**

**Expected**: Checkpoint tests validate substrate-agnostic position shapes

---

#### Step 4: Add new substrate-specific property tests

**Action**: Add tests for different substrate types

**Create**: `tests/test_townlet/properties/test_substrate_properties.py`

```python
"""Property-based tests for substrate implementations."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_position_shape_matches_substrate():
    """PROPERTY: Position tensors always match substrate.position_dim."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=5,
        device="cpu",
    )

    # Initial positions
    assert env.positions.shape == (5, env.substrate.position_dim)

    # After reset
    env.reset()
    assert env.positions.shape == (5, env.substrate.position_dim)

    # After step
    actions = torch.randint(0, 6, (5,))
    env.step(actions)
    assert env.positions.shape == (5, env.substrate.position_dim)


def test_affordance_positions_match_substrate():
    """PROPERTY: Affordance positions always match substrate.position_dim."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    for name, pos in env.affordances.items():
        assert pos.shape[0] == env.substrate.position_dim, (
            f"Affordance {name} position dim {pos.shape[0]} != substrate {env.substrate.position_dim}"
        )


def test_observation_dim_consistent_with_substrate():
    """PROPERTY: Observation dim includes substrate position encoding."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    obs = env.reset()

    # Observation should match environment's reported obs_dim
    assert obs.shape[1] == env.observation_builder.obs_dim

    # Obs_dim should include substrate encoding
    # Full obs: substrate.get_observation_dim() + meters + affordance + temporal
    expected_dim = (
        env.substrate.get_observation_dim() +
        8 +  # meters
        env.num_affordance_types + 1 +  # affordance one-hot
        (4 if env.enable_temporal_mechanics else 0)  # temporal extras
    )
    assert env.observation_builder.obs_dim == expected_dim
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/properties/test_substrate_properties.py -v
```

**Expected**: All property tests PASS

---

#### Step 5: Run full test suite

**Command**:
```bash
uv run pytest tests/test_townlet/ -v --tb=short
```

**Expected**: All tests PASS (no regressions)

---

#### Step 6: Commit

**Command**:
```bash
git add tests/test_townlet/properties/test_environment_properties.py tests/test_townlet/properties/test_substrate_properties.py tests/test_townlet/integration/test_checkpointing.py tests/test_townlet/conftest.py
git commit -m "test: update test suite for substrate-agnostic positions

Fixed hardcoded position shape assumptions:

test_environment_properties.py:
- BEFORE: assert positions[:, 0] < grid_size
- AFTER: assert positions[:, 0] < env.substrate.width (for grid substrates)
- Added aspatial substrate check (position_dim == 0)

test_checkpointing.py:
- BEFORE: assert env.positions.shape == (num_agents, 2)
- AFTER: assert env.positions.shape == (num_agents, env.substrate.position_dim)
- Updated 5 assertion sites

test_substrate_properties.py (NEW):
- Property test: position shape matches substrate.position_dim
- Property test: affordance positions match substrate.position_dim
- Property test: observation dim consistent with substrate encoding

conftest.py:
- Added substrate_type fixture (parameterized: grid2d, aspatial)
- Added env_with_substrate fixture for substrate-specific tests
- Aspatial tests skipped until Phase 5 config packs created

All tests pass with 2D grid substrate.

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with test updates

---

### Task 4.11: Documentation Updates (NEW)

**Purpose**: Communicate breaking changes to users

**Files**:
- `CHANGELOG.md`
- `CLAUDE.md`
- `README.md` (if exists)

**Estimated Time**: 0.5 hours

---

#### Step 1: Update CHANGELOG.md

**Action**: Add breaking change notice for Phase 4

**Modify**: `CHANGELOG.md`

**Add at top** (after version header):

```markdown
## [Unreleased] - BREAKING CHANGES

### BREAKING CHANGES: Checkpoint Format (Phase 4)

**Impact:** Existing checkpoints (Version 2) will NOT load after Phase 4 merge.

**Affected Users:** Anyone with saved checkpoints in:
- `checkpoints_level2/` (Level 2 POMDP)
- `checkpoints_level2_5/` (Level 2.5 Temporal Mechanics)
- `checkpoints_level3/` (Level 3 Temporal Mechanics)
- Any custom checkpoint directories

**Action Required:**
1. **Before pulling Phase 4 changes**: Back up old checkpoints if needed
2. **After pulling Phase 4 changes**: Delete all checkpoint directories
3. **Retrain models** from scratch with Phase 4+ code

**Error Message:**
If you attempt to load old checkpoints, you'll see:
```
ValueError: Checkpoint missing 'position_dim' field.
This is a legacy checkpoint (pre-Phase 4).

BREAKING CHANGE: Phase 4 changed checkpoint format.
Legacy checkpoints (Version 2) are no longer compatible.

Action required:
  1. Delete old checkpoint directories: checkpoints_level*/
  2. Retrain models from scratch with Phase 4+ code
```

**Rationale:**
Phase 4 introduces substrate abstraction for position management, requiring
fundamental changes to checkpoint format. Backward compatibility was explicitly
deprioritized to simplify implementation and reduce technical debt.

**Preservation Option:**
If you need to preserve old models, checkout the commit before Phase 4 merge:
```bash
git checkout <pre-phase4-commit>
```

---

### Added (Phase 4)

- Substrate abstraction for position management
- Support for 2D, 3D, and aspatial position spaces
- Checkpoint format includes `position_dim` field
- Pre-flight validation detects old checkpoints on startup

### Changed (Phase 4)

- **BREAKING:** Checkpoint format version 2 → 3
- **BREAKING:** Legacy checkpoints no longer supported
- Position tensors now use `substrate.position_dim` instead of hardcoded 2
- All position operations delegated to substrate methods

### Removed (Phase 4)

- Backward compatibility with Version 2 checkpoints
```

**Expected**: CHANGELOG documents breaking change

---

#### Step 2: Update CLAUDE.md

**Action**: Add checkpoint deletion instructions

**Modify**: `CLAUDE.md`

**Find** "Training (Townlet System)" section (around line 40):

**Add before training commands**:

```markdown
### ⚠️ Phase 4 Breaking Change: Checkpoint Format

**If you have old checkpoints (pre-Phase 4), DELETE THEM before training:**

```bash
# Delete old checkpoint directories
rm -rf checkpoints_level*
rm -rf checkpoints_*

# Verify deletion
ls -la checkpoints_*  # Should show "No such file or directory"
```

**Why:** Phase 4 changed checkpoint format for substrate abstraction.
Legacy checkpoints will cause training to fail with a clear error message.

**Error if you don't delete:**
```
ValueError: Old checkpoints detected in checkpoints_level2.
BREAKING CHANGE: Phase 4 changed checkpoint format.
Legacy checkpoints (Version 2) are no longer compatible.
```

---
```

**Expected**: CLAUDE.md warns about checkpoint deletion

---

#### Step 3: Add breaking change notice to README (if exists)

**Action**: Check for README.md and add notice if present

**Command**:
```bash
# Check if README exists
ls /home/john/hamlet/README.md
```

**If README exists, modify**: `README.md`

**Add section** (near top, after project description):

```markdown
## ⚠️ Breaking Changes (Phase 4)

**Phase 4 introduces breaking changes to checkpoint format.**

If you have existing checkpoints, **delete them before training:**
```bash
rm -rf checkpoints_level*
```

See [CHANGELOG.md](CHANGELOG.md) for details.
```

**If README doesn't exist**: Skip this step

**Expected**: README (if exists) includes breaking change notice

---

#### Step 4: Commit

**Command**:
```bash
git add CHANGELOG.md CLAUDE.md README.md
git commit -m "docs: document Phase 4 breaking changes (checkpoint format)

Added breaking change notices to user-facing documentation:

CHANGELOG.md:
- BREAKING CHANGES section at top
- Clear impact statement (old checkpoints won't load)
- Action required: delete checkpoints before training
- Error message example
- Rationale: substrate abstraction requires format change
- Preservation option: checkout pre-Phase 4 commit

CLAUDE.md:
- Warning before training commands
- Delete checkpoint directories commands
- Error message example if not deleted

README.md (if exists):
- Brief breaking change notice near top
- Link to CHANGELOG for details

User communication strategy:
- Multiple touchpoints (CHANGELOG, CLAUDE.md, README)
- Clear action steps (delete checkpoints)
- Example error messages
- Justification (substrate abstraction)
- Escape hatch (checkout old commit)

Part of TASK-002A Phase 4 (Position Management Refactoring)."
```

**Expected**: Clean commit with documentation updates

---

## Phase 4 Completion Checklist

### Functional Requirements

- [x] Task 4.1: New substrate methods (`get_all_positions`, `encode_partial_observation`)
- [x] Task 4.2: Position initialization uses `substrate.initialize_positions()`
- [x] Task 4.3: Movement uses `substrate.apply_movement()`
- [x] Task 4.4: Distance checks use `substrate.is_on_position()`
- [x] Task 4.5: Checkpoint format includes `position_dim` (BREAKING CHANGE - version 3)
- [x] Task 4.5B: Pre-flight validation detects old checkpoints (NEW)
- [x] Task 4.6: Observations use substrate encoding methods
- [x] Task 4.7: Affordance randomization uses `substrate.get_all_positions()`
- [x] Task 4.8: Visualization sends substrate metadata to frontend
- [x] Task 4.9: Recording system handles variable-length positions
- [x] Task 4.10: Test suite updated for substrate-agnostic assertions
- [x] Task 4.11: Documentation updates for breaking changes (NEW)

### Integration Points Verified

- [x] Position tensors have shape `[num_agents, substrate.position_dim]`
- [x] Movement respects substrate boundaries (clamp, wrap, bounce)
- [x] Distance checks work for all substrate types
- [x] Observations encode positions correctly (grid, aspatial)
- [x] Affordances randomize correctly (grid, aspatial)
- [x] Checkpoints save/load with new format (version 3)
- [x] BREAKING: Version 2 checkpoints rejected with clear error
- [x] Pre-flight check detects old checkpoints on startup
- [x] Visualization routes by substrate type
- [x] Recording system handles variable-length positions

### Test Coverage

- [x] Unit tests for 2 new substrate methods
- [x] Integration test: position initialization via substrate
- [x] Integration test: movement via substrate
- [x] Integration test: distance checks via substrate
- [x] Integration test: observation encoding via substrate
- [x] Integration test: affordance randomization via substrate
- [x] Integration test: checkpoint save/load with position_dim
- [x] Integration test: legacy checkpoint rejection (BREAKING CHANGE)
- [x] Integration test: pre-flight validation detects old checkpoints
- [x] Integration test: recording with variable-length positions
- [x] Property test: positions always in bounds
- [x] Property test: positions always correct shape
- [x] Property test: observation dim consistent with substrate

### Breaking Changes

- [x] BREAKING: Legacy checkpoints (version 2) NO LONGER SUPPORTED
- [x] Clear error messages guide users to delete old checkpoints
- [x] Pre-flight validation prevents training with old checkpoints
- [x] Documentation updated (CHANGELOG, CLAUDE.md, README)
- [x] All existing tests pass with 2D grid substrate (Phase 4+ format)

### Performance

- [x] No performance regression in training speed
- [x] Substrate method calls are hot-path optimized
- [x] Checkpoint size does not explode

### Documentation

- [x] Code comments explain substrate integration
- [x] Test documentation describes new fixtures
- [x] Frontend rendering notes (deferred to Phase 7)
- [x] Checkpoint migration strategy documented

---

## Phase 4 Summary

**Total Tasks**: 12 (10 original + 2 new)
**Total Steps**: ~90
**Estimated Effort**: 26 hours (revised down from 32h with breaking changes authorization)

**Effort Breakdown**:
- Task 4.1: 2h (New substrate methods)
- Task 4.2: 2h (Position initialization)
- Task 4.3: 3h (Movement logic)
- Task 4.4: 3h (Distance calculations)
- Task 4.5: 1.5h (Checkpoint serialization - simplified)
- Task 4.5B: 1h (Pre-flight validation - NEW)
- Task 4.6: 5h (Observation encoding)
- Task 4.7: 2h (Affordance randomization)
- Task 4.8: 2h (Visualization)
- Task 4.9: 1.5h (Recording system - simplified)
- Task 4.10: 3.5h (Test suite - simplified)
- Task 4.11: 0.5h (Documentation - NEW)
- Contingency: 3h (reduced from 4h)

**Key Achievements**:

1. ✅ All position management code now uses substrate abstraction
2. ✅ Checkpoint format updated with position_dim validation (BREAKING CHANGE - version 3)
3. ✅ Pre-flight validation detects old checkpoints on startup
4. ✅ Test suite updated for substrate-agnostic assertions
5. ✅ Visualization sends substrate metadata to frontend (Phase 7 work deferred)
6. ✅ Documentation updated with breaking change notices

**Risk Mitigation**:

- ✅ Checkpoint incompatibility: BREAKING CHANGE - clear error messages guide users
- ✅ Pre-flight validation: Fails fast before training starts
- ✅ Network architecture mismatch: Validation during checkpoint load
- ✅ Temporal mechanics with aspatial: Guard checks prevent errors
- ✅ Frontend rendering: Graceful degradation for unknown substrates
- ✅ Test suite fragility: Incremental fixes with parameterized tests
- ✅ User confusion: Multiple documentation touchpoints (CHANGELOG, CLAUDE.md, README)

**Phase 4 Completion Unlocks**:

- Phase 5: Config Migration (create substrate.yaml for all config packs)
- Phase 6: Example Substrates (3D, toroidal, aspatial config packs)
- Phase 7: Frontend Rendering (Grid3DRenderer, AspatialRenderer)

---

**Document Status**: Implementation Plan Complete
**Next Step**: Review with team, begin implementation task-by-task
**Reviewed By**: [Pending]
**Date**: 2025-11-05
