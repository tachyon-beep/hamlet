# TASK-002A Phase 6: Observation Builder Integration - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date**: 2025-11-05
**Status**: Ready for Implementation
**Dependencies**: Phases 0-5 Complete
**Estimated Effort**: 16-20 hours

---

⚠️ **BREAKING CHANGES NOTICE** ⚠️

Phase 6 introduces breaking changes to observation dimensions.

**Impact:**
- Observation dimensions will change when using coordinate encoding
- Existing checkpoints trained with one-hot encoding (obs_dim=91 for 8×8 grid) will NOT load if substrate switches to coordinate encoding (obs_dim=29)
- Networks must be retrained when changing position encoding strategy
- RecurrentSpatialQNetwork signature changes (requires `position_dim` parameter)

**Affected Configurations:**
- Grid2D with coordinate encoding: obs_dim changes from 64+27=91 to 2+27=29
- Grid3D: obs_dim uses coordinates (3 dims) instead of one-hot (512+ dims)
- Aspatial: obs_dim has no position encoding (0 dims instead of grid dims)

**Rationale:**
One-hot position encoding prevents 3D substrates (512+ dimensions infeasible).
Coordinate encoding enables 3D, larger grids, and transfer learning across grid sizes.
This change is essential for TASK-002A's goal of supporting configurable spatial substrates.

**Migration Path:**
1. For 2D grids ≤8×8: Default remains one-hot (backward compatible)
2. For 2D grids >8×8: Auto-switches to coordinate encoding (BREAKING)
3. For 3D grids: Always uses coordinate encoding (new feature)
4. For aspatial: No position encoding (new feature)

Operators must delete old checkpoints when changing substrate type or grid size that triggers encoding change.

See Task 6.8 for documentation updates.

---

## Executive Summary

Phase 6 integrates substrate-based observation encoding into `VectorizedHamletEnv` and `ObservationBuilder`, replacing hardcoded grid encoding logic. This involves **4 core integration points** across **3 main files**, with **moderate risk** due to observation dimension changes.

**Key Finding**: Current system has hardcoded position encoding in two places:
1. **Full observability**: One-hot grid encoding (64 dims for 8×8) in `observation_builder.py`
2. **Partial observability (POMDP)**: Normalized coordinates (2 dims) + local 5×5 window in `observation_builder.py`

Both must be replaced with substrate methods to support Grid2D (one-hot or coords), Grid3D (coords only), and Aspatial (no position encoding).

**Critical Insight**: L2 POMDP **already uses coordinate encoding successfully** (normalized x, y), proving that networks can learn spatial reasoning from coordinates instead of one-hot. This validates the substrate approach for all grid types.

**New Substrate Methods Required** (from Task 6.1):
1. `get_observation_dim(partial_observability, vision_range) -> int` - Query obs dimension
2. `encode_observation(agent_positions, affordance_positions) -> Tensor` - Full obs encoding
3. `encode_partial_observation(positions, affordances, vision_range) -> Tensor` - POMDP encoding (already added in Phase 5, but needs affordance overlay)

**Updated Signature** (from Phase 5 Task 6.1):
- Phase 5 added basic `encode_partial_observation()` without affordance overlay
- Phase 6 enhances it to handle affordance positions within local window

---

## Phase 6 Task Breakdown

### Task 6.1: Add Substrate Observation Methods

**Purpose**: Extend substrate interface with observation encoding methods needed for Phase 6

**Files**:
- `src/townlet/substrate/base.py`
- `src/townlet/substrate/grid2d.py`
- `src/townlet/substrate/aspatial.py`

**Estimated Time**: 4 hours

---

#### Step 0: Verify Phase 5 Complete (REQUIRED BEFORE CONTINUING)

**Action**: Run Phase 5 integration tests to verify position management is complete

**Command**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Verify Phase 5 integration tests pass
uv run pytest tests/test_townlet/integration/test_substrate_position_init.py -v

# Verify all substrate methods are integrated
grep -n "substrate.initialize_positions\|substrate.apply_movement\|substrate.get_all_positions" src/townlet/environment/vectorized_env.py
```

**Expected**:
- All Phase 5 integration tests PASS
- Grep shows all 3 methods are being used:
  - `initialize_positions()` at line 246
  - `apply_movement()` at line 431
  - `get_all_positions()` at line 676

**Phase 5 Completion Checklist** (must all be TRUE):
- [ ] Position initialization uses `substrate.initialize_positions()`
- [ ] Movement logic uses `substrate.apply_movement()`
- [ ] Affordance randomization uses `substrate.get_all_positions()`
- [ ] Aspatial guard exists in affordance randomization
- [ ] All position management integration tests pass

**If checklist fails**: STOP - Phase 5 is incomplete. Complete Phase 5 before starting Phase 6.

**If all checks pass**: Proceed to Step 1.

---

#### Step 1: Write test for get_observation_dim()

**Action**: Add tests for observation dimension query

**Modify**: `tests/test_townlet/unit/test_substrate_base.py`

Add to end of file:

```python
def test_grid2d_get_observation_dim_full_obs_onehot():
    """Grid2D with one-hot encoding should return grid_size² dimensions."""
    substrate = Grid2DSubstrate(width=8, height=8, position_encoding="onehot")

    obs_dim = substrate.get_observation_dim(partial_observability=False, vision_range=0)

    assert obs_dim == 64  # 8×8 grid


def test_grid2d_get_observation_dim_full_obs_coords():
    """Grid2D with coordinate encoding should return 2 dimensions."""
    substrate = Grid2DSubstrate(width=8, height=8, position_encoding="coords")

    obs_dim = substrate.get_observation_dim(partial_observability=False, vision_range=0)

    assert obs_dim == 2  # Normalized (x, y)


def test_grid2d_get_observation_dim_pomdp():
    """Grid2D POMDP should return window² + 2 dimensions."""
    substrate = Grid2DSubstrate(width=8, height=8)

    obs_dim = substrate.get_observation_dim(partial_observability=True, vision_range=2)

    # 5×5 window + normalized (x, y) position
    assert obs_dim == 25 + 2


def test_aspatial_get_observation_dim():
    """Aspatial should return 0 dimensions (no position encoding)."""
    substrate = AspatialSubstrate()

    full_obs_dim = substrate.get_observation_dim(partial_observability=False, vision_range=0)
    pomdp_dim = substrate.get_observation_dim(partial_observability=True, vision_range=2)

    assert full_obs_dim == 0  # No position
    assert pomdp_dim == 0  # No local window either
```

**Run test**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_get_observation_dim_full_obs_onehot -v
```

**Expected**: FAIL (method not implemented)

---

#### Step 2: Add get_observation_dim() to SpatialSubstrate interface

**Action**: Extend abstract interface

**Modify**: `src/townlet/substrate/base.py`

Add after `encode_partial_observation()` method (around line 340):

```python
    @abstractmethod
    def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
        """Get observation dimension for this substrate.

        This dimension represents ONLY the position encoding portion of the
        observation. The full observation also includes meters, affordances,
        and temporal features, which are substrate-independent.

        Args:
            partial_observability: If True, calculate POMDP dimension
            vision_range: Vision radius for POMDP (ignored for full obs)

        Returns:
            Observation dimension (position encoding only)

        Examples:
            Grid2D (8×8) with one-hot:
                Full obs: 64 (grid cells)
                POMDP: 25 + 2 (5×5 window + position)

            Grid2D (8×8) with coords:
                Full obs: 2 (normalized x, y)
                POMDP: 25 + 2 (5×5 window + position)

            Grid3D (8×8×3):
                Full obs: 3 (normalized x, y, z)
                POMDP: 125 + 3 (5×5×5 cube + position)

            Aspatial:
                Full obs: 0 (no position)
                POMDP: 0 (no local window)

        Used for:
            - VectorizedHamletEnv: Calculate total observation_dim at initialization
            - Network creation: Determine input layer size
        """
        pass
```

**Expected**: Abstract method added to interface

---

#### Step 3: Implement get_observation_dim() in Grid2DSubstrate

**Action**: Add dimension query for 2D grids

**Modify**: `src/townlet/substrate/grid2d.py`

Add after `encode_partial_observation()` method (around line 680):

```python
    def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
        """Get observation dimension for Grid2D substrate.

        Args:
            partial_observability: If True, return POMDP dimension
            vision_range: Vision radius for POMDP (e.g., 2 for 5×5 window)

        Returns:
            Observation dimension (position encoding only)

        Position encoding strategy:
            - One-hot: Returns width × height (e.g., 8×8 = 64)
            - Coordinates: Returns 2 (normalized x, y)

        POMDP always uses local window + position:
            - Local window: (2*vision_range + 1)²
            - Position: 2 (normalized x, y)
            - Total: window² + 2

        Examples:
            Grid2D (8×8, one-hot):
                Full obs: 64
                POMDP (vision_range=2): 25 + 2 = 27

            Grid2D (8×8, coords):
                Full obs: 2
                POMDP (vision_range=2): 25 + 2 = 27
        """
        if partial_observability:
            # POMDP: local window + normalized position
            window_size = 2 * vision_range + 1
            local_window_dim = window_size * window_size
            position_dim = 2  # Always use normalized (x, y) for POMDP
            return local_window_dim + position_dim
        else:
            # Full observability: depends on encoding strategy
            if self.position_encoding == "onehot":
                return self.width * self.height
            elif self.position_encoding == "coords":
                return 2  # Normalized (x, y)
            else:
                raise ValueError(f"Unknown position encoding: {self.position_encoding}")
```

**Expected**: Returns correct dimension for 2D grids (one-hot or coords)

---

#### Step 4: Implement get_observation_dim() in AspatialSubstrate

**Action**: Add dimension query for aspatial

**Modify**: `src/townlet/substrate/aspatial.py`

Add after `encode_partial_observation()` method (around line 850):

```python
    def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
        """Get observation dimension for aspatial substrate.

        Aspatial universes have no position encoding, so dimension is always 0.

        Args:
            partial_observability: Ignored (aspatial has no local window)
            vision_range: Ignored

        Returns:
            0 (no position encoding)

        Note:
            Aspatial observations consist ONLY of meters + affordances + temporal features.
            There's no concept of "where" the agent is, only "what state" the agent is in.
        """
        return 0  # No position encoding
```

**Expected**: Returns 0 for aspatial substrate

---

#### Step 5: Run tests to verify get_observation_dim()

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_get_observation_dim_full_obs_onehot -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_get_observation_dim_full_obs_coords -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_get_observation_dim_pomdp -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_aspatial_get_observation_dim -v
```

**Expected**: All tests PASS

---

#### Step 6: Write test for encode_observation() with affordance overlay

**Action**: Add tests for full observability encoding

**Modify**: `tests/test_townlet/unit/test_substrate_base.py`

Add to end of file:

```python
def test_grid2d_encode_observation_onehot():
    """Grid2D one-hot encoding should mark agent and affordance positions."""
    substrate = Grid2DSubstrate(width=4, height=4, position_encoding="onehot")

    positions = torch.tensor([[1, 1], [2, 2]], dtype=torch.long)
    affordances = {
        "Bed": torch.tensor([0, 0], dtype=torch.long),
        "Hospital": torch.tensor([3, 3], dtype=torch.long),
    }

    encoding = substrate.encode_observation(positions, affordances)

    # Shape: [2 agents, 4×4=16 cells]
    assert encoding.shape == (2, 16)

    # Agent 0 at (1,1): flat index = 1*4 + 1 = 5
    # Should have value > 1.0 (agent + possibly affordance)
    assert encoding[0, 5] >= 1.0

    # Bed at (0,0): flat index = 0*4 + 0 = 0
    # Should be marked for all agents (value = 1.0)
    assert encoding[0, 0] == 1.0
    assert encoding[1, 0] == 1.0

    # Hospital at (3,3): flat index = 3*4 + 3 = 15
    assert encoding[0, 15] == 1.0
    assert encoding[1, 15] == 1.0


def test_grid2d_encode_observation_coords():
    """Grid2D coordinate encoding should return normalized positions."""
    substrate = Grid2DSubstrate(width=8, height=8, position_encoding="coords")

    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    affordances = {}  # Affordances not encoded in coord mode

    encoding = substrate.encode_observation(positions, affordances)

    # Shape: [2 agents, 2 dims]
    assert encoding.shape == (2, 2)

    # Agent 0 at (0, 0) → normalized to (0.0, 0.0)
    assert torch.allclose(encoding[0], torch.tensor([0.0, 0.0]))

    # Agent 1 at (7, 7) → normalized to (1.0, 1.0)
    assert torch.allclose(encoding[1], torch.tensor([1.0, 1.0]))


def test_aspatial_encode_observation():
    """Aspatial should return empty tensor (no position encoding)."""
    substrate = AspatialSubstrate()

    positions = torch.zeros((3, 0))  # 3 agents, 0-dimensional positions
    affordances = {}

    encoding = substrate.encode_observation(positions, affordances)

    assert encoding.shape == (3, 0)  # No position encoding
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_encode_observation_onehot -v
```

**Expected**: FAIL (method not fully implemented or needs affordance support)

---

#### Step 7: Update encode_observation() in Grid2DSubstrate

**Action**: Add affordance overlay support

**Modify**: `src/townlet/substrate/grid2d.py`

Find `encode_observation()` method (around line 500) and replace with:

```python
    def encode_observation(
        self,
        agent_positions: torch.Tensor,
        affordance_positions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode agent positions (and optionally affordances) into observation.

        Args:
            agent_positions: [num_agents, 2] agent positions (x, y)
            affordance_positions: Optional dict of affordance_name -> [2] position tensor
                For one-hot: affordances overlaid on same grid
                For coordinates: affordances NOT encoded (handled separately in affordance encoding)

        Returns:
            [num_agents, obs_dim] encoded positions

            obs_dim depends on encoding:
                - One-hot: width × height (e.g., 8×8 = 64)
                - Coordinates: 2 (normalized x, y)

        One-Hot Strategy:
            1. Mark affordances on grid (value = 1.0) - broadcast to all agents
            2. Mark agent positions (add 1.0, so agent on affordance = 2.0)

        Coordinate Strategy:
            1. Normalize agent positions to [0, 1]
            2. Affordances NOT encoded in position (handled by affordance encoding)
        """
        num_agents = agent_positions.shape[0]
        device = agent_positions.device

        if self.position_encoding == "onehot":
            # One-hot grid: mark affordances first, then add agents
            grid_encoding = torch.zeros(
                num_agents, self.width * self.height, device=device, dtype=torch.float32
            )

            # Mark affordances (broadcast to all agents)
            if affordance_positions:
                for affordance_pos in affordance_positions.values():
                    x, y = affordance_pos[0].item(), affordance_pos[1].item()
                    flat_idx = y * self.width + x
                    grid_encoding[:, flat_idx] = 1.0

            # Mark agent positions (add 1.0, so agent on affordance = 2.0)
            agent_x = agent_positions[:, 0]
            agent_y = agent_positions[:, 1]
            flat_indices = agent_y * self.width + agent_x
            grid_encoding.scatter_add_(
                1,
                flat_indices.unsqueeze(1),
                torch.ones(num_agents, 1, device=device),
            )

            return grid_encoding

        elif self.position_encoding == "coords":
            # Coordinate encoding: just normalize agent positions
            # Affordances NOT encoded in position (handled by affordance encoding)
            normalized_x = agent_positions[:, 0].float() / (self.width - 1)
            normalized_y = agent_positions[:, 1].float() / (self.height - 1)
            return torch.stack([normalized_x, normalized_y], dim=1)

        else:
            raise ValueError(f"Unknown position encoding: {self.position_encoding}")
```

**Expected**: Encodes positions with affordance overlay (one-hot) or coordinates

---

#### Step 8: Update encode_observation() in AspatialSubstrate

**Action**: Ensure aspatial returns empty tensor

**Modify**: `src/townlet/substrate/aspatial.py`

Find `encode_observation()` method (around line 700) and verify/update:

```python
    def encode_observation(
        self,
        agent_positions: torch.Tensor,
        affordance_positions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return empty tensor (aspatial has no position encoding).

        In aspatial universes, there's no concept of "position" or "spatial layout."
        Agents exist in a pure state space defined by meters and affordances.

        Args:
            agent_positions: [num_agents, 0] empty tensor
            affordance_positions: Ignored (no spatial positions)

        Returns:
            [num_agents, 0] empty tensor

        Note:
            Aspatial observations consist ONLY of:
            - Meters (energy, health, etc.)
            - Affordance encoding (which affordance is being interacted with)
            - Temporal features (time, interaction progress)

            There is NO position component.
        """
        num_agents = agent_positions.shape[0]
        device = agent_positions.device
        return torch.zeros((num_agents, 0), device=device, dtype=torch.float32)
```

**Expected**: Returns empty tensor for aspatial substrate

---

#### Step 9: Run tests to verify encode_observation()

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_encode_observation_onehot -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_encode_observation_coords -v
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_aspatial_encode_observation -v
```

**Expected**: All tests PASS

---

#### Step 10: Commit

**Command**:
```bash
git add src/townlet/substrate/base.py src/townlet/substrate/grid2d.py src/townlet/substrate/aspatial.py tests/test_townlet/unit/test_substrate_base.py
git commit -m "feat: add observation encoding methods to substrate

Extended substrate interface with two new methods for Phase 5:

1. get_observation_dim(partial_observability, vision_range) -> int
   - Returns position encoding dimension (excludes meters/affordances/temporal)
   - Grid2D one-hot: width × height
   - Grid2D coords: 2
   - Grid3D: 3 (future)
   - Aspatial: 0

2. encode_observation(agent_positions, affordance_positions) -> Tensor
   - Encodes positions into observation space
   - Grid2D one-hot: marks agents + affordances on grid
   - Grid2D coords: normalized (x, y) positions
   - Aspatial: empty tensor

Updated encode_observation() to handle affordance overlay for one-hot mode.

Implemented in Grid2DSubstrate and AspatialSubstrate.
All tests pass.

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with new observation encoding methods

---

### Task 6.2: Update VectorizedHamletEnv obs_dim Calculation

**Purpose**: Replace hardcoded obs_dim formulas with `substrate.get_observation_dim()`

**Files**:
- `src/townlet/environment/vectorized_env.py`

**Estimated Time**: 2 hours

---

#### Step 1: Write test for substrate-based obs_dim calculation

**Action**: Add integration test

**Modify**: `tests/test_townlet/integration/test_observation_dimensions.py` (create new file)

```python
"""Test environment calculates observation dimensions using substrate."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_obs_dim_uses_substrate_grid2d_onehot():
    """Environment should calculate obs_dim using substrate for Grid2D one-hot."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # Grid2D (8×8) one-hot: 64 position + 8 meters + 15 affordances + 4 temporal = 91
    assert env.observation_dim == 91

    # Verify breakdown
    substrate_dim = env.substrate.get_observation_dim(
        partial_observability=False, vision_range=0
    )
    assert substrate_dim == 64  # One-hot grid


def test_obs_dim_uses_substrate_pomdp():
    """Environment should calculate obs_dim using substrate for POMDP."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L2_partial_observability"),
        num_agents=1,
        device="cpu",
    )

    # POMDP: 25 window + 2 position + 8 meters + 15 affordances + 4 temporal = 54
    assert env.observation_dim == 54

    # Verify breakdown
    substrate_dim = env.substrate.get_observation_dim(
        partial_observability=True, vision_range=2
    )
    assert substrate_dim == 27  # 5×5 window + (x, y)


def test_obs_dim_changes_with_encoding():
    """Observation dimension should change when substrate encoding changes."""
    # Simulate Grid2D with coordinate encoding
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate_onehot = Grid2DSubstrate(width=8, height=8, position_encoding="onehot")
    substrate_coords = Grid2DSubstrate(width=8, height=8, position_encoding="coords")

    # One-hot: 64 dims
    assert substrate_onehot.get_observation_dim(False, 0) == 64

    # Coords: 2 dims (62 dims smaller!)
    assert substrate_coords.get_observation_dim(False, 0) == 2
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_observation_dimensions.py::test_obs_dim_uses_substrate_grid2d_onehot -v
```

**Expected**: FAIL (still using hardcoded formulas)

---

#### Step 2: Locate current obs_dim calculation

**Action**: Identify hardcoded formulas

**Read**: `src/townlet/environment/vectorized_env.py` (Lines 129-142)

Current code:
```python
# Observation dimensions depend on observability mode
if partial_observability:
    # Level 2 POMDP: local window + position + meters + current affordance type
    window_size = 2 * vision_range + 1  # 5×5 for vision_range=2
    # Grid + position + meter_count meters + affordance type one-hot (N+1 for "none")
    self.observation_dim = window_size * window_size + 2 + meter_count + (self.num_affordance_types + 1)
else:
    # Level 1: full grid one-hot + meters + current affordance type
    # Grid one-hot + meter_count meters + affordance type (N+1 for "none")
    self.observation_dim = grid_size * grid_size + meter_count + (self.num_affordance_types + 1)

# Always add temporal features for forward compatibility (4 features)
# time_sin, time_cos, interaction_progress, lifetime_progress
self.observation_dim += 4
```

**Observation**: Hardcoded `grid_size * grid_size` and `window_size * window_size + 2`

---

#### Step 3: Replace with substrate.get_observation_dim()

**Action**: Use substrate method for position encoding dimension

**Modify**: `src/townlet/environment/vectorized_env.py`

Replace lines 129-142 with:

```python
# Observation dimensions: substrate position encoding + meters + affordances + temporal
# Query substrate for position encoding dimension
substrate_obs_dim = self.substrate.get_observation_dim(
partial_observability, vision_range
        )

        # Total observation = position + meters + affordances + temporal
        # - Position: substrate-specific (grid one-hot, coords, or empty)
        # - Meters: 8 values (energy, health, satiation, money, mood, social, fitness, hygiene)
        # - Affordances: (num_affordance_types + 1) one-hot encoding (includes "none")
        # - Temporal: 4 values (time_sin, time_cos, interaction_progress, lifetime_progress)
        self.observation_dim = (
            substrate_obs_dim
            + meter_count
            + (self.num_affordance_types + 1)
            + 4  # Temporal features
        )
```

**Expected**: obs_dim now calculated using substrate

---

#### Step 4: Add logging for obs_dim breakdown

**Action**: Help operators understand dimension breakdown

**Modify**: `src/townlet/environment/vectorized_env.py`

Add after obs_dim calculation (around line 150):

```python
        # Log observation dimension breakdown for debugging
        logger.info(
            f"Observation dimension breakdown: "
            f"substrate={substrate_obs_dim}, "
            f"meters={meter_count}, "
            f"affordances={self.num_affordance_types + 1}, "
            f"temporal=4, "
            f"total={self.observation_dim}"
        )

        # Warn if position encoding differs from expected
        if isinstance(self.substrate, Grid2DSubstrate):
            if self.substrate.position_encoding == "coords" and substrate_obs_dim > 3:
                logger.warning(
                    f"Grid2D coordinate encoding should have ≤3 dims, got {substrate_obs_dim}. "
                    "Check substrate configuration."
                )
```

**Expected**: Clear logging of dimension breakdown

---

#### Step 5: Run tests to verify obs_dim calculation

**Command**:
```bash
uv run pytest tests/test_townlet/integration/test_observation_dimensions.py::test_obs_dim_uses_substrate_grid2d_onehot -v
uv run pytest tests/test_townlet/integration/test_observation_dimensions.py::test_obs_dim_uses_substrate_pomdp -v
```

**Expected**: Both tests PASS

---

#### Step 6: Run full test suite to check for regressions

**Command**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py -v
uv run pytest tests/test_townlet/integration/test_episode_execution.py -v
```

**Expected**: All observation tests PASS

---

#### Step 7: Commit

**Command**:
```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/integration/test_observation_dimensions.py
git commit -m "refactor: use substrate.get_observation_dim() in VectorizedHamletEnv

Replaced hardcoded obs_dim formulas with substrate method:

Before:
  obs_dim = grid_size² + meter_count + affordances + temporal

After:
  substrate_obs_dim = substrate.get_observation_dim(partial_obs, vision_range)
  obs_dim = substrate_obs_dim + meter_count + affordances + temporal

Benefits:
- Supports different encodings (one-hot, coordinates)
- Works with Grid2D, Grid3D (future), Aspatial
- Auto-adapts to substrate configuration
- Clear dimension breakdown in logs

All tests pass.

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with substrate-based obs_dim

---

### Task 6.3: Refactor ObservationBuilder Constructor

**Purpose**: Replace `grid_size` parameter with `substrate` reference

**Files**:
- `src/townlet/environment/observation_builder.py`
- `src/townlet/environment/vectorized_env.py` (update call site)

**Estimated Time**: 1 hour

---

#### Step 1: Locate ObservationBuilder constructor

**Action**: Identify current signature

**Read**: `src/townlet/environment/observation_builder.py` (Lines 20-50)

Current signature:
```python
def __init__(
    self,
    num_agents: int,
    grid_size: int,  # ← REMOVE
    device: torch.device,
    partial_observability: bool,
    vision_range: int,
    enable_temporal_mechanics: bool,
    num_affordance_types: int,
    affordance_names: list[str],
):
    self.grid_size = grid_size
```

**Observation**: `grid_size` is scalar, assumes square 2D grid

---

#### Step 2: Update ObservationBuilder constructor signature

**Action**: Replace grid_size with substrate

**Modify**: `src/townlet/environment/observation_builder.py`

Replace constructor (lines 20-50) with:

```python
    def __init__(
        self,
        num_agents: int,
        substrate: SpatialSubstrate,  # ← CHANGED: substrate instead of grid_size
        device: torch.device,
        partial_observability: bool,
        vision_range: int,
        enable_temporal_mechanics: bool,
        num_affordance_types: int,
        affordance_names: list[str],
    ):
        """Initialize observation builder with substrate abstraction.

        Args:
            num_agents: Number of agents in environment
            substrate: Spatial substrate (Grid2D, Grid3D, Aspatial, etc.)
            device: PyTorch device (cuda or cpu)
            partial_observability: If True, use local window (POMDP)
            vision_range: Vision radius for POMDP (e.g., 2 for 5×5 window)
            enable_temporal_mechanics: If True, include time-based features
            num_affordance_types: Number of affordance types in environment
            affordance_names: List of affordance names (for debugging)
        """
        self.num_agents = num_agents
        self.substrate = substrate  # ← NEW: store substrate reference
        self.position_dim = substrate.position_dim  # ← NEW: for validation
        self.device = device
        self.partial_observability = partial_observability
        self.vision_range = vision_range
        self.enable_temporal_mechanics = enable_temporal_mechanics
        self.num_affordance_types = num_affordance_types
        self.affordance_names = affordance_names

        # Remove grid_size attribute (replaced by substrate)
```

**Expected**: Constructor now accepts substrate instead of grid_size

---

#### Step 3: Add import for SpatialSubstrate

**Action**: Import substrate base class

**Modify**: `src/townlet/environment/observation_builder.py`

Add to imports at top of file (around line 5):

```python
from townlet.substrate.base import SpatialSubstrate
```

**Expected**: Import added for type hint

---

#### Step 4: Update call site in VectorizedHamletEnv

**Action**: Pass substrate instead of grid_size

**Modify**: `src/townlet/environment/vectorized_env.py`

Find ObservationBuilder instantiation (around line 200) and update:

Before:
```python
self.observation_builder = ObservationBuilder(
    num_agents=num_agents,
    grid_size=grid_size,  # ← REMOVE
    device=device,
    partial_observability=partial_observability,
    vision_range=vision_range,
    enable_temporal_mechanics=enable_temporal_mechanics,
    num_affordance_types=self.num_affordance_types,
    affordance_names=self.affordance_names,
)
```

After:
```python
self.observation_builder = ObservationBuilder(
    num_agents=num_agents,
    substrate=self.substrate,  # ← CHANGED: pass substrate
    device=device,
    partial_observability=partial_observability,
    vision_range=vision_range,
    enable_temporal_mechanics=enable_temporal_mechanics,
    num_affordance_types=self.num_affordance_types,
    affordance_names=self.affordance_names,
)
```

**Expected**: Call site updated to pass substrate

---

#### Step 5: Update test fixtures

**Action**: Update all tests that create ObservationBuilder

**Modify**: `tests/test_townlet/unit/environment/test_observations.py`

Find fixture (around line 15) and update:

Before:
```python
@pytest.fixture
def observation_builder():
    return ObservationBuilder(
        num_agents=2,
        grid_size=8,  # ← REMOVE
        device=torch.device("cpu"),
        ...
    )
```

After:
```python
@pytest.fixture
def observation_builder():
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8)  # ← ADD

    return ObservationBuilder(
        num_agents=2,
        substrate=substrate,  # ← CHANGED
        device=torch.device("cpu"),
        ...
    )
```

**Expected**: Test fixtures updated to pass substrate

---

#### Step 6: Run tests to verify signature change

**Command**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py -v
```

**Expected**: All tests PASS

---

#### Step 7: Commit

**Command**:
```bash
git add src/townlet/environment/observation_builder.py src/townlet/environment/vectorized_env.py tests/test_townlet/unit/environment/test_observations.py
git commit -m "refactor: replace grid_size with substrate in ObservationBuilder

Updated ObservationBuilder constructor to accept substrate instead of scalar grid_size:

Before:
  ObservationBuilder(num_agents, grid_size, ...)

After:
  ObservationBuilder(num_agents, substrate, ...)

Benefits:
- Works with any substrate (Grid2D, Grid3D, Aspatial)
- No hardcoded assumptions about grid dimensions
- Position dimension available via substrate.position_dim

Updated:
- ObservationBuilder.__init__() signature
- VectorizedHamletEnv call site
- Test fixtures

All tests pass.

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with signature change

---

### Task 6.4: Refactor Full Observability Encoding

**Purpose**: Replace hardcoded grid encoding with `substrate.encode_observation()`

**Files**:
- `src/townlet/environment/observation_builder.py`

**Estimated Time**: 2 hours

---

#### Step 1: Locate current full observability encoding

**Action**: Identify hardcoded logic

**Read**: `src/townlet/environment/observation_builder.py` (Lines 104-146)

Current implementation:
```python
def _build_full_observations(self, positions, meters, affordances):
    # Grid encoding: mark BOTH agent position AND affordance positions
    grid_encoding = torch.zeros(self.num_agents, self.grid_size * self.grid_size, device=self.device)

    # Mark affordance positions (value = 1.0) for all agents
    for affordance_pos in affordances.values():
        affordance_flat_idx = affordance_pos[1] * self.grid_size + affordance_pos[0]
        grid_encoding[:, affordance_flat_idx] = 1.0

    # Mark agent position (add 1.0, so if on affordance it becomes 2.0)
    flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
    grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), torch.ones(self.num_agents, 1, device=self.device))

    # Get affordance encoding
    affordance_encoding = self._build_affordance_encoding(positions, affordances)

    # Concatenate: grid + meters + affordance
    observations = torch.cat([grid_encoding, meters, affordance_encoding], dim=1)

    return observations
```

**Observation**: Hardcoded one-hot grid encoding with `grid_size`

---

#### Step 2: Write test for substrate-based full observation encoding

**Action**: Add test for new implementation

**Modify**: `tests/test_townlet/unit/environment/test_observations.py`

Add new test:

```python
def test_full_observation_uses_substrate_encoding():
    """Full observations should use substrate.encode_observation()."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=4, height=4, position_encoding="onehot")
    builder = ObservationBuilder(
        num_agents=2,
        substrate=substrate,
        device=torch.device("cpu"),
        partial_observability=False,
        vision_range=0,
        enable_temporal_mechanics=False,
        num_affordance_types=2,
        affordance_names=["Bed", "Hospital"],
    )

    positions = torch.tensor([[1, 1], [2, 2]], dtype=torch.long)
    meters = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]] * 2)
    affordances = {
        "Bed": torch.tensor([0, 0], dtype=torch.long),
        "Hospital": torch.tensor([3, 3], dtype=torch.long),
    }

    observations = builder.build_observations(positions, meters, affordances)

    # Observation should contain substrate encoding
    # 4×4 grid=16 + 8 meters + 3 affordances + 4 temporal = 31 dims
    assert observations.shape == (2, 31)


def test_full_observation_coordinate_encoding():
    """Full observations should work with coordinate encoding."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8, position_encoding="coords")
    builder = ObservationBuilder(
        num_agents=2,
        substrate=substrate,
        device=torch.device("cpu"),
        partial_observability=False,
        vision_range=0,
        enable_temporal_mechanics=False,
        num_affordance_types=2,
        affordance_names=["Bed", "Hospital"],
    )

    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    meters = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]] * 2)
    affordances = {}

    observations = builder.build_observations(positions, meters, affordances)

    # Observation should contain coordinate encoding
    # 2 coords + 8 meters + 3 affordances + 4 temporal = 17 dims
    assert observations.shape == (2, 17)

    # First agent at (0,0) should have position (0.0, 0.0)
    # Last agent at (7,7) should have position (1.0, 1.0)
    assert torch.allclose(observations[0, :2], torch.tensor([0.0, 0.0]))
    assert torch.allclose(observations[1, :2], torch.tensor([1.0, 1.0]))
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py::test_full_observation_uses_substrate_encoding -v
```

**Expected**: FAIL (still using hardcoded grid encoding)

---

#### Step 3: Refactor _build_full_observations()

**Action**: Replace grid encoding with substrate method

**Modify**: `src/townlet/environment/observation_builder.py`

Replace `_build_full_observations()` method (lines 104-146) with:

```python
    def _build_full_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build full observability observations using substrate encoding.

        Args:
            positions: [num_agents, position_dim] agent positions
            meters: [num_agents, 8] meter values
            affordances: {name: [position_dim]} affordance positions

        Returns:
            [num_agents, obs_dim] observations

        Observation structure:
            [position_encoding | meters | affordance_encoding | temporal_features]

        Position encoding varies by substrate:
            - Grid2D one-hot: 64 dims (8×8 grid with agents + affordances marked)
            - Grid2D coords: 2 dims (normalized x, y)
            - Grid3D coords: 3 dims (normalized x, y, z)
            - Aspatial: 0 dims (no position)
        """
        # Delegate position encoding to substrate
        position_encoding = self.substrate.encode_observation(
            agent_positions=positions,
            affordance_positions=affordances,
        )  # [num_agents, substrate_obs_dim]

        # Get affordance encoding (substrate-independent)
        # This encodes "which affordance is agent currently on" (one-hot)
        affordance_encoding = self._build_affordance_encoding(positions, affordances)

        # Concatenate: position + meters + affordance
        # If position_encoding is empty [num_agents, 0] for aspatial, cat still works!
        observations = torch.cat(
            [position_encoding, meters, affordance_encoding], dim=1
        )

        return observations
```

**Expected**: Full observations now use substrate encoding

---

#### Step 4: Remove hardcoded grid_size references

**Action**: Search for remaining grid_size usage in observation_builder.py

**Command**:
```bash
cd /home/john/hamlet
grep -n "self.grid_size" src/townlet/environment/observation_builder.py
```

**Expected**: Should find references only in `_build_partial_observations()` (will fix in Task 6.5)

---

#### Step 5: Run tests to verify full observation encoding

**Command**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py::test_full_observation_uses_substrate_encoding -v
uv run pytest tests/test_townlet/unit/environment/test_observations.py::test_full_observation_coordinate_encoding -v
```

**Expected**: Both tests PASS

---

#### Step 6: Run full observation test suite

**Command**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py -k "full" -v
```

**Expected**: All full observability tests PASS

---

#### Step 7: Commit

**Command**:
```bash
git add src/townlet/environment/observation_builder.py tests/test_townlet/unit/environment/test_observations.py
git commit -m "refactor: use substrate.encode_observation() for full observability

Replaced hardcoded grid encoding with substrate method in _build_full_observations():

Before:
  grid_encoding = torch.zeros(num_agents, grid_size²)
  # Manual affordance overlay
  # Manual agent position marking

After:
  position_encoding = substrate.encode_observation(positions, affordances)

Benefits:
- Works with one-hot (Grid2D ≤8×8) and coordinate encoding (Grid2D >8×8, Grid3D)
- Substrate handles affordance overlay for one-hot mode
- Supports aspatial (no position encoding)
- No hardcoded grid_size references

All tests pass.

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with substrate-based full obs encoding

---

### Task 6.5: Refactor Partial Observability Encoding (POMDP)

**Purpose**: Replace manual window extraction with `substrate.encode_partial_observation()`

**Files**:
- `src/townlet/environment/observation_builder.py`

**Estimated Time**: 3 hours

---

#### Step 1: Locate current POMDP encoding

**Action**: Identify manual window extraction logic

**Read**: `src/townlet/environment/observation_builder.py` (Lines 148-209)

Current implementation:
```python
def _build_partial_observations(self, positions, meters, affordances):
    window_size = 2 * self.vision_range + 1
    local_grids = []

    for agent_idx in range(self.num_agents):
        agent_pos = positions[agent_idx]
        local_grid = torch.zeros(window_size * window_size, device=self.device)

        # Extract local window centered on agent
        for dy in range(-self.vision_range, self.vision_range + 1):
            for dx in range(-self.vision_range, self.vision_range + 1):
                world_x = agent_pos[0] + dx
                world_y = agent_pos[1] + dy

                # Check if position is within grid bounds
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    # Check if there's an affordance at this position
                    has_affordance = False
                    for affordance_pos in affordances.values():
                        if affordance_pos[0] == world_x and affordance_pos[1] == world_y:
                            has_affordance = True
                            break

                    # Encode in local grid
                    if has_affordance:
                        local_y = dy + self.vision_range
                        local_x = dx + self.vision_range
                        local_idx = local_y * window_size + local_x
                        local_grid[local_idx] = 1.0

        local_grids.append(local_grid)

    # Stack all local grids
    local_grids_batch = torch.stack(local_grids)

    # Normalize positions to [0, 1]
    normalized_positions = positions.float() / (self.grid_size - 1)

    # Get affordance encoding
    affordance_encoding = self._build_affordance_encoding(positions, affordances)

    # Concatenate: local_grid + position + meters + affordance
    observations = torch.cat([local_grids_batch, normalized_positions, meters, affordance_encoding], dim=1)

    return observations
```

**Observation**: Manual nested loops for window extraction, hardcoded grid_size boundary checks

---

#### Step 2: Write test for substrate-based POMDP encoding

**Action**: Add test for new implementation

**Modify**: `tests/test_townlet/unit/environment/test_observations.py`

Add new test:

```python
def test_partial_observation_uses_substrate_encoding():
    """Partial observations should use substrate.encode_partial_observation()."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8)
    builder = ObservationBuilder(
        num_agents=2,
        substrate=substrate,
        device=torch.device("cpu"),
        partial_observability=True,
        vision_range=2,
        enable_temporal_mechanics=False,
        num_affordance_types=2,
        affordance_names=["Bed", "Hospital"],
    )

    positions = torch.tensor([[4, 4], [1, 1]], dtype=torch.long)
    meters = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]] * 2)
    affordances = {
        "Bed": torch.tensor([3, 3], dtype=torch.long),
        "Hospital": torch.tensor([6, 6], dtype=torch.long),
    }

    observations = builder.build_observations(positions, meters, affordances)

    # Observation should contain local window + position
    # 5×5 window=25 + 2 position + 8 meters + 3 affordances + 4 temporal = 42 dims
    assert observations.shape == (2, 42)


def test_partial_observation_aspatial():
    """Aspatial substrate should have no local window in POMDP."""
    from townlet.substrate.aspatial import AspatialSubstrate

    substrate = AspatialSubstrate()
    builder = ObservationBuilder(
        num_agents=2,
        substrate=substrate,
        device=torch.device("cpu"),
        partial_observability=True,  # Ignored for aspatial
        vision_range=2,  # Ignored for aspatial
        enable_temporal_mechanics=False,
        num_affordance_types=2,
        affordance_names=["Bed", "Hospital"],
    )

    positions = torch.zeros((2, 0))  # Aspatial has 0-dimensional positions
    meters = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]] * 2)
    affordances = {}

    observations = builder.build_observations(positions, meters, affordances)

    # Observation should have no position encoding
    # 0 position + 8 meters + 3 affordances + 4 temporal = 15 dims
    assert observations.shape == (2, 15)
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py::test_partial_observation_uses_substrate_encoding -v
```

**Expected**: FAIL (still using manual window extraction)

---

#### Step 3: Refactor _build_partial_observations()

**Action**: Replace manual window extraction with substrate method

**Modify**: `src/townlet/environment/observation_builder.py`

Replace `_build_partial_observations()` method (lines 148-209) with:

```python
    def _build_partial_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build partial observability observations using substrate encoding.

        Args:
            positions: [num_agents, position_dim] agent positions
            meters: [num_agents, 8] meter values
            affordances: {name: [position_dim]} affordance positions

        Returns:
            [num_agents, obs_dim] observations

        Observation structure:
            [local_window | normalized_position | meters | affordance_encoding | temporal_features]

        Local window varies by substrate:
            - Grid2D: 5×5 grid around agent (25 dims)
            - Grid3D: 5×5×5 cube around agent (125 dims) - future
            - Aspatial: empty (0 dims)

        Position encoding always uses normalized coordinates:
            - Grid2D: 2 dims (x, y)
            - Grid3D: 3 dims (x, y, z) - future
            - Aspatial: 0 dims

        Note: For POMDP, substrate.encode_partial_observation() returns BOTH
        local window AND normalized position as separate tensors, which we
        concatenate here.
        """
        # Delegate local window extraction to substrate
        # This method was added in Phase 4 Task 4.1 but we now use it fully
        local_window_encoding = self.substrate.encode_partial_observation(
            positions=positions,
            affordances=affordances,
            vision_range=self.vision_range,
        )  # [num_agents, window_size² + position_dim]

        # Note: encode_partial_observation() returns combined tensor:
        # [local_window | normalized_position]
        # We don't need to normalize positions separately!

        # Get affordance encoding (substrate-independent)
        affordance_encoding = self._build_affordance_encoding(positions, affordances)

        # Concatenate: local_window+position (from substrate) + meters + affordance
        observations = torch.cat(
            [local_window_encoding, meters, affordance_encoding], dim=1
        )

        return observations
```

**Wait!** Looking at Phase 4 Task 4.1, `encode_partial_observation()` returns only the local grid, not position. We need to handle position separately. Let me correct:

```python
    def _build_partial_observations(
        self,
        positions: torch.Tensor,
        meters: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build partial observability observations using substrate encoding.

        Args:
            positions: [num_agents, position_dim] agent positions
            meters: [num_agents, 8] meter values
            affordances: {name: [position_dim]} affordance positions

        Returns:
            [num_agents, obs_dim] observations

        Observation structure:
            [local_window | normalized_position | meters | affordance_encoding | temporal_features]

        Local window encoding varies by substrate:
            - Grid2D: 5×5 grid around agent (25 dims)
            - Grid3D: 5×5×5 cube around agent (125 dims) - future
            - Aspatial: empty (0 dims)

        Position encoding (normalized coordinates):
            - Grid2D: 2 dims (x, y normalized to [0, 1])
            - Grid3D: 3 dims (x, y, z normalized to [0, 1]) - future
            - Aspatial: 0 dims
        """
        # Delegate local window extraction to substrate
        local_window_encoding = self.substrate.encode_partial_observation(
            positions=positions,
            affordances=affordances,
            vision_range=self.vision_range,
        )  # [num_agents, window_size²]

        # Normalize positions for POMDP
        # Substrate handles normalization based on its dimensions
        if self.substrate.position_dim > 0:
            # Grid2D: normalize to [0, 1] based on grid dimensions
            # This assumes Grid2DSubstrate, we should delegate to substrate
            # For now, use substrate's encode_observation with coords mode conceptually
            # Actually, we need normalized positions separately
            # Let's call substrate to get normalized positions

            # TODO: This is hacky - substrate should provide a separate method for position normalization
            # For Phase 5, we'll handle this inline with knowledge of substrate type
            from townlet.substrate.grid2d import Grid2DSubstrate
            from townlet.substrate.aspatial import AspatialSubstrate

            if isinstance(self.substrate, Grid2DSubstrate):
                # Normalize positions for Grid2D
                normalized_x = positions[:, 0].float() / (self.substrate.width - 1)
                normalized_y = positions[:, 1].float() / (self.substrate.height - 1)
                normalized_positions = torch.stack([normalized_x, normalized_y], dim=1)
            elif isinstance(self.substrate, AspatialSubstrate):
                # No position for aspatial
                normalized_positions = torch.zeros(
                    (positions.shape[0], 0), device=self.device
                )
            else:
                # Generic fallback: assume position_dim dimensions, normalize assuming 0 to substrate dimensions
                raise NotImplementedError(
                    f"Position normalization not implemented for substrate type: {type(self.substrate)}"
                )
        else:
            # Aspatial: no position
            normalized_positions = torch.zeros(
                (positions.shape[0], 0), device=self.device
            )

        # Get affordance encoding (substrate-independent)
        affordance_encoding = self._build_affordance_encoding(positions, affordances)

        # Concatenate: local_window + normalized_position + meters + affordance
        observations = torch.cat(
            [local_window_encoding, normalized_positions, meters, affordance_encoding],
            dim=1,
        )

        return observations
```

**Note**: This implementation is not ideal - we're doing isinstance checks. Better approach would be to have substrate provide normalized positions separately. But for Phase 5, this works and we can refactor later.

**Expected**: POMDP now uses substrate for local window extraction

---

#### Step 4: Run tests to verify POMDP encoding

**Command**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py::test_partial_observation_uses_substrate_encoding -v
uv run pytest tests/test_townlet/unit/environment/test_observations.py::test_partial_observation_aspatial -v
```

**Expected**: Both tests PASS

---

#### Step 5: Run full POMDP test suite

**Command**:
```bash
uv run pytest tests/test_townlet/unit/environment/test_observations.py -k "partial" -v
```

**Expected**: All partial observability tests PASS

---

#### Step 6: Verify no remaining grid_size references

**Command**:
```bash
grep -n "self.grid_size" src/townlet/environment/observation_builder.py
```

**Expected**: No matches (all grid_size references removed)

---

#### Step 7: Commit

**Command**:
```bash
git add src/townlet/environment/observation_builder.py tests/test_townlet/unit/environment/test_observations.py
git commit -m "refactor: use substrate.encode_partial_observation() for POMDP

Replaced manual window extraction with substrate method in _build_partial_observations():

Before:
  # Manual nested loops for 5×5 window extraction
  # Hardcoded grid_size boundary checks
  # Manual affordance checks per cell

After:
  local_window = substrate.encode_partial_observation(positions, affordances, vision_range)
  # Substrate handles window extraction and boundaries

Benefits:
- Works with Grid2D (5×5 window), Grid3D (5×5×5 cube future), Aspatial (no window)
- No hardcoded grid_size references
- Substrate handles boundary conditions
- Cleaner, more maintainable code

Note: Position normalization still uses isinstance checks - can be improved
in future refactor with dedicated substrate method for position normalization.

All tests pass.

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with substrate-based POMDP encoding

---

### Task 6.6: Update RecurrentSpatialQNetwork

**Purpose**: Add `position_dim` parameter to support 3D substrates (position_dim=3)

**Files**:
- `src/townlet/agent/networks.py`
- `src/townlet/population/vectorized.py` (update call site)

**Estimated Time**: 2 hours

---

#### Step 1: Locate RecurrentSpatialQNetwork architecture

**Action**: Identify hardcoded position_dim=2

**Read**: `src/townlet/agent/networks.py` (Lines 162-194)

Current forward method:
```python
def forward(self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
    # Split observation components with dynamic indices
    grid_size_flat = self.window_size * self.window_size
    idx = 0

    # Extract grid
    grid = obs[:, idx : idx + grid_size_flat]
    idx += grid_size_flat

    # Extract position (HARDCODED 2 dims)
    position = obs[:, idx : idx + 2]
    idx += 2

    # Extract meters
    meters = obs[:, idx : idx + self.num_meters]
    idx += self.num_meters

    # Extract affordance
    affordance = obs[:, idx : idx + self.num_affordance_dims]
    idx += self.num_affordance_dims
```

**Observation**: Hardcoded `position_dim = 2`

---

#### Step 2: Update RecurrentSpatialQNetwork constructor

**Action**: Add position_dim parameter

**Modify**: `src/townlet/agent/networks.py`

Find `__init__` method (around line 90) and update signature:

Before:
```python
def __init__(
    self,
    action_dim: int,
    window_size: int,
    num_meters: int,
    num_affordance_types: int,
    enable_temporal_features: bool,
    hidden_dim: int = 256,
    device: torch.device = torch.device("cpu"),
):
```

After:
```python
def __init__(
    self,
    action_dim: int,
    local_window_dim: int,  # ← CHANGED: replaces window_size (could be 25 for 2D, 125 for 3D)
    position_dim: int,  # ← NEW: number of position dimensions (2 for 2D, 3 for 3D, 0 for aspatial)
    num_meters: int,
    num_affordance_types: int,
    enable_temporal_features: bool,
    hidden_dim: int = 256,
    device: torch.device = torch.device("cpu"),
):
    """Initialize recurrent spatial Q-network for POMDP.

    Args:
        action_dim: Number of actions (5: UP, DOWN, LEFT, RIGHT, INTERACT)
        local_window_dim: Flattened local window size (e.g., 25 for 5×5, 125 for 5×5×5)
        position_dim: Number of position dimensions (2 for Grid2D, 3 for Grid3D, 0 for Aspatial)
        num_meters: Number of meter values (8: energy, health, etc.)
        num_affordance_types: Number of affordance types
        enable_temporal_features: If True, include temporal features (4 dims)
        hidden_dim: LSTM hidden dimension
        device: PyTorch device
    """
    super().__init__()
    self.action_dim = action_dim
    self.local_window_dim = local_window_dim  # Replaces window_size²
    self.position_dim = position_dim  # NEW
    self.num_meters = num_meters
    self.num_affordance_types = num_affordance_types
    self.enable_temporal_features = enable_temporal_features
    self.hidden_dim = hidden_dim
    self.device = device

    # Vision encoder: processes local window (2D or 3D)
    # For 2D: 5×5 = 25 dims → 128 features
    # For 3D: 5×5×5 = 125 dims → 128 features
    self.vision_encoder = nn.Sequential(
        nn.Linear(local_window_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )

    # Position encoder: processes normalized position coordinates
    # For 2D: 2 dims (x, y) → 32 features
    # For 3D: 3 dims (x, y, z) → 32 features
    # For Aspatial: 0 dims → skip this encoder
    if position_dim > 0:
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
    else:
        # Aspatial: no position encoding
        self.position_encoder = None

    # ... rest of encoders (meters, affordance, temporal)
```

**Expected**: Constructor now accepts dynamic position_dim

---

#### Step 3: Update forward() method to use position_dim

**Action**: Replace hardcoded 2 with self.position_dim

**Modify**: `src/townlet/agent/networks.py`

Find forward() method (around line 162) and update:

```python
def forward(
    self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
):
    """Forward pass through recurrent network.

    Args:
        obs: [batch_size, obs_dim] observations
        hidden: Optional LSTM hidden state (h, c)

    Returns:
        q_values: [batch_size, action_dim] Q-values for each action
        hidden: (h, c) updated LSTM hidden state
    """
    # Split observation components with dynamic indices
    idx = 0

    # Extract local window (dimension varies by substrate)
    local_window = obs[:, idx : idx + self.local_window_dim]
    idx += self.local_window_dim

    # Extract position (dimension varies by substrate)
    position = obs[:, idx : idx + self.position_dim]
    idx += self.position_dim

    # Extract meters (substrate-independent)
    meters = obs[:, idx : idx + self.num_meters]
    idx += self.num_meters

    # Extract affordance (substrate-independent)
    affordance = obs[:, idx : idx + self.num_affordance_dims]
    idx += self.num_affordance_dims

    # Extract temporal features (substrate-independent)
    if self.enable_temporal_features:
        temporal = obs[:, idx : idx + 4]
        idx += 4

    # Encode components
    vision_features = self.vision_encoder(local_window)  # [batch, 128]

    if self.position_dim > 0:
        position_features = self.position_encoder(position)  # [batch, 32]
    else:
        # Aspatial: no position features
        position_features = torch.zeros((obs.shape[0], 32), device=obs.device)

    meter_features = self.meter_encoder(meters)  # [batch, 32]
    affordance_features = self.affordance_encoder(affordance)  # [batch, 32]

    if self.enable_temporal_features:
        temporal_features = self.temporal_encoder(temporal)  # [batch, 32]
        combined = torch.cat(
            [
                vision_features,
                position_features,
                meter_features,
                affordance_features,
                temporal_features,
            ],
            dim=1,
        )  # [batch, 128+32+32+32+32=256]
    else:
        combined = torch.cat(
            [vision_features, position_features, meter_features, affordance_features],
            dim=1,
        )  # [batch, 128+32+32+32=224]

    # LSTM for temporal dependencies
    lstm_input = combined.unsqueeze(1)  # [batch, 1, features]
    lstm_out, hidden = self.lstm(lstm_input, hidden)  # [batch, 1, hidden_dim]
    lstm_features = lstm_out.squeeze(1)  # [batch, hidden_dim]

    # Q-value head
    q_values = self.q_head(lstm_features)  # [batch, action_dim]

    return q_values, hidden
```

**Expected**: forward() now uses dynamic position_dim

---

#### Step 4: Update call site in VectorizedPopulation

**Action**: Pass position_dim from substrate

**Modify**: `src/townlet/population/vectorized.py`

Find network creation (around line 147) and update:

Before:
```python
if network_type == "recurrent":
    window_size = 2 * vision_range + 1
    self.q_network = RecurrentSpatialQNetwork(
        action_dim=action_dim,
        window_size=window_size,  # ← REMOVE
        num_meters=8,
        num_affordance_types=env.num_affordance_types,
        enable_temporal_features=enable_temporal_features,
        hidden_dim=hidden_dim,
        device=device,
    )
```

After:
```python
if network_type == "recurrent":
    # Get dimensions from substrate
    local_window_dim = env.substrate.get_observation_dim(
        partial_observability=True, vision_range=vision_range
    )
    position_dim = env.substrate.position_dim

    # For POMDP, substrate.get_observation_dim() returns local_window + position
    # We need to split them for network architecture
    window_size = 2 * vision_range + 1
    if isinstance(env.substrate, Grid2DSubstrate):
        # Grid2D: local_window_dim = 25 + 2, so actual window is 25
        actual_window_dim = window_size * window_size
    elif isinstance(env.substrate, AspatialSubstrate):
        # Aspatial: no local window
        actual_window_dim = 0
    else:
        # Generic: assume local_window_dim includes position, subtract it
        actual_window_dim = local_window_dim - position_dim

    self.q_network = RecurrentSpatialQNetwork(
        action_dim=action_dim,
        local_window_dim=actual_window_dim,  # ← CHANGED
        position_dim=position_dim,  # ← NEW
        num_meters=8,
        num_affordance_types=env.num_affordance_types,
        enable_temporal_features=enable_temporal_features,
        hidden_dim=hidden_dim,
        device=device,
    )
```

**Note**: This is getting complicated. Better approach: have substrate provide `get_local_window_dim()` and `get_position_dim()` separately. But for Phase 5, this works.

**Expected**: Network creation uses substrate dimensions

---

#### Step 5: Write test for recurrent network with different substrates

**Action**: Add test for 2D and aspatial

**Create**: `tests/test_townlet/unit/agent/test_recurrent_network_substrates.py`

```python
"""Test RecurrentSpatialQNetwork with different substrates."""
import pytest
import torch
from townlet.agent.networks import RecurrentSpatialQNetwork
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


def test_recurrent_network_grid2d():
    """RecurrentSpatialQNetwork should work with Grid2D substrate."""
    substrate = Grid2DSubstrate(width=8, height=8)

    # Get dimensions for POMDP
    local_window_dim = 25  # 5×5 window
    position_dim = 2  # (x, y)

    network = RecurrentSpatialQNetwork(
        action_dim=5,
        local_window_dim=local_window_dim,
        position_dim=position_dim,
        num_meters=8,
        num_affordance_types=14,
        enable_temporal_features=True,
        hidden_dim=256,
    )

    # Create mock observation: window + position + meters + affordances + temporal
    # 25 + 2 + 8 + 15 + 4 = 54 dims
    obs = torch.randn(4, 54)

    q_values, hidden = network(obs)

    assert q_values.shape == (4, 5)  # [batch, actions]
    assert hidden[0].shape == (1, 4, 256)  # LSTM hidden state


def test_recurrent_network_aspatial():
    """RecurrentSpatialQNetwork should work with aspatial substrate."""
    substrate = AspatialSubstrate()

    # Aspatial: no local window, no position
    local_window_dim = 0
    position_dim = 0

    network = RecurrentSpatialQNetwork(
        action_dim=5,
        local_window_dim=local_window_dim,
        position_dim=position_dim,
        num_meters=8,
        num_affordance_types=14,
        enable_temporal_features=True,
        hidden_dim=256,
    )

    # Create mock observation: 0 window + 0 position + 8 meters + 15 affordances + 4 temporal = 27 dims
    obs = torch.randn(4, 27)

    q_values, hidden = network(obs)

    assert q_values.shape == (4, 5)  # [batch, actions]
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/agent/test_recurrent_network_substrates.py -v
```

**Expected**: Both tests PASS

---

#### Step 6: Commit

**Command**:
```bash
git add src/townlet/agent/networks.py src/townlet/population/vectorized.py tests/test_townlet/unit/agent/test_recurrent_network_substrates.py
git commit -m "refactor: add position_dim parameter to RecurrentSpatialQNetwork

Updated RecurrentSpatialQNetwork to support variable position dimensions:

Before:
  - Hardcoded position_dim = 2
  - window_size² parameter (assumes 2D)

After:
  - local_window_dim parameter (works for 2D/3D/aspatial)
  - position_dim parameter (2 for Grid2D, 3 for Grid3D, 0 for Aspatial)

Changes:
1. Constructor accepts local_window_dim + position_dim
2. Forward method uses self.position_dim for dynamic slicing
3. Position encoder skipped if position_dim = 0 (aspatial)
4. VectorizedPopulation passes dimensions from substrate

Benefits:
- Works with Grid2D (5×5 window, 2-dim position)
- Works with Grid3D future (5×5×5 cube, 3-dim position)
- Works with Aspatial (no window, no position)

All tests pass.

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with dynamic position_dim support

---

### Task 6.7: Update Test Suite

**Purpose**: Update all tests to use substrate-based observation encoding

**Files**:
- `tests/test_townlet/unit/environment/test_observations.py`
- `tests/test_townlet/integration/test_data_flows.py`
- `tests/test_townlet/integration/test_episode_execution.py`

**Estimated Time**: 4 hours

---

#### Step 1: Run full test suite to identify failures

**Command**:
```bash
uv run pytest tests/test_townlet/ -v --tb=short
```

**Expected**: Some tests may fail due to observation dimension changes

---

#### Step 2: Update observation unit tests

**Action**: Fix any hardcoded dimension assumptions

**Modify**: `tests/test_townlet/unit/environment/test_observations.py`

Review all tests and update:

1. Remove grid_size parameters
2. Pass substrate instead
3. Update dimension assertions if substrate encoding changed

Example fix:
```python
# Before
def test_observation_dimensions():
    builder = ObservationBuilder(num_agents=2, grid_size=8, ...)
    obs = builder.build_observations(...)
    assert obs.shape[1] == 91  # Hardcoded

# After
def test_observation_dimensions():
    substrate = Grid2DSubstrate(width=8, height=8, position_encoding="onehot")
    builder = ObservationBuilder(num_agents=2, substrate=substrate, ...)
    obs = builder.build_observations(...)

    # Calculate expected dimension dynamically
    substrate_dim = substrate.get_observation_dim(False, 0)  # 64
    expected_dim = substrate_dim + 8 + 15 + 4  # 91
    assert obs.shape[1] == expected_dim
```

**Expected**: All observation unit tests updated and passing

---

#### Step 3: Update data flow integration tests

**Action**: Fix integration tests that create environments

**Modify**: `tests/test_townlet/integration/test_data_flows.py`

Update any tests that:
1. Create VectorizedHamletEnv (should now use substrate from config)
2. Check observation dimensions (should use env.observation_dim)
3. Assume specific obs_dim values (should calculate dynamically)

**Expected**: Data flow tests pass

---

#### Step 4: Update episode execution tests

**Action**: Fix integration tests for episode execution

**Modify**: `tests/test_townlet/integration/test_episode_execution.py`

Similar updates to ensure tests work with substrate-based observations.

**Expected**: Episode execution tests pass

---

#### Step 5: Run full test suite again

**Command**:
```bash
uv run pytest tests/test_townlet/ -v
```

**Expected**: All tests PASS

---

#### Step 6: Add new integration test for substrate observation consistency

**Action**: Verify observations consistent across substrates

**Create**: `tests/test_townlet/integration/test_substrate_observation_consistency.py`

```python
"""Test observation encoding consistency across substrates."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_observation_dimension_matches_substrate():
    """Observation dimension should match substrate calculation."""
    for config in ["L1_full_observability", "L2_partial_observability"]:
        env = VectorizedHamletEnv(
            config_pack_path=Path(f"configs/{config}"),
            num_agents=1,
            device="cpu",
        )

        obs, _ = env.reset()

        # Observation shape should match calculated dimension
        assert obs.shape[1] == env.observation_dim

        # Verify breakdown
        substrate_dim = env.substrate.get_observation_dim(
            env.partial_observability, env.vision_range
        )
        meter_dim = 8
        affordance_dim = env.num_affordance_types + 1
        temporal_dim = 4

        expected_dim = substrate_dim + meter_dim + affordance_dim + temporal_dim
        assert env.observation_dim == expected_dim


def test_observations_valid_after_step():
    """Observations should remain valid after environment steps."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=3,
        device="cpu",
    )

    obs, _ = env.reset()
    assert obs.shape == (3, env.observation_dim)

    # Take random actions
    actions = torch.randint(0, 5, (3,))
    obs, rewards, dones, truncated, info = env.step(actions)

    # Observations should still be valid
    assert obs.shape == (3, env.observation_dim)
    assert not torch.isnan(obs).any()
    assert not torch.isinf(obs).any()
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_observation_consistency.py -v
```

**Expected**: All consistency tests PASS

---

#### Step 7: Commit

**Command**:
```bash
git add tests/
git commit -m "test: update test suite for substrate-based observations

Updated all tests to use substrate abstraction:

1. Observation unit tests:
   - Pass substrate instead of grid_size
   - Calculate expected dimensions dynamically
   - Test multiple substrate types

2. Integration tests:
   - Update data flow tests for substrate
   - Update episode execution tests
   - Add substrate observation consistency tests

3. New tests:
   - test_substrate_observation_consistency.py
   - Verifies obs_dim matches substrate calculation
   - Tests observations valid after steps

All tests pass.

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with all tests updated

---

### Task 6.8: Documentation

**Purpose**: Update CLAUDE.md and create research summary

**Files**:
- `CLAUDE.md`
- `docs/research/research-task-002a-phase5-summary.md`

**Estimated Time**: 1 hour

---

#### Step 1: Update CLAUDE.md observation section

**Action**: Document substrate-based observation encoding

**Modify**: `CLAUDE.md`

Find "State Representation" section (around line 200) and update:

Before:
```markdown
**Full Observability**:

- Grid encoding: grid_size × grid_size one-hot
- Meters: 8 normalized values
- Affordance at position: 15 one-hot

**Observation dimensions by level**:

- **L0_0_minimal**: 36 dims (3×3 grid=9 + ...)
- **L1_full_observability**: 91 dims (8×8 grid=64 + ...)
```

After:
```markdown
**State Representation**

**Position Encoding Strategy** (Substrate-Based):

The environment uses **substrate-based position encoding**, which varies by substrate type and configuration:

**Grid2D (Square Grid)**:
- **One-hot encoding** (default for ≤8×8 grids): Marks agent + affordance positions on grid
  - 8×8 grid: 64 dimensions
  - Agent on affordance: value = 2.0 (both marked)
- **Coordinate encoding** (auto-selected for >8×8 grids): Normalized (x, y) positions
  - Any grid size: 2 dimensions
  - Enables transfer learning across grid sizes

**Grid3D (Cubic Grid)** (Future):
- **Always coordinate encoding**: Normalized (x, y, z) positions
  - Any grid size: 3 dimensions
  - One-hot infeasible (512+ dimensions for 8×8×3)

**Aspatial**:
- **No position encoding**: 0 dimensions
  - Pure resource management without spatial positioning

**Full Observability**:

- Position encoding: substrate-specific (see above)
- Meters: 8 normalized values (energy, health, satiation, money, mood, social, fitness, hygiene)
- Affordance at position: 15 one-hot (14 affordances + "none")
- Temporal extras: 4 values (time_of_day, retirement_age, interaction_progress, interaction_ticks)

**Observation dimensions by level**:

- **L0_0_minimal** (3×3 grid, one-hot): 36 dims (9 grid + 8 meters + 15 affordances + 4 temporal)
- **L0_5_dual_resource** (7×7 grid, one-hot): 76 dims (49 grid + 8 meters + 15 affordances + 4 temporal)
- **L1_full_observability** (8×8 grid, one-hot): 91 dims (64 grid + 8 meters + 15 affordances + 4 temporal)
- **L1_full_observability** (8×8 grid, coords): 29 dims (2 coords + 8 meters + 15 affordances + 4 temporal)

**Partial Observability (Level 2 POMDP)**:

- Local grid: 5×5 window (25 dims) - agent only sees local region
- Position: normalized (x, y) (2 dims) - where am I on the grid?
- Meters: 8 normalized values (8 dims)
- Affordance at position: 15 one-hot (15 dims)
- Temporal extras: 4 values (4 dims)
- **Total**: 54 dimensions

**Key insight**: Position encoding is **substrate-specific**. Coordinate encoding enables:
- 3D substrates (3 dims instead of 512+)
- Larger grids (2 dims instead of 256+ for 16×16)
- Transfer learning (same network works on different grid sizes)
- Aspatial universes (0 dims, pure resource management)
```

**Expected**: State representation section updated with substrate details

---

#### Step 2: Add breaking changes notice to CLAUDE.md

**Action**: Warn about observation dimension changes

**Modify**: `CLAUDE.md`

Add to "Common Pitfalls" section (around line 800):

```markdown
6. **Observation dimension changes with substrate encoding**:
   - Grid2D one-hot (≤8×8): Uses one-hot encoding (backward compatible)
   - Grid2D coords (>8×8 or explicit config): Uses coordinate encoding (BREAKING - changes obs_dim)
   - Grid3D: Always uses coordinate encoding (new feature)
   - Aspatial: No position encoding (new feature)
   - **Checkpoints are NOT compatible** when encoding changes
   - Delete old checkpoints when changing substrate type or grid size
```

**Expected**: Warning added to common pitfalls

---

#### Step 3: Create Phase 5 summary document

**Action**: Summarize implementation

**Create**: `docs/research/research-task-002a-phase5-summary.md`

```markdown
# TASK-002A Phase 5: Observation Builder Integration - Summary

**Date**: 2025-11-05
**Status**: Complete
**Phase**: Phase 5 of Configurable Spatial Substrates
**Actual Effort**: [To be filled after completion]

---

## Executive Summary

Phase 5 successfully integrated substrate-based observation encoding into VectorizedHamletEnv and ObservationBuilder, replacing all hardcoded grid encoding logic. This enables support for Grid2D (one-hot or coordinate encoding), Grid3D (coordinate encoding), and Aspatial substrates.

**Key Achievements**:

1. ✅ Added 3 substrate methods: `get_observation_dim()`, `encode_observation()`, enhanced `encode_partial_observation()`
2. ✅ Refactored VectorizedHamletEnv to use `substrate.get_observation_dim()` for obs_dim calculation
3. ✅ Replaced ObservationBuilder grid_size parameter with substrate reference
4. ✅ Refactored full observability encoding to use `substrate.encode_observation()`
5. ✅ Refactored POMDP encoding to use `substrate.encode_partial_observation()`
6. ✅ Updated RecurrentSpatialQNetwork to support variable position_dim (2D/3D/aspatial)
7. ✅ Updated entire test suite for substrate-based observations
8. ✅ Documentation updated with breaking changes notice

**Breaking Changes**:

- Observation dimensions change when switching from one-hot to coordinate encoding
- Existing checkpoints incompatible with different encoding strategies
- RecurrentSpatialQNetwork signature changed (requires position_dim parameter)

**Impact**:

- Enables 3D substrates (512+ dims → 3 dims with coordinate encoding)
- Enables larger 2D grids (256+ dims → 2 dims for 16×16)
- Enables aspatial substrates (no position encoding)
- Enables transfer learning (same network works on different grid sizes)

---

## Implementation Details

### Task 6.1: Substrate Observation Methods (4 hours)

Added three methods to substrate interface:

1. **get_observation_dim(partial_observability, vision_range) -> int**
   - Returns position encoding dimension only (excludes meters/affordances/temporal)
   - Grid2D one-hot: width × height
   - Grid2D coords: 2
   - Aspatial: 0

2. **encode_observation(agent_positions, affordance_positions) -> Tensor**
   - Grid2D one-hot: marks agents + affordances on same grid
   - Grid2D coords: normalized (x, y) positions
   - Aspatial: empty tensor

3. **encode_partial_observation(positions, affordances, vision_range) -> Tensor**
   - Enhanced from Phase 4 to handle affordance overlay
   - Grid2D: extracts 5×5 window around agent
   - Aspatial: returns empty tensor

Implemented in Grid2DSubstrate and AspatialSubstrate.

### Task 6.2: VectorizedHamletEnv obs_dim (2 hours)

Replaced hardcoded formulas:

Before:
```python
obs_dim = grid_size² + meter_count + affordances + temporal
```

After:
```python
substrate_obs_dim = substrate.get_observation_dim(partial_obs, vision_range)
obs_dim = substrate_obs_dim + meter_count + affordances + temporal
```

Added logging for dimension breakdown.

### Task 6.3: ObservationBuilder Constructor (1 hour)

Changed signature:

Before: `ObservationBuilder(num_agents, grid_size, ...)`
After: `ObservationBuilder(num_agents, substrate, ...)`

Updated call sites in VectorizedHamletEnv and test fixtures.

### Task 6.4: Full Observability Encoding (2 hours)

Replaced manual grid encoding:

Before: 15 lines of hardcoded one-hot logic
After: 1 line `substrate.encode_observation(positions, affordances)`

Works with one-hot, coordinates, or aspatial.

### Task 6.5: POMDP Encoding (3 hours)

Replaced manual window extraction:

Before: 40 lines of nested loops with boundary checks
After: 1 line `substrate.encode_partial_observation(positions, affordances, vision_range)`

Position normalization still uses isinstance checks (can be improved in future).

### Task 6.6: RecurrentSpatialQNetwork (2 hours)

Added position_dim parameter:

Before: Hardcoded `position_dim = 2`
After: Dynamic `position_dim` from substrate (0, 2, or 3)

Updated forward() to use `self.position_dim` for slicing.

### Task 6.7: Test Suite (4 hours)

Updated all tests:
- Unit tests: Pass substrate instead of grid_size
- Integration tests: Calculate expected dimensions dynamically
- New tests: Substrate observation consistency

All tests pass.

### Task 6.8: Documentation (1 hour)

Updated CLAUDE.md:
- State representation section with substrate encoding strategies
- Common pitfalls with checkpoint compatibility warning

---

## Lessons Learned

**What Worked Well**:

1. Substrate abstraction cleanly separated position encoding from observation building
2. Coordinate encoding proven viable by L2 POMDP success
3. TDD approach caught dimension mismatches early
4. Breaking changes authorization simplified implementation

**What Could Be Improved**:

1. Position normalization in POMDP still uses isinstance checks
   - Future: Add `substrate.normalize_positions()` method
2. Network dimension calculation has some complexity
   - Future: Have substrate provide `get_local_window_dim()` separately from `get_position_dim()`
3. Documentation could include migration guide for existing configs

**Technical Debt**:

1. Position normalization hardcoded for Grid2D/Aspatial in `_build_partial_observations()`
2. VectorizedPopulation network creation has isinstance checks for substrate type
3. RecurrentSpatialQNetwork still has temporal feature handling complexity

---

## Next Steps

**Phase 6 (Future)**: Observation Builder Performance Optimization
- Vectorize POMDP window extraction (currently loops over agents)
- Benchmark performance with num_agents=100
- Optimize affordance checks in local window

**Phase 7 (Future)**: Position Normalization Abstraction
- Add `substrate.normalize_positions()` method
- Remove isinstance checks from ObservationBuilder
- Cleaner separation of concerns

**Phase 8 (Future)**: 3D Substrate Implementation
- Implement Grid3DSubstrate with 5×5×5 POMDP window
- Test RecurrentSpatialQNetwork with 3-dimensional positions
- Validate obs_dim: 125 (cube) + 3 (position) + 8 + 15 + 4 = 155 dims

---

## Validation

**Success Criteria** (from Phase 5 plan):

- [x] get_observation_dim() implemented for Grid2D, Aspatial
- [x] encode_observation() handles one-hot and coordinate encoding
- [x] encode_partial_observation() extracts local windows
- [x] VectorizedHamletEnv uses substrate.get_observation_dim()
- [x] ObservationBuilder uses substrate.encode_observation()
- [x] RecurrentSpatialQNetwork supports variable position_dim
- [x] All tests pass
- [x] Documentation updated

**Regression Testing**:

- [x] L0_0_minimal still works (3×3 grid, one-hot)
- [x] L0_5_dual_resource still works (7×7 grid, one-hot)
- [x] L1_full_observability still works (8×8 grid, one-hot)
- [x] L2_partial_observability still works (8×8 grid, POMDP)
- [x] L3_temporal_mechanics still works (temporal features)

**New Functionality**:

- [ ] Grid2D coordinate encoding tested (manual test with config)
- [ ] Aspatial substrate tested (requires new config pack)
- [ ] Transfer learning tested (train on 8×8, test on 16×16)

---

## Conclusion

Phase 5 successfully abstracted observation encoding into substrates, removing all hardcoded grid assumptions from ObservationBuilder and VectorizedHamletEnv. This unblocks 3D substrates, aspatial substrates, and enables transfer learning across grid sizes.

**Estimated vs Actual Effort**: 16-20 hours estimated, [actual to be filled]

**Phase 5 Status**: ✅ Complete

**Ready for Phase 6**: Yes (performance optimization, optional)
```

**Expected**: Summary document created

---

#### Step 4: Commit

**Command**:
```bash
git add CLAUDE.md docs/research/research-task-002a-phase5-summary.md
git commit -m "docs: update documentation for Phase 5 observation builder integration

Updated documentation for substrate-based observation encoding:

1. CLAUDE.md:
   - Updated state representation section with substrate encoding strategies
   - Documented one-hot vs coordinate encoding trade-offs
   - Added breaking changes warning for checkpoint compatibility
   - Updated observation dimensions for all levels

2. research-task-002a-phase5-summary.md:
   - Comprehensive Phase 5 implementation summary
   - Task breakdown with effort tracking
   - Lessons learned and technical debt notes
   - Validation checklist

Breaking Changes Notice:
- Observation dimensions change with encoding strategy
- Checkpoints incompatible across encoding types
- RecurrentSpatialQNetwork signature changed

Part of TASK-002A Phase 5 (Observation Builder Integration)."
```

**Expected**: Clean commit with documentation updates

---

## Phase 5 Completion Checklist

### Core Implementation

- [ ] Task 6.1: Add substrate observation methods (4 hours)
  - [ ] get_observation_dim() in Grid2DSubstrate, AspatialSubstrate
  - [ ] encode_observation() with affordance overlay
  - [ ] Tests pass

- [ ] Task 6.2: Update VectorizedHamletEnv obs_dim (2 hours)
  - [ ] Replace hardcoded formulas with substrate.get_observation_dim()
  - [ ] Add dimension breakdown logging
  - [ ] Tests pass

- [ ] Task 6.3: Refactor ObservationBuilder constructor (1 hour)
  - [ ] Replace grid_size with substrate parameter
  - [ ] Update call sites and test fixtures
  - [ ] Tests pass

- [ ] Task 6.4: Refactor full observability encoding (2 hours)
  - [ ] Use substrate.encode_observation()
  - [ ] Remove hardcoded grid encoding
  - [ ] Tests pass

- [ ] Task 6.5: Refactor POMDP encoding (3 hours)
  - [ ] Use substrate.encode_partial_observation()
  - [ ] Remove manual window extraction
  - [ ] Tests pass

- [ ] Task 6.6: Update RecurrentSpatialQNetwork (2 hours)
  - [ ] Add position_dim parameter
  - [ ] Update forward() for dynamic slicing
  - [ ] Update call sites
  - [ ] Tests pass

- [ ] Task 6.7: Update test suite (4 hours)
  - [ ] Update observation unit tests
  - [ ] Update integration tests
  - [ ] Add substrate consistency tests
  - [ ] All tests pass

- [ ] Task 6.8: Documentation (1 hour)
  - [ ] Update CLAUDE.md state representation section
  - [ ] Add breaking changes warning
  - [ ] Create Phase 5 summary document

### Validation

- [ ] No hardcoded grid_size references in observation_builder.py
- [ ] No hardcoded position_dim=2 in networks.py
- [ ] All existing curriculum levels work (L0, L0.5, L1, L2, L3)
- [ ] Observation dimensions match substrate.get_observation_dim()
- [ ] Full test suite passes

### Regression Testing

Run each curriculum level:

```bash
# L0_0_minimal (3×3 one-hot)
uv run scripts/run_demo.py --config configs/L0_0_minimal

# L0_5_dual_resource (7×7 one-hot)
uv run scripts/run_demo.py --config configs/L0_5_dual_resource

# L1_full_observability (8×8 one-hot)
uv run scripts/run_demo.py --config configs/L1_full_observability

# L2_partial_observability (8×8 POMDP)
uv run scripts/run_demo.py --config configs/L2_partial_observability

# L3_temporal_mechanics (8×8 temporal)
uv run scripts/run_demo.py --config configs/L3_temporal_mechanics
```

Verify:
- [ ] Environment initializes without errors
- [ ] Observations have correct dimensions
- [ ] Agents can take actions and receive rewards
- [ ] Training proceeds normally (at least 100 episodes)

---

## Risk Mitigation

### Risk 1: Observation Dimension Mismatch

**Symptom**: Network expects obs_dim=91, substrate provides obs_dim=29

**Cause**: Encoding changed from one-hot to coordinates

**Mitigation**:
1. Checkpoint validation checks obs_dim on load
2. Clear error message: "Checkpoint obs_dim mismatch - delete old checkpoints"
3. Logging shows dimension breakdown on env creation

**Probability**: Medium (operators changing configs)
**Severity**: High (breaks training)
**Status**: Mitigated

---

### Risk 2: POMDP Network Breaks with 3D

**Symptom**: RecurrentSpatialQNetwork crashes with position_dim=3

**Cause**: Hardcoded position slicing

**Mitigation**:
1. Dynamic position_dim parameter added to network
2. Forward() uses self.position_dim for slicing
3. Tests verify with Grid2D (position_dim=2) and Aspatial (position_dim=0)

**Probability**: High (when 3D implemented)
**Severity**: High (blocks 3D)
**Status**: Mitigated

---

### Risk 3: Aspatial Affordance Interaction Undefined

**Symptom**: Aspatial agents can't interact (no position)

**Cause**: Interaction requires distance check

**Mitigation**:
1. Aspatial substrate returns is_on_position()=True for all affordances
2. All affordances always accessible in aspatial
3. Action masking allows INTERACT without movement

**Probability**: High (aspatial is new)
**Severity**: Medium (blocks aspatial)
**Status**: Mitigated (from Phase 4)

---

### Risk 4: Performance Regression from Looping

**Symptom**: POMDP observation construction takes 10× longer

**Cause**: encode_partial_observation() called per-agent (not vectorized)

**Mitigation**:
1. Defer optimization to Phase 6
2. Current approach works for num_agents ≤ 10
3. Benchmark if needed with larger populations

**Probability**: Low (current approach sufficient)
**Severity**: Low (only affects large populations)
**Status**: Acceptable (defer to Phase 6)

---

## Estimated Effort Breakdown

| Task | Description | Estimated Hours | Actual Hours |
|------|-------------|----------------|--------------|
| 5.1 | Add substrate observation methods | 4 | [TBD] |
| 5.2 | Update VectorizedHamletEnv obs_dim | 2 | [TBD] |
| 5.3 | Refactor ObservationBuilder constructor | 1 | [TBD] |
| 5.4 | Refactor full observability encoding | 2 | [TBD] |
| 5.5 | Refactor POMDP encoding | 3 | [TBD] |
| 5.6 | Update RecurrentSpatialQNetwork | 2 | [TBD] |
| 5.7 | Update test suite | 4 | [TBD] |
| 5.8 | Documentation | 1 | [TBD] |
| **Subtotal** | **Core Implementation** | **19** | [TBD] |
| **Buffer** | Edge cases, debugging | **1** | [TBD] |
| **Total** | **Phase 5 Complete** | **20** | [TBD] |

---

## Success Criteria

Phase 5 is complete when:

1. ✅ All substrate methods implemented (get_observation_dim, encode_observation, encode_partial_observation)
2. ✅ VectorizedHamletEnv uses substrate.get_observation_dim()
3. ✅ ObservationBuilder uses substrate.encode_observation()
4. ✅ RecurrentSpatialQNetwork supports variable position_dim
5. ✅ No hardcoded grid_size or position_dim=2 references
6. ✅ All tests pass
7. ✅ All curriculum levels (L0-L3) work with substrate encoding
8. ✅ Documentation updated with breaking changes notice

**Phase 5 Status**: Ready for Implementation

**Next Phase**: Phase 6 (Optional) - Performance Optimization
