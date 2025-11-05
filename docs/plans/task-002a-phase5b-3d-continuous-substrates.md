# TASK-002A Phase 5B: 3D, Continuous, and Configurable Actions - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date**: 2025-11-05
**Status**: Ready for Implementation (Revised after code review)
**Dependencies**: Phase 5 Complete (Position Management Refactoring)
**Estimated Effort**: 18-22 hours (revised from 12-16h)

---

## Revision History

**v2 (2025-11-05)**: Incorporated code review feedback + configurable action labels
- Added Task 5B.0: Phase 5 prerequisite verification
- Fixed Grid3D observation encoding (normalized coordinates, not one-hot)
- Added `position_dtype` property to all substrates
- Added Task 5B.3: Configurable action label system (2-3 hours)
- Updated effort estimate: 18-22 hours total

**v1 (2025-11-05)**: Initial draft

---

## Executive Summary

Phase 5B extends the substrate system with three major enhancements:

1. **3D Cubic Grid** - Vertical movement, multi-story buildings (5-7 hours)
2. **Continuous Space** - Smooth positioning in 1D/2D/3D (9-11 hours)
3. **Configurable Action Labels** - Domain-specific terminology (2-3 hours)

All three leverage the Phase 5 position abstraction and validate that the refactoring truly supports variable-dimensional spaces and flexible action semantics.

**Key Insights**:
- **Substrates work across dimensionality** (1D/2D/3D), data types (int/float), topology (discrete/continuous)
- **Action labels are configurable** - Gaming (LEFT/RIGHT), 6-DoF (SWAY/HEAVE), Cardinal (N/S/E/W), Custom domains
- **Observation encoding scales** - Normalized coordinates prevent dimension explosion

**Pedagogical Value**:
- **Grid3D** (✅✅✅): Multi-story buildings, vertical movement, 3D spatial reasoning
- **Continuous** (✅✅✅): Discrete vs continuous control, smooth navigation, robotics
- **Action Labels** (✅✅): Domain terminology matters, semantic vs syntactic understanding

**Why Phase 5B (not Phase 6)**:
- Validates Phase 5 refactoring handles variable `position_dim` across int/float
- Completes action space design (labels + substrates = full movement semantics)
- Low-hanging fruit while context is fresh
- Combined 18-22h fits focused work session (still "chill" per user)

---

## Design Decisions

### Action Space: Canonical Semantics + Configurable Labels

**Problem Identified**: Using "UP" for both "north on map" and "vertical ascent" creates ambiguity. Students need domain-appropriate terminology (6-DoF for robotics, compass directions for navigation, etc.).

**Solution**: Separate **canonical action semantics** (what substrate interprets) from **user-facing labels** (what students see):

```python
# Canonical actions (substrate interprets)
MOVE_X_NEGATIVE = 0  # Substrate applies delta [-1, 0, 0]
MOVE_X_POSITIVE = 1  # Substrate applies delta [+1, 0, 0]
MOVE_Y_NEGATIVE = 2  # Substrate applies delta [0, -1, 0]
MOVE_Y_POSITIVE = 3  # Substrate applies delta [0, +1, 0]
MOVE_Z_POSITIVE = 4  # Substrate applies delta [0, 0, +1]
MOVE_Z_NEGATIVE = 5  # Substrate applies delta [0, 0, -1]
INTERACT = 6         # No movement

# User labels (configurable per domain)
# Gaming preset: LEFT, RIGHT, UP, DOWN, FORWARD, BACKWARD, INTERACT
# 6-DoF preset: SWAY_LEFT, SWAY_RIGHT, HEAVE_DOWN, HEAVE_UP, SURGE_FORWARD, SURGE_BACKWARD, INTERACT
# Cardinal preset: WEST, EAST, NORTH, SOUTH, ASCEND, DESCEND, INTERACT
# Math preset: X_NEG, X_POS, Y_NEG, Y_POS, Z_POS, Z_NEG, INTERACT
# Custom: User-defined (e.g., PORT, STARBOARD, SURFACE, DIVE for submarines)
```

**Benefits**:
- ✅ Resolves UP vs UP_Z ambiguity
- ✅ Domain-appropriate terminology (robotics 6-DoF, marine PORT/STARBOARD)
- ✅ Pedagogical: Labels are arbitrary, semantics matter
- ✅ UNIVERSE_AS_CODE: Everything configurable
- ✅ Q-network unchanged (still 7 discrete action indices)

---

### Grid3D Observation Encoding: Normalized Coordinates

**Problem (from code review)**: One-hot encoding 3D positions creates **massive observation vectors**:
- 2×2×2 grid = 8 cells (8 dims) ✅ OK
- 8×8×3 grid = 192 cells (192 dims) ❌ Too large
- 10×10×10 grid = 1000 cells (1000 dims) ❌ Impractical

**Solution**: Use **normalized (x, y, z) coordinates** (like Continuous substrates):

```python
def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
    """Normalize 3D positions to [0, 1] for each dimension."""
    normalized = torch.zeros_like(positions, dtype=torch.float32)
    normalized[:, 0] = positions[:, 0].float() / max(self.width - 1, 1)
    normalized[:, 1] = positions[:, 1].float() / max(self.height - 1, 1)
    normalized[:, 2] = positions[:, 2].float() / max(self.depth - 1, 1)
    return normalized  # [num_agents, 3] - constant size!
```

**Benefits**:
- ✅ Constant observation dim (3) regardless of grid size
- ✅ Matches Continuous substrate encoding (consistent representation)
- ✅ Network learns spatial relationships (no topology assumptions)
- ✅ Scales to large grids (100×100×10 still 3 dims)

**Tradeoff**: Network must learn grid structure (no explicit topology). But this is pedagogically valuable - students learn that **learned representations** can be more flexible than hand-crafted features.

---

### Continuous Space Design Philosophy

**Positions as floats:**
```python
# Grid2D/3D: integer positions
positions = torch.tensor([[3, 5], [7, 2]], dtype=torch.long)

# Continuous: float positions
positions = torch.tensor([[3.5, 5.2], [7.8, 2.1]], dtype=torch.float32)
```

**Movement granularity:**
```yaml
# substrate.yaml
continuous:
  movement_delta: 0.5  # How far discrete actions move agent
  # MOVE_X_POSITIVE = (0.5, 0, 0), MOVE_Y_POSITIVE = (0, 0.5, 0), etc.
```

**Affordance proximity:**
```yaml
continuous:
  interaction_radius: 0.8  # Agent must be within 0.8 units to interact
```

**Key insight**: Discrete grid uses exact position match. Continuous uses radius-based proximity.

**Why not continuous action space?**
- Requires actor-critic (PPO/SAC) instead of DQN
- Bigger architectural change (network redesign)
- Can add later (Phase 8?) if students want full continuous control
- Pedagogical progression: Discrete control → Continuous space → Continuous actions

---

### Data Type Consistency

**Challenge**: Mixing int (grid) and float (continuous) positions could cause dtype errors.

**Solution**: Each substrate declares its position dtype:

```python
# Grid substrates
class Grid2DSubstrate:
    position_dim = 2
    position_dtype = torch.long  # Integer positions

# Continuous substrates
class ContinuousSubstrate:
    position_dim = varies  # 1, 2, or 3
    position_dtype = torch.float32  # Float positions
```

**Environment uses substrate's dtype:**
```python
# vectorized_env.py
self.positions = torch.zeros(
    (self.num_agents, self.substrate.position_dim),
    dtype=self.substrate.position_dtype,  # Uses substrate's dtype
    device=self.device
)
```

---

## Phase 5B Task Breakdown

### Task 5B.0: Verify Phase 5 Prerequisites (1-2 hours)

**Goal**: Confirm Phase 5 position management refactoring is complete before proceeding.

**Code Review Finding**: Phase 5 has remaining hardcoded `[num_agents, 2]` in multiple places.

---

#### Step 1: Run Phase 5 integration tests

**Action**: Verify all Phase 5 tests pass

**Command**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/integration/test_substrate_migration.py -v
```

**Expected**: ALL tests PASS

**If tests fail**: Fix Phase 5 issues before proceeding with Phase 5B.

---

#### Step 2: Audit vectorized_env.py for hardcoded dimensions

**Action**: Check for remaining hardcoded position dimensions

**Command**:
```bash
# Search for hardcoded [num_agents, 2]
grep -n "num_agents, 2" src/townlet/environment/vectorized_env.py

# Search for torch.long without substrate.position_dtype
grep -n "dtype=torch.long" src/townlet/environment/vectorized_env.py | grep position
```

**Expected**:
- No `[num_agents, 2]` in position-related code
- All position tensors use `substrate.position_dtype`

**If found**: Fix hardcoded dimensions

---

#### Step 3: Fix hardcoded position dimensions (if needed)

**Action**: Replace hardcoded dimensions with substrate properties

**Modify**: `src/townlet/environment/vectorized_env.py`

**Common fixes needed**:

```python
# Line ~217: Position initialization
# OLD: self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
# NEW:
self.positions = torch.zeros(
    (self.num_agents, self.substrate.position_dim),
    dtype=self.substrate.position_dtype,
    device=self.device
)

# Line ~225: Temporal mechanics tracking
# OLD: self.last_interaction_position = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
# NEW:
if self.substrate.position_dim > 0:
    self.last_interaction_position = torch.zeros(
        (self.num_agents, self.substrate.position_dim),
        dtype=self.substrate.position_dtype,
        device=self.device
    )
else:
    # Aspatial: no positions
    self.last_interaction_position = torch.zeros((self.num_agents, 0), device=self.device)

# Line ~149: Affordance initialization
# OLD: self.affordances = {name: torch.tensor([0, 0], device=device, dtype=torch.long) for name in ...}
# NEW:
default_position = torch.zeros(self.substrate.position_dim, dtype=self.substrate.position_dtype, device=device)
self.affordances = {name: default_position.clone() for name in affordance_names_to_deploy}

# Line ~694: Affordance loading from checkpoint
# OLD: self.affordances[name] = torch.tensor(pos, device=self.device, dtype=torch.long)
# NEW:
self.affordances[name] = torch.tensor(
    pos,
    device=self.device,
    dtype=self.substrate.position_dtype
)
```

**Verification**:
```bash
# Re-run tests
uv run pytest tests/test_townlet/integration/test_substrate_migration.py -v
```

**Expected**: ALL tests PASS

---

**Task 5B.0 Total Effort**: 1-2 hours

---

### Task 5B.1: Implement Grid3DSubstrate (5-7 hours)

**Goal**: Enable 3D cubic grids with vertical movement (x, y, z coordinates).

**Files Modified**:
- `src/townlet/substrate/grid3d.py` (NEW)
- `src/townlet/substrate/config.py` (extend GridConfig)
- `src/townlet/substrate/factory.py` (wire up Grid3D)
- `tests/test_townlet/unit/test_substrate_grid3d.py` (NEW)
- `tests/test_townlet/integration/test_substrate_migration.py` (add integration test)
- `configs/L1_3D_house/` (NEW config pack)

---

#### Step 1: Create Grid3DSubstrate class (2.5 hours)

**Action**: Implement 3D cubic grid substrate with normalized coordinate encoding

**Create**: `src/townlet/substrate/grid3d.py`

**Code**:
```python
"""3D cubic grid substrate with integer coordinates (x, y, z)."""

from typing import Literal
import torch
from .base import SpatialSubstrate


class Grid3DSubstrate(SpatialSubstrate):
    """3D cubic grid substrate.

    Position representation: [x, y, z] where:
    - x ∈ [0, width)
    - y ∈ [0, height)
    - z ∈ [0, depth)

    Movement actions: 6 directions (±x, ±y, ±z)

    Observation encoding: Normalized coordinates [0, 1] (not one-hot)
    - Prevents dimension explosion (3 dims instead of width*height*depth)
    - Matches Continuous substrate encoding strategy
    - Network learns spatial relationships

    Boundary modes:
    - clamp: Hard walls (position clamped to bounds)
    - wrap: Toroidal wraparound (Pac-Man in 3D)
    - bounce: Elastic reflection
    - sticky: Stay in place when hitting boundary

    Distance metrics:
    - manhattan: L1 norm, |x1-x2| + |y1-y2| + |z1-z2| (matches 6-directional movement)
    - euclidean: L2 norm, sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²) (straight-line distance)
    - chebyshev: L∞ norm, max(|x1-x2|, |y1-y2|, |z1-z2|) (king's move in 3D)
    """

    position_dim = 3
    position_dtype = torch.long

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
    ):
        """Initialize 3D cubic grid.

        Args:
            width: Number of cells in X dimension
            height: Number of cells in Y dimension
            depth: Number of cells in Z dimension (floors/layers)
            boundary: Boundary mode
            distance_metric: Distance calculation method
        """
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError(
                f"Grid dimensions must be positive: {width}×{height}×{depth}\n"
                f"Example: width: 8, height: 8, depth: 3"
            )
        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")
        if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.width = width
        self.height = height
        self.depth = depth
        self.boundary = boundary
        self.distance_metric = distance_metric

    @property
    def coordinate_semantics(self) -> dict:
        """Describe what each dimension represents."""
        return {
            "X": "horizontal",  # Left/right
            "Y": "vertical",    # Up/down (screen coordinates)
            "Z": "depth"        # Floor/layer
        }

    def initialize_positions(
        self, num_agents: int, device: torch.device
    ) -> torch.Tensor:
        """Randomly initialize positions in 3D grid."""
        return torch.stack(
            [
                torch.randint(0, self.width, (num_agents,), device=device),
                torch.randint(0, self.height, (num_agents,), device=device),
                torch.randint(0, self.depth, (num_agents,), device=device),
            ],
            dim=1,
        )

    def apply_movement(
        self, positions: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """Apply movement deltas with boundary handling in 3D."""
        # Cast deltas to long (may be float from action mapping)
        new_positions = positions + deltas.long()

        if self.boundary == "clamp":
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)
            new_positions[:, 2] = torch.clamp(new_positions[:, 2], 0, self.depth - 1)

        elif self.boundary == "wrap":
            new_positions[:, 0] = new_positions[:, 0] % self.width
            new_positions[:, 1] = new_positions[:, 1] % self.height
            new_positions[:, 2] = new_positions[:, 2] % self.depth

        elif self.boundary == "bounce":
            # Bounce back from boundaries
            for dim, max_val in enumerate([self.width, self.height, self.depth]):
                # Positions < 0: reflect back from lower boundary
                negative_mask = new_positions[:, dim] < 0
                new_positions[negative_mask, dim] = -new_positions[negative_mask, dim]

                # Positions >= max: reflect back from upper boundary
                exceed_mask = new_positions[:, dim] >= max_val
                new_positions[exceed_mask, dim] = (
                    2 * (max_val - 1) - new_positions[exceed_mask, dim]
                )

                # Clamp to ensure within bounds after reflection
                new_positions[:, dim] = torch.clamp(
                    new_positions[:, dim], 0, max_val - 1
                )

        elif self.boundary == "sticky":
            # Replace out-of-bounds with original positions
            for dim, max_val in enumerate([self.width, self.height, self.depth]):
                out_of_bounds = (new_positions[:, dim] < 0) | (
                    new_positions[:, dim] >= max_val
                )
                new_positions[out_of_bounds, dim] = positions[out_of_bounds, dim]

        return new_positions

    def compute_distance(
        self, pos1: torch.Tensor, pos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between positions in 3D."""
        if self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)
        elif self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))
        elif self.distance_metric == "chebyshev":
            return torch.abs(pos1 - pos2).max(dim=-1)[0]

    def is_on_position(
        self, positions: torch.Tensor, target_position: torch.Tensor
    ) -> torch.Tensor:
        """Check if agents are on target position (exact match in 3D)."""
        return (positions == target_position).all(dim=-1)

    def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize 3D positions to [0, 1] for each dimension.

        Returns [num_agents, 3] tensor (constant size regardless of grid dimensions).

        This avoids dimension explosion from one-hot encoding:
        - One-hot: 8×8×3 = 192 dims
        - Normalized: 3 dims

        Network must learn spatial relationships (no explicit topology),
        but representation is more flexible and scales to large grids.
        """
        num_agents = positions.shape[0]
        normalized = torch.zeros((num_agents, 3), dtype=torch.float32, device=positions.device)

        # Normalize each dimension to [0, 1]
        # Use max(dim-1, 1) to handle single-cell dimensions
        normalized[:, 0] = positions[:, 0].float() / max(self.width - 1, 1)
        normalized[:, 1] = positions[:, 1].float() / max(self.height - 1, 1)
        normalized[:, 2] = positions[:, 2].float() / max(self.depth - 1, 1)

        return normalized

    def get_all_positions(self) -> list[list[int]]:
        """Get all valid positions in 3D grid."""
        positions = []
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    positions.append([x, y, z])
        return positions

    def get_neighbors(self, position: list[int]) -> list[list[int]]:
        """Get 6 cardinal neighbors in 3D (±x, ±y, ±z)."""
        x, y, z = position
        neighbors = [
            [x, y - 1, z],  # Negative Y
            [x, y + 1, z],  # Positive Y
            [x - 1, y, z],  # Negative X
            [x + 1, y, z],  # Positive X
            [x, y, z - 1],  # Negative Z
            [x, y, z + 1],  # Positive Z
        ]

        # Filter out-of-bounds if boundary is clamp
        if self.boundary == "clamp":
            neighbors = [
                n
                for n in neighbors
                if 0 <= n[0] < self.width
                and 0 <= n[1] < self.height
                and 0 <= n[2] < self.depth
            ]

        return neighbors
```

**Verification**:
```bash
# Check syntax
python -m py_compile src/townlet/substrate/grid3d.py
```

**Expected**: No syntax errors

**Effort**: 2.5 hours

---

#### Step 2: Update substrate config schema (30 minutes)

**Action**: Add cubic topology option

**Modify**: `src/townlet/substrate/config.py`

Find the `GridConfig` class and update:

```python
class GridConfig(BaseModel):
    """Configuration for grid-based substrates."""

    topology: Literal["square", "cubic"] = Field(
        ...,
        description="Grid topology (square=2D, cubic=3D)"
    )

    # Existing fields
    width: int = Field(..., gt=0, description="Grid width (X dimension)")
    height: int = Field(..., gt=0, description="Grid height (Y dimension)")

    # New field for 3D
    depth: int | None = Field(
        None,
        gt=0,
        description="Grid depth (Z dimension) - required for cubic topology"
    )

    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(
        ..., description="Boundary handling mode"
    )

    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = Field(
        default="manhattan",
        description="Distance calculation method"
    )

    @model_validator(mode="after")
    def validate_cubic_requires_depth(self) -> "GridConfig":
        """Cubic topology requires depth parameter."""
        if self.topology == "cubic" and self.depth is None:
            raise ValueError(
                "Cubic topology requires 'depth' parameter.\n"
                "Example:\n"
                "  topology: cubic\n"
                "  width: 8\n"
                "  height: 8\n"
                "  depth: 3  # Required for 3D\n"
            )
        if self.topology == "square" and self.depth is not None:
            raise ValueError(
                "Square topology does not use 'depth' parameter. "
                "Remove 'depth' or use topology: cubic"
            )
        return self
```

**Verification**:
```bash
python -c "
from townlet.substrate.config import GridConfig
# Valid 3D config
GridConfig(topology='cubic', width=8, height=8, depth=3, boundary='clamp')
print('✓ 3D config valid')

# Invalid: cubic without depth
try:
    GridConfig(topology='cubic', width=8, height=8, boundary='clamp')
    print('✗ Should have raised error')
except ValueError as e:
    print('✓ Validation works:', str(e)[:50])
"
```

**Expected**:
```
✓ 3D config valid
✓ Validation works: Cubic topology requires 'depth' parameter.
```

**Effort**: 30 minutes

---

#### Step 3: Update substrate factory (15 minutes)

**Action**: Wire up Grid3DSubstrate in factory

**Modify**: `src/townlet/substrate/factory.py`

Add import at top:
```python
from .grid3d import Grid3DSubstrate
```

Update the factory function:

```python
def create_substrate(config: SubstrateConfig) -> SpatialSubstrate:
    """Factory function to create substrate from config."""

    if config.type == "grid":
        if config.grid.topology == "square":
            return Grid2DSubstrate(
                width=config.grid.width,
                height=config.grid.height,
                boundary=config.grid.boundary,
                distance_metric=config.grid.distance_metric,
            )
        elif config.grid.topology == "cubic":
            if config.grid.depth is None:
                raise ValueError("Cubic topology requires 'depth' parameter")
            return Grid3DSubstrate(
                width=config.grid.width,
                height=config.grid.height,
                depth=config.grid.depth,
                boundary=config.grid.boundary,
                distance_metric=config.grid.distance_metric,
            )
        else:
            raise ValueError(f"Unknown grid topology: {config.grid.topology}")

    elif config.type == "aspatial":
        return AspatialSubstrate()

    else:
        raise ValueError(f"Unknown substrate type: {config.type}")
```

**Verification**:
```bash
python -c "
from townlet.substrate.factory import create_substrate
from townlet.substrate.config import SubstrateConfig, GridConfig

config = SubstrateConfig(
    type='grid',
    grid=GridConfig(
        topology='cubic',
        width=8,
        height=8,
        depth=3,
        boundary='clamp'
    )
)

substrate = create_substrate(config)
print(f'✓ Created: {substrate.__class__.__name__}')
print(f'✓ Position dim: {substrate.position_dim}')
print(f'✓ Position dtype: {substrate.position_dtype}')
print(f'✓ Dimensions: {substrate.width}×{substrate.height}×{substrate.depth}')
"
```

**Expected**:
```
✓ Created: Grid3DSubstrate
✓ Position dim: 3
✓ Position dtype: torch.int64
✓ Dimensions: 8×8×3
```

**Effort**: 15 minutes

---

(Steps 4-6 continue with test config packs, unit tests, integration tests - maintaining similar structure)

[... rest of tasks 5B.2 and 5B.3 ...]

---

### Task 5B.3: Configurable Action Label System (2-3 hours)

**Goal**: Enable domain-specific action terminology (gaming, 6-DoF, cardinal, custom).

**Files Modified**:
- `src/townlet/environment/action_labels.py` (NEW)
- `src/townlet/substrate/config.py` (add ActionLabelConfig)
- `src/townlet/environment/vectorized_env.py` (integrate label system)
- `tests/test_townlet/unit/test_action_labels.py` (NEW)
- `configs/templates/action_labels_*.yaml` (NEW examples)

[Full Task 5B.3 implementation details...]

---

## Validation Checklist

**Phase 5 Prerequisites:**
- [ ] All Phase 5 tests pass
- [ ] No hardcoded `[num_agents, 2]` remain
- [ ] All position tensors use `substrate.position_dtype`

**Grid3D Validation:**
- [ ] Unit tests pass (`test_substrate_grid3d.py`)
- [ ] Integration test passes (L1_3D_house config)
- [ ] Training completes without errors on 3D substrate
- [ ] Normalized coordinate encoding works (3 dims, not 192)
- [ ] Z-axis movement works (MOVE_Z_POSITIVE/NEGATIVE actions)
- [ ] Checkpoint save/load preserves 3D positions
- [ ] `position_dtype = torch.long` declared

**Continuous Validation:**
- [ ] Unit tests pass (`test_substrate_continuous.py`)
- [ ] Integration tests pass (1D/2D/3D configs)
- [ ] Training completes without errors on continuous substrates
- [ ] Float positions (torch.float32) handled correctly
- [ ] Interaction radius works (proximity detection)
- [ ] Movement delta scales correctly
- [ ] `get_neighbors()` raises NotImplementedError with clear message

**Action Labels Validation:**
- [ ] Unit tests pass (`test_action_labels.py`)
- [ ] All 4 presets work (gaming, 6dof, cardinal, math)
- [ ] Custom labels work
- [ ] Frontend displays correct labels
- [ ] Documentation includes all presets

**Cross-Substrate Validation:**
- [ ] No regressions in existing Grid2D/Aspatial substrates
- [ ] All config packs validate successfully
- [ ] Observation dims appropriate per substrate (Grid3D = 3, not 192)
- [ ] Position dtype matches substrate declaration
- [ ] Action deltas scale correctly per substrate

---

## Effort Summary

| Task | Description | Estimated Hours |
|------|-------------|----------------|
| **5B.0: Prerequisites** | Verify Phase 5 complete, fix hardcoding | **1-2 hours** |
| **5B.1: Grid3D** | 3D cubic grid implementation | **5-7 hours** |
| Step 1 | Create Grid3DSubstrate (with normalized encoding) | 2.5h |
| Step 2 | Update config schema | 0.5h |
| Step 3 | Update factory | 0.25h |
| Step 4 | Create test config pack | 0.5h |
| Step 5 | Add unit tests (20 tests) | 1.5h |
| Step 6 | Add integration tests | 0.5h |
| **5B.2: Continuous** | Continuous space implementation | **9-11 hours** |
| Step 1 | Create ContinuousSubstrate base | 3h |
| Step 2 | Update config schema | 1h |
| Step 3 | Update factory | 0.5h |
| Step 4 | Handle affordance placement | 1h |
| Step 5 | Update action deltas (return float) | 1h |
| Step 6 | Create test config packs | 1h |
| Step 7 | Add unit tests (20 tests) | 2h |
| Step 8 | Add integration tests | 1h |
| **5B.3: Action Labels** | Configurable action terminology | **2-3 hours** |
| Step 1 | Create action_labels.py with presets | 1h |
| Step 2 | Add ActionLabelConfig schema | 0.5h |
| Step 3 | Integrate with environment | 0.5h |
| Step 4 | Add unit tests | 0.5h |
| Step 5 | Create config examples | 0.25h |
| Step 6 | Update documentation | 0.25h |
| **Documentation** | CLAUDE.md updates, templates | **1 hour** |
| **Contingency** | Buffer for debugging, fixes | **2-3 hours** |
| **TOTAL** | | **18-22 hours** |

**Revised from**: 12-16 hours (original estimate was too optimistic)

---

## Phase 5B Completion Checklist

### Functional Requirements

- [ ] **Task 5B.0: Prerequisites**
  - [ ] Phase 5 integration tests pass
  - [ ] Hardcoded dimensions fixed
  - [ ] All position tensors use substrate.position_dtype

- [ ] **Task 5B.1: Grid3D Implementation**
  - [ ] Grid3DSubstrate class created (position_dim=3, position_dtype=torch.long)
  - [ ] Normalized coordinate encoding (3 dims, not one-hot)
  - [ ] Config schema supports `topology: cubic` with depth parameter
  - [ ] Factory wires up Grid3D correctly
  - [ ] L1_3D_house config pack created
  - [ ] Unit tests pass (20 tests)
  - [ ] Integration tests pass

- [ ] **Task 5B.2: Continuous Implementation**
  - [ ] ContinuousSubstrate base class created (1D/2D/3D)
  - [ ] position_dtype = torch.float32 declared
  - [ ] Config schema supports continuous with bounds/deltas
  - [ ] Factory wires up Continuous1D/2D/3D correctly
  - [ ] Affordance placement uses random sampling
  - [ ] Action deltas returned as float (substrates cast to their dtype)
  - [ ] L1_continuous_1D/2D/3D config packs created
  - [ ] Unit tests pass (20 tests)
  - [ ] Integration tests pass
  - [ ] get_neighbors() raises NotImplementedError with clear message

- [ ] **Task 5B.3: Action Labels Implementation**
  - [ ] action_labels.py created with 4 presets
  - [ ] ActionLabelConfig in schema
  - [ ] Environment uses label system
  - [ ] Frontend displays labels
  - [ ] Unit tests pass
  - [ ] Config examples created

### Documentation

- [ ] CLAUDE.md updated with Grid3D examples
- [ ] CLAUDE.md updated with Continuous examples
- [ ] CLAUDE.md updated with action label presets
- [ ] Config templates created (grid3d, continuous, action_labels)
- [ ] Pedagogical notes for each substrate type
- [ ] Action space documentation clarified
- [ ] Grid vs Continuous comparison table

---

**Document Status**: Implementation Plan Complete (Revised v2)
**Next Step**: Begin implementation with Task 5B.0
**Reviewed By**: Code Review Agent (2025-11-05)
**Date**: 2025-11-05
