# TASK-002A Phase 5B: 3D, Continuous, and Configurable Actions - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date**: 2025-11-05
**Completion Date**: 2025-11-06
**Status**: ✅ COMPLETE
**Dependencies**: Phase 5 Complete (Position Management Refactoring)
**Estimated Effort**: 19-23 hours (revised from 18-22h)
**Actual Effort**: ~22 hours

---

## Revision History

**v4 (2025-11-06)**: ✅ PHASE 5B COMPLETE
- Completed Task 5B.1: Grid3DSubstrate implementation (commit a01250d)
- Completed Task 5B.2: Continuous Substrates implementation (commits d5b8304, ddb1aa7)
- Completed Task 5B.3: Configurable Action Label System (commit f6941fe)
- All tests passing (100%): 43 continuous unit + 5 integration + 31 action label unit
- All validation checklist items complete
- Actual effort: ~22 hours (within 19-23h estimate)

**v3 (2025-11-05)**: Fixed critical technical issues from code review
- Fixed `encode_observation()` signature: added `affordances` parameter
- Fixed method naming: `get_neighbors()` → `get_valid_neighbors()`
- Added Task 5B.0 Step 2: Add `position_dtype` property to existing substrates
- Fixed test paths: `test_substrate_migration.py` → `tests/test_townlet/phase5/`
- Updated effort estimate: 19-23 hours total

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

#### Step 1: Run Phase 5 integration tests (15 minutes)

**Action**: Verify all Phase 5 tests pass

**Command**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run Phase 5 substrate tests
uv run pytest tests/test_townlet/phase5/ -v

# Also check for any substrate integration tests
uv run pytest tests/test_townlet/integration/ -v -k substrate
```

**Expected**: ALL tests PASS

**If tests fail**: Fix Phase 5 issues before proceeding with Phase 5B.

**Effort**: 15 minutes

---

#### Step 2: Add position_dtype property to existing substrates (30 minutes)

**Goal**: Add position_dtype class attribute to Grid2D and Aspatial substrates (required for Phase 5B)

**Code Review Finding**: The position_dtype property doesn't exist in current codebase yet. Phase 5B assumes it exists.

**Action**: Add position_dtype to existing substrate classes

**Modify**: `src/townlet/substrate/grid2d.py`

```python
class Grid2DSubstrate(SpatialSubstrate):
    """2D square grid substrate with integer coordinates.

    ... (existing docstring) ...
    """

    position_dim = 2
    position_dtype = torch.long  # ADD THIS LINE

    def __init__(self, ...):
        # ... rest of class ...
```

**Modify**: `src/townlet/substrate/aspatial.py`

```python
class AspatialSubstrate(SpatialSubstrate):
    """Aspatial substrate with no positioning.

    ... (existing docstring) ...
    """

    position_dim = 0
    position_dtype = torch.long  # ADD THIS LINE (any dtype, no positions exist)

    def __init__(self):
        # ... rest of class ...
```

**Verification**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

python -c "
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate
import torch

# Test Grid2D
grid = Grid2DSubstrate(width=8, height=8, boundary='clamp')
assert hasattr(grid, 'position_dtype'), 'Grid2D missing position_dtype'
assert grid.position_dtype == torch.long, 'Grid2D position_dtype should be torch.long'
print('✓ Grid2DSubstrate has position_dtype = torch.long')

# Test Aspatial
aspatial = AspatialSubstrate()
assert hasattr(aspatial, 'position_dtype'), 'Aspatial missing position_dtype'
assert aspatial.position_dtype == torch.long, 'Aspatial position_dtype should be torch.long'
print('✓ AspatialSubstrate has position_dtype = torch.long')
"
```

**Expected**:
```
✓ Grid2DSubstrate has position_dtype = torch.long
✓ AspatialSubstrate has position_dtype = torch.long
```

**Effort**: 30 minutes

---

#### Step 3: Audit vectorized_env.py for hardcoded dimensions (15 minutes)

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

**If found**: Proceed to Step 4 to fix

**If not found**: Skip Step 4, proceed to Task 5B.1

**Effort**: 15 minutes

---

#### Step 4: Fix hardcoded position dimensions (if needed) (1-2 hours)

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

**Task 5B.0 Total Effort**: 2-3 hours (increased from 1-2h to add position_dtype setup)

**Breakdown**:
- Step 1: Run tests (15 min)
- Step 2: Add position_dtype property (30 min)
- Step 3: Audit hardcoded dimensions (15 min)
- Step 4: Fix hardcoded dimensions if needed (1-2h contingent)

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

    def encode_observation(
        self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Normalize 3D positions to [0, 1] for each dimension.

        Args:
            positions: Agent positions [num_agents, 3]
            affordances: Dict of affordance positions (not used for Grid3D encoding)

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

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get 6 cardinal neighbors in 3D (±x, ±y, ±z).

        Args:
            position: Position tensor [3] or list of [x, y, z]

        Returns:
            List of neighbor position tensors
        """
        # Convert to list if tensor
        if isinstance(position, torch.Tensor):
            x, y, z = position.tolist()
        else:
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

        # Convert to tensors
        return [torch.tensor(n, dtype=torch.long) for n in neighbors]
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

#### Step 4: Create test config pack (30 minutes)

**Action**: Create L1_3D_house config pack for testing

**Create directory**: `configs/L1_3D_house/`

**Create**: `configs/L1_3D_house/substrate.yaml`

```yaml
version: "1.0"
description: "3D cubic grid - 3-story house with vertical movement"

type: "grid"

grid:
  topology: "cubic"
  width: 8
  height: 8
  depth: 3  # Three floors
  boundary: "clamp"
  distance_metric: "manhattan"
```

**Copy remaining files from L1_full_observability**:
```bash
cd configs
cp L1_full_observability/{bars.yaml,cascades.yaml,affordances.yaml,cues.yaml,training.yaml} L1_3D_house/
```

**Verification**:
```bash
cd /home/john/hamlet
python -c "
from pathlib import Path
from townlet.substrate.factory import load_substrate_config

config = load_substrate_config(Path('configs/L1_3D_house'))
print(f'✓ Loaded: {config.type} substrate')
print(f'✓ Topology: {config.grid.topology}')
print(f'✓ Dimensions: {config.grid.width}×{config.grid.height}×{config.grid.depth}')
"
```

**Expected**:
```
✓ Loaded: grid substrate
✓ Topology: cubic
✓ Dimensions: 8×8×3
```

**Effort**: 30 minutes

---

#### Step 5: Add unit tests (1.5 hours)

**Action**: Create comprehensive test suite for Grid3D

**Create**: `tests/test_townlet/unit/test_substrate_grid3d.py`

```python
"""Unit tests for Grid3DSubstrate."""

import pytest
import torch
from townlet.substrate.grid3d import Grid3DSubstrate


class TestGrid3DInitialization:
    """Tests for Grid3D initialization."""

    def test_initialization_valid(self):
        """Valid grid initializes."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        assert substrate.position_dim == 3
        assert substrate.position_dtype == torch.long
        assert substrate.width == 8
        assert substrate.height == 8
        assert substrate.depth == 3

    def test_initialization_invalid_dimensions(self):
        """Invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            Grid3DSubstrate(width=0, height=8, depth=3, boundary="clamp")

        with pytest.raises(ValueError, match="dimensions must be positive"):
            Grid3DSubstrate(width=8, height=-1, depth=3, boundary="clamp")

    def test_initialization_invalid_boundary(self):
        """Invalid boundary raises ValueError."""
        with pytest.raises(ValueError, match="Unknown boundary mode"):
            Grid3DSubstrate(width=8, height=8, depth=3, boundary="invalid")


class TestGrid3DPositionInitialization:
    """Tests for position initialization."""

    def test_initialize_positions_shape(self):
        """Positions have correct shape."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = substrate.initialize_positions(100, torch.device("cpu"))

        assert positions.shape == (100, 3)
        assert positions.dtype == torch.long

    def test_initialize_positions_in_bounds(self):
        """Positions within grid bounds."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = substrate.initialize_positions(1000, torch.device("cpu"))

        assert (positions[:, 0] >= 0).all()
        assert (positions[:, 0] < 8).all()
        assert (positions[:, 1] >= 0).all()
        assert (positions[:, 1] < 8).all()
        assert (positions[:, 2] >= 0).all()
        assert (positions[:, 2] < 3).all()


class TestGrid3DMovement:
    """Tests for 3D movement."""

    def test_movement_x_axis(self):
        """Movement along X axis."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = torch.tensor([[4, 4, 1]], dtype=torch.long)
        deltas = torch.tensor([[1, 0, 0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        assert torch.equal(new_positions, torch.tensor([[5, 4, 1]]))

    def test_movement_z_axis(self):
        """Movement along Z axis (vertical)."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = torch.tensor([[4, 4, 1]], dtype=torch.long)
        deltas = torch.tensor([[0, 0, 1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Z from 1 → 2 (going up one floor)
        assert torch.equal(new_positions, torch.tensor([[4, 4, 2]]))

    def test_movement_clamp_boundary(self):
        """Clamp boundary prevents out of bounds."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = torch.tensor([[7, 7, 2]], dtype=torch.long)
        deltas = torch.tensor([[1, 1, 1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # All dimensions clamped to max
        assert torch.equal(new_positions, torch.tensor([[7, 7, 2]]))

    def test_movement_wrap_boundary(self):
        """Wrap boundary uses toroidal wraparound in 3D."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="wrap"
        )
        positions = torch.tensor([[7, 7, 2]], dtype=torch.long)
        deltas = torch.tensor([[1, 1, 1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Wraps: (8 % 8, 8 % 8, 3 % 3) = (0, 0, 0)
        assert torch.equal(new_positions, torch.tensor([[0, 0, 0]]))

    def test_movement_batch(self):
        """Batch movement works."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = torch.tensor(
            [[1, 2, 0], [3, 4, 1], [5, 6, 2]],
            dtype=torch.long
        )
        deltas = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float32
        )

        new_positions = substrate.apply_movement(positions, deltas)

        expected = torch.tensor([[2, 2, 0], [3, 5, 1], [5, 6, 2]])
        assert torch.equal(new_positions, expected)


class TestGrid3DDistance:
    """Tests for distance calculations."""

    def test_distance_manhattan(self):
        """Manhattan distance in 3D."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3,
            boundary="clamp",
            distance_metric="manhattan"
        )
        pos1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 4, 2]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # |3| + |4| + |2| = 9
        assert distance[0].item() == 9

    def test_distance_euclidean(self):
        """Euclidean distance in 3D."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3,
            boundary="clamp",
            distance_metric="euclidean"
        )
        pos1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 4, 0]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # sqrt(9 + 16) = 5
        assert torch.isclose(distance[0], torch.tensor(5.0))

    def test_distance_chebyshev(self):
        """Chebyshev distance in 3D."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3,
            boundary="clamp",
            distance_metric="chebyshev"
        )
        pos1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 7, 1]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # max(3, 7, 1) = 7
        assert distance[0].item() == 7


class TestGrid3DObservationEncoding:
    """Tests for observation encoding."""

    def test_encode_observation_shape(self):
        """Observation encoding returns constant size."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = torch.tensor(
            [[0, 0, 0], [7, 7, 2], [4, 4, 1]],
            dtype=torch.long
        )

        obs = substrate.encode_observation(positions)

        # Should be [num_agents, 3] regardless of grid size
        assert obs.shape == (3, 3)
        assert obs.dtype == torch.float32

    def test_encode_observation_normalization(self):
        """Observations normalized to [0, 1]."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        positions = torch.tensor([[0, 0, 0], [7, 7, 2]], dtype=torch.long)

        obs = substrate.encode_observation(positions)

        # Min corner: [0, 0, 0]
        assert torch.allclose(obs[0], torch.tensor([0.0, 0.0, 0.0]))

        # Max corner: [7, 7, 2]
        # Normalized: [7/7, 7/7, 2/2] = [1, 1, 1]
        assert torch.allclose(obs[1], torch.tensor([1.0, 1.0, 1.0]))

    def test_encode_observation_scales_with_grid_size(self):
        """Large grids still produce 3-dim observations."""
        substrate = Grid3DSubstrate(
            width=100, height=100, depth=10, boundary="clamp"
        )
        positions = torch.tensor([[50, 50, 5]], dtype=torch.long)

        obs = substrate.encode_observation(positions)

        # Still 3 dims, not 100*100*10=100K dims!
        assert obs.shape == (1, 3)

        # Middle of grid ≈ [0.5, 0.5, 0.5]
        expected = torch.tensor([50/99, 50/99, 5/9])
        assert torch.allclose(obs[0], expected, atol=0.01)


class TestGrid3DPositionChecks:
    """Tests for position checking."""

    def test_is_on_position_exact_match(self):
        """is_on_position returns True for exact match."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        agent_positions = torch.tensor([[3, 4, 1], [5, 6, 2]], dtype=torch.long)
        target_position = torch.tensor([3, 4, 1], dtype=torch.long)

        on_position = substrate.is_on_position(agent_positions, target_position)

        assert on_position[0].item() == True
        assert on_position[1].item() == False

    def test_is_on_position_no_match(self):
        """is_on_position returns False for different positions."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        agent_positions = torch.tensor([[3, 4, 1]], dtype=torch.long)
        target_position = torch.tensor([3, 4, 2], dtype=torch.long)

        on_position = substrate.is_on_position(agent_positions, target_position)

        # Different floor (z=1 vs z=2)
        assert on_position[0].item() == False


class TestGrid3DNeighbors:
    """Tests for neighbor enumeration."""

    def test_get_valid_neighbors_interior(self):
        """Interior position has 6 neighbors."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        neighbors = substrate.get_valid_neighbors(torch.tensor([4, 4, 1]))

        assert len(neighbors) == 6

        expected = {
            (4, 3, 1),  # Y-
            (4, 5, 1),  # Y+
            (3, 4, 1),  # X-
            (5, 4, 1),  # X+
            (4, 4, 0),  # Z-
            (4, 4, 2),  # Z+
        }
        assert {tuple(n.tolist()) for n in neighbors} == expected

    def test_get_valid_neighbors_corner_clamp(self):
        """Corner position with clamp has fewer neighbors."""
        substrate = Grid3DSubstrate(
            width=8, height=8, depth=3, boundary="clamp"
        )
        neighbors = substrate.get_valid_neighbors(torch.tensor([0, 0, 0]))

        # Only 3 neighbors (no negatives)
        assert len(neighbors) == 3

        expected = {
            (0, 1, 0),  # Y+
            (1, 0, 0),  # X+
            (0, 0, 1),  # Z+
        }
        assert {tuple(n.tolist()) for n in neighbors} == expected

    def test_get_all_positions(self):
        """get_all_positions returns all grid cells."""
        substrate = Grid3DSubstrate(
            width=2, height=2, depth=2, boundary="clamp"
        )
        all_positions = substrate.get_all_positions()

        # 2*2*2 = 8 cells
        assert len(all_positions) == 8

        # Should contain all corners
        assert [0, 0, 0] in all_positions
        assert [1, 1, 1] in all_positions


class TestGrid3DConfiguration:
    """Tests for config integration."""

    def test_config_cubic_topology(self):
        """Config with cubic topology creates Grid3D."""
        from townlet.substrate.config import SubstrateConfig, GridConfig
        from townlet.substrate.factory import create_substrate

        config = SubstrateConfig(
            type="grid",
            grid=GridConfig(
                topology="cubic",
                width=8,
                height=8,
                depth=3,
                boundary="clamp"
            )
        )

        substrate = create_substrate(config)

        assert isinstance(substrate, Grid3DSubstrate)
        assert substrate.position_dim == 3
        assert substrate.position_dtype == torch.long
```

**Run tests**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_grid3d.py -v
```

**Expected**: All 20+ tests PASS

**Effort**: 1.5 hours

---

#### Step 6: Add integration test (30 minutes)

**Action**: Test Grid3D in full training loop

**Modify**: `tests/test_townlet/integration/test_substrate_migration.py`

Add parametrized test:

```python
def test_training_with_grid3d_substrate(tmp_path):
    """Training runs with 3D cubic grid."""
    from pathlib import Path
    from townlet.demo.runner import DemoRunner

    config_dir = Path("configs/L1_3D_house")
    if not config_dir.exists():
        pytest.skip("L1_3D_house config not found")

    with DemoRunner(
        config_dir=config_dir,
        db_path=tmp_path / "test.db",
        checkpoint_dir=tmp_path / "checkpoints",
        max_episodes=5,
        training_config_path=config_dir / "training.yaml",
    ) as runner:
        runner.run()

        # Verify 3D positions
        assert runner.env.positions.shape[1] == 3
        assert runner.env.positions.dtype == torch.long

        # Verify Z dimension in bounds
        assert (runner.env.positions[:, 2] >= 0).all()
        assert (runner.env.positions[:, 2] < 3).all()
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_migration.py::test_training_with_grid3d_substrate -v
```

**Expected**: Test PASSES

**Effort**: 30 minutes

---

**Task 5B.1 Total Effort**: 5-7 hours

---

### Task 5B.2: Implement Continuous Substrates (9-11 hours)

**Goal**: Enable continuous float-based positioning in 1D/2D/3D with configurable bounds, movement granularity, and proximity detection.

**Files Modified**:
- `src/townlet/substrate/continuous.py` (NEW)
- `src/townlet/substrate/config.py` (add ContinuousConfig)
- `src/townlet/substrate/factory.py` (wire up Continuous1D/2D/3D)
- `src/townlet/environment/vectorized_env.py` (update affordance placement & action deltas)
- `tests/test_townlet/unit/test_substrate_continuous.py` (NEW, 20 tests)
- `tests/test_townlet/integration/test_substrate_migration.py` (add integration tests)
- `configs/L1_continuous_1D/`, `L1_continuous_2D/`, `L1_continuous_3D/` (NEW config packs)
- `configs/templates/substrate_continuous_2d.yaml` (NEW template)

---

#### Step 1: Create ContinuousSubstrate base class (3 hours)

**Action**: Implement base continuous substrate with 1D/2D/3D specializations

**Create**: `src/townlet/substrate/continuous.py`

```python
"""Continuous space substrates with float-based positioning."""

from typing import Literal
import torch
from .base import SpatialSubstrate


class ContinuousSubstrate(SpatialSubstrate):
    """Base class for continuous space substrates.

    Position representation: float coordinates in bounded space
    - 1D: [x] where x ∈ [min_x, max_x]
    - 2D: [x, y] where x ∈ [min_x, max_x], y ∈ [min_y, max_y]
    - 3D: [x, y, z] where x ∈ [min_x, max_x], y ∈ [min_y, max_y], z ∈ [min_z, max_z]

    Movement: Discrete actions move agent by fixed `movement_delta`
    - MOVE_X_NEGATIVE = delta = (-movement_delta, 0, 0)
    - MOVE_X_POSITIVE = delta = (+movement_delta, 0, 0)
    - etc.

    Interaction: Agent must be within `interaction_radius` of affordance
    - Uses distance metric (euclidean or manhattan)
    - Proximity-based, not exact position match

    Observation encoding: Normalized coordinates [0, 1] per dimension
    - Same as Grid3D (consistent representation)
    - Constant size regardless of bounds
    """

    position_dtype = torch.float32

    def __init__(
        self,
        dimensions: int,
        bounds: list[tuple[float, float]],
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        """Initialize continuous substrate.

        Args:
            dimensions: Number of dimensions (1, 2, or 3)
            bounds: List of (min, max) tuples for each dimension
            boundary: Boundary handling mode
            movement_delta: Distance discrete actions move agent
            interaction_radius: Distance threshold for affordance interaction
            distance_metric: Distance calculation method
        """
        if dimensions not in (1, 2, 3):
            raise ValueError(f"Continuous substrates support 1-3 dimensions, got {dimensions}")

        if len(bounds) != dimensions:
            raise ValueError(
                f"Number of bounds ({len(bounds)}) must match dimensions ({dimensions}). "
                f"Example for 2D: bounds=[(0.0, 10.0), (0.0, 10.0)]"
            )

        for i, (min_val, max_val) in enumerate(bounds):
            if min_val >= max_val:
                raise ValueError(
                    f"Bound {i} invalid: min ({min_val}) must be < max ({max_val})"
                )

            # Check space is large enough for interaction
            range_size = max_val - min_val
            if range_size < interaction_radius:
                raise ValueError(
                    f"Dimension {i} range ({range_size}) < interaction_radius ({interaction_radius}). "
                    f"Space too small for affordance interaction."
                )

        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        if distance_metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        if movement_delta <= 0:
            raise ValueError(f"movement_delta must be positive, got {movement_delta}")

        if interaction_radius <= 0:
            raise ValueError(f"interaction_radius must be positive, got {interaction_radius}")

        # Warn if interaction_radius < movement_delta
        if interaction_radius < movement_delta:
            import warnings
            warnings.warn(
                f"interaction_radius ({interaction_radius}) < movement_delta ({movement_delta}). "
                f"Agent may step over affordances without interaction. "
                f"This may be intentional for challenge, but verify configuration.",
                UserWarning
            )

        self.dimensions = dimensions
        self.bounds = bounds
        self.boundary = boundary
        self.movement_delta = movement_delta
        self.interaction_radius = interaction_radius
        self.distance_metric = distance_metric

    @property
    def position_dim(self) -> int:
        """Number of dimensions."""
        return self.dimensions

    @property
    def coordinate_semantics(self) -> dict:
        """Describe what each dimension represents."""
        names = {1: {"X": "position"},
                 2: {"X": "horizontal", "Y": "vertical"},
                 3: {"X": "horizontal", "Y": "vertical", "Z": "depth"}}
        return names.get(self.dimensions, {})

    def initialize_positions(
        self, num_agents: int, device: torch.device
    ) -> torch.Tensor:
        """Randomly initialize positions in continuous space."""
        positions = []
        for min_val, max_val in self.bounds:
            dim_positions = torch.rand(num_agents, device=device) * (max_val - min_val) + min_val
            positions.append(dim_positions)

        return torch.stack(positions, dim=1)

    def apply_movement(
        self, positions: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """Apply continuous movement with boundary handling."""
        new_positions = positions + deltas.float()

        for dim in range(self.dimensions):
            min_val, max_val = self.bounds[dim]

            if self.boundary == "clamp":
                new_positions[:, dim] = torch.clamp(
                    new_positions[:, dim], min_val, max_val
                )

            elif self.boundary == "wrap":
                # Toroidal wraparound
                range_size = max_val - min_val
                # Shift to [0, range_size), wrap, shift back
                new_positions[:, dim] = (
                    (new_positions[:, dim] - min_val) % range_size
                ) + min_val

            elif self.boundary == "bounce":
                # Elastic reflection
                range_size = max_val - min_val

                # Normalize to [0, range_size)
                normalized = new_positions[:, dim] - min_val

                # Reflect about boundaries (multiple bounces)
                # Fold into [0, 2*range_size)
                normalized = normalized % (2 * range_size)

                # If in second half, reflect back
                exceed_half = normalized >= range_size
                normalized[exceed_half] = 2 * range_size - normalized[exceed_half]

                # Denormalize back
                new_positions[:, dim] = normalized + min_val

            elif self.boundary == "sticky":
                # Stay in place if out of bounds
                out_of_bounds = (new_positions[:, dim] < min_val) | (
                    new_positions[:, dim] > max_val
                )
                new_positions[out_of_bounds, dim] = positions[out_of_bounds, dim]

        return new_positions

    def compute_distance(
        self, pos1: torch.Tensor, pos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between positions in continuous space."""
        if self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))
        elif self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)

    def is_on_position(
        self, positions: torch.Tensor, target_position: torch.Tensor
    ) -> torch.Tensor:
        """Check if agents are within interaction radius of target.

        For continuous space, this is proximity-based (not exact match).
        """
        distance = self.compute_distance(positions, target_position)
        return distance <= self.interaction_radius

    def encode_observation(
        self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Normalize positions to [0, 1] for each dimension.

        Args:
            positions: Agent positions [num_agents, dimensions]
            affordances: Dict of affordance positions (not used for Continuous encoding)

        Returns [num_agents, dimensions] tensor.
        Matches Grid3D encoding strategy (constant size).
        """
        num_agents = positions.shape[0]
        normalized = torch.zeros(
            (num_agents, self.dimensions),
            dtype=torch.float32,
            device=positions.device
        )

        for dim in range(self.dimensions):
            min_val, max_val = self.bounds[dim]
            range_size = max_val - min_val
            normalized[:, dim] = (positions[:, dim] - min_val) / range_size

        return normalized

    def get_all_positions(self) -> list[list[float]]:
        """Raise error - continuous space has infinite positions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has infinite positions (continuous space). "
            f"Use random sampling for affordance placement instead. "
            f"See vectorized_env.py randomize_affordance_positions()."
        )

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Raise error - continuous space has no discrete neighbors.

        Args:
            position: Position tensor (not used)

        Raises:
            NotImplementedError: Continuous substrates don't have discrete neighbors
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has continuous positions. "
            f"No discrete neighbors exist. "
            f"Use compute_distance() and interaction_radius for proximity detection."
        )

    def supports_enumerable_positions(self) -> bool:
        """Continuous substrates have infinite positions."""
        return False


class Continuous1DSubstrate(ContinuousSubstrate):
    """1D continuous line."""

    def __init__(
        self,
        min_x: float,
        max_x: float,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        super().__init__(
            dimensions=1,
            bounds=[(min_x, max_x)],
            boundary=boundary,
            movement_delta=movement_delta,
            interaction_radius=interaction_radius,
            distance_metric=distance_metric,
        )
        self.min_x = min_x
        self.max_x = max_x


class Continuous2DSubstrate(ContinuousSubstrate):
    """2D continuous plane."""

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        super().__init__(
            dimensions=2,
            bounds=[(min_x, max_x), (min_y, max_y)],
            boundary=boundary,
            movement_delta=movement_delta,
            interaction_radius=interaction_radius,
            distance_metric=distance_metric,
        )
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


class Continuous3DSubstrate(ContinuousSubstrate):
    """3D continuous space."""

    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_z: float,
        max_z: float,
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        movement_delta: float,
        interaction_radius: float,
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        super().__init__(
            dimensions=3,
            bounds=[(min_x, max_x), (min_y, max_y), (min_z, max_z)],
            boundary=boundary,
            movement_delta=movement_delta,
            interaction_radius=interaction_radius,
            distance_metric=distance_metric,
        )
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z
```

**Verification**:
```bash
python -m py_compile src/townlet/substrate/continuous.py
```

**Expected**: No syntax errors

**Effort**: 3 hours

---

#### Step 2: Update substrate config schema (1 hour)

**Action**: Add ContinuousConfig to config system

**Modify**: `src/townlet/substrate/config.py`

Add new config class:

```python
class ContinuousConfig(BaseModel):
    """Configuration for continuous substrates."""

    dimensions: int = Field(
        ...,
        ge=1,
        le=3,
        description="Number of dimensions (1, 2, or 3)"
    )

    bounds: list[tuple[float, float]] = Field(
        ...,
        description="Bounds for each dimension [(min, max), ...]"
    )

    boundary: Literal["clamp", "wrap", "bounce", "sticky"] = Field(
        ...,
        description="Boundary handling mode"
    )

    movement_delta: float = Field(
        ...,
        gt=0,
        description="Distance discrete actions move agent"
    )

    interaction_radius: float = Field(
        ...,
        gt=0,
        description="Distance threshold for affordance interaction"
    )

    distance_metric: Literal["euclidean", "manhattan"] = Field(
        default="euclidean",
        description="Distance calculation method"
    )

    @model_validator(mode="after")
    def validate_bounds_match_dimensions(self) -> "ContinuousConfig":
        """Validate bounds and interaction parameters."""
        if len(self.bounds) != self.dimensions:
            raise ValueError(
                f"Number of bounds ({len(self.bounds)}) must match dimensions ({self.dimensions}). "
                f"Example for 2D: bounds=[(0.0, 10.0), (0.0, 10.0)]"
            )

        for i, (min_val, max_val) in enumerate(self.bounds):
            if min_val >= max_val:
                raise ValueError(
                    f"Bound {i} invalid: min ({min_val}) must be < max ({max_val})"
                )

            # Check space is large enough for interaction
            range_size = max_val - min_val
            if range_size < self.interaction_radius:
                raise ValueError(
                    f"Dimension {i} range ({range_size}) < interaction_radius ({self.interaction_radius}). "
                    f"Space too small for affordance interaction."
                )

        # Warn if interaction_radius < movement_delta
        if self.interaction_radius < self.movement_delta:
            import warnings
            warnings.warn(
                f"interaction_radius ({self.interaction_radius}) < movement_delta ({self.movement_delta}). "
                f"Agent may step over affordances without interaction.",
                UserWarning
            )

        return self
```

Update `SubstrateConfig` to include continuous:

```python
class SubstrateConfig(BaseModel):
    """Top-level substrate configuration."""

    type: Literal["grid", "continuous", "aspatial"] = Field(...)

    grid: GridConfig | None = Field(
        default=None,
        description="Grid substrate configuration (required if type='grid')"
    )

    continuous: ContinuousConfig | None = Field(
        default=None,
        description="Continuous substrate configuration (required if type='continuous')"
    )

    @model_validator(mode="after")
    def validate_type_matches_config(self) -> "SubstrateConfig":
        """Ensure type matches provided config."""
        if self.type == "grid" and self.grid is None:
            raise ValueError("type='grid' requires 'grid' config")
        if self.type == "continuous" and self.continuous is None:
            raise ValueError("type='continuous' requires 'continuous' config")
        if self.type == "aspatial" and (self.grid is not None or self.continuous is not None):
            raise ValueError("type='aspatial' should not have 'grid' or 'continuous' config")
        return self
```

**Verification**:
```bash
python -c "
from townlet.substrate.config import ContinuousConfig

# Valid 2D config
config = ContinuousConfig(
    dimensions=2,
    bounds=[(0.0, 10.0), (0.0, 10.0)],
    boundary='clamp',
    movement_delta=0.5,
    interaction_radius=0.8
)
print('✓ 2D continuous config valid')
print(f'✓ Bounds: {config.bounds}')
print(f'✓ Movement delta: {config.movement_delta}')
"
```

**Expected**:
```
✓ 2D continuous config valid
✓ Bounds: [(0.0, 10.0), (0.0, 10.0)]
✓ Movement delta: 0.5
```

**Effort**: 1 hour

---

#### Step 3: Update substrate factory (30 minutes)

**Action**: Wire up Continuous substrates in factory

**Modify**: `src/townlet/substrate/factory.py`

Add import:
```python
from .continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
```

Update factory function:

```python
def create_substrate(config: SubstrateConfig) -> SpatialSubstrate:
    """Factory function to create substrate from config."""

    if config.type == "grid":
        # ... existing grid logic ...

    elif config.type == "continuous":
        if config.continuous.dimensions == 1:
            (min_x, max_x) = config.continuous.bounds[0]
            return Continuous1DSubstrate(
                min_x=min_x,
                max_x=max_x,
                boundary=config.continuous.boundary,
                movement_delta=config.continuous.movement_delta,
                interaction_radius=config.continuous.interaction_radius,
                distance_metric=config.continuous.distance_metric,
            )

        elif config.continuous.dimensions == 2:
            (min_x, max_x), (min_y, max_y) = config.continuous.bounds
            return Continuous2DSubstrate(
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                boundary=config.continuous.boundary,
                movement_delta=config.continuous.movement_delta,
                interaction_radius=config.continuous.interaction_radius,
                distance_metric=config.continuous.distance_metric,
            )

        elif config.continuous.dimensions == 3:
            (min_x, max_x), (min_y, max_y), (min_z, max_z) = config.continuous.bounds
            return Continuous3DSubstrate(
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                min_z=min_z,
                max_z=max_z,
                boundary=config.continuous.boundary,
                movement_delta=config.continuous.movement_delta,
                interaction_radius=config.continuous.interaction_radius,
                distance_metric=config.continuous.distance_metric,
            )
        else:
            raise ValueError(f"Unsupported continuous dimensions: {config.continuous.dimensions}")

    elif config.type == "aspatial":
        return AspatialSubstrate()

    else:
        raise ValueError(f"Unknown substrate type: {config.type}")
```

**Verification**:
```bash
python -c "
from townlet.substrate.factory import create_substrate
from townlet.substrate.config import SubstrateConfig, ContinuousConfig

config = SubstrateConfig(
    type='continuous',
    continuous=ContinuousConfig(
        dimensions=2,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        boundary='clamp',
        movement_delta=0.5,
        interaction_radius=0.8
    )
)

substrate = create_substrate(config)
print(f'✓ Created: {substrate.__class__.__name__}')
print(f'✓ Position dim: {substrate.position_dim}')
print(f'✓ Position dtype: {substrate.position_dtype}')
"
```

**Expected**:
```
✓ Created: Continuous2DSubstrate
✓ Position dim: 2
✓ Position dtype: torch.float32
```

**Effort**: 30 minutes

---

#### Step 4: Handle affordance placement for continuous (1 hour)

**Action**: Update affordance placement to use random sampling for continuous substrates

**Modify**: `src/townlet/environment/vectorized_env.py`

Add helper method to check if substrate supports enumerable positions:

```python
def supports_enumerable_positions(self) -> bool:
    """Check if substrate has finite enumerable positions."""
    return hasattr(self.substrate, 'supports_enumerable_positions') and \
           self.substrate.supports_enumerable_positions()
```

Add method to randomize affordance positions:

```python
def randomize_affordance_positions(self) -> None:
    """Randomize affordance positions using substrate.

    Grid substrates: Shuffle all positions
    Continuous substrates: Random sampling
    Aspatial: No positions
    """
    # Aspatial substrates don't have positions
    if self.substrate.position_dim == 0:
        self.affordance_positions = torch.zeros(
            (len(self.affordances), 0),
            dtype=self.substrate.position_dtype,
            device=self.device
        )
        return

    # Check if substrate supports enumerable positions
    if self.supports_enumerable_positions():
        # Grid substrates: shuffle all positions
        import random
        all_positions = self.substrate.get_all_positions()

        if len(all_positions) < len(self.affordances):
            raise ValueError(
                f"Not enough positions for affordances. "
                f"Substrate has {len(all_positions)} positions, "
                f"but {len(self.affordances)} affordances enabled."
            )

        random.shuffle(all_positions)
        selected = all_positions[: len(self.affordances)]

        self.affordance_positions = torch.tensor(
            selected,
            dtype=self.substrate.position_dtype,
            device=self.device
        )
    else:
        # Continuous/other: random sampling
        self.affordance_positions = self.substrate.initialize_positions(
            num_agents=len(self.affordances),
            device=self.device
        )
```

Update `__init__` to call this method after affordances are created:

```python
# In __init__, after self.affordances is populated:
self.randomize_affordance_positions()
```

**Effort**: 1 hour

---

#### Step 5: Update action deltas to return float (1 hour)

**Action**: Make `_action_to_deltas()` return float deltas (substrates cast to their dtype)

**Modify**: `src/townlet/environment/vectorized_env.py`

Update `_action_to_deltas` method:

```python
def _action_to_deltas(self, actions: torch.Tensor) -> torch.Tensor:
    """Map action indices to movement deltas.

    Returns float deltas. Substrates cast to their dtype as needed.

    Canonical actions:
    - 0: MOVE_X_NEGATIVE
    - 1: MOVE_X_POSITIVE
    - 2: MOVE_Y_NEGATIVE
    - 3: MOVE_Y_POSITIVE
    - 4: MOVE_Z_POSITIVE (3D only)
    - 5: MOVE_Z_NEGATIVE (3D only)
    - 6+: INTERACT

    Returns:
        [num_agents, 3] tensor of float deltas (padded to max 3 dimensions)
    """
    num_agents = actions.shape[0]
    # Return float32 (substrates will cast to their dtype as needed)
    deltas = torch.zeros((num_agents, 3), dtype=torch.float32, device=self.device)

    # MOVE_X_NEGATIVE (0)
    deltas[actions == 0, 0] = -1.0
    # MOVE_X_POSITIVE (1)
    deltas[actions == 1, 0] = 1.0
    # MOVE_Y_NEGATIVE (2)
    deltas[actions == 2, 1] = -1.0
    # MOVE_Y_POSITIVE (3)
    deltas[actions == 3, 1] = 1.0
    # MOVE_Z_POSITIVE (4)
    deltas[actions == 4, 2] = 1.0
    # MOVE_Z_NEGATIVE (5)
    deltas[actions == 5, 2] = -1.0
    # INTERACT (6+): no movement (already zeros)

    # Apply movement_delta scaling for continuous substrates
    if hasattr(self.substrate, 'movement_delta'):
        deltas *= self.substrate.movement_delta

    return deltas
```

**Also update Grid substrates** to cast to long in their `apply_movement`:

**Modify**: `src/townlet/substrate/grid2d.py` and `src/townlet/substrate/grid3d.py`

```python
def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply movement deltas with boundary handling."""
    # Cast deltas to long for grid substrates
    new_positions = positions + deltas.long()
    # ... rest of boundary handling logic ...
```

**Effort**: 1 hour

---

#### Step 6: Create test config packs (1 hour)

**Action**: Create L1_continuous_1D, L1_continuous_2D, L1_continuous_3D config packs

**Create**: `configs/L1_continuous_1D/substrate.yaml`

```yaml
version: "1.0"
description: "1D continuous line - simple navigation challenge"

type: "continuous"

continuous:
  dimensions: 1
  bounds:
    - [0.0, 10.0]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"
```

**Create**: `configs/L1_continuous_2D/substrate.yaml`

```yaml
version: "1.0"
description: "2D continuous plane with smooth navigation"

type: "continuous"

continuous:
  dimensions: 2
  bounds:
    - [0.0, 10.0]
    - [0.0, 10.0]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"
```

**Create**: `configs/L1_continuous_3D/substrate.yaml`

```yaml
version: "1.0"
description: "3D continuous space with volumetric navigation"

type: "continuous"

continuous:
  dimensions: 3
  bounds:
    - [0.0, 10.0]
    - [0.0, 10.0]
    - [0.0, 10.0]
  boundary: "clamp"
  movement_delta: 0.5
  interaction_radius: 0.8
  distance_metric: "euclidean"
```

**Copy remaining files**:
```bash
cd configs
for config in L1_continuous_{1D,2D,3D}; do
  mkdir -p $config
  cp L1_full_observability/{bars.yaml,cascades.yaml,affordances.yaml,cues.yaml,training.yaml} $config/
done
```

**Verification**:
```bash
cd /home/john/hamlet
for config in L1_continuous_{1D,2D,3D}; do
  python -c "
from pathlib import Path
from townlet.substrate.factory import load_substrate_config
config = load_substrate_config(Path('configs/$config'))
print(f'✓ $config: {config.continuous.dimensions}D continuous')
  "
done
```

**Effort**: 1 hour

---

#### Step 7: Add unit tests (2 hours)

**Action**: Create comprehensive test suite for continuous substrates

**Create**: `tests/test_townlet/unit/test_substrate_continuous.py`

(Full 300+ line test file with 20+ tests covering initialization, movement, distance, observation encoding, boundary modes, dtype consistency - see research agent output for complete code)

**Run tests**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_continuous.py -v
```

**Expected**: All 20+ tests PASS

**Effort**: 2 hours

---

#### Step 8: Add integration tests (1 hour)

**Action**: Test continuous substrates in full training loop

**Modify**: `tests/test_townlet/integration/test_substrate_migration.py`

Add parametrized tests for continuous substrates and observation dimension validation.

**Run tests**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_migration.py -v -k continuous
```

**Expected**: All integration tests PASS

**Effort**: 1 hour

---

**Task 5B.2 Total Effort**: 9-11 hours

---

### Task 5B.3: Configurable Action Label System (2-3 hours)

**Goal**: Enable domain-specific action terminology (gaming, 6-DoF, cardinal, custom).

**Files Modified**:
- `src/townlet/environment/action_labels.py` (NEW)
- `src/townlet/substrate/config.py` (add ActionLabelConfig)
- `src/townlet/environment/vectorized_env.py` (integrate label system)
- `tests/test_townlet/unit/test_action_labels.py` (NEW)
- `configs/templates/action_labels_*.yaml` (NEW examples)

---

#### Step 1: Create action_labels.py with presets (1 hour)

**Action**: Implement action label system with 4 presets

**Create**: `src/townlet/environment/action_labels.py`

(Full implementation with CanonicalAction enum, ActionLabels dataclass, 4 presets, and get_labels() function - see research agent output for complete 200+ line code)

**Verification**:
```bash
python -c "
from townlet.environment.action_labels import get_labels, PRESET_LABELS

for name in PRESET_LABELS.keys():
    labels = get_labels(name)
    print(f'✓ {name}: {labels.get_all_labels()}')
"
```

**Expected**: All 4 presets display correctly

**Effort**: 1 hour

---

#### Step 2: Add ActionLabelConfig schema (30 minutes)

**Action**: Add action label configuration to schema

**Modify**: `src/townlet/substrate/config.py`

Add ActionLabelConfig class with preset and custom label support.

**Verification**:
```bash
python -c "
from townlet.substrate.config import ActionLabelConfig

config = ActionLabelConfig(preset='gaming')
print('✓ Preset config valid')
"
```

**Effort**: 30 minutes

---

#### Step 3: Integrate with environment (30 minutes)

**Action**: Use action labels in VectorizedHamletEnv

**Modify**: `src/townlet/environment/vectorized_env.py`

Add action label loading in `__init__` and expose `action_label_names` property.

**Effort**: 30 minutes

---

#### Step 4: Add unit tests (30 minutes)

**Action**: Test action label system

**Create**: `tests/test_townlet/unit/test_action_labels.py`

(Full test suite with 10+ tests covering presets, customization, validation - see research agent output for complete code)

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_action_labels.py -v
```

**Expected**: All 10+ tests PASS

**Effort**: 30 minutes

---

#### Step 5: Create config examples (15 minutes)

**Action**: Create action label config examples

**Create**: `configs/templates/action_labels_gaming.yaml`, `action_labels_6dof.yaml`, `action_labels_cardinal.yaml`, `action_labels_submarine.yaml`

**Effort**: 15 minutes

---

#### Step 6: Update documentation (15 minutes)

**Action**: Document action label system in CLAUDE.md

**Modify**: `/home/john/hamlet/CLAUDE.md`

Add section explaining configurable action labels with presets and examples.

**Effort**: 15 minutes

---

**Task 5B.3 Total Effort**: 2-3 hours

---

## Validation Checklist

**Phase 5 Prerequisites:**
- [x] All Phase 5 tests pass
- [x] No hardcoded `[num_agents, 2]` remain
- [x] All position tensors use `substrate.position_dtype`

**Grid3D Validation:**
- [x] Unit tests pass (`test_substrate_grid3d.py`)
- [x] Integration test passes (L1_3D_house config)
- [x] Training completes without errors on 3D substrate
- [x] Normalized coordinate encoding works (3 dims, not 192)
- [x] Z-axis movement works (MOVE_Z_POSITIVE/NEGATIVE actions)
- [x] Checkpoint save/load preserves 3D positions
- [x] `position_dtype = torch.long` declared

**Continuous Validation:**
- [x] Unit tests pass (`test_substrate_continuous.py`) - 43 tests passing
- [x] Integration tests pass (1D/2D/3D configs) - 5 tests passing
- [x] Training completes without errors on continuous substrates
- [x] Float positions (torch.float32) handled correctly
- [x] Interaction radius works (proximity detection)
- [x] Movement delta scales correctly
- [x] `get_neighbors()` raises NotImplementedError with clear message

**Action Labels Validation:**
- [x] Unit tests pass (`test_action_labels.py`) - 31 tests passing
- [x] All 4 presets work (gaming, 6dof, cardinal, math)
- [x] Custom labels work
- [x] Frontend displays correct labels (via `get_action_label_names()`)
- [x] Documentation includes all presets (CLAUDE.md updated)

**Cross-Substrate Validation:**
- [x] No regressions in existing Grid2D/Aspatial substrates
- [x] All config packs validate successfully
- [x] Observation dims appropriate per substrate (Grid3D = 3, not 192)
- [x] Position dtype matches substrate declaration
- [x] Action deltas scale correctly per substrate

---

## Effort Summary

| Task | Description | Estimated Hours |
|------|-------------|----------------|
| **5B.0: Prerequisites** | Verify Phase 5, add position_dtype, fix hardcoding | **2-3 hours** |
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
| **TOTAL** | | **19-23 hours** |

**Revised from**:
- v1: 12-16 hours (original estimate was too optimistic)
- v2: 18-22 hours (after code review feedback)
- v3: 19-23 hours (after adding position_dtype setup)

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

**Document Status**: Implementation Plan Complete (v3 - Critical fixes applied)
**Next Step**: Begin implementation with Task 5B.0
**Reviewed By**: Code Review Agent (2025-11-05)
**Critical Fixes Applied**: 2025-11-05
**Date**: 2025-11-05
