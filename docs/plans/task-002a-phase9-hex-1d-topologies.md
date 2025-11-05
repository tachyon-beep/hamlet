# TASK-002A Phase 9: Simple Alternative Topologies (Hex + 1D) - Implementation Plan

**Date**: 2025-11-05 (Updated: 2025-11-05 post-risk-assessment)
**Status**: Ready for Implementation
**Dependencies**: Phase 5C Complete (N-Dimensional Substrates, Observation Encoding Retrofit)
**Estimated Effort**: 20-26 hours total (revised from 13-17h after risk assessment)
**Supersedes**: Original Phase 5D plan (split into Phase 9 + Phase 10 due to complexity)

---

## Executive Summary

Phase 9 adds two simple alternative topologies that **do not require infrastructure changes**:

1. **1D Grid** (6-8 hours) - Linear movement, simplest spatial case, surfaces 2D assumptions
2. **Hexagonal Grid** (10-12 hours) - Uniform distances, axial coordinates, strategy games
3. **Frontend Visualization** (3-4 hours) - Hex rendering + 1D mode (added post-risk-assessment)
4. **Documentation** (1-2 hours) - CLAUDE.md updates + regression testing

**Why the Split?**

The original Phase 5D plan included Graph-Based substrates (18-24h estimated), which require significant infrastructure changes:
- Action mapping refactoring (3-4h)
- Q-network dynamic sizing (2-3h)
- Replay buffer schema changes (2-3h)
- Frontend visualization updates (8-12h)

**Graph substrate deferred to Phase 10** to avoid blocking simple topologies.

**Implementation Order**: 1D → Hex → Frontend → Docs (revised from Hex→1D after risk assessment)

**Rationale for 1D-First**:
- 1D is simplest, will surface any hardcoded 2D assumptions early
- Fixing 2D assumptions benefits Hex implementation
- Builds confidence before tackling coordinate systems
- Tests action_dim=3 before action_dim=7

**Key Technical Challenges**:
- **1D**: Will expose hardcoded 2D assumptions (action mapping, observation dims)
- **Hex**: Axial coordinate system math (6 neighbors, uniform distances)
- **Frontend**: Hex requires axial→pixel conversion, SVG polygon rendering

**Pedagogical Value**:
- **1D** (✅): Edge case validator, teaches dimensionality concepts
- **Hex** (✅✅): Teaches coordinate system design, uniform distance metrics
- **Frontend** (✅✅): Essential for debugging and visual pedagogy

---

**⚠️ IMPLEMENTATION ORDER**:
1. **FIRST**: Scroll down to Task 9.1 (1D Grid) - Line ~793
2. **SECOND**: Return to Task 9.2 (Hex Grid) below
3. **THIRD**: Task 9.3 (Documentation)
4. **FOURTH**: Task 9.4 (Frontend Visualization)

**Rationale**: 1D Grid will surface hardcoded 2D assumptions early, providing cleaner architecture for Hex Grid.

---

## Task 9.2: Hexagonal Grid Substrate (10-12 hours) - IMPLEMENT SECOND (see line ~793 for Task 9.1)

### Overview

Implements 2D hexagonal grid with axial coordinates (q, r). All hexes have uniform distance to 6 neighbors (no diagonal ambiguity). Uses proven axial coordinate math for distance and neighbor detection.

**Files to Create**:
- `src/townlet/substrate/hexgrid.py` (new)
- `tests/test_townlet/unit/test_substrate_hexgrid.py` (new)
- `configs/L1_hex_strategy/` (new config pack)

**Files to Modify**:
- `src/townlet/substrate/config.py` (add hex config parsing)
- `src/townlet/substrate/factory.py` (wire up hex substrate)

---

### Step 2.1: Write unit tests for HexGridSubstrate (1 hour)

**Purpose**: Test-driven development - write tests first to define behavior

**Action**: Create comprehensive test suite for hex substrate

**Create**: `tests/test_townlet/unit/test_substrate_hexgrid.py`

```python
"""Unit tests for HexGridSubstrate."""
import pytest
import torch
from townlet.substrate.hexgrid import HexGridSubstrate


def test_hexgrid_initialization():
    """Hex grid should initialize with valid parameters."""
    substrate = HexGridSubstrate(
        radius=3,
        boundary="clamp",
        distance_metric="hex_manhattan",
        orientation="flat_top"
    )

    assert substrate.radius == 3
    assert substrate.position_dim == 2
    assert substrate.position_dtype == torch.long
    assert substrate.action_space_size == 7  # 6 directions + INTERACT


def test_hexgrid_invalid_initialization():
    """Hex grid should reject invalid parameters."""
    with pytest.raises(ValueError, match="Radius must be positive"):
        HexGridSubstrate(radius=0, boundary="clamp")

    with pytest.raises(ValueError, match="Unknown boundary"):
        HexGridSubstrate(radius=3, boundary="invalid")


def test_hexgrid_valid_positions():
    """Hex grid should generate valid positions within radius constraint."""
    substrate = HexGridSubstrate(radius=2, boundary="clamp")

    valid_positions = substrate.valid_positions

    # Radius 2 hex grid has 19 hexes
    assert len(valid_positions) == 19

    # Center hex should be valid
    assert (0, 0) in valid_positions

    # Edge hexes should be valid
    assert (2, 0) in valid_positions
    assert (0, 2) in valid_positions

    # Outside radius should be invalid
    assert (3, 0) not in valid_positions


def test_hexgrid_initialize_positions():
    """Hex grid should place agents at valid hex positions."""
    substrate = HexGridSubstrate(radius=3, boundary="clamp")

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    assert positions.shape == (10, 2)
    assert positions.dtype == torch.long

    # All positions should be within radius
    for pos in positions:
        q, r = pos.tolist()
        assert abs(q) + abs(r) <= 3


def test_hexgrid_apply_action_movement():
    """Hex grid should apply 6-directional movement correctly."""
    substrate = HexGridSubstrate(radius=5, boundary="clamp")

    start = torch.tensor([[0, 0]], dtype=torch.long)

    # Test all 6 directions
    actions = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    expected_deltas = [
        [+1,  0],  # EAST
        [+1, -1],  # NORTHEAST
        [ 0, -1],  # NORTHWEST
        [-1,  0],  # WEST
        [-1, +1],  # SOUTHWEST
        [ 0, +1],  # SOUTHEAST
    ]

    for action_idx, expected_delta in zip(actions, expected_deltas):
        action = action_idx.unsqueeze(0)
        new_pos = substrate.apply_action(start, action)

        expected_pos = torch.tensor([[expected_delta[0], expected_delta[1]]], dtype=torch.long)
        assert torch.equal(new_pos, expected_pos)


def test_hexgrid_apply_action_interact():
    """INTERACT action should not move agent."""
    substrate = HexGridSubstrate(radius=5, boundary="clamp")

    start = torch.tensor([[2, 1]], dtype=torch.long)
    interact_action = torch.tensor([6], dtype=torch.long)  # Action index 6 = INTERACT

    new_pos = substrate.apply_action(start, interact_action)

    assert torch.equal(new_pos, start)  # No movement


def test_hexgrid_boundary_clamp():
    """Clamping boundary should prevent movement beyond radius."""
    substrate = HexGridSubstrate(radius=2, boundary="clamp")

    # Start at edge hex (2, 0)
    edge = torch.tensor([[2, 0]], dtype=torch.long)

    # Try to move EAST (would go to 3, 0 - outside radius)
    move_east = torch.tensor([0], dtype=torch.long)
    clamped_pos = substrate.apply_action(edge, move_east)

    # Should stay at (2, 0)
    assert torch.equal(clamped_pos, edge)


def test_hexgrid_boundary_wrap():
    """Wrapping boundary should teleport to opposite side (toroidal hex)."""
    substrate = HexGridSubstrate(radius=2, boundary="wrap")

    # Start at edge hex (2, 0)
    edge = torch.tensor([[2, 0]], dtype=torch.long)

    # Try to move EAST
    move_east = torch.tensor([0], dtype=torch.long)
    wrapped_pos = substrate.apply_action(edge, move_east)

    # Should wrap to opposite side (-2, 0)
    expected = torch.tensor([[-2, 0]], dtype=torch.long)
    assert torch.equal(wrapped_pos, expected)


def test_hexgrid_distance_hex_manhattan():
    """Hex manhattan distance should be correct."""
    substrate = HexGridSubstrate(radius=5, distance_metric="hex_manhattan")

    pos1 = torch.tensor([[0, 0]], dtype=torch.long)
    pos2 = torch.tensor([[1, 0]], dtype=torch.long)

    distance = substrate.compute_distance(pos1, pos2)
    assert distance.item() == 1.0

    # Distance from (0,0) to (2, -1) should be 2 (not Euclidean!)
    pos3 = torch.tensor([[2, -1]], dtype=torch.long)
    distance2 = substrate.compute_distance(pos1, pos3)
    assert distance2.item() == 2.0


def test_hexgrid_distance_euclidean():
    """Euclidean distance should be computed in hex coordinate space."""
    substrate = HexGridSubstrate(radius=5, distance_metric="euclidean")

    pos1 = torch.tensor([[0, 0]], dtype=torch.long)
    pos2 = torch.tensor([[1, 0]], dtype=torch.long)

    distance = substrate.compute_distance(pos1, pos2)
    # Euclidean in axial coords: sqrt((1-0)^2 + (0-0)^2) = 1.0
    assert pytest.approx(distance.item(), 0.01) == 1.0


def test_hexgrid_get_valid_neighbors():
    """Hex grid should return 6 valid neighbors for interior hex."""
    substrate = HexGridSubstrate(radius=3, boundary="clamp")

    center = torch.tensor([0, 0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(center)

    # Center hex has 6 neighbors
    assert len(neighbors) == 6

    # Check expected neighbor coordinates
    expected = [
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    ]
    for neighbor in neighbors:
        q, r = neighbor.tolist()
        assert (q, r) in expected
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_hexgrid.py -v
```

**Expected**: ALL FAIL (implementation not written yet - TDD!)

---

### Step 2.2: Implement HexGridSubstrate (4-5 hours)

**Purpose**: Implement hex grid with axial coordinates

**Create**: `src/townlet/substrate/hexgrid.py`

```python
"""Hexagonal grid substrate with axial coordinates.

Implements 2D hexagonal tiling using axial (q, r) coordinate system.
All hexes have uniform distance to 6 neighbors (no diagonal ambiguity).

**Coordinate System**:
- Axial coordinates: (q, r) where q=horizontal, r=diagonal
- 6 neighbors per hex (vs 4 for square grid)
- Uniform distance to all neighbors
- No "diagonal cheating" problem

**Use Cases**:
- Strategy games (Civilization, Heroes of Might and Magic)
- Natural terrain representation
- Uniform movement cost environments

**Pedagogical Value**:
- Teaches alternative coordinate systems
- Demonstrates coordinate system design choices
- Shows how topology affects distance metrics

References:
- https://www.redblobgames.com/grids/hexagons/
"""
from typing import Literal

import torch

from townlet.substrate.base import SpatialSubstrate


class HexGridSubstrate(SpatialSubstrate):
    """Hexagonal grid substrate with axial coordinates.

    Positions are (q, r) axial coordinates within a radius constraint.
    Actions: 6 directional movements + INTERACT.
    """

    # Hex neighbor offsets (axial coordinates)
    HEX_DIRECTIONS = torch.tensor([
        [+1,  0],  # EAST
        [+1, -1],  # NORTHEAST
        [ 0, -1],  # NORTHWEST
        [-1,  0],  # WEST
        [-1, +1],  # SOUTHWEST
        [ 0, +1],  # SOUTHEAST
    ], dtype=torch.long)

    def __init__(
        self,
        radius: int,
        boundary: Literal["clamp", "wrap"] = "clamp",
        distance_metric: Literal["hex_manhattan", "euclidean"] = "hex_manhattan",
        orientation: Literal["flat_top", "pointy_top"] = "flat_top",
    ):
        """Initialize hexagonal grid substrate.

        Args:
            radius: Maximum distance from origin (hex grid radius)
            boundary: How to handle boundary crossings
                - "clamp": Stay at current hex if move would exit radius
                - "wrap": Wrap to opposite side (toroidal hex grid)
            distance_metric: How to measure distance between hexes
                - "hex_manhattan": Sum of axial coordinate differences (standard)
                - "euclidean": Euclidean distance in axial coordinate space
            orientation: Hex orientation (affects rendering, not logic)
                - "flat_top": Flat edge on top
                - "pointy_top": Pointy vertex on top

        Raises:
            ValueError: If radius <= 0 or invalid boundary/distance_metric
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")

        if boundary not in ["clamp", "wrap"]:
            raise ValueError(f"Unknown boundary: {boundary}")

        if distance_metric not in ["hex_manhattan", "euclidean"]:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        if orientation not in ["flat_top", "pointy_top"]:
            raise ValueError(f"Unknown orientation: {orientation}")

        self.radius = radius
        self.boundary = boundary
        self.distance_metric_name = distance_metric
        self.orientation = orientation

        # Generate valid hex positions within radius
        self._valid_positions = self._generate_valid_positions()

    @property
    def position_dim(self) -> int:
        """Hexagonal grid uses 2D axial coordinates (q, r)."""
        return 2

    @property
    def position_dtype(self) -> torch.dtype:
        """Hex positions are discrete integers."""
        return torch.long

    @property
    def action_space_size(self) -> int:
        """6 directional movements + INTERACT."""
        return 7

    @property
    def valid_positions(self) -> set[tuple[int, ...]]:
        """Set of valid (q, r) positions within radius."""
        return self._valid_positions

    def _generate_valid_positions(self) -> set[tuple[int, int]]:
        """Generate all valid hex positions within radius constraint."""
        positions = set()
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                # Hex is valid if within radius using axial distance
                if abs(q) + abs(r) <= self.radius:
                    positions.add((q, r))
        return positions

    def _is_valid_position(self, q: int, r: int) -> bool:
        """Check if (q, r) is within radius."""
        return abs(q) + abs(r) <= self.radius

    def initialize_positions(
        self, num_agents: int, device: torch.device
    ) -> torch.Tensor:
        """Randomly place agents at valid hex positions.

        Args:
            num_agents: Number of agents to place
            device: Torch device

        Returns:
            [num_agents, 2] tensor of (q, r) positions
        """
        valid_list = list(self._valid_positions)
        indices = torch.randint(0, len(valid_list), (num_agents,), device=device)

        positions = torch.zeros((num_agents, 2), dtype=torch.long, device=device)
        for i, idx in enumerate(indices):
            q, r = valid_list[idx]
            positions[i, 0] = q
            positions[i, 1] = r

        return positions

    def apply_action(
        self, positions: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Apply actions to move agents on hex grid.

        Args:
            positions: [num_agents, 2] current (q, r) positions
            actions: [num_agents] action indices

        Returns:
            [num_agents, 2] new (q, r) positions
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Compute deltas for each action
        deltas = torch.zeros((num_agents, 2), dtype=torch.long, device=device)

        for action_idx in range(6):  # 6 directional movements
            mask = actions == action_idx
            if mask.any():
                direction = self.HEX_DIRECTIONS[action_idx].to(device)
                deltas[mask] = direction

        # INTERACT (action 6) has zero delta (already initialized)

        # Apply deltas
        new_positions = positions + deltas

        # Apply boundary conditions
        if self.boundary == "clamp":
            new_positions = self._apply_clamp_boundary(positions, new_positions)
        elif self.boundary == "wrap":
            new_positions = self._apply_wrap_boundary(new_positions)

        return new_positions

    def _apply_clamp_boundary(
        self, old_positions: torch.Tensor, new_positions: torch.Tensor
    ) -> torch.Tensor:
        """Clamp to valid hex positions (stay put if move would exit radius)."""
        clamped = old_positions.clone()

        for i in range(new_positions.shape[0]):
            q, r = new_positions[i].tolist()
            if self._is_valid_position(q, r):
                clamped[i] = new_positions[i]
            # else: stay at old position

        return clamped

    def _apply_wrap_boundary(self, positions: torch.Tensor) -> torch.Tensor:
        """Wrap to opposite side of hex grid (toroidal topology)."""
        # Toroidal hex wrapping: if outside radius, wrap to opposite side
        wrapped = positions.clone()

        for i in range(positions.shape[0]):
            q, r = positions[i].tolist()
            if not self._is_valid_position(q, r):
                # Simple wrapping: negate coordinates
                wrapped[i, 0] = -q
                wrapped[i, 1] = -r

        return wrapped

    def compute_distance(
        self, pos1: torch.Tensor, pos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between hex positions.

        Args:
            pos1: [batch_size, 2] first positions (q, r)
            pos2: [batch_size, 2] second positions (q, r)

        Returns:
            [batch_size] distances
        """
        if self.distance_metric_name == "hex_manhattan":
            # Hex Manhattan distance in axial coordinates
            dq = torch.abs(pos1[:, 0] - pos2[:, 0])
            dr = torch.abs(pos1[:, 1] - pos2[:, 1])
            ds = torch.abs(dq + dr)  # s = -q - r (cube coordinate)
            return torch.maximum(torch.maximum(dq, dr), ds).float()

        elif self.distance_metric_name == "euclidean":
            # Euclidean distance in axial coordinate space
            diff = pos1 - pos2
            return torch.norm(diff.float(), dim=1)

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric_name}")

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get valid neighboring hex positions.

        Args:
            position: [2] current (q, r) position

        Returns:
            List of [2] neighboring positions within radius
        """
        q, r = position.tolist()
        neighbors = []

        for dq, dr in self.HEX_DIRECTIONS.tolist():
            nq, nr = q + dq, r + dr
            if self._is_valid_position(nq, nr):
                neighbors.append(torch.tensor([nq, nr], dtype=torch.long))

        return neighbors
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_hexgrid.py -v
```

**Expected**: ALL PASS (implementation matches test expectations)

---

### Step 2.3: Add hex config support (1 hour)

**Modify**: `src/townlet/substrate/config.py`

Add hex substrate configuration DTO after existing substrate configs:

```python
@dataclass
class HexGridSubstrateConfig:
    """Configuration for hexagonal grid substrate."""
    type: Literal["hexgrid"]
    radius: int
    boundary: Literal["clamp", "wrap"] = "clamp"
    distance_metric: Literal["hex_manhattan", "euclidean"] = "hex_manhattan"
    orientation: Literal["flat_top", "pointy_top"] = "flat_top"

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("radius must be positive")
```

Update `SubstrateConfig` union to include hex:

```python
SubstrateConfig = Union[
    Grid2DSubstrateConfig,
    Grid3DSubstrateConfig,
    ContinuousSubstrateConfig,
    NDimensionalSubstrateConfig,
    HexGridSubstrateConfig,  # NEW
]
```

**Modify**: `src/townlet/substrate/factory.py`

Add hex substrate creation:

```python
def create_substrate(config: SubstrateConfig) -> SpatialSubstrate:
    """Create substrate from configuration."""
    if config.type == "grid2d":
        from townlet.substrate.grid2d import Grid2DSubstrate
        return Grid2DSubstrate(...)
    # ... existing cases ...
    elif config.type == "hexgrid":
        from townlet.substrate.hexgrid import HexGridSubstrate
        return HexGridSubstrate(
            radius=config.radius,
            boundary=config.boundary,
            distance_metric=config.distance_metric,
            orientation=config.orientation,
        )
    else:
        raise ValueError(f"Unknown substrate type: {config.type}")
```

**Test config loading**:
```bash
python -c "
from townlet.substrate.config import load_substrate_config
config = load_substrate_config({'type': 'hexgrid', 'radius': 5})
print(config)
"
```

**Expected**: Config loads successfully

---

### Step 2.4: Create hex config pack (2 hours)

**Purpose**: Example config for hex grid training

**Create**: `configs/L1_hex_strategy/substrate.yaml`

```yaml
substrate:
  type: hexgrid
  radius: 4
  boundary: clamp
  distance_metric: hex_manhattan
  orientation: flat_top
```

**Create**: `configs/L1_hex_strategy/bars.yaml`, `cascades.yaml`, `affordances.yaml`, `cues.yaml`, `training.yaml`

Copy from `configs/L1_full_observability/` and adjust:

- `training.yaml`: Set `max_steps_per_episode: 300` (smaller hex grid)
- Keep all other configs identical for consistency

**Test config pack loads**:
```bash
python -c "
from townlet.demo.runner import DemoRunner
runner = DemoRunner(
    config_dir='configs/L1_hex_strategy',
    db_path='demo_hex.db',
    checkpoint_dir='checkpoints_hex',
)
print('Config loaded successfully')
"
```

**Expected**: No errors

---

### Step 2.5: Integration test (2 hours)

**Purpose**: Verify hex substrate works in full training loop

**Create**: `tests/test_townlet/integration/test_hexgrid_training.py`

```python
"""Integration test for hexagonal grid training."""
import pytest
import torch

from townlet.substrate.hexgrid import HexGridSubstrate
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_hexgrid_environment_initialization():
    """Hex substrate should work with vectorized environment."""
    substrate = HexGridSubstrate(radius=3, boundary="clamp")

    # Create minimal environment config
    env_config = {
        "num_agents": 4,
        "substrate": substrate,
        # ... minimal config ...
    }

    env = VectorizedHamletEnv(**env_config)

    assert env.num_agents == 4
    assert env.substrate.action_space_size == 7


def test_hexgrid_training_step():
    """Hex environment should handle training step correctly."""
    substrate = HexGridSubstrate(radius=3, boundary="clamp")

    env = VectorizedHamletEnv(
        num_agents=4,
        substrate=substrate,
        # ... minimal config ...
    )

    # Reset environment
    obs = env.reset()
    assert obs.shape[0] == 4  # num_agents

    # Take random actions
    actions = torch.randint(0, 7, (4,))  # 7 possible actions
    obs, rewards, dones, info = env.step(actions)

    assert obs.shape[0] == 4
    assert rewards.shape[0] == 4
    assert dones.shape[0] == 4
```

**Run integration test**:
```bash
uv run pytest tests/test_townlet/integration/test_hexgrid_training.py -v
```

**Expected**: PASS

**Run short training**:
```bash
uv run scripts/run_demo.py --config configs/L1_hex_strategy --max-episodes 10
```

**Expected**: Training runs without errors

---

### Step 2.6: Commit hex substrate (30 min)

```bash
git add src/townlet/substrate/hexgrid.py \
        tests/test_townlet/unit/test_substrate_hexgrid.py \
        tests/test_townlet/integration/test_hexgrid_training.py \
        src/townlet/substrate/config.py \
        src/townlet/substrate/factory.py \
        configs/L1_hex_strategy/

git commit -m "feat(substrate): add hexagonal grid substrate

Implements 2D hexagonal grid with axial coordinates.

**Features**:
- Axial (q, r) coordinate system
- 6-neighbor movement (uniform distances)
- Boundary conditions: clamp, wrap
- Distance metrics: hex_manhattan, euclidean

**Use Cases**:
- Strategy games (Civilization, Heroes of Might and Magic)
- Natural terrain representation
- Uniform movement cost environments

**Testing**:
- 12 unit tests (all passing)
- Integration test with vectorized environment
- Example config pack: L1_hex_strategy

**Pedagogical Value**:
- Teaches alternative coordinate systems
- Demonstrates coordinate system design choices
- Shows how topology affects distance metrics

Part of TASK-002A Phase 9 Task 2 (Hexagonal Grid - IMPLEMENT SECOND)."
```

---

## Task 9.1: 1D Grid Substrate (6-8 hours) - IMPLEMENT FIRST

### Overview

Simplest spatial substrate - linear 1D grid with scalar positions. Only 2 directional actions (LEFT/RIGHT). Edge case validation for N=1 dimension.

**Why More Complex Than Expected?**

Original estimate: 2-4 hours. Revised to 6-8 hours after risk assessment due to:
- Action mapping currently hardcoded for 2D (4 directions)
- Will expose ANY hardcoded 2D assumptions in environment (VALUABLE cleanup!)
- Need to add 1D action mapping edge case (2 directions)
- Q-network needs action_dim=3 (not 5)
- Budget includes debugging time for surfacing/fixing 2D assumptions

**Files to Create**:
- `src/townlet/substrate/grid1d.py` (new)
- `tests/test_townlet/unit/test_substrate_grid1d.py` (new)
- `configs/L0_1D_line/` (new config pack)

**Files to Modify**:
- `src/townlet/substrate/config.py` (add 1D config parsing)
- `src/townlet/substrate/factory.py` (wire up 1D substrate)
- `src/townlet/environment/vectorized_env.py` (fix action mapping for 1D)

---

### Step 1.1: Write unit tests for Grid1DSubstrate (1 hour)

**Create**: `tests/test_townlet/unit/test_substrate_grid1d.py`

```python
"""Unit tests for Grid1DSubstrate."""
import pytest
import torch
from townlet.substrate.grid1d import Grid1DSubstrate


def test_grid1d_initialization():
    """1D grid should initialize with valid parameters."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")

    assert substrate.length == 10
    assert substrate.position_dim == 1
    assert substrate.position_dtype == torch.long
    assert substrate.action_space_size == 3  # LEFT, RIGHT, INTERACT


def test_grid1d_invalid_initialization():
    """1D grid should reject invalid parameters."""
    with pytest.raises(ValueError, match="Length must be positive"):
        Grid1DSubstrate(length=0, boundary="clamp")


def test_grid1d_initialize_positions():
    """1D grid should place agents at random positions."""
    substrate = Grid1DSubstrate(length=20, boundary="clamp")

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    assert positions.shape == (10, 1)
    assert positions.dtype == torch.long
    assert (positions >= 0).all()
    assert (positions < 20).all()


def test_grid1d_apply_action_movement():
    """1D grid should apply LEFT/RIGHT movement correctly."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")

    start = torch.tensor([[5]], dtype=torch.long)

    # Move LEFT (action 0)
    move_left = torch.tensor([0], dtype=torch.long)
    new_pos = substrate.apply_action(start, move_left)
    assert new_pos[0, 0].item() == 4

    # Move RIGHT (action 1)
    move_right = torch.tensor([1], dtype=torch.long)
    new_pos = substrate.apply_action(start, move_right)
    assert new_pos[0, 0].item() == 6


def test_grid1d_apply_action_interact():
    """INTERACT action should not move agent."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")

    start = torch.tensor([[5]], dtype=torch.long)
    interact = torch.tensor([2], dtype=torch.long)  # INTERACT

    new_pos = substrate.apply_action(start, interact)
    assert torch.equal(new_pos, start)


def test_grid1d_boundary_clamp():
    """Clamping should prevent movement beyond edges."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")

    # At left edge
    left_edge = torch.tensor([[0]], dtype=torch.long)
    move_left = torch.tensor([0], dtype=torch.long)
    clamped = substrate.apply_action(left_edge, move_left)
    assert clamped[0, 0].item() == 0  # Stay at 0

    # At right edge
    right_edge = torch.tensor([[9]], dtype=torch.long)
    move_right = torch.tensor([1], dtype=torch.long)
    clamped = substrate.apply_action(right_edge, move_right)
    assert clamped[0, 0].item() == 9  # Stay at 9


def test_grid1d_boundary_wrap():
    """Wrapping should teleport to opposite edge."""
    substrate = Grid1DSubstrate(length=10, boundary="wrap")

    # Wrap left
    left_edge = torch.tensor([[0]], dtype=torch.long)
    move_left = torch.tensor([0], dtype=torch.long)
    wrapped = substrate.apply_action(left_edge, move_left)
    assert wrapped[0, 0].item() == 9  # Wrap to right edge

    # Wrap right
    right_edge = torch.tensor([[9]], dtype=torch.long)
    move_right = torch.tensor([1], dtype=torch.long)
    wrapped = substrate.apply_action(right_edge, move_right)
    assert wrapped[0, 0].item() == 0  # Wrap to left edge


def test_grid1d_compute_distance():
    """Distance should be absolute difference."""
    substrate = Grid1DSubstrate(length=20, boundary="clamp")

    pos1 = torch.tensor([[5]], dtype=torch.long)
    pos2 = torch.tensor([[10]], dtype=torch.long)

    distance = substrate.compute_distance(pos1, pos2)
    assert distance.item() == 5.0


def test_grid1d_get_valid_neighbors():
    """1D grid should return left and right neighbors."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")

    center = torch.tensor([5], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(center)

    assert len(neighbors) == 2
    assert neighbors[0][0].item() == 4  # LEFT
    assert neighbors[1][0].item() == 6  # RIGHT


def test_grid1d_edge_neighbors():
    """Edge positions should have only 1 neighbor."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")

    # Left edge
    left = torch.tensor([0], dtype=torch.long)
    left_neighbors = substrate.get_valid_neighbors(left)
    assert len(left_neighbors) == 1
    assert left_neighbors[0][0].item() == 1

    # Right edge
    right = torch.tensor([9], dtype=torch.long)
    right_neighbors = substrate.get_valid_neighbors(right)
    assert len(right_neighbors) == 1
    assert right_neighbors[0][0].item() == 8
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_grid1d.py -v
```

**Expected**: ALL FAIL (not implemented yet)

---

### Step 1.2: Implement Grid1DSubstrate (2-3 hours)

**Create**: `src/townlet/substrate/grid1d.py`

```python
"""1D grid substrate (linear spatial topology).

Simplest spatial case - scalar positions on a line segment.
Only 2 directional movements (LEFT/RIGHT) + INTERACT.

**Use Cases**:
- Pedagogical progression (simplest spatial case)
- Conveyor belt simulations
- Linear sequences (gene sequences, timelines)

**Pedagogical Value**:
- Teaches dimensionality concepts (N=1 edge case)
- Simplest example of spatial substrate
- Gateway to understanding 2D/3D grids
"""
from typing import Literal

import torch

from townlet.substrate.base import SpatialSubstrate


class Grid1DSubstrate(SpatialSubstrate):
    """1D grid substrate with scalar positions.

    Positions are integers [0, length).
    Actions: LEFT, RIGHT, INTERACT.
    """

    def __init__(
        self,
        length: int,
        boundary: Literal["clamp", "wrap"] = "clamp",
        distance_metric: Literal["manhattan", "euclidean"] = "manhattan",
    ):
        """Initialize 1D grid substrate.

        Args:
            length: Grid length (number of positions)
            boundary: How to handle boundary crossings
                - "clamp": Stay at edge if move would exit grid
                - "wrap": Wrap to opposite edge (circular topology)
            distance_metric: Distance measure (both equivalent for 1D)

        Raises:
            ValueError: If length <= 0
        """
        if length <= 0:
            raise ValueError("Length must be positive")

        if boundary not in ["clamp", "wrap"]:
            raise ValueError(f"Unknown boundary: {boundary}")

        self.length = length
        self.boundary = boundary
        self.distance_metric_name = distance_metric

    @property
    def position_dim(self) -> int:
        """1D grid uses scalar positions."""
        return 1

    @property
    def position_dtype(self) -> torch.dtype:
        """Positions are discrete integers."""
        return torch.long

    @property
    def action_space_size(self) -> int:
        """LEFT, RIGHT, INTERACT = 3 actions."""
        return 3

    @property
    def valid_positions(self) -> set[tuple[int, ...]]:
        """All positions [0, length) are valid."""
        return {(i,) for i in range(self.length)}

    def initialize_positions(
        self, num_agents: int, device: torch.device
    ) -> torch.Tensor:
        """Randomly place agents on 1D line.

        Args:
            num_agents: Number of agents
            device: Torch device

        Returns:
            [num_agents, 1] tensor of positions
        """
        positions = torch.randint(
            0, self.length, (num_agents, 1), dtype=torch.long, device=device
        )
        return positions

    def apply_action(
        self, positions: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Apply actions to move agents on 1D line.

        Args:
            positions: [num_agents, 1] current positions
            actions: [num_agents] action indices
                - 0: LEFT  (position -= 1)
                - 1: RIGHT (position += 1)
                - 2: INTERACT (no movement)

        Returns:
            [num_agents, 1] new positions
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Compute deltas
        deltas = torch.zeros((num_agents, 1), dtype=torch.long, device=device)
        deltas[actions == 0] = -1  # LEFT
        deltas[actions == 1] = +1  # RIGHT
        # INTERACT (action 2) has zero delta

        # Apply deltas
        new_positions = positions + deltas

        # Apply boundary conditions
        if self.boundary == "clamp":
            new_positions = torch.clamp(new_positions, 0, self.length - 1)
        elif self.boundary == "wrap":
            new_positions = new_positions % self.length

        return new_positions

    def compute_distance(
        self, pos1: torch.Tensor, pos2: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between positions.

        Args:
            pos1: [batch_size, 1] first positions
            pos2: [batch_size, 1] second positions

        Returns:
            [batch_size] distances (absolute difference)
        """
        return torch.abs(pos1 - pos2).squeeze(1).float()

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get valid neighboring positions (left and right).

        Args:
            position: [1] current position

        Returns:
            List of [1] neighboring positions within bounds
        """
        pos = position[0].item()
        neighbors = []

        # Left neighbor
        if pos > 0:
            neighbors.append(torch.tensor([pos - 1], dtype=torch.long))

        # Right neighbor
        if pos < self.length - 1:
            neighbors.append(torch.tensor([pos + 1], dtype=torch.long))

        return neighbors
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_grid1d.py -v
```

**Expected**: ALL PASS

---

### Step 1.3: Add config support and create config pack (2-3 hours)

**Modify**: `src/townlet/substrate/config.py`

```python
@dataclass
class Grid1DSubstrateConfig:
    """Configuration for 1D grid substrate."""
    type: Literal["grid1d"]
    length: int
    boundary: Literal["clamp", "wrap"] = "clamp"
    distance_metric: Literal["manhattan", "euclidean"] = "manhattan"

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError("length must be positive")
```

Update union:
```python
SubstrateConfig = Union[
    Grid2DSubstrateConfig,
    Grid3DSubstrateConfig,
    ContinuousSubstrateConfig,
    NDimensionalSubstrateConfig,
    HexGridSubstrateConfig,
    Grid1DSubstrateConfig,  # NEW
]
```

**Modify**: `src/townlet/substrate/factory.py`

```python
elif config.type == "grid1d":
    from townlet.substrate.grid1d import Grid1DSubstrate
    return Grid1DSubstrate(
        length=config.length,
        boundary=config.boundary,
        distance_metric=config.distance_metric,
    )
```

**Create**: `configs/L0_1D_line/substrate.yaml`

```yaml
substrate:
  type: grid1d
  length: 15
  boundary: clamp
  distance_metric: manhattan
```

Copy other configs from L0_0_minimal and adjust for 1D:

**Test training**:
```bash
uv run scripts/run_demo.py --config configs/L0_1D_line --max-episodes 10
```

**Expected**: Training runs successfully

**Commit**:
```bash
git add src/townlet/substrate/grid1d.py \
        tests/test_townlet/unit/test_substrate_grid1d.py \
        src/townlet/substrate/config.py \
        src/townlet/substrate/factory.py \
        configs/L0_1D_line/

git commit -m "feat(substrate): add 1D grid substrate

Implements linear 1D spatial topology with scalar positions.

**Features**:
- Scalar position [0, length)
- 2-directional movement (LEFT/RIGHT) + INTERACT
- Boundary conditions: clamp, wrap
- Simplest spatial substrate

**Use Cases**:
- Pedagogical progression (N=1 edge case)
- Conveyor belt simulations
- Linear sequences

**Testing**:
- 11 unit tests (all passing)
- Example config pack: L0_1D_line

**Pedagogical Value**:
- Gateway to understanding spatial substrates
- Demonstrates dimensionality concepts
- Edge case validation (N=1)

Part of TASK-002A Phase 9 Task 1 (1D Grid - IMPLEMENT FIRST)."
```

---

## Task 9.3: Documentation & Examples (1-2 hours)

### Step 3.1: Update CLAUDE.md (1 hour)

**Modify**: `CLAUDE.md`

Add section on Phase 9 topologies:

```markdown
### Phase 9: Simple Alternative Topologies

**Hexagonal Grid** (configs/L1_hex_strategy/):
- 2D hexagonal tiling with axial coordinates
- 6-neighbor movement (uniform distances)
- Use cases: Strategy games, natural terrain
- Action space: 7 (6 directions + INTERACT)

**1D Grid** (configs/L0_1D_line/):
- Linear 1D grid (simplest spatial case)
- 2-neighbor movement (LEFT/RIGHT)
- Use cases: Pedagogical progression, conveyor belts
- Action space: 3 (LEFT, RIGHT, INTERACT)

**Key Concepts**:
- **Hex Grid**: Teaches coordinate system design (axial coords)
- **1D Grid**: Edge case validation (N=1 dimension)

**Note**: Graph-Based substrate deferred to Phase 10 (requires infrastructure changes).
```

---

### Step 3.2: Regression testing (1 hour) - ADDED POST-RISK-ASSESSMENT

**Purpose**: Verify that 1D Grid changes didn't break existing substrates

**Action**: Run full test suite to ensure no regressions

**Test existing substrates**:
```bash
# Test 2D Grid (baseline)
uv run pytest tests/test_townlet/unit/test_substrate_grid2d.py -v

# Test 3D Grid
uv run pytest tests/test_townlet/unit/test_substrate_grid3d.py -v

# Test N-Dimensional
uv run pytest tests/test_townlet/unit/test_substrate_ndimensional.py -v

# Run quick training test on 2D to ensure no environment breakage
uv run scripts/run_demo.py --config configs/L1_full_observability --max-episodes 5
```

**Expected**: All existing substrate tests PASS, 2D training runs without errors

**Why This Matters**: 1D Grid may have required fixes to hardcoded 2D assumptions. This step verifies those fixes didn't break existing functionality.

---

### Step 3.3: Commit documentation (30 min)

```bash
git add CLAUDE.md

git commit -m "docs: add Phase 9 topology documentation

Added documentation for two new topologies:
- Hexagonal Grid: Axial coordinates, uniform distances
- 1D Grid: Linear movement, pedagogical progression

Phase 9 focuses on simple topologies that do not require
infrastructure changes (Graph substrate deferred to Phase 10).

Part of TASK-002A Phase 9 (Simple Alternative Topologies)."
```

---

## Task 9.4: Frontend Visualization (3-4 hours) - ADDED POST-RISK-ASSESSMENT

**⚠️ CRITICAL ADDITION**: Risk assessment identified frontend visualization as essential for debugging and pedagogy. This task was completely missing from original plan.

### Overview

Add frontend rendering support for Hex Grid and 1D Grid substrates. Without visualization, debugging coordinate systems is nearly impossible and pedagogical value is lost.

**Files to Create**:
- `frontend/src/components/HexGridVisualization.vue` (new)
- `frontend/src/components/Grid1DVisualization.vue` (new)

**Files to Modify**:
- `frontend/src/App.vue` (add substrate type switching)
- `frontend/src/websocket.js` (handle hex/1D position data)

---

### Step 4.1: Hex Grid visualization component (2-2.5 hours)

**Purpose**: Render hexagonal grid with axial coordinates

**Create**: `frontend/src/components/HexGridVisualization.vue`

**Key Features**:
- Axial→pixel coordinate conversion (using Red Blob Games formulas)
- SVG polygon rendering for hexagons
- Agent position rendering on hex grid
- Affordance placement on hexes

**Axial to Pixel Conversion** (flat-top orientation):
```javascript
function axialToPixel(q, r, hexSize) {
  const x = hexSize * (3/2 * q);
  const y = hexSize * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
  return { x, y };
}

function hexPolygonPoints(centerX, centerY, size) {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angleDeg = 60 * i;
    const angleRad = Math.PI / 180 * angleDeg;
    points.push({
      x: centerX + size * Math.cos(angleRad),
      y: centerY + size * Math.sin(angleRad)
    });
  }
  return points;
}
```

**Test**: Load L1_hex_strategy config and verify hex grid renders correctly

---

### Step 4.2: 1D Grid visualization component (1-1.5 hours)

**Purpose**: Render linear 1D grid

**Create**: `frontend/src/components/Grid1DVisualization.vue`

**Key Features**:
- Linear horizontal layout
- Agent position as circle on line
- Affordances as markers on line segments
- Position labels (0, 1, 2, ..., N-1)

**Simple Layout**:
```javascript
function positionToPixel(pos, length, containerWidth) {
  const spacing = containerWidth / length;
  return spacing * pos + spacing / 2;
}
```

**Test**: Load L0_1D_line config and verify linear grid renders correctly

---

### Step 4.3: Integration and commit (30 min)

**Modify**: `frontend/src/App.vue`

Add substrate type detection:
```javascript
computed: {
  substrateType() {
    // Detect from config or WebSocket data
    if (this.config.substrate.type === 'hexgrid') return 'hex';
    if (this.config.substrate.type === 'grid1d') return '1d';
    if (this.config.substrate.type === 'grid2d') return '2d';
    // ... etc
  }
}
```

Conditionally render components:
```vue
<HexGridVisualization v-if="substrateType === 'hex'" :agents="agents" />
<Grid1DVisualization v-if="substrateType === '1d'" :agents="agents" />
<GridVisualization v-else :agents="agents" />  <!-- 2D fallback -->
```

**Commit**:
```bash
git add frontend/src/components/HexGridVisualization.vue \
        frontend/src/components/Grid1DVisualization.vue \
        frontend/src/App.vue \
        frontend/src/websocket.js

git commit -m "feat(frontend): add Hex Grid and 1D Grid visualization

Essential for debugging and pedagogical value.

**Hex Grid Visualization**:
- Axial→pixel coordinate conversion
- SVG polygon rendering for hexagons
- Supports L1_hex_strategy config pack

**1D Grid Visualization**:
- Linear horizontal layout
- Agent position rendering on line
- Supports L0_1D_line config pack

**Integration**:
- Automatic substrate type detection
- Conditional component rendering

Part of TASK-002A Phase 9 Task 4 (Frontend Visualization)."
```

---

## Phase 9 Completion Checklist (UPDATED POST-RISK-ASSESSMENT)

**⚠️ IMPLEMENT IN ORDER**: Task 9.1 (1D) → 9.2 (Hex) → 9.3 (Docs) → 9.4 (Frontend)

### Task 9.1: 1D Substrate (6-8 hours) - FIRST
- [ ] Step 1.1: Unit tests (1h)
- [ ] Step 1.2: Implement Grid1DSubstrate (2-3h)
- [ ] Step 1.3: Config & commit (2-3h)
- [ ] Verify: action_dim=3, no 2D hardcodes remaining

### Task 9.2: Hex Substrate (10-12 hours) - SECOND
- [ ] Step 2.1: Unit tests (1h)
- [ ] Step 2.2: Implement HexGridSubstrate (4-5h)
- [ ] Step 2.3: Config support (1h)
- [ ] Step 2.4: Config pack (2h)
- [ ] Step 2.5: Integration test (2h)
- [ ] Step 2.6: Commit (30min)

### Task 9.3: Documentation (2-2.5 hours) - THIRD
- [ ] Step 3.1: Update CLAUDE.md (1h)
- [ ] Step 3.2: Regression testing (1h) - NEW
- [ ] Step 3.3: Commit docs (30min)

### Task 9.4: Frontend Visualization (3-4 hours) - FOURTH (NEW)
- [ ] Step 4.1: Hex Grid component (2-2.5h)
- [ ] Step 4.2: 1D Grid component (1-1.5h)
- [ ] Step 4.3: Integration & commit (30min)

**Total Estimated Effort**: 20-26 hours (revised from 13-17h after risk assessment)
**Delta from Original**: +7-9 hours (frontend +3-4h, 1D debugging +1-2h, regression +1h, estimates +2-3h)

---

## Success Criteria (UPDATED)

Phase 9 is complete when:

**Task 9.1: 1D Substrate**
- [ ] All unit tests pass (11 tests)
- [ ] Training runs on 1D line
- [ ] Config pack L0_1D_line created
- [ ] Q-network properly sizes to action_dim=3
- [ ] No hardcoded 2D assumptions remain in environment

**Task 9.2: Hex Substrate**
- [ ] All unit tests pass (12 tests)
- [ ] Integration test passes
- [ ] Training runs on hex grid
- [ ] Config pack L1_hex_strategy created
- [ ] Axial coordinate math validated

**Task 9.3: Documentation**
- [ ] CLAUDE.md updated with both substrates
- [ ] Regression tests PASS (2D, 3D, ND still work)
- [ ] Documentation complete

**Task 9.4: Frontend Visualization**
- [ ] Hex grid renders correctly with axial coords
- [ ] 1D grid renders correctly as linear layout
- [ ] Substrate type detection works
- [ ] Both visualizations tested with live inference

**Overall Integration:**
- [ ] No regressions in existing substrates (2D, 3D, ND)
- [ ] All four topologies coexist peacefully
- [ ] Full test suite passes

---

**Phase 9 Status**: Ready for Implementation
**Implementation Order**: 1D → Hex → Docs → Frontend (revised from Hex→1D)
**Total Effort**: 20-26 hours (revised from 13-17h post-risk-assessment)
**Next Phase**: Phase 10 (Graph substrate + infrastructure prerequisites)
