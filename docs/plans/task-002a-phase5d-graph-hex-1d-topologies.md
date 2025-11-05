# TASK-002A Phase 5D: Alternative Topologies (Graph, Hex, 1D) - Implementation Plan

**Date**: 2025-11-05
**Status**: Ready for Implementation
**Dependencies**: Phase 5C Complete (N-Dimensional Substrates, Observation Encoding Retrofit)
**Estimated Effort**: 33-45 hours total

---

## Executive Summary

Phase 5D adds three alternative topologies to the configurable spatial substrate system:

1. **Hexagonal Grid** (8-10 hours) - Uniform distances, axial coordinates, strategy games
2. **1D Grid** (2-4 hours) - Linear movement, simplest spatial case, pedagogical progression
3. **Graph-Based** (18-24 hours) - Variable action spaces, graph RL, non-Euclidean reasoning

**Implementation Order**: Hex → 1D → Graph (build confidence with simpler topologies first)

**Key Technical Challenges**:
- **Hex**: Axial coordinate system math (6 neighbors, uniform distances)
- **1D**: Edge case validation (N=1 dimension)
- **Graph**: Action masking infrastructure (variable action spaces per node)

**Pedagogical Value**:
- **Hex** (✅✅): Teaches coordinate system design, uniform distance metrics
- **1D** (✅): Teaches dimensionality concepts, edge case reasoning
- **Graph** (✅✅✅): Teaches graph RL, topological reasoning, action masking

---

## Task 5D.1: Hexagonal Grid Substrate (8-10 hours)

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

### Step 1.1: Write unit tests for HexGridSubstrate (1 hour)

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
    assert (-2, 0) in valid_positions
    assert (0, -2) in valid_positions

    # Corners should be valid
    assert (1, 1) in valid_positions
    assert (-1, -1) in valid_positions

    # Out-of-bounds should NOT be valid
    assert (3, 0) not in valid_positions
    assert (0, 3) not in valid_positions


def test_hexgrid_initialize_positions():
    """Hex grid should randomly place agents on valid positions."""
    substrate = HexGridSubstrate(radius=3, boundary="clamp")

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    # Shape: [10, 2] (10 agents, 2D axial coords)
    assert positions.shape == (10, 2)
    assert positions.dtype == torch.long

    # All positions should be valid
    for i in range(10):
        q, r = positions[i, 0].item(), positions[i, 1].item()
        assert (q, r) in substrate.valid_positions


def test_hexgrid_distance_hex_manhattan():
    """Hex manhattan distance should be correct."""
    substrate = HexGridSubstrate(radius=5, boundary="clamp", distance_metric="hex_manhattan")

    # Test distances from origin
    pos1 = torch.tensor([[0, 0]], dtype=torch.long)  # Origin
    pos2 = torch.tensor([1, 0], dtype=torch.long)    # 1 hex east

    distance = substrate.compute_distance(pos1, pos2)
    assert distance.item() == 1.0

    # Distance to (2, 0) should be 2
    pos2 = torch.tensor([2, 0], dtype=torch.long)
    distance = substrate.compute_distance(pos1, pos2)
    assert distance.item() == 2.0

    # Distance to (1, 1) should be 2 (not √2!)
    pos2 = torch.tensor([1, 1], dtype=torch.long)
    distance = substrate.compute_distance(pos1, pos2)
    assert distance.item() == 2.0


def test_hexgrid_movement():
    """Hex grid should move agents along 6 hex directions."""
    substrate = HexGridSubstrate(radius=5, boundary="clamp")

    # Start at origin
    positions = torch.tensor([[0, 0]], dtype=torch.long)

    # Move EAST (direction 0: +1, 0)
    deltas = torch.tensor([[1, 0]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert (new_positions == torch.tensor([[1, 0]])).all()

    # Move NORTHEAST (direction 1: +1, -1)
    deltas = torch.tensor([[1, -1]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert (new_positions == torch.tensor([[1, -1]])).all()


def test_hexgrid_boundary_clamp():
    """Hex grid should clamp to valid positions at boundary."""
    substrate = HexGridSubstrate(radius=2, boundary="clamp")

    # Start at edge (2, 0)
    positions = torch.tensor([[2, 0]], dtype=torch.long)

    # Try to move EAST (would go out of bounds)
    deltas = torch.tensor([[1, 0]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)

    # Should stay at (2, 0) since (3, 0) is invalid
    assert (new_positions == torch.tensor([[2, 0]])).all()


def test_hexgrid_get_valid_neighbors():
    """Hex grid should return 6 neighbors (or fewer at boundary)."""
    substrate = HexGridSubstrate(radius=3, boundary="clamp")

    # Center hex has 6 neighbors
    position = torch.tensor([0, 0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)
    assert len(neighbors) == 6

    # Edge hex has fewer neighbors
    position = torch.tensor([3, 0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)
    assert len(neighbors) < 6  # Some directions go out of bounds


def test_hexgrid_encode_observation():
    """Hex grid should encode positions as normalized coordinates."""
    substrate = HexGridSubstrate(radius=4, boundary="clamp")

    # Position at origin (0, 0) → normalized to (0.5, 0.5)
    positions = torch.tensor([[0, 0]], dtype=torch.long)
    encoded = substrate.encode_observation(positions, {})

    assert encoded.shape == (1, 2)
    assert torch.allclose(encoded, torch.tensor([[0.5, 0.5]]), atol=0.01)

    # Position at (4, 0) → normalized to (1.0, 0.5)
    positions = torch.tensor([[4, 0]], dtype=torch.long)
    encoded = substrate.encode_observation(positions, {})
    assert torch.allclose(encoded[0, 0], torch.tensor(1.0), atol=0.01)


def test_hexgrid_get_all_positions():
    """Hex grid should return all valid hex positions."""
    substrate = HexGridSubstrate(radius=1, boundary="clamp")

    all_positions = substrate.get_all_positions()

    # Radius 1 has 7 hexes
    assert len(all_positions) == 7

    # Should include center and 6 neighbors
    assert [0, 0] in all_positions
    assert [1, 0] in all_positions
    assert [-1, 0] in all_positions


def test_hexgrid_is_on_position():
    """Hex grid should detect if agent is on target position."""
    substrate = HexGridSubstrate(radius=3, boundary="clamp")

    agent_positions = torch.tensor([[1, 1], [2, 0], [0, 2]], dtype=torch.long)
    target_position = torch.tensor([2, 0], dtype=torch.long)

    on_position = substrate.is_on_position(agent_positions, target_position)

    # Only agent 1 is at (2, 0)
    assert on_position[0] == False
    assert on_position[1] == True
    assert on_position[2] == False


def test_hexgrid_observation_dim():
    """Hex grid observation dim should be 2 (normalized q, r)."""
    substrate = HexGridSubstrate(radius=5, boundary="clamp")
    assert substrate.get_observation_dim() == 2
```

**Run tests**:
```bash
cd /home/user/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_hexgrid.py -v
```

**Expected**: ALL tests FAIL (HexGridSubstrate not yet implemented)

---

### Step 1.2: Implement HexGridSubstrate class (3-4 hours)

**Purpose**: Implement hex grid substrate to pass all tests

**Action**: Create substrate with axial coordinate math

**Create**: `src/townlet/substrate/hexgrid.py`

```python
"""Hexagonal grid substrate with axial coordinates."""

from typing import Literal
import torch
from .base import SpatialSubstrate


class HexGridSubstrate(SpatialSubstrate):
    """Hexagonal grid substrate with axial coordinates (q, r).

    Uses axial coordinate system where:
    - q axis points "east"
    - r axis points "southeast"
    - Uniform distance to all 6 neighbors

    Movement actions: 6 directions (E, NE, NW, W, SW, SE) + INTERACT = 7 total

    Distance metrics:
    - hex_manhattan: max(|q1-q2|, |r1-r2|, |q1-q2 + r1-r2|)  [RECOMMENDED]
    - cube_manhattan: (|x1-x2| + |y1-y2| + |z1-z2|) / 2  [Alternative]

    Key Properties:
    - All 6 neighbors are equidistant (distance = 1)
    - No diagonal movement ambiguity (unlike square grids)
    - Grid shape is hexagonal (not rectangular)
    - Position validation required (not all (q, r) are valid)

    Pedagogical Value:
    - Teaches coordinate system design (alternatives to Cartesian)
    - Teaches uniform distance metrics
    - Used for strategy games, natural terrain, biological simulations

    References:
    - Red Blob Games Hex Guide: https://www.redblobgames.com/grids/hexagons/
    """

    position_dim = 2  # Axial coordinates (q, r)
    position_dtype = torch.long

    # Hex direction vectors (axial coordinates)
    # Order: EAST, NORTHEAST, NORTHWEST, WEST, SOUTHWEST, SOUTHEAST
    HEX_DIRECTIONS = torch.tensor([
        [+1,  0],  # EAST (action 0)
        [+1, -1],  # NORTHEAST (action 1)
        [ 0, -1],  # NORTHWEST (action 2)
        [-1,  0],  # WEST (action 3)
        [-1, +1],  # SOUTHWEST (action 4)
        [ 0, +1],  # SOUTHEAST (action 5)
    ], dtype=torch.long)

    def __init__(
        self,
        radius: int,
        boundary: Literal["clamp"] = "clamp",  # "wrap" deferred
        distance_metric: Literal["hex_manhattan", "cube_manhattan"] = "hex_manhattan",
        orientation: Literal["flat_top", "pointy_top"] = "flat_top",
    ):
        """Initialize hexagonal grid substrate.

        Args:
            radius: Hex grid radius (q, r range: [-radius, radius])
            boundary: Boundary handling (only "clamp" supported initially)
            distance_metric: Distance calculation method
            orientation: Hex tile orientation (for visualization, doesn't affect logic)

        Raises:
            ValueError: If radius <= 0 or boundary/distance_metric invalid
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive: {radius}")
        if boundary not in ("clamp",):  # Only clamp for Phase 5D
            raise ValueError(f"Unknown boundary mode: {boundary} (only 'clamp' supported)")
        if distance_metric not in ("hex_manhattan", "cube_manhattan"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        if orientation not in ("flat_top", "pointy_top"):
            raise ValueError(f"Unknown orientation: {orientation}")

        self.radius = radius
        self.boundary = boundary
        self.distance_metric = distance_metric
        self.orientation = orientation

        # Precompute valid hex positions (hex grid is not rectangular!)
        self.valid_positions = self._generate_valid_positions()

    @property
    def action_space_size(self) -> int:
        """6 movement directions + 1 INTERACT = 7 actions."""
        return 7

    def _generate_valid_positions(self) -> set[tuple[int, int]]:
        """Generate all valid (q, r) positions within radius.

        Hex grid constraint: |q| + |r| + |q + r| <= 2 * radius
        (Equivalent to: hex is within radius steps from origin)

        Returns:
            Set of (q, r) tuples representing valid hex positions
        """
        valid = set()
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                # Hex grid constraint (cube coordinate sum)
                # Convert to cube: x=q, z=r, y=-q-r
                # Constraint: max(|x|, |y|, |z|) <= radius
                s = -q - r  # Third cube coordinate
                if max(abs(q), abs(r), abs(s)) <= self.radius:
                    valid.add((q, r))
        return valid

    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Randomly place agents on valid hex positions.

        Args:
            num_agents: Number of agents to place
            device: PyTorch device

        Returns:
            [num_agents, 2] tensor of axial coordinates
        """
        valid_list = list(self.valid_positions)
        indices = torch.randint(0, len(valid_list), (num_agents,), device=device)
        positions = torch.tensor(
            [valid_list[idx] for idx in indices],
            device=device,
            dtype=torch.long
        )
        return positions

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply movement with hex-aware boundary handling.

        Args:
            positions: [num_agents, 2] current axial coordinates
            deltas: [num_agents, 2] movement deltas

        Returns:
            [num_agents, 2] new axial coordinates after boundary handling
        """
        new_positions = positions + deltas

        if self.boundary == "clamp":
            # Clamp to valid hex positions
            for i in range(new_positions.shape[0]):
                q, r = new_positions[i, 0].item(), new_positions[i, 1].item()
                if (q, r) not in self.valid_positions:
                    # Stay at current position if out of bounds
                    new_positions[i] = positions[i]

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute hex distance between positions.

        Args:
            pos1: [num_agents, 2] agent positions (axial)
            pos2: [2] target position (axial)

        Returns:
            [num_agents] distances (number of hex steps)
        """
        if self.distance_metric == "hex_manhattan":
            # Axial hex distance: max(|Δq|, |Δr|, |Δq + Δr|)
            dq = torch.abs(pos1[:, 0] - pos2[0])
            dr = torch.abs(pos1[:, 1] - pos2[1])
            ds = torch.abs(dq + dr)
            return torch.max(torch.max(dq, dr), ds).float()

        elif self.distance_metric == "cube_manhattan":
            # Cube coordinate distance: (|Δx| + |Δy| + |Δz|) / 2
            # Convert axial → cube
            x1, z1 = pos1[:, 0], pos1[:, 1]
            y1 = -x1 - z1

            x2, z2 = pos2[0], pos2[1]
            y2 = -x2 - z2

            return (torch.abs(x1 - x2) + torch.abs(y1 - y2) + torch.abs(z1 - z2)).float() / 2

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode hex positions as normalized coordinates.

        Normalizes q, r to [0, 1] based on grid radius.
        Range [-radius, radius] → [0, 1]

        Args:
            positions: [num_agents, 2] axial coordinates
            affordances: Dict of affordance positions (reserved for future)

        Returns:
            [num_agents, 2] normalized coordinates in [0, 1]
        """
        normalized = positions.float() / (2 * self.radius)  # [-radius, radius] → [-0.5, 0.5]
        normalized += 0.5  # Shift to [0, 1]
        return normalized

    def get_observation_dim(self) -> int:
        """Hex position encoding is 2-dimensional (normalized q, r)."""
        return 2

    def get_all_positions(self) -> list[list[int]]:
        """Return all valid hex positions.

        Returns:
            List of [q, r] positions for all valid hexes
        """
        return [[q, r] for q, r in self.valid_positions]

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get 6 hex neighbors (or fewer if at boundary).

        Args:
            position: [2] axial coordinates

        Returns:
            List of neighbor position tensors (up to 6 neighbors)
        """
        q, r = position[0].item(), position[1].item()
        neighbors = []

        for dq, dr in self.HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if (nq, nr) in self.valid_positions:
                neighbors.append(torch.tensor([nq, nr], dtype=torch.long))

        return neighbors

    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor
    ) -> torch.Tensor:
        """Check if agents are on target hex (exact match).

        Args:
            agent_positions: [num_agents, 2] agent positions
            target_position: [2] target position

        Returns:
            [num_agents] boolean tensor (True if on position)
        """
        return (agent_positions == target_position).all(dim=1)

    # Visualization helper (optional, for frontend)
    def to_pixel_coords(self, q: int, r: int, size: float = 1.0) -> tuple[float, float]:
        """Convert axial hex coords to pixel coords for rendering.

        Args:
            q: Axial q coordinate
            r: Axial r coordinate
            size: Hex size multiplier

        Returns:
            (x, y) pixel coordinates for hex center
        """
        import math

        if self.orientation == "flat_top":
            x = size * (3/2 * q)
            y = size * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
        else:  # "pointy_top"
            x = size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
            y = size * (3/2 * r)
        return (x, y)
```

**Verify implementation**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_hexgrid.py -v
```

**Expected**: ALL tests PASS

---

### Step 1.3: Add hex configuration support (1 hour)

**Purpose**: Wire up hex substrate to config system

**Action**: Extend config schema and factory

**Modify**: `src/townlet/substrate/config.py`

Add to `GridConfig` dataclass:

```python
@dataclass
class GridConfig:
    """Grid substrate configuration."""
    topology: Literal["square", "cubic", "hypercube", "hexagonal", "line", "graph"]

    # Existing fields...
    width: int | None = None
    height: int | None = None
    depth: int | None = None

    # NEW: Hex-specific fields
    radius: int | None = None  # For hexagonal topology
    orientation: Literal["flat_top", "pointy_top"] | None = "flat_top"

    # Validation...
    def __post_init__(self):
        if self.topology == "hexagonal":
            if self.radius is None or self.radius <= 0:
                raise ValueError(
                    f"Hexagonal topology requires positive radius, got {self.radius}"
                )
        # ... existing validation
```

**Modify**: `src/townlet/substrate/factory.py`

Add hex case to factory:

```python
def build(config: SubstrateConfig, device: torch.device) -> SpatialSubstrate:
    """Build substrate from configuration."""

    if config.type == "grid":
        grid_config = config.grid

        if grid_config.topology == "hexagonal":
            from .hexgrid import HexGridSubstrate

            return HexGridSubstrate(
                radius=grid_config.radius,
                boundary=grid_config.boundary,
                distance_metric=grid_config.distance_metric,
                orientation=grid_config.orientation or "flat_top",
            )

        # ... existing topologies
```

**Test config loading**:

Create test config: `configs/test_hex/substrate.yaml`

```yaml
type: "grid"

grid:
  topology: "hexagonal"
  radius: 3
  boundary: "clamp"
  distance_metric: "hex_manhattan"
  orientation: "flat_top"
  observation_encoding: "relative"
```

**Verify**:
```bash
python -c "
from townlet.substrate.config import load_substrate_config
from townlet.substrate.factory import SubstrateFactory
import torch

config = load_substrate_config('configs/test_hex/substrate.yaml')
substrate = SubstrateFactory.build(config, torch.device('cpu'))

print(f'✓ Hex substrate created: {substrate}')
print(f'✓ Radius: {substrate.radius}')
print(f'✓ Valid positions: {len(substrate.valid_positions)}')
print(f'✓ Action space size: {substrate.action_space_size}')
"
```

**Expected output**:
```
✓ Hex substrate created: <HexGridSubstrate object>
✓ Radius: 3
✓ Valid positions: 37
✓ Action space size: 7
```

---

### Step 1.4: Create hex config pack (1 hour)

**Purpose**: Demonstrate hex substrate in full environment

**Action**: Create training config pack with hex substrate

**Create directory**: `configs/L1_hex_strategy/`

**Create**: `configs/L1_hex_strategy/substrate.yaml`

```yaml
type: "grid"

grid:
  topology: "hexagonal"
  radius: 5  # ~91 hexes (comparable to 8×8 = 64 square grid)
  boundary: "clamp"
  distance_metric: "hex_manhattan"
  orientation: "flat_top"
  observation_encoding: "relative"
```

**Create**: `configs/L1_hex_strategy/training.yaml`

```yaml
environment:
  enabled_affordances: ["Bed", "Hospital", "HomeMeal", "Job", "Gym", "Bar", "Store", "Park"]
  partial_observability: false
  vision_range: 2
  enable_temporal_mechanics: false
  move_energy_cost: 0.5
  wait_energy_cost: 0.1
  interact_energy_cost: 0.3
  agent_lifespan: 1000

population:
  num_agents: 1
  learning_rate: 0.00025
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: "simple"

curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000

exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 100

training:
  device: "cuda"
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
```

**Copy remaining configs** (bars.yaml, cascades.yaml, affordances.yaml, cues.yaml) from L1_full_observability.

**Test training**:
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run scripts/run_demo.py --config configs/L1_hex_strategy
```

**Expected**: Training starts successfully, agent moves on hexagonal grid

---

### Step 1.5: Integration test (1 hour)

**Purpose**: Verify hex substrate works in full training loop

**Create**: `tests/test_townlet/integration/test_hex_substrate.py`

```python
"""Integration test for hexagonal grid substrate."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_hex_substrate_training_loop():
    """Hex substrate should work in full training environment."""

    # Use test hex config
    config_path = Path("configs/test_hex")

    # Create environment
    env = VectorizedHamletEnv(
        config_pack_path=config_path,
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Verify substrate type
    from townlet.substrate.hexgrid import HexGridSubstrate
    assert isinstance(env.substrate, HexGridSubstrate)

    # Reset environment
    obs = env.reset()

    # Verify observation shape includes hex position encoding (2 dims)
    assert obs.shape[1] > 2  # At least 2 dims for position

    # Run several steps
    for _ in range(10):
        # Random hex actions (0-5 = directions, 6 = INTERACT)
        actions = torch.randint(0, 7, (1,), device=torch.device("cpu"))
        obs, rewards, dones, info = env.step(actions)

        # Verify positions are valid hex coords
        q, r = env.positions[0, 0].item(), env.positions[0, 1].item()
        assert (q, r) in env.substrate.valid_positions


def test_hex_distance_symmetry():
    """Hex distances should be symmetric (d(A,B) = d(B,A))."""
    from townlet.substrate.hexgrid import HexGridSubstrate

    substrate = HexGridSubstrate(radius=5, boundary="clamp")

    pos_a = torch.tensor([[2, 1]], dtype=torch.long)
    pos_b = torch.tensor([3, 2], dtype=torch.long)

    dist_ab = substrate.compute_distance(pos_a, pos_b)
    dist_ba = substrate.compute_distance(pos_b.unsqueeze(0), pos_a[0])

    assert torch.allclose(dist_ab, dist_ba)


def test_hex_uniform_neighbor_distance():
    """All 6 hex neighbors should be distance 1 from center."""
    from townlet.substrate.hexgrid import HexGridSubstrate

    substrate = HexGridSubstrate(radius=5, boundary="clamp")

    center = torch.tensor([[0, 0]], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(center[0])

    # Should have 6 neighbors
    assert len(neighbors) == 6

    # All should be distance 1
    for neighbor in neighbors:
        distance = substrate.compute_distance(center, neighbor)
        assert torch.allclose(distance, torch.tensor(1.0))
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_hex_substrate.py -v
```

**Expected**: ALL tests PASS

---

### Step 1.6: Commit hex substrate (30 min)

**Action**: Commit all hex substrate code

```bash
git add src/townlet/substrate/hexgrid.py
git add tests/test_townlet/unit/test_substrate_hexgrid.py
git add tests/test_townlet/integration/test_hex_substrate.py
git add src/townlet/substrate/config.py
git add src/townlet/substrate/factory.py
git add configs/L1_hex_strategy/
git add configs/test_hex/

git commit -m "feat(substrate): add hexagonal grid substrate

Implements 2D hexagonal grid with axial coordinates:

**Features**:
- Axial coordinate system (q, r) with 6-neighbor movement
- Uniform distances to all neighbors (no diagonal ambiguity)
- Hex distance metrics (hex_manhattan, cube_manhattan)
- Valid position generation (hex grid shape constraint)
- Action space: 6 directions + INTERACT = 7 actions

**Pedagogical Value** (✅✅):
- Teaches coordinate system design
- Teaches uniform distance metrics
- Use cases: Strategy games, natural terrain, biological sims

**Implementation**:
- HexGridSubstrate class (278 lines)
- Comprehensive unit tests (12 tests, 100% coverage)
- Integration tests (3 tests)
- Config pack: L1_hex_strategy (for demonstration)

**Testing**:
- All unit tests pass (12/12)
- All integration tests pass (3/3)
- Training verified on hex grid

Part of TASK-002A Phase 5D (Alternative Topologies)."
```

---

## Task 5D.2: 1D Grid Substrate (2-4 hours)

### Overview

Implements simplest spatial substrate - 1D linear grid. Validates substrate abstraction at N=1 edge case. Useful for pedagogical progression (1D → 2D → 3D).

**Files to Create**:
- `src/townlet/substrate/grid1d.py` (new)
- `tests/test_townlet/unit/test_substrate_grid1d.py` (new)
- `configs/L0_1D_line/` (new config pack)

---

### Step 2.1: Write unit tests for Grid1DSubstrate (30 min)

**Purpose**: TDD - define 1D behavior through tests

**Create**: `tests/test_townlet/unit/test_substrate_grid1d.py`

```python
"""Unit tests for Grid1DSubstrate."""
import pytest
import torch
from townlet.substrate.grid1d import Grid1DSubstrate


def test_grid1d_initialization():
    """1D grid should initialize with valid parameters."""
    substrate = Grid1DSubstrate(width=20, boundary="clamp")

    assert substrate.width == 20
    assert substrate.position_dim == 1
    assert substrate.position_dtype == torch.long
    assert substrate.action_space_size == 3  # LEFT, RIGHT, INTERACT


def test_grid1d_initialize_positions():
    """1D grid should randomly place agents on line."""
    substrate = Grid1DSubstrate(width=10, boundary="clamp")

    positions = substrate.initialize_positions(num_agents=5, device=torch.device("cpu"))

    assert positions.shape == (5, 1)
    assert positions.dtype == torch.long
    assert (positions >= 0).all()
    assert (positions < 10).all()


def test_grid1d_movement():
    """1D grid should move agents left/right."""
    substrate = Grid1DSubstrate(width=10, boundary="clamp")

    # Start at position 5
    positions = torch.tensor([[5]], dtype=torch.long)

    # Move right (+1)
    deltas = torch.tensor([[1]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert new_positions[0, 0].item() == 6

    # Move left (-1)
    deltas = torch.tensor([[-1]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert new_positions[0, 0].item() == 4


def test_grid1d_boundary_clamp():
    """1D grid should clamp at boundaries."""
    substrate = Grid1DSubstrate(width=10, boundary="clamp")

    # At left edge
    positions = torch.tensor([[0]], dtype=torch.long)
    deltas = torch.tensor([[-1]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert new_positions[0, 0].item() == 0  # Clamped

    # At right edge
    positions = torch.tensor([[9]], dtype=torch.long)
    deltas = torch.tensor([[1]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert new_positions[0, 0].item() == 9  # Clamped


def test_grid1d_boundary_wrap():
    """1D grid should wrap at boundaries."""
    substrate = Grid1DSubstrate(width=10, boundary="wrap")

    # Wrap from left edge
    positions = torch.tensor([[0]], dtype=torch.long)
    deltas = torch.tensor([[-1]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert new_positions[0, 0].item() == 9  # Wrapped to right edge

    # Wrap from right edge
    positions = torch.tensor([[9]], dtype=torch.long)
    deltas = torch.tensor([[1]], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, deltas)
    assert new_positions[0, 0].item() == 0  # Wrapped to left edge


def test_grid1d_distance():
    """1D distance should be absolute difference."""
    substrate = Grid1DSubstrate(width=20, boundary="clamp")

    pos1 = torch.tensor([[5], [10]], dtype=torch.long)
    pos2 = torch.tensor([7], dtype=torch.long)

    distances = substrate.compute_distance(pos1, pos2)

    assert distances[0].item() == 2  # |5 - 7| = 2
    assert distances[1].item() == 3  # |10 - 7| = 3


def test_grid1d_encode_observation():
    """1D grid should encode position as normalized scalar."""
    substrate = Grid1DSubstrate(width=10, boundary="clamp")

    positions = torch.tensor([[0], [5], [9]], dtype=torch.long)
    encoded = substrate.encode_observation(positions, {})

    assert encoded.shape == (3, 1)
    assert torch.allclose(encoded[0], torch.tensor([0.0]), atol=0.01)
    assert torch.allclose(encoded[1], torch.tensor([0.5555]), atol=0.01)
    assert torch.allclose(encoded[2], torch.tensor([1.0]), atol=0.01)


def test_grid1d_get_all_positions():
    """1D grid should return all positions on line."""
    substrate = Grid1DSubstrate(width=5, boundary="clamp")

    all_positions = substrate.get_all_positions()

    assert len(all_positions) == 5
    assert [0] in all_positions
    assert [4] in all_positions


def test_grid1d_get_valid_neighbors():
    """1D grid should return left/right neighbors (2 max)."""
    substrate = Grid1DSubstrate(width=10, boundary="clamp")

    # Middle position has 2 neighbors
    position = torch.tensor([5], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)
    assert len(neighbors) == 2

    # Left edge has 1 neighbor
    position = torch.tensor([0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)
    assert len(neighbors) == 1
    assert neighbors[0][0].item() == 1

    # Right edge has 1 neighbor
    position = torch.tensor([9], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)
    assert len(neighbors) == 1
    assert neighbors[0][0].item() == 8
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_grid1d.py -v
```

**Expected**: ALL tests FAIL (not yet implemented)

---

### Step 2.2: Implement Grid1DSubstrate (1 hour)

**Purpose**: Simple 1D implementation to pass all tests

**Create**: `src/townlet/substrate/grid1d.py`

```python
"""1D line grid substrate."""

from typing import Literal
import torch
from .base import SpatialSubstrate


class Grid1DSubstrate(SpatialSubstrate):
    """1D line grid substrate.

    Positions are scalar integers in [0, width).
    Movement actions: LEFT, RIGHT, INTERACT (3 total).

    This is the simplest spatial substrate - essentially a number line.

    Pedagogical Value:
    - Teaches dimensionality concepts (1D → 2D → 3D progression)
    - Edge case validation (N=1 dimension)
    - Use cases: Conveyor belts, number lines, sequential decisions
    """

    position_dim = 1
    position_dtype = torch.long

    def __init__(
        self,
        width: int,
        boundary: Literal["clamp", "wrap", "bounce"] = "clamp",
    ):
        """Initialize 1D line grid.

        Args:
            width: Number of positions on line (0 to width-1)
            boundary: Boundary handling mode

        Raises:
            ValueError: If width <= 0 or boundary invalid
        """
        if width <= 0:
            raise ValueError(f"Width must be positive: {width}")
        if boundary not in ("clamp", "wrap", "bounce"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        self.width = width
        self.boundary = boundary

    @property
    def action_space_size(self) -> int:
        """2 movement directions + 1 INTERACT = 3 actions."""
        return 3

    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Random positions on 1D line.

        Args:
            num_agents: Number of agents to place
            device: PyTorch device

        Returns:
            [num_agents, 1] tensor of 1D positions
        """
        return torch.randint(0, self.width, (num_agents, 1), device=device)

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply movement with 1D boundary handling.

        Args:
            positions: [num_agents, 1] current positions
            deltas: [num_agents, 1] movement deltas

        Returns:
            [num_agents, 1] new positions after boundary handling
        """
        new_positions = positions + deltas

        if self.boundary == "clamp":
            new_positions = torch.clamp(new_positions, 0, self.width - 1)

        elif self.boundary == "wrap":
            new_positions = new_positions % self.width

        elif self.boundary == "bounce":
            out_of_bounds = (new_positions < 0) | (new_positions >= self.width)
            new_positions[out_of_bounds] = positions[out_of_bounds]

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """1D distance is absolute difference.

        Args:
            pos1: [num_agents, 1] agent positions
            pos2: [1] target position

        Returns:
            [num_agents] distances
        """
        return torch.abs(pos1[:, 0] - pos2[0]).float()

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Normalize 1D positions to [0, 1].

        Args:
            positions: [num_agents, 1] 1D positions
            affordances: Dict of affordance positions (reserved)

        Returns:
            [num_agents, 1] normalized positions
        """
        return positions.float() / max(self.width - 1, 1)

    def get_observation_dim(self) -> int:
        """1D position encoding is 1-dimensional."""
        return 1

    def get_all_positions(self) -> list[list[int]]:
        """Return all positions on 1D line.

        Returns:
            List of [x] positions for all positions on line
        """
        return [[x] for x in range(self.width)]

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get left and right neighbors (2 max, fewer at boundaries).

        Args:
            position: [1] 1D position

        Returns:
            List of neighbor positions (up to 2)
        """
        x = position[0].item()
        neighbors = []

        if x > 0:
            neighbors.append(torch.tensor([x - 1], dtype=torch.long))
        if x < self.width - 1:
            neighbors.append(torch.tensor([x + 1], dtype=torch.long))

        return neighbors

    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor
    ) -> torch.Tensor:
        """Check if agents are on target position (exact match).

        Args:
            agent_positions: [num_agents, 1] agent positions
            target_position: [1] target position

        Returns:
            [num_agents] boolean tensor
        """
        return (agent_positions[:, 0] == target_position[0])
```

**Verify**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_grid1d.py -v
```

**Expected**: ALL tests PASS

---

### Step 2.3: Wire up 1D config and commit (1 hour)

**Action**: Add 1D to config system, create config pack, commit

**Modify `config.py` and `factory.py`** (similar to hex, but for "line" topology)

**Create config pack**: `configs/L0_1D_line/` with substrate.yaml:

```yaml
type: "grid"

grid:
  topology: "line"
  width: 20
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
```

**Test and commit**:

```bash
# Test training
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run scripts/run_demo.py --config configs/L0_1D_line

# Commit
git add src/townlet/substrate/grid1d.py tests/test_townlet/unit/test_substrate_grid1d.py
git add configs/L0_1D_line/
git commit -m "feat(substrate): add 1D line grid substrate

Implements simplest spatial substrate - 1D linear grid:

**Features**:
- 1D scalar positions [0, width)
- 2 movement directions (LEFT/RIGHT) + INTERACT = 3 actions
- Boundary modes: clamp, wrap, bounce
- Distance: Absolute difference

**Pedagogical Value** (✅):
- Edge case validation (N=1 dimension)
- Pedagogical progression (1D → 2D → 3D)
- Use cases: Conveyor belts, number lines, sequences

**Implementation**:
- Grid1DSubstrate class (96 lines, very simple)
- Unit tests (11 tests, 100% coverage)
- Config pack: L0_1D_line

Part of TASK-002A Phase 5D (Alternative Topologies)."
```

---

## Task 5D.3: Graph-Based Substrate (18-24 hours)

### Overview

Most complex topology - graph nodes connected by edges. Requires **action masking infrastructure** (variable action spaces). Enables graph RL, non-Euclidean reasoning.

**Files to Create**:
- `src/townlet/substrate/graph.py` (new)
- `tests/test_townlet/unit/test_substrate_graph.py` (new)
- `tests/test_townlet/unit/test_action_masking.py` (new)
- `configs/L1_graph_subway/` (new config pack)

**Files to Modify**:
- `src/townlet/substrate/base.py` (add `get_valid_actions()` method)
- `src/townlet/environment/vectorized_env.py` (add action masking)
- `src/townlet/population/vectorized.py` (mask invalid actions during exploration)

---

### Step 3.1: Add action masking to base interface (1 hour)

**Purpose**: Extend substrate interface to support variable action spaces

**Action**: Add `get_valid_actions()` method to base class

**Modify**: `src/townlet/substrate/base.py`

Add after `get_valid_neighbors()` method:

```python
def get_valid_actions(self, position: torch.Tensor) -> list[int]:
    """Get valid action indices for a given position.

    Used for action masking when action space varies by state.
    Default implementation: All actions valid (no masking needed).

    Args:
        position: [position_dim] agent position

    Returns:
        List of valid action indices [0, action_space_size)

    Examples:
        Grid2D: Always returns [0, 1, 2, 3, 4] (all 5 actions valid)
        Graph: Returns edge indices + INTERACT based on node's neighbors
               Node with 3 neighbors: [0, 1, 2, max_edges] (3 edges + INTERACT)
    """
    # Default: All actions valid (no masking)
    return list(range(self.action_space_size))
```

**Add tests**:

Create `tests/test_townlet/unit/test_action_masking.py`:

```python
"""Tests for action masking interface."""
import torch
from townlet.substrate.grid2d import Grid2DSubstrate


def test_grid2d_all_actions_valid():
    """Grid2D should return all actions as valid (no masking needed)."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")

    position = torch.tensor([3, 3], dtype=torch.long)
    valid_actions = substrate.get_valid_actions(position)

    # All 5 actions should be valid
    assert valid_actions == [0, 1, 2, 3, 4]
    assert len(valid_actions) == substrate.action_space_size
```

**Run test**:
```bash
uv run pytest tests/test_townlet/unit/test_action_masking.py -v
```

**Expected**: PASS (default implementation returns all actions)

**Commit**:
```bash
git add src/townlet/substrate/base.py tests/test_townlet/unit/test_action_masking.py
git commit -m "feat(substrate): add action masking interface

Adds get_valid_actions() method to SpatialSubstrate base class.

**Purpose**:
- Support variable action spaces (needed for graph substrate)
- Enable action masking during training

**Default Behavior**:
- Returns all actions as valid (no masking)
- Grid substrates (2D/3D/ND) use default (all actions always valid)
- Graph substrates override to return only valid edges

**Testing**:
- Added test_action_masking.py
- Verified Grid2D returns all actions valid

Part of TASK-002A Phase 5D Task 3 (Graph Substrate - Action Masking)."
```

---

### Step 3.2: Write graph substrate unit tests (2-3 hours)

**Purpose**: TDD - define graph behavior through comprehensive tests

**Create**: `tests/test_townlet/unit/test_substrate_graph.py`

```python
"""Unit tests for GraphSubstrate."""
import pytest
import torch
from townlet.substrate.graph import GraphSubstrate


def test_graph_simple_initialization():
    """Graph should initialize with simple edge list."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Square graph
    substrate = GraphSubstrate(num_nodes=4, edges=edges, directed=False)

    assert substrate.num_nodes == 4
    assert substrate.position_dim == 1
    assert substrate.position_dtype == torch.long
    assert len(substrate.edges) == 4


def test_graph_action_space_size():
    """Graph action space size should be max_edges + 1."""
    # Graph with varying node degrees
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4)]  # Node 1 has 4 neighbors
    substrate = GraphSubstrate(num_nodes=5, edges=edges, directed=False)

    # Max degree is 4, so action space = 4 + 1 = 5
    assert substrate.action_space_size == 5


def test_graph_adjacency_list():
    """Graph should build correct adjacency list."""
    edges = [(0, 1), (1, 2), (2, 3)]
    substrate = GraphSubstrate(num_nodes=4, edges=edges, directed=False)

    # Node 0 connects to node 1
    assert 1 in substrate.adjacency_list[0]

    # Node 1 connects to nodes 0 and 2 (undirected)
    assert 0 in substrate.adjacency_list[1]
    assert 2 in substrate.adjacency_list[1]

    # Node 3 connects only to node 2
    assert 2 in substrate.adjacency_list[3]
    assert len(substrate.adjacency_list[3]) == 1


def test_graph_directed_vs_undirected():
    """Graph should handle directed vs undirected edges."""
    edges = [(0, 1), (1, 2)]

    # Undirected
    undirected = GraphSubstrate(num_nodes=3, edges=edges, directed=False)
    assert 1 in undirected.adjacency_list[0]
    assert 0 in undirected.adjacency_list[1]  # Bidirectional

    # Directed
    directed = GraphSubstrate(num_nodes=3, edges=edges, directed=True)
    assert 1 in directed.adjacency_list[0]
    assert 0 not in directed.adjacency_list[1]  # Only 0→1, not 1→0


def test_graph_initialize_positions():
    """Graph should randomly place agents at nodes."""
    substrate = GraphSubstrate(num_nodes=10, edges=[], directed=False)

    positions = substrate.initialize_positions(num_agents=5, device=torch.device("cpu"))

    assert positions.shape == (5, 1)
    assert positions.dtype == torch.long
    assert (positions >= 0).all()
    assert (positions < 10).all()


def test_graph_get_valid_actions():
    """Graph should return valid actions based on node's neighbors."""
    edges = [(0, 1), (0, 2), (0, 3)]  # Node 0 has 3 neighbors
    substrate = GraphSubstrate(num_nodes=4, edges=edges, directed=False)

    # Node 0 has 3 neighbors → actions [0, 1, 2, 3] (3 edges + INTERACT)
    position = torch.tensor([0], dtype=torch.long)
    valid_actions = substrate.get_valid_actions(position)

    assert len(valid_actions) == 4
    assert 0 in valid_actions  # Edge 0 → node 1
    assert 1 in valid_actions  # Edge 1 → node 2
    assert 2 in valid_actions  # Edge 2 → node 3
    assert 3 in valid_actions  # INTERACT (max_edges)

    # Node 1 has 1 neighbor → actions [0, 3] (1 edge + INTERACT)
    position = torch.tensor([1], dtype=torch.long)
    valid_actions = substrate.get_valid_actions(position)
    assert len(valid_actions) == 2
    assert 0 in valid_actions  # Edge to node 0
    assert 3 in valid_actions  # INTERACT


def test_graph_movement():
    """Graph should move agents along edges."""
    edges = [(0, 1), (1, 2), (2, 3)]
    substrate = GraphSubstrate(num_nodes=4, edges=edges, directed=False)

    # Agent at node 0
    positions = torch.tensor([[0]], dtype=torch.long)

    # Action 0: Traverse to first neighbor (node 1)
    actions = torch.tensor([0], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, actions)
    assert new_positions[0, 0].item() == 1

    # Agent at node 1, action 1: Traverse to second neighbor (node 2)
    positions = torch.tensor([[1]], dtype=torch.long)
    actions = torch.tensor([1], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, actions)
    assert new_positions[0, 0].item() == 2


def test_graph_invalid_action():
    """Graph should ignore invalid actions (stay in place)."""
    edges = [(0, 1)]  # Node 0 has only 1 neighbor
    substrate = GraphSubstrate(num_nodes=2, edges=edges, directed=False)

    positions = torch.tensor([[0]], dtype=torch.long)

    # Action 1 is invalid (node 0 has no second neighbor)
    actions = torch.tensor([1], dtype=torch.long)
    new_positions = substrate.apply_movement(positions, actions)

    # Should stay at node 0
    assert new_positions[0, 0].item() == 0


def test_graph_shortest_path_distance():
    """Graph should compute shortest path distances correctly."""
    # Line graph: 0 - 1 - 2 - 3
    edges = [(0, 1), (1, 2), (2, 3)]
    substrate = GraphSubstrate(num_nodes=4, edges=edges, directed=False)

    # Distance from node 0 to node 3 should be 3 (0→1→2→3)
    pos1 = torch.tensor([[0]], dtype=torch.long)
    pos2 = torch.tensor([3], dtype=torch.long)
    distance = substrate.compute_distance(pos1, pos2)
    assert distance.item() == 3.0

    # Distance from node 1 to node 2 should be 1
    pos1 = torch.tensor([[1]], dtype=torch.long)
    pos2 = torch.tensor([2], dtype=torch.long)
    distance = substrate.compute_distance(pos1, pos2)
    assert distance.item() == 1.0


def test_graph_encode_observation():
    """Graph should encode node ID as normalized position."""
    substrate = GraphSubstrate(num_nodes=10, edges=[], directed=False)

    # Node 0 → 0.0
    positions = torch.tensor([[0]], dtype=torch.long)
    encoded = substrate.encode_observation(positions, {})
    assert torch.allclose(encoded, torch.tensor([[0.0]]), atol=0.01)

    # Node 9 → 1.0
    positions = torch.tensor([[9]], dtype=torch.long)
    encoded = substrate.encode_observation(positions, {})
    assert torch.allclose(encoded, torch.tensor([[1.0]]), atol=0.01)

    # Node 5 → ~0.555
    positions = torch.tensor([[5]], dtype=torch.long)
    encoded = substrate.encode_observation(positions, {})
    assert torch.allclose(encoded, torch.tensor([[0.5555]]), atol=0.01)


def test_graph_get_all_positions():
    """Graph should return all node IDs as positions."""
    substrate = GraphSubstrate(num_nodes=5, edges=[], directed=False)

    all_positions = substrate.get_all_positions()

    assert len(all_positions) == 5
    assert [0] in all_positions
    assert [4] in all_positions


def test_graph_get_valid_neighbors():
    """Graph should return neighbor nodes."""
    edges = [(0, 1), (0, 2), (1, 2)]  # Triangle
    substrate = GraphSubstrate(num_nodes=3, edges=edges, directed=False)

    # Node 0 has 2 neighbors (1, 2)
    position = torch.tensor([0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)
    assert len(neighbors) == 2

    neighbor_ids = [n[0].item() for n in neighbors]
    assert 1 in neighbor_ids
    assert 2 in neighbor_ids
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_graph.py -v
```

**Expected**: ALL tests FAIL (not yet implemented)

---

### Step 3.3: Implement GraphSubstrate class (6-8 hours)

**Purpose**: Implement graph substrate to pass all tests

**Create**: `src/townlet/substrate/graph.py`

```python
"""Graph-based substrate with edge traversal movement."""

import torch
from collections import deque
from .base import SpatialSubstrate


class GraphSubstrate(SpatialSubstrate):
    """Graph-based substrate with edge traversal movement.

    Positions are node IDs in the graph. Movement is edge traversal.
    Action space varies per node (different nodes have different degrees).

    Key Differences from Grid Substrates:
    - position_dim = 1 (node ID)
    - action_space_size varies per state (requires action masking)
    - Movement is action-dependent (not delta-dependent)
    - Distance is shortest path length (not Euclidean)

    Pedagogical Value (✅✅✅):
    - Teaches graph RL and non-Euclidean reasoning
    - Teaches action masking (variable action spaces)
    - Teaches topological reasoning vs metric reasoning
    - Use cases: Subway systems, social networks, state machines

    References:
    - Graph RL: https://arxiv.org/abs/2004.10756
    """

    position_dim = 1  # Node ID
    position_dtype = torch.long

    def __init__(
        self,
        num_nodes: int,
        edges: list[tuple[int, int]],
        directed: bool = False,
        max_edges: int | None = None,
    ):
        """Initialize graph substrate.

        Args:
            num_nodes: Number of nodes in graph
            edges: List of (from_node, to_node) edges
            directed: Whether edges are directed (default: undirected)
            max_edges: Maximum edges per node (for action space sizing)
                       If None, auto-computed as max(node_degrees)

        Raises:
            ValueError: If num_nodes <= 0 or invalid edges
        """
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive: {num_nodes}")

        self.num_nodes = num_nodes
        self.edges = edges
        self.directed = directed

        # Build adjacency list
        self.adjacency_list = self._build_adjacency_list(edges, directed)

        # Compute max edges per node (for action space size)
        if max_edges is None:
            max_edges = max(len(neighbors) for neighbors in self.adjacency_list.values())
        self.max_edges = max_edges

        # Precompute all-pairs shortest paths (for distance metric)
        self.shortest_paths = self._compute_shortest_paths()

    @property
    def action_space_size(self) -> int:
        """Maximum action space size = max_edges + 1 (INTERACT)."""
        return self.max_edges + 1

    def _build_adjacency_list(
        self,
        edges: list[tuple[int, int]],
        directed: bool
    ) -> dict[int, list[int]]:
        """Build adjacency list from edge list.

        Args:
            edges: List of (from, to) edges
            directed: Whether edges are directed

        Returns:
            Dict mapping node_id → list of neighbor node_ids
        """
        adj = {i: [] for i in range(self.num_nodes)}

        for u, v in edges:
            if u < 0 or u >= self.num_nodes or v < 0 or v >= self.num_nodes:
                raise ValueError(f"Invalid edge ({u}, {v}): nodes must be in [0, {self.num_nodes})")

            adj[u].append(v)
            if not directed:
                adj[v].append(u)

        return adj

    def _compute_shortest_paths(self) -> torch.Tensor:
        """Precompute all-pairs shortest paths using BFS.

        Returns:
            [num_nodes, num_nodes] tensor of shortest path lengths
            Distance is inf if no path exists (disconnected components)
        """
        shortest_paths = torch.full(
            (self.num_nodes, self.num_nodes),
            float('inf'),
            dtype=torch.float32
        )

        # BFS from each node
        for start_node in range(self.num_nodes):
            visited = {start_node: 0}
            queue = deque([start_node])

            while queue:
                node = queue.popleft()
                dist = visited[node]

                for neighbor in self.adjacency_list[node]:
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append(neighbor)

            # Fill shortest paths
            for node, dist in visited.items():
                shortest_paths[start_node, node] = dist

        return shortest_paths

    def get_valid_actions(self, position: torch.Tensor) -> list[int]:
        """Get valid action indices for a given node.

        This is the key method for action masking.

        Args:
            position: [1] node ID

        Returns:
            List of valid action indices (edge indices + INTERACT)

        Example:
            Node with 3 neighbors: [0, 1, 2, max_edges]
            (3 edge actions + INTERACT action)
        """
        node_id = position[0].item()
        neighbors = self.adjacency_list[node_id]

        # Valid actions = edge traversal (0 to num_neighbors-1) + INTERACT
        valid_actions = list(range(len(neighbors))) + [self.max_edges]
        return valid_actions

    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Randomly place agents at graph nodes.

        Args:
            num_agents: Number of agents to place
            device: PyTorch device

        Returns:
            [num_agents, 1] node IDs
        """
        return torch.randint(0, self.num_nodes, (num_agents, 1), device=device)

    def apply_movement(self, positions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Move agents along graph edges.

        Args:
            positions: [num_agents, 1] current node IDs
            actions: [num_agents] action indices (which edge to traverse)

        Returns:
            [num_agents, 1] new node IDs after movement
        """
        new_positions = positions.clone()

        for i in range(positions.shape[0]):
            node_id = positions[i, 0].item()
            action = actions[i].item()
            neighbors = self.adjacency_list[node_id]

            # If valid edge action, traverse to neighbor
            if action < len(neighbors):
                new_positions[i, 0] = neighbors[action]
            # Else: INTERACT or invalid → stay in place

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute shortest path distance between nodes.

        Args:
            pos1: [num_agents, 1] agent node IDs
            pos2: [1] target node ID

        Returns:
            [num_agents] shortest path lengths (number of hops)
        """
        node_ids_1 = pos1[:, 0].long()
        node_id_2 = pos2[0].long()
        return self.shortest_paths[node_ids_1, node_id_2]

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode node ID as normalized position [0, 1].

        Args:
            positions: [num_agents, 1] node IDs
            affordances: Dict of affordance positions (reserved)

        Returns:
            [num_agents, 1] normalized node IDs
        """
        return positions.float() / max(self.num_nodes - 1, 1)

    def get_observation_dim(self) -> int:
        """Graph position encoding is 1-dimensional (normalized node ID)."""
        return 1

    def get_all_positions(self) -> list[list[int]]:
        """Return all node IDs as positions.

        Returns:
            List of [node_id] positions
        """
        return [[node_id] for node_id in range(self.num_nodes)]

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Return neighbor nodes.

        Args:
            position: [1] node ID

        Returns:
            List of neighbor node tensors
        """
        node_id = position[0].item()
        neighbors = self.adjacency_list[node_id]
        return [torch.tensor([n], dtype=torch.long) for n in neighbors]

    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor
    ) -> torch.Tensor:
        """Check if agents are at target node (exact match).

        Args:
            agent_positions: [num_agents, 1] agent node IDs
            target_position: [1] target node ID

        Returns:
            [num_agents] boolean tensor
        """
        return (agent_positions[:, 0] == target_position[0])
```

**Verify**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_graph.py -v
```

**Expected**: ALL tests PASS

---

### Step 3.4: Add action masking to environment (4-6 hours)

**Purpose**: Wire up action masking in training loop

**Modify**: `src/townlet/environment/vectorized_env.py`

Find `_execute_actions()` method and update action selection to use masking:

```python
def _execute_actions(self, actions: torch.Tensor):
    """Execute agent actions with action masking support.

    Args:
        actions: [num_agents] action indices
    """
    # Get valid actions for each agent's current position
    # (Only needed for substrates with variable action spaces)
    valid_action_masks = self._get_valid_action_masks()

    # Mask invalid actions (for safety, though Q-network should already handle this)
    # If an action is invalid, replace with INTERACT action
    for i in range(self.num_agents):
        if not valid_action_masks[i, actions[i]]:
            # Invalid action detected - replace with INTERACT
            actions[i] = self.substrate.action_space_size - 1  # INTERACT

    # ... rest of movement logic
```

Add helper method:

```python
def _get_valid_action_masks(self) -> torch.Tensor:
    """Get valid action masks for all agents.

    Returns:
        [num_agents, action_space_size] boolean tensor
        True = action is valid, False = action is invalid
    """
    action_space_size = self.substrate.action_space_size
    masks = torch.zeros(
        (self.num_agents, action_space_size),
        dtype=torch.bool,
        device=self.device
    )

    for i in range(self.num_agents):
        valid_actions = self.substrate.get_valid_actions(self.positions[i])
        masks[i, valid_actions] = True

    return masks
```

**Modify**: `src/townlet/population/vectorized.py`

Update `select_actions()` to mask invalid actions:

```python
def select_actions(
    self,
    observations: torch.Tensor,
    valid_action_masks: torch.Tensor | None = None,
    epsilon: float = 0.0
) -> torch.Tensor:
    """Select actions with optional action masking.

    Args:
        observations: [num_agents, obs_dim] observations
        valid_action_masks: [num_agents, action_dim] boolean mask (optional)
        epsilon: Epsilon-greedy exploration rate

    Returns:
        [num_agents] action indices
    """
    with torch.no_grad():
        q_values = self.q_network(observations)  # [num_agents, action_dim]

        # Apply action masking if provided
        if valid_action_masks is not None:
            # Set invalid actions to -inf (will never be selected)
            q_values[~valid_action_masks] = -float('inf')

        # Epsilon-greedy with masking
        if epsilon > 0:
            random_mask = torch.rand(len(observations), device=self.device) < epsilon

            # For random actions, sample uniformly from valid actions only
            if valid_action_masks is not None:
                for i in range(len(observations)):
                    if random_mask[i]:
                        valid_actions = torch.where(valid_action_masks[i])[0]
                        if len(valid_actions) > 0:
                            q_values[i, :] = -float('inf')
                            q_values[i, valid_actions] = torch.rand(len(valid_actions))
            else:
                # Standard epsilon-greedy (no masking)
                q_values[random_mask] = torch.rand_like(q_values[random_mask])

        # Greedy selection (max Q-value among valid actions)
        actions = q_values.argmax(dim=1)

    return actions
```

**Test action masking**:

Add to `tests/test_townlet/unit/test_action_masking.py`:

```python
def test_graph_action_masking():
    """Graph substrate should mask invalid actions correctly."""
    from townlet.substrate.graph import GraphSubstrate

    # Simple graph: Node 0 has 2 neighbors, Node 1 has 1 neighbor
    edges = [(0, 1), (0, 2)]
    substrate = GraphSubstrate(num_nodes=3, edges=edges, directed=False)

    # Node 0: Valid actions [0, 1, 2] (2 edges + INTERACT)
    position = torch.tensor([0], dtype=torch.long)
    valid_actions = substrate.get_valid_actions(position)
    assert len(valid_actions) == 3

    # Node 2: Valid actions [0, 2] (1 edge + INTERACT)
    position = torch.tensor([2], dtype=torch.long)
    valid_actions = substrate.get_valid_actions(position)
    assert len(valid_actions) == 2
    assert 0 in valid_actions  # Edge to node 0
    assert 2 in valid_actions  # INTERACT
    assert 1 not in valid_actions  # Invalid (no second neighbor)
```

---

### Step 3.5: Wire up graph config, create config pack, test (3-4 hours)

**Action**: Add graph topology to config system

**Modify `config.py`** to support graph edges:

```python
@dataclass
class GridConfig:
    topology: Literal["square", "cubic", "hypercube", "hexagonal", "line", "graph"]

    # ... existing fields

    # NEW: Graph-specific fields
    num_nodes: int | None = None
    edges: list[tuple[int, int]] | None = None
    directed: bool = False
    max_edges: int | None = None

    def __post_init__(self):
        if self.topology == "graph":
            if self.num_nodes is None or self.num_nodes <= 0:
                raise ValueError("Graph topology requires positive num_nodes")
            if self.edges is None:
                raise ValueError("Graph topology requires edges list")
```

**Modify `factory.py`** to build graph:

```python
if grid_config.topology == "graph":
    from .graph import GraphSubstrate

    return GraphSubstrate(
        num_nodes=grid_config.num_nodes,
        edges=grid_config.edges,
        directed=grid_config.directed,
        max_edges=grid_config.max_edges,
    )
```

**Create config pack**: `configs/L1_graph_subway/substrate.yaml`

```yaml
type: "grid"

grid:
  topology: "graph"

  # Simple subway network (16 stations)
  num_nodes: 16

  # Edge list (undirected)
  edges:
    - [0, 1]   # Line 1: Station 0 ↔ 1
    - [1, 2]   # Station 1 ↔ 2
    - [2, 3]   # Station 2 ↔ 3
    - [3, 4]   # Station 3 ↔ 4
    - [5, 6]   # Line 2: Station 5 ↔ 6
    - [6, 7]   # Station 6 ↔ 7
    - [7, 8]   # Station 7 ↔ 8
    - [1, 6]   # Transfer: Line 1 ↔ Line 2 at stations 1, 6
    - [3, 7]   # Transfer: Line 1 ↔ Line 2 at stations 3, 7
    - [9, 10]  # Line 3: Station 9 ↔ 10
    - [10, 11] # Station 10 ↔ 11
    - [11, 12] # Station 11 ↔ 12
    - [2, 10]  # Transfer: Line 1 ↔ Line 3
    - [13, 14] # Line 4: Station 13 ↔ 14
    - [14, 15] # Station 14 ↔ 15
    - [4, 14]  # Transfer: Line 1 ↔ Line 4
    - [8, 15]  # Transfer: Line 2 ↔ Line 4

  directed: false
  distance_metric: "shortest_path"
  observation_encoding: "relative"
```

**Test training**:
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run scripts/run_demo.py --config configs/L1_graph_subway
```

**Expected**: Training runs successfully with graph substrate + action masking

---

### Step 3.6: Integration tests and commit (2-3 hours)

**Create**: `tests/test_townlet/integration/test_graph_substrate.py`

```python
"""Integration test for graph substrate."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_graph_substrate_full_training():
    """Graph substrate should work in full training environment."""

    config_path = Path("configs/L1_graph_subway")

    env = VectorizedHamletEnv(
        config_pack_path=config_path,
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Verify graph substrate
    from townlet.substrate.graph import GraphSubstrate
    assert isinstance(env.substrate, GraphSubstrate)

    # Reset
    obs = env.reset()

    # Run episode
    for step in range(50):
        # Get valid actions for current position
        position = env.positions[0]
        valid_actions = env.substrate.get_valid_actions(position)

        # Sample from valid actions
        action = torch.tensor([valid_actions[0]], device=torch.device("cpu"))

        obs, rewards, dones, info = env.step(action)

        # Verify position is valid node
        node_id = env.positions[0, 0].item()
        assert 0 <= node_id < env.substrate.num_nodes


def test_graph_action_masking_in_training():
    """Action masking should prevent invalid actions during training."""

    from townlet.substrate.graph import GraphSubstrate

    # Simple graph
    edges = [(0, 1), (1, 2)]
    substrate = GraphSubstrate(num_nodes=3, edges=edges, directed=False)

    # Agent at node 0 (has 1 neighbor)
    position = torch.tensor([0], dtype=torch.long)

    # Get valid actions
    valid_actions = substrate.get_valid_actions(position)

    # Should have 2 valid actions (1 edge + INTERACT)
    assert len(valid_actions) == 2

    # Action 0 should be valid (edge to node 1)
    assert 0 in valid_actions

    # INTERACT should be valid
    assert substrate.action_space_size - 1 in valid_actions
```

**Run all graph tests**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_graph.py -v
uv run pytest tests/test_townlet/unit/test_action_masking.py -v
uv run pytest tests/test_townlet/integration/test_graph_substrate.py -v
```

**Expected**: ALL tests PASS

**Commit**:
```bash
git add src/townlet/substrate/graph.py
git add src/townlet/substrate/base.py
git add src/townlet/environment/vectorized_env.py
git add src/townlet/population/vectorized.py
git add tests/test_townlet/unit/test_substrate_graph.py
git add tests/test_townlet/unit/test_action_masking.py
git add tests/test_townlet/integration/test_graph_substrate.py
git add configs/L1_graph_subway/
git commit -m "feat(substrate): add graph-based substrate with action masking

Implements graph substrate with variable action spaces:

**Features**:
- Graph nodes as positions (node IDs)
- Edge traversal movement (not spatial deltas)
- Action masking (variable action spaces per node)
- Shortest path distance (BFS precomputation)
- Directed/undirected graphs
- Action space size: max_edges + 1 (INTERACT)

**Action Masking Infrastructure**:
- Added get_valid_actions() to SpatialSubstrate base
- Integrated masking into action selection
- Masked epsilon-greedy exploration
- Prevents invalid actions during training

**Pedagogical Value** (✅✅✅):
- Teaches graph RL and non-Euclidean reasoning
- Teaches action masking (variable action spaces)
- Teaches topological vs metric reasoning
- Use cases: Subway systems, social networks, state machines

**Implementation**:
- GraphSubstrate class (312 lines)
- Action masking in vectorized_env.py and vectorized.py
- Comprehensive tests (15 unit + 2 integration)
- Config pack: L1_graph_subway (16-station subway network)

**Testing**:
- All unit tests pass (15/15)
- All integration tests pass (2/2)
- Action masking verified in training loop

Part of TASK-002A Phase 5D (Alternative Topologies)."
```

---

## Task 5D.4: Documentation & Examples (2-3 hours)

### Step 4.1: Update CLAUDE.md (1 hour)

**Modify**: `CLAUDE.md`

Add section on Phase 5D topologies:

```markdown
### Phase 5D: Alternative Topologies

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

**Graph-Based** (configs/L1_graph_subway/):
- Graph nodes connected by edges
- Variable action space per node (requires action masking)
- Use cases: Subway systems, social networks, state machines
- Action space: Variable (max_edges + INTERACT)

**Key Concepts**:
- **Hex Grid**: Teaches coordinate system design (axial coords)
- **1D Grid**: Edge case validation (N=1 dimension)
- **Graph**: Action masking, non-Euclidean reasoning
```

---

### Step 4.2: Create comparison documentation (1 hour)

**Create**: `docs/research/COMPARISON-TOPOLOGIES-5D.md`

Document comparing all Phase 5D topologies with existing ones:

```markdown
# Substrate Topology Comparison

| Topology | Position Dim | Action Space | Distance Metric | Use Cases | Pedagogical Value |
|----------|--------------|--------------|-----------------|-----------|-------------------|
| **Square Grid** (2D) | 2 | 5 (4 dirs + INT) | Manhattan/Euclidean | Buildings, farms | ✅ (baseline) |
| **Cubic Grid** (3D) | 3 | 7 (6 dirs + INT) | Manhattan/Euclidean | Multi-story buildings | ✅✅ (vertical movement) |
| **Hexagonal Grid** | 2 | 7 (6 dirs + INT) | Hex Manhattan | Strategy games, terrain | ✅✅ (uniform distances) |
| **1D Grid** | 1 | 3 (2 dirs + INT) | Absolute difference | Pedagogical, sequences | ✅ (edge case) |
| **Graph** | 1 | Variable | Shortest path | Subway, social networks | ✅✅✅ (graph RL) |
| **Continuous 2D** | 2 | 5 (4 dirs + INT) | Euclidean | Robotics, smooth nav | ✅✅✅ (continuous control) |
| **N-Dimensional** | N | 2N+1 | Manhattan/Euclidean | Abstract state spaces | ✅✅ (multi-objective) |

...
```

---

### Step 4.3: Commit documentation (30 min)

```bash
git add CLAUDE.md docs/research/COMPARISON-TOPOLOGIES-5D.md
git commit -m "docs: add Phase 5D topology documentation

Added documentation for three new topologies:
- Hexagonal Grid: Axial coordinates, uniform distances
- 1D Grid: Linear movement, pedagogical progression
- Graph-Based: Action masking, variable action spaces

Includes comparison table of all substrate topologies.

Part of TASK-002A Phase 5D (Alternative Topologies)."
```

---

## Phase 5D Completion Checklist

### Hex Substrate (8-10 hours)
- [x] Step 1.1: Unit tests (1h)
- [x] Step 1.2: Implement HexGridSubstrate (3-4h)
- [x] Step 1.3: Config support (1h)
- [x] Step 1.4: Config pack (1h)
- [x] Step 1.5: Integration test (1h)
- [x] Step 1.6: Commit (30min)

### 1D Substrate (2-4 hours)
- [x] Step 2.1: Unit tests (30min)
- [x] Step 2.2: Implement Grid1DSubstrate (1h)
- [x] Step 2.3: Config & commit (1h)

### Graph Substrate (18-24 hours)
- [x] Step 3.1: Action masking interface (1h)
- [x] Step 3.2: Unit tests (2-3h)
- [x] Step 3.3: Implement GraphSubstrate (6-8h)
- [x] Step 3.4: Action masking in env (4-6h)
- [x] Step 3.5: Config & training (3-4h)
- [x] Step 3.6: Integration tests & commit (2-3h)

### Documentation (2-3 hours)
- [x] Step 4.1: Update CLAUDE.md (1h)
- [x] Step 4.2: Comparison docs (1h)
- [x] Step 4.3: Commit docs (30min)

**Total Estimated Effort**: 30-40 hours (including buffer)

---

## Success Criteria

Phase 5D is complete when:

**Hex Substrate:**
- [ ] All unit tests pass (12 tests)
- [ ] Integration test passes
- [ ] Training runs on hex grid
- [ ] Config pack L1_hex_strategy created

**1D Substrate:**
- [ ] All unit tests pass (11 tests)
- [ ] Training runs on 1D line
- [ ] Config pack L0_1D_line created

**Graph Substrate:**
- [ ] All unit tests pass (15 tests)
- [ ] Integration tests pass (2 tests)
- [ ] Action masking working correctly
- [ ] Training runs on graph
- [ ] Config pack L1_graph_subway created

**Integration:**
- [ ] No regressions in existing substrates
- [ ] All Phase 5D topologies coexist peacefully
- [ ] Documentation complete

---

**Phase 5D Status**: Ready for Implementation
**Recommended Order**: Hex → 1D → Graph
**Total Effort**: 30-40 hours
