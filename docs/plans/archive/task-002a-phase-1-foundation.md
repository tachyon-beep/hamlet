# TASK-000: Configurable Spatial Substrates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Abstract the hardcoded 2D square grid substrate into a configurable system supporting 2D/3D grids, hexagonal grids, graph topologies, and aspatial (pure state machine) universes.

**Architecture:** Create abstract `SpatialSubstrate` interface with concrete implementations (Grid2D, Grid3D, Aspatial), integrate with environment/observation/network layers, migrate all configs to `substrate.yaml` schema.

**Tech Stack:** Python 3.11+, Pydantic 2.x, PyTorch, YAML, Abstract Base Classes

**Research Findings Summary:**

- Spatial substrate hardcoded in ~15 core files
- Position tensors always `[num_agents, 2]` shape
- Manhattan distance used in 4 locations
- Observation dim depends on `grid_size²`
- Frontend assumes 2D SVG rendering
- Estimated effort: 15-22 hours

**Key Insight:** Meters (bars) are the true universe - spatial substrate is just an optional overlay for positioning and navigation.

---

## Phase 0: Research Validation and Setup

### Task 0.1: Verify Research Findings

**Files:**

- Read: `docs/research/2025-11-04-spatial-substrates-research.md`
- Verify: Key files from research report

**Step 1: Read research report**

```bash
cd /home/john/hamlet
cat docs/research/2025-11-04-spatial-substrates-research.md | head -100
```

Expected: Research report exists with comprehensive findings

**Step 2: Spot-check critical findings**

Verify position shape hardcoding:

```bash
grep -n "torch.zeros((self.num_agents, 2)" src/townlet/environment/vectorized_env.py
```

Expected: Line 160 shows `[num_agents, 2]` hardcoding

**Step 3: Verify distance calculation pattern**

```bash
grep -n "torch.abs(self.positions - affordance_pos).sum(dim=1)" src/townlet/environment/
```

Expected: 4 locations found (vectorized_env.py, observation_builder.py)

**Step 4: Count config files needing migration**

```bash
ls configs/*/training.yaml | wc -l
```

Expected: ~7 config packs (L0, L0.5, L1, L2, L3, templates, test)

**Step 5: Document verification results**

Create checklist of validated findings vs research claims.

---

## Phase 1: Abstract Substrate Interface (Foundation)

### Task 1.1: Create Substrate Module Structure

**Files:**

- Create: `src/townlet/substrate/__init__.py`
- Create: `src/townlet/substrate/base.py`

**Step 1: Write test for substrate module import**

Create: `tests/test_townlet/unit/test_substrate_base.py`

```python
"""Test spatial substrate abstract interface."""
import pytest
from townlet.substrate.base import SpatialSubstrate


def test_substrate_module_exists():
    """Substrate module should be importable."""
    assert SpatialSubstrate is not None


def test_substrate_is_abstract():
    """SpatialSubstrate should not be instantiable directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        SpatialSubstrate()
```

**Step 2: Run test to verify it fails**

```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_base.py -v
```

Expected: FAIL (module does not exist)

**Step 3: Create substrate module structure**

Create: `src/townlet/substrate/__init__.py`

```python
"""Spatial substrate abstractions for UNIVERSE_AS_CODE.

The spatial substrate defines the coordinate system, topology, and distance metric
for agent positioning and navigation. This is an OPTIONAL component - aspatial
universes (pure resource management) are perfectly valid.
"""

from townlet.substrate.base import SpatialSubstrate

__all__ = ["SpatialSubstrate"]
```

**Step 4: Create abstract substrate interface**

Create: `src/townlet/substrate/base.py`

```python
"""Abstract base class for spatial substrates."""

from abc import ABC, abstractmethod
import torch


class SpatialSubstrate(ABC):
    """Abstract interface for spatial substrates.

    A spatial substrate defines:
    - How positions are represented (dimensionality, dtype)
    - How positions are initialized (random, fixed, etc.)
    - How movement is applied (deltas, boundaries)
    - How distance is computed (Manhattan, Euclidean, graph distance)
    - How positions are encoded in observations

    Key insight: The substrate is OPTIONAL. Aspatial universes (pure state
    machines without positioning) are valid and reveal that meters (bars)
    are the true universe.

    Design Principles:
    - Conceptual Agnosticism: Don't assume 2D, Euclidean, or grid-based
    - Permissive Semantics: Allow 3D, hexagonal, continuous, graph, aspatial
    - Structural Enforcement: Validate tensor shapes, boundary behaviors
    """

    @property
    @abstractmethod
    def position_dim(self) -> int:
        """Dimensionality of position vectors.

        Returns:
            0 for aspatial (no positioning)
            2 for 2D grids
            3 for 3D grids
            N for N-dimensional continuous spaces
        """
        pass

    @abstractmethod
    def initialize_positions(
        self, num_agents: int, device: torch.device
    ) -> torch.Tensor:
        """Initialize random positions for agents.

        Args:
            num_agents: Number of agents to initialize
            device: PyTorch device (cuda/cpu)

        Returns:
            Tensor of shape [num_agents, position_dim]
            For aspatial substrates: [num_agents, 0]
        """
        pass

    @abstractmethod
    def apply_movement(
        self,
        positions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Apply movement deltas to positions, respecting boundaries.

        Args:
            positions: [num_agents, position_dim] current positions
            deltas: [num_agents, position_dim] movement deltas

        Returns:
            [num_agents, position_dim] new positions after movement

        Boundary handling (clamp, wrap, bounce) is substrate-specific.
        """
        pass

    @abstractmethod
    def compute_distance(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance between positions.

        Args:
            pos1: [num_agents, position_dim] or [position_dim]
            pos2: [num_agents, position_dim] or [position_dim]

        Returns:
            [num_agents] tensor of distances

        Distance metric is substrate-specific:
        - Grid: Manhattan, Euclidean, or Chebyshev
        - Graph: Shortest path distance
        - Aspatial: Zero (no meaningful distance)
        """
        pass

    @abstractmethod
    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions and affordances into observation space.

        Args:
            positions: [num_agents, position_dim] agent positions
            affordances: {name: [position_dim]} affordance positions

        Returns:
            [num_agents, observation_dim] position encoding

        observation_dim is substrate-specific:
        - Grid2D (8×8): 64 (one-hot grid cells)
        - Grid3D (8×8×3): 192 (one-hot 3D cells)
        - Aspatial: 0 (no position encoding)
        """
        pass

    @abstractmethod
    def get_observation_dim(self) -> int:
        """Return the dimensionality of position encoding in observations.

        Returns:
            Number of features in position encoding:
            - Grid2D: width × height
            - Grid3D: width × height × depth
            - Aspatial: 0
        """
        pass

    @abstractmethod
    def get_valid_neighbors(
        self,
        position: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get valid neighbor positions for action validation.

        Args:
            position: [position_dim] single position

        Returns:
            List of [position_dim] neighbor positions

        Used for action masking (boundary checks).
        """
        pass

    @abstractmethod
    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        """Check if agents are on the target position (for interactions).

        Args:
            agent_positions: [num_agents, position_dim]
            target_position: [position_dim]

        Returns:
            [num_agents] bool tensor (True if on target)

        For discrete grids: exact match
        For continuous spaces: proximity threshold
        For aspatial: always True (no positioning concept)
        """
        pass
```

**Step 5: Run test to verify import succeeds**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py -v
```

Expected: PASS (both tests pass)

**Step 6: Commit**

```bash
git add src/townlet/substrate/ tests/test_townlet/unit/test_substrate_base.py
git commit -m "feat: add abstract SpatialSubstrate interface

Created abstract base class for spatial substrates with 8 core methods:
- position_dim: Dimensionality (0 for aspatial, 2 for 2D, 3 for 3D)
- initialize_positions: Random agent placement
- apply_movement: Movement with boundary handling
- compute_distance: Substrate-specific distance metric
- encode_observation: Position encoding for obs space
- get_observation_dim: Position encoding size
- get_valid_neighbors: For action masking
- is_on_position: Exact/proximity check for interactions

Key insight: Substrate is OPTIONAL - aspatial universes are valid.

Part of TASK-000 (Configurable Spatial Substrates)."
```

---

### Task 1.2: Implement Grid2DSubstrate (Current Behavior)

**Files:**

- Create: `src/townlet/substrate/grid2d.py`
- Modify: `src/townlet/substrate/__init__.py`

**Step 1: Write test for Grid2DSubstrate**

Modify: `tests/test_townlet/unit/test_substrate_base.py`

Add to end of file:

```python
from townlet.substrate.grid2d import Grid2DSubstrate


def test_grid2d_substrate_creation():
    """Grid2DSubstrate should instantiate with width/height."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    assert substrate.position_dim == 2
    assert substrate.get_observation_dim() == 64  # 8×8


def test_grid2d_initialize_positions():
    """Grid2D should initialize random positions in valid range."""
    substrate = Grid2DSubstrate(width=8, height=8)

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    assert positions.shape == (10, 2)
    assert torch.all(positions >= 0)
    assert torch.all(positions[:, 0] < 8)  # x < width
    assert torch.all(positions[:, 1] < 8)  # y < height


def test_grid2d_apply_movement_clamp():
    """Grid2D with clamp boundary should keep agents in bounds."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")

    # Agent at top-left corner tries to move up-left
    positions = torch.tensor([[0, 0]], dtype=torch.long)
    deltas = torch.tensor([[-1, -1]], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should clamp to [0, 0] (can't go negative)
    assert torch.equal(new_positions, torch.tensor([[0, 0]], dtype=torch.long))


def test_grid2d_compute_distance_manhattan():
    """Grid2D should compute Manhattan distance correctly."""
    substrate = Grid2DSubstrate(width=8, height=8, distance_metric="manhattan")

    pos1 = torch.tensor([[0, 0], [3, 4]], dtype=torch.long)
    pos2 = torch.tensor([5, 7], dtype=torch.long)  # Single position

    distances = substrate.compute_distance(pos1, pos2)

    # Manhattan: |0-5| + |0-7| = 12, |3-5| + |4-7| = 5
    assert torch.equal(distances, torch.tensor([12, 5], dtype=torch.long))


def test_grid2d_is_on_position():
    """Grid2D should check exact position match."""
    substrate = Grid2DSubstrate(width=8, height=8)

    agent_positions = torch.tensor([[3, 4], [5, 7], [3, 4]], dtype=torch.long)
    target_position = torch.tensor([3, 4], dtype=torch.long)

    on_target = substrate.is_on_position(agent_positions, target_position)

    assert torch.equal(on_target, torch.tensor([True, False, True]))
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_grid2d_substrate_creation -v
```

Expected: FAIL (Grid2DSubstrate not defined)

**Step 3: Implement Grid2DSubstrate**

Create: `src/townlet/substrate/grid2d.py`

```python
"""2D square grid substrate (replicates current HAMLET behavior)."""

import torch
from townlet.substrate.base import SpatialSubstrate


class Grid2DSubstrate(SpatialSubstrate):
    """2D square grid with configurable boundaries and distance metrics.

    This replicates the current hardcoded behavior of HAMLET (vectorized_env.py).

    Coordinate system:
    - positions: [x, y] where x is column, y is row
    - Origin: top-left corner is [0, 0]
    - x increases rightward, y increases downward

    Supported boundaries:
    - clamp: Hard walls (default, current behavior)
    - wrap: Toroidal wraparound (Pac-Man style)
    - bounce: Elastic reflection

    Supported distance metrics:
    - manhattan: |x1-x2| + |y1-y2| (default, current behavior)
    - euclidean: sqrt((x1-x2)² + (y1-y2)²)
    - chebyshev: max(|x1-x2|, |y1-y2|)
    """

    def __init__(
        self,
        width: int,
        height: int,
        boundary: str = "clamp",
        distance_metric: str = "manhattan",
    ):
        """Initialize 2D grid substrate.

        Args:
            width: Grid width (number of columns)
            height: Grid height (number of rows)
            boundary: Boundary mode ("clamp", "wrap", "bounce")
            distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive: width={width}, height={height}")

        if boundary not in ("clamp", "wrap", "bounce"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.width = width
        self.height = height
        self.boundary = boundary
        self.distance_metric = distance_metric

    @property
    def position_dim(self) -> int:
        """2D grid has 2-dimensional positions."""
        return 2

    def initialize_positions(
        self, num_agents: int, device: torch.device
    ) -> torch.Tensor:
        """Initialize random positions uniformly across the grid."""
        return torch.stack(
            [
                torch.randint(0, self.width, (num_agents,), device=device),
                torch.randint(0, self.height, (num_agents,), device=device),
            ],
            dim=1,
        )

    def apply_movement(
        self,
        positions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Apply movement deltas with boundary handling."""
        new_positions = positions + deltas

        if self.boundary == "clamp":
            # Hard walls: clamp to valid range
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)

        elif self.boundary == "wrap":
            # Toroidal wraparound
            new_positions[:, 0] = new_positions[:, 0] % self.width
            new_positions[:, 1] = new_positions[:, 1] % self.height

        elif self.boundary == "bounce":
            # Elastic reflection: if out of bounds, stay in place
            out_of_bounds_x = (new_positions[:, 0] < 0) | (new_positions[:, 0] >= self.width)
            out_of_bounds_y = (new_positions[:, 1] < 0) | (new_positions[:, 1] >= self.height)

            new_positions[out_of_bounds_x, 0] = positions[out_of_bounds_x, 0]
            new_positions[out_of_bounds_y, 1] = positions[out_of_bounds_y, 1]

        return new_positions

    def compute_distance(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance between positions using configured metric."""
        # Handle broadcasting: pos2 might be single position [2] or batch [N, 2]
        if pos2.dim() == 1:
            pos2 = pos2.unsqueeze(0)  # [2] → [1, 2]

        if self.distance_metric == "manhattan":
            # L1 distance: |x1-x2| + |y1-y2|
            return torch.abs(pos1 - pos2).sum(dim=-1)

        elif self.distance_metric == "euclidean":
            # L2 distance: sqrt((x1-x2)² + (y1-y2)²)
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))

        elif self.distance_metric == "chebyshev":
            # L∞ distance: max(|x1-x2|, |y1-y2|)
            return torch.abs(pos1 - pos2).max(dim=-1)[0]

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode positions as one-hot grid cells.

        Creates a grid_size × grid_size one-hot encoding where:
        - Affordances are marked with 1.0
        - Agent position adds 1.0 (so agent on affordance = 2.0)
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Initialize grid encoding [num_agents, width * height]
        grid_encoding = torch.zeros(num_agents, self.width * self.height, device=device)

        # Mark affordance positions
        for affordance_pos in affordances.values():
            affordance_flat_idx = affordance_pos[1] * self.width + affordance_pos[0]
            grid_encoding[:, affordance_flat_idx] = 1.0

        # Mark agent positions (add 1.0, so overlaps become 2.0)
        flat_indices = positions[:, 1] * self.width + positions[:, 0]
        ones = torch.ones(num_agents, 1, device=device)
        grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), ones)

        return grid_encoding

    def get_observation_dim(self) -> int:
        """Grid observation is width × height (flattened)."""
        return self.width * self.height

    def get_valid_neighbors(
        self,
        position: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get valid 4-connected neighbors (UP, DOWN, LEFT, RIGHT).

        For clamp boundary: only returns in-bounds neighbors
        For wrap/bounce: returns all 4 neighbors (wrapping/bouncing handled in apply_movement)
        """
        x, y = position[0].item(), position[1].item()

        neighbors = [
            torch.tensor([x, y - 1], dtype=torch.long, device=position.device),  # UP
            torch.tensor([x, y + 1], dtype=torch.long, device=position.device),  # DOWN
            torch.tensor([x - 1, y], dtype=torch.long, device=position.device),  # LEFT
            torch.tensor([x + 1, y], dtype=torch.long, device=position.device),  # RIGHT
        ]

        if self.boundary == "clamp":
            # Filter out-of-bounds neighbors
            neighbors = [
                n for n in neighbors
                if 0 <= n[0] < self.width and 0 <= n[1] < self.height
            ]

        return neighbors

    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        """Check if agents are exactly on target position (exact match)."""
        # For discrete grids, agents must be on exact cell
        distances = self.compute_distance(agent_positions, target_position)
        return distances == 0
```

**Step 4: Update **init**.py**

Modify: `src/townlet/substrate/__init__.py`

```python
"""Spatial substrate abstractions for UNIVERSE_AS_CODE."""

from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate

__all__ = ["SpatialSubstrate", "Grid2DSubstrate"]
```

**Step 5: Run tests to verify implementation**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/townlet/substrate/grid2d.py src/townlet/substrate/__init__.py tests/test_townlet/unit/test_substrate_base.py
git commit -m "feat: implement Grid2DSubstrate (replicates current behavior)

Implemented 2D square grid substrate with:
- Configurable width/height (replaces single grid_size)
- 3 boundary modes: clamp (current), wrap (toroidal), bounce (elastic)
- 3 distance metrics: manhattan (current), euclidean, chebyshev
- One-hot grid encoding for observations
- 4-connected neighbor detection

All methods tested. Replicates current vectorized_env.py behavior.

Part of TASK-000 (Configurable Spatial Substrates)."
```

---

### Task 1.3: Implement AspatialSubstrate (State Machine)

**Files:**

- Create: `src/townlet/substrate/aspatial.py`
- Modify: `src/townlet/substrate/__init__.py`

**Step 1: Write test for AspatialSubstrate**

Modify: `tests/test_townlet/unit/test_substrate_base.py`

Add to end of file:

```python
from townlet.substrate.aspatial import AspatialSubstrate


def test_aspatial_substrate_creation():
    """AspatialSubstrate should represent no positioning."""
    substrate = AspatialSubstrate()

    assert substrate.position_dim == 0  # No position!
    assert substrate.get_observation_dim() == 0  # No position encoding


def test_aspatial_initialize_positions():
    """Aspatial should return empty position tensors."""
    substrate = AspatialSubstrate()

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    assert positions.shape == (10, 0)  # Empty position vectors


def test_aspatial_compute_distance():
    """Aspatial should return zero distance (no spatial meaning)."""
    substrate = AspatialSubstrate()

    pos1 = torch.zeros((5, 0))  # 5 agents with no position
    pos2 = torch.zeros((0,))    # Target with no position

    distances = substrate.compute_distance(pos1, pos2)

    assert distances.shape == (5,)
    assert torch.all(distances == 0)  # All distances are zero


def test_aspatial_is_on_position():
    """Aspatial should always return True (no positioning concept)."""
    substrate = AspatialSubstrate()

    agent_positions = torch.zeros((10, 0))
    target_position = torch.zeros((0,))

    on_target = substrate.is_on_position(agent_positions, target_position)

    assert torch.all(on_target == True)  # All agents are "everywhere"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py::test_aspatial_substrate_creation -v
```

Expected: FAIL (AspatialSubstrate not defined)

**Step 3: Implement AspatialSubstrate**

Create: `src/townlet/substrate/aspatial.py`

```python
"""Aspatial substrate (no positioning - pure state machine)."""

import torch
from townlet.substrate.base import SpatialSubstrate


class AspatialSubstrate(SpatialSubstrate):
    """Substrate with no spatial positioning (pure state machine).

    Key insight: The meters (bars) are the true universe. Spatial positioning
    is just an OPTIONAL overlay for navigation and affordance placement.

    An aspatial universe reveals this truth:
    - No concept of "position" or "distance"
    - All affordances are "everywhere and nowhere"
    - Agents interact directly without movement
    - Pure resource management (no navigation)

    Pedagogical value:
    - Reveals that positioning is a design choice, not fundamental
    - Simplifies universe design (no grid to configure)
    - Focuses learning on resource management, not navigation

    Use cases:
    - Abstract planning problems (no physical space)
    - Resource management games (Factorio-like)
    - State machines (FSM without spatial component)
    """

    @property
    def position_dim(self) -> int:
        """Aspatial has zero-dimensional positions (no positioning)."""
        return 0

    def initialize_positions(
        self, num_agents: int, device: torch.device
    ) -> torch.Tensor:
        """Return empty position tensors (agents have no position)."""
        return torch.zeros((num_agents, 0), dtype=torch.long, device=device)

    def apply_movement(
        self,
        positions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """No movement possible in aspatial universe (return unchanged)."""
        return positions

    def compute_distance(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """Return zero distance (no spatial meaning in aspatial universe)."""
        num_agents = pos1.shape[0]
        return torch.zeros(num_agents, device=pos1.device)

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return empty observation encoding (no position to encode)."""
        num_agents = positions.shape[0]
        device = positions.device
        return torch.zeros((num_agents, 0), device=device)

    def get_observation_dim(self) -> int:
        """Aspatial has zero observation dimensions (no position encoding)."""
        return 0

    def get_valid_neighbors(
        self,
        position: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return empty list (no spatial neighbors in aspatial universe)."""
        return []

    def is_on_position(
        self,
        agent_positions: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        """Return all True (agents are 'everywhere' in aspatial universe).

        In aspatial universes, there's no concept of being "on" a position.
        All agents can interact with all affordances at any time.
        """
        num_agents = agent_positions.shape[0]
        return torch.ones(num_agents, dtype=torch.bool, device=agent_positions.device)
```

**Step 4: Update **init**.py**

Modify: `src/townlet/substrate/__init__.py`

```python
"""Spatial substrate abstractions for UNIVERSE_AS_CODE."""

from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate

__all__ = ["SpatialSubstrate", "Grid2DSubstrate", "AspatialSubstrate"]
```

**Step 5: Run tests to verify implementation**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_base.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/townlet/substrate/aspatial.py src/townlet/substrate/__init__.py tests/test_townlet/unit/test_substrate_base.py
git commit -m "feat: implement AspatialSubstrate (pure state machine)

Implemented aspatial substrate with:
- position_dim = 0 (no positioning concept)
- Empty position tensors [N, 0]
- Zero distance (no spatial meaning)
- is_on_position always True (agents are 'everywhere')

Key insight: Reveals that meters are the true universe, positioning is optional.

Use cases: Abstract planning, resource management, state machines.

Part of TASK-000 (Configurable Spatial Substrates)."
```
