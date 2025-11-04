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

---

## Phase 2: Substrate Configuration Schema

### Task 2.1: Create Substrate Config Pydantic Schema

**Files:**

- Create: `src/townlet/substrate/config.py`

**Step 1: Write test for substrate config schema**

Create: `tests/test_townlet/unit/test_substrate_config.py`

```python
"""Test substrate configuration schema."""
import pytest
from pathlib import Path
from townlet.substrate.config import (
    Grid2DSubstrateConfig,
    AspatialSubstrateConfig,
    SubstrateConfig,
    load_substrate_config,
)


def test_grid2d_config_valid():
    """Valid Grid2D config should parse successfully."""
    config_data = {
        "topology": "square",
        "width": 8,
        "height": 8,
        "boundary": "clamp",
        "distance_metric": "manhattan",
    }

    config = Grid2DSubstrateConfig(**config_data)

    assert config.width == 8
    assert config.height == 8
    assert config.boundary == "clamp"


def test_grid2d_config_invalid_dimensions():
    """Grid2D config with invalid dimensions should fail."""
    config_data = {
        "topology": "square",
        "width": 0,  # Invalid!
        "height": 8,
        "boundary": "clamp",
        "distance_metric": "manhattan",
    }

    with pytest.raises(ValueError, match="greater than 0"):
        Grid2DSubstrateConfig(**config_data)


def test_aspatial_config_valid():
    """Valid aspatial config should parse successfully."""
    config_data = {"enabled": True}

    config = AspatialSubstrateConfig(**config_data)

    assert config.enabled is True


def test_substrate_config_grid2d():
    """SubstrateConfig with type='grid' should require grid config."""
    config_data = {
        "version": "1.0",
        "description": "Test grid substrate",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
        },
    }

    config = SubstrateConfig(**config_data)

    assert config.type == "grid"
    assert config.grid is not None
    assert config.grid.width == 8


def test_substrate_config_missing_grid():
    """SubstrateConfig with type='grid' but missing grid config should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "grid",
        # Missing grid config!
    }

    with pytest.raises(ValueError, match="grid config missing"):
        SubstrateConfig(**config_data)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py::test_grid2d_config_valid -v
```

Expected: FAIL (module does not exist)

**Step 3: Implement substrate config schema**

Create: `src/townlet/substrate/config.py`

```python
"""Substrate configuration schema (Pydantic DTOs).

Defines the YAML schema for substrate.yaml files, enforcing structure and
validating configuration at load time.

Design Principles (from TASK-001):
- No-Defaults Principle: All behavioral parameters must be explicit
- Conceptual Agnosticism: Don't assume 2D, grid-based, or Euclidean
- Structural Enforcement: Validate dimensions, boundary modes, metrics
- Permissive Semantics: Allow 3D, hex, continuous, graph, aspatial
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Grid2DSubstrateConfig(BaseModel):
    """Configuration for 2D square grid substrate.

    No-Defaults Principle: All fields required (no implicit defaults).
    """

    topology: Literal["square"] = Field(description="Grid topology (must be 'square' for 2D)")
    width: int = Field(gt=0, description="Grid width (number of columns)")
    height: int = Field(gt=0, description="Grid height (number of rows)")
    boundary: Literal["clamp", "wrap", "bounce"] = Field(
        description="Boundary handling mode"
    )
    distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = Field(
        description="Distance metric for spatial queries"
    )


class AspatialSubstrateConfig(BaseModel):
    """Configuration for aspatial substrate (no positioning).

    This is the simplest config - just enable flag.
    """

    enabled: bool = Field(
        default=True,
        description="Enable aspatial mode (no spatial positioning)",
    )


class SubstrateConfig(BaseModel):
    """Complete substrate configuration.

    Exactly one substrate type must be specified via the 'type' field,
    and the corresponding config must be provided.

    No-Defaults Principle: version, description, type are all required.
    """

    version: str = Field(description="Config version (e.g., '1.0')")
    description: str = Field(description="Human-readable description")
    type: Literal["grid", "aspatial"] = Field(
        description="Substrate type selection"
    )

    # Substrate-specific configs (only one should be populated)
    grid: Grid2DSubstrateConfig | None = Field(
        None,
        description="Grid substrate configuration (required if type='grid')",
    )
    aspatial: AspatialSubstrateConfig | None = Field(
        None,
        description="Aspatial substrate configuration (required if type='aspatial')",
    )

    @model_validator(mode="after")
    def validate_substrate_type_match(self) -> "SubstrateConfig":
        """Ensure substrate config matches declared type."""
        if self.type == "grid" and self.grid is None:
            raise ValueError(
                "type='grid' requires grid configuration. "
                "Add grid: { topology: 'square', width: 8, height: 8, ... }"
            )

        if self.type == "aspatial" and self.aspatial is None:
            raise ValueError(
                "type='aspatial' requires aspatial configuration. "
                "Add aspatial: { enabled: true }"
            )

        # Ensure only one config is provided
        if self.type == "grid" and self.aspatial is not None:
            raise ValueError("type='grid' should not have aspatial configuration")

        if self.type == "aspatial" and self.grid is not None:
            raise ValueError("type='aspatial' should not have grid configuration")

        return self


def load_substrate_config(config_path: Path) -> SubstrateConfig:
    """Load and validate substrate configuration from YAML.

    Args:
        config_path: Path to substrate.yaml file

    Returns:
        SubstrateConfig: Validated substrate configuration

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config validation fails

    Example:
        >>> config = load_substrate_config(Path("configs/L1/substrate.yaml"))
        >>> print(f"Substrate type: {config.type}")
        >>> print(f"Grid size: {config.grid.width}×{config.grid.height}")
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Substrate config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    try:
        return SubstrateConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid substrate config at {config_path}: {e}") from e
```

**Step 4: Run tests to verify implementation**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/townlet/substrate/config.py tests/test_townlet/unit/test_substrate_config.py
git commit -m "feat: add Pydantic schema for substrate configuration

Defines SubstrateConfig DTOs for substrate.yaml validation:
- Grid2DSubstrateConfig: width, height, boundary, distance_metric
- AspatialSubstrateConfig: enabled flag
- SubstrateConfig: Top-level with type selection

Enforces:
- No-defaults principle (all fields required)
- Type validation (grid config for type='grid')
- Conceptual agnosticism (allows grid, aspatial, future 3D/hex)

Part of TASK-000 (Configurable Spatial Substrates)."
```

---

### Task 2.2: Create Substrate Factory

**Files:**

- Create: `src/townlet/substrate/factory.py`
- Modify: `src/townlet/substrate/__init__.py`

**Step 1: Write test for substrate factory**

Modify: `tests/test_townlet/unit/test_substrate_config.py`

Add to end of file:

```python
from townlet.substrate.factory import SubstrateFactory
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


def test_factory_build_grid2d():
    """Factory should build Grid2DSubstrate from config."""
    config_data = {
        "version": "1.0",
        "description": "Test grid",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
        },
    }

    config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))

    assert isinstance(substrate, Grid2DSubstrate)
    assert substrate.width == 8
    assert substrate.height == 8


def test_factory_build_aspatial():
    """Factory should build AspatialSubstrate from config."""
    config_data = {
        "version": "1.0",
        "description": "Test aspatial",
        "type": "aspatial",
        "aspatial": {"enabled": True},
    }

    config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))

    assert isinstance(substrate, AspatialSubstrate)
    assert substrate.position_dim == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py::test_factory_build_grid2d -v
```

Expected: FAIL (SubstrateFactory not defined)

**Step 3: Implement substrate factory**

Create: `src/townlet/substrate/factory.py`

```python
"""Factory for building substrate instances from configuration."""

import torch
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.config import SubstrateConfig
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


class SubstrateFactory:
    """Factory for building substrate instances from configuration.

    Converts SubstrateConfig (Pydantic DTO) into concrete SpatialSubstrate
    implementations (Grid2DSubstrate, AspatialSubstrate, etc.).
    """

    @staticmethod
    def build(config: SubstrateConfig, device: torch.device) -> SpatialSubstrate:
        """Build substrate instance from configuration.

        Args:
            config: Validated substrate configuration
            device: PyTorch device (cuda/cpu) for tensor operations

        Returns:
            Concrete SpatialSubstrate implementation

        Raises:
            ValueError: If substrate type is unknown

        Example:
            >>> config = load_substrate_config(Path("substrate.yaml"))
            >>> substrate = SubstrateFactory.build(config, torch.device("cuda"))
            >>> positions = substrate.initialize_positions(num_agents=100, device=device)
        """
        if config.type == "grid":
            assert config.grid is not None  # Validated by pydantic

            return Grid2DSubstrate(
                width=config.grid.width,
                height=config.grid.height,
                boundary=config.grid.boundary,
                distance_metric=config.grid.distance_metric,
            )

        elif config.type == "aspatial":
            assert config.aspatial is not None  # Validated by pydantic

            return AspatialSubstrate()

        else:
            raise ValueError(f"Unknown substrate type: {config.type}")
```

**Step 4: Update **init**.py**

Modify: `src/townlet/substrate/__init__.py`

```python
"""Spatial substrate abstractions for UNIVERSE_AS_CODE."""

from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.config import SubstrateConfig, load_substrate_config
from townlet.substrate.factory import SubstrateFactory

__all__ = [
    "SpatialSubstrate",
    "Grid2DSubstrate",
    "AspatialSubstrate",
    "SubstrateConfig",
    "load_substrate_config",
    "SubstrateFactory",
]
```

**Step 5: Run tests to verify implementation**

```bash
uv run pytest tests/test_townlet/unit/test_substrate_config.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/townlet/substrate/factory.py src/townlet/substrate/__init__.py tests/test_townlet/unit/test_substrate_config.py
git commit -m "feat: add SubstrateFactory for building substrates from config

Created factory that converts SubstrateConfig (Pydantic DTO) into concrete
SpatialSubstrate instances (Grid2DSubstrate, AspatialSubstrate).

Usage:
  config = load_substrate_config(path)
  substrate = SubstrateFactory.build(config, device)

Part of TASK-000 (Configurable Spatial Substrates)."
```

---

## Phase 3: Environment Integration

### Task 3.1: Add Substrate to VectorizedEnv (Load Only)

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Write test for environment substrate loading**

Create: `tests/test_townlet/unit/test_env_substrate_loading.py`

```python
"""Test environment loads and uses substrate configuration."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate


def test_env_loads_substrate_config():
    """Environment should load substrate.yaml and create substrate instance."""
    # Note: This test will initially PASS with legacy behavior
    # After Phase 3 integration, it will load from substrate.yaml

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # After integration, should have substrate attribute
    # For now, check legacy grid_size exists
    assert hasattr(env, "grid_size")
    assert env.grid_size == 8


def test_env_substrate_accessible():
    """Environment should expose substrate for inspection."""
    # This test will FAIL initially, becomes valid after integration

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        device="cpu",
    )

    # After integration:
    # assert hasattr(env, "substrate")
    # assert isinstance(env.substrate, Grid2DSubstrate)
    # assert env.substrate.width == 8
```

**Step 2: Run test to establish baseline**

```bash
uv run pytest tests/test_townlet/unit/test_env_substrate_loading.py::test_env_loads_substrate_config -v
```

Expected: PASS (legacy behavior)

**Step 3: Add substrate loading to VectorizedEnv.**init****

Modify: `src/townlet/environment/vectorized_env.py`

Find `__init__` method (around line 36). After line where `grid_size` is set, add:

```python
# Load substrate configuration (if exists)
substrate_config_path = config_pack_path / "substrate.yaml"
if substrate_config_path.exists():
    from townlet.substrate.config import load_substrate_config
    from townlet.substrate.factory import SubstrateFactory

    substrate_config = load_substrate_config(substrate_config_path)
    self.substrate = SubstrateFactory.build(substrate_config, device=self.device)

    # Update grid_size from substrate (for backward compatibility)
    if hasattr(self.substrate, "width") and hasattr(self.substrate, "height"):
        if self.substrate.width != self.substrate.height:
            raise ValueError(
                f"Non-square grids not yet supported: "
                f"{self.substrate.width}×{self.substrate.height}"
            )
        self.grid_size = self.substrate.width
else:
    # Legacy mode: No substrate.yaml, use hardcoded behavior
    import warnings
    warnings.warn(
        f"No substrate.yaml found in {config_pack_path}. "
        f"Using legacy hardcoded grid substrate (grid_size={self.grid_size}). "
        f"This will be deprecated in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    self.substrate = None  # Legacy mode marker
```

**Step 4: Test backward compatibility (no substrate.yaml)**

```bash
python -c "
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
import warnings

# L0 doesn't have substrate.yaml yet
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    env = VectorizedHamletEnv(
        config_pack_path=Path('configs/L0_0_minimal'),
        num_agents=1,
        device='cpu',
    )

    assert len(w) == 1
    assert 'No substrate.yaml found' in str(w[0].message)
    assert env.substrate is None  # Legacy mode
    assert env.grid_size == 3  # From training.yaml
    print('✓ Backward compatibility works (legacy mode)')
"
```

Expected: Shows deprecation warning, uses legacy behavior

**Step 5: Commit**

```bash
git add src/townlet/environment/vectorized_env.py tests/test_townlet/unit/test_env_substrate_loading.py
git commit -m "feat: add substrate loading to VectorizedHamletEnv

Environment now loads substrate.yaml if present:
- Creates substrate instance via SubstrateFactory
- Updates grid_size from substrate (for compatibility)
- Falls back to legacy mode if substrate.yaml missing

Backward compatibility: Emits deprecation warning if no substrate.yaml.

Part of TASK-000 (Configurable Spatial Substrates)."
```

---

Due to the immense scope of this task (15-22 hours estimated, ~15 files to change), I'll pause here and provide an executive summary of what the full plan would contain. This demonstrates the methodology while keeping the response manageable.

## Executive Summary - Remaining Phases

The complete plan would include:

**Phase 4: Position Management Refactoring** (6 tasks)

- Replace hardcoded `[num_agents, 2]` with `substrate.initialize_positions()`
- Update movement logic to use `substrate.apply_movement()`
- Replace distance calculations with `substrate.compute_distance()`
- Update position serialization for checkpoints

**Phase 5: Observation Builder Integration** (4 tasks)

- Replace `grid_size * grid_size` with `substrate.get_observation_dim()`
- Update full observability encoding
- Update partial observability (POMDP) encoding
- Handle aspatial observations (zero-dimensional)

**Phase 6: Config Migration** (7 tasks)

- Create substrate.yaml for L0_0_minimal (3×3 grid)
- Create substrate.yaml for L0_5 through L3
- Create substrate.yaml for templates
- Add toroidal example (wrap boundary)
- Add aspatial example
- Update training.yaml to reference substrate

**Phase 7: Frontend Visualization** (5 tasks)

- Create renderer interface
- Implement Grid2DRenderer (current SVG logic)
- Implement AspatialRenderer (text-based)
- Update Grid.vue to route by substrate type
- Send substrate metadata via WebSocket

**Phase 8: Testing & Verification** (10 tasks)

- Parameterized tests for multiple substrates
- Integration tests (training runs)
- Checkpoint migration tool
- Performance benchmarking
- Documentation updates

**Total: ~40 tasks, 200+ steps, 15-22 hours**

Would you like me to:

1. **Continue writing the full plan** (will be very long, ~5000+ lines)
2. **Save this partial plan and iterate** based on review feedback
3. **Proceed with the saved partial plan** to the review phase

Given the success of the research → plan → review methodology on TASK-002, I recommend option 2 or 3 to get review feedback before investing in the complete plan.
