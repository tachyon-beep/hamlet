# Phase 5C Part 2: N-Dimensional Substrates - Detailed Implementation Plan

**Status**: Draft (Pending Review)
**Prerequisites**: Part 1 complete (observation_encoding retrofitted to existing substrates)

---

## Part 2 Overview

**Goal**: Add GridND and ContinuousND substrates for 4D+ abstract state spaces

**Estimated Time**: 12-14 hours

**Tasks**:
1. **Task 2.1**: Implement GridND Base Class (120 min)
2. **Task 2.2**: Implement ContinuousND Class (90 min)
3. **Task 2.3**: Add GridND/ContinuousND to Config Schema (45 min)
4. **Task 2.4**: Update Factory for GridND/ContinuousND (30 min)
5. **Task 2.5**: Integration Testing (90 min)
6. **Task 2.6**: Documentation (60 min)

---

## Task 2.1: Implement GridND Base Class (120 min)

**Context**: GridND handles 4D-100D grid substrates with auto-generated 2N+1 action spaces. Based on research, all core methods generalize from Grid2D/Grid3D patterns using loops over dimensions.

**Files**:
- Create: `src/townlet/substrate/gridnd.py`
- Test: `tests/test_townlet/phase5/test_gridnd_basic.py` (NEW)
- Test: `tests/test_townlet/phase5/test_gridnd_scaling.py` (NEW)
- Test: `tests/test_townlet/phase5/test_gridnd_edge_cases.py` (NEW)

### Step 2.1.1: Write failing tests for GridND initialization (10 min)

**Create**: `tests/test_townlet/phase5/test_gridnd_basic.py`

```python
"""Test GridND substrate for N-dimensional grids (N≥4)."""
import pytest
import torch
from townlet.substrate.gridnd import GridNDSubstrate


def test_gridnd_4d_initialization():
    """GridND should initialize correctly for 4D hypercube."""
    substrate = GridNDSubstrate(
        dimension_sizes=[8, 8, 8, 8],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    assert substrate.position_dim == 4
    assert substrate.position_dtype == torch.long
    assert substrate.action_space_size == 9  # 2*4 + 1
    assert substrate.get_observation_dim() == 4  # Normalized coordinates


def test_gridnd_requires_minimum_4_dimensions():
    """GridND should reject dimensions < 4."""
    with pytest.raises(ValueError, match="GridND requires at least 4 dimensions"):
        GridNDSubstrate(
            dimension_sizes=[8, 8, 8],  # Only 3D
            boundary="clamp",
            distance_metric="manhattan",
        )


def test_gridnd_validates_positive_dimensions():
    """GridND should reject non-positive dimension sizes."""
    with pytest.raises(ValueError, match="Dimension sizes must be positive"):
        GridNDSubstrate(
            dimension_sizes=[8, 8, 0, 8],  # Zero size
            boundary="clamp",
            distance_metric="manhattan",
        )


def test_gridnd_asymmetric_dimensions():
    """GridND should support different sizes per dimension."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 5, 3, 7],  # Different sizes
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    assert substrate.position_dim == 4
    assert substrate.dimension_sizes == [10, 5, 3, 7]


def test_gridnd_warns_at_10_dimensions():
    """GridND should emit warning at N≥10."""
    with pytest.warns(UserWarning, match="10 dimensions"):
        substrate = GridNDSubstrate(
            dimension_sizes=[3] * 10,
            boundary="clamp",
            distance_metric="manhattan",
        )

    assert substrate.action_space_size == 21  # 2*10 + 1
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py::test_gridnd_4d_initialization -v`

**Expected**: FAIL - ModuleNotFoundError: No module named 'townlet.substrate.gridnd'

### Step 2.1.2: Create GridND class skeleton (10 min)

**Create**: `src/townlet/substrate/gridnd.py`

```python
"""N-dimensional grid substrate (N≥4 dimensions)."""

from typing import Literal
import torch
import warnings

from townlet.substrate.base import SpatialSubstrate


class GridNDSubstrate(SpatialSubstrate):
    """N-dimensional hypercube grid for abstract state spaces.

    GridND supports 4D to 100D discrete grid substrates. For 2D/3D grids,
    use Grid2DSubstrate or Grid3DSubstrate for better ergonomics.

    Coordinate system:
    - positions: [d0, d1, d2, ..., dN] where d0 is dimension 0, etc.
    - Origin: all zeros [0, 0, 0, ...]
    - Each dimension increases positively from 0 to (size - 1)

    Observation encoding:
    - relative: Normalized coordinates [0, 1] (N dimensions)
    - scaled: Normalized + dimension sizes (2N dimensions)
    - absolute: Raw unnormalized coordinates (N dimensions)

    Use cases:
    - High-dimensional RL research
    - Abstract state space experiments
    - Transfer learning from low-D to high-D
    """

    def __init__(
        self,
        dimension_sizes: list[int],
        boundary: Literal["clamp", "wrap", "bounce", "sticky"],
        distance_metric: Literal["manhattan", "euclidean", "chebyshev"] = "manhattan",
        observation_encoding: Literal["relative", "scaled", "absolute"] = "relative",
    ):
        """Initialize N-dimensional grid substrate.

        Args:
            dimension_sizes: Size of each dimension [d0_size, d1_size, ..., dN_size]
            boundary: Boundary mode ("clamp", "wrap", "bounce", "sticky")
            distance_metric: Distance metric ("manhattan", "euclidean", "chebyshev")
            observation_encoding: Position encoding strategy ("relative", "scaled", "absolute")

        Raises:
            ValueError: If dimensions < 4 or any size <= 0

        Warnings:
            UserWarning: If dimensions >= 10 (action space size warning)
        """
        # Validate dimension count
        num_dims = len(dimension_sizes)
        if num_dims < 4:
            raise ValueError(
                f"GridND requires at least 4 dimensions, got {num_dims}. "
                f"Use Grid2DSubstrate (2D) or Grid3DSubstrate (3D) instead."
            )

        if num_dims > 100:
            raise ValueError(f"GridND dimension count ({num_dims}) exceeds limit (100)")

        # Warn at N≥10 (action space grows large)
        if num_dims >= 10:
            warnings.warn(
                f"GridND with {num_dims} dimensions has {2*num_dims+1} actions. "
                f"Large action spaces may be challenging to train. "
                f"Verify this is intentional for your research.",
                UserWarning,
            )

        # Validate dimension sizes
        for i, size in enumerate(dimension_sizes):
            if size <= 0:
                raise ValueError(
                    f"Dimension sizes must be positive. Dimension {i} has size {size}."
                )

        # Validate parameters
        if boundary not in ("clamp", "wrap", "bounce", "sticky"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        if distance_metric not in ("manhattan", "euclidean", "chebyshev"):
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        if observation_encoding not in ("relative", "scaled", "absolute"):
            raise ValueError(f"Unknown observation encoding: {observation_encoding}")

        # Store configuration
        self.dimension_sizes = dimension_sizes
        self.boundary = boundary
        self.distance_metric = distance_metric
        self.observation_encoding = observation_encoding

    @property
    def position_dim(self) -> int:
        """Return number of dimensions."""
        return len(self.dimension_sizes)

    @property
    def position_dtype(self) -> torch.dtype:
        """Grid positions are integers (discrete cells)."""
        return torch.long

    # Placeholder methods (to be implemented in following steps)
    def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError("To be implemented")

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("To be implemented")

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("To be implemented")

    def encode_observation(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("To be implemented")

    def get_observation_dim(self) -> int:
        raise NotImplementedError("To be implemented")

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError("To be implemented")

    def is_on_position(self, agent_positions: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("To be implemented")

    def get_all_positions(self) -> list[list[int]]:
        raise NotImplementedError("To be implemented")

    def supports_enumerable_positions(self) -> bool:
        return True
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py::test_gridnd_4d_initialization -v`

**Expected**: FAIL - NotImplementedError: To be implemented (initialize_positions)

### Step 2.1.3: Implement initialize_positions() (5 min)

**Modify**: `src/townlet/substrate/gridnd.py`

Replace `initialize_positions()` placeholder:

```python
def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
    """Initialize random positions uniformly across N-dimensional grid.

    Returns:
        [num_agents, N] tensor of integer positions
    """
    return torch.stack(
        [torch.randint(0, dim_size, (num_agents,), device=device, dtype=torch.long)
         for dim_size in self.dimension_sizes],
        dim=1,
    )
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py -k initialization -v`

**Expected**: PASS (initialization tests), FAIL on others (NotImplementedError)

### Step 2.1.4: Implement apply_movement() with all boundary modes (15 min)

**Modify**: `src/townlet/substrate/gridnd.py`

Replace `apply_movement()` placeholder:

```python
def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply movement deltas with boundary handling.

    Args:
        positions: [num_agents, N] current positions (long)
        deltas: [num_agents, N] movement deltas (float32)

    Returns:
        [num_agents, N] new positions after boundary handling
    """
    # Cast deltas to long for grid substrates
    new_positions = positions + deltas.long()

    # Apply boundary handling per dimension
    if self.boundary == "clamp":
        for dim_idx, dim_size in enumerate(self.dimension_sizes):
            new_positions[:, dim_idx] = torch.clamp(
                new_positions[:, dim_idx], 0, dim_size - 1
            )

    elif self.boundary == "wrap":
        for dim_idx, dim_size in enumerate(self.dimension_sizes):
            new_positions[:, dim_idx] = new_positions[:, dim_idx] % dim_size

    elif self.boundary == "bounce":
        for dim_idx, dim_size in enumerate(self.dimension_sizes):
            # Handle negative positions (reflect across 0)
            negative_mask = new_positions[:, dim_idx] < 0
            new_positions[negative_mask, dim_idx] = -new_positions[negative_mask, dim_idx]

            # Handle positions >= dim_size (reflect across upper boundary)
            exceed_mask = new_positions[:, dim_idx] >= dim_size
            new_positions[exceed_mask, dim_idx] = (
                2 * (dim_size - 1) - new_positions[exceed_mask, dim_idx]
            )

            # Safety clamp (in case of large velocities)
            new_positions[:, dim_idx] = torch.clamp(
                new_positions[:, dim_idx], 0, dim_size - 1
            )

    elif self.boundary == "sticky":
        for dim_idx, dim_size in enumerate(self.dimension_sizes):
            out_of_bounds = (new_positions[:, dim_idx] < 0) | (new_positions[:, dim_idx] >= dim_size)
            new_positions[out_of_bounds, dim_idx] = positions[out_of_bounds, dim_idx]

    return new_positions
```

**Write test** (add to `test_gridnd_basic.py`):

```python
def test_gridnd_movement_with_clamp_boundary():
    """Test movement with clamp boundary (hard walls)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
    )

    # Agent at corner [0, 0, 0, 0]
    positions = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)

    # Try to move negative (should be clamped)
    deltas = torch.tensor([[-1, -1, -1, -1]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Should stay at [0, 0, 0, 0]
    assert torch.equal(new_positions, torch.tensor([[0, 0, 0, 0]], dtype=torch.long))


def test_gridnd_movement_with_wrap_boundary():
    """Test movement with wrap boundary (toroidal)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="wrap",
        distance_metric="manhattan",
    )

    # Agent at [0, 0, 0, 0]
    positions = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)

    # Move negative (should wrap to [4, 4, 4, 4])
    deltas = torch.tensor([[-1, -1, -1, -1]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    assert torch.equal(new_positions, torch.tensor([[4, 4, 4, 4]], dtype=torch.long))
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py -k movement -v`

**Expected**: PASS

### Step 2.1.5: Implement compute_distance() for all metrics (10 min)

**Modify**: `src/townlet/substrate/gridnd.py`

Replace `compute_distance()` placeholder:

```python
def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """Compute distance between positions using configured metric.

    Handles broadcasting: pos2 can be [N] or [batch, N].

    Args:
        pos1: [batch, N] positions
        pos2: [N] or [batch, N] positions

    Returns:
        [batch] distances
    """
    # Handle broadcasting: pos2 might be single position [N] or batch [batch, N]
    if pos2.dim() == 1:
        pos2 = pos2.unsqueeze(0)  # [N] → [1, N]

    if self.distance_metric == "manhattan":
        # L1 distance: sum of absolute differences
        return torch.abs(pos1 - pos2).sum(dim=-1)

    elif self.distance_metric == "euclidean":
        # L2 distance: sqrt(sum of squared differences)
        return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))

    elif self.distance_metric == "chebyshev":
        # L∞ distance: max of absolute differences
        return torch.abs(pos1 - pos2).max(dim=-1)[0]
```

**Write test** (add to `test_gridnd_basic.py`):

```python
def test_gridnd_distance_manhattan():
    """Test Manhattan distance in 4D."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="manhattan",
    )

    pos1 = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    pos2 = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    distance = substrate.compute_distance(pos1, pos2)

    # Manhattan: |3-0| + |4-0| + |5-0| + |6-0| = 18
    assert distance[0] == 18


def test_gridnd_distance_euclidean():
    """Test Euclidean distance in 4D."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="euclidean",
    )

    pos1 = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    pos2 = torch.tensor([[3, 4, 0, 0]], dtype=torch.long)

    distance = substrate.compute_distance(pos1, pos2)

    # Euclidean: sqrt(3^2 + 4^2 + 0^2 + 0^2) = sqrt(25) = 5.0
    assert torch.allclose(distance, torch.tensor([5.0]))
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py -k distance -v`

**Expected**: PASS

### Step 2.1.6: Implement observation encoding (relative/scaled/absolute) (15 min)

**Modify**: `src/townlet/substrate/gridnd.py`

Add helper methods and replace `encode_observation()` and `get_observation_dim()` placeholders:

```python
def _encode_relative(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode positions as normalized coordinates [0, 1] per dimension."""
    num_agents = positions.shape[0]
    device = positions.device

    normalized = torch.zeros((num_agents, len(self.dimension_sizes)), dtype=torch.float32, device=device)

    for dim_idx, dim_size in enumerate(self.dimension_sizes):
        normalized[:, dim_idx] = positions[:, dim_idx].float() / max(dim_size - 1, 1)

    return normalized


def _encode_scaled(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode positions as normalized coordinates + dimension sizes."""
    num_agents = positions.shape[0]
    device = positions.device

    # Get normalized positions
    relative = self._encode_relative(positions, affordances)

    # Add dimension sizes
    sizes_tensor = torch.tensor(
        [float(size) for size in self.dimension_sizes],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).expand(num_agents, -1)

    return torch.cat([relative, sizes_tensor], dim=1)


def _encode_absolute(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode positions as raw unnormalized coordinates."""
    return positions.float()


def encode_observation(self, positions: torch.Tensor, affordances: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode agent positions into observation space.

    Args:
        positions: [num_agents, N] agent positions
        affordances: {name: [N]} affordance positions (currently unused)

    Returns:
        Encoded observations:
        - relative: [num_agents, N]
        - scaled: [num_agents, 2N]
        - absolute: [num_agents, N]
    """
    if self.observation_encoding == "relative":
        return self._encode_relative(positions, affordances)
    elif self.observation_encoding == "scaled":
        return self._encode_scaled(positions, affordances)
    elif self.observation_encoding == "absolute":
        return self._encode_absolute(positions, affordances)
    else:
        raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}")


def get_observation_dim(self) -> int:
    """Return dimensionality of position encoding.

    Returns:
        - relative: N (normalized coordinates)
        - scaled: 2N (normalized + sizes)
        - absolute: N (raw coordinates)
    """
    if self.observation_encoding == "relative":
        return len(self.dimension_sizes)
    elif self.observation_encoding == "scaled":
        return 2 * len(self.dimension_sizes)
    elif self.observation_encoding == "absolute":
        return len(self.dimension_sizes)
    else:
        raise ValueError(f"Invalid observation_encoding: {self.observation_encoding}")
```

**Write test** (add to `test_gridnd_basic.py`):

```python
def test_gridnd_observation_encoding_relative():
    """Test relative encoding (normalized [0,1])."""
    substrate = GridNDSubstrate(
        dimension_sizes=[8, 8, 8, 8],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Corner: [0, 0, 0, 0] → [0.0, 0.0, 0.0, 0.0]
    # Opposite corner: [7, 7, 7, 7] → [1.0, 1.0, 1.0, 1.0]
    positions = torch.tensor([[0, 0, 0, 0], [7, 7, 7, 7]], dtype=torch.long)

    encoded = substrate.encode_observation(positions, {})

    assert encoded.shape == (2, 4)
    assert torch.allclose(encoded[0], torch.zeros(4))
    assert torch.allclose(encoded[1], torch.ones(4))
    assert substrate.get_observation_dim() == 4


def test_gridnd_observation_encoding_scaled():
    """Test scaled encoding (normalized + sizes)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 5, 3, 7],  # Different sizes
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="scaled",
    )

    positions = torch.tensor([[5, 2, 1, 3]], dtype=torch.long)

    encoded = substrate.encode_observation(positions, {})

    # First N dims: normalized positions
    # Last N dims: dimension sizes
    assert encoded.shape == (1, 8)  # 2 * 4 dimensions

    # Verify sizes in last N dims
    assert torch.allclose(encoded[0, 4:], torch.tensor([10.0, 5.0, 3.0, 7.0]))
    assert substrate.get_observation_dim() == 8
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py -k observation -v`

**Expected**: PASS

### Step 2.1.7: Implement get_valid_neighbors() (2N cardinal neighbors) (10 min)

**Modify**: `src/townlet/substrate/gridnd.py`

Replace `get_valid_neighbors()` placeholder:

```python
def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
    """Get cardinal neighbors in N dimensions (2N neighbors).

    For each dimension, generates ±1 neighbors (positive and negative direction).

    Args:
        position: [N] position tensor or list

    Returns:
        List of [N] neighbor positions
        - For clamp boundary: only in-bounds neighbors
        - For other boundaries: all 2N neighbors (boundary handling in apply_movement)
    """
    if isinstance(position, torch.Tensor):
        coords = position.tolist()
    else:
        coords = list(position)

    neighbors = []

    # For each dimension, generate ±1 neighbors
    for dim_idx in range(len(self.dimension_sizes)):
        # Negative direction
        neighbor_neg = coords.copy()
        neighbor_neg[dim_idx] -= 1
        neighbors.append(neighbor_neg)

        # Positive direction
        neighbor_pos = coords.copy()
        neighbor_pos[dim_idx] += 1
        neighbors.append(neighbor_pos)

    if self.boundary == "clamp":
        # Filter out-of-bounds neighbors
        neighbors = [
            n for n in neighbors
            if all(0 <= n[dim_idx] < self.dimension_sizes[dim_idx]
                   for dim_idx in range(len(self.dimension_sizes)))
        ]

    return [torch.tensor(n, dtype=torch.long) for n in neighbors]
```

**Write test** (add to `test_gridnd_basic.py`):

```python
def test_gridnd_neighbors_4d_interior():
    """Test interior position has 8 neighbors (2*4) in 4D."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="manhattan",
    )

    # Interior position
    position = torch.tensor([5, 5, 5, 5], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)

    # Should have 8 neighbors (2*4 dimensions)
    assert len(neighbors) == 8

    # Verify each neighbor is ±1 in exactly one dimension
    for neighbor in neighbors:
        diff = torch.abs(neighbor - position).sum()
        assert diff == 1, "Neighbor should differ by 1 in exactly one dimension"


def test_gridnd_neighbors_4d_corner():
    """Test corner position has fewer neighbors (clamp boundary)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
    )

    # Corner position [0, 0, 0, 0]
    position = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)

    # Should have 4 neighbors (only positive directions)
    assert len(neighbors) == 4

    # All neighbors should be [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
    for neighbor in neighbors:
        assert (neighbor >= 0).all()
        assert (neighbor.sum() == 1)  # Exactly one dimension = 1
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py -k neighbors -v`

**Expected**: PASS

### Step 2.1.8: Implement is_on_position() (5 min)

**Modify**: `src/townlet/substrate/gridnd.py`

Replace `is_on_position()` placeholder:

```python
def is_on_position(self, agent_positions: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
    """Check if agents are exactly on target position (exact match in N dimensions).

    Args:
        agent_positions: [num_agents, N] agent positions
        target_position: [N] target position

    Returns:
        [num_agents] boolean tensor (True if agent on target)
    """
    return (agent_positions == target_position).all(dim=-1)
```

**Write test** (add to `test_gridnd_basic.py`):

```python
def test_gridnd_is_on_position():
    """Test exact position matching in N dimensions."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="manhattan",
    )

    agent_positions = torch.tensor([
        [5, 5, 5, 5],  # On target
        [5, 5, 5, 6],  # Off by 1 in one dimension
        [0, 0, 0, 0],  # Far from target
    ], dtype=torch.long)

    target_position = torch.tensor([5, 5, 5, 5], dtype=torch.long)

    on_position = substrate.is_on_position(agent_positions, target_position)

    assert on_position[0] == True
    assert on_position[1] == False
    assert on_position[2] == False
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_basic.py -k is_on -v`

**Expected**: PASS

### Step 2.1.9: Implement get_all_positions() with warning (10 min)

**Modify**: `src/townlet/substrate/gridnd.py`

Replace `get_all_positions()` placeholder:

```python
def get_all_positions(self) -> list[list[int]]:
    """Enumerate all positions in N-dimensional grid.

    Warning: Combinatorial explosion for high N or large grids!
    - 4D 3×3×3×3: 81 positions (manageable)
    - 10D 3^10: 59,049 positions (slow)
    - 10D 10^10: 10 billion positions (memory error)

    Returns:
        List of [N] positions

    Raises:
        MemoryError: If total positions > 10 million

    Warns:
        UserWarning: If total positions > 100,000
    """
    import itertools

    # Calculate total positions
    total_positions = 1
    for size in self.dimension_sizes:
        total_positions *= size

    # Error on absurd counts
    if total_positions > 10_000_000:
        raise MemoryError(
            f"get_all_positions() would generate {total_positions:,} positions. "
            f"This is too large for memory. Consider using initialize_positions() "
            f"for random sampling instead."
        )

    # Warn on large counts
    if total_positions > 100_000:
        warnings.warn(
            f"get_all_positions() generating {total_positions:,} positions. "
            f"This may be slow and use significant memory.",
            UserWarning,
        )

    # Generate all combinations using Cartesian product
    ranges = [range(dim_size) for dim_size in self.dimension_sizes]
    return [list(coords) for coords in itertools.product(*ranges)]
```

**Write test** (add to `test_gridnd_edge_cases.py`):

```python
"""Test edge cases and warnings for GridND."""
import pytest
import torch
from townlet.substrate.gridnd import GridNDSubstrate


def test_gridnd_get_all_positions_small_grid():
    """Test get_all_positions() on small 4D grid."""
    substrate = GridNDSubstrate(
        dimension_sizes=[2, 2, 2, 2],
        boundary="clamp",
        distance_metric="manhattan",
    )

    positions = substrate.get_all_positions()

    # 2^4 = 16 positions
    assert len(positions) == 16

    # Verify all positions are unique
    assert len(set(tuple(p) for p in positions)) == 16


def test_gridnd_get_all_positions_warns_on_large_grid():
    """Test get_all_positions() warns on large grids."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10] * 4,  # 10^4 = 10,000 positions
        boundary="clamp",
        distance_metric="manhattan",
    )

    # Should NOT warn (10K < 100K threshold)
    positions = substrate.get_all_positions()
    assert len(positions) == 10_000


def test_gridnd_get_all_positions_raises_on_huge_grid():
    """Test get_all_positions() raises on absurdly large grids."""
    substrate = GridNDSubstrate(
        dimension_sizes=[100] * 4,  # 100^4 = 100M positions
        boundary="clamp",
        distance_metric="manhattan",
    )

    with pytest.raises(MemoryError, match="too large for memory"):
        substrate.get_all_positions()
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_edge_cases.py -v`

**Expected**: PASS

### Step 2.1.10: Add property-based tests (10 min)

**Create**: `tests/test_townlet/phase5/test_nd_properties.py`

```python
"""Property-based tests for N-dimensional substrates.

These tests verify universal invariants that must hold for ALL N.
"""
import pytest
import torch
from townlet.substrate.gridnd import GridNDSubstrate


@pytest.mark.parametrize("dimensions,size,metric", [
    (4, 3, "manhattan"),
    (4, 3, "euclidean"),
    (4, 3, "chebyshev"),
    (7, 3, "manhattan"),
    (10, 2, "euclidean"),
])
def test_distance_symmetric_gridnd(dimensions, size, metric):
    """PROPERTY: Distance is symmetric d(a,b) == d(b,a) for all N."""
    substrate = GridNDSubstrate(
        dimension_sizes=[size] * dimensions,
        boundary="clamp",
        distance_metric=metric,
    )

    # Two random positions
    pos_a = torch.randint(0, size, (1, dimensions), dtype=torch.long)
    pos_b = torch.randint(0, size, (1, dimensions), dtype=torch.long)

    # PROPERTY: d(a,b) == d(b,a)
    dist_ab = substrate.compute_distance(pos_a, pos_b)
    dist_ba = substrate.compute_distance(pos_b, pos_a)

    assert torch.allclose(dist_ab, dist_ba, atol=1e-6), \
        f"Distance not symmetric in {dimensions}D {metric}"


@pytest.mark.parametrize("dimensions,size,boundary_mode", [
    (4, 5, "clamp"),
    (7, 3, "clamp"),
    (4, 5, "wrap"),
    (7, 3, "sticky"),
])
def test_boundary_idempotence_gridnd(dimensions, size, boundary_mode):
    """PROPERTY: boundary(boundary(x)) == boundary(x)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[size] * dimensions,
        boundary=boundary_mode,
        distance_metric="manhattan",
    )

    # Out-of-bounds position
    positions = torch.full((1, dimensions), size * 2, dtype=torch.long)
    deltas = torch.zeros((1, dimensions), dtype=torch.float32)

    # Apply boundary once
    bounded_once = substrate.apply_movement(positions, deltas)

    # Apply boundary again
    bounded_twice = substrate.apply_movement(bounded_once, deltas)

    # PROPERTY: boundary(boundary(x)) == boundary(x)
    assert torch.equal(bounded_once, bounded_twice), \
        f"Boundary not idempotent in {dimensions}D {boundary_mode}"


@pytest.mark.parametrize("dimensions,size", [
    (4, 5),
    (7, 3),
    (10, 3),
])
def test_movement_reversible_gridnd(dimensions, size):
    """PROPERTY: move + reverse = identity (for wrap boundary)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[size] * dimensions,
        boundary="wrap",
        distance_metric="manhattan",
    )

    # Interior position
    center = size // 2
    positions = torch.full((1, dimensions), center, dtype=torch.long)

    # Random movement delta
    deltas = torch.randint(-1, 2, (1, dimensions), dtype=torch.float32)

    # Move forward
    moved = substrate.apply_movement(positions, deltas)

    # Move backward
    reversed_pos = substrate.apply_movement(moved, -deltas)

    # PROPERTY: move + reverse = identity
    assert torch.equal(reversed_pos, positions), \
        f"Movement not reversible in {dimensions}D"
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_nd_properties.py -v`

**Expected**: PASS

### Step 2.1.11: Add scaling tests (7D, 10D) (10 min)

**Create**: `tests/test_townlet/phase5/test_gridnd_scaling.py`

```python
"""Test GridND scaling behavior for 7D, 10D substrates."""
import pytest
import torch
from townlet.substrate.gridnd import GridNDSubstrate


def test_gridnd_7d_initialization():
    """Test 7D grid substrate."""
    substrate = GridNDSubstrate(
        dimension_sizes=[3] * 7,
        boundary="clamp",
        distance_metric="manhattan",
    )

    assert substrate.position_dim == 7
    assert substrate.action_space_size == 15  # 2*7 + 1

    # Test position initialization
    positions = substrate.initialize_positions(100, torch.device("cpu"))
    assert positions.shape == (100, 7)
    assert (positions >= 0).all()
    assert (positions < 3).all()


def test_gridnd_10d_initialization_with_warning():
    """Test 10D grid substrate (warning threshold)."""
    with pytest.warns(UserWarning, match="10 dimensions"):
        substrate = GridNDSubstrate(
            dimension_sizes=[2] * 10,  # Small sizes to keep fast
            boundary="clamp",
            distance_metric="manhattan",
        )

    assert substrate.position_dim == 10
    assert substrate.action_space_size == 21  # 2*10 + 1

    # Test position initialization
    positions = substrate.initialize_positions(100, torch.device("cpu"))
    assert positions.shape == (100, 10)


def test_gridnd_7d_movement_all_boundaries():
    """Test 7D movement with all boundary modes."""
    for boundary_mode in ["clamp", "wrap", "bounce", "sticky"]:
        substrate = GridNDSubstrate(
            dimension_sizes=[5] * 7,
            boundary=boundary_mode,
            distance_metric="manhattan",
        )

        # Agent at center
        positions = torch.full((1, 7), 2, dtype=torch.long)

        # Move in all directions
        deltas = torch.ones((1, 7), dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Verify movement applied
        assert new_positions.shape == (1, 7)
        assert (new_positions >= 0).all()
        assert (new_positions < 5).all()


def test_gridnd_10d_observation_encoding():
    """Test 10D observation encoding."""
    substrate = GridNDSubstrate(
        dimension_sizes=[3] * 10,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    positions = torch.zeros((10, 10), dtype=torch.long)

    encoded = substrate.encode_observation(positions, {})

    # relative: 10 dimensions
    assert encoded.shape == (10, 10)
    assert substrate.get_observation_dim() == 10
```

**Run**: `uv run pytest tests/test_townlet/phase5/test_gridnd_scaling.py -v`

**Expected**: PASS

### Step 2.1.12: Commit GridND implementation (5 min)

**Run full GridND test suite**:

```bash
uv run pytest tests/test_townlet/phase5/test_gridnd*.py tests/test_townlet/phase5/test_nd_properties.py -v
```

**Expected**: All tests pass

**Commit**:

```bash
git add src/townlet/substrate/gridnd.py tests/test_townlet/phase5/test_gridnd*.py tests/test_townlet/phase5/test_nd_properties.py
git commit -m "feat(substrate): implement GridND for N-dimensional grids (N≥4)

- Add GridNDSubstrate class with 4D-100D support
- Implement all boundary modes (clamp/wrap/bounce/sticky)
- Implement all distance metrics (manhattan/euclidean/chebyshev)
- Implement all observation encodings (relative/scaled/absolute)
- 2N cardinal neighbors for N-dimensional movement
- Warning at N≥10 (action space size)
- Memory protection for get_all_positions()
- Property-based tests (distance symmetry, boundary idempotence, etc.)
- Scaling tests for 7D, 10D substrates
- Part of Phase 5C Part 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

**(Continue with Tasks 2.2-2.6 in similar detail...)**

**Note**: Due to length constraints, I've shown the complete detailed TDD approach for Task 2.1. Tasks 2.2-2.6 would follow the same pattern:

- **Task 2.2**: ContinuousND (90 min) - Similar TDD steps, but simpler since ContinuousSubstrate already generalizes
- **Task 2.3**: Config Schema (45 min) - Add GridNDConfig, extend ContinuousConfig
- **Task 2.4**: Factory Integration (30 min) - Add gridnd branch, extend continuous branch
- **Task 2.5**: Integration Testing (90 min) - Full training with GridND/ContinuousND
- **Task 2.6**: Documentation (60 min) - Update CLAUDE.md, create templates

Each task would have 8-12 detailed TDD steps similar to the GridND implementation above.

---

## Execution Strategy

Same as Part 1: Use `superpowers:executing-plans` skill to execute tasks sequentially with commits between tasks.
