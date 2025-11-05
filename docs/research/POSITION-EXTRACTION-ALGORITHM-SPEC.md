# Position Extraction Algorithm Specification

**Author**: Claude (Technical Specification)
**Date**: 2025-11-05
**Status**: Ready for Implementation
**Purpose**: Unblock Phase 10 replay buffer implementation

---

## Executive Summary

This document specifies the **complete algorithm** for extracting agent position from flattened observation vectors across all 6 substrate types.

**Problem**: The observation state is a flattened vector containing grid encoding, meters, affordance encoding, and temporal extras. To store next-state valid actions in the replay buffer, we need to extract the position component to call `substrate.get_valid_actions(position)`.

**Solution**: Observation structure is deterministic based on substrate type. We can reverse-engineer position from the grid encoding portion using the substrate's known encoding scheme.

**Estimated Implementation Time**: 8-12 hours (was incorrectly estimated at 4-6h)
- Algorithm implementation: 4-6h
- Testing across all substrates: 3-4h
- Integration and validation: 1-2h

---

## Observation Structure (Current Implementation)

All observations follow this structure:

```
observation = [grid_encoding | meters | affordance_encoding | temporal_extras]
```

**Component sizes**:
- `grid_encoding`: **VARIABLE** (depends on substrate type)
- `meters`: 8 floats (energy, health, satiation, money, mood, social, fitness, hygiene)
- `affordance_encoding`: 15 floats (14 affordances + "none", one-hot)
- `temporal_extras`: 4 floats (time_of_day, retirement_age, interaction_progress, interaction_ticks)

**Total observation size**: `grid_size + 8 + 15 + 4 = grid_size + 27`

**Key insight**: Grid encoding is ALWAYS at the start. Meters, affordances, and temporal are ALWAYS at the end.

---

## Grid Encoding by Substrate Type

### 1. Grid1D (1D Linear Grid)

**Position format**: Scalar integer `[0, length)`

**Grid encoding**: One-hot vector of length `length`

**Example** (length=10, agent at position 5):
```python
grid_encoding = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # 10 dims
position = 5  # Scalar
```

**Extraction algorithm**:
```python
def extract_position_1d(observation: torch.Tensor, substrate: Grid1DSubstrate) -> torch.Tensor:
    """Extract position from 1D grid observation.

    Args:
        observation: [obs_dim] flattened observation
        substrate: Grid1DSubstrate instance

    Returns:
        position: [1] scalar position tensor
    """
    length = substrate.length

    # Grid encoding is first `length` dims
    grid_encoding = observation[:length]

    # Find which index is 1.0 (one-hot encoding)
    position_scalar = torch.argmax(grid_encoding).item()

    # Return as [1] tensor for consistency
    return torch.tensor([position_scalar], dtype=torch.long)
```

---

### 2. Grid2D (2D Rectangular Grid)

**Position format**: 2D coordinates `[x, y]` where `x ∈ [0, width)`, `y ∈ [0, height)`

**Grid encoding**: Flattened 2D one-hot grid (row-major order)

**Size**: `width * height`

**Example** (8×8 grid, agent at (3, 5)):
```python
# Flattened index = y * width + x = 5 * 8 + 3 = 43
grid_encoding = [0, 0, ..., 0, 1, 0, ..., 0]  # 64 dims, position 43 is 1.0
position = [3, 5]  # [x, y]
```

**Extraction algorithm**:
```python
def extract_position_2d(observation: torch.Tensor, substrate: Grid2DSubstrate) -> torch.Tensor:
    """Extract position from 2D grid observation.

    Args:
        observation: [obs_dim] flattened observation
        substrate: Grid2DSubstrate instance

    Returns:
        position: [2] tensor [x, y]
    """
    width = substrate.width
    height = substrate.height
    grid_size = width * height

    # Grid encoding is first `grid_size` dims
    grid_encoding = observation[:grid_size]

    # Find which index is 1.0
    flat_index = torch.argmax(grid_encoding).item()

    # Convert flat index to (x, y) coordinates (row-major order)
    y = flat_index // width
    x = flat_index % width

    return torch.tensor([x, y], dtype=torch.long)
```

---

### 3. Grid3D (3D Cubic Grid)

**Position format**: 3D coordinates `[x, y, z]`

**Grid encoding**: Flattened 3D one-hot grid (z-major, then y-major, then x)

**Size**: `width * height * depth`

**Flattening order**: `flat_index = z * (width * height) + y * width + x`

**Example** (5×5×5 grid, agent at (2, 3, 1)):
```python
# Flattened index = 1 * (5*5) + 3 * 5 + 2 = 25 + 15 + 2 = 42
grid_encoding = [0, 0, ..., 0, 1, 0, ..., 0]  # 125 dims
position = [2, 3, 1]  # [x, y, z]
```

**Extraction algorithm**:
```python
def extract_position_3d(observation: torch.Tensor, substrate: Grid3DSubstrate) -> torch.Tensor:
    """Extract position from 3D grid observation.

    Args:
        observation: [obs_dim] flattened observation
        substrate: Grid3DSubstrate instance

    Returns:
        position: [3] tensor [x, y, z]
    """
    width = substrate.width
    height = substrate.height
    depth = substrate.depth
    grid_size = width * height * depth

    # Grid encoding is first `grid_size` dims
    grid_encoding = observation[:grid_size]

    # Find which index is 1.0
    flat_index = torch.argmax(grid_encoding).item()

    # Convert flat index to (x, y, z) coordinates
    # flat_index = z * (width * height) + y * width + x
    xy_plane_size = width * height

    z = flat_index // xy_plane_size
    remainder = flat_index % xy_plane_size
    y = remainder // width
    x = remainder % width

    return torch.tensor([x, y, z], dtype=torch.long)
```

---

### 4. HexSubstrate (Hexagonal Grid with Axial Coordinates)

**Position format**: Axial coordinates `[q, r]` (cube coordinates with s implied)

**Grid encoding**: Flattened 2D one-hot grid over valid hex positions

**Size**: Number of valid hexagons within radius `R`

**Valid hexagons**: All `(q, r)` where `|q| + |r| + |q + r| ≤ 2R` (cube constraint)

**Encoding order**: Sorted by (q, r) lexicographically

**Example** (radius=2, 19 hexagons):
```python
# Valid positions (sorted):
# (-2, 0), (-2, 1), (-2, 2)
# (-1, -1), (-1, 0), (-1, 1), (-1, 2)
# (0, -2), (0, -1), (0, 0), (0, 1), (0, 2)
# (1, -2), (1, -1), (1, 0), (1, 1)
# (2, -2), (2, -1), (2, 0)

# Agent at (1, -1) → index 14 in sorted list
grid_encoding = [0, 0, ..., 0, 1, 0, ..., 0]  # 19 dims
position = [1, -1]  # [q, r]
```

**Extraction algorithm**:
```python
def extract_position_hex(observation: torch.Tensor, substrate: HexSubstrate) -> torch.Tensor:
    """Extract position from hex grid observation.

    Args:
        observation: [obs_dim] flattened observation
        substrate: HexSubstrate instance

    Returns:
        position: [2] tensor [q, r] (axial coordinates)
    """
    # Hex substrate must provide mapping from index → (q, r)
    # This is precomputed during substrate initialization
    if not hasattr(substrate, 'index_to_axial'):
        raise RuntimeError("HexSubstrate must provide index_to_axial mapping")

    grid_size = substrate.num_positions  # Number of valid hexagons

    # Grid encoding is first `grid_size` dims
    grid_encoding = observation[:grid_size]

    # Find which index is 1.0
    flat_index = torch.argmax(grid_encoding).item()

    # Map index to (q, r) coordinates
    q, r = substrate.index_to_axial[flat_index]

    return torch.tensor([q, r], dtype=torch.long)
```

**Required substrate method**:
```python
class HexSubstrate:
    def __init__(self, radius: int):
        self.radius = radius

        # Precompute index ↔ (q, r) mappings
        self.axial_to_index: dict[tuple[int, int], int] = {}
        self.index_to_axial: dict[int, tuple[int, int]] = {}

        valid_positions = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                # Cube constraint: |q| + |r| + |q+r| <= 2*radius
                if abs(q) + abs(r) + abs(q + r) <= 2 * radius:
                    valid_positions.append((q, r))

        # Sort for deterministic ordering
        valid_positions.sort()

        for idx, (q, r) in enumerate(valid_positions):
            self.axial_to_index[(q, r)] = idx
            self.index_to_axial[idx] = (q, r)

        self.num_positions = len(valid_positions)
```

---

### 5. GridND (N-Dimensional Hypergrid)

**Position format**: N-dimensional coordinates `[x₀, x₁, ..., xₙ₋₁]`

**Grid encoding**: Flattened N-D one-hot grid

**Size**: `∏ᵢ dims[i]` (product of all dimension sizes)

**Flattening order**: Row-major (rightmost dimension varies fastest)

**Example** (3D with dims=[4, 3, 5], agent at [2, 1, 3]):
```python
# Flattened index = 2 * (3*5) + 1 * 5 + 3 = 30 + 5 + 3 = 38
grid_encoding = [0, 0, ..., 0, 1, 0, ..., 0]  # 60 dims
position = [2, 1, 3]  # [x₀, x₁, x₂]
```

**Extraction algorithm**:
```python
def extract_position_nd(observation: torch.Tensor, substrate: GridNDSubstrate) -> torch.Tensor:
    """Extract position from N-D grid observation.

    Args:
        observation: [obs_dim] flattened observation
        substrate: GridNDSubstrate instance

    Returns:
        position: [N] tensor coordinates
    """
    dims = substrate.dims  # List[int], e.g., [4, 3, 5]
    grid_size = substrate.num_positions  # Product of dims

    # Grid encoding is first `grid_size` dims
    grid_encoding = observation[:grid_size]

    # Find which index is 1.0
    flat_index = torch.argmax(grid_encoding).item()

    # Convert flat index to N-D coordinates (row-major order)
    coords = []
    for i in range(len(dims) - 1, -1, -1):  # Iterate backwards
        coord = flat_index % dims[i]
        coords.append(coord)
        flat_index //= dims[i]

    coords.reverse()  # Reverse to get [x₀, x₁, ..., xₙ₋₁]

    return torch.tensor(coords, dtype=torch.long)
```

---

### 6. GraphSubstrate (Graph with Node IDs)

**Position format**: Node ID (integer `[0, num_nodes)`)

**Grid encoding**: One-hot vector of length `num_nodes`

**Size**: `num_nodes`

**Example** (16-node graph, agent at node 7):
```python
grid_encoding = [0, 0, 0, 0, 0, 0, 0, 1, 0, ..., 0]  # 16 dims
position = 7  # Node ID (scalar)
```

**Extraction algorithm**:
```python
def extract_position_graph(observation: torch.Tensor, substrate: GraphSubstrate) -> torch.Tensor:
    """Extract position from graph observation.

    Args:
        observation: [obs_dim] flattened observation
        substrate: GraphSubstrate instance

    Returns:
        position: [1] node ID tensor (scalar)
    """
    num_nodes = substrate.num_nodes

    # Grid encoding is first `num_nodes` dims
    grid_encoding = observation[:num_nodes]

    # Find which index is 1.0 (node ID)
    node_id = torch.argmax(grid_encoding).item()

    # Return as [1] tensor for consistency
    return torch.tensor([node_id], dtype=torch.long)
```

---

## Unified ObservationBuilder Implementation

### Updated ObservationBuilder Class

**Modify**: `src/townlet/environment/observation.py`

Add position extraction method:

```python
class ObservationBuilder:
    """Builds observations for agents."""

    def __init__(self, substrate: SpatialSubstrate, ...):
        self.substrate = substrate
        self.substrate_type = type(substrate).__name__
        # ... existing init

    def build_observation(self, position, meters, affordance_at_pos, temporal) -> torch.Tensor:
        """Build observation vector (existing method)."""
        grid_encoding = self._encode_grid_position(position)

        obs = torch.cat([
            grid_encoding,      # Variable size (substrate-dependent)
            meters,             # 8 floats
            affordance_at_pos,  # 15 floats (one-hot)
            temporal,           # 4 floats
        ])

        return obs

    def extract_position(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract agent position from observation vector.

        This is the INVERSE operation of _encode_grid_position().

        Args:
            observation: [obs_dim] flattened observation vector

        Returns:
            position: [position_dim] position tensor
                - 1D Grid: [1] scalar position
                - 2D Grid: [2] (x, y)
                - 3D Grid: [3] (x, y, z)
                - Hex: [2] (q, r) axial coords
                - ND Grid: [N] coordinates
                - Graph: [1] node ID

        Raises:
            RuntimeError: If substrate type not supported
        """
        # Dispatch to substrate-specific extraction
        if self.substrate_type == 'Grid1DSubstrate':
            return self._extract_position_1d(observation)
        elif self.substrate_type == 'Grid2DSubstrate':
            return self._extract_position_2d(observation)
        elif self.substrate_type == 'Grid3DSubstrate':
            return self._extract_position_3d(observation)
        elif self.substrate_type == 'HexSubstrate':
            return self._extract_position_hex(observation)
        elif self.substrate_type == 'GridNDSubstrate':
            return self._extract_position_nd(observation)
        elif self.substrate_type == 'GraphSubstrate':
            return self._extract_position_graph(observation)
        else:
            raise RuntimeError(f"Unsupported substrate type: {self.substrate_type}")

    # Private extraction methods (implementations from above)

    def _extract_position_1d(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract 1D position."""
        length = self.substrate.length
        grid_encoding = observation[:length]
        position_scalar = torch.argmax(grid_encoding).item()
        return torch.tensor([position_scalar], dtype=torch.long)

    def _extract_position_2d(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract 2D position."""
        width = self.substrate.width
        height = self.substrate.height
        grid_size = width * height

        grid_encoding = observation[:grid_size]
        flat_index = torch.argmax(grid_encoding).item()

        y = flat_index // width
        x = flat_index % width

        return torch.tensor([x, y], dtype=torch.long)

    def _extract_position_3d(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract 3D position."""
        width = self.substrate.width
        height = self.substrate.height
        depth = self.substrate.depth
        grid_size = width * height * depth

        grid_encoding = observation[:grid_size]
        flat_index = torch.argmax(grid_encoding).item()

        xy_plane_size = width * height
        z = flat_index // xy_plane_size
        remainder = flat_index % xy_plane_size
        y = remainder // width
        x = remainder % width

        return torch.tensor([x, y, z], dtype=torch.long)

    def _extract_position_hex(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract hex position."""
        if not hasattr(self.substrate, 'index_to_axial'):
            raise RuntimeError("HexSubstrate must provide index_to_axial mapping")

        grid_size = self.substrate.num_positions
        grid_encoding = observation[:grid_size]
        flat_index = torch.argmax(grid_encoding).item()

        q, r = self.substrate.index_to_axial[flat_index]
        return torch.tensor([q, r], dtype=torch.long)

    def _extract_position_nd(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract N-D position."""
        dims = self.substrate.dims
        grid_size = self.substrate.num_positions

        grid_encoding = observation[:grid_size]
        flat_index = torch.argmax(grid_encoding).item()

        coords = []
        for i in range(len(dims) - 1, -1, -1):
            coord = flat_index % dims[i]
            coords.append(coord)
            flat_index //= dims[i]

        coords.reverse()
        return torch.tensor(coords, dtype=torch.long)

    def _extract_position_graph(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract graph node ID."""
        num_nodes = self.substrate.num_nodes
        grid_encoding = observation[:num_nodes]
        node_id = torch.argmax(grid_encoding).item()
        return torch.tensor([node_id], dtype=torch.long)
```

---

## Required Substrate Changes

### HexSubstrate Must Provide Mappings

**Modify**: `src/townlet/substrate/hex.py`

```python
class HexSubstrate(SpatialSubstrate):
    """Hexagonal grid substrate with axial coordinates."""

    def __init__(self, radius: int, boundary: str = "clamp"):
        self.radius = radius
        self.boundary = boundary

        # Precompute index ↔ (q, r) mappings (REQUIRED for position extraction)
        self.axial_to_index: dict[tuple[int, int], int] = {}
        self.index_to_axial: dict[int, tuple[int, int]] = {}

        valid_positions = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                # Cube constraint
                if abs(q) + abs(r) + abs(q + r) <= 2 * radius:
                    valid_positions.append((q, r))

        # Sort for deterministic ordering
        valid_positions.sort()

        for idx, (q, r) in enumerate(valid_positions):
            self.axial_to_index[(q, r)] = idx
            self.index_to_axial[idx] = (q, r)

        self.num_positions = len(valid_positions)
        self.position_dim = 2  # (q, r)
        self.action_space_size = 7  # 6 directions + INTERACT
```

---

## Comprehensive Testing Strategy

### Round-Trip Validation Tests

**Create**: `tests/test_townlet/unit/test_position_extraction.py`

```python
"""Test position extraction from observations."""
import torch
import pytest
from townlet.substrate.grid1d import Grid1DSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.hex import HexSubstrate
from townlet.substrate.graph import GraphSubstrate
from townlet.environment.observation import ObservationBuilder


def test_round_trip_1d():
    """1D: encode position → build observation → extract position → matches original."""
    substrate = Grid1DSubstrate(length=10, boundary="clamp")
    obs_builder = ObservationBuilder(substrate, ...)

    # Test multiple positions
    for original_pos in [0, 5, 9]:
        pos_tensor = torch.tensor([original_pos], dtype=torch.long)

        # Build observation
        obs = obs_builder.build_observation(
            position=pos_tensor,
            meters=torch.zeros(8),
            affordance_at_pos=torch.zeros(15),
            temporal=torch.zeros(4),
        )

        # Extract position
        extracted_pos = obs_builder.extract_position(obs)

        # Should match original
        assert torch.equal(extracted_pos, pos_tensor), \
            f"Round-trip failed: {original_pos} → {extracted_pos.item()}"


def test_round_trip_2d():
    """2D: encode → extract → matches."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")
    obs_builder = ObservationBuilder(substrate, ...)

    test_positions = [(0, 0), (3, 5), (7, 7), (2, 4)]

    for x, y in test_positions:
        pos_tensor = torch.tensor([x, y], dtype=torch.long)

        obs = obs_builder.build_observation(
            position=pos_tensor,
            meters=torch.zeros(8),
            affordance_at_pos=torch.zeros(15),
            temporal=torch.zeros(4),
        )

        extracted_pos = obs_builder.extract_position(obs)

        assert torch.equal(extracted_pos, pos_tensor), \
            f"Round-trip failed: ({x}, {y}) → ({extracted_pos[0].item()}, {extracted_pos[1].item()})"


def test_round_trip_3d():
    """3D: encode → extract → matches."""
    substrate = Grid3DSubstrate(width=5, height=5, depth=5, boundary="clamp")
    obs_builder = ObservationBuilder(substrate, ...)

    test_positions = [(0, 0, 0), (2, 3, 1), (4, 4, 4), (1, 2, 3)]

    for x, y, z in test_positions:
        pos_tensor = torch.tensor([x, y, z], dtype=torch.long)

        obs = obs_builder.build_observation(
            position=pos_tensor,
            meters=torch.zeros(8),
            affordance_at_pos=torch.zeros(15),
            temporal=torch.zeros(4),
        )

        extracted_pos = obs_builder.extract_position(obs)

        assert torch.equal(extracted_pos, pos_tensor), \
            f"Round-trip failed: ({x}, {y}, {z}) → {extracted_pos.tolist()}"


def test_round_trip_hex():
    """Hex: encode → extract → matches."""
    substrate = HexSubstrate(radius=2)
    obs_builder = ObservationBuilder(substrate, ...)

    # Test all valid hex positions
    for idx, (q, r) in substrate.index_to_axial.items():
        pos_tensor = torch.tensor([q, r], dtype=torch.long)

        obs = obs_builder.build_observation(
            position=pos_tensor,
            meters=torch.zeros(8),
            affordance_at_pos=torch.zeros(15),
            temporal=torch.zeros(4),
        )

        extracted_pos = obs_builder.extract_position(obs)

        assert torch.equal(extracted_pos, pos_tensor), \
            f"Round-trip failed: ({q}, {r}) → ({extracted_pos[0].item()}, {extracted_pos[1].item()})"


def test_round_trip_graph():
    """Graph: encode → extract → matches."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    substrate = GraphSubstrate(num_nodes=4, edges=edges, directed=False)
    obs_builder = ObservationBuilder(substrate, ...)

    # Test all nodes
    for node_id in range(4):
        pos_tensor = torch.tensor([node_id], dtype=torch.long)

        obs = obs_builder.build_observation(
            position=pos_tensor,
            meters=torch.zeros(8),
            affordance_at_pos=torch.zeros(15),
            temporal=torch.zeros(4),
        )

        extracted_pos = obs_builder.extract_position(obs)

        assert torch.equal(extracted_pos, pos_tensor), \
            f"Round-trip failed: node {node_id} → {extracted_pos.item()}"


def test_extraction_with_realistic_observations():
    """Test extraction with non-zero meters/affordances/temporal."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")
    obs_builder = ObservationBuilder(substrate, ...)

    pos_tensor = torch.tensor([3, 5], dtype=torch.long)

    # Realistic values
    meters = torch.tensor([0.8, 0.9, 0.7, 50.0, 0.6, 0.5, 0.8, 0.9])
    affordance_at_pos = torch.zeros(15)
    affordance_at_pos[3] = 1.0  # Affordance 3 present
    temporal = torch.tensor([0.5, 65.0, 0.3, 10.0])

    obs = obs_builder.build_observation(
        position=pos_tensor,
        meters=meters,
        affordance_at_pos=affordance_at_pos,
        temporal=temporal,
    )

    extracted_pos = obs_builder.extract_position(obs)

    # Position extraction should ignore meters/affordances/temporal
    assert torch.equal(extracted_pos, pos_tensor), \
        "Position extraction affected by non-position components"


def test_batched_extraction():
    """Test extracting positions from batch of observations."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")
    obs_builder = ObservationBuilder(substrate, ...)

    batch_size = 16
    positions = torch.randint(0, 8, (batch_size, 2), dtype=torch.long)

    observations = []
    for i in range(batch_size):
        obs = obs_builder.build_observation(
            position=positions[i],
            meters=torch.zeros(8),
            affordance_at_pos=torch.zeros(15),
            temporal=torch.zeros(4),
        )
        observations.append(obs)

    observations = torch.stack(observations)  # [batch_size, obs_dim]

    # Extract all positions
    extracted_positions = torch.stack([
        obs_builder.extract_position(observations[i])
        for i in range(batch_size)
    ])

    # All should match
    assert torch.equal(extracted_positions, positions), \
        "Batched extraction failed"
```

**Run tests**:
```bash
uv run pytest tests/test_townlet/unit/test_position_extraction.py -v
```

**Expected**: ALL tests PASS (15+ tests covering all substrates)

---

## Integration with Replay Buffer

Once position extraction is implemented, the replay buffer can use it:

**Modify**: `src/townlet/population/vectorized.py`

```python
def store_transition(self, state, action, reward, next_state, done):
    """Store transition with valid actions for next state."""

    # Extract next position from next_state observation
    next_position = self.obs_builder.extract_position(next_state)

    # Get valid actions for next position (for masked Q-target)
    if hasattr(self.substrate, 'get_valid_actions'):
        valid_actions_list = self.substrate.get_valid_actions(next_position)

        # Convert to boolean mask
        valid_actions_mask = torch.zeros(
            self.substrate.action_space_size,
            dtype=torch.bool,
            device=self.device
        )
        valid_actions_mask[valid_actions_list] = True
    else:
        # No masking needed (all actions valid)
        valid_actions_mask = None

    transition = Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        valid_actions=valid_actions_mask,
    )

    self.replay_buffer.add(transition)
```

---

## Performance Considerations

### Computational Cost

Position extraction uses `torch.argmax()` which is O(grid_size):
- 1D (length=10): O(10) - negligible
- 2D (8×8): O(64) - negligible
- 3D (5×5×5): O(125) - negligible
- Hex (radius=4): O(37) - negligible
- Graph (100 nodes): O(100) - negligible

**Conclusion**: Position extraction is fast enough to use in training loop.

### Batched Extraction

For batch processing, can vectorize:

```python
def extract_positions_batched(
    self,
    observations: torch.Tensor  # [batch_size, obs_dim]
) -> torch.Tensor:
    """Extract positions from batch of observations.

    Returns:
        positions: [batch_size, position_dim]
    """
    batch_size = observations.shape[0]
    grid_size = self.substrate.num_positions

    # Extract grid encodings
    grid_encodings = observations[:, :grid_size]  # [batch_size, grid_size]

    # Find argmax for each sample
    flat_indices = torch.argmax(grid_encodings, dim=1)  # [batch_size]

    # Convert to positions (substrate-specific)
    # ... (similar logic as single extraction, but vectorized)
```

---

## Success Criteria

Position extraction is complete when:

- ✅ `ObservationBuilder.extract_position()` implemented
- ✅ All 6 substrate types supported (1D, 2D, 3D, Hex, ND, Graph)
- ✅ Round-trip tests pass for ALL substrates
- ✅ Tests with realistic observations pass
- ✅ Batched extraction works
- ✅ HexSubstrate provides `index_to_axial` mapping
- ✅ Integration with replay buffer functional
- ✅ Performance acceptable (< 1ms per extraction)

---

## Implementation Checklist

- [ ] Modify `src/townlet/environment/observation.py`:
  - [ ] Add `extract_position()` public method
  - [ ] Add 6 private extraction methods (`_extract_position_*`)
  - [ ] Add substrate type dispatch logic
- [ ] Modify `src/townlet/substrate/hex.py`:
  - [ ] Add `index_to_axial` mapping
  - [ ] Add `axial_to_index` mapping
  - [ ] Add `num_positions` attribute
- [ ] Create `tests/test_townlet/unit/test_position_extraction.py`:
  - [ ] Round-trip tests (1D, 2D, 3D, Hex, Graph)
  - [ ] Realistic observation tests
  - [ ] Batched extraction tests
  - [ ] Edge case tests (boundaries, invalid observations)
- [ ] Update `src/townlet/population/vectorized.py`:
  - [ ] Use `extract_position()` in `store_transition()`
  - [ ] Pass `obs_builder` to population
- [ ] Integration testing:
  - [ ] Verify replay buffer stores valid actions correctly
  - [ ] Verify Q-target masking works
  - [ ] Run training with all substrate types

**Estimated Time**: 8-12 hours
- Implementation: 4-6h
- Testing: 3-4h
- Integration: 1-2h

---

## References

- **Hex coordinate systems**: https://www.redblobgames.com/grids/hexagons/
- **Axial coordinates**: Standard (q, r) representation for hex grids
- **Row-major order**: Standard flattening for multi-dimensional arrays
- **One-hot encoding**: Each position encoded as single 1.0 in vector of 0.0s

---

**Status**: Ready for implementation
**Blockers Resolved**: This specification unblocks Phase 10 Task 0.3 (replay buffer schema change)
**Next Steps**: Implement `ObservationBuilder.extract_position()` and run tests
