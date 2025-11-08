# TASK-002A: Configurable Spatial Substrates

**STATUS**: ✅ **COMPLETED** (2025-11-06)

All 8 phases complete. Substrate abstraction fully implemented and tested.

## Problem: Hardcoded 2D Square Grid Violates UAC Principle

### Current State

The spatial substrate (coordinate system, topology, boundaries) is **hardcoded in Python**, assuming a 2D square grid with Manhattan distance and clamped boundaries.

**Current Implementation** (`src/townlet/environment/vectorized_env.py`):

```python
# Hardcoded 2D topology
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)

# Hardcoded square grid
self.grid_size = grid_size  # Single dimension for both axes

# Hardcoded Manhattan distance
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)

# Hardcoded clamped boundaries
new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
```

### Why This Violates UAC

1. **No Topology Flexibility**: Cannot model universes with alternative spatial structures:
   - 3D environments (multi-story buildings, Minecraft-like)
   - Hexagonal grids (better distance metrics, strategy games)
   - Continuous spaces (smooth movement, robotics)
   - Graph-based spaces (subway systems, social networks)
   - Aspatial universes (pure resource management, no positioning)

2. **Hardcoded Assumptions**:
   - Space must be 2D
   - Grid must be square (not hex, triangular, or other tessellation)
   - Boundaries must be clamped (not wraparound/toroidal, bounce, or fail)
   - Distance must be Manhattan (not Euclidean, Chebyshev, or graph distance)

3. **Cannot Experiment**: Operator cannot:
   - Add vertical dimension (Z-axis for floors)
   - Create toroidal worlds (Pac-Man wraparound)
   - Use hexagonal grids (more natural terrain)
   - Remove spatial substrate entirely (pure state machine)

4. **Spatial Substrate Split**: Some in training.yaml, most hardcoded in Python:

   ```yaml
   # training.yaml (partial)
   grid_size: 8
   ```

   But topology, boundaries, distance metric are Python code!

## Solution: Config-Driven Spatial Substrates

### Proposed Architecture

Define the complete spatial substrate in YAML configuration, making it a first-class citizen of UNIVERSE_AS_CODE.

**Create `substrate.yaml` in config packs:**

```yaml
version: "1.0"
description: "Spatial substrate configuration for HAMLET village"

# Substrate type selection
substrate:
  type: "grid"  # grid, continuous, graph, aspatial

  # Grid-specific configuration (for discrete grid worlds)
  grid:
    topology: "square"  # square, hexagonal, triangular, cubic
    dimensions: [8, 8]  # [width, height] or [width, height, depth] for 3D
    boundary: "clamp"   # clamp, wrap, bounce, fail
    distance_metric: "manhattan"  # manhattan, euclidean, chebyshev

  # Continuous-specific configuration (for smooth movement)
  continuous:
    dimensions: 2  # 2D or 3D
    bounds: [[0.0, 10.0], [0.0, 10.0]]  # [min, max] per dimension
    boundary: "clamp"
    discretization: 0.1  # Step size for movement actions

  # Graph-specific configuration (for abstract topologies)
  graph:
    nodes: 16  # Number of nodes (stations, rooms, etc.)
    adjacency: "config"  # config, complete, random, spatial
    # If adjacency: "config", define edges explicitly
    edges:
      - [0, 1]  # Node 0 connects to Node 1
      - [1, 2]
      - [2, 3]

  # Aspatial configuration (no positioning, pure state machine)
  aspatial:
    enabled: true
    # No positioning - affordances accessed directly without movement
```

### Key Insight: Substrate Is Optional, Not Fundamental

The meters (energy, health, money, etc.) **ARE** the universe. The spatial substrate is just an **optional overlay** for:

- Positioning affordances on a map
- Enabling navigation mechanics
- Creating distance-based challenges

An **aspatial HAMLET** is perfectly valid - pure resource management without any concept of "position."

### Example Substrates

**Example 1: 3D Cubic Grid (Multi-Story Building)**

```yaml
# configs/L1_3D_house/substrate.yaml
version: "1.0"
description: "3-story house with stairs"

substrate:
  type: "grid"
  grid:
    topology: "cubic"
    dimensions: [8, 8, 3]  # 8×8 floor plan, 3 floors
    boundary: "clamp"
    distance_metric: "manhattan"
```

**Actions needed**: 6 directions (UP, DOWN, LEFT, RIGHT, UP_FLOOR, DOWN_FLOOR) or special affordances (Stairs, Elevator)

**Example 2: Toroidal 2D Grid (Pac-Man Wraparound)**

```yaml
# configs/L1_toroidal/substrate.yaml
version: "1.0"
description: "Toroidal world with wraparound boundaries"

substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [8, 8]
    boundary: "wrap"  # Edges connect! Top→Bottom, Left→Right
    distance_metric: "manhattan"
```

**Pedagogical value**: Teaches topology, periodic boundary conditions, no boundary bias

**Example 3: Hexagonal Grid (Strategy Game)**

```yaml
# configs/L1_hexagonal/substrate.yaml
version: "1.0"
description: "Hexagonal grid for natural terrain"

substrate:
  type: "grid"
  grid:
    topology: "hexagonal"
    dimensions: [8, 8]  # Axial coordinates (q, r)
    boundary: "clamp"
    distance_metric: "hexagonal"  # Uniform distance to all 6 neighbors
```

**Actions needed**: 6 directions (E, NE, NW, W, SW, SE)

**Example 4: Continuous 2D Space (Smooth Robot)**

```yaml
# configs/L1_continuous/substrate.yaml
version: "1.0"
description: "Continuous space for smooth movement"

substrate:
  type: "continuous"
  continuous:
    dimensions: 2
    bounds: [[0.0, 8.0], [0.0, 8.0]]
    boundary: "clamp"
    discretization: 0.1  # Agent moves in 0.1 increments
```

**Actions needed**: Continuous movement with (dx, dy) deltas

**Example 5: Subway System (Graph)**

```yaml
# configs/L1_subway/substrate.yaml
version: "1.0"
description: "Subway system with stations and routes"

substrate:
  type: "graph"
  graph:
    nodes: 10  # 10 subway stations
    adjacency: "config"
    edges:
      - [0, 1]  # Red line: Central → North
      - [1, 2]  # Red line: North → Airport
      - [0, 3]  # Blue line: Central → West
      - [3, 4]  # Blue line: West → Harbor
      # ... etc
```

**Actions needed**: "Move to adjacent station" (variable action space per node!)

**Example 6: Aspatial (Pure Resource Management)**

```yaml
# configs/L1_aspatial/substrate.yaml
version: "1.0"
description: "No spatial substrate - pure state machine"

substrate:
  type: "aspatial"
  aspatial:
    enabled: true
    # No positioning - all affordances "available" without movement
```

**Actions needed**: No movement actions, only INTERACT and WAIT

**Key insight**: Reveals that the grid is just window dressing. The meters are the true universe.

## Implementation Plan

### Phase 1: Create Substrate Abstraction Layer

**File**: `src/townlet/environment/substrate.py`

```python
from abc import ABC, abstractmethod
import torch

class SpatialSubstrate(ABC):
    """Abstract interface for spatial substrates."""

    @property
    @abstractmethod
    def position_dim(self) -> int:
        """Dimensionality of position vectors (2 for 2D, 3 for 3D, 0 for aspatial)"""
        pass

    @abstractmethod
    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        """Initialize random positions for agents. Shape: [num_agents, position_dim]"""
        pass

    @abstractmethod
    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply movement deltas to positions, respecting boundaries."""
        pass

    @abstractmethod
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute distance between positions."""
        pass

    @abstractmethod
    def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode positions into observation space."""
        pass

    @abstractmethod
    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get valid neighbor positions (for action validation)."""
        pass


class SquareGridSubstrate(SpatialSubstrate):
    """2D square grid substrate (current implementation)."""

    def __init__(self, width: int, height: int, boundary: str = "clamp", distance_metric: str = "manhattan"):
        self.width = width
        self.height = height
        self.boundary = boundary
        self.distance_metric = distance_metric

    @property
    def position_dim(self) -> int:
        return 2

    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        return torch.stack([
            torch.randint(0, self.width, (num_agents,)),
            torch.randint(0, self.height, (num_agents,))
        ], dim=1)

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        new_positions = positions + deltas

        if self.boundary == "clamp":
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)
        elif self.boundary == "wrap":
            # Toroidal wraparound
            new_positions[:, 0] = new_positions[:, 0] % self.width
            new_positions[:, 1] = new_positions[:, 1] % self.height
        elif self.boundary == "bounce":
            # Reflect at boundaries
            for i in range(new_positions.shape[0]):
                if new_positions[i, 0] < 0 or new_positions[i, 0] >= self.width:
                    new_positions[i, 0] = positions[i, 0]
                if new_positions[i, 1] < 0 or new_positions[i, 1] >= self.height:
                    new_positions[i, 1] = positions[i, 1]

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)
        elif self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))
        elif self.distance_metric == "chebyshev":
            return torch.abs(pos1 - pos2).max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
        # One-hot encoding of grid cells
        flat_indices = positions[:, 1] * self.width + positions[:, 0]
        one_hot = torch.zeros((positions.shape[0], self.width * self.height))
        one_hot.scatter_(1, flat_indices.unsqueeze(1), 1)
        return one_hot

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        x, y = position
        neighbors = [
            torch.tensor([x, y - 1]),  # UP
            torch.tensor([x, y + 1]),  # DOWN
            torch.tensor([x - 1, y]),  # LEFT
            torch.tensor([x + 1, y]),  # RIGHT
        ]

        if self.boundary == "clamp":
            neighbors = [n for n in neighbors
                        if 0 <= n[0] < self.width and 0 <= n[1] < self.height]
        # Wrap and bounce allow all neighbors (wrapping/bouncing happens in apply_movement)

        return neighbors


class CubicGridSubstrate(SpatialSubstrate):
    """3D cubic grid substrate for multi-level environments."""

    def __init__(self, width: int, height: int, depth: int, boundary: str = "clamp"):
        self.width = width
        self.height = height
        self.depth = depth
        self.boundary = boundary

    @property
    def position_dim(self) -> int:
        return 3  # (x, y, z)

    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        return torch.stack([
            torch.randint(0, self.width, (num_agents,)),
            torch.randint(0, self.height, (num_agents,)),
            torch.randint(0, self.depth, (num_agents,))
        ], dim=1)

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        new_positions = positions + deltas

        if self.boundary == "clamp":
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)
            new_positions[:, 2] = torch.clamp(new_positions[:, 2], 0, self.depth - 1)

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # 3D Manhattan distance
        return torch.abs(pos1 - pos2).sum(dim=-1)

    def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
        # Flatten 3D grid to 1D indices
        flat_indices = (positions[:, 2] * self.width * self.height +
                       positions[:, 1] * self.width +
                       positions[:, 0])
        one_hot = torch.zeros((positions.shape[0], self.width * self.height * self.depth))
        one_hot.scatter_(1, flat_indices.unsqueeze(1), 1)
        return one_hot

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        x, y, z = position
        neighbors = [
            torch.tensor([x, y - 1, z]),  # UP
            torch.tensor([x, y + 1, z]),  # DOWN
            torch.tensor([x - 1, y, z]),  # LEFT
            torch.tensor([x + 1, y, z]),  # RIGHT
            torch.tensor([x, y, z - 1]),  # DOWN_FLOOR
            torch.tensor([x, y, z + 1]),  # UP_FLOOR
        ]

        if self.boundary == "clamp":
            neighbors = [n for n in neighbors
                        if 0 <= n[0] < self.width and
                           0 <= n[1] < self.height and
                           0 <= n[2] < self.depth]

        return neighbors


class AspatialSubstrate(SpatialSubstrate):
    """No spatial substrate - pure state machine without positioning."""

    @property
    def position_dim(self) -> int:
        return 0  # No position!

    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        # Empty tensor - agents have no position
        return torch.zeros((num_agents, 0))

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        # No movement possible in aspatial universe
        return positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # No spatial distance - all agents are "everywhere and nowhere"
        return torch.zeros(pos1.shape[0])

    def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
        # No position encoding
        return torch.zeros((positions.shape[0], 0))

    def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        # No spatial neighbors in aspatial universe
        return []
```

### Phase 2: Create Substrate Config Schema

**File**: `src/townlet/substrate/config.py`

Task 002A ultimately landed the substrate DTOs in `townlet.substrate.config`. The shipped schema includes:

- `GridConfig` with explicit `topology` (`"square"` or `"cubic"`), `width`, `height`, optional `depth`, `boundary`, `distance_metric`, and `observation_encoding` (`"relative" | "scaled" | "absolute"`).
- `GridNDConfig` for ≥4D grids, with `dimension_sizes`, shared boundary/metric fields, observation encoding, and a `topology="hypercube"` marker.
- `ContinuousConfig` / `ContinuousNDConfig` describing float-based substrates (bounds per dimension, movement parameters, observation encoding).
- `AspatialSubstrateConfig` as a marker type.
- `SubstrateConfig` tying everything together with `type: Literal["grid", "gridnd", "continuous", "continuousnd", "aspatial"]` and mutually exclusive child configs enforced via a validator.

These DTOs are the ones referenced by VectorizedHamletEnv today; the original prototype (`environment/substrate_config.py`) was replaced during implementation.

### Phase 3: Update VectorizedEnv to Use Substrate

**File**: `src/townlet/environment/vectorized_env.py`

```python
# BEFORE: Hardcoded
self.grid_size = grid_size
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)

# AFTER: Load from substrate config
substrate_config_path = config_pack_path / "substrate.yaml"
substrate_config = load_substrate_config(substrate_config_path)

# Build substrate from config
self.substrate = SubstrateFactory.build_substrate(substrate_config, device=self.device)

# Initialize positions using substrate
self.positions = self.substrate.initialize_positions(self.num_agents)

# Update step() to use substrate
new_positions = self.substrate.apply_movement(self.positions, movement_deltas)
```

### Phase 4: Migrate Existing Configs

Create `substrate.yaml` for all existing config packs, replicating current behavior:

**configs/L0_0_minimal/substrate.yaml**:

```yaml
version: "1.0"
description: "2D square grid for L0 minimal"

substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [3, 3]
    boundary: "clamp"
    distance_metric: "manhattan"
```

**configs/L1_full_observability/substrate.yaml**:

```yaml
version: "1.0"
description: "2D square grid for L1 full observability"

substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [8, 8]
    boundary: "clamp"
    distance_metric: "manhattan"
```

### Phase 5: Create Example Alternative Substrates

**configs/L1_3D_house/substrate.yaml** (3D cubic grid):

```yaml
version: "1.0"
description: "3-story house simulation"

substrate:
  type: "grid"
  grid:
    topology: "cubic"
    dimensions: [8, 8, 3]
    boundary: "clamp"
    distance_metric: "manhattan"
```

**configs/L1_toroidal/substrate.yaml** (toroidal wraparound):

```yaml
version: "1.0"
description: "Pac-Man style wraparound world"

substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [8, 8]
    boundary: "wrap"  # Toroidal!
    distance_metric: "manhattan"
```

## Benefits

1. **Full UAC Compliance**: Spatial substrate now defined in config, not code
2. **Topology Flexibility**: Can model 2D, 3D, hex, continuous, graph, aspatial universes
3. **Experimentation**: Operators can:
   - Test 3D multi-story environments
   - Create toroidal worlds (wraparound boundaries)
   - Experiment with hexagonal grids
   - Remove spatial substrate entirely (pure resource management)
4. **Pedagogical Value**: Students learn topology, coordinate systems, continuous vs discrete
5. **Reveals Deep Insight**: Spatial substrate is optional - meters are the true universe

## Success Criteria - ✅ ALL COMPLETE

### Core Substrate Implementation

- [x] ✅ `SpatialSubstrate` abstract interface defined (`src/townlet/substrate/base.py`)
- [x] ✅ `substrate.yaml` schema defined with Pydantic DTOs (`src/townlet/substrate/config.py`)
- [x] ✅ `Grid2DSubstrate` implemented (replicates current behavior)
- [x] ✅ `Grid3DSubstrate` implemented (3D extension)
- [x] ✅ `GridNDSubstrate` implemented (4D-100D grids)
- [x] ✅ `Continuous1D/2D/3DSubstrate` implemented
- [x] ✅ `ContinuousNDSubstrate` implemented (4D-100D continuous)
- [x] ✅ Toroidal boundary support (wraparound) + bounce + sticky modes
- [x] ✅ `AspatialSubstrate` implemented (no positioning)
- [x] ✅ All existing configs have `substrate.yaml`
- [x] ✅ Can switch between 2D/3D by editing substrate.yaml (no code changes)

### Problem 1: obs_dim Variability

- [x] ✅ Substrate has `position_dim` property (not encoding_dim)
- [x] ✅ Environment aggregates substrate obs dims into total `observation_dim`
- [x] ✅ Different substrates produce correct obs_dim (Grid2D=29, Aspatial=13 with test config)

### Problem 5: Distance Semantics

- [x] ✅ Substrate has `compute_distance(pos1, pos2) → float` method
- [x] ✅ Substrate has `interaction_radius` for proximity checks
- [x] ✅ Continuous substrates use radius-based interaction detection
- [x] ✅ Grid substrates use exact position matching

### Problem 6: Observation Encoding (CRITICAL)

- [x] ✅ **Coordinate encoding works for all substrates** (relative, scaled, absolute modes)
- [x] ✅ Substrate has `normalize_positions()` method for observation encoding
- [x] ✅ Auto-selection via `observation_encoding` in substrate.yaml
- [x] ✅ **Transfer learning: network trained on 3×3 works on 8×8** (same obs_dim=29)
- [x] ✅ Backward compatibility: Grid2D uses "relative" encoding by default

### Problem 3: Visualization

- [x] ✅ Frontend supports multi-substrate rendering (Grid2D and Aspatial)
- [x] ✅ Grid2D: SVG-based 2D grid with heat maps
- [x] ✅ Aspatial: Meters-only dashboard (no fake grid)
- [x] ✅ WebSocket protocol includes substrate metadata
- [x] ✅ Substrate type detection in frontend

### Problem 4: Affordance Placement

- [x] ✅ `randomize_affordance_positions()` works for all grid types
- [x] ✅ Continuous substrates place affordances within bounds
- [x] ✅ Aspatial substrates have no affordance positions (empty tensor)
- [x] ✅ Substrate factory handles all types

### Validation & Testing

- [x] ✅ Substrate compilation errors caught at load time (Pydantic validation)
- [x] ✅ **All 1,159 tests pass** with new substrate system
- [x] ✅ **3D feasibility proven**: Grid3D implemented and tested
- [x] ✅ **Transfer learning verified**: Parameterized tests confirm obs_dim consistency
- [x] ✅ **23 Phase 8 tests**: Property-based, integration, regression tests all passing
- [x] ✅ **77% code coverage** overall, 85%+ for substrate modules

## Actual Implementation Summary

### Implementation Phases (As Executed)

**Total**: 8 phases completed (Nov 2024 - Nov 2025)

| Phase | Description | Commits | Status |
|-------|-------------|---------|--------|
| **0-2** | Foundation & Config Schema | 4034bd3-0afc680 | ✅ Complete |
| **3** | Config Migration (All Packs) | 1265142-af1abeb | ✅ Complete |
| **4** | Environment Integration | e53ea10-f612981 | ✅ Complete |
| **5** | Position Management | 32702c5-8ea4837 | ✅ Complete |
| **5B** | 3D + Continuous Substrates | a01250d-f6941fe | ✅ Complete |
| **5C** | N-Dimensional Substrates | (integrated with 5B) | ✅ Complete |
| **6** | Observation Builder | 8ea4837-367c43e | ✅ Complete |
| **7** | Frontend Visualization | 1dfbd32-606fdca | ✅ Complete |
| **8** | Testing & Verification | e989fcb-3d067e0 | ✅ Complete |

### Key Deliverables

**Substrate Types Implemented** (6 total):
- Grid2DSubstrate (2D discrete grid)
- Grid3DSubstrate (3D discrete grid)
- GridNDSubstrate (4D-100D discrete grids)
- Continuous1D/2D/3DSubstrate (smooth movement)
- ContinuousNDSubstrate (4D-100D continuous)
- AspatialSubstrate (no positioning)

**Configuration System**:
- `substrate.yaml` schema with Pydantic validation
- 3 observation encoding modes (relative, scaled, absolute)
- 4 boundary modes (clamp, wrap, bounce, sticky)
- 3 distance metrics (manhattan, euclidean, chebyshev)

**Testing Coverage**:
- 1,159 total tests (all passing)
- 23 substrate-specific tests (property-based, integration, regression)
- 77% overall coverage, 85%+ substrate modules

## Scope Evolution & Creep Analysis

### Original Plan (Nov 2024)

**Estimated Effort**: 15-22 hours (5 phases)

**Planned Scope**:
1. Phase 1: Substrate abstraction layer (6-8h)
2. Phase 2: Config schema (2-3h)
3. Phase 3: Environment integration (4-6h)
4. Phase 4: Config migration (3-5h)
5. Phase 5: Testing & validation (not estimated)

**Substrate Types Planned**:
- SquareGridSubstrate (2D)
- CubicGridSubstrate (3D)
- AspatialSubstrate (no positioning)

### Revised Estimate After Research (Nov 4, 2024)

**Revised Effort**: 51-65 hours (+140-195% increase)

**Reason**: Research uncovered 6 unsolved problems requiring significant work:
1. Observation dimension variability (3D grids explode obs_dim)
2. Action space compatibility (N-dimensional movement)
3. Visualization requirements (multi-substrate rendering)
4. Affordance placement (continuous vs discrete)
5. Distance semantics (graph/aspatial substrates)
6. **Coordinate encoding** (CRITICAL: 512 dims → 3 dims for 3D)

**Research Document**: `docs/research/RESEARCH-TASK-002A-UNSOLVED-PROBLEMS-CONSOLIDATED.md`

### Actual Implementation Scope Creep

**Final Scope Additions** (not in original plan):

#### Major Additions (+300% substrate types)
1. **GridNDSubstrate** (4D-100D discrete grids)
   - Supports up to 100-dimensional hypercubes
   - Dynamic action space: 2N + 2 actions
   - Configurable via `dimension_sizes` array
   - **Reason**: Generalizing 3D → N dimensions was minimal extra work

2. **Continuous Substrates** (3 types)
   - Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
   - Smooth movement with configurable `movement_delta`
   - Interaction radius for proximity detection
   - **Reason**: Robotics use cases, smooth pathfinding research

3. **ContinuousNDSubstrate** (4D-100D continuous)
   - Generalized continuous spaces
   - Configurable bounds per dimension
   - **Reason**: Abstract RL research (state spaces beyond 3D)

4. **Action Label System** (Phase 5B.3)
   - Configurable action terminology (gaming, robotics, naval, math)
   - Presets: gaming, 6dof, cardinal, math
   - Custom label support
   - **Reason**: Pedagogical value - reveals action semantics are arbitrary

#### Feature Additions
5. **Observation Encoding Modes** (3 modes instead of 1)
   - `relative`: Normalized coordinates [0,1] (transfer learning)
   - `scaled`: Normalized + dimension metadata (size-aware strategies)
   - `absolute`: Raw coordinates (physical simulation)
   - **Reason**: Different research paradigms need different encodings

6. **Boundary Modes** (4 modes instead of 2)
   - `clamp`: Hard walls (original)
   - `wrap`: Toroidal wraparound (original)
   - `bounce`: Elastic reflection (added)
   - `sticky`: Sticky walls (added)
   - **Reason**: Completeness, physics simulation support

7. **Frontend Multi-Substrate Rendering** (Phase 7)
   - Grid2D: SVG-based grid with heat maps
   - Aspatial: Meters-only dashboard (no fake grid)
   - WebSocket substrate metadata protocol
   - Substrate type detection and dispatch
   - **Reason**: Aspatial needed different UI, led to full multi-substrate system

8. **Comprehensive Testing Framework** (Phase 8)
   - Property-based tests (Hypothesis)
   - Parameterized integration tests
   - Regression tests (backward compatibility)
   - 23 substrate-specific tests
   - **Reason**: Complex system needed rigorous validation

### Scope Creep Summary

| Category | Original Plan | Actual Implementation | Growth |
|----------|---------------|----------------------|--------|
| **Substrate Types** | 3 types | 6 types | +100% |
| **Encoding Modes** | 1 mode | 3 modes | +200% |
| **Boundary Modes** | 2 modes | 4 modes | +100% |
| **Dimensionality Support** | 2D, 3D | 1D-100D | +infinite |
| **Config Files Created** | 5 packs | 8 packs + templates | +60% |
| **Test Files Created** | 0 (implicit) | 3 files, 23 tests | N/A |
| **Frontend Components** | 0 (out of scope) | 2 components | N/A |

### Why the Scope Expanded

**1. Generalization Was Cheap** (GridND, ContinuousND)
- Once Grid3D existed, GridND was 2-3 hours extra work
- Pattern-based implementation: loop over `dimension_sizes`
- Marginal cost for massive capability increase

**2. Research Value** (Continuous substrates, N-dimensional)
- Enables robotics research (smooth movement)
- Enables abstract RL research (high-dimensional state spaces)
- PhD students can experiment without code changes

**3. Pedagogical Completeness** (Action labels, encoding modes)
- Teaching moment: Labels are arbitrary, semantics matter
- Teaching moment: Encoding choice affects learning dynamics
- Reveals deep insights about RL representation learning

**4. Production Quality** (Phase 8 testing)
- Original plan lacked formal test strategy
- Property-based testing catches edge cases
- Regression tests prevent backward compatibility breaks

**5. User Experience** (Frontend Phase 7)
- Aspatial substrates needed different visualization
- Once solving for one substrate type, generalized to all
- WebSocket protocol upgrade needed anyway for metadata

### Actual Effort Estimate

**Not formally tracked**, but based on commit history:
- **Phases 0-5**: ~40-50 hours (foundation + core substrates)
- **Phase 5B-5C**: ~15-20 hours (3D, continuous, ND substrates)
- **Phase 6**: ~8-12 hours (observation builder integration)
- **Phase 7**: ~10-15 hours (frontend multi-substrate rendering)
- **Phase 8**: ~8-10 hours (testing & verification)

**Total Estimated**: **81-107 hours** (vs original 15-22h = **+368-586% growth**)

### Lessons Learned

**What Went Well**:
- Abstraction paid off: Adding GridND was trivial after Grid3D
- Property-based testing caught bugs early
- Comprehensive research (6 problems) prevented rework

**What Caused Bloat**:
- Feature creep: "While we're here, let's add N-dimensional..."
- Perfectionism: 3 encoding modes instead of 1
- Production quality: Extensive testing framework

**Was It Worth It?**
- **Yes**: System now supports 1D-100D substrates with 4 boundary modes
- **Yes**: Transfer learning works (train on 3×3, run on 8×8)
- **Yes**: Aspatial reveals deep insight (positioning is optional)
- **Maybe**: N-dimensional substrates rarely used in practice (but cheap to add)

## Original Effort Estimate (Pre-Research)

**⚠️ OBSOLETE - See "Scope Evolution" Above**

**Original Estimate**: 15-22h → **Revised**: 51-65h → **Actual**: ~81-107h

### Detailed Phase Breakdown

- **Phase 1** (substrate abstraction layer): **18-25 hours** (was 6-8h)
  - Create `SpatialSubstrate` interface: 2h
  - Implement `SquareGridSubstrate`: 2h
  - Implement `CubicGridSubstrate`: 2h
  - Implement `AspatialSubstrate`: 1h
  - **Add `position_encoding_dim` property** (Problem 1): 3h
  - **Add `is_adjacent()` + `compute_distance()`** (Problem 5): 4-6h
  - **Implement coordinate encoding** (Problem 6 - CRITICAL): 5h
  - Tests for abstraction layer: 2-3h

- **Phase 2** (config schema): **2-3 hours**
  - Pydantic DTOs for `substrate.yaml`: 2-3h

- **Phase 3** (environment integration): **12-16 hours** (was 4-6h)
  - Update `VectorizedHamletEnv` to use substrate: 4-6h
  - Use `substrate.position_encoding_dim` for obs_dim: 2h
  - Use `substrate.is_adjacent()` for interactions: 2-3h
  - Use `substrate.encode_position()` for observations: 3-4h
  - **Extend `randomize_affordance_positions()`** (Problem 4): 1h

- **Phase 4** (text visualization + config migration): **7-10 hours** (was 3-5h)
  - **Text renderer for debugging** (Problem 3): 4-6h
  - Create `substrate.yaml` for L0, L0.5, L1, L2, L3: 3-4h

- **Phase 5** (testing & validation): **12-11 hours** (NEW)
  - Test all substrate types: 4h
  - **Test coordinate encoding** (Problem 6): 3h
  - **Test transfer learning** (same network, different grid sizes): 3h
  - **Prove 3D feasibility** (512 dims → 3 dims): 2h

- **Total**: **51-65 hours**

### Critical Path Items

🔴 **Must implement** for 3D substrate support:

1. Coordinate encoding (Problem 6): 20h total
2. Distance semantics (Problem 5): 8-12h total
3. obs_dim property (Problem 1): 8-11h total

✅ **Deferred** to later tasks:

- Action validation → TASK-004A (Compiler): 17h saved
- GUI visualization → TASK-006 (separate project): 24-64h saved
- Explicit affordance positioning → Phase 2: 6h saved

## Risks

### 🚨 CRITICAL RISK: One-Hot Encoding Prevents 3D Substrates

**Problem**: Current one-hot position encoding **explodes for large/3D grids**:

| Substrate | One-Hot Dims | Feasible? |
|-----------|--------------|-----------|
| 2D (8×8) | 64 | ✅ Yes |
| 2D (16×16) | 256 | ⚠️ Marginal |
| **3D (8×8×3)** | **512** | ❌ **NO** |
| **3D (16×16×4)** | **1024** | ❌ **NO** |

**Impact**: Without mitigation, 3D cubic substrates cannot be implemented (512+ dimensions infeasible for input layer).

**Mitigation**: **Implement coordinate encoding** (normalized floats instead of one-hot)

- 3D (8×8×3): 512 dims → 3 dims (170× reduction!)
- 2D (16×16): 256 dims → 2 dims (128× reduction!)
- Enables transfer learning (same network works on any grid size)

**Evidence**: L2 POMDP already uses coordinate encoding successfully (`observation_builder.py:201`). Proven approach!

**Effort**: +20 hours (critical path, included in Phase 1/3/5)

**Status**: Research complete (Problem 6), implementation required for Phase 2

---

### Other Risks

- **Network Architecture**: Observation dim depends on substrate (2D vs 3D vs aspatial)
  - **Mitigation**: Substrate computes `position_encoding_dim` property, env aggregates into total `observation_dim`
  - **Status**: Research complete (Problem 1), implementation straightforward

- **Action Space Compatibility**: 3D needs 6 actions, hex needs 6, aspatial needs 0
  - **Mitigation**: Compile-time validation in TASK-004A (Universe Compiler Stage 4)
  - **Status**: Research complete (Problem 2), deferred to TASK-004A (17h)

- **Visualization**: Frontend assumes 2D square grid
  - **Mitigation**: Phase 1 uses text-based viz (4-6h), GUI deferred to TASK-006 (24-64h)
  - **Status**: Research complete (Problem 3), text viz sufficient for experimentation

- **Affordance Placement**: Different substrates need different position formats
  - **Mitigation**: Phase 1 extends random placement to all substrates (2h), explicit positioning deferred to Phase 2 (6h)
  - **Status**: Research complete (Problem 4), current approach works perfectly

- **Distance Semantics**: Pure distance metric fails for graph/aspatial substrates
  - **Mitigation**: Hybrid approach with `is_adjacent()` (primary) + `compute_distance()` (optional)
  - **Status**: Research complete (Problem 5), implementation required for Phase 1 (8-12h)

## Relationship to Other Tasks

**TASK-002A (Spatial Substrates)** establishes the foundation. Other tasks build on it:

- **TASK-001 (Schema Validation)**: Validates substrate.yaml structure
- **TASK-003 (Action Space)**: Actions must be compatible with substrate (6-way for hex, 0-way for aspatial)
- **TASK-002B (Universe Compilation)**: Validates substrate + actions + affordances work together
- **TASK-005 (BRAIN_AS_CODE)**: Network architecture depends on substrate observation encoding

**Recommended order**: TASK-002A → TASK-001 → TASK-003 → TASK-002B → TASK-005

## Design Principles

**Conceptual Agnosticism**: The substrate system should NOT assume:

- ❌ Space must be 2D or 3D (could be 4D, could be 0D aspatial)
- ❌ Grid must be square (could be hex, triangular, irregular graph)
- ❌ Distance must be Euclidean (could be Manhattan, graph distance, or meaningless for aspatial)
- ❌ Positioning is fundamental (aspatial universes have no concept of "position")

**Structural Enforcement**: The schema MUST enforce:

- ✅ Position tensor shape matches substrate.position_dim
- ✅ Boundary behavior is well-defined (clamp, wrap, bounce, fail)
- ✅ Distance metric matches topology (hexagonal distance for hex grids)
- ✅ Actions are valid for this substrate (caught in TASK-003)

**Permissive Semantics, Strict Syntax**:

- ✅ Allow: 3D, 4D, hexagonal, continuous, graph, aspatial
- ✅ Allow: Weird topologies students want to experiment with
- ❌ Reject: Invalid topology names, incompatible dimensions
- ❌ Reject: Type errors (string dimensions when expecting int)

**Key Insight**: The meters (bars) already form a continuous multidimensional state space. The spatial substrate is just an **optional overlay**. Making this explicit enables:

- Aspatial universes (pure resource management)
- Graph-based universes (abstract topologies)
- Hybrid universes (spatial + aspatial components)

This reveals that **positioning is a choice, not a requirement**.
