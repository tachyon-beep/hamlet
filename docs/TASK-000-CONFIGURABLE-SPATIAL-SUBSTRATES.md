# TASK-000: Configurable Spatial Substrates

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

**File**: `src/townlet/environment/substrate_config.py`

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal

class GridSubstrateConfig(BaseModel):
    """Grid substrate configuration."""
    topology: Literal["square", "hexagonal", "triangular", "cubic"]  # REQUIRED
    dimensions: list[int] = Field(min_length=2, max_length=3)  # REQUIRED: [w, h] or [w, h, d]
    boundary: Literal["clamp", "wrap", "bounce", "fail"]  # REQUIRED
    distance_metric: Literal["manhattan", "euclidean", "chebyshev", "hexagonal"]  # REQUIRED

    @model_validator(mode="after")
    def validate_dimensions(self) -> "GridSubstrateConfig":
        if self.topology == "cubic" and len(self.dimensions) != 3:
            raise ValueError("Cubic topology requires 3 dimensions [width, height, depth]")
        if self.topology in ["square", "hexagonal", "triangular"] and len(self.dimensions) != 2:
            raise ValueError(f"{self.topology} topology requires 2 dimensions [width, height]")
        return self

class ContinuousSubstrateConfig(BaseModel):
    """Continuous substrate configuration."""
    dimensions: int = Field(ge=2, le=3)  # REQUIRED: 2D or 3D
    bounds: list[list[float]] = Field(min_length=2, max_length=3)  # REQUIRED: [[min, max], ...]
    boundary: Literal["clamp", "wrap"]  # REQUIRED
    discretization: float = Field(gt=0.0)  # REQUIRED: step size for actions

class GraphSubstrateConfig(BaseModel):
    """Graph substrate configuration."""
    nodes: int = Field(gt=0)  # REQUIRED: number of nodes
    adjacency: Literal["config", "complete", "random", "spatial"]  # REQUIRED
    edges: list[list[int]] | None = None  # Required if adjacency="config"

    @model_validator(mode="after")
    def validate_edges(self) -> "GraphSubstrateConfig":
        if self.adjacency == "config" and not self.edges:
            raise ValueError("adjacency='config' requires explicit edge list")
        return self

class AspatialSubstrateConfig(BaseModel):
    """Aspatial substrate configuration (no positioning)."""
    enabled: bool = True

class SubstrateConfig(BaseModel):
    """Complete spatial substrate configuration."""
    version: str  # REQUIRED
    description: str  # REQUIRED (metadata)
    type: Literal["grid", "continuous", "graph", "aspatial"]  # REQUIRED

    # Substrate-specific configs (only one should be populated based on type)
    grid: GridSubstrateConfig | None = None
    continuous: ContinuousSubstrateConfig | None = None
    graph: GraphSubstrateConfig | None = None
    aspatial: AspatialSubstrateConfig | None = None

    @model_validator(mode="after")
    def validate_substrate_type(self) -> "SubstrateConfig":
        """Ensure substrate config matches type."""
        if self.type == "grid" and self.grid is None:
            raise ValueError("type='grid' but grid config missing")
        if self.type == "continuous" and self.continuous is None:
            raise ValueError("type='continuous' but continuous config missing")
        if self.type == "graph" and self.graph is None:
            raise ValueError("type='graph' but graph config missing")
        if self.type == "aspatial" and self.aspatial is None:
            raise ValueError("type='aspatial' but aspatial config missing")
        return self

def load_substrate_config(config_path: Path) -> SubstrateConfig:
    """Load and validate substrate configuration."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return SubstrateConfig(**data)
```

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

**configs/L0_minimal/substrate.yaml**:
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

## Success Criteria

### Core Substrate Implementation
- [ ] `SpatialSubstrate` abstract interface defined
- [ ] `substrate.yaml` schema defined with Pydantic DTOs
- [ ] `SquareGridSubstrate` implemented (replicates current behavior)
- [ ] `CubicGridSubstrate` implemented (3D extension)
- [ ] Toroidal boundary support (wraparound)
- [ ] `AspatialSubstrate` implemented (no positioning)
- [ ] All existing configs have `substrate.yaml`
- [ ] Can switch between 2D/3D by editing substrate.yaml (no code changes)

### Problem 1: obs_dim Variability
- [ ] Substrate has `position_encoding_dim` property
- [ ] Environment aggregates `substrate.position_encoding_dim` into total `observation_dim`
- [ ] Different substrates produce correct obs_dim (2D=91, 3D=539, aspatial=27)

### Problem 5: Distance Semantics
- [ ] Substrate has `is_adjacent(pos1, pos2) → bool` method
- [ ] Substrate has `compute_distance(pos1, pos2) → float` method
- [ ] Interactions use `is_adjacent()` check (not raw distance)
- [ ] Aspatial substrate returns `True` for `is_adjacent()` (everything accessible)

### Problem 6: Observation Encoding (CRITICAL)
- [ ] **Coordinate encoding works for 3D cubic grids** (512 dims → 3 dims)
- [ ] Substrate has `encode_position(positions) → torch.Tensor` method
- [ ] Auto-selection: small grids use one-hot, large/3D use coordinates
- [ ] **Transfer learning: network trained on 8×8 works on 16×16** (same obs_dim)
- [ ] L1 backward compatibility: one-hot encoding still works for ≤8×8

### Problem 3: Visualization
- [ ] Text visualization renders all substrate types for debugging
- [ ] 2D square grid ASCII rendering
- [ ] 3D cubic floor-by-floor projection
- [ ] Aspatial meters-only dashboard

### Problem 4: Affordance Placement
- [ ] `randomize_affordance_positions()` works for 2D grids
- [ ] `randomize_affordance_positions()` works for 3D cubic grids
- [ ] `randomize_affordance_positions()` works for aspatial (returns empty tensor)

### Validation & Testing
- [ ] Substrate compilation errors caught at load time
- [ ] All tests pass with new substrate system
- [ ] **3D feasibility proof**: 8×8×3 grid runs without memory issues
- [ ] **Transfer learning test**: same network works on different grid sizes

## Estimated Effort

**⚠️ REVISED AFTER RESEARCH (2025-11-04)**: Original estimate 15-22h → **51-65h** (+140-195%)

Research into 6 unsolved problems revealed critical additions needed for 3D substrate support and proper abstraction.
See: `docs/research/RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md`

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
- Action validation → TASK-004 (Compiler): 17h saved
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
  - **Mitigation**: Compile-time validation in TASK-004 (Universe Compiler Stage 4)
  - **Status**: Research complete (Problem 2), deferred to TASK-004 (17h)

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

**TASK-000 (Spatial Substrates)** establishes the foundation. Other tasks build on it:

- **TASK-001 (Schema Validation)**: Validates substrate.yaml structure
- **TASK-002 (Action Space)**: Actions must be compatible with substrate (6-way for hex, 0-way for aspatial)
- **TASK-003 (Universe Compilation)**: Validates substrate + actions + affordances work together
- **TASK-004 (BRAIN_AS_CODE)**: Network architecture depends on substrate observation encoding

**Recommended order**: TASK-000 → TASK-001 → TASK-002 → TASK-003 → TASK-004

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
- ✅ Actions are valid for this substrate (caught in TASK-002)

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
