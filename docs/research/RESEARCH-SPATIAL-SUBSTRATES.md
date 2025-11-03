# Research: Configurable Spatial Substrates for UNIVERSE_AS_CODE

## Executive Summary

**Question**: Should HAMLET allow the "fabric of the universe" (spatial substrate) to be config-defined, enabling arbitrary topologies beyond 2D square grids?

**Answer**: **YES** - This is highly valuable for pedagogical experimentation and aligns perfectly with UNIVERSE_AS_CODE philosophy.

**Key Insight**: The bars (meters) already form a continuous multidimensional state space. The spatial grid is just an **optional overlay** for positioning affordances and enabling navigation mechanics. We should make this explicit and configurable.

---

## Current Hardcoded Assumptions

The current implementation hardcodes several spatial assumptions:

```python
# src/townlet/environment/vectorized_env.py

# 1. 2D topology (x, y coordinates)
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)

# 2. Square grid (x ∈ [0, grid_size), y ∈ [0, grid_size))
self.grid_size = grid_size  # Single dimension for both axes

# 3. Manhattan distance for proximity
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)  # L1 norm

# 4. Clamped boundaries (can't move outside grid)
new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)

# 5. Hardcoded movement deltas (UP/DOWN/LEFT/RIGHT)
# UP = (0, -1), DOWN = (0, +1), LEFT = (-1, 0), RIGHT = (+1, 0)
```

**These assumptions prevent**:

- 3D environments (villagers in multi-story buildings)
- Hexagonal grids (better distance metrics, more natural movement)
- Toroidal topologies (wraparound boundaries)
- Graph-based spaces (subway system, abstract state machines)
- Continuous spaces (smooth movement instead of discrete jumps)
- Non-spatial universes (pure state machines with no "position" concept)

---

## Spectrum of Spatial Substrates

### 1. Discrete Grid Topologies

**2D Square Grid** (current implementation)

- **Pros**: Simple to implement, intuitive visualization, maps to 2D arrays
- **Cons**: Diagonal distance ambiguity (√2 vs 1), axis-aligned bias
- **Use Cases**: Buildings, farms, simple navigation
- **Pedagogical Value**: ✅ (baseline, students understand it immediately)

**2D Hexagonal Grid**

- **Pros**: Uniform distance to all 6 neighbors, no diagonal ambiguity, better for natural terrain
- **Cons**: More complex indexing (axial/cube coordinates), harder to visualize in terminal
- **Use Cases**: Wargames, biological simulations, natural environments
- **Pedagogical Value**: ✅✅ (teaches coordinate system design, distance metrics)
- **Implementation**: Axial coordinates (q, r) or cube coordinates (x, y, z where x+y+z=0)

**2D Triangular Grid**

- **Pros**: Maximum packing density, interesting neighbor relationships
- **Cons**: Complex indexing, rare in practice
- **Use Cases**: Crystalline structures, experimental topologies
- **Pedagogical Value**: ✅ (niche but interesting for advanced students)

**3D Cubic Grid**

- **Pros**: Natural extension of 2D, enables vertical movement (stairs, elevators, floors)
- **Cons**: 3x more cells (memory), 6 movement directions (or 26 with diagonals)
- **Use Cases**: Multi-story buildings, Minecraft-like worlds, spatial planning
- **Pedagogical Value**: ✅✅✅ (very high! 3D navigation is a meaningful challenge)
- **Implementation**: Positions become (x, y, z), observation encoding needs depth channel

**4D+ Grids**

- **Pros**: Abstract reasoning, time as spatial dimension, embedding spaces
- **Cons**: Hard to visualize, unclear pedagogical value beyond "because we can"
- **Use Cases**: Time-travel mechanics, abstract state machines, theoretical exploration
- **Pedagogical Value**: ✅? (interesting for grad students exploring representation learning)

### 2. Boundary Conditions

**Clamped** (current)

- Movement outside bounds is clamped to boundary
- Creates "walls" at edges
- Pros: Intuitive, no surprises
- Cons: Boundary bias, corner states are different

**Toroidal (Wraparound)**

- Edges wrap around (Pac-Man style)
- Top edge connects to bottom, left to right
- Pros: No boundary bias, all states equivalent, useful for studying periodicity
- Cons: Confusing for beginners
- Pedagogical Value: ✅✅ (teaches topology, periodic boundary conditions)

**Bounce/Reflect**

- Movement outside bounds bounces back
- Pros: Preserves "attempted direction" momentum
- Cons: Rare in practice

**Fail/Invalid**

- Movement outside bounds is invalid action (no-op or penalty)
- Pros: Forces agent to learn boundaries explicitly
- Cons: Harsh, may slow learning

### 3. Continuous Spaces

**Continuous 2D/3D**

- Positions are floats, not ints: (x, y) ∈ [0.0, 1.0] × [0.0, 1.0]
- Actions are continuous deltas: move(dx, dy) where |dx|, |dy| ≤ max_speed
- Affordances have radius (collision detection)
- **Pros**: More realistic, enables smooth movement, velocity/acceleration
- **Cons**: More complex collision detection, harder to visualize, action space explosion
- **Use Cases**: Robotics, vehicle control, smooth navigation
- **Pedagogical Value**: ✅✅✅ (teaches continuous control, crucial for real-world RL)

**Hybrid: Discrete Affordances on Continuous Space**

- Agent moves continuously, but affordances are at fixed grid positions
- Combines smooth navigation with discrete interaction points
- **Pedagogical Value**: ✅✅ (teaches discretization vs continuity tradeoffs)

### 4. Graph-Based Spaces

**Arbitrary Graph**

- States are nodes, actions are edges
- Positions are graph node IDs, not coordinates
- Actions are "move to neighbor N" (variable action space per node!)
- **Pros**: Maximally flexible, can represent any topology (subway system, state machines, social networks)
- **Cons**: No spatial visualization, hard to reason about "distance"
- **Use Cases**: Transit systems, abstract state machines, non-Euclidean spaces
- **Pedagogical Value**: ✅✅✅ (teaches graph RL, action masking, topological reasoning)

**Examples**:

- Subway system: Nodes = stations, edges = routes
- Social network: Nodes = people, edges = relationships
- Workflow: Nodes = stages, edges = valid transitions

### 5. No Spatial Substrate (Pure State Space)

**Abstract/Aspatial**

- No concept of "position" at all
- Agent exists purely in meter space (energy, health, money, etc.)
- Affordances have no location - they're just "available" or "not available"
- Actions are pure state transitions (no movement)
- **Pros**: Simplest possible representation, focuses on resource management
- **Cons**: No navigation challenge, less intuitive visualization
- **Use Cases**: Economic simulations, process optimization, abstract MDPs
- **Pedagogical Value**: ✅✅✅ (teaches that spatial reasoning is optional, not fundamental)

**Key Insight**: This reveals that the meters ARE the universe. The grid is just window dressing for spatial navigation. An aspatial HAMLET would be pure resource management.

---

## Implementation Strategy: Substrate Abstraction Layer

### Proposed Architecture

**Create `substrate.yaml` in config packs:**

```yaml
version: "1.0"
description: "Spatial substrate configuration"

# Substrate type selection
substrate:
  type: "grid"  # grid, continuous, graph, aspatial

  # Grid-specific configuration
  grid:
    topology: "square"  # square, hexagonal, triangular, cubic
    dimensions: [8, 8]  # [width, height] or [width, height, depth]
    boundary: "clamp"   # clamp, wrap, bounce, fail
    distance_metric: "manhattan"  # manhattan, euclidean, chebyshev

  # Continuous-specific configuration
  continuous:
    dimensions: 2  # 2D or 3D
    bounds: [[0.0, 10.0], [0.0, 10.0]]  # [min, max] per dimension
    boundary: "clamp"
    discretization: 0.1  # Step size for movement actions

  # Graph-specific configuration
  graph:
    nodes: 16  # Number of nodes (stations, rooms, etc.)
    adjacency: "config"  # config, complete, random, spatial
    # If adjacency: "config", define edges explicitly
    edges:
      - [0, 1]  # Node 0 connects to Node 1
      - [1, 2]
      - [2, 3]
      # ... etc
    # If adjacency: "spatial", generate graph from spatial positions
    spatial_threshold: 2.0  # Connect nodes within distance

  # Aspatial configuration (no positioning)
  aspatial:
    enabled: true
    # Affordances exist in abstract "availability" space
    # Agent has no position, just accesses affordances directly
```

**Coordinate System Interface**:

```python
# src/townlet/environment/substrate.py

from abc import ABC, abstractmethod

class SpatialSubstrate(ABC):
    """Abstract interface for spatial substrates."""

    @abstractmethod
    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        """Initialize random positions for agents."""
        pass

    @abstractmethod
    def apply_action(self, positions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply movement actions to positions."""
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
    def get_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        """Get valid neighbor positions (for action validation)."""
        pass

    @property
    @abstractmethod
    def position_dim(self) -> int:
        """Dimensionality of position vectors (2 for 2D, 3 for 3D, etc.)"""
        pass

class SquareGridSubstrate(SpatialSubstrate):
    """2D square grid substrate (current implementation)."""

    def __init__(self, width: int, height: int, boundary: str = "clamp"):
        self.width = width
        self.height = height
        self.boundary = boundary

    @property
    def position_dim(self) -> int:
        return 2

    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        return torch.stack([
            torch.randint(0, self.width, (num_agents,)),
            torch.randint(0, self.height, (num_agents,))
        ], dim=1)

    def apply_action(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        new_positions = positions + deltas

        if self.boundary == "clamp":
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)
        elif self.boundary == "wrap":
            new_positions[:, 0] = new_positions[:, 0] % self.width
            new_positions[:, 1] = new_positions[:, 1] % self.height

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Manhattan distance
        return torch.abs(pos1 - pos2).sum(dim=-1)

    def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
        # One-hot encoding of grid cells
        flat_indices = positions[:, 1] * self.width + positions[:, 0]
        one_hot = torch.zeros((positions.shape[0], self.width * self.height))
        one_hot.scatter_(1, flat_indices.unsqueeze(1), 1)
        return one_hot

    def get_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        # 4 cardinal neighbors
        x, y = position
        neighbors = [
            torch.tensor([x, y - 1]),  # UP
            torch.tensor([x, y + 1]),  # DOWN
            torch.tensor([x - 1, y]),  # LEFT
            torch.tensor([x + 1, y]),  # RIGHT
        ]
        # Filter out-of-bounds if boundary is clamp
        if self.boundary == "clamp":
            neighbors = [n for n in neighbors
                        if 0 <= n[0] < self.width and 0 <= n[1] < self.height]
        return neighbors

class HexagonalGridSubstrate(SpatialSubstrate):
    """2D hexagonal grid substrate using axial coordinates."""

    def __init__(self, size: int, boundary: str = "clamp"):
        self.size = size  # Hexagonal grid radius
        self.boundary = boundary

    @property
    def position_dim(self) -> int:
        return 2  # Axial coordinates (q, r)

    def apply_action(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        # Hexagonal movement uses axial coordinate deltas
        # 6 directions: E, NE, NW, W, SW, SE
        return positions + deltas

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Hex distance = (|q1-q2| + |r1-r2| + |q1+r1-q2-r2|) / 2
        q1, r1 = pos1[:, 0], pos1[:, 1]
        q2, r2 = pos2[:, 0], pos2[:, 1]
        return (torch.abs(q1 - q2) + torch.abs(r1 - r2) + torch.abs(q1 + r1 - q2 - r2)) / 2

    def get_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        # 6 hex directions in axial coordinates
        q, r = position
        return [
            torch.tensor([q + 1, r]),      # E
            torch.tensor([q + 1, r - 1]),  # NE
            torch.tensor([q, r - 1]),      # NW
            torch.tensor([q - 1, r]),      # W
            torch.tensor([q - 1, r + 1]),  # SW
            torch.tensor([q, r + 1]),      # SE
        ]

class CubicGridSubstrate(SpatialSubstrate):
    """3D cubic grid substrate."""

    def __init__(self, width: int, height: int, depth: int, boundary: str = "clamp"):
        self.width = width
        self.height = height
        self.depth = depth
        self.boundary = boundary

    @property
    def position_dim(self) -> int:
        return 3  # (x, y, z)

    def apply_action(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        new_positions = positions + deltas

        if self.boundary == "clamp":
            new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
            new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)
            new_positions[:, 2] = torch.clamp(new_positions[:, 2], 0, self.depth - 1)

        return new_positions

    def get_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        # 6 cardinal neighbors in 3D
        x, y, z = position
        return [
            torch.tensor([x, y - 1, z]),  # UP
            torch.tensor([x, y + 1, z]),  # DOWN
            torch.tensor([x - 1, y, z]),  # LEFT
            torch.tensor([x + 1, y, z]),  # RIGHT
            torch.tensor([x, y, z - 1]),  # UP (Z-axis)
            torch.tensor([x, y, z + 1]),  # DOWN (Z-axis)
        ]

class GraphSubstrate(SpatialSubstrate):
    """Graph-based substrate (nodes + edges)."""

    def __init__(self, num_nodes: int, adjacency_matrix: torch.Tensor):
        self.num_nodes = num_nodes
        self.adjacency = adjacency_matrix  # [num_nodes, num_nodes] boolean

    @property
    def position_dim(self) -> int:
        return 1  # Node ID

    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        # Random node IDs
        return torch.randint(0, self.num_nodes, (num_agents, 1))

    def apply_action(self, positions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Actions are "move to neighbor N"
        # This requires action masking! Can only move to adjacent nodes.
        # For now, treat action as target node ID and validate
        new_positions = actions.clone()

        # Validate: Can only move to adjacent nodes
        for i in range(positions.shape[0]):
            current_node = positions[i, 0]
            target_node = actions[i, 0]
            if not self.adjacency[current_node, target_node]:
                # Invalid move, stay in place
                new_positions[i, 0] = current_node

        return new_positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Graph distance (shortest path) - expensive to compute!
        # For now, simple: 0 if same node, 1 if adjacent, infinity otherwise
        same_node = (pos1 == pos2).all(dim=-1)
        adjacent = self.adjacency[pos1[:, 0], pos2[:, 0]]
        return torch.where(same_node, 0.0, torch.where(adjacent, 1.0, float('inf')))

    def get_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        node_id = position[0].item()
        neighbor_ids = torch.where(self.adjacency[node_id])[0]
        return [torch.tensor([nid]) for nid in neighbor_ids]

class AspatialSubstrate(SpatialSubstrate):
    """No spatial substrate - pure state machine."""

    @property
    def position_dim(self) -> int:
        return 0  # No position!

    def initialize_positions(self, num_agents: int) -> torch.Tensor:
        # Empty tensor - no positions
        return torch.zeros((num_agents, 0))

    def apply_action(self, positions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # No movement - positions don't change (they don't exist!)
        return positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # No spatial distance - all agents are "at the same place" (nowhere)
        return torch.zeros(pos1.shape[0])

    def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
        # No position encoding
        return torch.zeros((positions.shape[0], 0))

    def get_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
        # No spatial neighbors
        return []
```

---

## Pedagogical Value Assessment

### High-Value Substrates (Implement First)

**3D Cubic Grid** ✅✅✅

- **Why**: Natural extension students understand immediately
- **Learning**: Vertical movement, 3D spatial reasoning, floor/level mechanics
- **Use Cases**: Multi-story house, apartment building, Minecraft-like
- **Implementation Effort**: Low (just add Z dimension)
- **Student Project**: "Add stairs and elevators to your 3D house"

**Continuous 2D Space** ✅✅✅

- **Why**: Teaches discrete vs continuous control (fundamental RL concept)
- **Learning**: Smooth movement, collision detection, continuous actions
- **Use Cases**: Robotics sim, vehicle control, realistic navigation
- **Implementation Effort**: Medium (need collision detection)
- **Student Project**: "Make your agent move smoothly instead of jumping between cells"

**Hexagonal Grid** ✅✅

- **Why**: Teaches coordinate system design, distance metrics
- **Learning**: Alternative representations, uniform distances, axial coordinates
- **Use Cases**: Strategy games, biological sims, natural terrain
- **Implementation Effort**: Medium (axial coordinate math)
- **Student Project**: "Why do hexagons make better battle maps?"

**Toroidal Boundaries (Wraparound)** ✅✅

- **Why**: Teaches topology, periodic boundary conditions (common in physics/biology sims)
- **Learning**: No boundary bias, equivalence of all states
- **Use Cases**: Pac-Man, toroidal worlds, studying periodicity
- **Implementation Effort**: Trivial (modulo instead of clamp)
- **Student Project**: "Make a Pac-Man universe where edges connect"

### Medium-Value Substrates (Implement If Students Request)

**Graph-Based** ✅✅

- **Why**: Teaches non-Euclidean spaces, action masking, graph RL
- **Learning**: Topological reasoning, variable action spaces
- **Use Cases**: Subway system, social networks, state machines
- **Implementation Effort**: High (need graph traversal, action masking)
- **Student Project**: "Model a subway system where agents commute between stations"

**Aspatial (Pure State Machine)** ✅✅

- **Why**: Reveals that spatial substrate is optional - meters ARE the universe
- **Learning**: Resource management without navigation, pure MDPs
- **Use Cases**: Economic sims, process optimization
- **Implementation Effort**: Low (remove positioning)
- **Student Project**: "What if there's no space, just resource management?"

### Low-Value Substrates (Skip Unless Compelling Use Case)

**4D+ Grids** ✅?

- **Why**: Hard to visualize, unclear pedagogical value
- **Learning**: Abstract reasoning, embedding spaces
- **Use Cases**: Time-travel, theoretical exploration
- **Implementation Effort**: Low (just more dimensions)
- **Student Project**: "What does time as a spatial dimension mean?"

**Triangular Grid** ✅

- **Why**: Rare in practice, no compelling advantage over hex
- **Learning**: Alternative tessellations
- **Implementation Effort**: Medium
- **Skip**: Unless student has specific use case

---

## Recommended Implementation Order

### Phase 1: Substrate Abstraction (Foundation)

1. Create `SpatialSubstrate` abstract interface
2. Refactor existing code to use `SquareGridSubstrate` (no behavior change)
3. Create `substrate.yaml` config schema
4. Update `VectorizedHamletEnv` to load substrate from config

**Goal**: Prove the abstraction works without changing behavior

### Phase 2: Low-Hanging Fruit (High Value, Low Effort)

1. Implement `ToroidalBoundary` (just change clamp to modulo)
2. Implement `CubicGridSubstrate` (3D extension)
3. Create example config packs:
   - `L1_3D_house` (3-story building)
   - `L1_toroidal` (Pac-Man wraparound)

**Goal**: Demonstrate substrate flexibility with minimal implementation

### Phase 3: Advanced Substrates (High Value, Medium Effort)

1. Implement `ContinuousSubstrate` (smooth movement)
2. Implement `HexagonalGridSubstrate` (axial coordinates)
3. Create example config packs:
   - `L1_continuous_robot` (smooth navigation)
   - `L1_hexagonal_terrain` (hex strategy game)

**Goal**: Cover major substrate types students will want

### Phase 4: Exotic Substrates (On Demand)

1. Implement `GraphSubstrate` (if student requests subway system)
2. Implement `AspatialSubstrate` (if student wants pure resource management)
3. Create example config packs as needed

**Goal**: Support experimental use cases as they arise

---

## Key Design Principles

### 1. Substrate Is Optional, Not Fundamental

The meters (energy, health, money, etc.) ARE the universe. The spatial substrate is just an **overlay** for:

- Positioning affordances
- Enabling navigation mechanics
- Creating distance-based challenges

An aspatial HAMLET is perfectly valid - pure resource management without movement.

### 2. Permissive Semantics, Strict Syntax

- ✅ **Allow**: 3D grids, hexagons, continuous spaces, graphs, aspatial
- ✅ **Allow**: Weird topologies students want to experiment with
- ❌ **Reject**: Invalid coordinate systems, undefined boundary behaviors
- ❌ **Reject**: Type errors (string position when expecting int)

### 3. Substrate-Agnostic Core Logic

The universe compiler should NOT assume:

- ❌ Positions must be 2D
- ❌ Distance must be Euclidean or Manhattan
- ❌ Boundaries must be clamped
- ❌ Actions must include movement

The compiler validates:

- ✅ Position tensor shape matches substrate.position_dim
- ✅ Actions are valid for this substrate (can't move UP in aspatial universe)
- ✅ Affordances fit in substrate (N affordances in M nodes/cells)

### 4. Fail Fast with Clear Errors

```
❌ SUBSTRATE COMPILATION FAILED
Substrate type 'continuous' requires action space with continuous movement.
Current actions.yaml defines discrete actions (UP, DOWN, LEFT, RIGHT).

For continuous substrate, define actions like:
  - name: "MOVE"
    type: "continuous_movement"
    max_delta: 0.5  # Maximum movement per step

Or switch substrate to 'grid' for discrete movement.
```

---

## Challenges and Mitigations

### Challenge 1: Observation Encoding Varies by Substrate

**Problem**: 2D grid uses one-hot encoding. 3D grid needs more dims. Continuous uses raw positions. Graph uses node IDs.

**Solution**: Each substrate implements `encode_observation()` method. Network input dim is computed from substrate config.

### Challenge 2: Action Space Depends on Substrate

**Problem**: Square grid has 4 movement actions. Hex has 6. 3D has 6 (or 26 with diagonals). Graph has variable actions per node. Continuous needs float deltas.

**Solution**:

- `actions.yaml` must be compatible with `substrate.yaml`
- Universe compiler validates compatibility
- Example: `substrate: continuous` requires `action_type: continuous_movement`

### Challenge 3: Visualization Depends on Substrate

**Problem**: Frontend currently assumes 2D square grid. How to visualize 3D? Hex? Graph?

**Solution**:

- Phase 1: Text-based visualization (show coordinates)
- Phase 2: 2D projections (3D → top-down view, graph → force-directed layout)
- Phase 3: 3D WebGL rendering (if students want it)

### Challenge 4: Distance Metrics Vary

**Problem**: Manhattan distance for square grid, hex distance for hexagons, Euclidean for continuous, shortest-path for graphs.

**Solution**: Each substrate implements `compute_distance()` method. Core logic is distance-agnostic.

---

## Recommendation

**YES, implement configurable spatial substrates.**

**Priority Order**:

1. **Phase 1**: Substrate abstraction layer (refactor current code)
2. **Phase 2**: Toroidal boundaries + 3D cubic grid (low effort, high pedagogy)
3. **Phase 3**: Continuous space + hexagonal grid (medium effort, high pedagogy)
4. **Phase 4**: Graph + aspatial (on demand, if students request)

**Rationale**:

- Aligns with UNIVERSE_AS_CODE philosophy ("everything configurable")
- High pedagogical value ("fuck around and find out")
- Reveals that spatial substrate is optional (meters are the true universe)
- Enables interesting student projects (3D houses, smooth robots, subway systems)
- Low bar for Phase 1-2 (mostly refactoring + one new dimension)

**Effort Estimate**:

- Phase 1 (abstraction): 6-8 hours
- Phase 2 (toroidal + 3D): 4-6 hours
- Phase 3 (continuous + hex): 8-12 hours
- Phase 4 (graph + aspatial): 6-10 hours (on demand)
- **Total (Phases 1-2)**: 10-14 hours for core flexibility + high-value substrates

**Success Metrics**:

- Student can switch between 2D/3D by editing substrate.yaml
- Student can create toroidal Pac-Man world in 10 minutes
- Student can experiment with hexagonal grids without code changes
- Substrate compilation errors are clear and actionable

---

## Conclusion

Configurable spatial substrates are **highly valuable** for HAMLET's pedagogical mission. They:

1. **Teach fundamental concepts**: Discrete vs continuous, topology, coordinate systems
2. **Enable experimentation**: "What if my world was a sphere?" "What if agents could fly?"
3. **Reveal deep insights**: Spatial substrate is optional - the meters ARE the universe
4. **Align with UNIVERSE_AS_CODE**: Everything should be configurable, including the fabric of reality

The bar should be low because "fuck around and find out" is half the point. Start with Phase 1-2 (abstraction + 3D + toroidal) to prove the concept, then expand based on student demand.

**Final recommendation: IMPLEMENT IT.** 🌍🔨
