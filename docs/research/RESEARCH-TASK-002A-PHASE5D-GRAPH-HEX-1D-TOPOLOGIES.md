# Research: Task-002a Phase 5D - Graph, Hexagonal, and 1D Topologies

**Date**: 2025-11-05
**Status**: Deep Dive Research (Docs-Only)
**Context**: Reviewing phase-5d requirements after completing phases 1-5c
**Scope**: Graph-Based, Hexagonal Grid, and 1D Grid topologies

---

## Executive Summary

This research document analyzes the implementation requirements for **Phase 5D: Alternative Topologies**, which adds three new substrate types to the configurable spatial substrate system:

1. **Graph-Based Substrate** (✅✅ High Priority) - Non-Euclidean spaces, variable action spaces, graph RL
2. **Hexagonal Grid Substrate** (✅✅ High Priority) - Uniform distances, axial coordinates, strategic games
3. **1D Grid Substrate** (✅ Medium Priority) - Linear movement, simplest spatial case

**Key Findings:**

- **Graph substrate is architecturally distinct** - Requires variable action spaces, edge-based movement, action masking
- **Hex substrate fits existing patterns** - Similar to Grid2D but with 6 neighbors and axial coordinate math
- **1D substrate is trivial** - Can be implemented in ~2 hours as a specialized Grid1DSubstrate

**Estimated Total Effort**: 28-38 hours

- Graph: 18-24 hours (complex, requires action masking infrastructure)
- Hex: 8-10 hours (moderate, requires coordinate system)
- 1D: 2-4 hours (simple, specialized case)

**Pedagogical Value Rankings:**

1. **Graph** (✅✅✅) - Teaches graph RL, action masking, non-Euclidean reasoning, variable action spaces
2. **Hex** (✅✅) - Teaches coordinate system design, uniform distance metrics, strategic thinking
3. **1D** (✅) - Teaches simplification, edge cases, pedagogical progression from 1D→2D→3D

**Dependencies:**

- Phase 5C must be complete (observation encoding retrofit, N-dimensional substrates)
- Action masking infrastructure needed for Graph substrate (not yet implemented)
- Frontend visualization updates needed for all three topologies

---

## Background: What's Been Implemented (Phases 1-5C)

Before analyzing Phase 5D requirements, let's review the existing substrate architecture:

### Existing Substrate Types (Phases 1-5C)

**Grid Substrates (Discrete Lattice):**
- `Grid2DSubstrate` - 2D square grid, 4 neighbors, integer positions
- `Grid3DSubstrate` - 3D cubic grid, 6 neighbors, integer positions
- `GridNDSubstrate` - N-dimensional hypercube (N≥4), 2N neighbors, integer positions

**Continuous Substrates (Smooth Space):**
- `Continuous1DSubstrate` - 1D line, float positions
- `Continuous2DSubstrate` - 2D plane, float positions
- `Continuous3DSubstrate` - 3D volume, float positions
- `ContinuousNDSubstrate` - N-dimensional space (N≥4), float positions

**Special Cases:**
- `AspatialSubstrate` - No positioning, pure state machine

### Key Substrate Interface Methods (from base.py)

```python
class SpatialSubstrate(ABC):
    position_dim: int                    # Number of dimensions (0 for aspatial)
    position_dtype: torch.dtype          # torch.long or torch.float32

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions (2*position_dim + 1 for grids)."""

    def initialize_positions(self, num_agents, device) -> Tensor:
        """Random agent placement."""

    def apply_movement(self, positions, deltas) -> Tensor:
        """Apply movement with boundary handling."""

    def compute_distance(self, pos1, pos2) -> Tensor:
        """Distance metric (manhattan/euclidean/chebyshev)."""

    def encode_observation(self, positions, affordances) -> Tensor:
        """Position encoding for observation space."""

    def get_observation_dim(self) -> int:
        """Size of position encoding."""

    def get_valid_neighbors(self, position) -> list[Tensor]:
        """Valid neighbor positions (for action masking)."""

    def is_on_position(self, agent_positions, target_position) -> Tensor:
        """Check if agents are on target (for interactions)."""

    def get_all_positions(self) -> list[list[int|float]]:
        """All valid positions (for affordance randomization)."""
```

### Configuration Discrimination (substrate factory)

The factory uses the `topology` field to select substrate class:

```yaml
# substrate.yaml
type: "grid"
grid:
  topology: "square"     # → Grid2DSubstrate
  topology: "cubic"      # → Grid3DSubstrate
  topology: "hypercube"  # → GridNDSubstrate
  topology: "hexagonal"  # → NOT YET IMPLEMENTED (Phase 5D)
  topology: "graph"      # → NOT YET IMPLEMENTED (Phase 5D)
  topology: "line"       # → NOT YET IMPLEMENTED (Phase 5D)
```

---

## Phase 5D Topology #1: Graph-Based Substrate

### Overview

**What It Is:**
- Positions are **graph node IDs**, not coordinates
- Movement is **edge traversal**, not spatial deltas
- Action space is **variable per node** (different nodes have different numbers of neighbors)
- Distance is **shortest path length** in graph

**Why It Matters:**
- Teaches **graph RL** (fundamental for many real-world problems)
- Teaches **action masking** (variable action spaces)
- Teaches **non-Euclidean reasoning** (topological vs metric spaces)
- Enables **abstract state machines** (workflows, processes, social networks)

**Pedagogical Value**: ✅✅✅ (Very High)

### Use Cases

1. **Subway/Transit System**
   - Nodes = Stations
   - Edges = Rail lines
   - Agent learns optimal routing
   - Example: "Get from Times Square to Brooklyn Bridge using fewest transfers"

2. **Social Network Navigation**
   - Nodes = People
   - Edges = Relationships (friend, colleague, family)
   - Agent learns social path finding
   - Example: "Reach the CEO through your network"

3. **Workflow State Machine**
   - Nodes = Process stages
   - Edges = Valid transitions
   - Agent learns process optimization
   - Example: "Navigate bureaucratic approval process"

4. **Abstract MDP**
   - Nodes = Abstract states
   - Edges = State transitions
   - Agent learns pure policy (no spatial component)
   - Example: "Optimize resource allocation across departments"

### Technical Challenges

#### Challenge 1: Variable Action Spaces

**Problem**: Different nodes have different numbers of neighbors (degree varies).

**Current Architecture Assumption**: Fixed action space size (5 for 2D, 7 for 3D, 2N+1 for ND).

**Solution Required**: Action masking infrastructure

```python
# Example: Node A has 3 neighbors, Node B has 5 neighbors
# Agent at Node A: Valid actions = [0, 1, 2, 6] (3 edges + INTERACT)
# Agent at Node B: Valid actions = [0, 1, 2, 3, 4, 6] (5 edges + INTERACT)
# Invalid actions must be masked in Q-network output
```

**Implementation Approach**:

1. **Fixed Action Space with Masking**
   - Define max_edges (e.g., 10) → action_space_size = max_edges + 1
   - Q-network always outputs 11 values
   - Mask invalid actions during action selection
   - Pro: No Q-network changes needed
   - Con: Wastes capacity for low-degree nodes

2. **Dynamic Action Space** (more complex)
   - Q-network output size varies per state
   - Requires padding/masking in batch processing
   - Pro: More efficient
   - Con: Requires Q-network architecture changes

**Recommendation**: Start with Fixed Action Space + Masking (simpler, compatible with existing DQN).

#### Challenge 2: Position Representation

**Problem**: Graph nodes don't have coordinates - they're just IDs.

**Current Architecture Assumption**: Positions are tensors of shape `[num_agents, position_dim]` with numeric coordinates.

**Solution Options**:

1. **Node ID as 1D Position** (Simplest)
   ```python
   position_dim = 1
   positions = torch.tensor([[3], [7], [2]], dtype=torch.long)  # Agents at nodes 3, 7, 2
   ```
   - Pro: Fits existing position tensor shape
   - Con: Implies ordering that doesn't exist (node 3 isn't "between" nodes 2 and 4)

2. **One-Hot Node Encoding**
   ```python
   position_dim = num_nodes
   positions = torch.tensor([[0,0,0,1,0,...], [0,0,0,0,0,0,0,1,...]], dtype=torch.float32)
   ```
   - Pro: No ordinal implications
   - Con: Observation dims explode (100 nodes = 100 dims)

3. **Learned Embeddings** (Advanced, defer to future)
   ```python
   # Node embeddings learned during training (like word2vec for graphs)
   positions = node_embedding_layer(node_ids)  # [num_agents, embedding_dim]
   ```
   - Pro: Learns topology structure
   - Con: Requires architecture changes, complex

**Recommendation**: Use Node ID as 1D Position for Phase 5D. Add learned embeddings in future phase if needed.

#### Challenge 3: Distance Metric

**Problem**: Euclidean distance meaningless for graph nodes.

**Solution**: Shortest path distance (BFS)

```python
def compute_distance(self, pos1: Tensor, pos2: Tensor) -> Tensor:
    """Compute shortest path distance in graph.

    Args:
        pos1: [num_agents, 1] node IDs
        pos2: [1] target node ID

    Returns:
        [num_agents] shortest path lengths (number of hops)
    """
    # Use precomputed all-pairs shortest paths (Floyd-Warshall or BFS)
    node_ids_1 = pos1[:, 0].long()  # [num_agents]
    node_id_2 = pos2[0].long()  # scalar

    return self.shortest_paths[node_ids_1, node_id_2]
```

**Performance**: Precompute shortest paths at initialization (O(n³) Floyd-Warshall or O(n²log(n)) with BFS from each node).

#### Challenge 4: Movement Semantics

**Problem**: Movement is edge traversal, not delta application.

**Current Architecture**: `apply_movement(positions, deltas)` assumes additive movement.

**Solution**: Override with edge-based movement

```python
def apply_movement(self, positions: Tensor, actions: Tensor) -> Tensor:
    """Move agents along graph edges.

    Args:
        positions: [num_agents, 1] current node IDs
        actions: [num_agents] action indices (which edge to traverse)

    Returns:
        [num_agents, 1] new node IDs after movement
    """
    new_positions = positions.clone()

    for agent_idx in range(len(positions)):
        current_node = positions[agent_idx, 0].item()
        action = actions[agent_idx].item()

        # Get neighbors of current node
        neighbors = self.adjacency_list[current_node]

        # If action is valid edge, move to neighbor
        if action < len(neighbors):
            new_positions[agent_idx, 0] = neighbors[action]
        # Else: INTERACT action or invalid action → stay in place

    return new_positions
```

**Key Difference**: Movement is **action-dependent**, not delta-dependent. This breaks the `apply_movement(positions, deltas)` abstraction.

**Refactoring Required**: Either:
1. Make `deltas` parameter optional (graph substrates ignore it)
2. Encode edge index in delta (e.g., delta = [edge_index])
3. Add new method `apply_graph_movement(positions, actions)`

#### Challenge 5: Affordance Placement

**Problem**: `get_all_positions()` returns list of positions for random placement.

**Solution**: Return list of node IDs

```python
def get_all_positions(self) -> list[list[int]]:
    """Return all graph nodes as positions."""
    return [[node_id] for node_id in range(self.num_nodes)]
```

**Simple** - graphs have finite enumerable positions (node IDs).

#### Challenge 6: Observation Encoding

**Problem**: How to encode node ID in observation space?

**Options**:

1. **Raw Node ID** (simplest)
   ```python
   def encode_observation(self, positions, affordances) -> Tensor:
       return positions.float()  # [num_agents, 1]
   ```
   - Pro: Simple, constant size (1 dim)
   - Con: Assumes ordinal relationship between nodes

2. **One-Hot Encoding**
   ```python
   def encode_observation(self, positions, affordances) -> Tensor:
       num_agents = positions.shape[0]
       one_hot = torch.zeros((num_agents, self.num_nodes), device=positions.device)
       one_hot.scatter_(1, positions.long(), 1.0)
       return one_hot  # [num_agents, num_nodes]
   ```
   - Pro: No ordinal assumptions
   - Con: Observation dims = num_nodes (100 nodes = 100 dims)

3. **Normalized Node ID**
   ```python
   def encode_observation(self, positions, affordances) -> Tensor:
       return positions.float() / max(self.num_nodes - 1, 1)  # [num_agents, 1] in [0,1]
   ```
   - Pro: Normalized [0,1] like other substrates
   - Con: Still assumes ordering

**Recommendation**: Start with Normalized Node ID (consistent with Grid3D/ContinuousND encoding). Add one-hot option in Phase 5C observation encoding retrofit.

---

### Graph Substrate Interface

```python
class GraphSubstrate(SpatialSubstrate):
    """Graph-based substrate with edge traversal movement.

    Positions are node IDs in the graph. Movement is edge traversal.
    Action space varies per node (different nodes have different degrees).

    Key Differences from Grid Substrates:
    - position_dim = 1 (node ID)
    - action_space_size varies per state (max_edges + 1)
    - Movement is action-dependent (not delta-dependent)
    - Distance is shortest path length (not Euclidean)
    """

    position_dim = 1
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
        """
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

    def get_valid_actions(self, position: Tensor) -> list[int]:
        """Get valid action indices for a given position (node).

        This is the key method for action masking.

        Args:
            position: [1] node ID

        Returns:
            List of valid action indices (edge indices + INTERACT)
        """
        node_id = position[0].item()
        neighbors = self.adjacency_list[node_id]

        # Valid actions = edge traversal (0 to num_neighbors-1) + INTERACT
        valid_actions = list(range(len(neighbors))) + [self.max_edges]
        return valid_actions

    def apply_movement(self, positions: Tensor, actions: Tensor) -> Tensor:
        """Move agents along graph edges.

        Args:
            positions: [num_agents, 1] current node IDs
            actions: [num_agents] action indices

        Returns:
            [num_agents, 1] new node IDs
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

    def compute_distance(self, pos1: Tensor, pos2: Tensor) -> Tensor:
        """Compute shortest path distance between nodes."""
        node_ids_1 = pos1[:, 0].long()
        node_id_2 = pos2[0].long()
        return self.shortest_paths[node_ids_1, node_id_2].float()

    def encode_observation(self, positions: Tensor, affordances: dict) -> Tensor:
        """Encode node ID as normalized position [0, 1]."""
        return positions.float() / max(self.num_nodes - 1, 1)

    def get_observation_dim(self) -> int:
        """Graph position encoding is 1-dimensional (normalized node ID)."""
        return 1

    def initialize_positions(self, num_agents: int, device: torch.device) -> Tensor:
        """Randomly place agents at graph nodes."""
        return torch.randint(0, self.num_nodes, (num_agents, 1), device=device)

    def get_all_positions(self) -> list[list[int]]:
        """Return all node IDs as positions."""
        return [[node_id] for node_id in range(self.num_nodes)]

    def get_valid_neighbors(self, position: Tensor) -> list[Tensor]:
        """Return neighbor nodes (for visualization/debugging)."""
        node_id = position[0].item()
        neighbors = self.adjacency_list[node_id]
        return [torch.tensor([n], dtype=torch.long) for n in neighbors]

    def is_on_position(self, agent_positions: Tensor, target_position: Tensor) -> Tensor:
        """Check if agents are at target node (exact match)."""
        return (agent_positions[:, 0] == target_position[0])

    # Helper methods
    def _build_adjacency_list(self, edges, directed) -> dict[int, list[int]]:
        """Build adjacency list from edge list."""
        adj = {i: [] for i in range(self.num_nodes)}
        for u, v in edges:
            adj[u].append(v)
            if not directed:
                adj[v].append(u)
        return adj

    def _compute_shortest_paths(self) -> Tensor:
        """Precompute all-pairs shortest paths using BFS."""
        import torch
        from collections import deque

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

            for node, dist in visited.items():
                shortest_paths[start_node, node] = dist

        return shortest_paths
```

---

### Graph Substrate Configuration

```yaml
# substrate.yaml
type: "grid"
grid:
  topology: "graph"

  # Graph structure
  num_nodes: 16

  # Edge list (undirected by default)
  edges:
    - [0, 1]    # Node 0 connects to Node 1
    - [0, 2]    # Node 0 connects to Node 2
    - [1, 3]
    - [2, 3]
    - [3, 4]
    # ... (define full graph structure)

  directed: false  # Undirected graph (bidirectional edges)

  # Action space sizing
  max_edges: 6  # Maximum edges per node (optional, auto-computed if omitted)

  # Distance metric (only "shortest_path" supported for graphs)
  distance_metric: "shortest_path"

  # Observation encoding
  observation_encoding: "relative"  # or "one_hot" (Phase 5C feature)

# Alternative: Load graph from file
graph:
  topology: "graph"
  graph_file: "data/subway_network.json"  # External graph definition
```

**Graph File Format** (JSON):
```json
{
  "nodes": [
    {"id": 0, "name": "Times Square"},
    {"id": 1, "name": "Grand Central"},
    {"id": 2, "name": "Penn Station"},
    ...
  ],
  "edges": [
    {"from": 0, "to": 1, "weight": 1},
    {"from": 0, "to": 2, "weight": 2},
    ...
  ]
}
```

---

### Action Masking Infrastructure

**Problem**: Existing DQN code doesn't support action masking.

**Where Masking Needed**:

1. **Action Selection** (during training)
   ```python
   # vectorized_env.py:_execute_actions()
   # BEFORE (no masking)
   q_values = self.q_network(observations)  # [num_agents, action_dim]
   actions = q_values.argmax(dim=1)  # Greedy

   # AFTER (with masking)
   q_values = self.q_network(observations)

   # Mask invalid actions (set to -inf)
   for i in range(num_agents):
       valid_actions = self.substrate.get_valid_actions(self.positions[i])
       mask = torch.ones(action_dim, dtype=torch.bool)
       mask[valid_actions] = False
       q_values[i, mask] = -float('inf')

   actions = q_values.argmax(dim=1)  # Now guaranteed to pick valid action
   ```

2. **Epsilon-Greedy Exploration**
   ```python
   # BEFORE (no masking)
   if random.random() < epsilon:
       actions = torch.randint(0, action_dim, (num_agents,))  # Random action

   # AFTER (with masking)
   if random.random() < epsilon:
       # Sample uniformly from valid actions only
       for i in range(num_agents):
           valid_actions = self.substrate.get_valid_actions(self.positions[i])
           actions[i] = random.choice(valid_actions)
   ```

3. **Q-Value Loss Computation** (optional, depends on algorithm)
   - Some algorithms (e.g., PPO) need masked Q-values in loss
   - DQN typically fine without masking in loss (just action selection)

**Implementation Plan**:

1. Add `get_valid_actions(position)` to SpatialSubstrate interface (default: all actions valid)
2. Update vectorized_env.py action selection to use masking when available
3. Add mask support to exploration strategies (epsilon-greedy, RND)
4. Test with Grid2D (should be no-op, all actions always valid)
5. Test with GraphSubstrate (masking critical)

**Estimated Effort**: 4-6 hours (action masking infrastructure is reusable for future substrates)

---

### Graph Substrate Implementation Checklist

**Core Substrate** (8-10 hours):
- [ ] Create `src/townlet/substrate/graph.py`
- [ ] Implement `GraphSubstrate` class with all interface methods
- [ ] Add adjacency list builder
- [ ] Add shortest path precomputation (BFS)
- [ ] Handle directed/undirected graphs
- [ ] Add graph validation (connected components, etc.)

**Action Masking** (4-6 hours):
- [ ] Add `get_valid_actions()` to base interface
- [ ] Update action selection in vectorized_env.py
- [ ] Update epsilon-greedy exploration
- [ ] Add masking tests

**Configuration** (2-3 hours):
- [ ] Extend config schema for graph topology
- [ ] Add edge list parsing
- [ ] Add graph file loader (JSON)
- [ ] Add graph validation in config loader
- [ ] Update SubstrateFactory

**Testing** (4-5 hours):
- [ ] Unit tests for GraphSubstrate methods
- [ ] Test various graph topologies (line, cycle, tree, complete graph)
- [ ] Test directed vs undirected
- [ ] Test action masking correctness
- [ ] Integration test (full training loop)

**Total Estimated Effort**: **18-24 hours**

---

## Phase 5D Topology #2: Hexagonal Grid Substrate

### Overview

**What It Is:**
- 2D grid with **hexagonal tiling** (6 neighbors instead of 4)
- **Uniform distances** to all neighbors (no diagonal ambiguity)
- Uses **axial coordinates** (q, r) or **cube coordinates** (x, y, z where x+y+z=0)

**Why It Matters:**
- Teaches **coordinate system design** (alternatives to Cartesian)
- Teaches **distance metrics** (hex distance vs square grid distance)
- **Better for strategy games** (uniform movement costs)
- **Better for natural terrain** (honeycomb is optimal packing)

**Pedagogical Value**: ✅✅ (High)

### Use Cases

1. **Strategy Games**
   - Wargames (Civilization, Total War style)
   - Board games (Settlers of Catan, etc.)
   - Tactical combat (uniform movement costs)

2. **Natural Simulations**
   - Biological cell structures (honeycomb pattern)
   - Molecular lattices (graphene, crystal structures)
   - Terrain modeling (hex tiles better represent natural features)

3. **Research**
   - Compare square vs hex grid learning
   - Study impact of neighbor topology on exploration

### Coordinate Systems for Hex Grids

#### Option 1: Axial Coordinates (q, r) - **RECOMMENDED**

```
        r →
   ↗ q

     (0,0)  (1,0)  (2,0)
  (-1,1) (0,1)  (1,1)
     (-1,2) (0,2)  (1,2)
```

**Properties:**
- 2D representation (fits existing position tensor shape)
- Simple to understand (two axes)
- Distance: `max(|q1-q2|, |r1-r2|, |q1-q2 + r1-r2|)`

**Neighbors** (6 directions):
```python
HEX_DIRECTIONS = [
    (+1,  0),  # East
    (+1, -1),  # Northeast
    ( 0, -1),  # Northwest
    (-1,  0),  # West
    (-1, +1),  # Southwest
    ( 0, +1),  # Southeast
]
```

#### Option 2: Cube Coordinates (x, y, z) where x+y+z=0

```
        y
       ↗
   x ← • → z

Constraint: x + y + z = 0
```

**Properties:**
- 3D representation with constraint
- More intuitive distance: `(|x1-x2| + |y1-y2| + |z1-z2|) / 2`
- Easier to reason about symmetry

**Conversion to/from Axial:**
```python
# Axial → Cube
x = q
z = r
y = -x - z

# Cube → Axial
q = x
r = z
```

**Recommendation**: Use Axial Coordinates (q, r) internally for efficiency, provide cube coordinate helpers for debugging/visualization.

---

### Hex vs Square Grid Comparison

| Feature | Square Grid | Hex Grid |
|---------|-------------|----------|
| Neighbors | 4 (cardinal) or 8 (with diagonals) | 6 (uniform) |
| Distance to neighbors | 1 (cardinal), √2 (diagonal) | 1 (all neighbors) |
| Movement cost | Uneven (diagonal shortcut) | Uniform (no shortcuts) |
| Coordinate system | Cartesian (x, y) | Axial (q, r) or Cube (x,y,z) |
| Tiling | Simple, aligned | Offset, tricky edges |
| Visualization | Easy (2D array) | Harder (offset rendering) |

**Key Pedagogical Insight**: Hex grids reveal that **coordinate systems are design choices**, not fundamental truths. This teaches students to think about representation.

---

### Hexagonal Grid Substrate Interface

```python
class HexGridSubstrate(SpatialSubstrate):
    """Hexagonal grid substrate with axial coordinates.

    Uses axial coordinate system (q, r) where:
    - q axis points "east"
    - r axis points "southeast"
    - Uniform distance to all 6 neighbors

    Movement actions: 6 directions (E, NE, NW, W, SW, SE) + INTERACT

    Distance metrics:
    - hex_manhattan: max(|q1-q2|, |r1-r2|, |q1-q2 + r1-r2|)  [RECOMMENDED]
    - cube_manhattan: (|x1-x2| + |y1-y2| + |z1-z2|) / 2  [Alternative]
    - euclidean: sqrt((q1-q2)² + (r1-r2)²) [NOT recommended, breaks symmetry]
    """

    position_dim = 2  # Axial coordinates (q, r)
    position_dtype = torch.long

    # Hex direction vectors (axial coordinates)
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
        radius: int,  # Hex grid radius (q ∈ [-radius, radius], r ∈ [-radius, radius])
        boundary: str = "clamp",
        distance_metric: str = "hex_manhattan",
        orientation: str = "flat_top",  # "flat_top" or "pointy_top" (for visualization)
    ):
        """Initialize hexagonal grid substrate.

        Args:
            radius: Hex grid radius (defines grid size)
            boundary: Boundary handling ("clamp", "wrap")
            distance_metric: Distance calculation ("hex_manhattan", "cube_manhattan")
            orientation: Hex tile orientation (affects visualization only)
        """
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

        Hex grid constraint: |q| + |r| ≤ radius (NOT a rectangle!)
        """
        valid = set()
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                # Hex grid constraint (cube coordinate sum)
                if abs(q) + abs(r) + abs(-q - r) <= 2 * self.radius:
                    valid.add((q, r))
        return valid

    def initialize_positions(self, num_agents: int, device: torch.device) -> Tensor:
        """Randomly place agents on valid hex positions."""
        valid_list = list(self.valid_positions)
        indices = torch.randint(0, len(valid_list), (num_agents,))
        positions = torch.tensor([valid_list[i] for i in indices], device=device, dtype=torch.long)
        return positions

    def apply_movement(self, positions: Tensor, deltas: Tensor) -> Tensor:
        """Apply movement with hex-aware boundary handling."""
        new_positions = positions + deltas

        if self.boundary == "clamp":
            # Clamp to valid hex positions
            for i in range(new_positions.shape[0]):
                q, r = new_positions[i, 0].item(), new_positions[i, 1].item()
                if (q, r) not in self.valid_positions:
                    # Stay at current position if out of bounds
                    new_positions[i] = positions[i]

        elif self.boundary == "wrap":
            # Hex wraparound (toroidal hex grid)
            # More complex than square grid, need modular arithmetic with hex constraint
            new_positions = self._wrap_hex_positions(new_positions)

        return new_positions

    def compute_distance(self, pos1: Tensor, pos2: Tensor) -> Tensor:
        """Compute hex distance between positions."""
        if self.distance_metric == "hex_manhattan":
            # Axial hex distance: max(|Δq|, |Δr|, |Δq + Δr|)
            dq = torch.abs(pos1[:, 0] - pos2[0])
            dr = torch.abs(pos1[:, 1] - pos2[1])
            ds = torch.abs(dq + dr)
            return torch.max(torch.max(dq, dr), ds).float()

        elif self.distance_metric == "cube_manhattan":
            # Cube coordinate distance: (|Δx| + |Δy| + |Δz|) / 2
            # Convert axial → cube
            x1, z1, y1 = pos1[:, 0], pos1[:, 1], -pos1[:, 0] - pos1[:, 1]
            x2, z2, y2 = pos2[0], pos2[1], -pos2[0] - pos2[1]
            return (torch.abs(x1 - x2) + torch.abs(y1 - y2) + torch.abs(z1 - z2)).float() / 2

    def encode_observation(self, positions: Tensor, affordances: dict) -> Tensor:
        """Encode hex positions as normalized coordinates.

        Normalize q, r to [0, 1] based on grid radius.
        """
        normalized = positions.float() / (2 * self.radius)  # Range [-radius, radius] → [-0.5, 0.5]
        normalized += 0.5  # Shift to [0, 1]
        return normalized

    def get_observation_dim(self) -> int:
        """Hex position encoding is 2-dimensional (normalized q, r)."""
        return 2

    def get_all_positions(self) -> list[list[int]]:
        """Return all valid hex positions."""
        return [[q, r] for q, r in self.valid_positions]

    def get_valid_neighbors(self, position: Tensor) -> list[Tensor]:
        """Get 6 hex neighbors (or fewer if at boundary)."""
        q, r = position[0].item(), position[1].item()
        neighbors = []

        for dq, dr in self.HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if (nq, nr) in self.valid_positions:
                neighbors.append(torch.tensor([nq, nr], dtype=torch.long))

        return neighbors

    def is_on_position(self, agent_positions: Tensor, target_position: Tensor) -> Tensor:
        """Check if agents are on target hex (exact match)."""
        return (agent_positions == target_position).all(dim=1)

    def _wrap_hex_positions(self, positions: Tensor) -> Tensor:
        """Wrap hex positions for toroidal boundary (complex, defer if needed)."""
        # Hex wraparound is non-trivial - requires careful modular arithmetic
        # For Phase 5D, we can defer this and only support "clamp" boundary
        raise NotImplementedError("Hex wraparound boundary not yet implemented")

    # Visualization helpers (optional, for frontend)
    def to_pixel_coords(self, q: int, r: int, size: float = 1.0) -> tuple[float, float]:
        """Convert axial hex coords to pixel coords for rendering.

        Returns (x, y) pixel coordinates for hex center.
        """
        if self.orientation == "flat_top":
            x = size * (3/2 * q)
            y = size * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
        else:  # "pointy_top"
            x = size * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
            y = size * (3/2 * r)
        return (x, y)
```

---

### Hex Grid Configuration

```yaml
# substrate.yaml
type: "grid"
grid:
  topology: "hexagonal"

  # Hex grid size (radius, not width×height!)
  radius: 5  # Grid contains ~91 hexes (for radius=5)

  # Boundary mode
  boundary: "clamp"  # "wrap" not yet supported for hex

  # Distance metric
  distance_metric: "hex_manhattan"  # or "cube_manhattan"

  # Orientation (affects visualization only)
  orientation: "flat_top"  # or "pointy_top"

  # Observation encoding
  observation_encoding: "relative"
```

**Grid Size Calculation**:
```
Hex grid with radius R contains approximately 3R² + 3R + 1 hexes
- radius=1:  7 hexes
- radius=2: 19 hexes
- radius=3: 37 hexes
- radius=4: 61 hexes
- radius=5: 91 hexes
- radius=8: 217 hexes (roughly equivalent to 8×8=64 square grid in area)
```

---

### Hex Grid Implementation Checklist

**Core Substrate** (4-5 hours):
- [ ] Create `src/townlet/substrate/hexgrid.py`
- [ ] Implement `HexGridSubstrate` with all interface methods
- [ ] Add axial coordinate helpers
- [ ] Add valid position generation (hex grid shape)
- [ ] Implement hex distance metrics
- [ ] Add boundary handling (clamp initially, defer wrap)

**Configuration** (1-2 hours):
- [ ] Extend config schema for hex topology
- [ ] Add radius parameter parsing
- [ ] Add orientation parameter
- [ ] Update SubstrateFactory

**Testing** (2-3 hours):
- [ ] Unit tests for hex distance
- [ ] Test hex neighbors
- [ ] Test position validation
- [ ] Integration test (training on hex grid)

**Documentation** (1 hour):
- [ ] Add hex grid examples to CLAUDE.md
- [ ] Explain axial vs cube coordinates
- [ ] Add visualization guide

**Total Estimated Effort**: **8-10 hours**

---

## Phase 5D Topology #3: 1D Grid Substrate

### Overview

**What It Is:**
- **Linear 1D grid** (positions are scalar integers)
- **2 movement directions** (LEFT/RIGHT) + INTERACT
- **Simplest spatial case** (1 dimension)

**Why It Matters:**
- **Pedagogical progression**: 1D → 2D → 3D teaches dimensionality concepts
- **Edge case testing**: Validates substrate abstraction at N=1
- **Specialized use cases**: Conveyor belts, number lines, sequences

**Pedagogical Value**: ✅ (Medium - mainly for completeness)

### Use Cases

1. **Pedagogical Progression**
   - Start students with 1D (LEFT/RIGHT only)
   - Progress to 2D (adds UP/DOWN)
   - Progress to 3D (adds vertical movement)
   - Teaches dimensionality concepts incrementally

2. **Specialized Scenarios**
   - Conveyor belt simulation (items move left/right)
   - Number line navigation (mathematical reasoning)
   - Sequential decision making (state sequences)

3. **Research**
   - Simplest testbed for RL algorithms
   - Benchmark for transfer learning (1D → 2D → 3D)
   - Control problem baseline

### 1D Grid Substrate Interface

```python
class Grid1DSubstrate(SpatialSubstrate):
    """1D line grid substrate.

    Positions are scalar integers in [0, width).
    Movement actions: LEFT, RIGHT, INTERACT (3 total).

    This is the simplest spatial substrate - essentially a number line.
    """

    position_dim = 1
    position_dtype = torch.long

    def __init__(
        self,
        width: int,
        boundary: str = "clamp",
        distance_metric: str = "manhattan",  # Only manhattan makes sense for 1D
    ):
        """Initialize 1D line grid.

        Args:
            width: Number of positions on line (0 to width-1)
            boundary: Boundary handling ("clamp", "wrap", "bounce")
            distance_metric: Always "manhattan" (only meaningful 1D metric)
        """
        if width <= 0:
            raise ValueError(f"Width must be positive: {width}")
        if boundary not in ("clamp", "wrap", "bounce"):
            raise ValueError(f"Unknown boundary mode: {boundary}")

        self.width = width
        self.boundary = boundary
        self.distance_metric = "manhattan"  # Only option for 1D

    @property
    def action_space_size(self) -> int:
        """2 movement directions + 1 INTERACT = 3 actions."""
        return 3

    def initialize_positions(self, num_agents: int, device: torch.device) -> Tensor:
        """Random positions on 1D line."""
        return torch.randint(0, self.width, (num_agents, 1), device=device)

    def apply_movement(self, positions: Tensor, deltas: Tensor) -> Tensor:
        """Apply movement with 1D boundary handling."""
        new_positions = positions + deltas

        if self.boundary == "clamp":
            new_positions = torch.clamp(new_positions, 0, self.width - 1)
        elif self.boundary == "wrap":
            new_positions = new_positions % self.width
        elif self.boundary == "bounce":
            out_of_bounds = (new_positions < 0) | (new_positions >= self.width)
            new_positions[out_of_bounds] = positions[out_of_bounds]

        return new_positions

    def compute_distance(self, pos1: Tensor, pos2: Tensor) -> Tensor:
        """1D distance is absolute difference."""
        return torch.abs(pos1[:, 0] - pos2[0]).float()

    def encode_observation(self, positions: Tensor, affordances: dict) -> Tensor:
        """Normalize 1D positions to [0, 1]."""
        return positions.float() / max(self.width - 1, 1)

    def get_observation_dim(self) -> int:
        """1D position encoding is 1-dimensional."""
        return 1

    def get_all_positions(self) -> list[list[int]]:
        """Return all positions on 1D line."""
        return [[x] for x in range(self.width)]

    def get_valid_neighbors(self, position: Tensor) -> list[Tensor]:
        """Get left and right neighbors (2 max, fewer at boundaries)."""
        x = position[0].item()
        neighbors = []

        if x > 0:
            neighbors.append(torch.tensor([x - 1], dtype=torch.long))
        if x < self.width - 1:
            neighbors.append(torch.tensor([x + 1], dtype=torch.long))

        return neighbors

    def is_on_position(self, agent_positions: Tensor, target_position: Tensor) -> Tensor:
        """Check if agents are on target position (exact match)."""
        return (agent_positions[:, 0] == target_position[0])
```

---

### 1D Grid Configuration

```yaml
# substrate.yaml
type: "grid"
grid:
  topology: "line"

  # 1D grid size
  width: 20  # Positions 0-19

  # Boundary mode
  boundary: "clamp"  # "wrap" or "bounce" also supported

  # Distance metric (only manhattan for 1D)
  distance_metric: "manhattan"

  # Observation encoding
  observation_encoding: "relative"
```

---

### 1D Grid Implementation Checklist

**Core Substrate** (1-1.5 hours):
- [ ] Create `src/townlet/substrate/grid1d.py`
- [ ] Implement `Grid1DSubstrate` (very similar to Grid2D but simpler)
- [ ] All interface methods

**Configuration** (30 min):
- [ ] Extend config schema for line topology
- [ ] Update SubstrateFactory

**Testing** (1 hour):
- [ ] Unit tests for 1D substrate
- [ ] Integration test

**Documentation** (30 min):
- [ ] Add 1D grid examples

**Total Estimated Effort**: **2-4 hours**

---

## Cross-Cutting Concerns

### Frontend Visualization Challenges

All three new topologies require frontend updates for visualization:

#### Graph Visualization

**Challenge**: How to render arbitrary graph topologies?

**Options**:
1. **Force-directed layout** (Physics-based node positioning)
   - Pro: Automatic, looks good for most graphs
   - Con: Complex to implement, may not converge

2. **Manual layout** (User specifies x,y coords for each node in config)
   - Pro: Simple, predictable
   - Con: Tedious for large graphs

3. **Grid projection** (Force nodes onto 2D grid for rendering)
   - Pro: Simple, fits existing SVG grid renderer
   - Con: May not reflect graph structure

**Recommendation**: Start with Manual Layout (config includes node coordinates). Add automatic layout in future phase.

```yaml
# substrate.yaml
graph:
  nodes:
    - {id: 0, name: "Station A", x: 100, y: 100}
    - {id: 1, name: "Station B", x: 200, y: 150}
    - {id: 2, name: "Station C", x: 150, y: 250}
  edges:
    - {from: 0, to: 1}
    - {from: 1, to: 2}
    - {from: 0, to: 2}
```

#### Hex Visualization

**Challenge**: Rendering hexagonal tiles in SVG.

**Solution**: Convert axial coords to pixel coords, render hex polygons.

```javascript
// Frontend: hexgrid.js
function axialToPixel(q, r, size, orientation="flat_top") {
  if (orientation === "flat_top") {
    const x = size * (3/2 * q);
    const y = size * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
    return {x, y};
  } else {  // "pointy_top"
    const x = size * (Math.sqrt(3) * q + Math.sqrt(3)/2 * r);
    const y = size * (3/2 * r);
    return {x, y};
  }
}

function drawHex(svg, q, r, size, fill) {
  const {x, y} = axialToPixel(q, r, size);
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angle = 2 * Math.PI / 6 * i;
    points.push([
      x + size * Math.cos(angle),
      y + size * Math.sin(angle)
    ]);
  }
  // Draw polygon with points...
}
```

**Recommendation**: Add hex rendering support to frontend (2-3 hours).

#### 1D Visualization

**Challenge**: How to visualize 1D line in 2D space?

**Options**:
1. Horizontal line (x-axis) - **RECOMMENDED**
2. Vertical line (y-axis)
3. Circular arrangement (positions around circle)

**Recommendation**: Render as horizontal line at bottom of screen. Trivial (~30 min).

---

### Testing Strategy

#### Unit Tests (Per Topology)

**Graph Substrate**:
- [ ] Test graph construction (adjacency list)
- [ ] Test shortest path computation (various graph types)
- [ ] Test action masking (different node degrees)
- [ ] Test edge traversal movement
- [ ] Test directed vs undirected
- [ ] Test graph validation (disconnected components, invalid edges)

**Hex Substrate**:
- [ ] Test axial coordinate math
- [ ] Test hex distance metrics
- [ ] Test valid position generation (hex shape)
- [ ] Test 6-neighbor movement
- [ ] Test boundary handling
- [ ] Test cube coordinate conversion

**1D Substrate**:
- [ ] Test 1D movement (left/right)
- [ ] Test boundary modes (clamp/wrap/bounce)
- [ ] Test distance calculation
- [ ] Test neighbor detection (1D edges)

#### Integration Tests (Per Topology)

For each topology:
- [ ] Full training loop (50 episodes)
- [ ] Checkpoint save/load
- [ ] Action selection with new action space
- [ ] Affordance interaction
- [ ] Observation encoding

#### Cross-Topology Tests

- [ ] Config validation for all topologies
- [ ] Substrate factory discrimination (correct class selected)
- [ ] Observation dimension calculation
- [ ] Action space size verification

---

## Implementation Plan Summary

### Phase 5D Task Breakdown

**Task 5D.1: Graph-Based Substrate** (18-24 hours)
- Subtask 1.1: Action Masking Infrastructure (4-6h)
- Subtask 1.2: Core Graph Substrate (8-10h)
- Subtask 1.3: Configuration & Factory (2-3h)
- Subtask 1.4: Testing (4-5h)

**Task 5D.2: Hexagonal Grid Substrate** (8-10 hours)
- Subtask 2.1: Core Hex Substrate (4-5h)
- Subtask 2.2: Configuration & Factory (1-2h)
- Subtask 2.3: Testing (2-3h)
- Subtask 2.4: Documentation (1h)

**Task 5D.3: 1D Grid Substrate** (2-4 hours)
- Subtask 3.1: Core 1D Substrate (1-1.5h)
- Subtask 3.2: Configuration & Factory (0.5h)
- Subtask 3.3: Testing (1h)
- Subtask 3.4: Documentation (0.5h)

**Task 5D.4: Frontend Visualization** (3-4 hours)
- Subtask 4.1: Graph rendering (manual layout) (1-1.5h)
- Subtask 4.2: Hex rendering (SVG polygons) (1.5-2h)
- Subtask 4.3: 1D rendering (horizontal line) (0.5h)

**Task 5D.5: Documentation & Examples** (2-3 hours)
- Subtask 5.1: Update CLAUDE.md (1h)
- Subtask 5.2: Create example configs (1h)
- Subtask 5.3: Comparison documentation (1h)

**Total Estimated Effort: 33-45 hours**

---

## Risk Assessment

### High-Risk Items

1. **Graph Action Masking** (Risk: High)
   - First time implementing variable action spaces
   - Requires changes to core DQN action selection
   - Potential for subtle bugs in masking logic
   - Mitigation: Extensive testing, start with simple graphs

2. **Hex Coordinate Math** (Risk: Medium)
   - Axial coordinate system unfamiliar to most developers
   - Easy to make off-by-one errors in neighbor detection
   - Hex distance metrics are non-intuitive
   - Mitigation: Use well-tested library (e.g., Red Blob Games hex guide)

3. **Frontend Graph Layout** (Risk: Medium)
   - Manual layout tedious for large graphs
   - Automatic layout algorithms complex
   - Graph rendering may not look good
   - Mitigation: Start with manual layout, defer automatic layout

### Medium-Risk Items

4. **Hex Grid Shape Validation** (Risk: Medium)
   - Hex grid is not rectangular - easy to validate wrong positions
   - Boundary handling more complex than square grid
   - Mitigation: Precompute valid positions, thorough testing

5. **Graph Shortest Path Precomputation** (Risk: Low-Medium)
   - O(n³) Floyd-Warshall may be slow for large graphs
   - Need to handle disconnected components
   - Mitigation: Use BFS (O(n²)) instead, warn on large graphs

### Low-Risk Items

6. **1D Substrate** (Risk: Low)
   - Very simple, well-understood
   - Essentially a specialized Grid2D
   - Mitigation: None needed

---

## Dependencies & Blockers

### Prerequisites (Must Be Complete)

1. **Phase 5C Complete**
   - N-dimensional substrates implemented
   - Observation encoding retrofit done
   - `action_space_size` property added to base class

2. **Phase 5B Complete**
   - 3D and Continuous substrates working
   - Configurable action labels implemented
   - Position management fully refactored

### Blocking Issues for Phase 5D

1. **Action Masking Infrastructure Missing**
   - Current DQN code doesn't support action masking
   - Need to add `get_valid_actions()` to base interface
   - Need to update action selection in vectorized_env.py
   - Blocks: Graph substrate (cannot work without masking)

2. **Frontend Substrate Abstraction**
   - Current frontend assumes 2D square grid
   - Need substrate-agnostic rendering interface
   - Blocks: Visualization for all three topologies

3. **Variable Action Space in Q-Network**
   - Current Q-network output size hardcoded (5 for 2D, 7 for 3D)
   - Need dynamic output sizing based on `substrate.action_space_size`
   - Partially blocks: Graph substrate (can work with fixed size + masking)

---

## Recommendations

### Priority Order

**Recommended Implementation Order:**

1. **Start with Hex Grid** (8-10 hours)
   - Fits existing architecture cleanly
   - No action masking needed
   - Validates substrate abstraction with alternative coordinate system
   - Provides immediate pedagogical value

2. **Then 1D Grid** (2-4 hours)
   - Quick win, validates N=1 edge case
   - Tests substrate abstraction at minimum dimensionality
   - Useful for pedagogical progression (1D → 2D → 3D)

3. **Finally Graph** (18-24 hours)
   - Most complex, requires action masking infrastructure
   - Build on lessons learned from Hex and 1D
   - Action masking infrastructure reusable for future substrates

**Total Effort if Done Sequentially**: 28-38 hours

### Alternative: Parallel Development

If multiple developers available:
- Developer A: Hex Grid (8-10h)
- Developer B: Action Masking + Graph (18-24h)
- Developer C: 1D Grid + Frontend (5-8h)

**Total Elapsed Time (Parallel)**: ~18-24 hours (with 3 developers)

### Defer to Future Phases

**Deferred Features:**
- Automatic graph layout algorithms (use manual layout for Phase 5D)
- Hex grid wraparound boundary (only clamp for Phase 5D)
- Weighted graph edges (all edges have weight=1 for Phase 5D)
- Graph editing tools (graphs are static config for Phase 5D)

---

## Success Criteria

### Phase 5D Complete When:

**Graph Substrate:**
- [ ] GraphSubstrate class implemented with all interface methods
- [ ] Action masking working correctly (invalid actions never selected)
- [ ] Shortest path distance computation correct
- [ ] Training runs on 3+ graph topologies (line, cycle, complete graph)
- [ ] Config packs created (e.g., L1_graph_subway)
- [ ] Frontend can render graph (manual layout)

**Hex Substrate:**
- [ ] HexGridSubstrate class implemented
- [ ] Axial coordinate math correct (distance, neighbors)
- [ ] Valid hex position generation correct (hex grid shape)
- [ ] Training runs on hex grid
- [ ] Config pack created (e.g., L1_hex_strategy)
- [ ] Frontend renders hexagons correctly

**1D Substrate:**
- [ ] Grid1DSubstrate class implemented
- [ ] 1D movement working (LEFT/RIGHT)
- [ ] Boundary modes working (clamp/wrap/bounce)
- [ ] Training runs on 1D grid
- [ ] Config pack created (e.g., L0_1D_line)
- [ ] Frontend renders as horizontal line

**Integration:**
- [ ] All unit tests passing (80+ new tests)
- [ ] All integration tests passing (3 new topologies)
- [ ] Documentation complete (CLAUDE.md updated)
- [ ] Comparison examples documented
- [ ] No regressions in existing substrates (Grid2D/3D, Continuous, Aspatial)

---

## Conclusion

Phase 5D adds three powerful new topologies to the configurable spatial substrate system:

1. **Graph-Based** (✅✅) - High complexity, high pedagogical value, requires action masking
2. **Hexagonal Grid** (✅✅) - Medium complexity, high pedagogical value, clean architecture fit
3. **1D Grid** (✅) - Low complexity, medium pedagogical value, edge case validation

**Total Estimated Effort**: 28-38 hours (33-45 hours with frontend and docs)

**Recommended Approach**: Implement sequentially (Hex → 1D → Graph) to build confidence and reuse infrastructure.

**Key Technical Challenges**:
- Action masking infrastructure (Graph)
- Axial coordinate system (Hex)
- Frontend rendering (all three)

**Pedagogical Impact**:
- Graph RL and non-Euclidean reasoning (Graph)
- Coordinate system design (Hex)
- Dimensionality concepts (1D)

This research document provides the foundation for creating a detailed implementation plan (task-002a-phase5d-implementation-plan.md) when Phase 5D is scheduled.

---

**Document Status**: Deep Dive Research Complete
**Next Steps**:
1. Review research findings with team
2. Prioritize topologies (recommend Hex → 1D → Graph)
3. Create detailed implementation plan when ready to proceed
4. Estimate frontend visualization effort more precisely

**Contributors**: Claude (research analysis)
**Review Date**: 2025-11-05
