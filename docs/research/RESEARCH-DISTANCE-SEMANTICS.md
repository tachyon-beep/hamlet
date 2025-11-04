# Research: Distance Semantics Across Spatial Substrates

## Problem Statement

**Question**: How should HAMLET handle "distance" and "adjacency" semantics across different spatial substrates?

**Context**: The current implementation hardcodes Manhattan distance (`torch.abs(pos1 - pos2).sum(dim=1)`) and "exact position match" (`distances == 0`) for interaction checks. This works for 2D square grids but becomes problematic when introducing:

- **3D cubic grids**: 3D Manhattan vs 3D Euclidean
- **Hexagonal grids**: Hexagonal distance (all 6 neighbors equidistant)
- **Toroidal boundaries**: Wraparound means multiple paths
- **Graph substrates**: Graph distance (edge hops) vs spatial distance
- **Aspatial substrates**: Distance is meaningless - everything is "adjacent"

TASK-000 partially addresses this with `substrate.compute_distance()` method, but validation rules and interaction range semantics are unclear.

### Current Usage of Distance in HAMLET

Distance is used in three places:

1. **Interaction range checks** (`vectorized_env.py:274, 462, 541`):

   ```python
   distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
   at_affordance = distances == 0
   ```

   Agent must be at **exact position** (distance == 0) to interact.

2. **Observation encoding** (`observation_builder.py:240`):

   ```python
   distances = torch.abs(positions - affordance_pos).sum(dim=1)
   on_affordance = distances == 0
   ```

   Used to determine "which affordance is agent currently on?"

3. **Action masking** (implicit in boundary checks):
   Movement validity depends on spatial boundaries (clamped, wrap, bounce).

**Key insight**: HAMLET currently requires **exact position match** for interactions, not "proximity within range." This simplifies logic but constrains design space.

---

## Design Space: Four Approaches

### Option A: Substrate-Computed Distance (Current TASK-000 Proposal)

Each substrate implements `compute_distance()` method. Environment calls this method for all distance calculations.

**Implementation**:

```python
# src/townlet/environment/substrate.py
class SpatialSubstrate(ABC):
    @abstractmethod
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute distance between positions."""
        pass

class SquareGridSubstrate(SpatialSubstrate):
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)
        elif self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))
        elif self.distance_metric == "chebyshev":
            return torch.abs(pos1 - pos2).max(dim=-1)[0]

class ToroidalGridSubstrate(SpatialSubstrate):
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Wraparound distance: shortest path considering wrapping
        dx = torch.abs(pos1[:, 0] - pos2[:, 0])
        dy = torch.abs(pos1[:, 1] - pos2[:, 1])
        dx = torch.min(dx, self.width - dx)   # Wrap if shorter
        dy = torch.min(dy, self.height - dy)  # Wrap if shorter
        return dx + dy  # Manhattan with wraparound

class HexagonalGridSubstrate(SpatialSubstrate):
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Hex distance in axial coordinates
        q1, r1 = pos1[:, 0], pos1[:, 1]
        q2, r2 = pos2[:, 0], pos2[:, 1]
        return (torch.abs(q1 - q2) + torch.abs(r1 - r2) + torch.abs(q1 + r1 - q2 - r2)) / 2

class GraphSubstrate(SpatialSubstrate):
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Graph distance = shortest path (expensive!)
        # Simplified: 0 if same, 1 if adjacent, inf otherwise
        same_node = (pos1 == pos2).all(dim=-1)
        adjacent = self.adjacency[pos1[:, 0], pos2[:, 0]]
        return torch.where(same_node, 0.0, torch.where(adjacent, 1.0, float('inf')))

class AspatialSubstrate(SpatialSubstrate):
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # No spatial distance - all agents are "everywhere"
        return torch.zeros(pos1.shape[0])
```

**Usage in environment**:

```python
# vectorized_env.py
distances = self.substrate.compute_distance(self.positions, affordance_pos)
at_affordance = distances == 0  # Still requires exact match
```

**Pros**:

- ✅ Flexible: Each substrate defines its own metric
- ✅ Clean abstraction: Environment doesn't need to know topology details
- ✅ Supports all substrate types (2D, 3D, hex, toroidal, graph)

**Cons**:

- ❌ Still hardcodes "distance == 0" for interactions (what about "within range 2"?)
- ❌ Graph distance is expensive (shortest path computation)
- ❌ Aspatial substrates have meaningless distance (always 0)
- ❌ Doesn't address "interaction range" configuration

**Validation**:

- Can detect unreachable affordances (distance == inf)
- Cannot validate "all affordances within interaction range" (no range concept)

---

### Option B: Configurable Distance Metric (substrate.yaml)

Add `distance_metric` field to substrate config, compute distance accordingly.

**Implementation**:

```yaml
# configs/L1_full_observability/substrate.yaml
substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [8, 8]
    boundary: "clamp"
    distance_metric: "manhattan"  # manhattan, euclidean, chebyshev

# configs/L1_toroidal/substrate.yaml
substrate:
  type: "grid"
  grid:
    topology: "square"
    dimensions: [8, 8]
    boundary: "wrap"
    distance_metric: "toroidal_manhattan"  # Wraparound-aware distance
```

**Pros**:

- ✅ Operator control: Can experiment with different metrics on same topology
- ✅ Config-driven: No code changes needed
- ✅ Clear: Distance metric is explicit in config

**Cons**:

- ❌ Limited flexibility: Only predefined metrics supported
- ❌ Metric must match topology (can't use "toroidal_manhattan" on clamp boundary)
- ❌ Still doesn't address interaction range
- ❌ Aspatial substrates still have meaningless distance

**Validation**:

- Can validate metric matches topology (no "toroidal" metric on clamped grid)

---

### Option C: Affordance-Specific Interaction Range

Each affordance defines `interaction_range` in YAML. Agent can interact if within range.

**Implementation**:

```yaml
# configs/L1_full_observability/affordances.yaml
affordances:
  - name: "Bed"
    interaction_range: 0  # Must be exact position (0 = adjacent only)
    effects:
      energy: 0.2

  - name: "Radio"
    interaction_range: 2  # Can interact from distance 2 (remote control?)
    effects:
      mood: 0.1

  - name: "FamilyTable"
    interaction_range: 1  # Can interact from adjacent cells (reach across table)
    effects:
      social: 0.15
```

**Usage**:

```python
# vectorized_env.py
for affordance_name, affordance_pos in self.affordances.items():
    distances = self.substrate.compute_distance(self.positions, affordance_pos)
    interaction_range = self.affordance_engine.get_interaction_range(affordance_name)
    can_interact = distances <= interaction_range
```

**Pros**:

- ✅ Flexible: Different affordances have different ranges
- ✅ Pedagogical: Teaches "reachability" vs "adjacency" (radio remote control!)
- ✅ Config-driven: No code changes for new affordances
- ✅ Supports "remote" interactions (distance > 0)

**Cons**:

- ❌ More complex: Operator must specify range for every affordance
- ❌ Default range unclear (0? 1? depends on substrate?)
- ❌ Graph substrates: Is range "hops" or spatial distance?
- ❌ Aspatial substrates: Range is meaningless

**Validation**:

- Can validate: No affordances with range > max_distance in substrate
- Cannot validate: "Unreachable" affordances (depends on runtime positions)

---

### Option D: Adjacency-Only (Binary "Adjacent" vs "Not Adjacent")

Replace distance with binary adjacency check: `substrate.is_adjacent(pos1, pos2)`.

**Implementation**:

```python
# src/townlet/environment/substrate.py
class SpatialSubstrate(ABC):
    @abstractmethod
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Return bool tensor: True if pos1 is adjacent to pos2."""
        pass

class SquareGridSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # 4-connected: adjacent if Manhattan distance == 1
        distances = torch.abs(pos1 - pos2).sum(dim=-1)
        return distances <= 1  # 0 (same cell) or 1 (adjacent)

class HexagonalGridSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Hex: All 6 neighbors are adjacent
        hex_distance = (torch.abs(pos1[:, 0] - pos2[:, 0]) +
                       torch.abs(pos1[:, 1] - pos2[:, 1]) +
                       torch.abs(pos1[:, 0] + pos1[:, 1] - pos2[:, 0] - pos2[:, 1])) / 2
        return hex_distance <= 1

class GraphSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Graph: Adjacent if edge exists OR same node
        same_node = (pos1 == pos2).all(dim=-1)
        connected = self.adjacency[pos1[:, 0], pos2[:, 0]]
        return same_node | connected

class AspatialSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Aspatial: Everything is "adjacent" (no positioning)
        return torch.ones(pos1.shape[0], dtype=torch.bool)
```

**Usage**:

```python
# vectorized_env.py
for affordance_name, affordance_pos in self.affordances.items():
    can_interact = self.substrate.is_adjacent(self.positions, affordance_pos)
```

**Pros**:

- ✅ Simple: Binary yes/no, no numeric distance
- ✅ Works for ALL substrates (2D, 3D, hex, graph, aspatial)
- ✅ Clear semantics: "Can I interact with this?" (no ambiguous thresholds)
- ✅ Aspatial substrates: Everything is adjacent (semantically correct!)
- ✅ No interaction range configuration needed (built into substrate)

**Cons**:

- ❌ Less flexible: Cannot have "remote" interactions (distance > 1)
- ❌ Cannot distinguish "on affordance" vs "next to affordance"
- ❌ 8-connected (diagonals) vs 4-connected ambiguity on square grids

**Validation**:

- Can validate: All affordances are "reachable" (some cell is adjacent)
- Simplified: No need to check interaction ranges

---

## Tradeoffs Analysis

| Criterion | Option A: compute_distance() | Option B: Config Metric | Option C: Affordance Range | Option D: is_adjacent() |
|-----------|------------------------------|-------------------------|---------------------------|-------------------------|
| **Flexibility** | High (substrate-specific) | Medium (predefined metrics) | Very High (per-affordance) | Low (binary only) |
| **Performance** | Medium (graph expensive) | Medium | Medium | Fast (no arithmetic) |
| **Clarity** | Medium (what is "distance"?) | High (explicit metric) | Low (range per affordance) | Very High (binary) |
| **Aspatial Support** | Poor (distance = 0 meaningless) | Poor | Poor | Excellent (all adjacent) |
| **Graph Support** | Poor (shortest path expensive) | Poor | Poor | Excellent (edge check) |
| **Validation** | Can detect unreachable | Can validate metric/topology | Can check max range | Can check reachability |
| **Pedagogical** | Medium (teaches metrics) | Medium | High (remote control!) | Low (simplest) |
| **Config Burden** | Low (substrate handles it) | Low | High (range per affordance) | Low (substrate handles it) |

---

## Special Cases Analysis

### Aspatial Substrates

**Option A (compute_distance)**:

```python
def compute_distance(self, pos1, pos2):
    return torch.zeros(pos1.shape[0])  # Always 0 (meaningless)
```

- Distance is meaningless, but "distance == 0" still allows interactions
- ✅ Works but semantically confusing

**Option D (is_adjacent)**:

```python
def is_adjacent(self, pos1, pos2):
    return torch.ones(pos1.shape[0], dtype=torch.bool)  # Always True
```

- Everything is "adjacent" (no positioning)
- ✅ Semantically correct! Aspatial means "everything is accessible"

**Winner**: Option D (is_adjacent) is semantically correct for aspatial.

---

### Toroidal Boundaries

**Option A (compute_distance)**:

```python
# Must compute shortest path considering wraparound
dx = torch.min(torch.abs(pos1[:, 0] - pos2[:, 0]), width - torch.abs(pos1[:, 0] - pos2[:, 0]))
dy = torch.min(torch.abs(pos1[:, 1] - pos2[:, 1]), height - torch.abs(pos1[:, 1] - pos2[:, 1]))
return dx + dy
```

- ✅ Computes correct wraparound distance

**Option D (is_adjacent)**:

```python
# Check if distance <= 1 considering wraparound
distances = toroidal_distance(pos1, pos2)  # Use Option A logic
return distances <= 1
```

- ✅ Still works, just checks adjacency after computing wraparound distance

**Winner**: Both work. Option A is more general (supports range > 1).

---

### Graph Substrates

**Option A (compute_distance)**:

```python
# Shortest path (expensive! Need BFS/Dijkstra)
def compute_distance(self, pos1, pos2):
    # Full shortest-path computation is O(V+E) per query
    return shortest_path_length(pos1, pos2)
```

- ❌ Expensive: Shortest path for every interaction check

**Option D (is_adjacent)**:

```python
# Edge check (cheap! Just adjacency matrix lookup)
def is_adjacent(self, pos1, pos2):
    same_node = (pos1 == pos2).all(dim=-1)
    connected = self.adjacency[pos1[:, 0], pos2[:, 0]]
    return same_node | connected
```

- ✅ Fast: O(1) adjacency matrix lookup

**Winner**: Option D (is_adjacent) is much faster for graphs.

---

### Hexagonal Grids

**Option A (compute_distance)**:

```python
# Hex distance formula
def compute_distance(self, pos1, pos2):
    q1, r1 = pos1[:, 0], pos1[:, 1]
    q2, r2 = pos2[:, 0], pos2[:, 1]
    return (torch.abs(q1 - q2) + torch.abs(r1 - r2) + torch.abs(q1 + r1 - q2 - r2)) / 2
```

- ✅ Correct hex distance

**Option D (is_adjacent)**:

```python
# Check if hex distance <= 1
def is_adjacent(self, pos1, pos2):
    return self.compute_distance(pos1, pos2) <= 1
```

- ✅ Works, delegates to distance computation

**Winner**: Both work. Option A is more general (if we ever want range > 1).

---

## Recommendation: Hybrid Approach (A + D)

**Use both `compute_distance()` and `is_adjacent()` in substrate interface.**

### Rationale

1. **is_adjacent()** is the **primary interface** for interaction checks:
   - Simple, fast, works for all substrates
   - Semantically clear: "Can I interact with this?"
   - Excellent performance for graphs and aspatial substrates

2. **compute_distance()** is **optional** for advanced use cases:
   - Observation encoding (how far is affordance?)
   - Curriculum progression (difficulty based on average distance)
   - Future features: Perception radius, remote interactions

3. **Default semantics**: Adjacency = "can interact"
   - Square grid: Manhattan distance <= 1 (same cell or orthogonally adjacent)
   - Hex grid: Hex distance <= 1 (same cell or any of 6 neighbors)
   - Graph: Same node OR connected by edge
   - Aspatial: Always true (everything accessible)

### Implementation Sketch

```python
# src/townlet/environment/substrate.py

class SpatialSubstrate(ABC):
    """Abstract interface for spatial substrates."""

    @abstractmethod
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Check if positions are adjacent (can interact).

        Returns:
            bool tensor [batch_size]: True if pos1 can interact with pos2
        """
        pass

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute distance between positions (optional, for advanced use).

        Default implementation: adjacency-based (0 if adjacent, inf otherwise).
        Substrates can override for continuous distance metrics.

        Returns:
            float tensor [batch_size]: Distance from pos1 to pos2
        """
        adjacent = self.is_adjacent(pos1, pos2)
        return torch.where(adjacent, 0.0, float('inf'))


class SquareGridSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # 4-connected: same cell OR orthogonally adjacent
        manhattan = torch.abs(pos1 - pos2).sum(dim=-1)
        return manhattan <= 1

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Override default: provide continuous distance
        if self.distance_metric == "manhattan":
            return torch.abs(pos1 - pos2).sum(dim=-1)
        elif self.distance_metric == "euclidean":
            return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))


class ToroidalGridSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Adjacency with wraparound
        dx = torch.abs(pos1[:, 0] - pos2[:, 0])
        dy = torch.abs(pos1[:, 1] - pos2[:, 1])
        dx = torch.min(dx, self.width - dx)
        dy = torch.min(dy, self.height - dy)
        toroidal_manhattan = dx + dy
        return toroidal_manhattan <= 1

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Wraparound distance
        dx = torch.abs(pos1[:, 0] - pos2[:, 0])
        dy = torch.abs(pos1[:, 1] - pos2[:, 1])
        dx = torch.min(dx, self.width - dx)
        dy = torch.min(dy, self.height - dy)
        return dx + dy


class HexagonalGridSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Hex distance <= 1 (same cell or any of 6 neighbors)
        q1, r1 = pos1[:, 0], pos1[:, 1]
        q2, r2 = pos2[:, 0], pos2[:, 1]
        hex_dist = (torch.abs(q1 - q2) + torch.abs(r1 - r2) + torch.abs(q1 + r1 - q2 - r2)) / 2
        return hex_dist <= 1

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Full hex distance
        q1, r1 = pos1[:, 0], pos1[:, 1]
        q2, r2 = pos2[:, 0], pos2[:, 1]
        return (torch.abs(q1 - q2) + torch.abs(r1 - r2) + torch.abs(q1 + r1 - q2 - r2)) / 2


class GraphSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Same node OR connected by edge
        same_node = (pos1 == pos2).all(dim=-1)
        connected = self.adjacency[pos1[:, 0], pos2[:, 0]]
        return same_node | connected

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Simplified: 0 if same, 1 if adjacent, inf otherwise
        # Full shortest-path is too expensive for real-time use
        adjacent = self.is_adjacent(pos1, pos2)
        same = (pos1 == pos2).all(dim=-1)
        return torch.where(same, 0.0, torch.where(adjacent, 1.0, float('inf')))


class AspatialSubstrate(SpatialSubstrate):
    def is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # Everything is "adjacent" (no positioning)
        return torch.ones(pos1.shape[0], dtype=torch.bool, device=pos1.device)

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        # No spatial distance
        return torch.zeros(pos1.shape[0], device=pos1.device)
```

### Usage in Environment

```python
# vectorized_env.py (interaction checks)

def get_action_masks(self):
    # Mask INTERACT action - only valid when adjacent to affordance
    can_interact = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)

    for affordance_name, affordance_pos in self.affordances.items():
        # Check adjacency (substrate-aware)
        adjacent = self.substrate.is_adjacent(self.positions, affordance_pos)

        # Check operating hours (temporal mechanics)
        if self.enable_temporal_mechanics:
            if not self.affordance_engine.is_affordance_open(affordance_name, self.time_of_day):
                adjacent = torch.zeros_like(adjacent)  # Closed = not adjacent

        can_interact |= adjacent

    action_masks[:, 4] = can_interact  # INTERACT action
```

```python
# observation_builder.py (affordance encoding)

def _build_affordance_encoding(self, positions, affordances):
    # Which affordance is agent on/adjacent to?
    affordance_encoding = torch.zeros(self.num_agents, self.num_affordance_types + 1)
    affordance_encoding[:, -1] = 1.0  # Default: "none"

    for affordance_idx, affordance_name in enumerate(self.affordance_names):
        if affordance_name in affordances:
            affordance_pos = affordances[affordance_name]
            adjacent = self.substrate.is_adjacent(positions, affordance_pos)
            if adjacent.any():
                affordance_encoding[adjacent, -1] = 0.0  # Clear "none"
                affordance_encoding[adjacent, affordance_idx] = 1.0

    return affordance_encoding
```

---

## Validation Rules

**Universe compiler validates**:

1. **Substrate consistency**:
   - `boundary: wrap` → Use `ToroidalGridSubstrate` (wraparound adjacency)
   - `topology: hexagonal` → Use `HexagonalGridSubstrate` (hex adjacency)

2. **Affordance reachability**:
   - Warning: "Affordance 'Radio' at position (7, 7) may be unreachable from spawn points"
   - Check: Simulate random walk, ensure all affordances are visited

3. **Action compatibility**:
   - Square grid: 4 movement actions (UP, DOWN, LEFT, RIGHT)
   - Hex grid: 6 movement actions (E, NE, NW, W, SW, SE)
   - Aspatial: 0 movement actions (INTERACT, WAIT only)

**Example validation error**:

```
❌ SUBSTRATE COMPILATION FAILED
Substrate type 'aspatial' has no positioning, but actions.yaml defines movement actions.

Aspatial substrates require action_type: "state_transition" (no movement).
Remove UP, DOWN, LEFT, RIGHT actions or switch substrate to 'grid'.
```

---

## Future Extension: Affordance-Specific Ranges (Optional)

If pedagogical value emerges (e.g., "radio remote control"), add `interaction_range` to affordances:

```yaml
# affordances.yaml
affordances:
  - name: "Radio"
    interaction_range: 2  # Can interact from distance <= 2
    effects:
      mood: 0.1

  - name: "Bed"
    # interaction_range: 0 (default, must be adjacent)
    effects:
      energy: 0.2
```

**Implementation**:

```python
# affordance_engine.py
def get_interaction_range(self, affordance_name: str) -> int:
    affordance = self.config.get_affordance(affordance_name)
    return affordance.interaction_range if affordance.interaction_range else 0  # Default: adjacent only

# vectorized_env.py
for affordance_name, affordance_pos in self.affordances.items():
    interaction_range = self.affordance_engine.get_interaction_range(affordance_name)

    if interaction_range == 0:
        # Use adjacency check (fast, substrate-aware)
        can_interact = self.substrate.is_adjacent(self.positions, affordance_pos)
    else:
        # Use distance check (slower, requires compute_distance)
        distances = self.substrate.compute_distance(self.positions, affordance_pos)
        can_interact = distances <= interaction_range
```

**This is OPTIONAL** - default behavior is adjacency-only (interaction_range = 0).

---

## Priority and Estimated Effort

**Priority**: **High** - Blocks TASK-000 (Spatial Substrates)

**Estimated Effort**:

1. **Phase 1: Add is_adjacent() to substrate interface** (2-3 hours)
   - Define abstract method
   - Implement for SquareGridSubstrate
   - Update vectorized_env.py and observation_builder.py to use is_adjacent()
   - Add tests

2. **Phase 2: Implement for all substrate types** (4-6 hours)
   - ToroidalGridSubstrate (wraparound adjacency)
   - CubicGridSubstrate (3D adjacency)
   - HexagonalGridSubstrate (hex adjacency)
   - GraphSubstrate (edge check)
   - AspatialSubstrate (always true)

3. **Phase 3: Add compute_distance() optional override** (2-3 hours)
   - Default implementation (0 if adjacent, inf otherwise)
   - Override in substrates that support continuous distance
   - Document when to use is_adjacent() vs compute_distance()

**Total**: 8-12 hours

---

## Conclusion

**Recommendation**: Implement **Hybrid Approach (A + D)** with `is_adjacent()` as primary interface and `compute_distance()` as optional override.

**Key Principles**:

1. **is_adjacent()** is the primary interaction check:
   - Simple, fast, clear semantics
   - Works for ALL substrates (2D, 3D, hex, graph, aspatial)
   - Substrate defines what "adjacent" means for its topology

2. **compute_distance()** is optional for advanced use:
   - Default: 0 if adjacent, inf otherwise (discrete)
   - Override: Continuous distance for substrates that support it
   - Used for: Observation encoding, curriculum metrics, future features

3. **Adjacency semantics by substrate**:
   - Square grid: Manhattan distance <= 1 (4-connected)
   - Hex grid: Hex distance <= 1 (6 neighbors equidistant)
   - Toroidal: Wraparound-aware adjacency
   - Graph: Same node OR edge connection
   - Aspatial: Everything is adjacent (no positioning)

4. **Future extension**: Add `interaction_range` to affordances if pedagogical value emerges (radio remote control, etc.)

**This design**:

- ✅ Works for all substrate types
- ✅ Clear semantics (adjacency = can interact)
- ✅ Fast (O(1) for graphs, cheap for grids)
- ✅ Extensible (can add ranges later)
- ✅ Semantically correct for aspatial substrates

**Next steps**: Implement in TASK-000 Phase 1 (Substrate Abstraction Layer).
