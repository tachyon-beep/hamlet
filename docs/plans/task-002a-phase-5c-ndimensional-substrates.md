# TASK-002A Phase 5C: N-Dimensional Substrates - Design Document (v4)

**Date**: 2025-11-05 (Rebranded as Phase 5C)
**Version**: 4.0
**Status**: Design Complete (Scheduled After Phase 5B)
**Dependencies**:
- Phase 5B Complete (3D, Continuous, Configurable Actions)
**Type**: Research Infrastructure Enhancement
**Scope**: Configurable Observation Encoding (All Dimensions) + N-Dimensional Substrates
**Estimated Effort**: TBD (will be determined during implementation planning)

**Revision History:**
- **v4 (2025-11-05)**: Rebranded as Phase 5C (scheduled after Phase 5B)
  - Changed from "Phase X" (unscheduled future work) to "Phase 5C" (next phase)
  - Updated all references throughout document
  - Status changed to "Scheduled After Phase 5B"
- **v3 (2025-11-05)**: Peer review fixes (against Phase 5A/5B)
  - CRITICAL: Fixed `encode_observation()` signatures - added `affordances` parameter to helper methods
  - CRITICAL: Corrected Grid3D documentation - acknowledged Phase 5B already uses normalized encoding
  - MAJOR: Merged observation encoding retrofit into Phase 5C scope (two-part implementation)
    - Part 1: Retrofit observation_encoding to existing substrates (Grid2D/3D, Continuous1D/2D/3D)
    - Part 2: Add N-dimensional substrates (GridND, ContinuousND)
  - MAJOR: Added `action_space_size` property to base class (SpatialSubstrate)
  - Updated executive summary to reflect unified "Configurable Observation Encoding" scope
  - Renamed file from `phase-5c` to `phase-X` for consistency
- **v2 (2025-11-05)**: Code review fixes
  - Removed 10D grid limit (now uncapped with warnings at N≥10, N≥20)
  - Added `action_space_size` property to GridND and ContinuousND
  - Added `get_valid_neighbors()` method to GridND
  - Added `supports_enumerable_positions()` to GridND
  - Added Phase 5C prerequisite (retrofit observation_encoding)
  - Added warning for ContinuousND at N≥100
- **v1 (2025-11-05)**: Initial design from brainstorming session

---

## Executive Summary

This document describes **Phase 5C: Configurable Observation Encoding & N-Dimensional Substrates**. This phase has two complementary goals:

1. **Make observation encoding configurable** across all substrates (1D through ND)
2. **Add N-dimensional substrates** (4D+) to support abstract state space research

**Phase 5C Scope:**

**Part 1: Retrofit Observation Encoding** (to existing Phase 5A/5B substrates)
- Add configurable `observation_encoding` parameter to Grid2D, Grid3D, Continuous1D/2D/3D
- Three options: `relative` (normalized), `scaled` (normalized + range info), `absolute` (raw)
- Default to `relative` (preserves current behavior, backward compatible)
- Enables comparative studies across dimensions

**Part 2: N-Dimensional Substrates** (new GridND, ContinuousND classes)
- **Grid substrates**: 4D to arbitrary dimensions (uncapped with warnings at N≥10)
- **Continuous substrates**: 4D to arbitrary dimensions (uncapped for research flexibility)
- **Auto-generated action spaces**: 2N+1 discrete actions (±movement per dimension + INTERACT)
- **Standard terminology where it exists**: 1D/2D/3D use specialized classes, 4D+ uses generic ND classes
- **Dynamic action space sizing**: New `action_space_size` property enables variable action spaces
- **Configurable observation encoding**: Built-in from day one (consistent with retrofitted substrates)

**Design Philosophy:**
- **Build on Phase 5B discovery**: Phase 5B found normalized encoding works well for Grid3D; Phase 5C extends this
- **"Sandcastle simplicity, rocket science available"**: Common cases (2D/3D) remain simple; complex cases (7D) don't impose complexity tax
- **User-defined limits with warnings**: No hard caps, emit warnings at N≥10 (Grid) and N≥100 (Continuous) to guide users
- **Explicit configuration over implicit magic**: All dimensionality explicit in config, no auto-detection
- **Clean separation of concerns**: Substrates handle position state, user-defined actions handle movement semantics

---

## Design Decisions

### 1. Purpose & Research Use Cases

**Primary Goal**: Support abstract state space navigation for RL research.

**Key Use Cases:**
- **Abstract State Space Navigation**: Teach that RL works in ANY state space, not just physical grids
  - Example: 5D space [temperature, pressure, pH, time, cost]
  - Affordances are "stable regions" in parameter space
- **Multi-Objective Optimization**: Each dimension is an objective to optimize
  - Example: 4D [health, wealth, happiness, reputation]
  - Study Pareto frontiers, trade-offs
- **Hyperparameter Search**: Dimensions are model hyperparameters
  - Example: 6D [learning_rate, batch_size, hidden_dim, dropout, temperature, momentum]
  - Agent learns to navigate hyperparameter space (AutoML)
- **Generalization Testing**: Train on 2D/3D, test transfer to 4D+
  - Does the network truly learn spatial reasoning or just memorize?

**Not Pedagogy-Only**: HAMLET is evolving into a formal experimentation platform with "universe as code" and "brain as code" features. N-dimensional substrates are research infrastructure.

---

### 2. Dimensionality Boundaries

**Grid Substrates (Discrete Lattice):**
- **Range**: 4 ≤ N < ∞ (no hard upper bound)
- **Rationale**: Discrete action space scales linearly as 2N+1
  - 10D = 20 movement actions + 1 interact = 21 actions (manageable)
  - 20D = 40 movement actions + 1 interact = 41 actions (still reasonable for deep RL)
  - 50D = 100 movement actions + 1 interact = 101 actions (challenging but possible)
- **Warning System**: Emit warnings for high dimensions to guide users
  - Warning at N ≥ 10: "GridND with N≥10 creates 2N+1 actions. Consider continuous substrates if action space becomes unwieldy."
  - Warning at N ≥ 20: "GridND with N≥20 creates very large action spaces (41+ actions). Ensure your network architecture and training approach can handle this."
- **User-Defined Limits**: Let researchers discover their own practical limits (same philosophy as continuous)
  - If N=100 causes training issues, they'll find out and can request guidance
  - Can add hard caps later based on empirical experience if needed

**Continuous Substrates (Smooth Space):**
- **Range**: 4 ≤ N < ∞ (no hard upper bound)
- **Rationale**:
  - No combinatorial position explosion (infinite positions anyway)
  - Proximity detection vs exact matching (more forgiving)
  - Discrete actions scale better here (no grid cell enumeration)
- **Practical Limits**: Let researchers discover their own limits
  - If N=500 causes memory issues, they'll find out and report
  - Can add guidance/warnings later based on empirical experience

**Why 4D Minimum for Both:**
- 1D/2D/3D have **specialized classes** with standard terminology (Surge/Sway/Heave for 3D)
- 4D+ uses **generic ND classes** with anonymous dimensions (DIM_0, DIM_1, ...)
- Clean separation: spatial (1-3D) vs abstract state spaces (4D+)

---

### 3. Dimension Naming & Semantics

**Standard Terminology Where It Exists:**
- **1D**: X-axis (position)
- **2D**: X (horizontal), Y (vertical)
- **3D**: 6-DoF terminology (Sway/Heave/Surge) from Phase 5B.3
  - X = Sway (lateral), Y = Heave (vertical), Z = Surge (longitudinal)

**Anonymous for 4D+:**
- Dimensions labeled as DIM_0, DIM_1, DIM_2, ..., DIM_N-1
- Rationale: No universal "standard" for 7D space semantics
- User adds semantic meaning via action_labels (from Phase 5B.3)

**Example: 7D Abstract State Space**
```yaml
# substrate.yaml
type: "grid"
grid:
  topology: "hypercube"
  dimensions: 7
  bounds: [[0,100], [0,50], [0,14], [0,24], [0,10], [0,5], [0,3]]

# System generates: DIM_0_NEGATIVE/POSITIVE, DIM_1_NEGATIVE/POSITIVE, etc.

# action_labels.yaml (from Phase 5B.3)
action_labels:
  DIM_0_POSITIVE: "TEMPERATURE_UP"
  DIM_0_NEGATIVE: "TEMPERATURE_DOWN"
  DIM_1_POSITIVE: "PRESSURE_INCREASE"
  DIM_1_NEGATIVE: "PRESSURE_DECREASE"
  # ... user defines semantics for each dimension
```

---

### 4. Observation Encoding Strategies

**Context from Phase 5B**: Grid3D already uses normalized coordinate encoding (3 dimensions regardless of grid size), not one-hot encoding. This avoided the explosion of observation dimensions (8×8×3 = 192 cells would require 192 dims for one-hot). Phase 5B discovered that normalized encoding scales much better.

**Phase 5C Extension**: Phase 5C generalizes this approach to N dimensions and makes it configurable.

**Three Encoding Options** (configurable per substrate):

#### A) `relative` - Position as Fraction of Bounds (CURRENT PHASE 5B APPROACH)
```python
# Output: [num_agents, N] floats in [0,1]^N
# Example 7D: [0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2]
```
- Network learns: "I'm 50% across dimension 0, 30% across dimension 1..."
- **Use when**: Spatial relationships matter, absolute scale doesn't
- **Observation size**: N dimensions

#### B) `scaled` - Position + Range Metadata
```python
# Output: [num_agents, 2N] floats
# Example 7D: [0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2,  # normalized positions
#              100, 50, 14, 24, 10, 5, 3]           # range sizes
```
- Network learns: "I'm 50% across a 100-unit range, 30% across a 50-unit range..."
- **Use when**: Scale matters semantically (temperature vs pH have different meanings)
- **Observation size**: 2N dimensions

#### C) `absolute` - Raw Unnormalized Coordinates
```python
# Output: [num_agents, N] floats (raw values)
# Example 7D: [50.0, 15.0, 1.4, 19.2, 6.0, 2.0, 0.6]
```
- Network learns: "I'm at absolute position [50, 15, 1.4, ...]"
- **Use when**: Absolute values have semantic meaning
- **Observation size**: N dimensions

**Configuration:**
```yaml
observation_encoding: "relative"  # or "scaled" or "absolute"
```

**Design Rationale:**
- `relative` is standard RL (normalized observations)
- `scaled` preserves metric structure for heterogeneous dimensions
- `absolute` for domains where raw values matter
- Configurable → researchers can run ablation studies

**What's Deferred:**
- Per-dimension step sizes (handled by user-defined actions)
- Affordance proximity encoding (add later if requested)
- Dimension-specific metadata (units, types, etc.)

---

### 5. Action Space Design

**Auto-Generated Discrete Actions:**

For N-dimensional substrate, system generates **2N + 1 canonical actions**:

```
Action Index | Canonical Name     | Delta Vector
-------------|-------------------|---------------------------
0            | DIM_0_NEGATIVE    | [-1, 0, 0, ..., 0]
1            | DIM_0_POSITIVE    | [+1, 0, 0, ..., 0]
2            | DIM_1_NEGATIVE    | [0, -1, 0, ..., 0]
3            | DIM_1_POSITIVE    | [0, +1, 0, ..., 0]
...          | ...               | ...
2N-2         | DIM_(N-1)_NEGATIVE| [0, 0, 0, ..., -1]
2N-1         | DIM_(N-1)_POSITIVE| [0, 0, 0, ..., +1]
2N           | INTERACT          | [0, 0, 0, ..., 0]
```

**Example: 7D Space**
- 14 movement actions (2 per dimension)
- 1 interact action
- Total: 15 discrete actions
- Q-network output shape: `[batch_size, 15]`

**Action-to-Delta Mapping:**
```python
def _action_to_deltas(self, actions: torch.Tensor) -> torch.Tensor:
    """Map action indices to N-dimensional movement deltas."""
    num_agents = actions.shape[0]
    N = self.substrate.position_dim

    deltas = torch.zeros((num_agents, N), dtype=torch.float32, device=self.device)

    # For each dimension, handle NEGATIVE and POSITIVE actions
    for dim in range(N):
        negative_action = 2 * dim
        positive_action = 2 * dim + 1

        deltas[actions == negative_action, dim] = -1.0
        deltas[actions == positive_action, dim] = +1.0

    # INTERACT action (2N) = no movement (already zeros)

    # Apply movement_delta scaling for continuous substrates
    if hasattr(self.substrate, 'movement_delta'):
        deltas *= self.substrate.movement_delta

    return deltas
```

**Encoding Pattern**: Dimension-major ordering (all actions for DIM_0, then all for DIM_1, etc.)

**Integration with Phase 5B.3 Action Labels:**
- System generates canonical actions (DIM_0_POSITIVE, etc.)
- User optionally defines semantic labels via `action_labels` config
- Example: `DIM_0_POSITIVE: "TEMPERATURE_UP"`

**Future: User-Defined Actions (Separate Task)**
- Users can define custom step sizes per action
- Users can define complex movements (teleportation, diagonal, multi-dim)
- Users can define semantic operations (HEAT moves temperature by +5)
- Substrate just executes `apply_movement(positions, deltas)` regardless

---

## Configuration Schema

### Grid N-Dimensional Substrate

```yaml
# substrate.yaml
type: "grid"

grid:
  topology: "hypercube"  # N-dimensional generalization of cube
  dimensions: 7          # Integer, N ≥ 4 (warnings at N≥10, N≥20)

  bounds:                # List of [min, max] per dimension
    - [0, 100]           # DIM_0: integer range [0, 100]
    - [0, 50]            # DIM_1: integer range [0, 50]
    - [0, 14]            # DIM_2: integer range [0, 14]
    - [0, 24]            # DIM_3: integer range [0, 24]
    - [0, 10]            # DIM_4: integer range [0, 10]
    - [0, 5]             # DIM_5: integer range [0, 5]
    - [0, 3]             # DIM_6: integer range [0, 3]

  boundary: "clamp"      # clamp | wrap | bounce | sticky
  distance_metric: "euclidean"  # manhattan | euclidean | chebyshev
  observation_encoding: "scaled"  # relative | scaled | absolute

# Validation:
# - len(bounds) must equal dimensions
# - 4 ≤ dimensions ≤ 10 (hard error otherwise)
# - All bounds must be integer ranges [min, max] where min < max
```

### Continuous N-Dimensional Substrate

```yaml
# substrate.yaml
type: "continuous"

continuous:
  dimensions: 7          # Integer, N ≥ 4 (no upper bound)

  bounds:                # List of [min, max] per dimension
    - [0.0, 100.0]       # DIM_0: float range [0.0, 100.0]
    - [0.0, 50.0]        # DIM_1: float range [0.0, 50.0]
    - [0.0, 14.0]        # DIM_2: float range [0.0, 14.0]
    - [0.0, 24.0]        # DIM_3: float range [0.0, 24.0]
    - [0.0, 10.0]        # DIM_4: float range [0.0, 10.0]
    - [0.0, 5.0]         # DIM_5: float range [0.0, 5.0]
    - [0.0, 3.0]         # DIM_6: float range [0.0, 3.0]

  boundary: "clamp"      # clamp | wrap | bounce | sticky
  movement_delta: 0.5    # Distance discrete actions move agent
  interaction_radius: 2.0  # Distance threshold for affordance interaction
  distance_metric: "euclidean"  # euclidean | manhattan
  observation_encoding: "scaled"  # relative | scaled | absolute

# Validation:
# - len(bounds) must equal dimensions
# - dimensions ≥ 4 (no upper hard limit)
# - All bounds must be float ranges [min, max] where min < max
# - Range size per dimension must be ≥ interaction_radius
```

### Backward Compatibility

**Existing 1D/2D/3D configs remain unchanged:**

```yaml
# 2D grid (existing Phase 5 format)
type: "grid"
grid:
  topology: "square"     # NOT "hypercube"
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "manhattan"

# 3D grid (Phase 5B format)
type: "grid"
grid:
  topology: "cubic"      # NOT "hypercube"
  width: 8
  height: 8
  depth: 3
  boundary: "clamp"
  distance_metric: "manhattan"
```

**Topology discrimination:**
- `square` → Grid2DSubstrate (specialized)
- `cubic` → Grid3DSubstrate (specialized)
- `hypercube` → GridNDSubstrate (generic N-dimensional)

---

## Technical Architecture

### Type Hierarchy

```python
# Base class (existing)
class SpatialSubstrate(ABC):
    position_dim: int         # Number of dimensions
    position_dtype: torch.dtype  # long for grid, float32 for continuous

    @abstractmethod
    def encode_observation(self, positions, affordances) -> torch.Tensor: ...
    @abstractmethod
    def apply_movement(self, positions, deltas) -> torch.Tensor: ...
    @abstractmethod
    def compute_distance(self, pos1, pos2) -> torch.Tensor: ...

# Specialized classes (existing, 1-3D)
class Grid2DSubstrate(SpatialSubstrate):
    position_dim = 2
    position_dtype = torch.long

class Grid3DSubstrate(SpatialSubstrate):
    position_dim = 3
    position_dtype = torch.long

class Continuous2DSubstrate(SpatialSubstrate):
    position_dim = 2
    position_dtype = torch.float32

# NEW: Generic N-dimensional classes (4D+)
class GridNDSubstrate(SpatialSubstrate):
    """N-dimensional hypercube grid (N ≥ 4, uncapped)."""
    position_dtype = torch.long

    def __init__(self, dimensions: int, bounds: list[tuple[int, int]],
                 boundary: str, distance_metric: str, observation_encoding: str):
        import warnings

        if dimensions < 4:
            raise ValueError(
                f"GridNDSubstrate requires dimensions ≥ 4, got {dimensions}. "
                f"Use Grid2D or Grid3D for lower dimensions."
            )

        # Emit warnings for high-dimensional spaces
        if dimensions >= 20:
            warnings.warn(
                f"GridND with N={dimensions} creates {2*dimensions+1} discrete actions. "
                f"Ensure your network architecture and training approach can handle this.",
                UserWarning
            )
        elif dimensions >= 10:
            warnings.warn(
                f"GridND with N={dimensions} creates {2*dimensions+1} discrete actions. "
                f"Consider continuous substrates if action space becomes unwieldy.",
                UserWarning
            )

        self.position_dim = dimensions
        self.bounds = bounds
        self.boundary = boundary
        self.distance_metric = distance_metric
        self.observation_encoding = observation_encoding

    @property
    def action_space_size(self) -> int:
        """Return number of discrete actions: 2N + 1 (±movement per dim + interact)."""
        return 2 * self.position_dim + 1

class ContinuousNDSubstrate(SpatialSubstrate):
    """N-dimensional continuous space (N ≥ 4, uncapped)."""
    position_dtype = torch.float32

    def __init__(self, dimensions: int, bounds: list[tuple[float, float]],
                 boundary: str, movement_delta: float, interaction_radius: float,
                 distance_metric: str, observation_encoding: str):
        import warnings

        if dimensions < 4:
            raise ValueError(
                f"ContinuousNDSubstrate requires dimensions ≥ 4, got {dimensions}. "
                f"Use Continuous1D/2D/3DSubstrate for lower dimensions."
            )

        # Optional warning for very high dimensions (no hard limit)
        if dimensions >= 100:
            warnings.warn(
                f"ContinuousND with N={dimensions} has very high dimensionality. "
                f"Monitor memory usage and training performance.",
                UserWarning
            )

        self.position_dim = dimensions
        self.bounds = bounds
        self.boundary = boundary
        self.movement_delta = movement_delta
        self.interaction_radius = interaction_radius
        self.distance_metric = distance_metric
        self.observation_encoding = observation_encoding

    @property
    def action_space_size(self) -> int:
        """Return number of discrete actions: 2N + 1 (±movement per dim + interact)."""
        return 2 * self.position_dim + 1
```

**Design Rationale:**
- 1D/2D/3D remain specialized (don't pay complexity tax for common cases)
- 4D+ uses generic ND classes (research-focused, complexity expected)
- Clean boundary at N=3/4 (spatial vs abstract)

---

## Implementation Details

### 1. Observation Encoding

**Three Encoding Methods:**

```python
class GridNDSubstrate(SpatialSubstrate):

    def encode_observation(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode positions according to observation_encoding config.

        Args:
            positions: Agent positions [num_agents, N]
            affordances: Dict of affordance positions (reserved for future proximity encoding)
        """
        if self.observation_encoding == "relative":
            return self._encode_relative(positions, affordances)
        elif self.observation_encoding == "scaled":
            return self._encode_scaled(positions, affordances)
        elif self.observation_encoding == "absolute":
            return self._encode_absolute(positions, affordances)

    def _encode_relative(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Normalize positions to [0, 1] per dimension.

        Args:
            positions: Agent positions [num_agents, N]
            affordances: Dict of affordance positions (currently unused, reserved for future)

        Returns: [num_agents, N] floats
        """
        # affordances parameter currently unused, reserved for future proximity encoding
        normalized = torch.zeros_like(positions, dtype=torch.float32)
        for dim in range(self.position_dim):
            min_val, max_val = self.bounds[dim]
            range_size = max(max_val - min_val, 1)
            normalized[:, dim] = positions[:, dim].float() / range_size
        return normalized

    def _encode_scaled(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Normalize + append range metadata.

        Args:
            positions: Agent positions [num_agents, N]
            affordances: Dict of affordance positions (currently unused, reserved for future)

        Returns: [num_agents, 2N] floats
        First N dims: normalized positions [0,1]^N
        Last N dims: range sizes for each dimension
        """
        # affordances parameter currently unused, reserved for future proximity encoding
        relative = self._encode_relative(positions, affordances)

        # Add range information for each dimension
        num_agents = positions.shape[0]
        ranges = torch.tensor(
            [max_val - min_val for min_val, max_val in self.bounds],
            dtype=torch.float32,
            device=positions.device
        ).unsqueeze(0).expand(num_agents, -1)

        return torch.cat([relative, ranges], dim=1)

    def _encode_absolute(
        self,
        positions: torch.Tensor,
        affordances: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return raw positions (unnormalized).

        Args:
            positions: Agent positions [num_agents, N]
            affordances: Dict of affordance positions (currently unused, reserved for future)

        Returns: [num_agents, N] floats (raw coordinate values)
        """
        # affordances parameter currently unused, reserved for future proximity encoding
        return positions.float()
```

**Continuous substrates use identical encoding logic** (just float inputs instead of long).

---

### 2. Distance Metrics

**All three standard metrics extend naturally to N dimensions:**

```python
def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """Compute distance in N-dimensional space.

    Args:
        pos1: [num_agents, N] positions
        pos2: [N] or [num_agents, N] target position(s)

    Returns:
        [num_agents] distances
    """
    if self.distance_metric == "manhattan":
        # L1 norm: |x₁-x₂| + |y₁-y₂| + ... + |xₙ-xₙ|
        return torch.abs(pos1 - pos2).sum(dim=-1)

    elif self.distance_metric == "euclidean":
        # L2 norm: sqrt((x₁-x₂)² + (y₁-y₂)² + ... + (xₙ-xₙ)²)
        return torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1))

    elif self.distance_metric == "chebyshev":
        # L∞ norm: max(|x₁-x₂|, |y₁-y₂|, ..., |xₙ-xₙ|)
        return torch.abs(pos1 - pos2).max(dim=-1)[0]
```

**No modifications needed** - existing Phase 5B distance code already works for arbitrary N.

**Deferred:**
- Weighted metrics (e.g., weighted Euclidean where dimensions have different importance)
- Custom distance functions
- Rationale: Not essential for basic N-dim functionality, can add when researchers request it

---

### 3. Boundary Handling

**All four boundary modes extend per-dimension:**

```python
def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply movement with N-dimensional boundary handling.

    Args:
        positions: [num_agents, N] current positions
        deltas: [num_agents, N] movement deltas

    Returns:
        [num_agents, N] new positions after boundary handling
    """
    new_positions = positions + deltas.long()  # Grid substrates cast to long

    # Apply boundary logic per dimension independently
    for dim in range(self.position_dim):
        min_val, max_val = self.bounds[dim]

        if self.boundary == "clamp":
            # Hard walls: clamp to bounds
            new_positions[:, dim] = torch.clamp(
                new_positions[:, dim], min_val, max_val
            )

        elif self.boundary == "wrap":
            # Toroidal wraparound (N-torus topology)
            range_size = max_val - min_val + 1
            new_positions[:, dim] = (
                (new_positions[:, dim] - min_val) % range_size
            ) + min_val

        elif self.boundary == "bounce":
            # Elastic reflection about boundaries
            range_size = max_val - min_val + 1
            normalized = new_positions[:, dim] - min_val
            normalized = normalized % (2 * range_size)
            exceed_half = normalized >= range_size
            normalized[exceed_half] = 2 * range_size - normalized[exceed_half]
            new_positions[:, dim] = normalized + min_val

        elif self.boundary == "sticky":
            # Stay in place if out of bounds
            out_of_bounds = (new_positions[:, dim] < min_val) | (
                new_positions[:, dim] > max_val
            )
            new_positions[out_of_bounds, dim] = positions[out_of_bounds, dim]

    return new_positions
```

**Key insight:** Boundary handling is **per-dimension** and independent. An N-dimensional hypercube with "wrap" creates an N-torus topology.

**Continuous substrates use identical logic** (just float arithmetic instead of long).

---

### 4. Affordance Placement

**Grid ND (N ≥ 4):**

```python
def get_all_positions(self) -> list[list[int]]:
    """Get all valid positions in N-dimensional grid.

    Warning: Exponential growth! Size = product of all dimension ranges.
    Example: 10×10×10×10 = 10,000 positions (ok)
             10^10 = 10 billion positions (not ok!)
             Consider affordance placement strategies for very large grids.
    """
    positions = []
    # Generate all combinations using itertools.product
    import itertools
    ranges = [range(min_val, max_val + 1) for min_val, max_val in self.bounds]
    for position in itertools.product(*ranges):
        positions.append(list(position))
    return positions

def supports_enumerable_positions(self) -> bool:
    """Grid substrates have finite enumerable positions."""
    return True

def get_valid_neighbors(self, position: torch.Tensor) -> list[torch.Tensor]:
    """Get 2N cardinal neighbors in N dimensions (±1 along each axis).

    Args:
        position: Position tensor [N] or list of N coordinates

    Returns:
        List of neighbor position tensors (up to 2N neighbors)
        Respects boundary conditions - neighbors outside bounds are excluded
    """
    # Convert to list if tensor
    if isinstance(position, torch.Tensor):
        pos_list = position.tolist()
    else:
        pos_list = list(position)

    neighbors = []

    # For each dimension, add ±1 neighbor if within bounds
    for dim in range(self.position_dim):
        min_val, max_val = self.bounds[dim]

        # Negative neighbor (dim - 1)
        neg_pos = pos_list.copy()
        neg_pos[dim] -= 1
        if neg_pos[dim] >= min_val:
            neighbors.append(torch.tensor(neg_pos, dtype=torch.long))

        # Positive neighbor (dim + 1)
        pos_pos = pos_list.copy()
        pos_pos[dim] += 1
        if pos_pos[dim] <= max_val:
            neighbors.append(torch.tensor(pos_pos, dtype=torch.long))

    return neighbors
```

**Continuous ND (N ≥ 4):**

```python
def get_all_positions(self) -> list[list[float]]:
    """Raise error - continuous space has infinite positions."""
    raise NotImplementedError(
        f"{self.__class__.__name__} has infinite positions (continuous space). "
        f"Use random sampling for affordance placement instead. "
        f"See vectorized_env.py randomize_affordance_positions()."
    )

def supports_enumerable_positions(self) -> bool:
    """Continuous substrates have infinite positions."""
    return False
```

**Random Placement (from Phase 5B):**

Existing `randomize_affordance_positions()` in vectorized_env.py already handles N-dimensional substrates correctly:
- Grid: Shuffles all enumerable positions
- Continuous: Random sampling via `substrate.initialize_positions()`

---

### 5. Position Initialization

**Grid ND:**

```python
def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
    """Randomly initialize positions in N-dimensional grid.

    Returns: [num_agents, N] long tensor
    """
    positions = []
    for min_val, max_val in self.bounds:
        dim_positions = torch.randint(
            min_val, max_val + 1,
            (num_agents,),
            device=device
        )
        positions.append(dim_positions)

    return torch.stack(positions, dim=1)
```

**Continuous ND:**

```python
def initialize_positions(self, num_agents: int, device: torch.device) -> torch.Tensor:
    """Randomly initialize positions in N-dimensional continuous space.

    Returns: [num_agents, N] float32 tensor
    """
    positions = []
    for min_val, max_val in self.bounds:
        dim_positions = torch.rand(num_agents, device=device) * (max_val - min_val) + min_val
        positions.append(dim_positions)

    return torch.stack(positions, dim=1)
```

---

### 6. Network Architecture Implications

**Q-Network Input/Output Shapes:**

For N-dimensional substrate:
- **Action space size**: 2N + 1 discrete actions
- **Q-network output**: `[batch_size, 2N+1]`

**Observation dimensions:**
- `relative` encoding: N dimensions
- `scaled` encoding: 2N dimensions
- `absolute` encoding: N dimensions

**Example: 7D Grid with `scaled` encoding**

```python
# Environment observation builder
obs_dim = (
    substrate.position_dim * 2  # 7*2 = 14 (normalized + ranges)
    + 8                          # 8 meters
    + 15                         # 15 affordance types (one-hot)
    + 4                          # 4 temporal extras
)  # Total: 41 dimensions

# Q-network architecture
input_dim = 41
action_dim = 2 * 7 + 1 = 15  # 14 movement + 1 interact

q_network = SimpleQNetwork(
    obs_dim=41,
    action_dim=15,
    hidden_dims=[256, 128]
)
```

**Scaling Considerations:**

| N  | Actions | Obs (relative) | Obs (scaled) | Notes |
|----|---------|----------------|--------------|-------|
| 4  | 9       | 4              | 8            | Manageable |
| 7  | 15      | 7              | 14           | Reasonable |
| 10 | 21      | 10             | 20           | Warning emitted for Grid |
| 20 | 41      | 20             | 40           | Stronger warning for Grid |
| 50 | 101     | 50             | 100          | Action space getting large |
| 100| 201     | 100            | 200          | Continuous gets warning at N≥100 |

**Network size scales linearly with N** (not exponentially), which is why normalized encoding works well.

---

## Phase 5C Scope: Two-Part Implementation

**Phase 5C delivers unified "Configurable Observation Encoding" for all dimensions:**

### Part 1: Retrofit Existing Substrates (Phase 5A/5B)

**Why This Is Part of Phase 5C**: Phase 5B discovered that normalized encoding works well for 3D. Phase 5C makes this approach configurable and extends it to N dimensions. For consistency, we retrofit the configuration to existing 2D/3D substrates in the same phase.

**Current State (Phase 5A/5B):**
- Grid2D, Grid3D: Use **fixed normalized encoding** (equivalent to `relative`)
- Continuous1D, Continuous2D, Continuous3D: Use **fixed normalized encoding** (equivalent to `relative`)
- No configuration option for encoding strategy

**Phase 5C Adds Configurability:**
- All substrates support `observation_encoding: "relative" | "scaled" | "absolute"`
- Consistent interface across 1D/2D/3D/ND substrates
- Enables comparative studies (e.g., "does `scaled` help in 2D vs 7D?")

### Part 2: N-Dimensional Substrates (GridND, ContinuousND)

**New substrate classes for 4D+ dimensions** with configurable observation encoding built-in.

---

## Implementation: Part 1 - Retrofit Observation Encoding

**Changes to Base Class (SpatialSubstrate):**

Add `action_space_size` property to base class for dynamic action space sizing:

```python
# src/townlet/substrate/base.py
class SpatialSubstrate(ABC):
    @property
    def action_space_size(self) -> int:
        """Return number of discrete actions supported by this substrate.

        Returns:
            For spatial substrates: 2*position_dim + 1 (movement + INTERACT)
            For aspatial: 1 (only INTERACT)

        Examples:
            Grid2D: 2*2+1 = 5 (UP/DOWN/LEFT/RIGHT/INTERACT)
            Grid3D: 2*3+1 = 7 (±X/±Y/±Z/INTERACT)
            Grid7D: 2*7+1 = 15 (±7 dims/INTERACT)
            Aspatial: 1 (INTERACT only)
        """
        if self.position_dim == 0:
            return 1  # Aspatial: only INTERACT
        return 2 * self.position_dim + 1
```

**Why This Matters**: Existing Phase 5B code hardcodes action space sizes (5 for Grid2D, 7 for Grid3D). This property enables environments to query dynamic action space sizes, which is essential for N-dimensional substrates.

**Compatibility**: This is NOT a breaking change - existing substrates (Grid2D/3D/Continuous) inherit the property automatically via the base class.

---

**Update All Existing Substrates:**

```python
# BEFORE (Phase 5A/5B - Fixed encoding)
class Grid2DSubstrate(SpatialSubstrate):
    def encode_observation(self, positions, affordances):
        # Always normalized (hardcoded)
        return positions.float() / torch.tensor([self.width, self.height])

# AFTER (Phase 5C - Configurable encoding)
class Grid2DSubstrate(SpatialSubstrate):
    def __init__(self, ..., observation_encoding: str = "relative"):
        # Add observation_encoding parameter with default
        self.observation_encoding = observation_encoding

    def encode_observation(self, positions, affordances):
        if self.observation_encoding == "relative":
            return self._encode_relative(positions)
        elif self.observation_encoding == "scaled":
            return self._encode_scaled(positions)
        elif self.observation_encoding == "absolute":
            return self._encode_absolute(positions)

    def _encode_relative(self, positions):
        # Existing normalization logic
        return positions.float() / torch.tensor([self.width, self.height])

    def _encode_scaled(self, positions):
        # Normalized + range metadata
        relative = self._encode_relative(positions)
        ranges = torch.tensor([self.width, self.height], dtype=torch.float32)
        return torch.cat([relative, ranges.expand(positions.shape[0], -1)], dim=1)

    def _encode_absolute(self, positions):
        # Raw coordinates
        return positions.float()
```

**Substrates to Update:**
- `Grid2DSubstrate` (src/townlet/substrate/grid2d.py)
- `Grid3DSubstrate` (src/townlet/substrate/grid3d.py)
- `Continuous1DSubstrate` (src/townlet/substrate/continuous.py)
- `Continuous2DSubstrate` (src/townlet/substrate/continuous.py)
- `Continuous3DSubstrate` (src/townlet/substrate/continuous.py)
- `AspatialSubstrate` (src/townlet/substrate/aspatial.py) - N/A, no position encoding

**Configuration Updates:**

```yaml
# substrate.yaml (for all configs)
type: "grid"
grid:
  topology: "square"
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"  # NEW: default to preserve existing behavior
```

**Backward Compatibility:**
- Default value: `observation_encoding: "relative"` (preserves current behavior)
- Existing configs without this field: Automatically use `relative` (no breaking change)
- New configs can opt into `scaled` or `absolute`

**Testing Requirements:**
- Add unit tests for all three encodings on Grid2D/3D
- Add integration tests ensuring existing training runs produce identical results with `relative`
- Verify observation dimensions change correctly with `scaled`

**Estimated Effort**: 4-6 hours
- Update 5 substrate classes (1h each)
- Add encoding tests (1-2h)
- Update config schema and validation (1h)

**Priority**: **BLOCKING** - Phase 5C Part 2 (N-dimensional substrates) cannot be implemented until Part 1 (retrofit) is complete.

### Benefits

**Consistency:**
- All substrates (1D/2D/3D/ND) use same observation encoding interface
- Researchers can run ablation studies across all dimensionalities

**Research Value:**
- Compare `relative` vs `scaled` encoding at different N
- Test if absolute coordinates help in some domains
- Enables systematic experimentation

**Future-Proof:**
- Clean foundation for additional encoding strategies
- Uniform interface reduces technical debt

---

## Future Extensions (Deferred)

These features were considered but deferred until researchers request them:

### 1. Per-Dimension Step Sizes
**What**: Different dimensions move by different amounts
**Example**: Temperature moves by 1°, pH moves by 0.1
**Why Defer**: User-defined actions task will handle this (actions define their own deltas)

### 2. Weighted Distance Metrics
**What**: `weighted_euclidean` where dimensions have different importance
**Example**: Distance = sqrt(w₁(x₁-x₂)² + w₂(y₁-y₂)² + ...)
**Why Defer**: Not essential for basic N-dim functionality, `scaled` encoding handles heterogeneous ranges

### 3. Affordance Proximity in Observations
**What**: Include distance to nearest affordance per dimension
**Example**: "Closest food is 5 units away in temperature dimension"
**Why Defer**: Can add as new `observation_encoding` option when requested

### 4. Per-Dimension Boundary Modes
**What**: Some dimensions wrap, others clamp
**Example**: Time wraps (24-hour cycle), temperature clamps
**Why Defer**: Adds config complexity, unclear if useful

### 5. Continuous Action Spaces (For Very High N)
**What**: Policy outputs continuous N-dimensional vector instead of discrete action index
**Example**: Actor-critic (PPO/SAC) instead of DQN for N=100+
**Why Defer**: Requires different training algorithm (separate Phase)
**Note**: Discrete actions scale well to N=50+, but continuous actions may be better for N>100

### 6. Dimension-Specific Metadata
**What**: Attach units, types, semantic meaning to each dimension
**Example**: `{name: "temperature", units: "celsius", type: "continuous"}`
**Why Defer**: Universe-as-code philosophy prefers minimal substrate, rich action labels

### 7. Hierarchical Dimensions
**What**: Nested state spaces (e.g., 3D physical space + 4D internal state)
**Example**: Agent has 3D position + 4D [hunger, thirst, fatigue, mood]
**Why Defer**: Complex compositional structure, unclear use case

---

## Testing Strategy

### Unit Tests

**GridNDSubstrate:**
- Initialization validation (N ≥ 4, warnings at N≥10, N≥20)
- Position initialization (random in bounds)
- Movement in each dimension
- Boundary handling per dimension (clamp/wrap/bounce/sticky)
- Distance metrics (manhattan/euclidean/chebyshev)
- Observation encoding (relative/scaled/absolute)
- Action-to-delta mapping (2N+1 actions)
- `action_space_size` property returns correct 2N+1
- `get_valid_neighbors()` returns 2N neighbors
- `supports_enumerable_positions()` returns True

**ContinuousNDSubstrate:**
- Initialization validation (N ≥ 4, warning at N≥100)
- Float position initialization
- Movement with movement_delta
- Boundary handling for floats
- Interaction radius proximity checks
- `action_space_size` property returns correct 2N+1
- `supports_enumerable_positions()` returns False
- Observation encoding (all three modes)

### Integration Tests

**Full Training Loop:**
- 4D grid training (minimal N)
- 10D grid training (max N)
- 7D continuous training (mid-range N)
- Test with different observation encodings
- Verify action space size matches 2N+1
- Checkpoint save/load with N-dimensional positions

### Config Validation Tests

**Schema Validation:**
- Grid: Error if N < 4, warnings at N≥10 and N≥20
- Continuous: Error if N < 4, warning at N≥100
- Error if len(bounds) ≠ dimensions
- Error if bounds invalid (min ≥ max)
- Warn if interaction_radius < movement_delta

---

## Pedagogical Value

**For Students:**
- **Abstraction**: RL isn't just "move around a grid," it's navigation in ANY state space
- **Dimensionality**: Experience curse of dimensionality firsthand
- **Generalization**: Do policies trained on 5D transfer to 6D?

**For Researchers:**
- **Multi-Objective RL**: Each dimension is an objective to optimize
- **Hyperparameter Search**: AutoML experiments in parameter space
- **Transfer Learning**: Study dimension scaling properties
- **State Representation**: Does observation encoding matter?

**For Advanced Users:**
- Formal experimentation platform for abstract RL research
- "Universe as code" enables reproducible high-dimensional experiments
- Action space design studies (discrete vs continuous thresholds)

---

## Implementation Checklist (Future Phase)

When this design is ready for implementation, create a separate implementation plan covering:

**Core Implementation:**
- [ ] **PREREQUISITE**: Complete Phase 5C (observation_encoding retrofit)
- [ ] Create `GridNDSubstrate` class (N ≥ 4, warnings at N≥10, N≥20)
- [ ] Create `ContinuousNDSubstrate` class (N ≥ 4, warning at N≥100)
- [ ] Add `action_space_size` property to both classes
- [ ] Add `get_valid_neighbors()` method to GridND
- [ ] Add `supports_enumerable_positions()` to both classes
- [ ] Update config schema with `topology: "hypercube"`
- [ ] Update factory to recognize hypercube topology
- [ ] Extend `_action_to_deltas()` for dynamic N dimensions
- [ ] Add dimension validation logic with warnings

**Testing:**
- [ ] Unit tests for GridND (20+ tests)
- [ ] Unit tests for ContinuousND (20+ tests)
- [ ] Integration tests (4D, 7D, 10D configs)
- [ ] Config validation tests
- [ ] Performance benchmarks (scaling with N)

**Documentation:**
- [ ] CLAUDE.md examples for N-dimensional substrates
- [ ] Config templates for 4D/7D/10D
- [ ] Research use case examples
- [ ] Action space documentation

**Config Packs:**
- [ ] `L1_4D_abstract/` (minimal N-dimensional example)
- [ ] `L1_7D_multiobjective/` (mid-range abstract state space)
- [ ] `L1_10D_hyperparameter/` (max discrete dimensions)

---

## Document Status

**Status**: Design Complete - v4 Rebranded as Phase 5C (Scheduled After Phase 5B)
**Next Step**: Create implementation plan after Phase 5B completion
**Dependencies**:
- Phase 5B (3D, Continuous, Configurable Actions) - MUST BE COMPLETE
**Estimated Effort**: TBD (likely 20-25 hours total for both parts)

**Design Decisions Made (v2):**
- ✅ Grid uncapped (N ≥ 4) with warnings at N≥10, N≥20
- ✅ Continuous uncapped (N ≥ 4) with warning at N≥100
- ✅ Three observation encodings (relative/scaled/absolute)
- ✅ Anonymous dimensions for 4D+ (DIM_0, DIM_1, ...)
- ✅ Auto-generated action spaces (2N+1 discrete)
- ✅ Dynamic action_space_size property
- ✅ get_valid_neighbors() method for GridND
- ✅ supports_enumerable_positions() for both substrates
- ✅ Per-dimension boundary handling
- ✅ Standard distance metrics (manhattan/euclidean/chebyshev)

**Deferred for Future Consideration:**
- Per-dimension step sizes (handled by user-defined actions)
- Weighted distance metrics
- Continuous action spaces (separate Phase)
- Dimension metadata/semantics beyond action labels

---

**Document Created**: 2025-11-05
**Contributors**: Design collaboration with user
**Review Status**: Design validated through Q&A process
