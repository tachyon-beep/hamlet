# Research: TASK-002A Phase 5 - Observation Builder Integration

**Date**: 2025-11-05
**Status**: Research Complete
**Phase**: Phase 5 of Configurable Spatial Substrates
**Context**: Phases 0-4 create substrate abstraction and integrate into position management; Phase 5 integrates substrate into observation encoding

---

## Executive Summary

Phase 5 must refactor `ObservationBuilder` to use substrate-based position encoding instead of hardcoded grid encoding. This research identifies **4 core integration points** in **2 main files** with **low risk** since substrate provides clean abstraction.

**Key Finding**: Current system has **two separate position encodings**:
1. **Full observability**: One-hot grid encoding (64 dims for 8×8)
2. **Partial observability (POMDP)**: Normalized coordinates (2 dims) + local 5×5 window

Both must be replaced with `substrate.encode_observation()` and `substrate.encode_partial_observation()`.

**Critical Insight**: L2 POMDP **already uses coordinate encoding** successfully, proving that networks can learn spatial reasoning from normalized positions. This validates the substrate approach.

**Estimated Effort**: **12-16 hours** (8h integration + 4h testing + 4h edge cases)

**Dependencies**: Phase 4 provides `substrate.initialize_positions()`, `substrate.compute_distance()`, `substrate.apply_movement()`. Phase 5 adds `substrate.encode_observation()` and `substrate.encode_partial_observation()`.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Substrate Methods Required](#2-substrate-methods-required)
3. [Integration Points](#3-integration-points)
4. [Observation Dimension Calculation](#4-observation-dimension-calculation)
5. [POMDP Local Window Strategy](#5-pomdp-local-window-strategy)
6. [Aspatial Substrate Handling](#6-aspatial-substrate-handling)
7. [Network Compatibility](#7-network-compatibility)
8. [Code Examples](#8-code-examples)
9. [Dependencies from Phase 4](#9-dependencies-from-phase-4)
10. [Risks and Mitigations](#10-risks-and-mitigations)
11. [Implementation Strategy](#11-implementation-strategy)
12. [Effort Estimates](#12-effort-estimates)

---

## 1. Current State Analysis

### 1.1 ObservationBuilder Architecture

**File**: `src/townlet/environment/observation_builder.py`

**Current Implementation** (248 lines):

```python
class ObservationBuilder:
    """Constructs observations for agents in vectorized Hamlet environment.

    Supports three observation modes:
    - Full observability: One-hot grid position + meters + affordance encoding
    - Partial observability (POMDP): Local 5×5 window + position + meters + affordance
    - Temporal mechanics: Adds time_of_day and interaction_progress features
    """

    def __init__(
        self,
        num_agents: int,
        grid_size: int,  # ← HARDCODED 2D ASSUMPTION
        device: torch.device,
        partial_observability: bool,
        vision_range: int,
        enable_temporal_mechanics: bool,
        num_affordance_types: int,
        affordance_names: list[str],
    ):
        self.grid_size = grid_size  # Single dimension for square grid
```

**Key Methods**:

1. **`build_observations()`** (Lines 52-102)
   - Entry point for observation construction
   - Delegates to `_build_full_observations()` or `_build_partial_observations()`
   - Appends temporal features (always included for forward compatibility)

2. **`_build_full_observations()`** (Lines 104-146)
   - **Grid encoding** (Lines 127-137): One-hot encoding of agent position + affordance positions
   - Uses hardcoded formula: `flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]`
   - Returns: `[num_agents, grid_size² + 8 + num_affordance_types + 1]`

3. **`_build_partial_observations()`** (Lines 148-209)
   - **Local window extraction** (Lines 167-195): Manual 5×5 window extraction
   - **Position encoding** (Lines 200-201): Normalized coordinates
   - Uses `normalized_positions = positions.float() / (self.grid_size - 1)`
   - Returns: `[num_agents, window² + 2 + 8 + num_affordance_types + 1]`

4. **`_build_affordance_encoding()`** (Lines 211-247)
   - One-hot encoding of current affordance under agent
   - Distance calculation: `distances = torch.abs(positions - affordance_pos).sum(dim=1)`
   - Substrate-independent (works for any position representation)

**Observations**:

- **Grid size is scalar** (`self.grid_size`), assumes square 2D grid
- **One-hot encoding hardcoded** for full observability
- **Local window manually extracted** with nested loops (Lines 169-195)
- **POMDP already uses coordinate encoding** (Line 201) - proves concept works!
- **Temporal features always appended** (Lines 79-100) - substrate-independent

---

### 1.2 VectorizedHamletEnv Integration

**File**: `src/townlet/environment/vectorized_env.py`

**Current obs_dim Calculation** (Lines 129-142):

```python
# Observation dimensions depend on observability mode
if partial_observability:
    # Level 2 POMDP: local window + position + meters + current affordance type
    window_size = 2 * vision_range + 1  # 5×5 for vision_range=2
    # Grid + position + meter_count meters + affordance type one-hot (N+1 for "none")
    self.observation_dim = window_size * window_size + 2 + meter_count + (self.num_affordance_types + 1)
else:
    # Level 1: full grid one-hot + meters + current affordance type
    # Grid one-hot + meter_count meters + affordance type (N+1 for "none")
    self.observation_dim = grid_size * grid_size + meter_count + (self.num_affordance_types + 1)

# Always add temporal features for forward compatibility (4 features)
# time_sin, time_cos, interaction_progress, lifetime_progress
self.observation_dim += 4
```

**Key Observations**:

- **obs_dim calculated in VectorizedHamletEnv**, not ObservationBuilder
- **Hardcoded formulas** for full obs (grid²) and POMDP (window² + 2)
- **POMDP position encoding is 2 dims** (normalized x, y)
- **Temporal features always +4** (substrate-independent)

**How obs_dim is Used**:

```python
# In runner.py (Line 371)
obs_dim = self.env.observation_dim

# In population/vectorized.py (Lines 120, 134)
self.q_network = SimpleQNetwork(obs_dim, action_dim, hidden_dim=128)
self.target_network = SimpleQNetwork(obs_dim, action_dim, hidden_dim=128)

# In exploration/adaptive_intrinsic.py (Line 56)
self.rnd_module = RNDExploration(obs_dim=obs_dim, embed_dim=embed_dim, device=device)

# In training/replay_buffer.py (Line 51)
obs_dim = observations.shape[1]
self.observations = torch.zeros(self.capacity, obs_dim, device=self.device)
```

**Critical**: obs_dim must be known **before** creating networks, exploration modules, and replay buffers.

---

### 1.3 Current Observation Dimensions

**Full Observability** (L0, L0.5, L1):

```
obs_dim = grid_size² + meter_count + (num_affordance_types + 1) + 4

L0_0_minimal (3×3):
  = 9 + 8 + 15 + 4 = 36 dims

L0_5_dual_resource (7×7):
  = 49 + 8 + 15 + 4 = 76 dims

L1_full_observability (8×8):
  = 64 + 8 + 15 + 4 = 91 dims
```

**Partial Observability** (L2 POMDP):

```
obs_dim = window_size² + 2 + meter_count + (num_affordance_types + 1) + 4

L2_partial_observability (5×5 window):
  = 25 + 2 + 8 + 15 + 4 = 54 dims
```

**Key Insight**: POMDP uses only **2 dims** for position (normalized coordinates) vs **64 dims** for full obs (one-hot). This is a **32× reduction** in position encoding size!

---

## 2. Substrate Methods Required

Phase 5 requires **two new substrate methods** for observation encoding:

### 2.1 `encode_observation()` - Full Observability

**Signature**:

```python
def encode_observation(self, positions: torch.Tensor) -> torch.Tensor:
    """Encode positions into observation space for full observability.

    Args:
        positions: [num_agents, position_dim] raw positions

    Returns:
        [num_agents, observation_dim] encoded positions
    """
```

**Implementation Strategy by Substrate**:

**Grid2D (Square Grid)**:

- **Auto-select encoding** based on grid size:
  - Small grids (≤8×8): One-hot encoding (current behavior)
  - Large grids (>8×8): Coordinate encoding (prevents explosion)
- **One-hot**: `flat_indices = y * width + x`, scatter to `[width * height]`
- **Coordinates**: Normalize to `[0, 1]` range

**Grid3D (Cubic Grid)**:

- **Force coordinate encoding** (one-hot would be 512+ dims!)
- Normalize (x, y, z) to `[0, 1]` range

**Aspatial**:

- Return empty tensor `[num_agents, 0]`

### 2.2 `encode_partial_observation()` - POMDP Local Window

**Signature**:

```python
def encode_partial_observation(
    self,
    agent_position: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor],
    vision_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode local vision window + position for POMDP.

    Args:
        agent_position: [position_dim] single agent position
        affordance_positions: Dict of affordance_name -> position tensor
        vision_range: Radius of vision window (2 = 5×5)

    Returns:
        local_grid: [window_size²] flattened local window
        normalized_position: [position_dim] normalized position
    """
```

**Implementation Strategy**:

**Grid2D (Square Grid)**:

- Extract 5×5 window centered on agent
- Mark affordances within window (value = 1.0)
- Return normalized (x, y) position

**Grid3D (Cubic Grid)**:

- Extract 5×5×5 cube centered on agent (125 dims)
- OR: Extract 5×5 slice at current floor (25 dims + floor indicator)
- Return normalized (x, y, z) position

**Aspatial**:

- No local window concept
- Return empty tensor `[0]` for local grid
- Return empty tensor `[0]` for position

---

### 2.3 `get_observation_dim()` - Dimension Query

**Signature**:

```python
def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
    """Get observation dimension for this substrate.

    Args:
        partial_observability: If True, calculate POMDP dimension
        vision_range: Vision radius for POMDP (ignored for full obs)

    Returns:
        Observation dimension (position encoding only, excludes meters/affordances/temporal)
    """
```

**Implementation**:

```python
# Grid2D
def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
    if partial_observability:
        window_size = 2 * vision_range + 1
        return window_size * window_size + 2  # Local grid + normalized (x, y)
    else:
        if self.position_encoding == "onehot":
            return self.width * self.height  # One-hot grid
        elif self.position_encoding == "coords":
            return 2  # Normalized (x, y)

# Grid3D
def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
    if partial_observability:
        window_size = 2 * vision_range + 1
        return window_size ** 3 + 3  # Local cube + normalized (x, y, z)
    else:
        return 3  # Always use coordinates (one-hot infeasible)

# Aspatial
def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
    return 0  # No position encoding
```

---

## 3. Integration Points

### 3.1 VectorizedHamletEnv.__init__() - obs_dim Calculation

**File**: `src/townlet/environment/vectorized_env.py` (Lines 129-142)

**Current**:

```python
# Observation dimensions depend on observability mode
if partial_observability:
    window_size = 2 * vision_range + 1
    self.observation_dim = window_size * window_size + 2 + meter_count + (self.num_affordance_types + 1)
else:
    self.observation_dim = grid_size * grid_size + meter_count + (self.num_affordance_types + 1)

# Always add temporal features
self.observation_dim += 4
```

**After Substrate Integration**:

```python
# Observation dimensions: substrate position encoding + meters + affordances + temporal
substrate_obs_dim = self.substrate.get_observation_dim(partial_observability, vision_range)
self.observation_dim = substrate_obs_dim + meter_count + (self.num_affordance_types + 1) + 4
```

**Changes Required**:

- Replace hardcoded formulas with `substrate.get_observation_dim()`
- Pass `partial_observability` and `vision_range` to substrate query
- Substrate must be loaded **before** obs_dim calculation

**Effort**: 1 hour (simple replacement)

---

### 3.2 ObservationBuilder.__init__() - Remove grid_size

**File**: `src/townlet/environment/observation_builder.py` (Lines 20-50)

**Current**:

```python
def __init__(
    self,
    num_agents: int,
    grid_size: int,  # ← REMOVE
    device: torch.device,
    partial_observability: bool,
    vision_range: int,
    enable_temporal_mechanics: bool,
    num_affordance_types: int,
    affordance_names: list[str],
):
    self.grid_size = grid_size
```

**After Substrate Integration**:

```python
def __init__(
    self,
    num_agents: int,
    substrate: SpatialSubstrate,  # ← ADD
    device: torch.device,
    partial_observability: bool,
    vision_range: int,
    enable_temporal_mechanics: bool,
    num_affordance_types: int,
    affordance_names: list[str],
):
    self.substrate = substrate
    self.position_dim = substrate.position_dim
```

**Changes Required**:

- Replace `grid_size: int` with `substrate: SpatialSubstrate`
- Store `substrate` reference for encoding methods
- Remove `self.grid_size` attribute

**Effort**: 30 minutes (signature change + update all call sites)

---

### 3.3 ObservationBuilder._build_full_observations() - Use substrate.encode_observation()

**File**: `src/townlet/environment/observation_builder.py` (Lines 104-146)

**Current** (Lines 124-137):

```python
# Grid encoding: mark BOTH agent position AND affordance positions
grid_encoding = torch.zeros(self.num_agents, self.grid_size * self.grid_size, device=self.device)

# Mark affordance positions (value = 1.0) for all agents
for affordance_pos in affordances.values():
    affordance_flat_idx = affordance_pos[1] * self.grid_size + affordance_pos[0]
    grid_encoding[:, affordance_flat_idx] = 1.0

# Mark agent position (add 1.0, so if on affordance it becomes 2.0)
flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), torch.ones(self.num_agents, 1, device=self.device))
```

**After Substrate Integration**:

```python
# Encode agent positions using substrate
position_encoding = self.substrate.encode_observation(positions)  # [num_agents, obs_dim]

# For Grid2D with one-hot encoding, substrate handles agent+affordance overlay
# For Grid2D with coordinate encoding, just get normalized positions
# For Grid3D, always get normalized (x, y, z)
# For Aspatial, get empty tensor [num_agents, 0]
```

**Critical Design Decision**: Should affordance positions be encoded in substrate or observation builder?

**Option A: Substrate encodes affordances** (RECOMMENDED)

```python
# In substrate
def encode_observation(
    self,
    agent_positions: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encode agent positions AND affordance positions into observation."""
```

**Pros**:
- Substrate owns complete spatial encoding
- Can optimize affordance overlay (e.g., broadcast to all agents)
- Grid2D one-hot encoding already does this (agent + affordances in same grid)

**Cons**:
- Couples substrate to affordance concept (but affordances ARE spatial!)
- More complex substrate interface

**Option B: ObservationBuilder overlays affordances** (CURRENT)

```python
# In observation builder
agent_encoding = self.substrate.encode_observation(agent_positions)
# Manually overlay affordance positions (how?)
```

**Pros**:
- Substrate only handles position encoding
- Simpler substrate interface

**Cons**:
- ObservationBuilder needs to know substrate encoding format (breaks abstraction)
- Can't overlay affordances on coordinate encoding (they're not in same space!)

**Verdict**: **Option A** - Substrate must encode affordances for Grid2D one-hot mode.

**Revised Substrate Signature**:

```python
def encode_observation(
    self,
    agent_positions: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Encode agent positions (and optionally affordance positions) into observation.

    Args:
        agent_positions: [num_agents, position_dim] agent positions
        affordance_positions: Optional dict of affordance_name -> position tensor
            For Grid2D one-hot: affordances overlaid on same grid
            For coordinate encoding: affordances not encoded in position (handled separately)

    Returns:
        [num_agents, observation_dim] encoded positions
    """
```

**Implementation in Grid2D**:

```python
def encode_observation(
    self,
    agent_positions: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if self.position_encoding == "onehot":
        # One-hot grid: mark affordances first, then add agents
        num_agents = agent_positions.shape[0]
        grid_encoding = torch.zeros(num_agents, self.width * self.height, device=agent_positions.device)

        # Mark affordances (broadcast to all agents)
        if affordance_positions:
            for affordance_pos in affordance_positions.values():
                flat_idx = affordance_pos[1] * self.width + affordance_pos[0]
                grid_encoding[:, flat_idx] = 1.0

        # Mark agent positions (add 1.0, so agent on affordance = 2.0)
        flat_indices = agent_positions[:, 1] * self.width + agent_positions[:, 0]
        grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), torch.ones(num_agents, 1, device=agent_positions.device))

        return grid_encoding

    elif self.position_encoding == "coords":
        # Coordinate encoding: just normalize agent positions
        # Affordances NOT encoded in position (handled by affordance encoding)
        normalized_x = agent_positions[:, 0].float() / (self.width - 1)
        normalized_y = agent_positions[:, 1].float() / (self.height - 1)
        return torch.stack([normalized_x, normalized_y], dim=1)
```

**Changes Required**:

- Replace grid encoding logic with `substrate.encode_observation(positions, affordances)`
- Handle both one-hot (affordances overlaid) and coordinate (affordances separate)
- Remove hardcoded `grid_size` references

**Effort**: 2 hours (design + implementation + testing)

---

### 3.4 ObservationBuilder._build_partial_observations() - Use substrate.encode_partial_observation()

**File**: `src/townlet/environment/observation_builder.py` (Lines 148-209)

**Current** (Lines 167-195):

```python
window_size = 2 * self.vision_range + 1
local_grids = []

for agent_idx in range(self.num_agents):
    agent_pos = positions[agent_idx]
    local_grid = torch.zeros(window_size * window_size, device=self.device)

    # Extract local window centered on agent
    for dy in range(-self.vision_range, self.vision_range + 1):
        for dx in range(-self.vision_range, self.vision_range + 1):
            world_x = agent_pos[0] + dx
            world_y = agent_pos[1] + dy

            # Check if position is within grid bounds
            if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                # Check if there's an affordance at this position
                has_affordance = False
                for affordance_pos in affordances.values():
                    if affordance_pos[0] == world_x and affordance_pos[1] == world_y:
                        has_affordance = True
                        break

                # Encode in local grid (1 = affordance, 0 = empty/out-of-bounds)
                if has_affordance:
                    local_y = dy + self.vision_range
                    local_x = dx + self.vision_range
                    local_idx = local_y * window_size + local_x
                    local_grid[local_idx] = 1.0

    local_grids.append(local_grid)

# Stack all local grids
local_grids_batch = torch.stack(local_grids)

# Normalize positions to [0, 1]
normalized_positions = positions.float() / (self.grid_size - 1)
```

**After Substrate Integration**:

```python
# Use substrate to extract local windows and encode positions
local_grids = []
normalized_positions = []

for agent_idx in range(self.num_agents):
    agent_pos = positions[agent_idx]

    # Delegate to substrate
    local_grid, normalized_pos = self.substrate.encode_partial_observation(
        agent_position=agent_pos,
        affordance_positions=affordances,
        vision_range=self.vision_range,
    )

    local_grids.append(local_grid)
    normalized_positions.append(normalized_pos)

# Stack batches
local_grids_batch = torch.stack(local_grids)
normalized_positions_batch = torch.stack(normalized_positions)
```

**Changes Required**:

- Replace manual window extraction with `substrate.encode_partial_observation()`
- Remove hardcoded `grid_size` references
- Handle per-agent encoding (loop required since window is agent-specific)

**Optimization Note**: Current implementation loops over agents sequentially. Could be vectorized in substrate for better performance, but not critical for Phase 5.

**Effort**: 2 hours (implementation + testing different substrates)

---

## 4. Observation Dimension Calculation

### 4.1 Current Formula

**Full Observability**:

```
obs_dim = position_encoding_dim + meter_count + affordance_encoding_dim + temporal_dim
        = grid_size² + meter_count + (num_affordance_types + 1) + 4
```

**Partial Observability**:

```
obs_dim = local_window_dim + position_dim + meter_count + affordance_encoding_dim + temporal_dim
        = window_size² + 2 + meter_count + (num_affordance_types + 1) + 4
```

### 4.2 After Substrate Integration

**Full Observability**:

```
obs_dim = substrate.get_observation_dim(partial_observability=False, vision_range=0)
        + meter_count
        + (num_affordance_types + 1)
        + 4
```

**Partial Observability**:

```
obs_dim = substrate.get_observation_dim(partial_observability=True, vision_range=2)
        + meter_count
        + (num_affordance_types + 1)
        + 4
```

### 4.3 Substrate Dimension Examples

**Grid2D (8×8) - One-Hot Encoding**:

- Full obs: `64 + 8 + 15 + 4 = 91` (unchanged from current)
- POMDP: `25 + 2 + 8 + 15 + 4 = 54` (unchanged from current)

**Grid2D (8×8) - Coordinate Encoding**:

- Full obs: `2 + 8 + 15 + 4 = 29` (62 dims smaller!)
- POMDP: `25 + 2 + 8 + 15 + 4 = 54` (unchanged)

**Grid3D (8×8×3) - Coordinate Encoding**:

- Full obs: `3 + 8 + 15 + 4 = 30` (one-hot would be 512 + 8 + 15 + 4 = 539!)
- POMDP: `125 + 3 + 8 + 15 + 4 = 155` (5×5×5 cube)

**Aspatial**:

- Full obs: `0 + 8 + 15 + 4 = 27` (no position encoding!)
- POMDP: Not applicable (no spatial window)

---

## 5. POMDP Local Window Strategy

### 5.1 Current Implementation (Grid2D)

**Window Extraction** (Lines 169-195):

- 5×5 window centered on agent (vision_range=2)
- Manual nested loops: `for dy in range(-2, 3): for dx in range(-2, 3):`
- Boundary check: `if 0 <= world_x < grid_size and 0 <= world_y < grid_size`
- Affordance check: iterate over all affordances, check if at position
- Mark affordances with 1.0, empty cells with 0.0

**Position Encoding** (Line 201):

- Normalize to [0, 1]: `positions.float() / (grid_size - 1)`
- Returns 2-dim vector: `[x_norm, y_norm]`

### 5.2 After Substrate Integration

**Grid2D**:

```python
def encode_partial_observation(
    self,
    agent_position: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor],
    vision_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 5×5 local window centered on agent."""
    window_size = 2 * vision_range + 1
    local_grid = torch.zeros(window_size * window_size, device=agent_position.device)

    # Extract window
    for dy in range(-vision_range, vision_range + 1):
        for dx in range(-vision_range, vision_range + 1):
            world_x = agent_position[0] + dx
            world_y = agent_position[1] + dy

            # Boundary check
            if 0 <= world_x < self.width and 0 <= world_y < self.height:
                # Check affordances
                for affordance_pos in affordance_positions.values():
                    if affordance_pos[0] == world_x and affordance_pos[1] == world_y:
                        local_y = dy + vision_range
                        local_x = dx + vision_range
                        local_idx = local_y * window_size + local_x
                        local_grid[local_idx] = 1.0
                        break

    # Normalize position
    normalized_x = agent_position[0].float() / (self.width - 1)
    normalized_y = agent_position[1].float() / (self.height - 1)
    normalized_position = torch.tensor([normalized_x, normalized_y], device=agent_position.device)

    return local_grid, normalized_position
```

**Grid3D**:

Two options for 3D POMDP:

**Option A: 5×5×5 Cube** (125 dims)

```python
def encode_partial_observation(
    self,
    agent_position: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor],
    vision_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 5×5×5 local cube centered on agent."""
    window_size = 2 * vision_range + 1
    local_cube = torch.zeros(window_size ** 3, device=agent_position.device)

    # Extract cube (3 nested loops: dx, dy, dz)
    for dz in range(-vision_range, vision_range + 1):
        for dy in range(-vision_range, vision_range + 1):
            for dx in range(-vision_range, vision_range + 1):
                world_x = agent_position[0] + dx
                world_y = agent_position[1] + dy
                world_z = agent_position[2] + dz

                # Boundary check
                if (0 <= world_x < self.width and
                    0 <= world_y < self.height and
                    0 <= world_z < self.depth):
                    # Check affordances
                    for affordance_pos in affordance_positions.values():
                        if (affordance_pos[0] == world_x and
                            affordance_pos[1] == world_y and
                            affordance_pos[2] == world_z):
                            local_z = dz + vision_range
                            local_y = dy + vision_range
                            local_x = dx + vision_range
                            local_idx = local_z * window_size ** 2 + local_y * window_size + local_x
                            local_cube[local_idx] = 1.0
                            break

    # Normalize position
    normalized_x = agent_position[0].float() / (self.width - 1)
    normalized_y = agent_position[1].float() / (self.height - 1)
    normalized_z = agent_position[2].float() / (self.depth - 1)
    normalized_position = torch.tensor([normalized_x, normalized_y, normalized_z], device=agent_position.device)

    return local_cube, normalized_position
```

**Option B: 5×5 Floor Slice + Floor Indicator** (25 + 1 = 26 dims)

```python
def encode_partial_observation(
    self,
    agent_position: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor],
    vision_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 5×5 local window on current floor + floor indicator."""
    window_size = 2 * vision_range + 1
    local_grid = torch.zeros(window_size * window_size + 1, device=agent_position.device)

    # Extract window on current floor (z = agent_position[2])
    current_floor = agent_position[2]
    for dy in range(-vision_range, vision_range + 1):
        for dx in range(-vision_range, vision_range + 1):
            world_x = agent_position[0] + dx
            world_y = agent_position[1] + dy

            # Boundary check (2D + floor check)
            if (0 <= world_x < self.width and
                0 <= world_y < self.height):
                # Check affordances on current floor
                for affordance_pos in affordance_positions.values():
                    if (affordance_pos[0] == world_x and
                        affordance_pos[1] == world_y and
                        affordance_pos[2] == current_floor):
                        local_y = dy + vision_range
                        local_x = dx + vision_range
                        local_idx = local_y * window_size + local_x
                        local_grid[local_idx] = 1.0
                        break

    # Add floor indicator (normalized z)
    local_grid[-1] = current_floor.float() / (self.depth - 1)

    # Normalize position
    normalized_x = agent_position[0].float() / (self.width - 1)
    normalized_y = agent_position[1].float() / (self.height - 1)
    normalized_z = current_floor.float() / (self.depth - 1)
    normalized_position = torch.tensor([normalized_x, normalized_y, normalized_z], device=agent_position.device)

    return local_grid, normalized_position
```

**Recommendation**: Start with **Option A** (5×5×5 cube) for simplicity. Option B is an optimization if 125 dims is too large.

**Aspatial**:

```python
def encode_partial_observation(
    self,
    agent_position: torch.Tensor,
    affordance_positions: dict[str, torch.Tensor],
    vision_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """No local window for aspatial substrate."""
    # Return empty tensors
    local_grid = torch.zeros(0, device=agent_position.device)
    normalized_position = torch.zeros(0, device=agent_position.device)
    return local_grid, normalized_position
```

---

## 6. Aspatial Substrate Handling

### 6.1 Aspatial Observations

**Position Encoding**: Empty tensor `[num_agents, 0]`

**Observation Composition**:

```
obs_dim = 0 + meter_count + (num_affordance_types + 1) + 4
        = 0 + 8 + 15 + 4 = 27 dims
```

**Observation Structure**:

```python
observations = torch.cat([
    # No position encoding (empty)
    meters,                    # [num_agents, 8]
    affordance_encoding,       # [num_agents, 15]
    temporal_features,         # [num_agents, 4]
], dim=1)
```

### 6.2 Aspatial Interaction Model

**Key Question**: How do agents interact with affordances without position?

**Answer**: Direct interaction (no movement required)

**Action Space**:

- UP, DOWN, LEFT, RIGHT: Invalid (no movement in aspatial)
- INTERACT: Valid if affordance available (action masking)
- WAIT: Valid

**Affordance Availability**:

Option A: All affordances always available (no spatial constraint)
Option B: One affordance at a time (state machine)
Option C: Configurable availability schedule (time-based)

**Recommendation**: Start with **Option A** (all affordances always available). Simplest model for pure resource management.

### 6.3 ObservationBuilder Changes for Aspatial

**Full Observability**:

```python
def _build_full_observations(self, positions, meters, affordances):
    # Encode positions using substrate
    position_encoding = self.substrate.encode_observation(
        agent_positions=positions,
        affordance_positions=affordances,
    )  # [num_agents, obs_dim] - could be [num_agents, 0] for aspatial!

    # Get affordance encoding (substrate-independent)
    affordance_encoding = self._build_affordance_encoding(positions, affordances)

    # Concatenate: position + meters + affordance
    # If position_encoding is empty [num_agents, 0], cat still works!
    observations = torch.cat([position_encoding, meters, affordance_encoding], dim=1)

    return observations
```

**Partial Observability**: Not applicable for aspatial (no local window)

---

## 7. Network Compatibility

### 7.1 SimpleQNetwork (Full Observability)

**Current**:

```python
class SimpleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),  # obs_dim can be ANY size
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
```

**Compatibility**: ✅ Works with ANY obs_dim

- Grid2D one-hot (8×8): obs_dim = 91 ✅
- Grid2D coords (8×8): obs_dim = 29 ✅
- Grid3D coords (8×8×3): obs_dim = 30 ✅
- Aspatial: obs_dim = 27 ✅

**No changes required** - network is substrate-agnostic!

### 7.2 RecurrentSpatialQNetwork (POMDP)

**Current** (Lines 162-194):

```python
def forward(self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
    # Split observation components with dynamic indices
    grid_size_flat = self.window_size * self.window_size
    idx = 0

    # Extract grid
    grid = obs[:, idx : idx + grid_size_flat]
    idx += grid_size_flat

    # Extract position
    position = obs[:, idx : idx + 2]
    idx += 2

    # Extract meters
    meters = obs[:, idx : idx + self.num_meters]
    idx += self.num_meters

    # Extract affordance
    affordance = obs[:, idx : idx + self.num_affordance_dims]
    idx += self.num_affordance_dims

    # Temporal features are ignored (or could be extracted)
```

**Compatibility Issues**:

1. **Hardcoded window_size² for grid** - Works for Grid2D, breaks for Grid3D (window_size³)
2. **Hardcoded position dim = 2** - Works for 2D, breaks for 3D (position dim = 3)

**Required Changes for POMDP Network**:

```python
class RecurrentSpatialQNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int,
        local_window_dim: int,      # ← NEW: window_size² or window_size³
        position_dim: int,          # ← NEW: 2 or 3
        num_meters: int,
        num_affordance_types: int,
        enable_temporal_features: bool,
        hidden_dim: int,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.local_window_dim = local_window_dim  # Replaces window_size²
        self.position_dim = position_dim          # Replaces hardcoded 2
        self.num_meters = num_meters
        # ... rest of init

    def forward(self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        # Split observation components with dynamic indices
        idx = 0

        # Extract local window (dimension varies by substrate)
        local_window = obs[:, idx : idx + self.local_window_dim]
        idx += self.local_window_dim

        # Extract position (dimension varies by substrate)
        position = obs[:, idx : idx + self.position_dim]
        idx += self.position_dim

        # Extract meters (substrate-independent)
        meters = obs[:, idx : idx + self.num_meters]
        idx += self.num_meters

        # Extract affordance (substrate-independent)
        affordance = obs[:, idx : idx + self.num_affordance_dims]
        idx += self.num_affordance_dims
```

**Effort**: 2 hours (network signature changes + testing)

---

## 8. Code Examples

### 8.1 Current vs Proposed - Full Observability

**Current** (`ObservationBuilder._build_full_observations()`):

```python
def _build_full_observations(self, positions, meters, affordances):
    # Grid encoding: hardcoded one-hot
    grid_encoding = torch.zeros(self.num_agents, self.grid_size * self.grid_size, device=self.device)

    # Mark affordance positions
    for affordance_pos in affordances.values():
        affordance_flat_idx = affordance_pos[1] * self.grid_size + affordance_pos[0]
        grid_encoding[:, affordance_flat_idx] = 1.0

    # Mark agent position
    flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
    grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), torch.ones(self.num_agents, 1, device=self.device))

    # Get affordance encoding
    affordance_encoding = self._build_affordance_encoding(positions, affordances)

    # Concatenate: grid + meters + affordance
    observations = torch.cat([grid_encoding, meters, affordance_encoding], dim=1)

    return observations
```

**Proposed** (Substrate-Based):

```python
def _build_full_observations(self, positions, meters, affordances):
    # Delegate position encoding to substrate
    position_encoding = self.substrate.encode_observation(
        agent_positions=positions,
        affordance_positions=affordances,
    )  # [num_agents, obs_dim] - varies by substrate!

    # Get affordance encoding (substrate-independent)
    affordance_encoding = self._build_affordance_encoding(positions, affordances)

    # Concatenate: position + meters + affordance
    observations = torch.cat([position_encoding, meters, affordance_encoding], dim=1)

    return observations
```

**Benefits**:

- No hardcoded grid_size references
- Works for Grid2D (one-hot or coords), Grid3D (coords), Aspatial (empty)
- Substrate handles affordance overlay (for one-hot) or ignores (for coords)

---

### 8.2 Current vs Proposed - POMDP

**Current** (`ObservationBuilder._build_partial_observations()`):

```python
def _build_partial_observations(self, positions, meters, affordances):
    window_size = 2 * self.vision_range + 1
    local_grids = []

    for agent_idx in range(self.num_agents):
        agent_pos = positions[agent_idx]
        local_grid = torch.zeros(window_size * window_size, device=self.device)

        # Manual window extraction (nested loops)
        for dy in range(-self.vision_range, self.vision_range + 1):
            for dx in range(-self.vision_range, self.vision_range + 1):
                world_x = agent_pos[0] + dx
                world_y = agent_pos[1] + dy

                # Boundary check
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    # Check affordances
                    has_affordance = False
                    for affordance_pos in affordances.values():
                        if affordance_pos[0] == world_x and affordance_pos[1] == world_y:
                            has_affordance = True
                            break

                    if has_affordance:
                        local_y = dy + self.vision_range
                        local_x = dx + self.vision_range
                        local_idx = local_y * window_size + local_x
                        local_grid[local_idx] = 1.0

        local_grids.append(local_grid)

    local_grids_batch = torch.stack(local_grids)

    # Normalize positions
    normalized_positions = positions.float() / (self.grid_size - 1)

    # Get affordance encoding
    affordance_encoding = self._build_affordance_encoding(positions, affordances)

    # Concatenate: local_grid + position + meters + affordance
    observations = torch.cat([local_grids_batch, normalized_positions, meters, affordance_encoding], dim=1)

    return observations
```

**Proposed** (Substrate-Based):

```python
def _build_partial_observations(self, positions, meters, affordances):
    # Delegate to substrate for window extraction and position encoding
    local_grids = []
    normalized_positions = []

    for agent_idx in range(self.num_agents):
        agent_pos = positions[agent_idx]

        # Substrate handles window extraction and normalization
        local_grid, normalized_pos = self.substrate.encode_partial_observation(
            agent_position=agent_pos,
            affordance_positions=affordances,
            vision_range=self.vision_range,
        )

        local_grids.append(local_grid)
        normalized_positions.append(normalized_pos)

    # Stack batches
    local_grids_batch = torch.stack(local_grids)
    normalized_positions_batch = torch.stack(normalized_positions)

    # Get affordance encoding (substrate-independent)
    affordance_encoding = self._build_affordance_encoding(positions, affordances)

    # Concatenate: local_grid + position + meters + affordance
    observations = torch.cat([local_grids_batch, normalized_positions_batch, meters, affordance_encoding], dim=1)

    return observations
```

**Benefits**:

- No manual window extraction logic
- No hardcoded grid_size or boundary checks
- Works for Grid2D (5×5), Grid3D (5×5×5 or 5×5+floor), Aspatial (empty)

---

### 8.3 Substrate Implementation Example - Grid2D

```python
class Grid2DSubstrate(SpatialSubstrate):
    def __init__(
        self,
        width: int,
        height: int,
        boundary: str = "clamp",
        distance_metric: str = "manhattan",
        position_encoding: str = "auto",
    ):
        self.width = width
        self.height = height
        self.boundary = boundary
        self.distance_metric = distance_metric

        # Auto-select encoding based on grid size
        if position_encoding == "auto":
            if width * height <= 64:  # 8×8 or smaller
                self.position_encoding = "onehot"
            else:
                self.position_encoding = "coords"
        else:
            self.position_encoding = position_encoding

    @property
    def position_dim(self) -> int:
        return 2  # (x, y)

    def get_observation_dim(self, partial_observability: bool, vision_range: int) -> int:
        """Get observation dimension for this substrate."""
        if partial_observability:
            window_size = 2 * vision_range + 1
            return window_size * window_size + 2  # Local grid + normalized (x, y)
        else:
            if self.position_encoding == "onehot":
                return self.width * self.height  # One-hot grid
            elif self.position_encoding == "coords":
                return 2  # Normalized (x, y)

    def encode_observation(
        self,
        agent_positions: torch.Tensor,
        affordance_positions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode agent positions (and optionally affordances) into observation."""
        num_agents = agent_positions.shape[0]
        device = agent_positions.device

        if self.position_encoding == "onehot":
            # One-hot grid: mark affordances first, then add agents
            grid_encoding = torch.zeros(num_agents, self.width * self.height, device=device)

            # Mark affordances (broadcast to all agents)
            if affordance_positions:
                for affordance_pos in affordance_positions.values():
                    flat_idx = affordance_pos[1] * self.width + affordance_pos[0]
                    grid_encoding[:, flat_idx] = 1.0

            # Mark agent positions (add 1.0, so agent on affordance = 2.0)
            flat_indices = agent_positions[:, 1] * self.width + agent_positions[:, 0]
            grid_encoding.scatter_add_(
                1,
                flat_indices.unsqueeze(1),
                torch.ones(num_agents, 1, device=device)
            )

            return grid_encoding

        elif self.position_encoding == "coords":
            # Coordinate encoding: just normalize agent positions
            # Affordances NOT encoded in position (handled by affordance encoding)
            normalized_x = agent_positions[:, 0].float() / (self.width - 1)
            normalized_y = agent_positions[:, 1].float() / (self.height - 1)
            return torch.stack([normalized_x, normalized_y], dim=1)

    def encode_partial_observation(
        self,
        agent_position: torch.Tensor,
        affordance_positions: dict[str, torch.Tensor],
        vision_range: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract 5×5 local window centered on agent."""
        window_size = 2 * vision_range + 1
        device = agent_position.device
        local_grid = torch.zeros(window_size * window_size, device=device)

        # Extract window
        for dy in range(-vision_range, vision_range + 1):
            for dx in range(-vision_range, vision_range + 1):
                world_x = agent_position[0] + dx
                world_y = agent_position[1] + dy

                # Boundary check
                if 0 <= world_x < self.width and 0 <= world_y < self.height:
                    # Check affordances
                    for affordance_pos in affordance_positions.values():
                        if affordance_pos[0] == world_x and affordance_pos[1] == world_y:
                            local_y = dy + vision_range
                            local_x = dx + vision_range
                            local_idx = local_y * window_size + local_x
                            local_grid[local_idx] = 1.0
                            break

        # Normalize position
        normalized_x = agent_position[0].float() / (self.width - 1)
        normalized_y = agent_position[1].float() / (self.height - 1)
        normalized_position = torch.tensor([normalized_x, normalized_y], device=device)

        return local_grid, normalized_position
```

---

## 9. Dependencies from Phase 4

Phase 5 requires Phase 4 to provide:

### 9.1 Substrate Instance in VectorizedHamletEnv

**Phase 4 Deliverable**:

```python
class VectorizedHamletEnv:
    def __init__(self, ...):
        # Phase 4: Load substrate from config
        self.substrate = load_substrate_from_config(config_pack_path / "substrate.yaml")

        # Phase 4: Initialize positions using substrate
        self.positions = self.substrate.initialize_positions(num_agents, device)
```

**Phase 5 Usage**:

```python
# In VectorizedHamletEnv.__init__()
# Calculate obs_dim using substrate
substrate_obs_dim = self.substrate.get_observation_dim(partial_observability, vision_range)
self.observation_dim = substrate_obs_dim + meter_count + (self.num_affordance_types + 1) + 4

# Pass substrate to ObservationBuilder
self.observation_builder = ObservationBuilder(
    num_agents=num_agents,
    substrate=self.substrate,  # ← From Phase 4
    device=device,
    partial_observability=partial_observability,
    vision_range=vision_range,
    enable_temporal_mechanics=enable_temporal_mechanics,
    num_affordance_types=self.num_affordance_types,
    affordance_names=self.affordance_names,
)
```

### 9.2 Substrate Methods Already Implemented in Phase 4

**From Phase 0-3**:

- `substrate.position_dim` - Number of position dimensions (2, 3, or 0)
- `substrate.initialize_positions(num_agents, device)` - Random starting positions
- `substrate.apply_movement(positions, deltas)` - Apply movement with boundaries
- `substrate.compute_distance(pos1, pos2)` - Distance metric

**Not Used by Phase 5**: Movement and distance are handled by environment, not observation builder.

### 9.3 New Methods Phase 5 Requires

**Phase 5 Must Add to Substrate Interface**:

1. `get_observation_dim(partial_observability, vision_range)` - Query obs dimension
2. `encode_observation(agent_positions, affordance_positions)` - Full obs encoding
3. `encode_partial_observation(agent_position, affordance_positions, vision_range)` - POMDP encoding

**Implementation Required in**:

- `src/townlet/substrate/base.py` - Abstract interface
- `src/townlet/substrate/grid2d.py` - Grid2D implementation
- `src/townlet/substrate/grid3d.py` - Grid3D implementation (if exists)
- `src/townlet/substrate/aspatial.py` - Aspatial implementation (if exists)

---

## 10. Risks and Mitigations

### 10.1 Risk: Observation Dimension Mismatch

**Symptom**: Network expects obs_dim=91, but substrate provides obs_dim=29

**Cause**: Changing from one-hot to coordinate encoding changes dimension

**Impact**: Runtime error when loading checkpoints or creating networks

**Mitigation**:

1. **Checkpoint validation**: Check obs_dim on load (already implemented in `population/vectorized.py:818`)
2. **Config-driven encoding**: Operators must specify encoding in substrate.yaml
3. **Auto-selection warning**: Log warning when auto-selecting encoding
4. **Test coverage**: Add tests for dimension consistency across substrates

**Probability**: Medium (operators changing configs)
**Severity**: High (breaks training)
**Mitigation Cost**: 2 hours (validation + logging)

---

### 10.2 Risk: POMDP Network Incompatibility with 3D

**Symptom**: RecurrentSpatialQNetwork expects 2-dim position, but Grid3D provides 3-dim

**Cause**: Hardcoded position_dim=2 in network architecture

**Impact**: Runtime error during forward pass

**Mitigation**:

1. **Update network signature**: Add `position_dim` parameter to RecurrentSpatialQNetwork
2. **Auto-detect from env**: `position_dim = env.substrate.position_dim`
3. **Vision encoder flexibility**: Use dynamic reshaping for local window (2D vs 3D)

**Probability**: High (if 3D POMDP is implemented)
**Severity**: High (blocks 3D POMDP)
**Mitigation Cost**: 2 hours (network signature + testing)

---

### 10.3 Risk: Aspatial Affordance Interaction Undefined

**Symptom**: Aspatial agents can't interact with affordances (no position)

**Cause**: Current interaction logic assumes `compute_distance(agent_pos, affordance_pos) == 0`

**Impact**: Aspatial environments unusable

**Mitigation**:

1. **Direct interaction model**: All affordances always available in aspatial
2. **Action masking**: INTERACT always valid (no movement required)
3. **Affordance encoding**: Current logic already works (distance check in `_build_affordance_encoding`)

**Probability**: High (aspatial is new paradigm)
**Severity**: Medium (blocks aspatial demos)
**Mitigation Cost**: 3 hours (interaction logic + action masking + testing)

---

### 10.4 Risk: Performance Regression from Looping

**Symptom**: POMDP observation construction takes 10× longer

**Cause**: `encode_partial_observation()` called per-agent (not vectorized)

**Impact**: Training slowdown for large num_agents

**Mitigation**:

1. **Vectorize later**: Start with per-agent loop (simpler)
2. **Benchmark**: Measure overhead with num_agents=100
3. **Optimize if needed**: Vectorize window extraction in Phase 6

**Probability**: Low (current loop works fine)
**Severity**: Low (only affects large populations)
**Mitigation Cost**: 0 hours (defer optimization)

---

## 11. Implementation Strategy

### 11.1 Phase 5 Task Breakdown

**Task 5.1: Add Substrate Methods** (4 hours)

1. Add abstract methods to `SpatialSubstrate` base class
2. Implement `get_observation_dim()` in Grid2D, Grid3D, Aspatial
3. Implement `encode_observation()` in Grid2D, Grid3D, Aspatial
4. Implement `encode_partial_observation()` in Grid2D, Grid3D, Aspatial
5. Write unit tests for each substrate

**Task 5.2: Update VectorizedHamletEnv** (2 hours)

1. Replace hardcoded obs_dim calculation with `substrate.get_observation_dim()`
2. Pass `substrate` to ObservationBuilder constructor
3. Remove `grid_size` parameter from ObservationBuilder
4. Update all call sites (runner.py, live_inference.py)

**Task 5.3: Refactor ObservationBuilder** (4 hours)

1. Update `__init__()` signature to accept `substrate`
2. Refactor `_build_full_observations()` to use `substrate.encode_observation()`
3. Refactor `_build_partial_observations()` to use `substrate.encode_partial_observation()`
4. Update tests to pass substrate instead of grid_size

**Task 5.4: Update RecurrentSpatialQNetwork** (2 hours)

1. Add `position_dim` parameter to constructor
2. Update forward() to use dynamic position extraction
3. Update call sites to pass `position_dim` from substrate
4. Test with 2D and 3D substrates

**Task 5.5: Testing and Validation** (4 hours)

1. Test Grid2D one-hot encoding (unchanged from current)
2. Test Grid2D coordinate encoding (new)
3. Test Grid3D coordinate encoding (new)
4. Test Aspatial empty encoding (new)
5. Test POMDP with Grid2D and Grid3D
6. Validate obs_dim matches across substrates

---

### 11.2 Testing Strategy

**Unit Tests** (Already Exist):

- `tests/test_townlet/unit/environment/test_observations.py` - Consolidated observation tests
- Update fixtures to pass substrate instead of grid_size
- Add new tests for Grid3D and Aspatial

**Integration Tests**:

- Test full training loop with Grid2D coords (vs one-hot baseline)
- Test POMDP with Grid3D (5×5×5 cube)
- Test Aspatial with direct affordance access

**Validation**:

- Compare obs_dim before/after for Grid2D one-hot (should be unchanged)
- Verify network can process all substrate encodings
- Check checkpoint loading fails gracefully on obs_dim mismatch

---

### 11.3 Rollout Plan

**Phase 5A: Grid2D One-Hot (Backward Compatible)**

1. Add substrate methods
2. Refactor ObservationBuilder to use substrate
3. Validate obs_dim unchanged for existing configs (8×8 one-hot)
4. Test L0, L0.5, L1 still work

**Phase 5B: Grid2D Coordinate Encoding**

1. Add auto-selection logic (one-hot for ≤8×8, coords for >8×8)
2. Create test config with coordinate encoding
3. Train agent and compare to one-hot baseline

**Phase 5C: Grid3D and Aspatial**

1. Implement Grid3D encoding (3-dim coords, 5×5×5 POMDP)
2. Implement Aspatial encoding (empty tensors)
3. Create demo configs for each

---

## 12. Effort Estimates

### 12.1 Task Breakdown

| Task | Description | Estimated Hours |
|------|-------------|----------------|
| 5.1 | Add substrate methods (get_observation_dim, encode_observation, encode_partial_observation) | 4 |
| 5.2 | Update VectorizedHamletEnv obs_dim calculation | 2 |
| 5.3 | Refactor ObservationBuilder to use substrate | 4 |
| 5.4 | Update RecurrentSpatialQNetwork for position_dim | 2 |
| 5.5 | Testing and validation (unit + integration) | 4 |
| **Total** | **Phase 5 Implementation** | **16** |

### 12.2 Risk Buffer

- **Edge cases** (aspatial interaction, 3D POMDP): +2 hours
- **Performance optimization** (vectorize window extraction): +2 hours (optional, defer to Phase 6)
- **Documentation** (update CLAUDE.md, research docs): +2 hours

**Total with Buffer**: 16 + 2 + 2 = **20 hours**

### 12.3 Critical Path

**Minimum Viable Phase 5** (12 hours):

1. Add substrate methods (4h)
2. Update VectorizedHamletEnv (2h)
3. Refactor ObservationBuilder full obs only (2h)
4. Basic testing (2h)
5. POMDP can be deferred to Phase 5B

---

## Conclusion

Phase 5 cleanly integrates substrate-based position encoding into observations by:

1. **Adding 3 substrate methods**: `get_observation_dim()`, `encode_observation()`, `encode_partial_observation()`
2. **Refactoring 2 files**: `VectorizedHamletEnv` (obs_dim calc), `ObservationBuilder` (encoding logic)
3. **Updating 1 network**: `RecurrentSpatialQNetwork` (dynamic position_dim)

**Key Benefits**:

- Removes all hardcoded grid_size references
- Enables 3D substrates (512+ dims avoided with coordinate encoding)
- Supports aspatial substrates (pure resource management)
- Preserves backward compatibility (Grid2D one-hot unchanged)

**Estimated Effort**: 16 hours implementation + 4 hours buffer = **20 hours total**

**Dependencies**: Phase 4 provides `substrate` instance, Phase 5 adds observation encoding methods

**Next Steps**: Implement Task 5.1 (Add substrate methods) with TDD approach
