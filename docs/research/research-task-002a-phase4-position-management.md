# Research: TASK-002A Phase 4 - Position Management Refactoring

**Date**: 2025-11-05
**Status**: Research Complete
**Phase**: Phase 4 of Configurable Spatial Substrates
**Context**: Phases 0-3 create substrate abstraction; Phase 4 integrates substrate into environment position management

---

## Executive Summary

Phase 4 must refactor all position management code to use the substrate abstraction created in Phases 0-3. This research identifies **9 integration points** across **5 core files** that require changes, with **moderate risk** due to checkpoint format changes.

**Key Finding**: Position tensors are hardcoded as `[num_agents, 2]` in 15+ locations. All must be changed to `[num_agents, substrate.position_dim]` to support 3D grids (position_dim=3) and aspatial universes (position_dim=0).

**Estimated Effort**: **12-16 hours** (was 6-8h in original plan, revised after thorough investigation)

**Critical Risk**: Checkpoint format changes require migration strategy for existing trained models.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Integration Points](#2-integration-points)
3. [Checkpoint Serialization Strategy](#3-checkpoint-serialization-strategy)
4. [Risks and Mitigations](#4-risks-and-mitigations)
5. [Dependencies from Phases 0-3](#5-dependencies-from-phases-0-3)
6. [Implementation Strategy](#6-implementation-strategy)
7. [Effort Estimates](#7-effort-estimates)

---

## 1. Current State Analysis

### 1.1 Position Initialization

**File**: `src/townlet/environment/vectorized_env.py`

**Current Implementation** (Lines 189, 197, 219):

```python
# __init__ (Line 189)
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)

# Temporal mechanics tracking (Line 197)
self.last_interaction_position = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)

# reset() (Line 219)
self.positions = torch.randint(0, self.grid_size, (self.num_agents, 2), device=self.device)
```

**Issues**:
- Hardcoded shape `[num_agents, 2]`
- Assumes 2D coordinates
- Cannot support 3D (would need `[num_agents, 3]`)
- Cannot support aspatial (should be `[num_agents, 0]`)

**Required Changes**:
```python
# __init__ (after substrate loaded)
self.positions = self.substrate.initialize_positions(self.num_agents, self.device)

# Temporal mechanics (only for spatial substrates)
if self.substrate.position_dim > 0:
    self.last_interaction_position = torch.zeros(
        (self.num_agents, self.substrate.position_dim),
        dtype=torch.long,
        device=self.device
    )
else:
    self.last_interaction_position = None  # Aspatial has no position tracking

# reset()
self.positions = self.substrate.initialize_positions(self.num_agents, self.device)
```

**Impact**:
- **3 initialization sites** need updating
- Temporal mechanics must check `substrate.position_dim > 0` before creating position tensors

---

### 1.2 Movement Logic

**File**: `src/townlet/environment/vectorized_env.py`

**Current Implementation** (Lines 388-409):

```python
def _execute_actions(self, actions: torch.Tensor) -> dict:
    """Execute movement, interaction, and wait actions."""
    old_positions = self.positions.clone() if self.enable_temporal_mechanics else None

    # Movement deltas (x, y) coordinates
    deltas = torch.tensor(
        [
            [0, -1],  # UP - decreases y
            [0, 1],   # DOWN - increases y
            [-1, 0],  # LEFT - decreases x
            [1, 0],   # RIGHT - increases x
            [0, 0],   # INTERACT (no movement)
            [0, 0],   # WAIT (no movement)
        ],
        device=self.device,
    )

    # Apply movement
    movement_deltas = deltas[actions]  # [num_agents, 2]
    new_positions = self.positions + movement_deltas

    # Clamp to grid boundaries
    new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)

    self.positions = new_positions
```

**Issues**:
- Hardcoded 2D movement deltas `[0, -1]`, `[-1, 0]`, etc.
- Hardcoded clamping assumes square grid boundaries
- Cannot support 3D movement (needs UP_FLOOR, DOWN_FLOOR actions)
- Cannot support wrap/bounce boundaries
- Cannot support aspatial (no movement concept)

**Required Changes**:
```python
def _execute_actions(self, actions: torch.Tensor) -> dict:
    """Execute movement, interaction, and wait actions."""
    # Store old positions for temporal mechanics (only if spatial)
    old_positions = None
    if self.enable_temporal_mechanics and self.substrate.position_dim > 0:
        old_positions = self.positions.clone()

    # Get movement deltas from action config (TASK-002B dependency)
    # For now, assume deltas are loaded from actions.yaml
    movement_deltas = self.action_deltas[actions]  # [num_agents, position_dim]

    # Apply movement using substrate (handles boundaries)
    new_positions = self.substrate.apply_movement(self.positions, movement_deltas)

    self.positions = new_positions

    # Reset temporal progress for agents that moved
    if self.enable_temporal_mechanics and old_positions is not None:
        for agent_idx in range(self.num_agents):
            if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
                self.interaction_progress[agent_idx] = 0
                self.last_interaction_affordance[agent_idx] = None
```

**Impact**:
- **1 movement application site** needs updating
- Movement deltas must come from actions.yaml (TASK-002B dependency)
- Boundary logic delegated to substrate
- Temporal mechanics position comparison still works (tensor equality)

---

### 1.3 Distance Calculations

**File**: `src/townlet/environment/vectorized_env.py`

**Current Implementation** (Lines 295-296, 470-471, 552-553):

```python
# Check if on affordance for interaction (3 locations)
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
on_this_affordance = distances == 0
```

**Issues**:
- Hardcoded Manhattan distance calculation
- Assumes coordinate subtraction works
- Cannot support graph substrates (distance is shortest path, not coordinate delta)
- Cannot support aspatial (distance meaningless)

**Required Changes**:
```python
# Use substrate's is_on_position() method (handles all substrate types)
on_this_affordance = self.substrate.is_on_position(self.positions, affordance_pos)
```

**File**: `src/townlet/environment/observation_builder.py`

**Current Implementation** (Lines 240-241):

```python
# Check which agents are on affordance
distances = torch.abs(positions - affordance_pos).sum(dim=1)
on_affordance = distances == 0
```

**Required Changes**:
```python
# Delegate to substrate
on_affordance = self.substrate.is_on_position(positions, affordance_pos)
```

**Impact**:
- **4 distance calculation sites** need updating (3 in vectorized_env.py, 1 in observation_builder.py)
- All use `substrate.is_on_position()` for consistency
- Substrate handles distance semantics (exact match for grid, proximity for continuous, always True for aspatial)

---

### 1.4 Position Encoding in Observations

**File**: `src/townlet/environment/observation_builder.py`

**Current Implementation** (Lines 127-137):

```python
def _build_full_observations(self, positions, meters, affordances):
    """Build full grid observations (Level 1)."""
    # Grid encoding: mark BOTH agent position AND affordance positions
    grid_encoding = torch.zeros(self.num_agents, self.grid_size * self.grid_size, device=self.device)

    # Mark affordance positions
    for affordance_pos in affordances.values():
        affordance_flat_idx = affordance_pos[1] * self.grid_size + affordance_pos[0]
        grid_encoding[:, affordance_flat_idx] = 1.0

    # Mark agent position (add 1.0, so overlaps become 2.0)
    flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
    grid_encoding.scatter_add_(1, flat_indices.unsqueeze(1), torch.ones(self.num_agents, 1, device=self.device))
```

**Issues**:
- Hardcoded grid size for observation dimension
- Hardcoded coordinate-to-index conversion (`y * grid_size + x`)
- Cannot support 3D grids (needs different indexing)
- Cannot support aspatial (no position encoding needed)

**Required Changes**:
```python
def _build_full_observations(self, positions, meters, affordances):
    """Build full observations using substrate encoding."""
    # Delegate position encoding to substrate
    grid_encoding = self.substrate.encode_observation(positions, affordances)

    # Rest of observation building continues...
```

**Partial Observability** (Lines 166-207):

Current implementation extracts local 5×5 window manually. After substrate integration:

```python
def _build_partial_observations(self, positions, meters, affordances):
    """Build partial observations (POMDP) using substrate encoding."""
    # For partial observability, substrate still provides local window encoding
    # But the window extraction logic moves to Grid2DSubstrate.encode_observation()

    # Normalized positions (for recurrent network position encoder)
    normalized_positions = positions.float() / (self.substrate.width - 1, self.substrate.height - 1)

    # Local window encoding from substrate
    local_grids = self.substrate.encode_partial_observation(
        positions, affordances, vision_range=self.vision_range
    )

    # Rest of observation building...
```

**Impact**:
- **2 observation encoding sites** need updating (full observability, partial observability)
- Substrate handles all position encoding logic
- Normalized position calculation must use substrate dimensions
- **NEW METHOD REQUIRED**: `substrate.encode_partial_observation()` for POMDP

---

### 1.5 Affordance Position Management

**File**: `src/townlet/environment/vectorized_env.py`

**Current Implementation** (Lines 646-671):

```python
def randomize_affordance_positions(self):
    """Randomize affordance positions for generalization testing.

    Ensures no two affordances occupy the same position.
    """
    import random

    # Validate that grid has enough cells for all affordances
    num_affordances = len(self.affordances)
    total_cells = self.grid_size * self.grid_size
    if num_affordances >= total_cells:
        raise ValueError(
            f"Grid has {total_cells} cells but {num_affordances} affordances + 1 agent need space."
        )

    # Generate list of all grid positions
    all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

    # Shuffle and assign to affordances
    random.shuffle(all_positions)

    for i, affordance_name in enumerate(self.affordances.keys()):
        new_pos = all_positions[i]
        self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
```

**Issues**:
- Hardcoded 2D position generation `[(x, y) for ...]`
- Hardcoded capacity check `grid_size * grid_size`
- Cannot support 3D grids (needs `[(x, y, z) for ...]`)
- Cannot support aspatial (affordances have no position)

**Required Changes**:
```python
def randomize_affordance_positions(self):
    """Randomize affordance positions using substrate.

    Ensures no two affordances occupy the same position.
    """
    import random

    # Aspatial substrates: no position randomization needed
    if self.substrate.position_dim == 0:
        # Clear positions (affordances have no spatial location)
        for affordance_name in self.affordances.keys():
            self.affordances[affordance_name] = torch.zeros(0, dtype=torch.long, device=self.device)
        return

    # Get all valid positions from substrate
    all_positions = self.substrate.get_all_positions()  # NEW METHOD NEEDED

    # Validate capacity
    num_affordances = len(self.affordances)
    total_cells = len(all_positions)
    if num_affordances >= total_cells:
        raise ValueError(
            f"Substrate has {total_cells} positions but {num_affordances} affordances + 1 agent need space."
        )

    # Shuffle and assign
    random.shuffle(all_positions)

    for i, affordance_name in enumerate(self.affordances.keys()):
        new_pos = all_positions[i]
        self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
```

**Impact**:
- **1 affordance randomization site** needs updating
- **NEW METHOD REQUIRED**: `substrate.get_all_positions() -> list[torch.Tensor]`
  - Grid2D: Returns `[(x, y) for x in range(width) for y in range(height)]`
  - Grid3D: Returns `[(x, y, z) for x in ... for y in ... for z in ...]`
  - Aspatial: Returns `[]` (no positions exist)

---

### 1.6 Checkpoint Serialization

**File**: `src/townlet/environment/vectorized_env.py`

**Current Implementation** (Lines 600-618):

```python
def get_affordance_positions(self) -> dict:
    """Get current affordance positions (P1.1 checkpointing).

    Returns:
        Dictionary with 'positions' and 'ordering' keys:
        - 'positions': Dict mapping affordance names to [x, y] positions
        - 'ordering': List of affordance names in consistent order
    """
    positions = {}
    for name, pos_tensor in self.affordances.items():
        # Convert tensor position to list (for JSON serialization)
        pos = pos_tensor.cpu().tolist()
        positions[name] = [int(pos[0]), int(pos[1])]  # Assumes 2D!

    return {
        "positions": positions,
        "ordering": self.affordance_names,
    }
```

**Issues**:
- Hardcoded conversion assumes 2D `[x, y]`
- Cannot serialize 3D positions `[x, y, z]`
- Cannot serialize aspatial positions `[]`

**Required Changes**:
```python
def get_affordance_positions(self) -> dict:
    """Get current affordance positions (substrate-agnostic checkpointing).

    Returns:
        Dictionary with 'positions', 'ordering', and 'position_dim' keys:
        - 'positions': Dict mapping affordance names to position lists
        - 'ordering': List of affordance names in consistent order
        - 'position_dim': Dimensionality for validation (0=aspatial, 2=2D, 3=3D)
    """
    positions = {}
    for name, pos_tensor in self.affordances.items():
        # Convert tensor to list (handles any dimensionality)
        pos = pos_tensor.cpu().tolist()

        # Ensure pos is a list (even for 0-dimensional positions)
        if isinstance(pos, int):
            pos = [pos]
        elif self.substrate.position_dim == 0:
            pos = []

        positions[name] = [int(x) for x in pos]

    return {
        "positions": positions,
        "ordering": self.affordance_names,
        "position_dim": self.substrate.position_dim,  # For validation
    }
```

**File**: `src/townlet/environment/vectorized_env.py`

**Current Implementation** (Lines 620-644):

```python
def set_affordance_positions(self, checkpoint_data: dict) -> None:
    """Set affordance positions from checkpoint (P1.1 checkpointing)."""
    # Handle backwards compatibility: old format is just positions dict
    if "positions" in checkpoint_data:
        positions = checkpoint_data["positions"]
        ordering = checkpoint_data.get("ordering", self.affordance_names)
    else:
        positions = checkpoint_data
        ordering = self.affordance_names

    # Restore ordering
    self.affordance_names = ordering
    self.num_affordance_types = len(self.affordance_names)

    # Rebuild affordances dict
    for name, pos in positions.items():
        if name in self.affordances:
            self.affordances[name] = torch.tensor(pos, device=self.device, dtype=torch.long)
```

**Required Changes**:
```python
def set_affordance_positions(self, checkpoint_data: dict) -> None:
    """Set affordance positions from checkpoint (backward compatible).

    Handles checkpoints from:
    - Phase 4+ (substrate-aware): position_dim field present
    - Legacy (pre-Phase 4): assumes 2D positions
    """
    # Extract position_dim for validation
    checkpoint_position_dim = checkpoint_data.get("position_dim", 2)  # Default to 2D for legacy

    # Validate compatibility
    if checkpoint_position_dim != self.substrate.position_dim:
        raise ValueError(
            f"Checkpoint position_dim mismatch: checkpoint has {checkpoint_position_dim}D positions, "
            f"but current substrate requires {self.substrate.position_dim}D positions. "
            f"Cannot load checkpoint from {'aspatial' if checkpoint_position_dim == 0 else str(checkpoint_position_dim) + 'D'} "
            f"substrate into {'aspatial' if self.substrate.position_dim == 0 else str(self.substrate.position_dim) + 'D'} substrate."
        )

    # Handle backwards compatibility: old format is just positions dict
    if "positions" in checkpoint_data:
        positions = checkpoint_data["positions"]
        ordering = checkpoint_data.get("ordering", self.affordance_names)
    else:
        positions = checkpoint_data
        ordering = self.affordance_names

    # Restore ordering
    self.affordance_names = ordering
    self.num_affordance_types = len(self.affordance_names)

    # Rebuild affordances dict (handles any dimensionality)
    for name, pos in positions.items():
        if name in self.affordances:
            self.affordances[name] = torch.tensor(pos, device=self.device, dtype=torch.long)
```

**Impact**:
- **2 checkpoint serialization sites** need updating (get/set)
- Add `position_dim` field to checkpoint format (for validation)
- Backward compatibility: assume 2D if `position_dim` missing
- **BREAKING**: Cannot load 2D checkpoint into 3D environment (validation error)

---

### 1.7 Visualization (Live Inference Server)

**File**: `src/townlet/demo/live_inference.py`

**Current Implementation** (Lines 664-665, 742-754):

```python
# Get agent position (unpack for frontend compatibility)
agent_pos = self.env.positions[0].cpu().tolist()

# ...

"grid": {
    "width": self.env.grid_size,
    "height": self.env.grid_size,
    "agents": [
        {
            "id": "agent_0",
            "x": agent_pos[0],  # Assumes 2D!
            "y": agent_pos[1],  # Assumes 2D!
            "color": "#4CAF50",
            "last_action": last_action,
        }
    ],
    "affordances": affordances,
},
```

**Issues**:
- Hardcoded unpacking assumes `[x, y]`
- Frontend (Grid.vue) assumes 2D SVG rendering
- Cannot render 3D positions (needs floor selection UI)
- Cannot render aspatial (no spatial visualization)

**Required Changes**:
```python
# Get agent position (substrate-agnostic)
agent_pos = self.env.positions[0].cpu().tolist()

# Build grid data based on substrate type
if isinstance(self.env.substrate, Grid2DSubstrate):
    grid_data = {
        "type": "grid2d",
        "width": self.env.substrate.width,
        "height": self.env.substrate.height,
        "agents": [
            {
                "id": "agent_0",
                "x": agent_pos[0],
                "y": agent_pos[1],
                "color": "#4CAF50",
                "last_action": last_action,
            }
        ],
        "affordances": affordances,
    }
elif isinstance(self.env.substrate, AspatialSubstrate):
    grid_data = {
        "type": "aspatial",
        # No position data, frontend shows meters-only dashboard
    }
else:
    # Future: Grid3DSubstrate, etc.
    grid_data = {
        "type": "unknown",
        "substrate_type": type(self.env.substrate).__name__,
    }
```

**Impact**:
- **1 visualization site** needs updating
- Frontend (Grid.vue) must detect substrate type and route to appropriate renderer
- 2D rendering unchanged
- Aspatial rendering shows meters-only (no grid)
- 3D rendering deferred to future work (can show text "3D rendering not implemented")

---

### 1.8 Recording System

**File**: `src/townlet/recording/recorder.py`

**Current Implementation** (Lines 71-73, 122-123):

```python
def record_step(
    self,
    positions: torch.Tensor,  # [2] Agent (x, y) position
    meters: torch.Tensor,
    action: int,
    # ...
):
    # ...
    position=(int(positions[0].item()), int(positions[1].item())),  # Assumes 2D!
```

**Issues**:
- Hardcoded tuple unpacking assumes 2D
- Recording format assumes `(x, y)` tuple

**Required Changes**:
```python
def record_step(
    self,
    positions: torch.Tensor,  # [position_dim] Agent position
    meters: torch.Tensor,
    action: int,
    # ...
):
    # Convert position to tuple (handles any dimensionality)
    position_tuple = tuple(int(positions[i].item()) for i in range(positions.shape[0]))

    # For backward compatibility, store position_dim in metadata
    # ...
```

**File**: `src/townlet/recording/data_structures.py`

**Current Implementation** (Lines 42-43, 96):

```python
# Convert affordance positions from lists to tuples
data["affordance_layout"] = {name: tuple(pos) for name, pos in data["affordance_layout"].items()}

# ...

affordance_layout: dict[str, tuple[int, int]]  # name → (x, y) - Assumes 2D!
```

**Required Changes**:
```python
# Position tuples now have variable length
affordance_layout: dict[str, tuple[int, ...]]  # name → (x, y) or (x, y, z) or ()

# Conversion still works (tuple() handles any length)
data["affordance_layout"] = {name: tuple(pos) for name, pos in data["affordance_layout"].items()}
```

**Impact**:
- **2 recording sites** need updating (recorder.py, data_structures.py)
- Recording format changes: positions are variable-length tuples
- Backward compatibility: old recordings have `(x, y)`, new recordings have `(x, y)`, `(x, y, z)`, or `()`
- **BREAKING**: Replay system must detect position_dim from recording metadata

---

### 1.9 Tests

**Files**: Multiple test files use hardcoded positions

**Property Tests** (`tests/test_townlet/properties/test_environment_properties.py`):

Lines 66-70:
```python
# PROPERTY: Positions always in bounds
positions = env.positions
assert torch.all(positions[:, 0] >= 0), f"X position {positions[0, 0]} < 0"
assert torch.all(positions[:, 0] < grid_size), f"X position {positions[0, 0]} >= {grid_size}"
assert torch.all(positions[:, 1] >= 0), f"Y position {positions[0, 1]} < 0"
assert torch.all(positions[:, 1] < grid_size), f"Y position {positions[0, 1]} >= {grid_size}"
```

**Required Changes**:
```python
# PROPERTY: Positions always in bounds (substrate-agnostic)
positions = env.positions

# For grid substrates, check bounds
if hasattr(env.substrate, 'width'):
    assert torch.all(positions[:, 0] >= 0), f"X position {positions[0, 0]} < 0"
    assert torch.all(positions[:, 0] < env.substrate.width), f"X position {positions[0, 0]} >= {env.substrate.width}"
    assert torch.all(positions[:, 1] >= 0), f"Y position {positions[0, 1]} < 0"
    assert torch.all(positions[:, 1] < env.substrate.height), f"Y position {positions[0, 1]} >= {env.substrate.height}"

# For aspatial substrates, positions should be empty
elif env.substrate.position_dim == 0:
    assert positions.shape[1] == 0, "Aspatial substrate should have 0-dimensional positions"
```

**Checkpoint Tests** (`tests/test_townlet/integration/test_checkpointing.py`):

Lines 71, 94, 129, 157, 162:
- Multiple assertions check `env.positions` shape and content
- All assume 2D positions

**Required Changes**:
- Update all tests to use `substrate.position_dim` instead of hardcoded 2
- Add parameterized tests for 2D vs aspatial substrates

**Impact**:
- **~10-15 test assertion sites** need updating across multiple files
- Add new tests for 3D and aspatial substrates
- Parameterize existing tests to run with different substrate types

---

## 2. Integration Points Summary

| # | Location | File | Lines | Description | Effort |
|---|----------|------|-------|-------------|--------|
| 1 | Position Initialization | `vectorized_env.py` | 189, 197, 219 | Replace `torch.zeros((N, 2))` with `substrate.initialize_positions()` | 2h |
| 2 | Movement Logic | `vectorized_env.py` | 388-409 | Replace hardcoded deltas with `substrate.apply_movement()` | 3h |
| 3 | Distance Calculations | `vectorized_env.py` | 295, 470, 552 | Replace Manhattan distance with `substrate.is_on_position()` | 2h |
| 4 | Distance (Observation) | `observation_builder.py` | 240-241 | Replace Manhattan distance with `substrate.is_on_position()` | 1h |
| 5 | Full Observations | `observation_builder.py` | 127-137 | Replace grid encoding with `substrate.encode_observation()` | 2h |
| 6 | Partial Observations | `observation_builder.py` | 166-207 | Replace local window with `substrate.encode_partial_observation()` | 3h |
| 7 | Affordance Randomization | `vectorized_env.py` | 646-671 | Replace 2D generation with `substrate.get_all_positions()` | 2h |
| 8 | Checkpoint Serialization | `vectorized_env.py` | 600-618, 620-644 | Add `position_dim` field, validate compatibility | 3h |
| 9 | Visualization | `live_inference.py` | 664-665, 742-754 | Detect substrate type, route to renderer | 2h |
| 10 | Recording System | `recorder.py`, `data_structures.py` | Various | Variable-length position tuples | 2h |
| 11 | Tests | Multiple files | Various | Update assertions, add parameterized tests | 4h |

**Total**: **26 hours** (was 12-16h in plan, revised after detailed investigation)

---

## 3. Checkpoint Serialization Strategy

### 3.1 Checkpoint Format Changes

**Current Format** (version 2):
```python
{
    "version": 2,
    "episode": 1234,
    "timestamp": 1699123456.789,
    "affordance_layout": {
        "positions": {
            "Bed": [2, 3],       # Always [x, y]
            "Hospital": [5, 7],
        },
        "ordering": ["Bed", "Hospital", ...]
    },
    "population_state": { ... },
    "curriculum_state": { ... },
}
```

**New Format** (version 3, Phase 4):
```python
{
    "version": 3,  # Increment for substrate support
    "episode": 1234,
    "timestamp": 1699123456.789,
    "substrate_metadata": {
        "type": "grid",  # or "aspatial"
        "position_dim": 2,  # 0 for aspatial, 2 for 2D, 3 for 3D
        "grid_width": 8,    # Only present for grid substrates
        "grid_height": 8,
    },
    "affordance_layout": {
        "positions": {
            "Bed": [2, 3],       # 2D: [x, y]
            "Hospital": [5, 7],  # 3D: [x, y, z]
                                 # Aspatial: []
        },
        "ordering": ["Bed", "Hospital", ...],
        "position_dim": 2,  # Redundant with substrate_metadata, but safer
    },
    "population_state": { ... },
    "curriculum_state": { ... },
}
```

### 3.2 Backward Compatibility Strategy

**Loading Old Checkpoints** (version 2):

```python
def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint with substrate migration."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Check version
    version = checkpoint.get("version", 1)

    if version < 3:
        # Legacy checkpoint: assume 2D grid substrate
        logger.warning(
            f"Loading legacy checkpoint (version {version}). "
            f"Assuming 2D grid substrate. "
            f"If current environment uses different substrate, loading will fail."
        )

        # Add substrate_metadata for validation
        if "substrate_metadata" not in checkpoint:
            checkpoint["substrate_metadata"] = {
                "type": "grid",
                "position_dim": 2,
                # Grid size unknown, will be inferred from positions
            }

        # Add position_dim to affordance_layout
        if "affordance_layout" in checkpoint:
            if "position_dim" not in checkpoint["affordance_layout"]:
                checkpoint["affordance_layout"]["position_dim"] = 2

    return checkpoint
```

**Validation During Load**:

```python
def validate_checkpoint_substrate_compatibility(checkpoint: dict, current_substrate: SpatialSubstrate):
    """Validate checkpoint can be loaded into current substrate."""
    metadata = checkpoint.get("substrate_metadata", {})

    checkpoint_position_dim = metadata.get("position_dim", 2)  # Default 2D

    if checkpoint_position_dim != current_substrate.position_dim:
        raise ValueError(
            f"Checkpoint incompatible with current substrate:\n"
            f"  Checkpoint: {checkpoint_position_dim}D "
            f"({'aspatial' if checkpoint_position_dim == 0 else 'grid'})\n"
            f"  Current: {current_substrate.position_dim}D "
            f"({'aspatial' if current_substrate.position_dim == 0 else 'grid'})\n"
            f"\n"
            f"Cannot load checkpoint from different substrate dimensionality.\n"
            f"Options:\n"
            f"  1. Use checkpoint from matching substrate type\n"
            f"  2. Train new model from scratch\n"
            f"  3. Use checkpoint migration tool (if available)"
        )

    # For grid substrates, validate grid size matches
    if hasattr(current_substrate, 'width'):
        checkpoint_width = metadata.get("grid_width")
        if checkpoint_width is not None and checkpoint_width != current_substrate.width:
            logger.warning(
                f"Grid size mismatch: checkpoint has {checkpoint_width}×{metadata.get('grid_height')}, "
                f"current environment has {current_substrate.width}×{current_substrate.height}. "
                f"Positions may be invalid."
            )
```

### 3.3 Migration Tool (Future Work)

**Not implemented in Phase 4**, but documented for future:

```bash
# Hypothetical migration tool
python -m townlet.tools.migrate_checkpoint \
    --input checkpoints_level2/checkpoint_ep01000.pt \
    --output checkpoints_level2_3d/checkpoint_ep01000_migrated.pt \
    --source-substrate 2d \
    --target-substrate 3d \
    --floor 0  # Place all agents/affordances on floor 0
```

This tool would:
1. Load checkpoint with old substrate format
2. Transform positions: `[x, y]` → `[x, y, 0]` (add Z=0 dimension)
3. Update metadata: `position_dim: 2` → `position_dim: 3`
4. Save as version 3 checkpoint

**Defer to future work** (not critical for Phase 4).

---

## 4. Risks and Mitigations

### Risk 1: Checkpoint Incompatibility

**Risk**: Existing trained models (Level 2 POMDP, Level 3 Temporal) use version 2 checkpoints. Loading fails after Phase 4 if validation is too strict.

**Impact**: HIGH - Cannot resume training, lose weeks of compute time

**Mitigation**:
1. **Backward compatibility mode**: Assume 2D grid if `substrate_metadata` missing
2. **Validation warnings**: Warn but don't fail if grid size mismatches
3. **Test loading**: Verify checkpoint loading with existing checkpoints before merging Phase 4
4. **Checkpoint conversion script**: Convert version 2 → version 3 offline before Phase 4 deployment

**Implementation**:
```python
# In runner.py load_checkpoint()
try:
    validate_checkpoint_substrate_compatibility(checkpoint, self.env.substrate)
except ValueError as e:
    # Allow legacy checkpoints with warning
    if checkpoint.get("version", 1) < 3:
        logger.warning(f"Legacy checkpoint compatibility issue (ignoring): {e}")
    else:
        raise
```

**Status**: MITIGATED

---

### Risk 2: Network Architecture Mismatch

**Risk**: Observation dimension changes with substrate, but Q-network input dim is fixed. Loading checkpoint into different substrate fails at network level.

**Impact**: HIGH - Network forward pass fails with shape mismatch

**Mitigation**:
1. **Observation dim validation**: Check `observation_dim` in checkpoint matches current environment
2. **Network metadata**: Store `observation_dim` in checkpoint for validation
3. **Fail-fast**: Detect mismatch during checkpoint load, not during first forward pass

**Implementation**:
```python
# In population.get_checkpoint_state()
checkpoint["observation_dim"] = self.obs_dim

# In population.load_checkpoint_state()
checkpoint_obs_dim = checkpoint.get("observation_dim")
if checkpoint_obs_dim != self.obs_dim:
    raise ValueError(
        f"Observation dimension mismatch: checkpoint has {checkpoint_obs_dim}, "
        f"current environment has {self.obs_dim}. "
        f"This indicates substrate or network architecture change."
    )
```

**Status**: MITIGATED

---

### Risk 3: Temporal Mechanics with Aspatial

**Risk**: Temporal mechanics code assumes positions exist (`self.last_interaction_position`). Aspatial substrates have no positions.

**Impact**: MEDIUM - Crashes when temporal mechanics enabled with aspatial substrate

**Mitigation**:
1. **Guard checks**: Only create `last_interaction_position` if `substrate.position_dim > 0`
2. **Validation**: Raise error if temporal mechanics enabled with aspatial substrate (invalid configuration)
3. **Documentation**: Clarify that temporal mechanics require spatial substrate

**Implementation**:
```python
# In vectorized_env.__init__()
if self.enable_temporal_mechanics and self.substrate.position_dim == 0:
    raise ValueError(
        "Temporal mechanics require spatial substrate (position_dim > 0). "
        "Cannot enable temporal mechanics with aspatial substrate. "
        "Set enable_temporal_mechanics=False or use grid substrate."
    )

if self.enable_temporal_mechanics:
    self.last_interaction_position = torch.zeros(
        (self.num_agents, self.substrate.position_dim),
        dtype=torch.long,
        device=self.device,
    )
```

**Status**: MITIGATED

---

### Risk 4: Frontend Rendering Assumptions

**Risk**: Frontend (Grid.vue) assumes 2D SVG grid rendering. Sending 3D or aspatial data crashes frontend.

**Impact**: MEDIUM - Visualization breaks, but training unaffected

**Mitigation**:
1. **Substrate type field**: Send `substrate.type` in WebSocket messages
2. **Graceful degradation**: Frontend shows "Unsupported substrate" message for 3D/aspatial
3. **Phase 7 work**: Proper rendering for all substrate types (deferred)

**Implementation**:
```python
# In live_inference.py _broadcast_state_update()
update = {
    "type": "state_update",
    "substrate": {
        "type": self.env.substrate.type,  # "grid2d", "grid3d", "aspatial"
        "position_dim": self.env.substrate.position_dim,
    },
    "grid": grid_data,  # Format depends on substrate type
    # ...
}
```

**Frontend** (Grid.vue):
```vue
<template>
  <div v-if="substrate.type === 'grid2d'">
    <!-- Existing 2D SVG rendering -->
  </div>
  <div v-else>
    <p>Substrate type "{{ substrate.type }}" rendering not yet implemented.</p>
    <p>Meters and training metrics still available.</p>
  </div>
</template>
```

**Status**: MITIGATED (graceful degradation)

---

### Risk 5: Test Suite Fragility

**Risk**: Many tests hardcode position shape `[N, 2]`. All break after Phase 4.

**Impact**: LOW - Tests fail, but easy to fix

**Mitigation**:
1. **Parameterized tests**: Run tests with 2D and aspatial substrates
2. **Substrate fixtures**: Create test fixtures for different substrate types
3. **Incremental fixes**: Fix tests incrementally during Phase 4 implementation

**Implementation**:
```python
# In conftest.py
@pytest.fixture(params=["grid2d", "aspatial"])
def substrate_type(request):
    """Parameterize tests across substrate types."""
    return request.param

@pytest.fixture
def env_with_substrate(substrate_type):
    """Create environment with specified substrate."""
    if substrate_type == "grid2d":
        config_pack = "configs/test"
    elif substrate_type == "aspatial":
        config_pack = "configs/test_aspatial"

    return VectorizedHamletEnv(config_pack_path=Path(config_pack), num_agents=1, device="cpu")
```

**Status**: MITIGATED (test infrastructure ready)

---

## 5. Dependencies from Phases 0-3

Phase 4 depends on successful completion of Phases 0-3:

### Phase 0: Research Validation
- ✅ Identify all hardcoded position assumptions (THIS DOCUMENT)
- ✅ Validate substrate abstraction design

### Phase 1: Substrate Abstraction Layer
- ✅ `SpatialSubstrate` abstract interface defined
- ✅ `Grid2DSubstrate` implemented (replicates current behavior)
- ✅ `AspatialSubstrate` implemented
- **Required for Phase 4**: All substrate methods available

### Phase 2: Substrate Configuration Schema
- ✅ `SubstrateConfig` Pydantic DTOs defined
- ✅ `SubstrateFactory.build()` creates substrate from config
- ✅ `load_substrate_config()` loads YAML
- **Required for Phase 4**: Environment can load substrate from config

### Phase 3: Environment Integration (Substrate Loading)
- ✅ `VectorizedHamletEnv` loads `substrate.yaml`
- ✅ Backward compatibility if `substrate.yaml` missing
- ✅ `self.substrate` attribute populated
- **Required for Phase 4**: `env.substrate` available in all Phase 4 integration points

### New Methods Needed for Phase 4

Phase 4 requires **2 new methods** to be added to `SpatialSubstrate` interface:

1. **`get_all_positions() -> list[torch.Tensor]`**
   - Purpose: Return all valid positions in substrate (for affordance randomization)
   - Grid2D: Returns `[(x, y) for x in range(width) for y in range(height)]`
   - Aspatial: Returns `[]`

2. **`encode_partial_observation(positions, affordances, vision_range) -> torch.Tensor`**
   - Purpose: Encode local window for POMDP (partial observability)
   - Grid2D: Extract 5×5 window centered on agent
   - Aspatial: Return empty tensor `[num_agents, 0]`

**These must be added during Phase 1 or early Phase 4**.

---

## 6. Implementation Strategy

### 6.1 Implementation Order (Sequential)

**Rationale**: Position management is tightly coupled. Partial changes break environment. Must implement atomically.

**Order**:

1. **Add new substrate methods** (2h)
   - `get_all_positions()`
   - `encode_partial_observation()`
   - Update `SpatialSubstrate` interface
   - Implement in `Grid2DSubstrate` and `AspatialSubstrate`

2. **Position initialization** (2h)
   - Replace `torch.zeros((N, 2))` in `__init__`
   - Replace `torch.randint(0, grid_size, (N, 2))` in `reset()`
   - Update `last_interaction_position` (temporal mechanics guard)

3. **Movement logic** (3h)
   - Replace hardcoded deltas with `substrate.apply_movement()`
   - Remove hardcoded boundary clamping
   - Update temporal mechanics position comparison

4. **Distance calculations** (3h)
   - Replace `torch.abs(...).sum(dim=1) == 0` with `substrate.is_on_position()`
   - Update 3 sites in `vectorized_env.py`
   - Update 1 site in `observation_builder.py`

5. **Observation encoding** (5h)
   - Replace grid encoding with `substrate.encode_observation()`
   - Update full observability
   - Update partial observability (use `encode_partial_observation()`)
   - Handle aspatial (empty position encoding)

6. **Affordance randomization** (2h)
   - Replace 2D position generation with `substrate.get_all_positions()`
   - Add aspatial guard (skip randomization)

7. **Checkpoint serialization** (3h)
   - Add `position_dim` field to affordance layout
   - Add `substrate_metadata` to checkpoint
   - Implement backward compatibility (assume 2D if missing)
   - Add validation during load

8. **Visualization** (2h)
   - Detect substrate type
   - Route 2D to existing renderer
   - Show "not implemented" for aspatial
   - Send substrate metadata to frontend

9. **Recording system** (2h)
   - Update position tuple conversion (variable length)
   - Update type hints (`tuple[int, int]` → `tuple[int, ...]`)

10. **Test updates** (4h)
    - Fix hardcoded `[N, 2]` assertions
    - Add parameterized tests for 2D vs aspatial
    - Update checkpoint tests
    - Add new substrate-specific tests

**Total: 28 hours**

### 6.2 Testing Strategy

**Unit Tests**:
- Test each integration point independently
- Parameterize across substrate types (2D, aspatial)
- Mock substrate for isolated testing

**Integration Tests**:
- Full training run with 2D grid (existing behavior)
- Full training run with aspatial substrate (new)
- Checkpoint save/load cycle with both substrate types
- Visualization with both substrate types

**Property Tests**:
- Positions always in bounds (for grid substrates)
- Positions always correct shape `[N, substrate.position_dim]`
- Checkpoint round-trip preserves positions exactly

**Regression Tests**:
- Load existing Level 2 POMDP checkpoint (version 2)
- Verify backward compatibility warnings
- Verify training continues correctly

---

## 7. Effort Estimates

### 7.1 Detailed Breakdown

| Task | Subtasks | Effort | Risk | Complexity |
|------|----------|--------|------|------------|
| **1. New Substrate Methods** | Interface update, Grid2D impl, Aspatial impl, tests | 2h | LOW | LOW |
| **2. Position Initialization** | 3 init sites, temporal guard, tests | 2h | LOW | LOW |
| **3. Movement Logic** | Remove deltas, substrate.apply_movement(), temporal, tests | 3h | MEDIUM | MEDIUM |
| **4. Distance Calculations** | 4 sites, substrate.is_on_position(), tests | 3h | LOW | LOW |
| **5. Observation Encoding** | Full obs, partial obs, aspatial, tests | 5h | HIGH | HIGH |
| **6. Affordance Randomization** | get_all_positions(), aspatial guard, tests | 2h | LOW | LOW |
| **7. Checkpoint Serialization** | Format update, validation, backward compat, tests | 3h | HIGH | MEDIUM |
| **8. Visualization** | Substrate routing, graceful degradation, tests | 2h | MEDIUM | MEDIUM |
| **9. Recording System** | Variable tuples, type hints, tests | 2h | LOW | LOW |
| **10. Test Updates** | Fix assertions, parameterize, add new tests | 4h | LOW | MEDIUM |
| **Contingency** | Unexpected issues, debugging, iteration | 4h | - | - |

**Total: 32 hours**

### 7.2 Critical Path

1. Phases 0-3 completion (dependency)
2. New substrate methods (blocks all position work)
3. Position initialization (blocks movement, observations)
4. Observation encoding (blocks training runs)
5. Checkpoint serialization (blocks training continuity)
6. Test updates (blocks validation)

**Critical path: 18 hours** (items 2-6)

### 7.3 Revised Estimates

**Original Plan Estimate**: 6-8 hours (Phase 3 in implementation plan)

**This Research Estimate**: 32 hours

**Difference**: +400% (24 hours underestimate)

**Reasons for Revision**:
1. **Observation encoding complexity**: Partial observability requires new substrate method (5h vs 2h)
2. **Checkpoint format changes**: Backward compatibility is complex (3h vs 1h)
3. **Test updates**: More test sites than anticipated (4h vs 1h)
4. **Visualization routing**: Frontend changes needed (2h vs 0h)
5. **Recording system**: Overlooked in original plan (2h vs 0h)

---

## 8. Code Examples

### Example 1: Position Initialization (Before/After)

**Before** (`vectorized_env.py` line 189):
```python
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
```

**After**:
```python
# Substrate initialized earlier in __init__
self.positions = self.substrate.initialize_positions(self.num_agents, self.device)
```

**Why**: Supports 2D `[N, 2]`, 3D `[N, 3]`, aspatial `[N, 0]` without hardcoding.

---

### Example 2: Distance Check (Before/After)

**Before** (`vectorized_env.py` line 295):
```python
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
on_this_affordance = distances == 0
```

**After**:
```python
on_this_affordance = self.substrate.is_on_position(self.positions, affordance_pos)
```

**Why**: Works for grid (exact match), continuous (proximity threshold), aspatial (always True).

---

### Example 3: Checkpoint Validation (New Code)

**New** (`runner.py` in `load_checkpoint()`):
```python
def load_checkpoint(self) -> int | None:
    """Load latest checkpoint with substrate validation."""
    checkpoint = torch.load(latest_checkpoint, weights_only=False)

    # Validate substrate compatibility
    if "substrate_metadata" in checkpoint:
        checkpoint_pos_dim = checkpoint["substrate_metadata"]["position_dim"]
        current_pos_dim = self.env.substrate.position_dim

        if checkpoint_pos_dim != current_pos_dim:
            raise ValueError(
                f"Checkpoint substrate mismatch: checkpoint={checkpoint_pos_dim}D, "
                f"current={current_pos_dim}D. Cannot load checkpoint from different substrate."
            )
    else:
        # Legacy checkpoint (version 2): assume 2D
        logger.warning("Loading legacy checkpoint (assuming 2D grid substrate)")
        if self.env.substrate.position_dim != 2:
            raise ValueError(
                f"Legacy checkpoint is 2D, but current substrate is {self.env.substrate.position_dim}D. "
                f"Cannot load legacy checkpoint into non-2D substrate."
            )

    # Continue with normal checkpoint loading...
```

**Why**: Prevents catastrophic failures when loading checkpoints into incompatible substrates.

---

## 9. Validation Checklist

Before merging Phase 4, verify:

### Functional Requirements
- [ ] All position tensors have shape `[num_agents, substrate.position_dim]`
- [ ] Movement respects substrate boundaries (clamp, wrap, bounce)
- [ ] Distance checks work for all substrate types
- [ ] Observations encode positions correctly (grid, aspatial)
- [ ] Affordances randomize correctly (grid, aspatial)
- [ ] Checkpoints save/load with new format (version 3)
- [ ] Backward compatibility: load version 2 checkpoints into 2D substrate
- [ ] Visualization routes by substrate type
- [ ] Recording system handles variable-length positions

### Test Coverage
- [ ] Unit tests for all 9 integration points
- [ ] Parameterized tests across substrate types (2D, aspatial)
- [ ] Integration test: full training run with 2D grid
- [ ] Integration test: full training run with aspatial
- [ ] Checkpoint save/load round-trip for both substrate types
- [ ] Load existing Level 2 checkpoint (backward compat)
- [ ] Property test: positions always in bounds
- [ ] Property test: positions always correct shape

### Performance
- [ ] No performance regression in training speed (2D grid)
- [ ] Substrate method calls are hot-path optimized
- [ ] Checkpoint size does not explode (version 3 vs version 2)

### Documentation
- [ ] Update CLAUDE.md with substrate usage
- [ ] Update checkpoint format documentation
- [ ] Add examples for 2D and aspatial substrates
- [ ] Document migration path from legacy checkpoints

---

## 10. Conclusion

Phase 4 is **more complex than originally estimated** due to:

1. **Tight coupling**: Position management touches 9 integration points across 5 files
2. **Checkpoint format**: Breaking change requires careful backward compatibility
3. **Observation encoding**: Partial observability needs new substrate method
4. **Testing overhead**: Many tests hardcode position shape

**Revised effort**: **32 hours** (original: 6-8h, +400%)

**Recommended approach**:
1. Implement sequentially (not parallel) due to tight coupling
2. Add new substrate methods early (blocks all other work)
3. Prioritize checkpoint compatibility (highest risk)
4. Use parameterized tests for substrate types (avoid duplication)

**Success criteria**:
- All existing tests pass with 2D grid substrate
- New tests pass with aspatial substrate
- Existing Level 2 checkpoints load correctly (backward compat)
- No performance regression in training speed

**Phase 4 completion unlocks**: Phase 5 (Config Migration), Phase 6 (Example Substrates), Phase 7 (Frontend Rendering)

---

## Appendix A: Files Modified

### Core Files (Must Change)
1. `src/townlet/environment/vectorized_env.py` (position init, movement, distance, affordances, checkpoints)
2. `src/townlet/environment/observation_builder.py` (observations, distance)
3. `src/townlet/substrate/base.py` (new methods)
4. `src/townlet/substrate/grid2d.py` (new method implementations)
5. `src/townlet/substrate/aspatial.py` (new method implementations)

### Supporting Files (Should Change)
6. `src/townlet/demo/runner.py` (checkpoint validation)
7. `src/townlet/demo/live_inference.py` (visualization routing)
8. `src/townlet/recording/recorder.py` (position tuples)
9. `src/townlet/recording/data_structures.py` (type hints)

### Test Files (Must Update)
10. `tests/test_townlet/properties/test_environment_properties.py`
11. `tests/test_townlet/integration/test_checkpointing.py`
12. `tests/test_townlet/unit/environment/test_observations.py`
13. `tests/test_townlet/conftest.py` (add substrate fixtures)

**Total: 13 files**

---

## Appendix B: Substrate Methods Reference

### Required Methods (Phases 0-3)
- `position_dim: int` - Dimensionality (0, 2, 3, ...)
- `initialize_positions(num_agents, device) -> Tensor[N, position_dim]`
- `apply_movement(positions, deltas) -> Tensor[N, position_dim]`
- `compute_distance(pos1, pos2) -> Tensor[N]`
- `encode_observation(positions, affordances) -> Tensor[N, obs_dim]`
- `get_observation_dim() -> int`
- `get_valid_neighbors(position) -> list[Tensor[position_dim]]`
- `is_on_position(agent_positions, target_position) -> Tensor[N] bool`

### New Methods (Phase 4)
- `get_all_positions() -> list[Tensor[position_dim]]` - For affordance randomization
- `encode_partial_observation(positions, affordances, vision_range) -> Tensor[N, window²]` - For POMDP

**Total: 10 methods**

---

**Document Status**: Research Complete
**Next Step**: Review with team, update implementation plan with revised effort estimates
**Reviewed By**: [Pending]
**Date**: 2025-11-05
