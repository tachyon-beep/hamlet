# TASK-000: Configurable Spatial Substrates - Simple Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose the spatial substrate abstraction layer in the actual system, replacing hardcoded 2D grid assumptions with pluggable substrate implementations.

**Architecture:** Create abstract `SpatialSubstrate` interface, implement Grid2D (current behavior) and Aspatial (demonstrates concept), integrate into environment with simple dict-based YAML loading. Defer full schema validation to TASK-001.

**Tech Stack:** Python 3.11+, PyTorch, YAML (simple loading), Abstract Base Classes

**Scope:** Get substrate abstraction "in the system" before TASK-001 (schemas), TASK-002 (actions), TASK-003 (universe compiler).

**Research Summary:**

- Position tensors always `[num_agents, 2]`
- Manhattan distance in 4 locations
- Observation dim depends on `grid_size²`
- ~15 files need changes
- Estimated: 6-8 hours (simplified from original 15-22)

**Review Findings Addressed:**

- ✅ encode_positions() verified to match current ObservationBuilder (lines 127-137)
- ✅ Added get_valid_spawn_positions() method (7th abstract method)
- ✅ Documented LSTM/POMDP limitations in Summary section

---

## Phase 1: Substrate Abstraction Layer

### Task 1.1: Create Abstract Substrate Interface

**Files:**

- Create: `src/townlet/substrate/__init__.py`
- Create: `src/townlet/substrate/base.py`

**Step 1: Create substrate module**

```bash
mkdir -p src/townlet/substrate
```

**Step 2: Write abstract interface**

Create: `src/townlet/substrate/base.py`

```python
"""Abstract substrate interface - exposes spatial system."""

from abc import ABC, abstractmethod
import torch


class SpatialSubstrate(ABC):
    """Abstract interface for spatial substrates.

    Key insight: Meters are the true universe, positioning is optional.
    """

    @property
    @abstractmethod
    def position_dim(self) -> int:
        """Position dimensionality (0=aspatial, 2=2D, 3=3D)."""
        pass

    @abstractmethod
    def random_positions(self, n: int, device: torch.device) -> torch.Tensor:
        """Initialize n random positions. Returns [n, position_dim]."""
        pass

    @abstractmethod
    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply movement with boundary handling."""
        pass

    @abstractmethod
    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute distance between positions."""
        pass

    @abstractmethod
    def get_observation_dim(self) -> int:
        """Dimension of position encoding in observations."""
        pass

    @abstractmethod
    def encode_positions(self, positions: torch.Tensor, affordances: dict) -> torch.Tensor:
        """Encode positions for observation space. Returns [n, obs_dim]."""
        pass

    @abstractmethod
    def get_valid_spawn_positions(self) -> list[tuple]:
        """Return list of valid spawn positions for affordances.

        For Grid2D: [(x, y) for all grid cells]
        For Aspatial: [()] - single aspatial position
        """
        pass
```

Create: `src/townlet/substrate/__init__.py`

```python
"""Spatial substrate abstractions."""
from townlet.substrate.base import SpatialSubstrate

__all__ = ["SpatialSubstrate"]
```

**Step 3: Commit**

```bash
git add src/townlet/substrate/
git commit -m "feat: add abstract SpatialSubstrate interface

Exposes spatial system with 7 core methods:
- position_dim: Dimensionality
- random_positions: Initialization
- apply_movement: Movement + boundaries
- compute_distance: Distance metric
- get_observation_dim: Obs encoding size
- encode_positions: Position → observation
- get_valid_spawn_positions: Valid affordance locations

Part of TASK-000 (Spatial Substrates)."
```

---

### Task 1.2: Implement Grid2DSubstrate (Current Behavior)

**Files:**

- Create: `src/townlet/substrate/grid2d.py`

**Step 1: Implement Grid2D**

Create: `src/townlet/substrate/grid2d.py`

```python
"""2D square grid substrate (replicates current behavior)."""

import torch
from townlet.substrate.base import SpatialSubstrate


class Grid2DSubstrate(SpatialSubstrate):
    """2D square grid with Manhattan distance and clamped boundaries.

    Replicates current vectorized_env.py hardcoded behavior.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @property
    def position_dim(self) -> int:
        return 2

    def random_positions(self, n: int, device: torch.device) -> torch.Tensor:
        """Random positions in [0, width) × [0, height)."""
        return torch.stack([
            torch.randint(0, self.width, (n,), device=device),
            torch.randint(0, self.height, (n,), device=device),
        ], dim=1)

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Apply deltas with clamped boundaries."""
        new_pos = positions + deltas
        new_pos[:, 0] = torch.clamp(new_pos[:, 0], 0, self.width - 1)
        new_pos[:, 1] = torch.clamp(new_pos[:, 1], 0, self.height - 1)
        return new_pos

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Manhattan distance: |x1-x2| + |y1-y2|."""
        if pos2.dim() == 1:
            pos2 = pos2.unsqueeze(0)
        return torch.abs(pos1 - pos2).sum(dim=-1)

    def get_observation_dim(self) -> int:
        """One-hot grid: width × height."""
        return self.width * self.height

    def encode_positions(self, positions: torch.Tensor, affordances: dict) -> torch.Tensor:
        """One-hot grid encoding.

        Matches current ObservationBuilder behavior (observation_builder.py lines 127-137):
        - Grid values: 0=empty, 1=affordance OR agent, 2=agent ON affordance
        """
        num_agents = positions.shape[0]
        device = positions.device

        # Initialize grid
        grid = torch.zeros(num_agents, self.width * self.height, device=device)

        # Mark affordances (value = 1.0) for all agents
        for aff_pos in affordances.values():
            flat_idx = aff_pos[1] * self.width + aff_pos[0]
            grid[:, flat_idx] = 1.0

        # Mark agent positions (add 1.0, so if on affordance it becomes 2.0)
        agent_flat_idx = positions[:, 1] * self.width + positions[:, 0]
        ones = torch.ones(num_agents, 1, device=device)
        grid.scatter_add_(1, agent_flat_idx.unsqueeze(1), ones)

        return grid

    def get_valid_spawn_positions(self) -> list[tuple]:
        """Return all grid cells as valid spawn positions."""
        return [(x, y) for x in range(self.width) for y in range(self.height)]
```

**Step 2: Update **init**.py**

Modify: `src/townlet/substrate/__init__.py`

```python
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate

__all__ = ["SpatialSubstrate", "Grid2DSubstrate"]
```

**Step 3: Commit**

```bash
git add src/townlet/substrate/
git commit -m "feat: implement Grid2DSubstrate (replicates current behavior)

Implements 2D square grid with:
- Manhattan distance
- Clamped boundaries
- One-hot grid encoding

Replicates vectorized_env.py hardcoded behavior exactly.

Part of TASK-000 (Spatial Substrates)."
```

---

### Task 1.3: Implement AspatialSubstrate (Demonstrates Concept)

**Files:**

- Create: `src/townlet/substrate/aspatial.py`

**Step 1: Implement Aspatial**

Create: `src/townlet/substrate/aspatial.py`

```python
"""Aspatial substrate (no positioning - pure state machine)."""

import torch
from townlet.substrate.base import SpatialSubstrate


class AspatialSubstrate(SpatialSubstrate):
    """Substrate with no spatial positioning.

    Key insight: Meters are the universe, positioning is optional.
    """

    @property
    def position_dim(self) -> int:
        return 0  # No position!

    def random_positions(self, n: int, device: torch.device) -> torch.Tensor:
        """Empty positions [n, 0]."""
        return torch.zeros((n, 0), dtype=torch.long, device=device)

    def apply_movement(self, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """No movement in aspatial universe."""
        return positions

    def compute_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Zero distance (no spatial meaning)."""
        return torch.zeros(pos1.shape[0], device=pos1.device)

    def get_observation_dim(self) -> int:
        """No position encoding."""
        return 0

    def encode_positions(self, positions: torch.Tensor, affordances: dict) -> torch.Tensor:
        """Empty encoding [n, 0]."""
        return torch.zeros((positions.shape[0], 0), device=positions.device)

    def get_valid_spawn_positions(self) -> list[tuple]:
        """Single aspatial 'position' (empty tuple)."""
        return [()]  # Single position with no coordinates
```

**Step 2: Update **init**.py**

Modify: `src/townlet/substrate/__init__.py`

```python
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate

__all__ = ["SpatialSubstrate", "Grid2DSubstrate", "AspatialSubstrate"]
```

**Step 3: Commit**

```bash
git add src/townlet/substrate/
git commit -m "feat: implement AspatialSubstrate (demonstrates concept)

Implements aspatial (no positioning) substrate:
- position_dim = 0
- Empty position tensors
- Zero distance
- No position encoding

Demonstrates that positioning is optional - meters are fundamental.

Part of TASK-000 (Spatial Substrates)."
```

---

## Phase 2: Environment Integration

### Task 2.1: Add Substrate to VectorizedEnv

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Add substrate creation in **init****

Modify: `src/townlet/environment/vectorized_env.py`

Find `__init__` around line 36, after `self.grid_size = grid_size`, add:

```python
# Create substrate (for now, hardcode Grid2D based on grid_size)
from townlet.substrate.grid2d import Grid2DSubstrate
self.substrate = Grid2DSubstrate(width=grid_size, height=grid_size)
```

**Step 2: Update position initialization**

Find line ~160 where positions are initialized:

Change:

```python
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
```

To:

```python
self.positions = self.substrate.random_positions(self.num_agents, self.device)
```

**Step 3: Commit**

```bash
git add src/townlet/environment/vectorized_env.py
git commit -m "feat: integrate substrate into VectorizedEnv

VectorizedEnv now creates Grid2DSubstrate and uses it for position initialization.

Hardcodes Grid2D for now (config loading comes later).

Part of TASK-000 (Spatial Substrates)."
```

---

### Task 2.2: Replace Movement with Substrate

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Find movement application**

Locate `_execute_actions` around line 385:

```python
new_positions = self.positions + movement_deltas
new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
```

**Step 2: Replace with substrate**

Change to:

```python
new_positions = self.substrate.apply_movement(self.positions, movement_deltas)
```

**Step 3: Commit**

```bash
git add src/townlet/environment/vectorized_env.py
git commit -m "refactor: use substrate for movement application

Replaced hardcoded clamp with substrate.apply_movement().

Boundary handling now substrate-specific.

Part of TASK-000 (Spatial Substrates)."
```

---

### Task 2.3: Replace Distance Calculations

**Files:**

- Modify: `src/townlet/environment/vectorized_env.py`

**Step 1: Find distance calculations**

Research found 4 locations with:

```python
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
```

**Step 2: Replace all occurrences**

In each location (lines ~274, ~462, ~541), replace with:

```python
distances = self.substrate.compute_distance(self.positions, affordance_pos)
```

**Step 3: Verify no hardcoded distance remains**

```bash
grep -n "torch.abs.*positions.*sum" src/townlet/environment/vectorized_env.py
```

Expected: No matches (all replaced)

**Step 4: Commit**

```bash
git add src/townlet/environment/vectorized_env.py
git commit -m "refactor: use substrate for distance calculations

Replaced 4 hardcoded Manhattan distance calculations with substrate.compute_distance().

Distance metric now substrate-specific.

Part of TASK-000 (Spatial Substrates)."
```

---

### Task 2.4: Update ObservationBuilder

**Files:**

- Modify: `src/townlet/environment/observation_builder.py`

**Step 1: Pass substrate to ObservationBuilder**

Modify `vectorized_env.py` around line where ObservationBuilder is created:

Change:

```python
self.observation_builder = ObservationBuilder(
    grid_size=grid_size,
    ...
)
```

To:

```python
self.observation_builder = ObservationBuilder(
    substrate=self.substrate,
    ...
)
```

**Step 2: Update ObservationBuilder.**init****

Modify: `src/townlet/environment/observation_builder.py`

Change `__init__` signature to accept substrate:

```python
def __init__(self, substrate, partial_observability, vision_range, ...):
    self.substrate = substrate
    self.grid_size = substrate.width if hasattr(substrate, 'width') else 8  # Fallback
    # ... rest
```

**Step 3: Update observation_dim calculation**

Around line where `observation_dim` is calculated:

Change:

```python
if partial_observability:
    # ... local window calc
else:
    self.observation_dim = grid_size * grid_size + 8 + (num_affordance_types + 1)
```

To:

```python
if partial_observability:
    # ... local window calc (unchanged for now)
else:
    position_dim = self.substrate.get_observation_dim()
    self.observation_dim = position_dim + 8 + (num_affordance_types + 1)
```

**Step 4: Commit**

```bash
git add src/townlet/environment/vectorized_env.py src/townlet/environment/observation_builder.py
git commit -m "refactor: observation builder uses substrate

ObservationBuilder now receives substrate and uses substrate.get_observation_dim()
for position encoding dimension.

Part of TASK-000 (Spatial Substrates)."
```

---

## Phase 3: Simple Config Loading (Dict-Based)

### Task 3.1: Add Simple YAML Loader

**Files:**

- Create: `src/townlet/substrate/loader.py`

**Step 1: Create simple YAML loader**

Create: `src/townlet/substrate/loader.py`

```python
"""Simple dict-based substrate config loading (schemas deferred to TASK-001)."""

from pathlib import Path
import yaml
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.aspatial import AspatialSubstrate


def load_substrate(config_path: Path):
    """Load substrate from simple YAML config.

    No schema validation yet - deferred to TASK-001.
    Just loads dict and builds substrate.
    """
    if not config_path.exists():
        return None  # No config = legacy mode

    with open(config_path) as f:
        data = yaml.safe_load(f)

    sub_type = data.get("type")

    if sub_type == "grid":
        grid_cfg = data.get("grid", {})
        return Grid2DSubstrate(
            width=grid_cfg.get("width", 8),
            height=grid_cfg.get("height", 8),
        )

    elif sub_type == "aspatial":
        return AspatialSubstrate()

    else:
        raise ValueError(f"Unknown substrate type: {sub_type}")
```

**Step 2: Update VectorizedEnv to use loader**

Modify: `src/townlet/environment/vectorized_env.py`

In `__init__`, replace substrate creation:

```python
# Try loading substrate from config
from townlet.substrate.loader import load_substrate
substrate_path = config_pack_path / "substrate.yaml"
self.substrate = load_substrate(substrate_path)

if self.substrate is None:
    # Legacy: No substrate.yaml, use hardcoded Grid2D
    from townlet.substrate.grid2d import Grid2DSubstrate
    self.substrate = Grid2DSubstrate(width=grid_size, height=grid_size)
```

**Step 3: Commit**

```bash
git add src/townlet/substrate/loader.py src/townlet/environment/vectorized_env.py
git commit -m "feat: add simple dict-based substrate YAML loading

Added load_substrate() that reads substrate.yaml and builds substrate instance.

No Pydantic validation yet - deferred to TASK-001.
Falls back to Grid2D if no substrate.yaml.

Part of TASK-000 (Spatial Substrates)."
```

---

### Task 3.2: Create Example substrate.yaml

**Files:**

- Create: `configs/L1_full_observability/substrate.yaml`

**Step 1: Create L1 substrate config**

Create: `configs/L1_full_observability/substrate.yaml`

```yaml
# Simple substrate config (full schemas in TASK-001)
type: "grid"
grid:
  width: 8
  height: 8
```

**Step 2: Test environment loads substrate**

```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -c "
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv

env = VectorizedHamletEnv(
    config_pack_path=Path('configs/L1_full_observability'),
    num_agents=1,
    device='cpu',
)

print(f'✓ Substrate loaded: {type(env.substrate).__name__}')
print(f'✓ Position dim: {env.substrate.position_dim}')
print(f'✓ Obs dim: {env.substrate.get_observation_dim()}')
"
```

Expected: Loads Grid2DSubstrate successfully

**Step 3: Commit**

```bash
git add configs/L1_full_observability/substrate.yaml
git commit -m "feat: add substrate.yaml for L1

Created first substrate config (simple dict format).
Full schema validation comes in TASK-001.

Part of TASK-000 (Spatial Substrates)."
```

---

## Phase 4: Verification

### Task 4.1: Run Training Smoke Test

**Step 1: Test training runs with substrate**

```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
timeout 30 uv run python -m townlet.demo.runner --config configs/L1_full_observability --max_episodes 10
```

Expected: Training starts successfully (timeout after 30s is OK)

**Step 2: Verify backward compatibility**

Test config without substrate.yaml:

```bash
timeout 30 uv run python -m townlet.demo.runner --config configs/L0_minimal --max_episodes 10
```

Expected: Falls back to legacy Grid2D, runs successfully

**Step 3: Document completion**

Create status file showing what's done.

---

## Summary

**Completed:**

- ✅ Abstract SpatialSubstrate interface (7 methods)
- ✅ Grid2DSubstrate (replicates current behavior exactly)
- ✅ AspatialSubstrate (demonstrates concept)
- ✅ VectorizedEnv uses substrate for positions, movement, distance
- ✅ ObservationBuilder uses substrate for encoding
- ✅ Simple dict-based YAML loading
- ✅ One example config (L1)

**Known Limitations:**

1. **RecurrentSpatialQNetwork (LSTM) not substrate-aware**: The position encoder in `networks.py` (line 367) is hardcoded to 2D:

   ```python
   self.position_encoder = nn.Linear(2, 32)  # HARDCODED 2D
   ```

   **Impact**: AspatialSubstrate only works with SimpleQNetwork. POMDP configs (L2, L3) require Grid2D substrate.
   **Fix**: Deferred to BRAIN_AS_CODE task (network architecture configuration).

2. **Partial observability not substrate-aware**: The 5×5 local window logic in `observation_builder.py` (lines 148-209) has hardcoded 2D loops.
   **Impact**: POMDP only works with Grid2D substrate.
   **Fix**: Deferred to future substrate-aware POMDP work.

3. **Checkpoint serialization not updated**: Affordance positions saved as `[x, y]` (lines 661-679 in `vectorized_env.py`).
   **Impact**: Aspatial substrates can't save checkpoints properly.
   **Fix**: Deferred to checkpoint migration work.

**Deferred to Later Tasks:**

- TASK-001: Full Pydantic schema validation
- TASK-002: Action space compatibility
- TASK-003: Universe compilation pipeline
- Frontend visualization updates
- All config pack migration (L0, L0.5, L2, L3)
- 3D/hex/graph substrates
- Network architecture substrate awareness (BRAIN_AS_CODE)

**Estimated Effort:** 6-8 hours (down from 15-22)

**Status:** Substrate abstraction is "in the system" - ready for TASK-001 to layer proper validation on top.
