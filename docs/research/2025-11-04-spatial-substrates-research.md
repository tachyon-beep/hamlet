# TASK-000 Spatial Substrate Research Report

## Executive Summary

This report documents **ALL hardcoded spatial substrate references** in the HAMLET/Townlet codebase for TASK-000 (Configurable Spatial Substrates). The current system assumes a **2D square grid with Manhattan distance**, which is deeply embedded across environment, observation, network, training, visualization, and configuration layers.

**Key Finding**: The spatial substrate is **thoroughly hardcoded** but **well-localized** to specific modules. Migration to abstract substrates will require changes in ~15 core files but is architecturally feasible.

---

## 1. Position Management Inventory

### 1.1 Position Tensor Initialization

**Primary Location**: `/home/john/hamlet/src/townlet/environment/vectorized_env.py`

#### Agent Positions (Line 160)
```python
self.positions = torch.zeros((self.num_agents, 2), dtype=torch.long, device=self.device)
```
- **Shape**: `[num_agents, 2]` - **HARDCODED 2D**
- **Dtype**: `torch.long` (discrete integer coordinates)
- **Device**: GPU/CPU compatible

#### Agent Position Reset (Line 198)
```python
self.positions = torch.randint(0, self.grid_size, (self.num_agents, 2), device=self.device)
```
- Random initialization in range `[0, grid_size)`
- **2D assumption**: always generates `[x, y]` pairs

#### Affordance Positions (Lines 105-108)
```python
self.affordances = {
    name: torch.tensor([0, 0], device=device, dtype=torch.long)
    for name in affordance_names_to_deploy
}
```
- **Fixed 2D shape**: `[x, y]` hardcoded as `[0, 0]` placeholder
- Actual positions assigned in `randomize_affordance_positions()` (line 707)

### 1.2 Position Update Locations

#### Movement Execution (Lines 366-387, vectorized_env.py)
```python
# Movement deltas (x, y) coordinates
deltas = torch.tensor([
    [0, -1],  # UP - decreases y, x unchanged
    [0, 1],   # DOWN - increases y, x unchanged
    [-1, 0],  # LEFT - decreases x, y unchanged
    [1, 0],   # RIGHT - increases x, y unchanged
    [0, 0],   # INTERACT (no movement)
    [0, 0],   # WAIT (no movement)
], device=self.device)

movement_deltas = deltas[actions]  # [num_agents, 2]
new_positions = self.positions + movement_deltas
```
- **Cardinal directions only** (no diagonals)
- **2D delta vectors**: `[dx, dy]`

### 1.3 Position Validation

#### Interaction Progress Tracking (Lines 389-395, vectorized_env.py)
```python
if self.enable_temporal_mechanics and old_positions is not None:
    for agent_idx in range(self.num_agents):
        if not torch.equal(old_positions[agent_idx], self.positions[agent_idx]):
            self.interaction_progress[agent_idx] = 0
```
- Position equality check: **assumes fixed-size position vectors**

### 1.4 Position Serialization

#### Checkpoint Save (Lines 661-679, vectorized_env.py)
```python
def get_affordance_positions(self) -> dict:
    positions = {}
    for name, pos_tensor in self.affordances.items():
        pos = pos_tensor.cpu().tolist()
        positions[name] = [int(pos[0]), int(pos[1])]  # [x, y]
    return {
        "positions": positions,
        "ordering": self.affordance_names,
    }
```
- **Explicit [x, y] indexing**: `pos[0]` = x, `pos[1]` = y
- **JSON serialization assumption**: list of 2 integers

---

## 2. Hardcoded Grid Constants

### 2.1 Grid Size Parameter

**Defined in**: All `configs/*/training.yaml` files

| Config Pack | Grid Size | Grid Type | Total Cells |
|-------------|-----------|-----------|-------------|
| L0_minimal | 3×3 | Square | 9 |
| L0_5_dual_resource | 7×7 | Square | 49 |
| L1_full_observability | 8×8 | Square | 64 |
| L2_partial_observability | 8×8 | Square | 64 |
| L3_temporal_mechanics | 8×8 | Square | 64 |

### 2.2 Grid Size Usage in Code

#### Environment Initialization (vectorized_env.py, lines 36, 72)
```python
def __init__(self, grid_size: int = 8, ...):
    self.grid_size = grid_size
```
- **Single dimension parameter**: Assumes square grid (width = height)

#### Observation Dimension Calculation (vectorized_env.py, lines 116-124)
```python
if partial_observability:
    window_size = 2 * vision_range + 1  # 5×5 for vision_range=2
    self.observation_dim = window_size * window_size + 2 + 8 + (num_affordance_types + 1)
else:
    self.observation_dim = grid_size * grid_size + 8 + (num_affordance_types + 1)
```
- **Square grid multiplication**: `grid_size * grid_size`
- **Fixed 2D position encoding**: `+ 2` (for x, y)

### 2.3 2D Coordinate Assumptions

#### Position Indexing (vectorized_env.py, line 254)
```python
# positions[:, 0] = x (column), positions[:, 1] = y (row)
```
- **Documented convention**: `[x, y]` = `[column, row]`

#### Boundary Checks (vectorized_env.py, lines 255-258)
```python
at_top = self.positions[:, 1] == 0  # y == 0
at_bottom = self.positions[:, 1] == self.grid_size - 1  # y == max
at_left = self.positions[:, 0] == 0  # x == 0
at_right = self.positions[:, 0] == self.grid_size - 1  # x == max
```
- **Explicit axis indexing**: `[:, 0]` = x-axis, `[:, 1]` = y-axis

---

## 3. Distance Computation Map

### 3.1 Distance Metric: Manhattan (L1)

**All distance calculations use Manhattan distance**.

#### Primary Pattern (Used 4 times):
```python
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
```

**Locations**:
1. **Action Masking** (vectorized_env.py, line 274)
2. **Multi-tick Interactions** (vectorized_env.py, line 462)
3. **Instant Interactions** (vectorized_env.py, line 541)
4. **Affordance Encoding** (observation_builder.py, line 240)

### 3.2 Distance Calculation Breakdown

#### Formula
```
Manhattan distance = |x1 - x2| + |y1 - y2|
```

#### Proximity Check (Adjacency)
```python
on_this_affordance = distances == 0  # Agent on same cell
```
- **Exact match only**: No proximity radius

---

## 4. Boundary Handling Inventory

### 4.1 Boundary Type: **Clamp (Hard Walls)**

#### Boundary Enforcement (vectorized_env.py, line 385)
```python
new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
```
- **Clamps to valid range**: `[0, grid_size-1]` for both x and y
- **Prevents out-of-bounds**: Moving into wall keeps agent at edge

### 4.2 Action Masking for Boundaries

#### Prevents Invalid Movement (vectorized_env.py, lines 253-264)
```python
at_top = self.positions[:, 1] == 0
at_bottom = self.positions[:, 1] == self.grid_size - 1
at_left = self.positions[:, 0] == 0
at_right = self.positions[:, 0] == self.grid_size - 1

action_masks[at_top, 0] = False     # Can't go UP
action_masks[at_bottom, 1] = False  # Can't go DOWN
action_masks[at_left, 2] = False    # Can't go LEFT
action_masks[at_right, 3] = False   # Can't go RIGHT
```

### 4.3 Wraparound Check: NONE

No wraparound logic found. Only **clamped boundaries** supported.

---

## 5. Observation Building Analysis

### 5.1 Full Observability

**File**: `observation_builder.py`, lines 104-146

#### Observation Structure
```python
# Grid encoding: grid_size × grid_size one-hot
grid_encoding = torch.zeros(num_agents, grid_size * grid_size, device=device)

# Flatten: flat_idx = y * width + x
flat_indices = positions[:, 1] * grid_size + positions[:, 0]
```

#### Observation Dimensions by Level

| Level | Grid | Obs Dim |
|-------|------|---------|
| L0 | 3×3 | 36 (9+8+15+4) |
| L0.5 | 7×7 | 76 (49+8+15+4) |
| L1 | 8×8 | 91 (64+8+15+4) |

### 5.2 Partial Observability (POMDP)

**File**: `observation_builder.py`, lines 148-209

#### Local Vision Window
```python
window_size = 2 * vision_range + 1  # 5×5 for vision_range=2

for dy in range(-vision_range, vision_range + 1):
    for dx in range(-vision_range, vision_range + 1):
        world_x = agent_pos[0] + dx
        world_y = agent_pos[1] + dy
```

#### Observation Structure (L2 POMDP)
- Local window: 25 dims (5×5)
- Position: 2 dims (x, y normalized)
- Meters: 8 dims
- Affordance: 15 dims
- Temporal: 4 dims
- **Total: 54 dims**

---

## 6. Affordance Placement Analysis

### 6.1 Randomized Placement (vectorized_env.py, lines 707-733)

```python
def randomize_affordance_positions(self):
    num_affordances = len(self.affordances)
    total_cells = self.grid_size * self.grid_size

    if num_affordances >= total_cells:
        raise ValueError(f"Grid has {total_cells} cells...")

    # Generate all grid positions
    all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

    # Shuffle and assign
    random.shuffle(all_positions)
    for i, affordance_name in enumerate(self.affordances.keys()):
        new_pos = all_positions[i]
        self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=device)
```

### 6.2 Spatial Validation

**Constraint**: `num_affordances + 1 (agent) < grid_size²`

**Current Limits**:
- 3×3 grid: max 8 affordances
- 7×7 grid: max 48 affordances
- 8×8 grid: max 63 affordances

---

## 7. Frontend Visualization Dependencies

### 7.1 Grid Rendering (SVG)

**File**: `frontend/src/components/Grid.vue`, lines 17-49

```vue
<svg :viewBox="`0 0 ${props.gridWidth * cellSize} ${props.gridHeight * cellSize}`">
  <g v-for="y in props.gridHeight">
    <rect
      v-for="x in props.gridWidth"
      :x="(x - 1) * cellSize"
      :y="(y - 1) * cellSize"
    />
  </g>
</svg>
```

**Assumptions**:
- Separate width/height props
- 2D SVG coordinates
- Nested loops for Cartesian rendering

### 7.2 Agent/Affordance Positioning

```vue
<!-- Affordances -->
<rect
  :x="affordance.x * cellSize"
  :y="affordance.y * cellSize"
/>

<!-- Agents -->
<circle
  :cx="agent.x * cellSize + cellSize / 2"
  :cy="agent.y * cellSize + cellSize / 2"
/>
```

**Coordinate convention**: `(x, y)` maps directly to SVG pixel space.

### 7.3 Video Rendering

**File**: `src/townlet/recording/video_renderer.py`, lines 133-150

```python
def _render_grid(self, ax, step_data: dict):
    ax.set_xlim(-0.5, self.grid_size - 0.5)
    ax.set_ylim(-0.5, self.grid_size - 0.5)

    for x in range(self.grid_size):
        for y in range(self.grid_size):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, ...)
```

---

## 8. Network Architecture Dependencies

### 8.1 SimpleQNetwork

**File**: `src/townlet/agent/networks.py`, lines 11-35

**Spatial dependency**: `obs_dim` includes `grid_size²` grid encoding
- L0 (3×3): `obs_dim = 36`
- L1 (8×8): `obs_dim = 91`

**Implication**: Changing grid size **changes network architecture** → **checkpoint incompatibility**.

### 8.2 RecurrentSpatialQNetwork

**File**: `src/townlet/agent/networks.py`, lines 38-244

```python
# Vision Encoder: CNN for 5×5 local window
self.vision_encoder = nn.Conv2d(1, 16, kernel_size=3, padding=1)

# Position Encoder: 2D input HARDCODED
self.position_encoder = nn.Linear(2, 32)
```

**Spatial dependencies**:
1. Vision encoder: `window_size × window_size` (25 cells)
2. Position encoder: **2D coordinates hardcoded**
3. CNN convolutions assume 2D spatial structure

---

## 9. Testing Hardcoded Assumptions

### 9.1 Observation Dimension Tests

**File**: `tests/test_townlet/test_observation_builder.py`

```python
expected_dim = 64 + 8 + 15 + 4  # Hardcoded for 8×8 grid
assert obs.shape == (4, expected_dim)
```

### 9.2 Test Files Needing Updates

1. `test_observation_builder.py` - Hardcoded dims
2. `test_observation_dimensions.py` - Grid size checks
3. `test_observation_updates.py` - Position changes
4. `test_vectorized_env_temporal.py` - Movement tests

---

## 10. Alternative Topology Feasibility Analysis

### 10.1 3D Grids (XYZ)

**Feasibility**: **Medium complexity**

**Changes needed**:
- Positions: `[N, 2]` → `[N, 3]`
- Movement: Add UP/DOWN (z-axis) → 6 directions
- Observation: `grid_size³` explosion (8³ = 512 dims!)
- Visualization: WebGL or isometric
- Network: Input dim scales cubically

**Challenges**:
- Observation explosion (512 vs 64 cells)
- 3D visualization complexity
- POMDP window (5³ = 125 cells)

### 10.2 Hexagonal Grids

**Feasibility**: **High complexity**

**Changes needed**:
- Coordinates: Axial `(q, r)` or cube `(x, y, z)`
- Movement: 6 directions (NE, E, SE, SW, W, NW)
- Distance: Hex distance metric
- Visualization: SVG polygons

**Challenges**:
- Coordinate math complexity
- Different adjacency rules
- Torus topology more natural

### 10.3 Graph Topologies

**Feasibility**: **Very high complexity**

**Changes needed**:
- Positions: Node IDs instead of coordinates
- Movement: Follow edges (adjacency)
- Actions: Dynamic (varies per node)
- Distance: Shortest path (BFS)
- Network: GNN recommended

**Challenges**:
- Variable action space per node
- No closed-form distance
- Complete movement rewrite

### 10.4 Aspatial (State Machine)

**Feasibility**: **Highest complexity** (philosophically different)

**Changes needed**:
- Positions: Abstract state IDs
- Movement: State transitions
- Distance: Semantic similarity
- Visualization: State diagram

**Challenges**:
- Loss of spatial intuition
- Loses "Sims-like" appeal
- When can agent interact?

---

## 11. Breaking Changes Risk Assessment

### 11.1 High-Risk (Break Checkpoints)

1. **Observation dimension changes**
   - Network input size mismatch
   - All saved checkpoints affected

2. **Position tensor shape**
   - `[N, 2]` → `[N, D]` where D ≠ 2
   - Checkpoint conversion needed

3. **Network architecture**
   - CNN assumes 2D (RecurrentSpatialQNetwork)
   - Need substrate-specific networks

### 11.2 Medium-Risk (Updates Needed)

1. **Frontend visualization**
   - Hardcoded SVG rendering
   - Need pluggable renderers

2. **Distance calculations**
   - Manhattan hardcoded (4 locations)
   - Abstract to substrate interface

3. **Config file schema**
   - `grid_size` assumes square
   - Need substrate-specific fields

### 11.3 Low-Risk (Easy Updates)

1. **Action space**
   - Already addressed in TASK-002

2. **Test files**
   - Parameterized tests needed

---

## 12. Dependency Graph

```
SPATIAL SUBSTRATE (grid_size, topology, boundary)
    ↓
POSITIONS [N, 2]
    ↓
OBSERVATIONS [N, obs_dim]  (obs_dim depends on grid_size²)
    ↓
Q-NETWORK (input_dim = obs_dim)
    ↓
ACTION SELECTION (with boundary masks)
    ↓
MOVEMENT (positions += deltas)
    ↓
BOUNDARY CLAMP
    ↓
DISTANCE CHECKS (Manhattan)
    ↓
REWARDS (substrate-agnostic!)
```

**Key insight**: Rewards are **already substrate-agnostic**.

---

## 13. Complete File Inventory

### Critical (Must Change)

| File | Lines | References |
|------|-------|------------|
| `vectorized_env.py` | 160, 198, 254-264, 366-387, 385, 707-733 | Position shape, deltas, boundaries |
| `observation_builder.py` | 64, 132, 180, 240 | Position indexing, distances |
| `networks.py` | 44, 52, 100-103 | Position encoder (2D) |
| `configs/*/training.yaml` | env section | `grid_size` parameter |

### High Priority

| File | Lines | References |
|------|-------|------------|
| `Grid.vue` | 18-49, 52-114 | SVG rendering |
| `video_renderer.py` | 133-150 | Matplotlib rendering |
| `population/vectorized.py` | indirect | Uses action masks |

### Medium Priority (Tests)

| File | References |
|------|------------|
| `test_observation_builder.py` | Hardcoded dims |
| `test_observation_dimensions.py` | Grid assumptions |
| `test_vectorized_env_temporal.py` | Movement tests |

---

## 14. Recommended Implementation Strategy

### Phase 1: Abstract Substrate Interface

Create `substrate/base.py`:
```python
class SpatialSubstrate(ABC):
    @abstractmethod
    def get_position_shape(self) -> tuple[int, ...]

    @abstractmethod
    def random_positions(self, n: int) -> torch.Tensor

    @abstractmethod
    def compute_distance(self, pos1, pos2) -> torch.Tensor

    @abstractmethod
    def clamp_positions(self, pos) -> torch.Tensor

    @abstractmethod
    def get_observation_dim(self) -> int

    @abstractmethod
    def encode_positions(self, pos) -> torch.Tensor
```

### Phase 2: Migrate Core Environment

Replace hardcoded logic in `vectorized_env.py`:
- `torch.randint(...)` → `substrate.random_positions(N)`
- `torch.clamp(...)` → `substrate.clamp_positions(pos)`
- Distance calcs → `substrate.compute_distance(...)`

### Phase 3: Update Observation Builder

- `grid_size * grid_size` → `substrate.get_observation_dim()`
- Flat index calc → `substrate.encode_positions(pos)`

### Phase 4: Network Adaptation

- **SimpleQNetwork**: Already flexible
- **RecurrentSpatialQNetwork**: Update position encoder dim

### Phase 5: Visualization Abstraction

- Create renderer interface
- Substrate-specific renderers
- Route by substrate type

### Phase 6: Config Schema

```yaml
substrate:
  type: "grid_2d"
  width: 8
  height: 8
  distance_metric: "manhattan"
  boundary_mode: "clamp"
```

### Phase 7: Testing & Migration

- Parameterized tests
- Checkpoint migration tool
- Documentation

---

## 15. Conclusion & Risk Mitigation

### Summary

1. **Spatial substrate is deeply hardcoded** but **well-localized** (~15 files).
2. **2D square grid with Manhattan distance** is universal.
3. **Observation dimensions depend on grid size** → network changes.
4. **Checkpoints will break** (high migration risk).
5. **Rewards are substrate-agnostic** (portability bonus).

### Critical Success Factors

1. Maintain backward compatibility
2. Abstract distance computation
3. Substrate-aware networks
4. Comprehensive testing
5. Gradual migration (Phase 1-3 before breaking changes)

### Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Checkpoint incompatibility | High | High | Version + migration tool |
| Frontend rewrite | Medium | Medium | Pluggable renderers |
| Test gaps | Medium | High | Parameterized tests |
| Performance regression | Low | Medium | Benchmark suite |
| Pedagogical loss | Low | High | Default to Grid2D |

---

**End of Report**

**Generated**: 2025-11-04
**Task**: TASK-000 Spatial Substrate Investigation
**Codebase**: Commit 0de8e42 (main)
