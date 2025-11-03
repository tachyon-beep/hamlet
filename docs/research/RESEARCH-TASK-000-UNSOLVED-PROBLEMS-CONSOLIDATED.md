# Research: TASK-000 Unsolved Problems - Consolidated Findings

**Date**: 2025-11-04
**Status**: Complete
**Methodology**: Research → Plan → Review Loop
**Related Task**: TASK-000 (Configurable Spatial Substrates)

---

## Executive Summary

This document consolidates research findings for **6 unsolved problems** identified in TASK-000 (Configurable Spatial Substrates). These problems emerged during initial design but were deferred with the note: "TASK-000 mentions but doesn't fully solve."

### Critical Discoveries

1. **🚨 BLOCKING ISSUE**: One-hot position encoding **prevents 3D substrates** (512 dims for 8×8×3 grid). **Must implement coordinate encoding** for TASK-000 Phase 2.

2. **✅ GOOD NEWS**: Current "lazy creation" pattern for `obs_dim` already works and needs minimal changes.

3. **✅ CAN DEFER**: Visualization (use text), affordance placement (use random), action validation (move to compiler).

### Impact on TASK-000 Timeline

**Original estimate**: 15-22 hours
**With research findings**: 51-65 hours (+140-195%)

**Critical path additions**:
- Coordinate encoding: +20h (MUST implement for 3D)
- Distance semantics: +8-12h (MUST implement for interactions)
- obs_dim property: +8-11h (MUST implement for variable dims)

**Deferrable work**:
- Action validation: Move to TASK-004 (Compiler) - 17h
- GUI visualization: Move to TASK-005 (separate project) - 24-64h
- Explicit positioning: Defer to Phase 2 - 6h

---

## Problem 1: Network obs_dim Variability

### Problem Statement

Different substrates produce vastly different observation dimensions:

| Substrate Type | Position Encoding | Example obs_dim |
|----------------|-------------------|-----------------|
| 2D Square (8×8) | 64-dim one-hot | 91 dims |
| **3D Cubic (8×8×3)** | **512-dim one-hot** | **539 dims** |
| Hexagonal (8×8) | 64-dim one-hot | 91 dims |
| Aspatial | 0-dim (no position!) | 27 dims |

**Challenge**: Network creation requires `obs_dim` at initialization time. How to compute from substrate config?

### Research Findings

**Optimal Solution**: **Lazy Creation** (current system already uses this!)

```python
# runner.py (CURRENT - NO CHANGES NEEDED)
self.env = VectorizedHamletEnv(config_pack_path=config_dir)
obs_dim = self.env.observation_dim  # ← Query after env creation
network = SimpleQNetwork(obs_dim, action_dim)
```

**Required Change**: Add `position_encoding_dim` property to substrate interface:

```python
class SpatialSubstrate(ABC):
    @property
    @abstractmethod
    def position_encoding_dim(self) -> int:
        """Dimension of position encoding contribution."""
        pass

class SquareGridSubstrate(SpatialSubstrate):
    @property
    def position_encoding_dim(self) -> int:
        return self.width * self.height  # One-hot: 64 for 8×8

class AspatialSubstrate(SpatialSubstrate):
    @property
    def position_encoding_dim(self) -> int:
        return 0  # No position encoding!
```

**Environment aggregates**:
```python
self.observation_dim = (
    self.substrate.position_encoding_dim +  # Variable by substrate
    8 +  # Meters (fixed)
    15 +  # Affordances (fixed)
    4  # Temporal (fixed)
)
```

### Implementation Requirements

**Effort**: 8-11 hours

**Phases**:
1. Add `position_encoding_dim` property to substrate interface (2-3h)
2. Implement for all substrate types (SquareGrid, Cubic, Hex, Aspatial) (3-4h)
3. Update `VectorizedHamletEnv` to use substrate property (2-3h)
4. Tests (1h)

**Files to modify**:
- `src/townlet/environment/substrate.py` (new file)
- `src/townlet/environment/vectorized_env.py`

**Priority**: 🔴 **Critical** - Required for TASK-000 Phase 1

**Dependencies**: None (independent change)

---

## Problem 2: Action Space Compatibility Validation

### Problem Statement

Different substrates require different action spaces:

| Substrate | Required Movement Actions | Example |
|-----------|--------------------------|---------|
| 2D square | 4 directional (UP, DOWN, LEFT, RIGHT) | `delta: [0, -1]` |
| 3D cubic | 6 directional (+ UP_FLOOR, DOWN_FLOOR) | `delta: [0, 0, 1]` |
| Hexagonal | 6 hex directions (E, NE, NW, W, SW, SE) | `delta: [1, -1]` |
| Aspatial | **0 movement actions** (movement forbidden) | No deltas |

**Challenge**: How to validate that `actions.yaml` is compatible with `substrate.yaml` before training starts?

### Research Findings

**Optimal Solution**: **Compile-Time Validation** (Stage 4 of Universe Compilation)

**Architecture**:
```python
class SubstrateActionValidator:
    """Validates substrate-action compatibility during compilation."""

    # Registry pattern for extensibility
    _validators: dict[tuple[str, str], Callable] = {}

    @classmethod
    def register_validator(cls, substrate_type: str, topology: str = "default"):
        """Decorator to register validator."""
        def decorator(fn):
            cls._validators[(substrate_type, topology)] = fn
            return fn
        return decorator

@SubstrateActionValidator.register_validator("grid", "square")
def validate_square_grid(substrate, actions) -> ValidationResult:
    """Validate 2D square grid requires 4-way movement."""
    required_deltas = {(0, -1), (0, 1), (-1, 0), (1, 0)}
    actual_deltas = {tuple(a.delta) for a in actions.actions if a.type == "movement"}

    missing = required_deltas - actual_deltas
    if missing:
        return ValidationResult(
            valid=False,
            errors=[f"Square grid requires 4-way movement. Missing deltas: {missing}"]
        )
    return ValidationResult(valid=True, errors=[])
```

**Integration with Universe Compiler**:
```python
# Stage 4: Validate substrate-action compatibility
validator = SubstrateActionValidator(substrate_config, action_config)
result = validator.validate()

if not result.valid:
    raise CompilationError(
        f"Substrate-Action Compatibility Error:\n" +
        "\n".join(f"  • {err}" for err in result.errors)
    )
```

**Pedagogical Error Messages**:
```
❌ Substrate-Action Compatibility Error:
   Config pack: configs/my_3d_house
   Substrate: grid (cubic)

   • Cubic grid requires 6-way movement (horizontal + vertical).
     Missing deltas: {(0, 0, -1), (0, 0, 1)}

     Add vertical movement actions to actions.yaml:
     - {id: 4, name: 'DOWN_FLOOR', type: 'movement', delta: [0, 0, -1]}
     - {id: 5, name: 'UP_FLOOR', type: 'movement', delta: [0, 0, 1]}
```

### Implementation Requirements

**Effort**: 17 hours

**Phases**:
1. Validation infrastructure (4h)
2. Substrate-specific validators (square, cubic, hex, aspatial) (6h)
3. Compiler integration (3h)
4. Testing (4h)

**Files to create**:
- `src/townlet/environment/substrate_action_validator.py` (new)
- `src/townlet/compilation/universe_compiler.py` (Stage 4 integration)

**Priority**: 🟡 **High** - But can be deferred to TASK-004 (Compiler Implementation)

**Dependencies**:
- TASK-000 (substrate types defined)
- TASK-003 (actions.yaml schema)
- TASK-004 (universe compiler pipeline)

**Recommendation**: **DEFER to TASK-004**. TASK-000 can ship without validation - compiler adds it later.

---

## Problem 3: Substrate-Agnostic Visualization

### Problem Statement

Current frontend assumes 2D square grid:
- `GridVisualization.vue` renders 8×8 square
- Agent positions use `(x, y)` coordinates
- Hardcoded grid layout

**Challenge**: How to render 3D cubic, hexagonal, graph, aspatial substrates?

### Research Findings

**Critical Insight**: **Visualization is DEFERRABLE** - should NOT block TASK-000!

**Phase 1 (TASK-000)**: Text-based visualization (4-6h)
```python
# Simple ASCII rendering
def render_2d_square(env):
    grid = [['.' for _ in range(8)] for _ in range(8)]
    for aff_id, pos in env.affordance_positions.items():
        grid[pos[1]][pos[0]] = aff_id[0]  # First letter
    for agent_pos in env.positions:
        grid[agent_pos[1]][agent_pos[0]] = 'A'
    print('\n'.join(''.join(row) for row in grid))
```

**Example output**:
```
. . . B . . . .
. . . . . . . .
. A . . . . H .
. . . . J . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . P .
```

**Phase 2-4 (TASK-005 - Separate Project)**: GUI rendering (24-64h)
- Hexagonal SVG: 8-12h
- 3D floor projection: 6-8h
- Graph D3.js: 10-14h
- Full 3D WebGL: 20-30h (optional)

### Implementation Requirements

**Effort**: 4-6 hours (text viz only)

**Phases**:
1. Text renderer interface (1-2h)
2. 2D square renderer (1h)
3. 3D cubic floor-by-floor renderer (1h)
4. Hex/graph/aspatial renderers (1-2h)

**Files to create**:
- `src/townlet/demo/text_viz.py` (new)

**Priority**: 🟢 **Low** - Nice-to-have for debugging, not critical

**Dependencies**: None

**Recommendation**: **Implement text viz in TASK-000 Phase 1**, defer GUI to TASK-005 as separate low-priority project.

---

## Problem 4: Affordance Placement Syntax

### Problem Statement

Current system: Affordance positions **not in config files** - randomized at runtime via `randomize_affordance_positions()`.

**Challenge**: Different substrates need different position representations:
- 2D square: `[x, y]`
- 3D cubic: `[x, y, z]`
- Hexagonal: `{q: 3, r: 4}` (axial coordinates)
- Graph: `7` (node ID)
- Aspatial: No position

### Research Findings

**Critical Insight**: **Current random placement works perfectly** - explicit positioning can be deferred!

**Recommended Approach**: Hybrid `position` field with substrate-specific interpretation

**Phase 1 (TASK-000)**: Keep random placement (2h effort)
```python
# Extend randomize_affordance_positions() for new substrates
def randomize_affordance_positions(self):
    if isinstance(self.substrate, SquareGridSubstrate):
        # Current: [x, y] in [0, grid_size)
        self.affordance_positions = torch.stack([
            torch.randint(0, self.grid_size, (num_affordances,)),
            torch.randint(0, self.grid_size, (num_affordances,))
        ], dim=1)

    elif isinstance(self.substrate, CubicGridSubstrate):
        # New: [x, y, z]
        self.affordance_positions = torch.stack([
            torch.randint(0, self.substrate.width, (num_affordances,)),
            torch.randint(0, self.substrate.height, (num_affordances,)),
            torch.randint(0, self.substrate.depth, (num_affordances,))
        ], dim=1)

    elif isinstance(self.substrate, AspatialSubstrate):
        # No positions at all!
        self.affordance_positions = torch.zeros((num_affordances, 0))
```

**Phase 2 (Future)**: Add optional explicit positioning (6h effort)
```yaml
# affordances.yaml (optional explicit positioning)
affordances:
  - id: "Bed"
    position: [2, 5]  # 2D: explicit position
    # ... rest of config

  - id: "Hospital"
    position: null  # Randomize (backward compatible)
    # ... rest of config
```

### Implementation Requirements

**Effort**: 2 hours (extend randomization)

**Phases**:
1. Extend `randomize_affordance_positions()` for 3D, hex, graph, aspatial (2h)

**Future effort**: 6 hours (add optional explicit positioning)

**Files to modify**:
- `src/townlet/environment/vectorized_env.py`

**Priority**: 🟢 **Low** - Random placement sufficient for experimentation

**Dependencies**: None

**Recommendation**: **Extend randomization in TASK-000 Phase 1** (2h), defer explicit positioning to Phase 2 when operators request it.

---

## Problem 5: Distance Semantics Across Substrates

### Problem Statement

Current implementation: Hardcoded Manhattan distance and "exact position match" (`distance == 0`) for interactions.

**Challenge**: Different substrates have different distance semantics:

| Substrate | Adjacency Definition | Distance Metric |
|-----------|---------------------|-----------------|
| 2D square | Manhattan ≤ 1 (4-connected) | Manhattan |
| 3D cubic | Manhattan ≤ 1 (6-connected) | 3D Manhattan |
| Hexagonal | Hex distance ≤ 1 (6 neighbors) | Hex distance |
| Toroidal | Wraparound-aware | Wraparound shortest path |
| Graph | Edge connection OR same node | Shortest path (hops) |
| Aspatial | **Always adjacent** (no position!) | Always 0 |

**Key insight**: Pure distance fails for graph (expensive shortest-path) and aspatial (no position).

### Research Findings

**Optimal Solution**: **Hybrid approach** with TWO substrate methods:

1. **`is_adjacent(pos1, pos2) → bool`** - Primary interface for interaction checks
2. **`compute_distance(pos1, pos2) → float`** - Optional for metrics/observations

**Why this is elegant**:
```python
# Aspatial substrate: Everything is adjacent!
class AspatialSubstrate:
    def is_adjacent(self, pos1, pos2) -> bool:
        return True  # No positioning = everything adjacent

    def compute_distance(self, pos1, pos2) -> float:
        return 0.0  # No spatial distance

# Graph substrate: Fast adjacency check, expensive distance
class GraphSubstrate:
    def is_adjacent(self, pos1, pos2) -> bool:
        return self.adjacency_matrix[pos1, pos2]  # O(1) lookup!

    def compute_distance(self, pos1, pos2) -> float:
        return self._shortest_path(pos1, pos2)  # Expensive, only when needed
```

**Interaction check** (primary use case):
```python
# vectorized_env.py
can_interact = self.substrate.is_adjacent(agent_position, affordance_position)
```

**Default implementation** (for simple substrates):
```python
class SpatialSubstrate(ABC):
    def is_adjacent(self, pos1, pos2) -> bool:
        """Default: adjacent if distance ≤ 1."""
        return self.compute_distance(pos1, pos2) <= 1.0

    @abstractmethod
    def compute_distance(self, pos1, pos2) -> float:
        """Subclass must implement distance metric."""
        pass
```

**Substrate-specific implementations**:
```python
class SquareGridSubstrate:
    def is_adjacent(self, pos1, pos2) -> bool:
        """4-connected: orthogonal neighbors only."""
        return torch.abs(pos1 - pos2).sum() == 1  # Manhattan distance exactly 1

class HexagonalGridSubstrate:
    def is_adjacent(self, pos1, pos2) -> bool:
        """Hexagonal: 6 equidistant neighbors."""
        q1, r1 = pos1
        q2, r2 = pos2
        return abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2)) <= 2

class ToroidalGridSubstrate:
    def is_adjacent(self, pos1, pos2) -> bool:
        """Wraparound-aware adjacency."""
        dx = min(abs(pos1[0] - pos2[0]), self.width - abs(pos1[0] - pos2[0]))
        dy = min(abs(pos1[1] - pos2[1]), self.height - abs(pos1[1] - pos2[1]))
        return dx + dy == 1
```

### Implementation Requirements

**Effort**: 8-12 hours

**Phases**:
1. Add `is_adjacent()` and `compute_distance()` to substrate interface (2-3h)
2. Implement for all substrate types (4-6h)
3. Update interaction logic in `VectorizedHamletEnv` (2-3h)

**Files to modify**:
- `src/townlet/environment/substrate.py` (interface)
- `src/townlet/environment/vectorized_env.py` (interaction checks)

**Priority**: 🔴 **Critical** - Required for TASK-000 Phase 1 (interactions must work)

**Dependencies**: None

**Recommendation**: **Implement in TASK-000 Phase 1** as part of substrate abstraction layer.

---

## Problem 6: Observation Encoding Strategy

### Problem Statement

Current: **One-hot encoding** for grid position
```python
# 8×8 grid = 64 dimensions
flat_indices = positions[:, 1] * width + positions[:, 0]
one_hot = torch.zeros((num_agents, 64))
one_hot.scatter_(1, flat_indices.unsqueeze(1), 1)
```

**🚨 CRITICAL ISSUE**: One-hot encoding **explodes for large/3D grids**:

| Substrate Type | One-Hot Dims | Coord Dims | Ratio | Feasible? |
|----------------|--------------|------------|-------|-----------|
| 2D (8×8) | 64 | 2 | 32× | ✅ Yes |
| 2D (16×16) | 256 | 2 | 128× | ⚠️ Marginal |
| **3D (8×8×3)** | **512** | **3** | **170×** | ❌ **NO** |
| **3D (16×16×4)** | **1024** | **3** | **341×** | ❌ **NO** |

**3D substrates are INFEASIBLE with one-hot encoding!**

### Research Findings

**🎯 CRITICAL DISCOVERY**: L2 POMDP **already uses coordinate encoding successfully**!

**Evidence** (`observation_builder.py:201`):
```python
# L2 POMDP: Normalized position coordinates (NOT one-hot!)
normalized_x = positions[:, 0] / (self.grid_size - 1)
normalized_y = positions[:, 1] / (self.grid_size - 1)
position_encoding = torch.stack([normalized_x, normalized_y], dim=1)
# → 2 dimensions instead of 64!
```

**This proves**: Networks CAN learn spatial reasoning from coordinate encoding!

**Recommended Solution**: **Hybrid with auto-selection**

**Auto-selection logic**:
```python
def select_encoding_strategy(substrate):
    if substrate.type == "aspatial":
        return "none"  # 0 dims

    elif substrate.type == "grid":
        if len(substrate.dimensions) == 3:
            return "coords"  # REQUIRED for 3D (512 → 3)
        elif max(substrate.dimensions) > 8:
            return "coords"  # Better for large grids (256 → 2)
        else:
            return "onehot"  # Preserves current behavior for ≤8×8

    elif substrate.type == "continuous":
        return "coords"  # Natural for continuous

    else:
        return "coords"  # Default
```

**Implementation**:
```python
class SquareGridSubstrate:
    def __init__(self, width, height, position_encoding="auto"):
        self.width = width
        self.height = height

        if position_encoding == "auto":
            # Auto-select based on grid size
            if max(width, height) > 8:
                self.position_encoding = "coords"
            else:
                self.position_encoding = "onehot"
        else:
            self.position_encoding = position_encoding

    def encode_position(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode positions using selected strategy."""
        if self.position_encoding == "onehot":
            # Current: 8×8 = 64 dims
            flat_indices = positions[:, 1] * self.width + positions[:, 0]
            one_hot = torch.zeros((positions.shape[0], self.width * self.height))
            one_hot.scatter_(1, flat_indices.unsqueeze(1), 1)
            return one_hot

        elif self.position_encoding == "coords":
            # New: Always 2 dims (normalized to [0, 1])
            normalized = positions.float() / torch.tensor([self.width - 1, self.height - 1])
            return normalized

    @property
    def position_encoding_dim(self) -> int:
        if self.position_encoding == "onehot":
            return self.width * self.height
        elif self.position_encoding == "coords":
            return 2

class CubicGridSubstrate:
    def __init__(self, width, height, depth):
        # FORCE coordinate encoding (one-hot would be 512+ dims!)
        self.position_encoding = "coords"

    def encode_position(self, positions: torch.Tensor) -> torch.Tensor:
        # Normalize [x, y, z] to [0, 1]
        normalized = positions.float() / torch.tensor([
            self.width - 1,
            self.height - 1,
            self.depth - 1
        ])
        return normalized  # Always 3 dims

    @property
    def position_encoding_dim(self) -> int:
        return 3  # Always coordinate encoding for 3D
```

### Transfer Learning Impact

**GAME-CHANGER**: Coordinate encoding enables **transfer learning across grid sizes**!

**Current limitation** (one-hot):
- Train on L0 (3×3): obs_dim = 36 (9 position + 27 other)
- Cannot transfer to L1 (8×8): obs_dim = 91 (64 position + 27 other)
- Network architecture incompatible (input layer size mismatch)

**With coordinate encoding**:
- Train on L0 (3×3): obs_dim = 29 (2 position + 27 other)
- Transfer to L1 (8×8): obs_dim = 29 (2 position + 27 other) ✅ **SAME!**
- Transfer to L1_large (16×16): obs_dim = 29 ✅ **STILL SAME!**

**Same network throughout curriculum!**

### Implementation Requirements

**Effort**: 20 hours (CRITICAL PATH)

**Phases**:
1. Add `encode_position()` to substrate interface (3h)
2. Implement coordinate encoding for all substrates (5h)
3. Auto-selection logic (3h)
4. Backward compatibility testing (L1 with one-hot vs coords) (5h)
5. Transfer learning tests (3D feasibility proof) (4h)

**Files to modify**:
- `src/townlet/environment/substrate.py` (interface + encoding methods)
- `src/townlet/environment/vectorized_env.py` (use substrate encoding)
- `src/townlet/agent/networks.py` (ensure works with coordinate inputs)

**Priority**: 🔴 **CRITICAL** - **BLOCKS TASK-000 Phase 2 (3D substrates)**

**Dependencies**: Problem 1 (obs_dim property) must be implemented first

**Recommendation**: **MUST implement in TASK-000**. Without coordinate encoding, 3D substrates are impossible.

---

## Consolidated Priority Matrix

### Must Implement in TASK-000

| Problem | Solution | Effort | Why Critical |
|---------|----------|--------|--------------|
| **6. Observation Encoding** | Coordinate encoding | 20h | **BLOCKS 3D substrates** (512 dims → 3 dims) |
| **5. Distance Semantics** | `is_adjacent()` abstraction | 8-12h | Required for interactions to work |
| **1. obs_dim Variability** | Substrate property | 8-11h | Required for variable substrates |

**Total critical path**: 36-43 hours

### Can Defer to Later Tasks

| Problem | Solution | Effort | Defer To | Why Deferrable |
|---------|----------|--------|----------|----------------|
| **2. Action Validation** | Compile-time validator | 17h | TASK-004 | Compiler not needed for Phase 1 |
| **3. Visualization** | Text viz + GUI deferred | 4-6h (text)<br>24-64h (GUI) | TASK-005 | Text viz sufficient for testing |
| **4. Affordance Placement** | Extend randomization | 2h (random)<br>6h (explicit) | Phase 2 | Random works perfectly |

**Total deferrable**: 47-89 hours

---

## Revised TASK-000 Implementation Plan

### Original TASK-000 Estimate: 15-22 hours

**Phase 1**: Abstraction layer (6-8h)
**Phase 2**: Config schema (2-3h)
**Phase 3**: Env integration (4-6h)
**Phase 4**: Migrate configs (1-2h)
**Phase 5**: Example alternatives (2-3h)

### Revised With Research Findings: 51-65 hours

**Phase 1: Substrate Abstraction Layer** (18-25h)
- ✅ Create `SpatialSubstrate` interface (2h)
- ✅ Implement `SquareGridSubstrate` (2h)
- ✅ Implement `CubicGridSubstrate` (2h)
- ✅ Implement `AspatialSubstrate` (1h)
- ✅ **Add `position_encoding_dim` property** (Problem 1) (3h)
- ✅ **Add `is_adjacent()` and `compute_distance()`** (Problem 5) (4-6h)
- ✅ **Implement coordinate encoding** (Problem 6) (5h)

**Phase 2: Config Schema** (2-3h)
- ✅ Pydantic DTOs for `substrate.yaml` (2-3h)

**Phase 3: Environment Integration** (12-16h)
- ✅ Update `VectorizedHamletEnv` to use substrate (4-6h)
- ✅ Use `substrate.position_encoding_dim` for obs_dim (2h)
- ✅ Use `substrate.is_adjacent()` for interactions (2-3h)
- ✅ Use `substrate.encode_position()` for observations (3-4h)
- ✅ **Extend `randomize_affordance_positions()`** (Problem 4) (1h)

**Phase 4: Text Visualization** (4-6h)
- ✅ **Text renderer for debugging** (Problem 3) (4-6h)

**Phase 5: Migrate Configs** (3-4h)
- ✅ Create `substrate.yaml` for L0, L0.5, L1, L2, L3 (3-4h)

**Phase 6: Testing & Validation** (12-11h)
- ✅ Test all substrate types (4h)
- ✅ Test coordinate encoding (3h)
- ✅ Test transfer learning (3h)
- ✅ Test 3D feasibility proof (2h)

**Total**: 51-65 hours (+140-195% from original)

---

## Dependencies Between Solutions

```
Problem 1 (obs_dim property)
    ↓ (provides position_encoding_dim)
    ↓
Problem 6 (Coordinate encoding)
    ↓ (uses position_encoding_dim for variable dims)
    ↓
Problem 5 (Distance semantics)
    ↓ (both use substrate abstraction)
    ↓
TASK-000 Phase 1 Complete
    ↓
    ├─→ Problem 4 (Affordance placement) [DEFER to Phase 2]
    ├─→ Problem 3 (Visualization) [DEFER to TASK-005]
    └─→ Problem 2 (Action validation) [DEFER to TASK-004]
```

**Critical path**: 1 → 6 → 5 (must implement in order)

**Parallel work**: 3, 4 can happen anytime (independent)

**Deferred**: 2 waits for TASK-004 (Compiler)

---

## Task Updates Required

### TASK-000: Configurable Spatial Substrates

**Update effort estimate**: 15-22h → 51-65h

**Add to implementation plan**:
- Phase 1: Add coordinate encoding support (Problem 6) - 20h
- Phase 1: Add distance/adjacency abstraction (Problem 5) - 8-12h
- Phase 1: Add obs_dim property (Problem 1) - 8-11h
- Phase 4: Add text visualization (Problem 3) - 4-6h
- Phase 4: Extend affordance randomization (Problem 4) - 2h

**Add to risks section**:
- **Risk**: One-hot encoding prevents 3D substrates
- **Mitigation**: Implement coordinate encoding with auto-selection

**Add to success criteria**:
- [ ] Coordinate encoding works for 3D cubic grids (512 dims → 3 dims)
- [ ] Transfer learning: train on 8×8, works on 16×16 (same obs_dim)
- [ ] `is_adjacent()` works for all substrate types
- [ ] Text visualization renders all substrate types

### TASK-001: Variable-Size Meter System

**Add dependency note**: Coordinate encoding (from TASK-000) enables transfer learning across meter counts AND grid sizes.

### TASK-002: UAC Contracts (DTO Validation)

**Add validation for**:
- `substrate.yaml`: topology, dimensions, boundary, distance_metric, position_encoding
- `affordances.yaml`: optional position field (list, dict, int, or null)

### TASK-004: Compiler Implementation

**Add Stage 4**: Substrate-Action Compatibility Validation
- **Effort**: +17h
- **Validator**: `SubstrateActionValidator` with registry pattern
- **Integration**: Compile-time error with pedagogical messages

**Add to cross-validation**:
- Validate affordance positions are in bounds for substrate
- Validate action deltas match substrate dimensionality

### TASK-005: BRAIN_AS_CODE

**Add dependency**: Needs `position_encoding` field from substrate config to handle variable encoding strategies.

**Add example**:
```yaml
# brain.yaml
network:
  architecture: "simple_q"
  observation_dim: null  # Auto-computed from universe
  encoding_aware: true   # Network must handle coordinate OR one-hot
```

### NEW TASK-006: Substrate-Agnostic Visualization

**Scope**: GUI rendering for all substrate types
- Phase 1: Hexagonal SVG (8-12h)
- Phase 2: 3D floor projection (6-8h)
- Phase 3: Graph D3.js (10-14h)
- Phase 4: Full 3D WebGL (20-30h, optional)

**Total effort**: 24-64h

**Priority**: Low (deferred after TASK-000 through TASK-005)

**Dependencies**: TASK-000 (substrate types), TASK-003 (action space for navigation)

---

## Critical Findings Summary

### 🚨 Blocking Issues

1. **One-hot encoding prevents 3D substrates** (Problem 6)
   - 8×8×3 grid = 512 dimensions (infeasible)
   - **MUST implement coordinate encoding** in TASK-000 Phase 1
   - Effort: 20 hours (critical path)

### ✅ Good News

1. **Current obs_dim pattern works** (Problem 1)
   - Lazy creation already implemented
   - Just need to add substrate property
   - Effort: 8-11 hours (straightforward)

2. **L2 POMDP already uses coordinate encoding** (Problem 6)
   - Proves networks can learn from coordinates
   - No new network architecture needed
   - Just extend to full observability mode

### 🎯 Strategic Opportunities

1. **Transfer learning across grid sizes** (Problem 6)
   - Train once on 8×8, works on any size
   - Same network from L0 to L1 to L1_large
   - Pedagogical value: students see generalization

2. **"Everything is adjacent" aspatial reveals substrate is optional** (Problem 5)
   - Meters are the true universe
   - Spatial substrate is just an overlay
   - Validates HAMLET's fundamental architecture

### 💡 Deferral Decisions

1. **Action validation → TASK-004** (Problem 2)
   - Requires compiler infrastructure
   - Not on critical path for TASK-000
   - Effort saved: 17 hours

2. **GUI visualization → TASK-005** (Problem 3)
   - Text viz sufficient for Phase 1
   - GUI is separate low-priority project
   - Effort saved: 24-64 hours

3. **Explicit positioning → Phase 2** (Problem 4)
   - Random placement works perfectly
   - Add explicit when operators request
   - Effort saved: 6 hours

**Total effort saved by deferral**: 47-87 hours

---

## Recommendations for Immediate Action

### 1. Update TASK-000 Estimate

**Change**: 15-22h → 51-65h (+140-195%)

**Rationale**: Research revealed critical work (coordinate encoding) that wasn't in original plan.

### 2. Prioritize Coordinate Encoding

**Action**: Make coordinate encoding the **highest priority** for TASK-000 Phase 1.

**Rationale**: Without it, 3D substrates are impossible (512 dims infeasible).

### 3. Defer Non-Critical Work

**Action**: Move action validation to TASK-004, GUI viz to TASK-005.

**Rationale**: Saves 47-87 hours without blocking TASK-000 core functionality.

### 4. Leverage L2 POMDP Success

**Action**: Study `observation_builder.py:201` for coordinate encoding implementation.

**Rationale**: Already proven to work, just need to extend to full observability.

### 5. Create NEW TASK-006

**Action**: Create TASK-006 for GUI visualization (separate low-priority project).

**Rationale**: Keeps TASK-000 focused, defers expensive GUI work (24-64h).

---

## Next Steps

1. **Review this consolidation** with user
2. **Update TASK-000** with revised estimates and implementation plan
3. **Update TASK-004** to include action validation (Stage 4)
4. **Create TASK-006** for GUI visualization
5. **Begin TASK-000 implementation** with coordinate encoding as priority

---

## Research Documents Reference

All detailed research saved to:
1. `RESEARCH-NETWORK-OBS-DIM-VARIABILITY.md`
2. `RESEARCH-ACTION-COMPATIBILITY-VALIDATION.md`
3. `RESEARCH-SUBSTRATE-AGNOSTIC-VISUALIZATION.md`
4. `RESEARCH-AFFORDANCE-PLACEMENT-SYNTAX.md`
5. `RESEARCH-DISTANCE-SEMANTICS.md`
6. `RESEARCH-OBSERVATION-ENCODING-STRATEGY.md`

**This document**: `RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md`

---

## Conclusion

The 6 unsolved problems from TASK-000 have been thoroughly researched. Key findings:

1. **🚨 Critical blocker discovered**: One-hot encoding prevents 3D substrates. Must implement coordinate encoding (+20h).

2. **✅ Good architecture validated**: Current obs_dim pattern works, just needs substrate property (+8-11h).

3. **💡 Strategic deferrals possible**: Action validation (→TASK-004), GUI viz (→TASK-005), explicit positioning (→Phase 2) saves 47-87 hours.

4. **🎯 Bonus discovery**: Coordinate encoding enables transfer learning across grid sizes (pedagogical value).

**Revised TASK-000 effort**: 51-65 hours (was 15-22h)
**Critical path additions**: 36-43 hours
**Deferrable work**: 47-87 hours

**Recommendation**: Proceed with updated TASK-000 plan, prioritizing coordinate encoding for 3D substrate support.
