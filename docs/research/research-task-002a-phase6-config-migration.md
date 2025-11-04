# Research: TASK-002A Phase 6 - Config Migration (substrate.yaml Creation)

**Date**: 2025-11-05
**Phase**: 6 (Config Migration - Create substrate.yaml Files)
**Status**: Research Complete
**Researcher**: Claude (Sonnet 4.5)

---

## Executive Summary

**Mission**: Create `substrate.yaml` files for all 7 existing config packs to enable configurable spatial substrates.

**Key Finding**: All configs use square 2D grids with "clamp" boundaries and "manhattan" distance. Migration is straightforward - create identical substrate.yaml for each config pack with their respective grid_size.

**Breaking Change**: ✅ Authorized - Old configs without substrate.yaml will fail with clear error message (per Phase 3 implementation plan).

**Critical Ordering**: Phase 6 MUST complete BEFORE Phase 3 (environment integration) to avoid breaking existing configs during development.

**Effort Estimate**: 2-3 hours (30min per config × 7 configs + testing)

---

## 1. Config Pack Inventory

### 1.1 Discovered Config Packs

Found **7 config packs** in `/home/john/hamlet/configs/`:

| Config Pack | Grid Size | Purpose | Status |
|-------------|-----------|---------|--------|
| `L0_0_minimal` | 3×3 | Pedagogical: Temporal credit assignment (single affordance) | Active |
| `L0_5_dual_resource` | 7×7 | Pedagogical: Multiple resource management (4 affordances) | Active |
| `L1_full_observability` | 8×8 | Baseline: Full observability with all affordances | Active |
| `L2_partial_observability` | 8×8 | POMDP: Partial observability (5×5 window) with LSTM | Active |
| `L3_temporal_mechanics` | 8×8 | Temporal: Time-based affordances + multi-tick interactions | Active |
| `templates` | 8×8 | Template for creating new config packs | Reference |
| `test` | 8×8 | Integration test config (lite, 200 episodes) | Testing |

### 1.2 Current Grid Configuration Location

**Location**: `environment.grid_size` in `training.yaml`

**Example** (from L1_full_observability):
```yaml
environment:
  grid_size: 8  # 8×8 grid world
  partial_observability: false
  vision_range: 8
  ...
```

### 1.3 Grid Configuration Analysis

**All grids are square** (width == height):
- ✅ L0: 3×3 (square)
- ✅ L0.5: 7×7 (square)
- ✅ L1, L2, L3, templates, test: 8×8 (square)

**No non-square grids found** - simplifies migration.

---

## 2. Current Spatial Behavior (Hardcoded)

### 2.1 Boundary Behavior

**Current implementation** (`src/townlet/environment/vectorized_env.py:407`):

```python
# Clamp to grid boundaries
new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
```

**Conclusion**: All configs use **"clamp" boundary** (hard walls, no wraparound).

**Evidence**: No wraparound/toroidal/bounce logic found in codebase.

### 2.2 Distance Metric

**Current implementation** (multiple locations):

```python
# Manhattan distance
distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
```

**Conclusion**: All configs use **"manhattan" distance** (L1 norm).

**Evidence**: No euclidean/chebyshev distance calculations found.

### 2.3 Coordinate System

**Convention** (from vectorized_env.py comments):
- Positions: `[x, y]` where x=column, y=row
- Origin: Top-left corner is `[0, 0]`
- x increases rightward, y increases downward

**Standard 2D grid topology**: "square" (not hexagonal, triangular, etc.)

---

## 3. Substrate Type Mapping

### 3.1 Recommended Substrate Types

| Config Pack | Substrate Type | Rationale |
|-------------|----------------|-----------|
| L0_0_minimal | `grid` (3×3) | Square grid with clamp boundary |
| L0_5_dual_resource | `grid` (7×7) | Square grid with clamp boundary |
| L1_full_observability | `grid` (8×8) | Square grid with clamp boundary |
| L2_partial_observability | `grid` (8×8) | Square grid with clamp boundary |
| L3_temporal_mechanics | `grid` (8×8) | Square grid with clamp boundary |
| templates | `grid` (8×8) | Default template |
| test | `grid` (8×8) | Integration test baseline |

**All configs should use `type: grid`** with Grid2DSubstrateConfig.

### 3.2 Alternative Substrate Examples (Future)

While current configs all use grid substrates, we should create **reference examples** for:

1. **Toroidal grid** (wrap boundary) - For Pac-Man style worlds
2. **Aspatial** - For pure resource management (no navigation)

These would be **additional example configs**, not migrations of existing configs.

---

## 4. substrate.yaml Schema

### 4.1 Required Fields (from Phase 2 schema)

Based on `SubstrateConfig` Pydantic model:

```yaml
version: "1.0"           # Config version (REQUIRED)
description: "..."       # Human-readable description (REQUIRED)
type: "grid"            # Substrate type: "grid" or "aspatial" (REQUIRED)

# For type="grid":
grid:
  topology: "square"           # Grid topology (REQUIRED)
  width: 8                     # Grid width in cells (REQUIRED, > 0)
  height: 8                    # Grid height in cells (REQUIRED, > 0)
  boundary: "clamp"            # Boundary mode: "clamp", "wrap", "bounce" (REQUIRED)
  distance_metric: "manhattan" # Distance metric: "manhattan", "euclidean", "chebyshev" (REQUIRED)
```

### 4.2 No-Defaults Principle Compliance

**All fields are REQUIRED** (no defaults allowed per PDR-002):
- ✅ `version`: Explicit versioning for schema evolution
- ✅ `description`: Forces documentation of substrate choice
- ✅ `type`: Explicit substrate selection
- ✅ `grid.topology`: Explicit topology declaration
- ✅ `grid.width/height`: Explicit dimensions (replaces single grid_size)
- ✅ `grid.boundary`: Explicit boundary behavior (no implicit "clamp")
- ✅ `grid.distance_metric`: Explicit distance metric (no implicit "manhattan")

**Rationale**: Operator must consciously choose every parameter. No hidden defaults.

---

## 5. substrate.yaml Templates

### 5.1 Template: L0_0_minimal (3×3 Grid)

**File**: `configs/L0_0_minimal/substrate.yaml`

```yaml
# Substrate Configuration for Level 0: Minimal
#
# Defines the spatial substrate (coordinate system, topology, boundaries)
# for the training environment.
#
# Level 0 uses a tiny 3×3 grid for fast pedagogical learning:
# - Minimal exploration space (9 cells)
# - Agent learns temporal credit assignment (spacing > spamming)
# - Simple enough to converge in ~100 episodes

version: "1.0"

description: "3×3 square grid with hard boundaries for Level 0 pedagogical training"

type: "grid"

grid:
  # Square 2D grid topology (standard Cartesian grid)
  topology: "square"

  # Grid dimensions: 3×3 (9 total cells)
  width: 3
  height: 3

  # Boundary behavior: "clamp" (hard walls, agents cannot move outside grid)
  # Alternatives: "wrap" (toroidal/Pac-Man), "bounce" (elastic reflection)
  boundary: "clamp"

  # Distance metric: "manhattan" (L1 norm, |x1-x2| + |y1-y2|)
  # Used for proximity calculations and action masking
  # Alternatives: "euclidean" (L2), "chebyshev" (L∞)
  distance_metric: "manhattan"
```

### 5.2 Template: L0_5_dual_resource (7×7 Grid)

**File**: `configs/L0_5_dual_resource/substrate.yaml`

```yaml
# Substrate Configuration for Level 0.5: Dual Resource Management
#
# Slightly larger grid to accommodate 4 affordances (Bed, Hospital, HomeMeal, Job)
# with room for navigation.

version: "1.0"

description: "7×7 square grid with hard boundaries for Level 0.5 multi-resource training"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 7×7 (49 total cells)
  width: 7
  height: 7

  boundary: "clamp"
  distance_metric: "manhattan"
```

### 5.3 Template: L1_full_observability (8×8 Grid)

**File**: `configs/L1_full_observability/substrate.yaml`

```yaml
# Substrate Configuration for Level 1: Full Observability Baseline
#
# Standard 8×8 grid with all 14 affordances. Agent sees complete grid
# (full observability). Baseline for comparing POMDP performance.

version: "1.0"

description: "8×8 square grid with hard boundaries for Level 1 full observability baseline"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  width: 8
  height: 8

  boundary: "clamp"
  distance_metric: "manhattan"
```

### 5.4 Template: L2_partial_observability (8×8 Grid)

**File**: `configs/L2_partial_observability/substrate.yaml`

```yaml
# Substrate Configuration for Level 2: Partial Observability (POMDP)
#
# Same 8×8 grid as L1, but agent sees only 5×5 local window. Agent must
# build mental map through exploration (LSTM memory). Spatial substrate
# identical to L1 - only observability differs (controlled in training.yaml).

version: "1.0"

description: "8×8 square grid with hard boundaries for Level 2 POMDP training"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  width: 8
  height: 8

  boundary: "clamp"
  distance_metric: "manhattan"
```

### 5.5 Template: L3_temporal_mechanics (8×8 Grid)

**File**: `configs/L3_temporal_mechanics/substrate.yaml`

```yaml
# Substrate Configuration for Level 3: Temporal Mechanics
#
# Same 8×8 grid as L2, with added temporal mechanics (operating hours,
# multi-tick interactions). Spatial substrate unchanged - temporal features
# controlled in training.yaml.

version: "1.0"

description: "8×8 square grid with hard boundaries for Level 3 temporal mechanics training"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  width: 8
  height: 8

  boundary: "clamp"
  distance_metric: "manhattan"
```

### 5.6 Template: templates (8×8 Grid)

**File**: `configs/templates/substrate.yaml`

```yaml
# Substrate Configuration - TEMPLATE
#
# Copy this file into your config pack directory and customize the values.
# Every pack should include:
#   - substrate.yaml          (this file)
#   - training.yaml           (training hyperparameters)
#   - affordances.yaml        (affordance definitions)
#   - bars.yaml               (meter definitions)
#   - cascades.yaml           (cascade physics)
#   - cues.yaml               (observable cues, optional)

version: "1.0"

description: "REPLACE_WITH_DESCRIPTION (e.g., '8×8 square grid for my custom level')"

type: "grid"  # Options: "grid" (2D grid), "aspatial" (no positioning)

# Grid substrate configuration (required if type="grid")
grid:
  # Topology: "square" (standard 2D grid)
  topology: "square"

  # Grid dimensions (must be positive integers)
  # Note: Non-square grids (width ≠ height) not yet supported
  width: 8
  height: 8

  # Boundary behavior when agent tries to move outside grid
  # Options:
  #   - "clamp": Hard walls (agent stays at edge)
  #   - "wrap": Toroidal wraparound (Pac-Man style)
  #   - "bounce": Elastic reflection (agent stays in place)
  boundary: "clamp"

  # Distance metric for proximity calculations
  # Options:
  #   - "manhattan": L1 norm (|x1-x2| + |y1-y2|), standard for grid worlds
  #   - "euclidean": L2 norm (sqrt((x1-x2)² + (y1-y2)²)), diagonal-aware
  #   - "chebyshev": L∞ norm (max(|x1-x2|, |y1-y2|)), diagonal = 1 step
  distance_metric: "manhattan"

# Aspatial substrate configuration (required if type="aspatial")
# Uncomment if using aspatial mode (no positioning, pure state machine)
# aspatial:
#   enabled: true
```

### 5.7 Template: test (8×8 Grid)

**File**: `configs/test/substrate.yaml`

```yaml
# Substrate Configuration for Integration Testing
#
# Lite version of L1 for automated testing. Same spatial substrate as L1
# (8×8 grid with clamp boundaries) but shorter training runs (200 episodes).

version: "1.0"

description: "8×8 square grid with hard boundaries for integration testing"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  width: 8
  height: 8

  boundary: "clamp"
  distance_metric: "manhattan"
```

---

## 6. Behavioral Equivalence Verification

### 6.1 Parameter Mapping

**Current system** (`training.yaml` → hardcoded logic):
```yaml
environment:
  grid_size: 8  # Single parameter
  # Implicit: square grid, clamp boundary, manhattan distance
```

**New system** (`substrate.yaml` → configurable):
```yaml
grid:
  width: 8      # Explicit width (replaces grid_size)
  height: 8     # Explicit height (replaces grid_size)
  boundary: "clamp"          # Explicit (previously hardcoded)
  distance_metric: "manhattan"  # Explicit (previously hardcoded)
```

**Equivalence conditions**:
1. `width == height == grid_size` (square grid maintained)
2. `boundary == "clamp"` (current hardcoded behavior)
3. `distance_metric == "manhattan"` (current hardcoded behavior)

### 6.2 Observation Dimension Validation

**Current observation dims** (from CLAUDE.md):
- L0_0_minimal: 36 dims (3×3 grid=9 + 8 meters + 15 affordances + 4 temporal)
- L0_5_dual_resource: 76 dims (7×7 grid=49 + 8 meters + 15 affordances + 4 temporal)
- L1/L2/L3/templates/test: 91 dims (8×8 grid=64 + 8 meters + 15 affordances + 4 temporal)

**Formula**: `obs_dim = grid_size² + 8 + 15 + 4`

**With substrate.yaml**: `obs_dim = width × height + 8 + 15 + 4`

**For behavioral equivalence**:
- L0: `3 × 3 = 9` ✅ (matches current 9)
- L0.5: `7 × 7 = 49` ✅ (matches current 49)
- L1+: `8 × 8 = 64` ✅ (matches current 64)

**Verification strategy**: Assert observation dimensions unchanged before/after migration.

### 6.3 Position Encoding Validation

**Current encoding** (from vectorized_env.py):
```python
# One-hot grid encoding
flat_indices = positions[:, 1] * self.grid_size + positions[:, 0]
```

**With substrate** (from Grid2DSubstrate):
```python
# One-hot grid encoding
flat_indices = positions[:, 1] * self.width + positions[:, 0]
```

**Equivalence**: When `width == grid_size`, encoding identical.

### 6.4 Movement Validation

**Current movement** (vectorized_env.py:407):
```python
new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)
```

**With substrate** (Grid2DSubstrate.apply_movement):
```python
new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, self.width - 1)
new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, self.height - 1)
```

**Equivalence**: When `width == height == grid_size`, behavior identical.

---

## 7. Testing Strategy

### 7.1 Unit Tests (Per Config Pack)

**Test file**: `tests/test_townlet/unit/test_substrate_configs.py`

```python
"""Test substrate.yaml files are valid and behaviorally equivalent."""
import pytest
from pathlib import Path
from townlet.substrate.config import load_substrate_config

@pytest.mark.parametrize("config_name,expected_width,expected_height", [
    ("L0_0_minimal", 3, 3),
    ("L0_5_dual_resource", 7, 7),
    ("L1_full_observability", 8, 8),
    ("L2_partial_observability", 8, 8),
    ("L3_temporal_mechanics", 8, 8),
    ("templates", 8, 8),
    ("test", 8, 8),
])
def test_substrate_config_valid(config_name, expected_width, expected_height):
    """Substrate config should load and match expected dimensions."""
    config_path = Path(f"configs/{config_name}/substrate.yaml")

    config = load_substrate_config(config_path)

    assert config.type == "grid"
    assert config.grid.width == expected_width
    assert config.grid.height == expected_height
    assert config.grid.boundary == "clamp"
    assert config.grid.distance_metric == "manhattan"
```

### 7.2 Integration Tests (Environment Loading)

**Test file**: `tests/test_townlet/integration/test_substrate_migration.py`

```python
"""Test environment loads substrate and produces identical observations."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv

@pytest.mark.parametrize("config_name,expected_obs_dim", [
    ("L0_0_minimal", 36),       # 9 + 8 + 15 + 4
    ("L0_5_dual_resource", 76), # 49 + 8 + 15 + 4
    ("L1_full_observability", 91),  # 64 + 8 + 15 + 4
])
def test_env_observation_dim_unchanged(config_name, expected_obs_dim):
    """Environment with substrate should produce same obs dims as legacy."""
    env = VectorizedHamletEnv(
        config_pack_path=Path(f"configs/{config_name}"),
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Observation dimension should be unchanged
    assert env.observation_dim == expected_obs_dim
```

### 7.3 Regression Tests (Training Smoke Test)

**Goal**: Verify configs still train without errors.

```bash
# Run short training session per config (50 episodes each)
for config in L0_0_minimal L0_5_dual_resource L1_full_observability; do
    echo "Testing $config..."
    uv run python -m townlet.demo.runner \
        --config configs/$config \
        --max-episodes 50 \
        || exit 1
done
```

### 7.4 Test Execution Plan

1. **Create substrate.yaml for test config first** (smallest blast radius)
2. **Run unit test** - verify schema loads correctly
3. **Run integration test** - verify environment loads and obs_dim correct
4. **Run smoke test** - verify 50 episodes train without errors
5. **Repeat for remaining configs** in order: L0 → L0.5 → L1 → L2 → L3 → templates

---

## 8. Effort Estimate

### 8.1 Per-Config Breakdown

| Task | Time per Config | Notes |
|------|----------------|-------|
| Create substrate.yaml | 10 min | Copy template, edit dimensions |
| Write/update unit test | 5 min | Add to parameterized test |
| Run unit test | 2 min | Verify schema validation |
| Run integration test | 5 min | Verify env loading + obs_dim |
| Run smoke test | 5-10 min | 50 episodes training |
| **Subtotal** | **30 min** | Per config pack |

### 8.2 Total Effort

| Phase | Time | Notes |
|-------|------|-------|
| Setup (test infrastructure) | 30 min | Create test files, parameterized tests |
| Config migration (7 × 30min) | 3.5 hours | All 7 config packs |
| Documentation updates | 30 min | CLAUDE.md, example docs |
| **Total** | **4.5 hours** | Conservative estimate |

### 8.3 Optimizations

**Parallel execution**: Configs are independent, can be done in parallel.

**Reduced estimate** (with parallelization):
- Setup: 30 min
- First config (establish pattern): 45 min
- Remaining 6 configs (copy pattern): 6 × 20min = 2 hours
- Documentation: 30 min
- **Total: 3.5 hours**

---

## 9. Dependencies

### 9.1 Prerequisites (Must Exist Before Phase 6)

✅ **Phase 1: Abstract Substrate Interface** (Completed per plan)
- `SpatialSubstrate` base class
- `Grid2DSubstrate` implementation
- `AspatialSubstrate` implementation

✅ **Phase 2: Substrate Configuration Schema** (Completed per plan)
- `SubstrateConfig` Pydantic schema
- `load_substrate_config()` loader
- `SubstrateFactory` builder

### 9.2 Blockers (Must Complete Before Phase 3)

⚠️ **CRITICAL**: Phase 6 (this phase) must complete BEFORE Phase 3 (environment integration).

**Rationale**:
- Phase 3 makes substrate.yaml REQUIRED (breaking change)
- If Phase 3 executes first, all configs break immediately
- Phase 6 creates substrate.yaml files BEFORE Phase 3 enforcement

**Correct order**:
1. Phase 1: Create substrate classes ✅
2. Phase 2: Create config schema ✅
3. **Phase 6: Create substrate.yaml files** ← DO THIS NEXT
4. Phase 3: Enforce substrate.yaml in VectorizedEnv ← THEN THIS
5. Phase 4: Refactor position management
6. Phase 5: Update observation builder

### 9.3 Testing Dependencies

**Unit tests** require:
- Pydantic schema (`SubstrateConfig`)
- Config loader (`load_substrate_config`)

**Integration tests** require:
- Phase 3 completion (environment loads substrate)
- VectorizedEnv updated to use substrate

**Smoke tests** require:
- Phase 3-5 completion (full substrate integration)

---

## 10. Additional Substrate Examples (Future)

While all current configs use standard grids, we should create **reference examples** for operators to learn from:

### 10.1 Toroidal Grid Example (Wraparound)

**File**: `configs/examples/toroidal_grid/substrate.yaml`

```yaml
version: "1.0"
description: "8×8 toroidal grid (Pac-Man style wraparound) for comparison with clamp boundary"
type: "grid"

grid:
  topology: "square"
  width: 8
  height: 8
  boundary: "wrap"  # ← Different! Agents wrap around edges
  distance_metric: "manhattan"
```

**Pedagogical value**: Shows how changing one parameter dramatically alters gameplay.

### 10.2 Aspatial Example (No Positioning)

**File**: `configs/examples/aspatial_resource/substrate.yaml`

```yaml
version: "1.0"
description: "Aspatial universe (pure resource management, no navigation)"
type: "aspatial"

aspatial:
  enabled: true
```

**Pedagogical value**: Reveals that spatial substrate is OPTIONAL - meters are the true universe.

### 10.3 Euclidean Distance Example

**File**: `configs/examples/euclidean_distance/substrate.yaml`

```yaml
version: "1.0"
description: "8×8 grid with Euclidean distance (diagonal-aware proximity)"
type: "grid"

grid:
  topology: "square"
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "euclidean"  # ← Different! Diagonal distance < manhattan
```

**Pedagogical value**: Shows impact of distance metric on agent behavior.

---

## 11. Documentation Updates

### 11.1 CLAUDE.md Updates

**Section**: "Configuration System" → "Config Pack Structure"

**Add after existing file list**:

```markdown
configs/L0_0_minimal/
├── substrate.yaml    # NEW: Spatial substrate definition
├── bars.yaml         # Meter definitions (energy, health, money, etc.)
├── cascades.yaml     # Meter relationships (low satiation → drains energy)
├── affordances.yaml  # Interaction definitions (Bed, Hospital, Job, etc.)
├── cues.yaml         # UI metadata for visualization
└── training.yaml     # Hyperparameters and enabled affordances
```

**Add new subsection**:

```markdown
### substrate.yaml Structure

Defines the spatial substrate (coordinate system, topology, boundaries):

```yaml
version: "1.0"
description: "8×8 square grid with hard boundaries"
type: "grid"  # Options: "grid", "aspatial"

grid:
  topology: "square"        # Grid topology
  width: 8                  # Grid width (columns)
  height: 8                 # Grid height (rows)
  boundary: "clamp"         # "clamp" (walls), "wrap" (toroidal), "bounce"
  distance_metric: "manhattan"  # "manhattan", "euclidean", "chebyshev"
```

**Key principle**: Substrate is OPTIONAL. Aspatial universes (pure state machines)
reveal that meters are the true universe - positioning is just an overlay.
```

### 11.2 Example Documentation

**File**: `docs/examples/substrate-yaml-examples.md`

Create new file with:
1. Standard grid example (L1)
2. Toroidal grid example
3. Aspatial example
4. Comparison table showing behavioral differences

---

## 12. Migration Checklist

### 12.1 Per-Config Migration Steps

- [ ] **L0_0_minimal** (3×3 grid)
  - [ ] Create `substrate.yaml` with 3×3 dimensions
  - [ ] Run unit test (schema validation)
  - [ ] Run integration test (env loading)
  - [ ] Run smoke test (50 episodes)

- [ ] **L0_5_dual_resource** (7×7 grid)
  - [ ] Create `substrate.yaml` with 7×7 dimensions
  - [ ] Run unit test
  - [ ] Run integration test
  - [ ] Run smoke test

- [ ] **L1_full_observability** (8×8 grid)
  - [ ] Create `substrate.yaml` with 8×8 dimensions
  - [ ] Run unit test
  - [ ] Run integration test
  - [ ] Run smoke test

- [ ] **L2_partial_observability** (8×8 grid)
  - [ ] Create `substrate.yaml` with 8×8 dimensions
  - [ ] Run unit test
  - [ ] Run integration test
  - [ ] Run smoke test (may take longer - LSTM)

- [ ] **L3_temporal_mechanics** (8×8 grid)
  - [ ] Create `substrate.yaml` with 8×8 dimensions
  - [ ] Run unit test
  - [ ] Run integration test
  - [ ] Run smoke test

- [ ] **templates** (8×8 grid)
  - [ ] Create `substrate.yaml` template with detailed comments
  - [ ] Run unit test
  - [ ] Verify well-documented for operators

- [ ] **test** (8×8 grid)
  - [ ] Create `substrate.yaml` with 8×8 dimensions
  - [ ] Run unit test
  - [ ] Run integration test (critical for CI)

### 12.2 Documentation Checklist

- [ ] Update CLAUDE.md "Configuration System" section
- [ ] Create `docs/examples/substrate-yaml-examples.md`
- [ ] Add substrate.yaml to config pack templates
- [ ] Update CHANGELOG.md with migration notes

### 12.3 Testing Checklist

- [ ] Create `tests/test_townlet/unit/test_substrate_configs.py`
- [ ] Create `tests/test_townlet/integration/test_substrate_migration.py`
- [ ] Run all unit tests (should pass immediately)
- [ ] Run all integration tests (requires Phase 3 completion)
- [ ] Run smoke tests per config (verify training works)

---

## 13. Risk Assessment

### 13.1 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Typo in substrate.yaml | Medium | High | Unit tests catch schema errors |
| Observation dim mismatch | Low | High | Integration tests verify obs_dim |
| Training regression | Low | Medium | Smoke tests detect training failures |
| Phase ordering error | Medium | High | **Do Phase 6 BEFORE Phase 3** |

### 13.2 Rollback Strategy

If substrate.yaml causes issues:

1. **Keep legacy grid_size support** in Phase 3 (fallback)
2. **Gradual rollout**: Migrate one config at a time
3. **Checkpoint compatibility**: Old checkpoints still work (grid_size → width/height)

**However**: User authorized breaking changes, so rollback not required.

---

## 14. Success Criteria

### 14.1 Definition of Done

✅ Phase 6 is complete when:

1. **All 7 config packs have substrate.yaml** (L0 through test)
2. **All substrate.yaml files pass schema validation** (unit tests green)
3. **Template substrate.yaml is well-documented** (for operators)
4. **CLAUDE.md updated** with substrate.yaml structure
5. **Example documentation created** (toroidal, aspatial, euclidean)
6. **Ready for Phase 3** (no blockers)

### 14.2 Verification Tests

**Before declaring Phase 6 complete**:

```bash
# 1. All substrate configs load without errors
for config in configs/*/substrate.yaml; do
    echo "Validating $config..."
    python -c "
from pathlib import Path
from townlet.substrate.config import load_substrate_config
config = load_substrate_config(Path('$config'))
print(f'✓ {config.description}')
" || exit 1
done

# 2. All unit tests pass
uv run pytest tests/test_townlet/unit/test_substrate_configs.py -v

# 3. Documentation exists
test -f docs/examples/substrate-yaml-examples.md || echo "Missing examples doc!"
```

### 14.3 Behavioral Equivalence Guarantee

**Critical assertion**: substrate.yaml must produce IDENTICAL behavior to legacy hardcoded grid.

**Test**:
1. Run L1 training for 100 episodes with legacy code (pre-migration)
2. Run L1 training for 100 episodes with substrate.yaml (post-migration)
3. Compare final checkpoint Q-values (should be similar, allowing for random seed)

**Acceptance**: Observation dims match, training completes, no crashes.

---

## 15. Recommendations

### 15.1 Execution Strategy

**Recommended order**:

1. **Start with `test` config** (smallest blast radius, used in CI)
2. **Validate pattern** with unit + integration tests
3. **Proceed to L0_0_minimal** (simplest pedagogical level)
4. **Then L0.5, L1, L2, L3** (increasing complexity)
5. **Finish with `templates`** (most important for operators)

**Rationale**: Test early failures on least-critical configs first.

### 15.2 Test-Driven Approach

**Write tests BEFORE creating substrate.yaml files**:

1. Write parameterized unit test for all 7 configs (will fail)
2. Create substrate.yaml for `test` config
3. Run test - should pass for `test`, fail for others
4. Create remaining substrate.yaml files
5. Run test - all should pass

**Benefit**: Ensures substrate.yaml matches test expectations.

### 15.3 Documentation Priority

**Create example docs EARLY** (during Phase 6, not after):

1. Operators will need substrate.yaml template immediately
2. Examples clarify intent and alternatives (toroidal, aspatial)
3. CLAUDE.md update prevents confusion during development

---

## 16. Conclusion

**Phase 6 is straightforward but CRITICAL**:

✅ **All configs use identical substrate** (square grid, clamp, manhattan)
✅ **Migration is mechanical** (copy template, edit dimensions)
✅ **Breaking changes authorized** (no backward compatibility needed)
✅ **Must complete BEFORE Phase 3** (correct ordering essential)

**Estimated effort**: 3.5-4.5 hours (7 configs × 30min + setup/docs)

**Deliverables**:
1. 7 × substrate.yaml files (one per config pack)
2. Unit tests (schema validation)
3. Example docs (toroidal, aspatial)
4. CLAUDE.md updates

**Next step**: Execute Phase 6 migration using this research.

---

**Research Complete** ✅
