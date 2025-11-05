# TASK-002A Phase 6: Config Migration (Create substrate.yaml Files) - Implementation Plan

**Date**: 2025-11-05
**Status**: Ready for Implementation
**Dependencies**: Phases 1-2 Complete (Substrate classes and schema exist)
**Blocks**: Phase 3 (Environment Integration - MUST wait for Phase 6)
**Estimated Effort**: 3.5-4.5 hours

---

⚠️ **CRITICAL PHASE ORDERING** ⚠️

**THIS PHASE MUST COMPLETE BEFORE PHASE 3**

Phase 3 will enforce substrate.yaml requirement in VectorizedEnv. If Phase 3 runs first, all existing configs will break immediately with "substrate.yaml not found" errors.

**Correct execution order:**
1. ✅ Phase 1: Create substrate abstraction (SpatialSubstrate interface)
2. ✅ Phase 2: Create substrate configuration schema (SubstrateConfig DTOs)
3. **→ Phase 6: Create substrate.yaml files for all configs** ← THIS PHASE
4. → Phase 3: Enforce substrate.yaml loading in VectorizedEnv ← THEN THIS
5. → Phase 4: Refactor position management
6. → Phase 5: Update observation builder

**Why this ordering:**
- Phase 6 creates substrate.yaml files WITHOUT changing any code
- Phase 3 changes VectorizedEnv to REQUIRE substrate.yaml
- If we do Phase 3 first → immediate breakage
- If we do Phase 6 first → files ready when Phase 3 needs them

---

⚠️ **BREAKING CHANGES NOTICE** ⚠️

Phase 6 introduces breaking changes to config pack structure.

**Impact:**
- All config packs MUST include substrate.yaml (after Phase 3)
- Config packs without substrate.yaml will fail with clear error message
- No backward compatibility for missing substrate.yaml

**Rationale:**
Breaking changes authorized per PDR-002 (No-Defaults Principle). Operators must explicitly specify spatial substrate parameters rather than relying on hardcoded defaults.

**Migration Path:**
This phase creates substrate.yaml for all existing configs. New configs created after Phase 3 must include substrate.yaml from the start (template provided).

---

## Executive Summary

Phase 6 creates `substrate.yaml` files for all 7 existing config packs, enabling configurable spatial substrates without modifying Python code.

**Key Finding**: All current configs use **identical spatial behavior** - only grid size differs. Migration is mechanical (copy template, edit dimensions).

**Scope**: 7 config packs requiring substrate.yaml:
- L0_0_minimal (3×3 grid)
- L0_5_dual_resource (7×7 grid)
- L1_full_observability (8×8 grid)
- L2_partial_observability (8×8 grid)
- L3_temporal_mechanics (8×8 grid)
- templates (8×8 grid template)
- test (8×8 grid for CI)

**Current Hardcoded Behavior** (to be made explicit):
- Topology: Square 2D grid
- Boundary: Clamp (hard walls, no wraparound)
- Distance: Manhattan (L1 norm)

**Deliverables**:
1. 7 × substrate.yaml files (one per config pack)
2. Unit tests for schema validation
3. Integration tests for behavioral equivalence
4. Example substrate configs (toroidal, aspatial, euclidean)
5. CLAUDE.md documentation updates

**Critical Success Factors**:
- All substrate.yaml files pass schema validation
- Observation dimensions unchanged (behavioral equivalence)
- Tests verify configs load correctly
- Phase 3 can proceed immediately after Phase 6

---

## Phase 6 Task Breakdown

### Task 6.1: Create Test Infrastructure

**Purpose**: Establish test framework for validating substrate.yaml files

**Files**:
- `tests/test_townlet/unit/test_substrate_configs.py` (NEW)
- `tests/test_townlet/integration/test_substrate_migration.py` (NEW)

**Estimated Time**: 45 minutes

---

#### Step 1: Create unit test file for schema validation

**Action**: Create parameterized test for all config packs

**Create**: `tests/test_townlet/unit/test_substrate_configs.py`

```python
"""Unit tests for substrate.yaml config files."""
import pytest
from pathlib import Path
from townlet.substrate.config import load_substrate_config, Grid2DSubstrateConfig


@pytest.mark.parametrize(
    "config_name,expected_width,expected_height,expected_obs_grid_dim",
    [
        ("L0_0_minimal", 3, 3, 9),  # 3×3 = 9
        ("L0_5_dual_resource", 7, 7, 49),  # 7×7 = 49
        ("L1_full_observability", 8, 8, 64),  # 8×8 = 64
        ("L2_partial_observability", 8, 8, 64),  # 8×8 = 64
        ("L3_temporal_mechanics", 8, 8, 64),  # 8×8 = 64
        ("templates", 8, 8, 64),  # 8×8 = 64
        ("test", 8, 8, 64),  # 8×8 = 64
    ],
)
def test_substrate_config_schema_valid(
    config_name, expected_width, expected_height, expected_obs_grid_dim
):
    """Substrate config should load and pass schema validation."""
    config_path = Path("configs") / config_name / "substrate.yaml"

    # Load config (will raise ValidationError if schema invalid)
    config = load_substrate_config(config_path)

    # Verify basic structure
    assert config.version == "1.0"
    assert config.type == "grid"
    assert config.description  # Non-empty description required

    # Verify grid configuration
    assert isinstance(config.grid, Grid2DSubstrateConfig)
    assert config.grid.topology == "square"
    assert config.grid.width == expected_width
    assert config.grid.height == expected_height
    assert config.grid.boundary == "clamp"
    assert config.grid.distance_metric == "manhattan"

    # Verify observation dimension calculation
    # obs_dim = grid_size + 8 meters + 15 affordances + 4 temporal
    obs_grid_dim = config.grid.width * config.grid.height
    assert obs_grid_dim == expected_obs_grid_dim


@pytest.mark.parametrize(
    "config_name",
    [
        "L0_0_minimal",
        "L0_5_dual_resource",
        "L1_full_observability",
        "L2_partial_observability",
        "L3_temporal_mechanics",
        "templates",
        "test",
    ],
)
def test_substrate_config_behavioral_equivalence(config_name):
    """Substrate config should produce identical behavior to legacy hardcoded grid."""
    config_path = Path("configs") / config_name / "substrate.yaml"
    config = load_substrate_config(config_path)

    # All current configs use same spatial behavior (only size differs)
    assert config.grid.topology == "square"  # Standard 2D grid
    assert config.grid.boundary == "clamp"  # Hard walls (not wrap/bounce)
    assert config.grid.distance_metric == "manhattan"  # L1 norm (not euclidean)

    # Grid must be square (width == height) for current configs
    assert config.grid.width == config.grid.height


def test_substrate_config_no_defaults():
    """Substrate config should require all fields (no-defaults principle)."""
    import yaml
    from pydantic import ValidationError

    # Attempt to load incomplete config (missing required fields)
    incomplete_yaml = """
version: "1.0"
type: "grid"
# Missing: description, grid section
"""
    incomplete_path = Path("/tmp/incomplete_substrate.yaml")
    incomplete_path.write_text(incomplete_yaml)

    # Should raise ValidationError (not fall back to defaults)
    with pytest.raises(ValidationError) as exc_info:
        load_substrate_config(incomplete_path)

    # Error message should mention missing field
    assert "description" in str(exc_info.value).lower()

    # Cleanup
    incomplete_path.unlink()


def test_substrate_config_file_exists():
    """All production config packs should have substrate.yaml."""
    production_configs = [
        "L0_0_minimal",
        "L0_5_dual_resource",
        "L1_full_observability",
        "L2_partial_observability",
        "L3_temporal_mechanics",
        "test",
    ]

    for config_name in production_configs:
        substrate_path = Path("configs") / config_name / "substrate.yaml"
        assert substrate_path.exists(), f"Missing substrate.yaml for {config_name}"
```

**Run test**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_configs.py -v
```

**Expected**: FAIL (substrate.yaml files don't exist yet)

---

#### Step 2: Create integration test file for environment loading

**Action**: Test that environments load substrate correctly and produce correct observation dims

**Create**: `tests/test_townlet/integration/test_substrate_migration.py`

```python
"""Integration tests for substrate.yaml migration (environment loading)."""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.mark.parametrize(
    "config_name,expected_obs_dim",
    [
        ("L0_0_minimal", 36),  # 9 grid + 8 meters + 15 affordances + 4 temporal
        ("L0_5_dual_resource", 76),  # 49 grid + 8 meters + 15 affordances + 4 temporal
        ("L1_full_observability", 91),  # 64 grid + 8 + 15 + 4
        ("L2_partial_observability", 91),  # Same as L1 (full obs_dim, not local window)
        ("L3_temporal_mechanics", 91),  # Same as L1
    ],
)
def test_env_observation_dim_unchanged(config_name, expected_obs_dim):
    """Environment with substrate.yaml should produce same obs dims as legacy."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs") / config_name,
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Observation dimension should match legacy hardcoded behavior
    assert env.observation_dim == expected_obs_dim

    # Verify substrate loaded correctly
    assert env.substrate is not None
    assert env.substrate.position_dim == 2  # 2D grid


@pytest.mark.parametrize(
    "config_name,expected_grid_size",
    [
        ("L0_0_minimal", 3),
        ("L0_5_dual_resource", 7),
        ("L1_full_observability", 8),
        ("L2_partial_observability", 8),
        ("L3_temporal_mechanics", 8),
    ],
)
def test_env_substrate_dimensions(config_name, expected_grid_size):
    """Environment substrate should have correct grid dimensions."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs") / config_name,
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Substrate should be Grid2D with correct dimensions
    assert env.substrate.width == expected_grid_size
    assert env.substrate.height == expected_grid_size
    assert env.substrate.width == env.substrate.height  # Square grid


def test_env_substrate_boundary_behavior():
    """Environment substrate should use clamp boundary (legacy behavior)."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs") / "L1_full_observability",
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Test boundary clamping (agent at edge trying to move out of bounds)
    positions = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")  # Top-left corner
    action_deltas = torch.tensor([[-1, -1]], dtype=torch.long, device="cpu")  # Try to move up-left

    new_positions = env.substrate.apply_movement(positions, action_deltas)

    # Should clamp to [0, 0] (not wrap or bounce)
    assert (new_positions == torch.tensor([[0, 0]], dtype=torch.long)).all()


def test_env_substrate_distance_metric():
    """Environment substrate should use manhattan distance (legacy behavior)."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs") / "L1_full_observability",
        num_agents=1,
        device=torch.device("cpu"),
    )

    # Test manhattan distance calculation
    pos1 = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")
    pos2 = torch.tensor([[3, 4]], dtype=torch.long, device="cpu")

    distance = env.substrate.compute_distance(pos1, pos2)

    # Manhattan distance: |3-0| + |4-0| = 7
    assert distance.item() == 7.0
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/test_substrate_migration.py -v
```

**Expected**: FAIL (substrate.yaml files don't exist yet, environment won't load after Phase 3)

**Note**: Integration tests will only pass after Phase 3 completes (environment loads substrate). For now, they document expected behavior.

---

#### Step 3: Commit test infrastructure

**Command**:
```bash
cd /home/john/hamlet
git add tests/test_townlet/unit/test_substrate_configs.py tests/test_townlet/integration/test_substrate_migration.py
git commit -m "test: add substrate.yaml validation tests

Added test infrastructure for Phase 6 config migration:

1. Unit tests (test_substrate_configs.py):
   - Schema validation for all 7 config packs
   - Behavioral equivalence verification
   - No-defaults principle enforcement
   - File existence checks

2. Integration tests (test_substrate_migration.py):
   - Environment loading with substrate.yaml
   - Observation dimension validation
   - Boundary behavior verification
   - Distance metric validation

Tests currently FAIL (substrate.yaml files don't exist yet).
Will pass after Tasks 6.2-6.8 complete.

Part of TASK-002A Phase 6 (Config Migration)."
```

**Expected**: Clean commit with failing tests (red phase of TDD)

---

### Task 6.2: Create L0_0_minimal Config

**Purpose**: Migrate smallest config (3×3 grid) as proof-of-concept

**Files**:
- `configs/L0_0_minimal/substrate.yaml` (NEW)

**Estimated Time**: 20 minutes

---

#### Step 1: Create substrate.yaml for L0_0_minimal

**Action**: Create substrate config with 3×3 dimensions

**Create**: `configs/L0_0_minimal/substrate.yaml`

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
#
# UNIVERSE_AS_CODE: All spatial parameters explicit (no hidden defaults).

version: "1.0"

description: "3×3 square grid with hard boundaries for Level 0 pedagogical training"

type: "grid"

grid:
  # Square 2D grid topology (standard Cartesian coordinate system)
  # Alternative topologies: (none implemented yet, reserved for future)
  topology: "square"

  # Grid dimensions: 3×3 (9 total cells)
  # Provides minimal space for single affordance (Bed) with navigation
  width: 3
  height: 3

  # Boundary behavior: "clamp" (hard walls, agents cannot move outside grid)
  # When agent tries to move outside: position clamped to grid edges
  #
  # Alternatives (not used in L0):
  #   - "wrap": Toroidal wraparound (Pac-Man style)
  #   - "bounce": Elastic reflection (agent stays in place)
  boundary: "clamp"

  # Distance metric: "manhattan" (L1 norm, |x1-x2| + |y1-y2|)
  # Used for proximity calculations and spatial reasoning
  #
  # Manhattan distance characteristics:
  #   - Only cardinal movement (no diagonal shortcuts)
  #   - Distance [0,0] to [2,2] = 4 steps (not 2.83)
  #   - Matches grid movement mechanics
  #
  # Alternatives (not used in L0):
  #   - "euclidean": L2 norm, sqrt((x1-x2)² + (y1-y2)²)
  #   - "chebyshev": L∞ norm, max(|x1-x2|, |y1-y2|)
  distance_metric: "manhattan"
```

**Expected**: substrate.yaml created with complete documentation

---

#### Step 2: Verify schema validation

**Command**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_schema_valid[L0_0_minimal-3-3-9] -v
```

**Expected**: PASS (L0_0_minimal substrate.yaml loads successfully)

---

#### Step 3: Verify file exists check

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_file_exists -v
```

**Expected**: PASS for L0_0_minimal (file exists now)

---

#### Step 4: Commit L0_0_minimal substrate

**Command**:
```bash
git add configs/L0_0_minimal/substrate.yaml
git commit -m "config: add substrate.yaml for L0_0_minimal

Created spatial substrate configuration for Level 0 pedagogical training:

- Grid: 3×3 (9 cells)
- Topology: Square 2D grid
- Boundary: Clamp (hard walls)
- Distance: Manhattan (L1 norm)

Behavioral equivalence:
- Observation dim: 36 (9 grid + 8 meters + 15 affordances + 4 temporal)
- Matches legacy hardcoded grid_size=3

Part of TASK-002A Phase 6 (Config Migration).
First config migrated (6 remaining)."
```

**Expected**: Clean commit with working substrate.yaml

---

### Task 6.3: Create L0_5_dual_resource Config

**Purpose**: Migrate medium-sized config (7×7 grid)

**Files**:
- `configs/L0_5_dual_resource/substrate.yaml` (NEW)

**Estimated Time**: 15 minutes

---

#### Step 1: Create substrate.yaml for L0_5_dual_resource

**Action**: Create substrate config with 7×7 dimensions

**Create**: `configs/L0_5_dual_resource/substrate.yaml`

```yaml
# Substrate Configuration for Level 0.5: Dual Resource Management
#
# Slightly larger grid to accommodate 4 affordances (Bed, Hospital, HomeMeal, Job)
# with room for navigation and resource management learning.
#
# Level 0.5 teaches balancing multiple resources:
# - Energy + Health cycles (Bed restores energy, Hospital restores health)
# - Economic loop (Job earns money, money enables affordances)
# - Spatial planning (navigate between 4 affordances)

version: "1.0"

description: "7×7 square grid with hard boundaries for Level 0.5 multi-resource training"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 7×7 (49 total cells)
  # Sufficient space for 4 affordances + agent navigation
  width: 7
  height: 7

  # Boundary behavior: "clamp" (hard walls)
  boundary: "clamp"

  # Distance metric: "manhattan" (L1 norm)
  distance_metric: "manhattan"
```

**Expected**: substrate.yaml created (less verbose than L0, reference L0 for full docs)

---

#### Step 2: Verify schema validation

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_schema_valid[L0_5_dual_resource-7-7-49] -v
```

**Expected**: PASS

---

#### Step 3: Commit L0_5_dual_resource substrate

**Command**:
```bash
git add configs/L0_5_dual_resource/substrate.yaml
git commit -m "config: add substrate.yaml for L0_5_dual_resource

Created spatial substrate configuration for Level 0.5 multi-resource training:

- Grid: 7×7 (49 cells)
- Topology: Square 2D grid
- Boundary: Clamp (hard walls)
- Distance: Manhattan (L1 norm)

Behavioral equivalence:
- Observation dim: 76 (49 grid + 8 meters + 15 affordances + 4 temporal)
- Matches legacy hardcoded grid_size=7

Part of TASK-002A Phase 6 (Config Migration).
2/7 configs complete."
```

**Expected**: Clean commit

---

### Task 6.4: Create L1_full_observability Config

**Purpose**: Migrate standard 8×8 config (full observability baseline)

**Files**:
- `configs/L1_full_observability/substrate.yaml` (NEW)

**Estimated Time**: 15 minutes

---

#### Step 1: Create substrate.yaml for L1_full_observability

**Action**: Create substrate config with 8×8 dimensions

**Create**: `configs/L1_full_observability/substrate.yaml`

```yaml
# Substrate Configuration for Level 1: Full Observability Baseline
#
# Standard 8×8 grid with all 14 affordances. Agent sees complete grid
# (full observability). Baseline for comparing POMDP performance.
#
# Level 1 characteristics:
# - All affordances deployed (14 total)
# - Complete grid visibility (no POMDP)
# - Standard MLP Q-network (~26K params)
# - Baseline survival metrics for curriculum progression

version: "1.0"

description: "8×8 square grid with hard boundaries for Level 1 full observability baseline"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  # Standard size for full affordance deployment
  width: 8
  height: 8

  # Boundary behavior: "clamp" (hard walls)
  boundary: "clamp"

  # Distance metric: "manhattan" (L1 norm)
  distance_metric: "manhattan"
```

**Expected**: substrate.yaml created

---

#### Step 2: Verify schema validation

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_schema_valid[L1_full_observability-8-8-64] -v
```

**Expected**: PASS

---

#### Step 3: Commit L1_full_observability substrate

**Command**:
```bash
git add configs/L1_full_observability/substrate.yaml
git commit -m "config: add substrate.yaml for L1_full_observability

Created spatial substrate configuration for Level 1 full observability baseline:

- Grid: 8×8 (64 cells)
- Topology: Square 2D grid
- Boundary: Clamp (hard walls)
- Distance: Manhattan (L1 norm)

Behavioral equivalence:
- Observation dim: 91 (64 grid + 8 meters + 15 affordances + 4 temporal)
- Matches legacy hardcoded grid_size=8

Part of TASK-002A Phase 6 (Config Migration).
3/7 configs complete."
```

**Expected**: Clean commit

---

### Task 6.5: Create L2_partial_observability Config

**Purpose**: Migrate POMDP config (8×8 grid with local vision)

**Files**:
- `configs/L2_partial_observability/substrate.yaml` (NEW)

**Estimated Time**: 15 minutes

---

#### Step 1: Create substrate.yaml for L2_partial_observability

**Action**: Create substrate config with 8×8 dimensions (same as L1, different observability)

**Create**: `configs/L2_partial_observability/substrate.yaml`

```yaml
# Substrate Configuration for Level 2: Partial Observability (POMDP)
#
# Same 8×8 grid as L1, but agent sees only 5×5 local window. Agent must
# build mental map through exploration (LSTM memory).
#
# Spatial substrate identical to L1 - only observability differs
# (controlled in training.yaml via partial_observability=true).
#
# Level 2 characteristics:
# - Same grid as L1 (8×8, 64 cells)
# - Local vision window (5×5, vision_range=2)
# - Recurrent Q-network with LSTM (~600K params)
# - Agent must build spatial memory

version: "1.0"

description: "8×8 square grid with hard boundaries for Level 2 POMDP training"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  # Same spatial substrate as L1 for direct comparison
  width: 8
  height: 8

  # Boundary behavior: "clamp" (hard walls)
  boundary: "clamp"

  # Distance metric: "manhattan" (L1 norm)
  distance_metric: "manhattan"
```

**Expected**: substrate.yaml created with POMDP context

---

#### Step 2: Verify schema validation

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_schema_valid[L2_partial_observability-8-8-64] -v
```

**Expected**: PASS

---

#### Step 3: Commit L2_partial_observability substrate

**Command**:
```bash
git add configs/L2_partial_observability/substrate.yaml
git commit -m "config: add substrate.yaml for L2_partial_observability

Created spatial substrate configuration for Level 2 POMDP training:

- Grid: 8×8 (64 cells)
- Topology: Square 2D grid
- Boundary: Clamp (hard walls)
- Distance: Manhattan (L1 norm)

Behavioral equivalence:
- Observation dim: 91 (same as L1, full grid encoding exists)
- Local window (5×5) handled by vision_range in training.yaml
- Matches legacy hardcoded grid_size=8

Part of TASK-002A Phase 6 (Config Migration).
4/7 configs complete."
```

**Expected**: Clean commit

---

### Task 6.6: Create L3_temporal_mechanics Config

**Purpose**: Migrate temporal mechanics config (8×8 grid with time-based features)

**Files**:
- `configs/L3_temporal_mechanics/substrate.yaml` (NEW)

**Estimated Time**: 15 minutes

---

#### Step 1: Create substrate.yaml for L3_temporal_mechanics

**Action**: Create substrate config with 8×8 dimensions (same spatial, adds temporal)

**Create**: `configs/L3_temporal_mechanics/substrate.yaml`

```yaml
# Substrate Configuration for Level 3: Temporal Mechanics
#
# Same 8×8 grid as L2, with added temporal mechanics (operating hours,
# multi-tick interactions). Spatial substrate unchanged - temporal features
# controlled in training.yaml via enable_temporal_mechanics=true.
#
# Level 3 characteristics:
# - Same grid as L2 (8×8, 64 cells)
# - 24-tick day/night cycle
# - Operating hours (Job 9am-5pm, Bar 6pm-2am)
# - Multi-tick interactions (75% linear + 25% completion bonus)
# - Time-based action masking

version: "1.0"

description: "8×8 square grid with hard boundaries for Level 3 temporal mechanics training"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  # Same spatial substrate as L1/L2 for curriculum progression
  width: 8
  height: 8

  # Boundary behavior: "clamp" (hard walls)
  boundary: "clamp"

  # Distance metric: "manhattan" (L1 norm)
  distance_metric: "manhattan"
```

**Expected**: substrate.yaml created with temporal context

---

#### Step 2: Verify schema validation

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_schema_valid[L3_temporal_mechanics-8-8-64] -v
```

**Expected**: PASS

---

#### Step 3: Commit L3_temporal_mechanics substrate

**Command**:
```bash
git add configs/L3_temporal_mechanics/substrate.yaml
git commit -m "config: add substrate.yaml for L3_temporal_mechanics

Created spatial substrate configuration for Level 3 temporal mechanics:

- Grid: 8×8 (64 cells)
- Topology: Square 2D grid
- Boundary: Clamp (hard walls)
- Distance: Manhattan (L1 norm)

Behavioral equivalence:
- Observation dim: 91 (same as L1/L2)
- Temporal mechanics (24-tick cycle) handled by training.yaml
- Matches legacy hardcoded grid_size=8

Part of TASK-002A Phase 6 (Config Migration).
5/7 configs complete."
```

**Expected**: Clean commit

---

### Task 6.7: Create test Config

**Purpose**: Migrate integration test config (8×8 grid, lite version)

**Files**:
- `configs/test/substrate.yaml` (NEW)

**Estimated Time**: 15 minutes

---

#### Step 1: Create substrate.yaml for test config

**Action**: Create substrate config for CI/testing

**Create**: `configs/test/substrate.yaml`

```yaml
# Substrate Configuration for Integration Testing
#
# Lite version of L1 for automated testing. Same spatial substrate as L1
# (8×8 grid with clamp boundaries) but shorter training runs (200 episodes).
#
# Used for:
# - CI/CD automated testing
# - Quick validation of code changes
# - Behavioral regression detection

version: "1.0"

description: "8×8 square grid with hard boundaries for integration testing"

type: "grid"

grid:
  topology: "square"

  # Grid dimensions: 8×8 (64 total cells)
  # Same as L1 for behavioral consistency
  width: 8
  height: 8

  # Boundary behavior: "clamp" (hard walls)
  boundary: "clamp"

  # Distance metric: "manhattan" (L1 norm)
  distance_metric: "manhattan"
```

**Expected**: substrate.yaml created for testing

---

#### Step 2: Verify schema validation

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_schema_valid[test-8-8-64] -v
```

**Expected**: PASS

---

#### Step 3: Verify all file existence checks pass

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_file_exists -v
```

**Expected**: PASS (all 6 production configs have substrate.yaml now)

---

#### Step 4: Commit test substrate

**Command**:
```bash
git add configs/test/substrate.yaml
git commit -m "config: add substrate.yaml for integration testing

Created spatial substrate configuration for automated testing:

- Grid: 8×8 (64 cells)
- Topology: Square 2D grid
- Boundary: Clamp (hard walls)
- Distance: Manhattan (L1 norm)

Behavioral equivalence:
- Observation dim: 91 (same as L1)
- Used in CI for regression testing
- Matches legacy hardcoded grid_size=8

Part of TASK-002A Phase 6 (Config Migration).
6/7 configs complete (templates remaining)."
```

**Expected**: Clean commit, CI-ready

---

### Task 6.8: Create templates Config

**Purpose**: Create well-documented template for operators creating new configs

**Files**:
- `configs/templates/substrate.yaml` (NEW)

**Estimated Time**: 30 minutes (extra time for comprehensive documentation)

---

#### Step 1: Create substrate.yaml template with extensive comments

**Action**: Create comprehensive template showing all options

**Create**: `configs/templates/substrate.yaml`

```yaml
# Substrate Configuration - TEMPLATE
#
# Copy this file into your config pack directory and customize the values.
#
# UNIVERSE_AS_CODE Principle:
# All spatial parameters must be explicitly specified. No hidden defaults allowed.
# This ensures reproducible configs and operator awareness of all behavioral parameters.
#
# Config Pack Structure:
# Every config pack should include these files:
#   - substrate.yaml    (this file - spatial substrate definition)
#   - bars.yaml         (meter definitions: energy, health, satiation, etc.)
#   - cascades.yaml     (cascade physics: meter interactions)
#   - affordances.yaml  (interaction definitions: Bed, Job, Hospital, etc.)
#   - training.yaml     (training hyperparameters and network config)
#   - cues.yaml         (optional - UI metadata for visualization)

version: "1.0"

# Human-readable description of this substrate configuration
# Should explain: grid size, boundary type, distance metric, intended use
description: "REPLACE_WITH_DESCRIPTION (e.g., '8×8 square grid for my custom level')"

# Substrate type: "grid" (spatial with positioning) or "aspatial" (pure state machine)
#
# "grid": Agents have positions on 2D grid, navigate via movement actions
# "aspatial": No positions, agents interact with affordances directly (no navigation)
type: "grid"  # Options: "grid", "aspatial"

# Grid substrate configuration (REQUIRED if type="grid", omit if type="aspatial")
grid:
  # Topology: Type of grid coordinate system
  # Currently only "square" (standard 2D Cartesian grid) is supported
  # Future: "hexagonal", "triangular", "3d_cube" (reserved for future use)
  topology: "square"

  # Grid dimensions (REQUIRED, must be positive integers)
  # width: Number of columns (x-axis, 0 to width-1)
  # height: Number of rows (y-axis, 0 to height-1)
  #
  # Current implementation: Non-square grids (width ≠ height) are supported
  # but uncommon. Most curriculum levels use square grids.
  #
  # Observation dimension impact: grid encoding uses width × height dimensions
  # Example: 8×8 grid = 64 dims, 3×3 grid = 9 dims
  width: 8
  height: 8

  # Boundary behavior: What happens when agent tries to move outside grid
  # Options:
  #   - "clamp": Hard walls - position clamped to grid edges (agent stays at edge)
  #   - "wrap": Toroidal wraparound - agent wraps to opposite edge (Pac-Man style)
  #   - "bounce": Elastic reflection - agent bounces back, stays in place
  #
  # Behavioral implications:
  #   - "clamp": Agent learns edges exist, corners are dead ends
  #   - "wrap": Infinite grid feel, shortest path may cross edges
  #   - "bounce": Similar to clamp but different semantics
  #
  # Most curriculum levels use "clamp" for clear spatial boundaries.
  boundary: "clamp"

  # Distance metric: How to measure distance between two positions
  # Options:
  #   - "manhattan": L1 norm, |x1-x2| + |y1-y2| (sum of absolute differences)
  #   - "euclidean": L2 norm, sqrt((x1-x2)² + (y1-y2)²) (straight-line distance)
  #   - "chebyshev": L∞ norm, max(|x1-x2|, |y1-y2|) (max of differences)
  #
  # Behavioral implications:
  #   - "manhattan": Only cardinal movement, diagonal distance = x + y steps
  #                  Example: [0,0] to [3,4] = 7 steps
  #                  Natural for grid worlds with 4-directional movement
  #   - "euclidean": Considers diagonal shortcuts, diagonal distance < manhattan
  #                  Example: [0,0] to [3,4] = 5.0 steps
  #                  More realistic for spatial reasoning, but doesn't match movement mechanics
  #   - "chebyshev": Diagonal = 1 step (8-directional movement)
  #                  Example: [0,0] to [3,4] = 4 steps
  #                  Natural for games with diagonal movement
  #
  # Most curriculum levels use "manhattan" to match 4-directional movement.
  distance_metric: "manhattan"

# Aspatial substrate configuration (REQUIRED if type="aspatial", omit if type="grid")
#
# Aspatial universes have NO positioning - agents exist in pure state space.
# Affordances are available without navigation. Reveals that meters are the
# true universe - positioning is just an optional overlay.
#
# Use cases:
#   - Pure resource management (no spatial planning)
#   - Baseline for comparing spatial vs aspatial learning
#   - Pedagogical: Show that substrate is OPTIONAL
#
# Uncomment if using aspatial mode:
# aspatial:
#   enabled: true

# Examples and Pedagogical Notes:
#
# Standard Grid (Most Common):
#   type: grid
#   grid: {topology: square, width: 8, height: 8, boundary: clamp, distance_metric: manhattan}
#   Use for: Standard curriculum levels, spatial planning
#
# Toroidal Grid (Wraparound):
#   type: grid
#   grid: {topology: square, width: 8, height: 8, boundary: wrap, distance_metric: manhattan}
#   Use for: Pac-Man style worlds, infinite grid feel
#
# Euclidean Distance (Diagonal-Aware):
#   type: grid
#   grid: {topology: square, width: 8, height: 8, boundary: clamp, distance_metric: euclidean}
#   Use for: Comparing distance metrics, realistic spatial reasoning
#
# Aspatial (No Positioning):
#   type: aspatial
#   aspatial: {enabled: true}
#   Use for: Pure resource management, baseline comparison
#
# Pedagogical Value:
# Changing ONE parameter (boundary, distance_metric) dramatically alters gameplay.
# Students learn how spatial abstractions shape agent behavior.
```

**Expected**: Comprehensive template with extensive documentation

---

#### Step 2: Verify schema validation

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py::test_substrate_config_schema_valid[templates-8-8-64] -v
```

**Expected**: PASS

---

#### Step 3: Run all substrate config tests

**Command**:
```bash
uv run pytest tests/test_townlet/unit/test_substrate_configs.py -v
```

**Expected**: ALL PASS (7/7 configs have valid substrate.yaml)

---

#### Step 4: Commit templates substrate

**Command**:
```bash
git add configs/templates/substrate.yaml
git commit -m "config: add substrate.yaml template with comprehensive documentation

Created well-documented substrate template for operators:

- Complete examples for all options
- Behavioral implications explained
- UNIVERSE_AS_CODE principle documented
- Examples: standard grid, toroidal, euclidean, aspatial

Documentation includes:
- Parameter descriptions and options
- Behavioral implications of each choice
- Example configurations for common use cases
- Pedagogical notes on spatial abstractions

Part of TASK-002A Phase 6 (Config Migration).
7/7 configs complete! ✅"
```

**Expected**: Clean commit, migration complete

---

### Task 6.9: Create Example Substrate Configs

**Purpose**: Provide reference examples for alternative substrate configurations

**Files**:
- `docs/examples/substrate-toroidal-grid.yaml` (NEW)
- `docs/examples/substrate-aspatial.yaml` (NEW)
- `docs/examples/substrate-euclidean-distance.yaml` (NEW)
- `docs/examples/substrate-comparison.md` (NEW)

**Estimated Time**: 30 minutes

---

#### Step 1: Create examples directory

**Command**:
```bash
cd /home/john/hamlet
mkdir -p docs/examples
```

**Expected**: Directory created

---

#### Step 2: Create toroidal grid example

**Action**: Show wraparound boundary behavior

**Create**: `docs/examples/substrate-toroidal-grid.yaml`

```yaml
# Example: Toroidal Grid (Pac-Man Style Wraparound)
#
# This substrate demonstrates "wrap" boundary behavior where agents
# wrap around edges to the opposite side (like Pac-Man).
#
# Behavioral differences from standard "clamp" boundary:
#   - No hard walls - agent can exit left edge and appear on right edge
#   - Shortest path may cross edges (wrap distance < straight distance)
#   - Infinite grid feel (no "corners" or "edges")
#
# Use cases:
#   - Pac-Man style games
#   - Comparing boundary behaviors (clamp vs wrap)
#   - Demonstrating impact of single parameter change

version: "1.0"
description: "8×8 toroidal grid (Pac-Man style wraparound) for comparison with clamp boundary"
type: "grid"

grid:
  topology: "square"
  width: 8
  height: 8
  boundary: "wrap"  # ← Different! Agents wrap around edges
  distance_metric: "manhattan"

# Example behaviors:
#
# Agent at [0, 4] moves LEFT:
#   - Standard (clamp): stays at [0, 4] (hit wall)
#   - Toroidal (wrap): wraps to [7, 4] (opposite edge)
#
# Distance [0, 0] to [7, 0]:
#   - Standard (clamp): 7 steps (must walk across)
#   - Toroidal (wrap): 1 step (wrap left to reach)
```

**Expected**: Toroidal example created

---

#### Step 3: Create aspatial example

**Action**: Show no-positioning configuration

**Create**: `docs/examples/substrate-aspatial.yaml`

```yaml
# Example: Aspatial Universe (No Positioning)
#
# This substrate demonstrates pure state machine mode where agents have
# NO position and NO navigation. Affordances are available without spatial location.
#
# Reveals core insight: Meters are the TRUE universe - positioning is just an overlay.
#
# Behavioral differences from grid substrate:
#   - No movement actions (UP/DOWN/LEFT/RIGHT disabled)
#   - No navigation planning (all affordances equally accessible)
#   - Pure resource management (energy, health, satiation cycles)
#   - Observation dim: 0 grid + 8 meters + 15 affordances + 4 temporal = 27 dims
#
# Use cases:
#   - Pure resource management baseline (no spatial planning)
#   - Comparing spatial vs aspatial learning curves
#   - Pedagogical: Show that substrate is OPTIONAL
#   - Faster training (no navigation exploration needed)

version: "1.0"
description: "Aspatial universe (pure resource management, no navigation)"
type: "aspatial"

aspatial:
  enabled: true

# Example behaviors:
#
# Agent state:
#   - No position (position_dim = 0)
#   - Meters still update (energy drains, cascades work)
#   - Affordances still available (INTERACT action on any affordance)
#
# Training implications:
#   - Faster convergence (no spatial exploration)
#   - Simpler Q-network (fewer input dims)
#   - Pure resource optimization (energy/health cycles)
#
# When to use:
#   - Baseline for curriculum Level 0 (pure credit assignment)
#   - Validating that spatial planning is NOT required for survival
#   - Isolating resource management from navigation
```

**Expected**: Aspatial example created

---

#### Step 4: Create euclidean distance example

**Action**: Show diagonal-aware distance metric

**Create**: `docs/examples/substrate-euclidean-distance.yaml`

```yaml
# Example: Euclidean Distance (Diagonal-Aware Proximity)
#
# This substrate demonstrates "euclidean" distance metric where proximity
# calculations consider diagonal distance (L2 norm, straight-line).
#
# Behavioral differences from standard "manhattan" distance:
#   - Diagonal positions are CLOSER than manhattan distance
#   - Proximity calculations more realistic (as the crow flies)
#   - May not match movement mechanics (if only 4-directional movement)
#
# Use cases:
#   - Comparing distance metrics (manhattan vs euclidean)
#   - More realistic spatial reasoning
#   - Pedagogical: Show impact of distance metric on behavior

version: "1.0"
description: "8×8 grid with Euclidean distance (diagonal-aware proximity)"
type: "grid"

grid:
  topology: "square"
  width: 8
  height: 8
  boundary: "clamp"
  distance_metric: "euclidean"  # ← Different! L2 norm instead of L1

# Example behaviors:
#
# Distance from [0, 0] to [3, 4]:
#   - Manhattan: |3-0| + |4-0| = 7 steps
#   - Euclidean: sqrt((3-0)² + (4-0)²) = sqrt(9 + 16) = 5.0 steps
#
# Distance from [0, 0] to [2, 2]:
#   - Manhattan: |2-0| + |2-0| = 4 steps
#   - Euclidean: sqrt((2-0)² + (2-0)²) = sqrt(8) = 2.83 steps
#
# Proximity implications:
#   - Diagonal affordances feel "closer" with euclidean
#   - Agent may prefer diagonal paths (even if movement is cardinal-only)
#   - Mismatch between distance metric and movement mechanics
#
# When to use:
#   - Realistic spatial reasoning (physical distance)
#   - Games with diagonal movement
#   - Comparing metric impact on learning
```

**Expected**: Euclidean example created

---

#### Step 5: Create comparison documentation

**Action**: Document behavioral differences between substrate types

**Create**: `docs/examples/substrate-comparison.md`

```markdown
# Substrate Configuration Examples - Behavioral Comparison

This document compares different substrate configurations and their impact on agent behavior.

## Quick Reference Table

| Substrate | Grid Size | Boundary | Distance | Position Dim | Obs Dim | Use Case |
|-----------|-----------|----------|----------|--------------|---------|----------|
| **Standard (L1)** | 8×8 | clamp | manhattan | 2 | 91 | Standard curriculum training |
| **Toroidal** | 8×8 | wrap | manhattan | 2 | 91 | Pac-Man style worlds |
| **Euclidean** | 8×8 | clamp | euclidean | 2 | 91 | Diagonal-aware distance |
| **Aspatial** | N/A | N/A | N/A | 0 | 27 | Pure resource management |

## Detailed Comparisons

### Standard vs Toroidal (Boundary Behavior)

**Standard (clamp boundary)**:
- Hard walls at grid edges
- Agent position clamped to [0, width-1] and [0, height-1]
- Corners are "dead ends" (agent can be trapped)
- Shortest path is always straight-line within grid

**Toroidal (wrap boundary)**:
- No walls - edges wrap to opposite side
- Agent at [0, y] moving LEFT wraps to [width-1, y]
- No corners or edges (infinite grid feel)
- Shortest path may cross edges (wrap distance < straight)

**Learning implications**:
- Standard: Agent learns edges exist, avoids corners
- Toroidal: Agent learns wraparound shortcuts, no edge aversion

**Example file**: `substrate-toroidal-grid.yaml`

---

### Manhattan vs Euclidean (Distance Metric)

**Manhattan (L1 norm)**:
- Distance = |x1-x2| + |y1-y2|
- Only cardinal directions (no diagonal shortcuts)
- Example: [0,0] to [3,4] = 7 steps
- Natural for 4-directional movement

**Euclidean (L2 norm)**:
- Distance = sqrt((x1-x2)² + (y1-y2)²)
- Straight-line distance (diagonal-aware)
- Example: [0,0] to [3,4] = 5.0 steps
- More realistic spatial reasoning

**Learning implications**:
- Manhattan: Agent learns grid topology, cardinal preferences
- Euclidean: Agent learns diagonal proximity, may prefer diagonal paths

**Caution**: If movement is 4-directional (UP/DOWN/LEFT/RIGHT only), euclidean distance creates mismatch between perceived distance and actual steps required.

**Example file**: `substrate-euclidean-distance.yaml`

---

### Grid vs Aspatial (Positioning)

**Grid (spatial substrate)**:
- Agents have positions on 2D grid
- Navigation via movement actions (UP/DOWN/LEFT/RIGHT)
- Spatial planning required (find affordances, navigate)
- Observation includes grid encoding (64 dims for 8×8)

**Aspatial (no positioning)**:
- Agents have NO position (pure state machine)
- No navigation (all affordances equally accessible)
- Pure resource management (energy/health cycles)
- Observation has no grid encoding (0 dims)

**Learning implications**:
- Grid: Agent must learn navigation + resource management
- Aspatial: Agent learns only resource management (faster convergence)

**Pedagogical value**: Reveals that meters are the TRUE universe - positioning is just an overlay. Spatial substrate is OPTIONAL.

**Example file**: `substrate-aspatial.yaml`

---

## Observation Dimension Calculation

Observation dimension varies by substrate type and grid size:

**Grid substrate**:
```
obs_dim = (width × height) + 8 meters + 15 affordances + 4 temporal
```

Examples:
- 3×3 grid: 9 + 8 + 15 + 4 = 36 dims
- 7×7 grid: 49 + 8 + 15 + 4 = 76 dims
- 8×8 grid: 64 + 8 + 15 + 4 = 91 dims

**Aspatial substrate**:
```
obs_dim = 0 + 8 meters + 15 affordances + 4 temporal = 27 dims
```

## When to Use Each Substrate

**Standard Grid (clamp + manhattan)**:
- ✅ Default for curriculum training
- ✅ Clear spatial boundaries (pedagogical)
- ✅ Matches 4-directional movement mechanics
- ✅ Most common in existing configs

**Toroidal Grid (wrap + manhattan)**:
- ✅ Pac-Man style games
- ✅ Infinite grid feel
- ✅ Comparing boundary behaviors
- ⚠️ May confuse beginners (no edges)

**Euclidean Distance (clamp + euclidean)**:
- ✅ Realistic spatial reasoning
- ✅ Diagonal-aware proximity
- ✅ Comparing distance metrics
- ⚠️ Mismatch with 4-directional movement

**Aspatial (no positioning)**:
- ✅ Pure resource management baseline
- ✅ Faster training (no navigation)
- ✅ Isolating resource optimization
- ⚠️ Cannot enable temporal mechanics (requires positions)

## Changing ONE Parameter

**Pedagogical insight**: Changing ONE parameter dramatically alters gameplay.

Examples:
- Change `boundary: clamp` → `boundary: wrap`: Agent learns wraparound shortcuts
- Change `distance_metric: manhattan` → `distance_metric: euclidean`: Diagonal affordances feel "closer"
- Change `type: grid` → `type: aspatial`: Navigation disappears entirely

**Teaching value**: Shows how spatial abstractions shape agent behavior. Students learn that substrate is a DESIGN CHOICE, not a fixed constraint.

---

## Creating Your Own Substrate

1. Copy `configs/templates/substrate.yaml`
2. Modify `width`, `height`, `boundary`, `distance_metric` as needed
3. Validate with: `uv run pytest tests/test_townlet/unit/test_substrate_configs.py`
4. Run training to verify behavior

See `configs/templates/substrate.yaml` for comprehensive parameter documentation.
```

**Expected**: Comparison documentation created

---

#### Step 6: Commit example substrate configs

**Command**:
```bash
cd /home/john/hamlet
git add docs/examples/
git commit -m "docs: add substrate configuration examples

Created reference examples for alternative substrate configurations:

1. substrate-toroidal-grid.yaml:
   - Wraparound boundary (Pac-Man style)
   - Shows impact of boundary parameter

2. substrate-aspatial.yaml:
   - No positioning (pure state machine)
   - Reveals that substrate is OPTIONAL

3. substrate-euclidean-distance.yaml:
   - Diagonal-aware distance metric
   - Shows impact of distance metric parameter

4. substrate-comparison.md:
   - Behavioral comparison table
   - When to use each substrate type
   - Observation dimension calculations
   - Pedagogical insights

Operators can reference these examples when creating custom configs.

Part of TASK-002A Phase 6 (Config Migration)."
```

**Expected**: Clean commit with examples

---

### Task 6.10: Update Documentation

**Purpose**: Update CLAUDE.md and project documentation with substrate.yaml structure

**Files**:
- `CLAUDE.md` (MODIFY)
- `docs/TASK-002A-configurable-spatial-substrates.md` (MODIFY - update status)

**Estimated Time**: 30 minutes

---

#### Step 1: Update CLAUDE.md config pack structure

**Action**: Add substrate.yaml to config pack documentation

**Modify**: `CLAUDE.md`

**Find** (around line 700):
```markdown
### Config Pack Structure (UNIVERSE_AS_CODE)

Each config pack directory contains:

```
configs/L0_0_minimal/
├── bars.yaml         # Meter definitions (energy, health, money, etc.)
├── cascades.yaml     # Meter relationships (low satiation → drains energy)
├── affordances.yaml  # Interaction definitions (Bed, Hospital, Job, etc.)
├── cues.yaml         # UI metadata for visualization
└── training.yaml     # Hyperparameters and enabled affordances
```
```

**Replace with**:
```markdown
### Config Pack Structure (UNIVERSE_AS_CODE)

Each config pack directory contains:

```
configs/L0_0_minimal/
├── substrate.yaml    # Spatial substrate definition (NEW in Phase 6)
├── bars.yaml         # Meter definitions (energy, health, money, etc.)
├── cascades.yaml     # Meter relationships (low satiation → drains energy)
├── affordances.yaml  # Interaction definitions (Bed, Hospital, Job, etc.)
├── cues.yaml         # UI metadata for visualization
└── training.yaml     # Hyperparameters and enabled affordances
```

**Key principle**: Everything configurable via YAML (UNIVERSE_AS_CODE). The system loads and validates these files at startup.
```

**Expected**: Config pack structure updated with substrate.yaml

---

#### Step 2: Add substrate.yaml structure section to CLAUDE.md

**Action**: Document substrate.yaml schema and options

**Modify**: `CLAUDE.md`

**Add after "Config Pack Structure" section** (around line 720):

```markdown
### substrate.yaml Structure

**NEW in TASK-002A Phase 6**: Defines spatial substrate (coordinate system, topology, boundaries).

**Purpose**: Makes spatial parameters explicit (no hidden defaults per PDR-002).

**Required Fields**:
```yaml
version: "1.0"                    # Config version
description: "..."                # Human-readable description
type: "grid"                      # "grid" (spatial) or "aspatial" (no positioning)

# For type="grid":
grid:
  topology: "square"              # Grid topology (currently only "square")
  width: 8                        # Grid width (columns)
  height: 8                       # Grid height (rows)
  boundary: "clamp"               # "clamp" (walls), "wrap" (toroidal), "bounce"
  distance_metric: "manhattan"    # "manhattan" (L1), "euclidean" (L2), "chebyshev" (L∞)
```

**Behavioral Impact**:
- `width × height`: Grid size (affects observation dimension)
- `boundary`: Movement at edges (clamp=walls, wrap=Pac-Man, bounce=reflection)
- `distance_metric`: Proximity calculations (manhattan=cardinal, euclidean=diagonal-aware)

**Observation Dimension**:
- Grid substrate: `obs_dim = (width × height) + 8 meters + 15 affordances + 4 temporal`
- Aspatial substrate: `obs_dim = 0 + 8 meters + 15 affordances + 4 temporal = 27 dims`

**Examples**:
- Standard grid (L1): 8×8, clamp, manhattan → 91 dims
- Toroidal grid: 8×8, wrap, manhattan → 91 dims (same obs, different behavior)
- Aspatial: No positioning → 27 dims

**Key Insight**: Substrate is OPTIONAL. Aspatial universes (pure state machines) reveal that meters are the true universe - positioning is just an overlay.

**See also**: `docs/examples/substrate-comparison.md` for behavioral comparisons.
```

**Expected**: substrate.yaml documentation added to CLAUDE.md

---

#### Step 3: Update training.yaml structure section

**Action**: Note that grid_size moved to substrate.yaml

**Modify**: `CLAUDE.md`

**Find** (around line 820):
```markdown
### Config Structure (training.yaml)

```yaml
environment:
  grid_size: 8
  partial_observability: false
  vision_range: 2
```

**Replace with**:
```markdown
### Config Structure (training.yaml)

```yaml
environment:
  # Note: grid_size moved to substrate.yaml (as width/height)
  partial_observability: false  # true for POMDP (Level 2+)
  vision_range: 2               # 5×5 window when partial_observability=true
```

**Historical note**: `environment.grid_size` was removed in Phase 6 (TASK-002A). Grid dimensions now specified in `substrate.yaml` as `grid.width` and `grid.height` for explicit configuration per UNIVERSE_AS_CODE principle.
```

**Expected**: training.yaml documentation updated to reflect grid_size migration

---

#### Step 4: Update TASK-002A status

**Action**: Mark Phase 6 as complete in task document

**Modify**: `docs/TASK-002A-configurable-spatial-substrates.md`

**Find** (Status section, usually near top):
```markdown
- [x] Phase 1: Abstract Substrate Interface
- [x] Phase 2: Substrate Configuration Schema
- [ ] Phase 3: Environment Integration
- [ ] Phase 4: Position Management Refactoring
- [ ] Phase 5: Observation Builder Update
- [ ] Phase 6: Config Migration
```

**Replace with**:
```markdown
- [x] Phase 1: Abstract Substrate Interface
- [x] Phase 2: Substrate Configuration Schema
- [ ] Phase 3: Environment Integration
- [ ] Phase 4: Position Management Refactoring
- [ ] Phase 5: Observation Builder Update
- [x] Phase 6: Config Migration ← **COMPLETED**
```

**Add completion notes** (at end of Phase 6 section):
```markdown
### Phase 6 Completion Notes

**Completed**: 2025-11-05
**Effort**: 4 hours (as estimated)

**Deliverables**:
- ✅ 7 substrate.yaml files (L0, L0.5, L1, L2, L3, templates, test)
- ✅ Unit tests (schema validation, behavioral equivalence)
- ✅ Integration tests (environment loading - ready for Phase 3)
- ✅ Example configs (toroidal, aspatial, euclidean)
- ✅ Documentation updates (CLAUDE.md, comparison docs)

**Behavioral Verification**:
- All substrate.yaml files pass schema validation ✅
- Observation dimensions unchanged (behavioral equivalence verified) ✅
- All configs use clamp boundary + manhattan distance (as expected) ✅

**Phase 3 Readiness**:
Phase 6 complete. Phase 3 (Environment Integration) can now proceed.
Phase 3 will enforce substrate.yaml requirement in VectorizedEnv.
```

**Expected**: Task status updated

---

#### Step 5: Commit documentation updates

**Command**:
```bash
cd /home/john/hamlet
git add CLAUDE.md docs/TASK-002A-configurable-spatial-substrates.md
git commit -m "docs: update CLAUDE.md and task status for Phase 6 completion

Updated project documentation to reflect substrate.yaml addition:

CLAUDE.md changes:
- Added substrate.yaml to config pack structure diagram
- Documented substrate.yaml schema and required fields
- Explained behavioral impact of each parameter
- Added observation dimension calculation formulas
- Noted grid_size migration from training.yaml to substrate.yaml

TASK-002A updates:
- Marked Phase 6 as complete
- Added completion notes with deliverables
- Verified behavioral equivalence
- Confirmed Phase 3 readiness

Phase 6 complete. Ready for Phase 3 (Environment Integration).

Part of TASK-002A Phase 6 (Config Migration)."
```

**Expected**: Clean commit with documentation updates

---

### Task 6.11: Validation and Smoke Testing

**Purpose**: Verify all substrate.yaml files are valid and configs still work

**Files**: None (testing only)

**Estimated Time**: 30 minutes

---

#### Step 1: Run all unit tests

**Command**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run pytest tests/test_townlet/unit/test_substrate_configs.py -v
```

**Expected**: ALL PASS (7/7 configs valid)

**Verification checklist**:
- [x] Schema validation passes for all configs
- [x] Behavioral equivalence verified (clamp + manhattan)
- [x] No-defaults principle enforced
- [x] File existence checks pass

---

#### Step 2: Validate substrate configs programmatically

**Action**: Create validation script for operators

**Create**: `scripts/validate_substrate_configs.py`

```python
#!/usr/bin/env python3
"""Validate all substrate.yaml configs before training.

Usage:
    python scripts/validate_substrate_configs.py

Validates:
- Schema correctness (Pydantic validation)
- File existence (all production configs have substrate.yaml)
- Behavioral equivalence (expected dimensions)
- No-defaults compliance (all required fields present)

Exit codes:
    0: All configs valid
    1: Validation errors found
"""
import sys
from pathlib import Path
from townlet.substrate.config import load_substrate_config

# Production config packs (must have substrate.yaml)
PRODUCTION_CONFIGS = [
    "L0_0_minimal",
    "L0_5_dual_resource",
    "L1_full_observability",
    "L2_partial_observability",
    "L3_temporal_mechanics",
    "test",
]

# Expected dimensions (for behavioral equivalence)
EXPECTED_DIMS = {
    "L0_0_minimal": (3, 3, 9),  # (width, height, grid_dims)
    "L0_5_dual_resource": (7, 7, 49),
    "L1_full_observability": (8, 8, 64),
    "L2_partial_observability": (8, 8, 64),
    "L3_temporal_mechanics": (8, 8, 64),
    "test": (8, 8, 64),
}


def validate_config(config_name: str) -> bool:
    """Validate single config pack's substrate.yaml.

    Returns:
        True if valid, False if errors found
    """
    substrate_path = Path("configs") / config_name / "substrate.yaml"

    # Check file exists
    if not substrate_path.exists():
        print(f"❌ {config_name}: substrate.yaml not found")
        return False

    try:
        # Load and validate schema
        config = load_substrate_config(substrate_path)

        # Verify required fields
        assert config.version == "1.0", f"Invalid version: {config.version}"
        assert config.description, "Missing description"
        assert config.type in ["grid", "aspatial"], f"Invalid type: {config.type}"

        if config.type == "grid":
            # Verify grid configuration
            assert config.grid is not None, "Missing grid config"
            assert config.grid.topology == "square", "Only square topology supported"
            assert config.grid.width > 0, "Width must be positive"
            assert config.grid.height > 0, "Height must be positive"
            assert config.grid.boundary in ["clamp", "wrap", "bounce"], "Invalid boundary"
            assert config.grid.distance_metric in [
                "manhattan",
                "euclidean",
                "chebyshev",
            ], "Invalid distance metric"

            # Verify expected dimensions (behavioral equivalence)
            if config_name in EXPECTED_DIMS:
                expected_width, expected_height, expected_grid_dims = EXPECTED_DIMS[
                    config_name
                ]
                if (
                    config.grid.width != expected_width
                    or config.grid.height != expected_height
                ):
                    print(
                        f"⚠️  {config_name}: Unexpected dimensions {config.grid.width}×{config.grid.height} "
                        f"(expected {expected_width}×{expected_height})"
                    )
                    return False

        print(f"✅ {config_name}: Valid")
        return True

    except Exception as e:
        print(f"❌ {config_name}: {e}")
        return False


def main():
    """Validate all production configs."""
    print("Validating substrate.yaml configs...\n")

    all_valid = True
    for config_name in PRODUCTION_CONFIGS:
        valid = validate_config(config_name)
        all_valid = all_valid and valid

    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All configs valid!")
        sys.exit(0)
    else:
        print("❌ Validation errors found")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Run script**:
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python scripts/validate_substrate_configs.py
```

**Expected**:
```
Validating substrate.yaml configs...

✅ L0_0_minimal: Valid
✅ L0_5_dual_resource: Valid
✅ L1_full_observability: Valid
✅ L2_partial_observability: Valid
✅ L3_temporal_mechanics: Valid
✅ test: Valid

============================================================
✅ All configs valid!
```

---

#### Step 3: Commit validation script

**Command**:
```bash
git add scripts/validate_substrate_configs.py
git commit -m "chore: add substrate config validation script

Added pre-training validation script for substrate.yaml files:

- Validates schema correctness (Pydantic)
- Checks file existence for all production configs
- Verifies behavioral equivalence (expected dimensions)
- Provides clear error messages

Usage:
    python scripts/validate_substrate_configs.py

Operators can run before training to catch config errors early.

Part of TASK-002A Phase 6 (Config Migration)."
```

**Expected**: Clean commit with validation tool

---

#### Step 4: Document smoke test procedure (manual)

**Action**: Document how to verify configs still train

**Note**: Full smoke testing requires Phase 3 completion (environment loads substrate). This step documents the procedure for future verification.

**Create**: `docs/testing/substrate-smoke-test.md`

```markdown
# Substrate Migration Smoke Test Procedure

**Purpose**: Verify configs with substrate.yaml still train correctly

**When to run**: After Phase 3 completion (environment loads substrate)

**Prerequisites**:
- Phase 3 complete (VectorizedEnv loads substrate.yaml)
- All substrate.yaml files created (Phase 6)

---

## Quick Smoke Test (50 episodes per config)

Run short training sessions to verify no crashes or errors:

```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Test each config with 50 episodes (fast validation)
for config in L0_0_minimal L0_5_dual_resource L1_full_observability L2_partial_observability L3_temporal_mechanics; do
    echo "=========================================="
    echo "Smoke testing: $config"
    echo "=========================================="

    uv run python -m townlet.demo.runner \
        --config configs/$config \
        --max-episodes 50 \
        || { echo "❌ $config failed!"; exit 1; }

    echo "✅ $config passed!"
done

echo ""
echo "=========================================="
echo "✅ All smoke tests passed!"
echo "=========================================="
```

**Expected**: All configs train without errors for 50 episodes

---

## Full Integration Test (test config)

Run complete test config (200 episodes):

```bash
uv run python -m townlet.demo.runner \
    --config configs/test \
    || { echo "❌ test config failed!"; exit 1; }

echo "✅ test config passed!"
```

**Expected**: Test config completes 200 episodes successfully

---

## Verification Checklist

After smoke testing, verify:

- [x] All configs load substrate.yaml without errors
- [x] Environment initializes with correct dimensions
- [x] Observation dimensions match expected values
- [x] Training completes without crashes
- [x] Agents survive and learn (survival steps increase)
- [x] Checkpoints save successfully

---

## Troubleshooting

**Error: "substrate.yaml not found"**
- Verify substrate.yaml exists in config pack directory
- Check Phase 6 completed for all configs

**Error: "Invalid substrate schema"**
- Run validation script: `python scripts/validate_substrate_configs.py`
- Check substrate.yaml against schema (Phase 2)

**Error: "Observation dimension mismatch"**
- Verify grid dimensions match expected: width × height
- Check training.yaml doesn't override grid_size (removed in Phase 6)

**Error: "Training crashes during episode"**
- Check boundary behavior (clamp/wrap/bounce)
- Verify distance metric (manhattan/euclidean/chebyshev)
- Review Phase 3 integration (position management)

---

## Success Criteria

Smoke test succeeds if:
1. All 5 curriculum configs train for 50 episodes without errors
2. Test config completes full 200 episodes
3. Observation dimensions match expected values
4. Survival steps increase (agents learning)
5. Checkpoints save and load successfully

If all criteria met: Phase 6 migration successful! ✅
```

**Expected**: Smoke test procedure documented

---

#### Step 5: Commit smoke test documentation

**Command**:
```bash
git add docs/testing/substrate-smoke-test.md
git commit -m "docs: add substrate smoke test procedure

Documented smoke testing procedure for substrate.yaml migration:

- Quick smoke test (50 episodes per config)
- Full integration test (test config, 200 episodes)
- Verification checklist
- Troubleshooting guide

Run after Phase 3 completion to verify configs train correctly.

Part of TASK-002A Phase 6 (Config Migration)."
```

**Expected**: Clean commit with test procedure

---

### Task 6.12: Phase 6 Completion Summary

**Purpose**: Final verification and completion documentation

**Files**: None (summary only)

**Estimated Time**: 15 minutes

---

#### Step 1: Run final validation

**Commands**:
```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# 1. Validate all substrate configs
echo "1. Validating substrate configs..."
python scripts/validate_substrate_configs.py
echo ""

# 2. Run all substrate unit tests
echo "2. Running unit tests..."
uv run pytest tests/test_townlet/unit/test_substrate_configs.py -v
echo ""

# 3. Verify file structure
echo "3. Checking file structure..."
for config in L0_0_minimal L0_5_dual_resource L1_full_observability L2_partial_observability L3_temporal_mechanics templates test; do
    if [ -f "configs/$config/substrate.yaml" ]; then
        echo "✅ configs/$config/substrate.yaml exists"
    else
        echo "❌ configs/$config/substrate.yaml MISSING"
    fi
done
echo ""

# 4. Verify documentation
echo "4. Checking documentation..."
test -f "docs/examples/substrate-toroidal-grid.yaml" && echo "✅ Toroidal example exists"
test -f "docs/examples/substrate-aspatial.yaml" && echo "✅ Aspatial example exists"
test -f "docs/examples/substrate-euclidean-distance.yaml" && echo "✅ Euclidean example exists"
test -f "docs/examples/substrate-comparison.md" && echo "✅ Comparison doc exists"
test -f "docs/testing/substrate-smoke-test.md" && echo "✅ Smoke test doc exists"
echo ""

echo "========================================"
echo "Phase 6 validation complete!"
echo "========================================"
```

**Expected**: All checks pass

---

#### Step 2: Generate Phase 6 completion report

**Action**: Summarize deliverables and verify success criteria

**Create**: `docs/completion/phase6-completion-report.md`

```markdown
# TASK-002A Phase 6 Completion Report

**Date**: 2025-11-05
**Phase**: 6 (Config Migration - Create substrate.yaml Files)
**Status**: ✅ COMPLETE
**Effort**: 4 hours (as estimated)

---

## Deliverables

### 1. substrate.yaml Files (7/7 complete)

| Config Pack | Grid Size | File Created | Schema Valid | Tests Pass |
|-------------|-----------|--------------|--------------|------------|
| L0_0_minimal | 3×3 | ✅ | ✅ | ✅ |
| L0_5_dual_resource | 7×7 | ✅ | ✅ | ✅ |
| L1_full_observability | 8×8 | ✅ | ✅ | ✅ |
| L2_partial_observability | 8×8 | ✅ | ✅ | ✅ |
| L3_temporal_mechanics | 8×8 | ✅ | ✅ | ✅ |
| templates | 8×8 | ✅ | ✅ | ✅ |
| test | 8×8 | ✅ | ✅ | ✅ |

### 2. Test Infrastructure

- ✅ Unit tests: `tests/test_townlet/unit/test_substrate_configs.py`
  - Schema validation (7 configs)
  - Behavioral equivalence verification
  - No-defaults principle enforcement
  - File existence checks

- ✅ Integration tests: `tests/test_townlet/integration/test_substrate_migration.py`
  - Environment loading (ready for Phase 3)
  - Observation dimension validation
  - Boundary behavior verification
  - Distance metric validation

### 3. Example Configurations

- ✅ `docs/examples/substrate-toroidal-grid.yaml` (wraparound boundary)
- ✅ `docs/examples/substrate-aspatial.yaml` (no positioning)
- ✅ `docs/examples/substrate-euclidean-distance.yaml` (diagonal-aware distance)
- ✅ `docs/examples/substrate-comparison.md` (behavioral comparisons)

### 4. Documentation

- ✅ CLAUDE.md updated (config structure, substrate.yaml schema)
- ✅ TASK-002A status updated (Phase 6 marked complete)
- ✅ Smoke test procedure documented
- ✅ Validation script created

### 5. Tooling

- ✅ `scripts/validate_substrate_configs.py` (pre-training validation)
- ✅ `docs/testing/substrate-smoke-test.md` (smoke test procedure)

---

## Success Criteria Verification

### ✅ All substrate.yaml files created (7/7)
- L0_0_minimal: 3×3 grid, clamp boundary, manhattan distance
- L0_5_dual_resource: 7×7 grid, clamp boundary, manhattan distance
- L1, L2, L3, templates, test: 8×8 grid, clamp boundary, manhattan distance

### ✅ All configs pass schema validation
```bash
$ python scripts/validate_substrate_configs.py
✅ L0_0_minimal: Valid
✅ L0_5_dual_resource: Valid
✅ L1_full_observability: Valid
✅ L2_partial_observability: Valid
✅ L3_temporal_mechanics: Valid
✅ test: Valid
✅ All configs valid!
```

### ✅ Behavioral equivalence verified
| Config | Expected Obs Dim | Actual Obs Dim | Match |
|--------|-----------------|----------------|-------|
| L0_0_minimal | 36 (9+8+15+4) | 36 | ✅ |
| L0_5_dual_resource | 76 (49+8+15+4) | 76 | ✅ |
| L1/L2/L3/test | 91 (64+8+15+4) | 91 | ✅ |

### ✅ All unit tests pass
```bash
$ uv run pytest tests/test_townlet/unit/test_substrate_configs.py -v
test_substrate_config_schema_valid[L0_0_minimal-3-3-9] PASSED
test_substrate_config_schema_valid[L0_5_dual_resource-7-7-49] PASSED
test_substrate_config_schema_valid[L1_full_observability-8-8-64] PASSED
test_substrate_config_schema_valid[L2_partial_observability-8-8-64] PASSED
test_substrate_config_schema_valid[L3_temporal_mechanics-8-8-64] PASSED
test_substrate_config_schema_valid[templates-8-8-64] PASSED
test_substrate_config_schema_valid[test-8-8-64] PASSED
test_substrate_config_behavioral_equivalence[...] PASSED (7 tests)
test_substrate_config_no_defaults PASSED
test_substrate_config_file_exists PASSED

==================== 16 passed ====================
```

### ✅ Example configs created (3 examples + comparison doc)
- Toroidal grid (wraparound)
- Aspatial (no positioning)
- Euclidean distance (diagonal-aware)
- Comparison documentation

### ✅ Documentation updated
- CLAUDE.md: Config structure, substrate.yaml schema
- TASK-002A: Phase 6 marked complete
- Smoke test procedure documented

---

## Phase 3 Readiness

**Phase 6 COMPLETE. Phase 3 can now proceed.**

**Rationale**:
- All substrate.yaml files created ✅
- Schema validation passing ✅
- Behavioral equivalence verified ✅
- Tests ready for Phase 3 integration ✅

**Phase 3 will**:
1. Load substrate.yaml in VectorizedEnv.__init__()
2. Enforce substrate.yaml requirement (breaking change)
3. Replace hardcoded grid_size with substrate.width/height
4. Enable configurable boundaries and distance metrics

**No blockers for Phase 3** ✅

---

## Breaking Changes Summary

**Config Pack Structure**:
- **BEFORE Phase 6**: No substrate.yaml (spatial behavior hardcoded)
- **AFTER Phase 6**: substrate.yaml required (explicit configuration)

**Impact**:
- Operators creating new configs MUST include substrate.yaml
- Existing configs updated in Phase 6 (no operator action needed)
- Clear error message if substrate.yaml missing (after Phase 3)

**Migration Path**:
- Existing configs: Already migrated (Phase 6) ✅
- New configs: Use `configs/templates/substrate.yaml` as starting point

---

## Lessons Learned

### What Went Well
1. **All configs use identical behavior** - Migration was mechanical (copy template, edit dimensions)
2. **Test-driven approach** - Writing tests first caught issues early
3. **Comprehensive documentation** - Template and examples make operator onboarding easy
4. **Validation tooling** - Script provides fast pre-training validation

### Challenges
1. **Phase ordering critical** - Phase 6 MUST precede Phase 3 to avoid breakage
2. **Documentation volume** - Template required extensive comments for operator clarity
3. **Behavioral equivalence testing** - Verifying obs_dim calculations manually was tedious

### Recommendations for Future Phases
1. **Maintain test-first approach** - Write failing tests before implementation
2. **Document examples early** - Examples clarify intent and prevent misunderstandings
3. **Validation scripts essential** - Pre-training validation saves debugging time
4. **Breaking changes need clear communication** - Template and error messages critical

---

## Next Steps

### Immediate (Phase 3)
1. Integrate substrate loading in VectorizedEnv.__init__()
2. Replace hardcoded grid_size with substrate.width/height
3. Add substrate.yaml requirement validation
4. Run integration tests (should pass after Phase 3)

### Future (Phase 4-5)
1. Refactor position management to use substrate.position_dim
2. Update observation builder for substrate-aware encoding
3. Run smoke tests (50 episodes per config)
4. Verify behavioral equivalence in training

---

## Conclusion

**Phase 6 complete.** All substrate.yaml files created, validated, and documented.

**Phase 3 ready to proceed.** No blockers.

**Breaking changes authorized and communicated.** Operators have clear migration path via template.

---

**Signed off**: Phase 6 Implementation Complete ✅
**Date**: 2025-11-05
**Next Phase**: Phase 3 (Environment Integration)
```

**Expected**: Comprehensive completion report

---

#### Step 3: Final commit - Phase 6 complete

**Command**:
```bash
cd /home/john/hamlet
git add docs/completion/phase6-completion-report.md
git commit -m "docs: Phase 6 completion report

Phase 6 (Config Migration) complete:

Deliverables:
- 7/7 substrate.yaml files created (L0 through test)
- Unit tests (schema validation, behavioral equivalence)
- Integration tests (ready for Phase 3)
- Example configs (toroidal, aspatial, euclidean)
- Documentation (CLAUDE.md, comparison docs, smoke tests)
- Tooling (validation script, smoke test procedure)

Success criteria met:
- All configs pass schema validation ✅
- Behavioral equivalence verified (obs dims unchanged) ✅
- All unit tests pass (16/16) ✅
- Documentation complete ✅

Phase 3 readiness:
- No blockers ✅
- All substrate.yaml files ready ✅
- Tests ready for integration ✅

Breaking changes: Config packs now require substrate.yaml (after Phase 3).
Migration path: Use configs/templates/substrate.yaml for new configs.

TASK-002A Phase 6 COMPLETE."
```

**Expected**: Final commit, phase complete

---

## Phase 6 Summary

### Total Effort: 4 hours (as estimated)

**Task Breakdown**:
- Task 6.1: Test infrastructure (45 min)
- Task 6.2: L0_0_minimal (20 min)
- Task 6.3: L0_5_dual_resource (15 min)
- Task 6.4: L1_full_observability (15 min)
- Task 6.5: L2_partial_observability (15 min)
- Task 6.6: L3_temporal_mechanics (15 min)
- Task 6.7: test config (15 min)
- Task 6.8: templates (30 min)
- Task 6.9: Example configs (30 min)
- Task 6.10: Documentation (30 min)
- Task 6.11: Validation (30 min)
- Task 6.12: Completion (15 min)

**Total**: ~4 hours

---

### Deliverables Checklist

**Config Files**:
- [x] L0_0_minimal/substrate.yaml (3×3 grid)
- [x] L0_5_dual_resource/substrate.yaml (7×7 grid)
- [x] L1_full_observability/substrate.yaml (8×8 grid)
- [x] L2_partial_observability/substrate.yaml (8×8 grid)
- [x] L3_temporal_mechanics/substrate.yaml (8×8 grid)
- [x] templates/substrate.yaml (template with extensive docs)
- [x] test/substrate.yaml (8×8 grid for CI)

**Test Infrastructure**:
- [x] Unit tests (test_substrate_configs.py)
- [x] Integration tests (test_substrate_migration.py)
- [x] Validation script (validate_substrate_configs.py)
- [x] Smoke test procedure (substrate-smoke-test.md)

**Examples and Documentation**:
- [x] Toroidal grid example
- [x] Aspatial example
- [x] Euclidean distance example
- [x] Comparison documentation
- [x] CLAUDE.md updates
- [x] TASK-002A status update
- [x] Completion report

**Verification**:
- [x] All configs pass schema validation
- [x] All unit tests pass (16/16)
- [x] Behavioral equivalence verified
- [x] Documentation complete

---

### Success Criteria

**✅ Phase 6 is COMPLETE when**:
1. All 7 config packs have substrate.yaml ✅
2. All substrate.yaml files pass schema validation ✅
3. Template substrate.yaml is well-documented ✅
4. CLAUDE.md updated with substrate.yaml structure ✅
5. Example documentation created ✅
6. Ready for Phase 3 (no blockers) ✅

---

### Phase 3 Readiness

**READY TO PROCEED**

Phase 6 created all substrate.yaml files WITHOUT changing Python code.
Phase 3 will now integrate substrate loading in VectorizedEnv.

**Correct execution order maintained**:
1. ✅ Phase 1: Substrate abstraction
2. ✅ Phase 2: Config schema
3. ✅ **Phase 6: Config migration** ← JUST COMPLETED
4. → **Phase 3: Environment integration** ← PROCEED NEXT

**No blockers. Phase 3 can proceed immediately.**

---

### Breaking Changes Notice (Final)

**After Phase 3 completes**:
- VectorizedEnv will REQUIRE substrate.yaml
- Config packs without substrate.yaml will fail with clear error
- No backward compatibility (authorized by user)

**Migration path**:
- Existing configs: Already migrated (Phase 6) ✅
- New configs: Copy `configs/templates/substrate.yaml` and customize

**Error message** (Phase 3):
```
FileNotFoundError: substrate.yaml not found in configs/my_config/

Config packs must include substrate.yaml (added in TASK-002A Phase 6).
Copy configs/templates/substrate.yaml as starting point.

See docs/examples/substrate-comparison.md for examples.
```

---

## Appendix: Quick Reference

### Validation Commands

```bash
# Validate all substrate configs
python scripts/validate_substrate_configs.py

# Run unit tests
uv run pytest tests/test_townlet/unit/test_substrate_configs.py -v

# Run integration tests (after Phase 3)
uv run pytest tests/test_townlet/integration/test_substrate_migration.py -v

# Smoke test (after Phase 3)
bash docs/testing/substrate-smoke-test.md
```

### File Locations

```
configs/
├── L0_0_minimal/substrate.yaml         # 3×3 grid
├── L0_5_dual_resource/substrate.yaml   # 7×7 grid
├── L1_full_observability/substrate.yaml # 8×8 grid
├── L2_partial_observability/substrate.yaml # 8×8 grid
├── L3_temporal_mechanics/substrate.yaml # 8×8 grid
├── templates/substrate.yaml            # Template with docs
└── test/substrate.yaml                 # 8×8 grid for CI

docs/
├── examples/
│   ├── substrate-toroidal-grid.yaml
│   ├── substrate-aspatial.yaml
│   ├── substrate-euclidean-distance.yaml
│   └── substrate-comparison.md
├── testing/
│   └── substrate-smoke-test.md
└── completion/
    └── phase6-completion-report.md

scripts/
└── validate_substrate_configs.py

tests/
├── test_townlet/
│   ├── unit/
│   │   └── test_substrate_configs.py
│   └── integration/
│       └── test_substrate_migration.py
```

---

**END OF PHASE 6 IMPLEMENTATION PLAN**
