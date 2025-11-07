# TASK-002C: TDD Implementation Plan
# Variable & Feature System (VFS) - Phase 1

**Created**: 2025-11-07
**Updated**: 2025-11-07 (Post peer review - READY FOR IMPLEMENTATION)
**Estimated Total Effort**: 40-52 hours (5-6.5 days)
**Approach**: Test-Driven Development (Red-Green-Refactor)

**CRITICAL CHANGES** (from peer review):
- **BLOCKER FIX**: Cycle 0 now documentation-only (no circular dependency on Cycle 3)
- Added Cycle 8: Integration Specification (2h) - was orphaned
- Clarified reference variables are TEMPORARY scaffolding (can be removed after Cycle 5)
- Added checkpoint backup to Pre-Implementation Setup (safety)
- Renamed VFS observation class: `ObservationBuilder` → `VFSObservationSpecBuilder`
- Updated total effort: 38-50h → 40-52h (includes integration spec)
- Updated scope semantics clarification for registry

---

## Peer Review Summary (2025-11-07)

**Status**: ✅ **READY FOR IMPLEMENTATION** (all blockers resolved)

**Key Changes Made**:

1. **BLOCKER FIX**: Eliminated Cycle 0 circular dependency
   - Was: Cycle 0 validation script depends on Cycle 3 VFSObservationSpecBuilder
   - Now: Cycle 0 is documentation-only, automated validation deferred to Cycle 5

2. **ORPHANED TASK FIX**: Added Cycle 8 (Integration Specification)
   - Was: Integration spec mentioned but not in cycle list
   - Now: Proper Cycle 8 with integration tests and documentation

3. **SAFETY ADDITION**: Checkpoint backup in Pre-Implementation Setup
   - Why: Dimension mismatch in Cycle 5 could corrupt checkpoints
   - Action: Backup all checkpoints before running any validation tests

4. **CLARITY FIX**: Reference variables are temporary scaffolding
   - Cycle 0 creates them for validation
   - Cycle 7 deletes them after validation passes
   - Only `configs/templates/variables.yaml` is permanent

5. **EFFORT CORRECTION**: Updated total from 38-50h → 40-52h
   - Includes integration specification (was missing)
   - Includes checkpoint backup time
   - Realistic estimate based on peer review

**Assessment**:
- Complexity: Appropriate for foundational infrastructure
- Risks: All CRITICAL risks mitigated with Cycle 0/5 changes
- Executability: All prerequisites met, integration points clean
- Confidence: 8.5/10 (was unscored)

---

## Overview

This plan breaks TASK-002C into **TDD cycles**, each following the Red-Green-Refactor pattern:
1. **RED**: Write failing tests first
2. **GREEN**: Implement minimal code to pass tests
3. **REFACTOR**: Clean up implementation while keeping tests green

**Key Principle**: No production code without a failing test first.

---

## Pre-Implementation Setup (45 minutes)

### 1. Backup Existing Checkpoints (CRITICAL SAFETY)

**⚠️ REQUIRED**: Backup checkpoints BEFORE running any dimension validation tests.

```bash
# Create backup directory
mkdir -p ../checkpoint_backups

# Backup all existing checkpoints
cp -r checkpoints_* ../checkpoint_backups/ 2>/dev/null || echo "No checkpoints to backup yet"

# Verify backups
ls -lh ../checkpoint_backups/
```

**Why**: If Cycle 5 dimension regression tests reveal incompatibilities, may need to restart from clean state.

### 2. Create Module Structure

```bash
# Create VFS module
mkdir -p src/townlet/vfs
touch src/townlet/vfs/__init__.py
touch src/townlet/vfs/schema.py
touch src/townlet/vfs/registry.py
touch src/townlet/vfs/observation_builder.py

# Create test directories
mkdir -p tests/test_townlet/unit/vfs
mkdir -p tests/test_townlet/integration/vfs
touch tests/test_townlet/unit/vfs/__init__.py
touch tests/test_townlet/integration/vfs/__init__.py

# Create test files (empty initially)
touch tests/test_townlet/unit/vfs/test_schema.py
touch tests/test_townlet/unit/vfs/test_registry.py
touch tests/test_townlet/unit/vfs/test_observation_builder.py
touch tests/test_townlet/unit/vfs/test_observation_dimension_regression.py
touch tests/test_townlet/unit/environment/test_action_config_extension.py
touch tests/test_townlet/integration/vfs/test_variable_to_observation_flow.py
```

### 3. Install Dependencies (if needed)

```bash
# Ensure Pydantic is available
uv sync
```

---

## TDD Cycle 0: Observation Structure Reverse Engineering (4-6 hours)

**⚠️ CRITICAL**: This cycle must complete BEFORE implementing VFS. It prevents observation dimension incompatibility that would break all existing checkpoints.

**⚠️ PEER REVIEW FIX**: This cycle is now **DOCUMENTATION-ONLY**. Automated validation deferred to Cycle 5 (after VFSObservationSpecBuilder exists). This eliminates circular dependency on Cycle 3.

### 0.1 Document Current Observation Formulas (1 hour)

**File**: `docs/vfs/observation-dimension-formulas.md` (NEW)

```markdown
# Current Observation Dimension Formulas

## Full Observability
obs_dim = substrate.get_observation_dim() + meter_count + (num_affordances + 1) + 4

Components:
- substrate.get_observation_dim(): Grid2D=2, Grid3D=3, GridND=N, Aspatial=0
- meter_count: Always 8 (energy, health, satiation, money, mood, social, fitness, hygiene)
- num_affordances + 1: One-hot encoding (affordance types + "none")
- 4: Temporal features (time_sin, time_cos, interaction_progress, lifetime_progress)

## Partial Observability (POMDP)
obs_dim = window_size^position_dim + position_dim + meter_count + (num_affordances + 1) + 4

Components:
- window_size: (2 * vision_range + 1) - e.g., vision_range=2 → 5×5 window
- position_dim: 2 for Grid2D, 3 for Grid3D
- window_size^position_dim: 25 for Grid2D 5×5 window
- (all other components same as full observability)

## Current Dimensions by Config
- L0_0_minimal: 29 dims (2 + 8 + 15 + 4)
- L0_5_dual_resource: 29 dims
- L1_full_observability: 29 dims
- L2_partial_observability: 54 dims (25 + 2 + 8 + 15 + 4)
- L3_temporal_mechanics: 29 dims
```

### 0.2 Create Reference Variable Definitions (2-3 hours)

**PURPOSE**: These are **TEMPORARY scaffolding** for dimension validation. They can be DELETED after Cycle 5 regression tests pass.

For EACH config, create reference `variables_reference.yaml` that maps current observation structure to VFS variables.

**File**: `configs/L1_full_observability/variables_reference.yaml` (EXAMPLE)

```yaml
# Reference variable definitions for L1 config
# These MUST produce observation_dim = 29

version: "1.0"
description: "Reference variables for L1 full observability (29 dims)"

variables:
  # Position: 2 dims (substrate.get_observation_dim())
  - id: "position"
    scope: "agent"
    type: "vec2i"
    lifetime: "episode"
    readable_by: ["agent"]
    writable_by: ["actions"]
    default: [0, 0]

  # Meters: 8 scalars = 8 dims
  - id: "energy"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent"]
    writable_by: ["actions"]
    default: 1.0
  # ... (repeat for all 8 meters: health, satiation, money, mood, social, fitness, hygiene)

  # Affordance at position: 15 dims (14 affordance types + 1 "none")
  # Represented as one-hot vector
  - id: "affordance_at_position"
    scope: "agent"
    type: "vecNi"
    dims: 15
    lifetime: "tick"
    readable_by: ["agent"]
    writable_by: ["engine"]
    default: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Default: "none"

  # Temporal features: 4 dims
  - id: "time_sin"
    scope: "global"
    type: "scalar"
    lifetime: "tick"
    readable_by: ["agent"]
    writable_by: ["engine"]
    default: 0.0

  - id: "time_cos"
    scope: "global"
    type: "scalar"
    lifetime: "tick"
    readable_by: ["agent"]
    writable_by: ["engine"]
    default: 1.0

  - id: "interaction_progress"
    scope: "agent"
    type: "scalar"
    lifetime: "tick"
    readable_by: ["agent"]
    writable_by: ["engine"]
    default: 0.0

  - id: "lifetime_progress"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent"]
    writable_by: ["engine"]
    default: 0.0

exposed_observations:
  - id: "obs_position"
    source_variable: "position"
    shape: [2]
  - id: "obs_energy"
    source_variable: "energy"
    shape: []
  # ... (all 8 meters)
  - id: "obs_affordance_at_position"
    source_variable: "affordance_at_position"
    shape: [15]
  - id: "obs_time_sin"
    source_variable: "time_sin"
    shape: []
  - id: "obs_time_cos"
    source_variable: "time_cos"
    shape: []
  - id: "obs_interaction_progress"
    source_variable: "interaction_progress"
    shape: []
  - id: "obs_lifetime_progress"
    source_variable: "lifetime_progress"
    shape: []

# Total dims: 2 + 8*1 + 15 + 4*1 = 2 + 8 + 15 + 4 = 29 ✓
```

**Repeat for**: L0_0_minimal, L0_5_dual_resource, L2_partial_observability, L3_temporal_mechanics

### 0.3 Manual Dimension Calculation (1-2 hours)

**⚠️ CHANGED**: No automated validation script in this cycle (circular dependency eliminated).

For EACH config, **manually calculate** expected observation_dim:

**Calculation Template**:
```markdown
# L1_full_observability Expected Dimensions

## Formula
obs_dim = substrate.get_observation_dim() + meter_count + (num_affordances + 1) + 4

## Components
- substrate.get_observation_dim(): 2 (Grid2D with "relative" encoding)
- meter_count: 8 (energy, health, satiation, money, mood, social, fitness, hygiene)
- num_affordances + 1: 15 (14 affordances + "none")
- temporal: 4 (time_sin, time_cos, interaction_progress, lifetime_progress)

## Total
2 + 8 + 15 + 4 = 29 dims ✓

## Variable Count Validation
- position: 2 dims
- energy, health, satiation, money, mood, social, fitness, hygiene: 8 × 1 = 8 dims
- affordance_at_position: 15 dims
- time_sin, time_cos, interaction_progress, lifetime_progress: 4 × 1 = 4 dims
TOTAL: 2 + 8 + 15 + 4 = 29 dims ✓
```

**Document in**: `docs/vfs/observation-dimension-manual-validation.md`

### 0.4 Verify Against Current Environment (30 minutes)

**Manual verification** - no script yet:

```bash
# For each config, check current env.observation_dim
python -c "
from pathlib import Path
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv

configs = [
    ('L0_0_minimal', 29),
    ('L0_5_dual_resource', 29),
    ('L1_full_observability', 29),
    ('L2_partial_observability', 54),
    ('L3_temporal_mechanics', 29),
]

for name, expected in configs:
    env = VectorizedHamletEnv(
        num_agents=1, grid_size=8,
        partial_observability=('partial' in name),
        vision_range=2,
        enable_temporal_mechanics=('temporal' in name),
        move_energy_cost=0.005, wait_energy_cost=0.004,
        interact_energy_cost=0.003, agent_lifespan=1000,
        device=torch.device('cpu'),
        config_pack_path=Path(f'configs/{name}')
    )
    actual = env.observation_dim
    status = '✓' if actual == expected else '❌'
    print(f'{status} {name}: expected={expected}, actual={actual}')
"
```

**Success Criteria**:
- All 5 configs have documented expected dimensions
- Reference variables match expected dimensions
- Manual calculation matches env.observation_dim
- Dimension formulas documented

**NOTE**: Automated validation script will be created in **Cycle 5** (after VFSObservationSpecBuilder exists).

---

## TDD Cycle 1: Variable Type Definitions (4 hours)

### 1.1 RED: Write Schema Validation Tests

**File**: `tests/test_townlet/unit/vfs/test_schema.py`

```python
"""Test VFS schema definitions and validation."""

import pytest
from pydantic import ValidationError
from townlet.vfs.schema import VariableDef, ObservationField, WriteSpec, NormalizationSpec


class TestVariableDef:
    """Test VariableDef schema validation."""

    def test_scalar_variable_valid(self):
        """Scalar variable with all required fields."""
        var = VariableDef(
            id="energy",
            scope="agent",
            type="scalar",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["actions", "engine"],
            default=1.0,
            description="Agent energy level"
        )
        assert var.id == "energy"
        assert var.type == "scalar"
        assert var.dims is None

    def test_vec2i_variable_valid(self):
        """2D integer vector variable."""
        var = VariableDef(
            id="position",
            scope="agent",
            type="vec2i",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["actions", "engine"],
            default=[0, 0]
        )
        assert var.type == "vec2i"
        assert var.dims is None

    def test_vecNi_requires_dims_field(self):
        """vecNi type must have dims field."""
        with pytest.raises(ValidationError, match="dims field"):
            VariableDef(
                id="position_7d",
                scope="agent",
                type="vecNi",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=[0] * 7
                # Missing dims field!
            )

    def test_vecNi_with_dims_valid(self):
        """vecNi with dims field is valid."""
        var = VariableDef(
            id="position_7d",
            scope="agent",
            type="vecNi",
            dims=7,
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["engine"],
            default=[0] * 7
        )
        assert var.type == "vecNi"
        assert var.dims == 7

    def test_scalar_cannot_have_dims(self):
        """Scalar type cannot have dims field."""
        with pytest.raises(ValidationError, match="cannot have dims"):
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                dims=1,  # Invalid!
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=1.0
            )

    def test_invalid_scope_rejected(self):
        """Invalid scope values are rejected."""
        with pytest.raises(ValidationError):
            VariableDef(
                id="test",
                scope="invalid_scope",  # Not in ["global", "agent", "agent_private"]
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=0.0
            )

    def test_invalid_type_rejected(self):
        """Invalid type values are rejected."""
        with pytest.raises(ValidationError):
            VariableDef(
                id="test",
                scope="agent",
                type="invalid_type",  # Not in valid types
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["engine"],
                default=0.0
            )

    def test_global_scope_variable(self):
        """Global scope variable."""
        var = VariableDef(
            id="world_config_hash",
            scope="global",
            type="scalar",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["engine"],
            default=0
        )
        assert var.scope == "global"

    def test_agent_private_scope_variable(self):
        """Agent private scope variable."""
        var = VariableDef(
            id="home_pos",
            scope="agent_private",
            type="vec2i",
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=[0, 0]
        )
        assert var.scope == "agent_private"


class TestObservationField:
    """Test ObservationField schema validation."""

    def test_scalar_observation_field(self):
        """Scalar observation field."""
        field = ObservationField(
            id="obs_energy",
            source_variable="energy",
            exposed_to=["agent"],
            shape=[]
        )
        assert field.shape == []
        assert field.normalization is None

    def test_vec2i_observation_field(self):
        """Vector observation field."""
        field = ObservationField(
            id="obs_position",
            source_variable="position",
            exposed_to=["agent"],
            shape=[2]
        )
        assert field.shape == [2]

    def test_observation_with_normalization(self):
        """Observation field with min/max normalization."""
        norm = NormalizationSpec(
            kind="minmax",
            min=[0.0, 0.0],
            max=[7.0, 7.0]
        )
        field = ObservationField(
            id="obs_position",
            source_variable="position",
            exposed_to=["agent"],
            shape=[2],
            normalization=norm
        )
        assert field.normalization.kind == "minmax"
        assert field.normalization.min == [0.0, 0.0]


class TestNormalizationSpec:
    """Test NormalizationSpec schema validation."""

    def test_minmax_normalization(self):
        """Min/max normalization."""
        norm = NormalizationSpec(
            kind="minmax",
            min=[0.0],
            max=[1.0]
        )
        assert norm.kind == "minmax"
        assert norm.min == [0.0]
        assert norm.max == [1.0]

    def test_standardization_normalization(self):
        """Standardization (z-score) normalization."""
        norm = NormalizationSpec(
            kind="standardization",
            mean=0.5,
            std=0.2
        )
        assert norm.kind == "standardization"
        assert norm.mean == 0.5
        assert norm.std == 0.2


class TestWriteSpec:
    """Test WriteSpec schema validation."""

    def test_simple_write_spec(self):
        """Simple variable write specification."""
        write = WriteSpec(
            variable_id="energy",
            expression="energy - 0.1"
        )
        assert write.variable_id == "energy"
        assert write.expression == "energy - 0.1"

    def test_complex_write_spec(self):
        """Complex write with clamp expression."""
        write = WriteSpec(
            variable_id="health",
            expression="clamp(health - 0.05, 0.0, 1.0)"
        )
        assert "clamp" in write.expression
```

**Run tests** (they should all fail - RED):
```bash
uv run pytest tests/test_townlet/unit/vfs/test_schema.py -v
# Expected: ImportError - module doesn't exist yet
```

### 1.2 GREEN: Implement Schema Definitions

**File**: `src/townlet/vfs/schema.py`

```python
"""VFS schema definitions for typed variable contracts."""

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class NormalizationSpec(BaseModel):
    """Normalization parameters for observations."""

    kind: Literal["minmax", "standardization"]
    min: Optional[List[float]] = None
    max: Optional[List[float]] = None
    mean: Optional[float] = None
    std: Optional[float] = None


class VariableDef(BaseModel):
    """Variable declaration in UAC.

    Defines a stateful variable with typed contract for scope,
    lifetime, access control, and default value.
    """

    id: str = Field(min_length=1)
    scope: Literal["global", "agent", "agent_private"]
    type: Literal["scalar", "vec2i", "vec3i", "vecNi", "vecNf", "bool"]
    dims: Optional[int] = Field(default=None, ge=1)
    lifetime: Literal["tick", "episode"]
    readable_by: List[str] = Field(min_length=1)
    writable_by: List[str] = Field(min_length=1)
    default: Any
    description: str = ""

    @field_validator("dims")
    @classmethod
    def validate_dims_for_type(cls, v: Optional[int], info) -> Optional[int]:
        """vecNi and vecNf require dims field; others cannot have it."""
        var_type = info.data.get("type")

        if var_type in ["vecNi", "vecNf"]:
            if v is None:
                raise ValueError(f"Type {var_type} requires dims field")
        else:
            if v is not None:
                raise ValueError(f"Type {var_type} cannot have dims field")

        return v


class ObservationField(BaseModel):
    """Observation exposure specification for BAC.

    Defines how a variable is exposed in the observation space,
    including shape, normalization, and access control.
    """

    id: str = Field(min_length=1)
    source_variable: str = Field(min_length=1)
    exposed_to: List[str] = Field(min_length=1)
    shape: List[int]
    normalization: Optional[NormalizationSpec] = None


class WriteSpec(BaseModel):
    """Action write declaration.

    Specifies which variable an action writes and the expression
    used to compute the new value.
    """

    variable_id: str = Field(min_length=1)
    expression: str = Field(min_length=1)
```

**File**: `src/townlet/vfs/__init__.py`

```python
"""Variable & Feature System (VFS) - Phase 1.

Provides typed contracts for variables, observations, and action dependencies.
"""

from townlet.vfs.schema import (
    VariableDef,
    ObservationField,
    WriteSpec,
    NormalizationSpec,
)

__all__ = [
    "VariableDef",
    "ObservationField",
    "WriteSpec",
    "NormalizationSpec",
]
```

**Run tests** (they should pass - GREEN):
```bash
uv run pytest tests/test_townlet/unit/vfs/test_schema.py -v
# Expected: All tests pass
```

### 1.3 REFACTOR: Clean Up Schema Code

- Add docstrings with examples
- Extract validation constants if needed
- Ensure consistent field ordering

**Run tests again** to ensure refactoring didn't break anything:
```bash
uv run pytest tests/test_townlet/unit/vfs/test_schema.py -v
```

---

## TDD Cycle 2: Variable Registry (10-12 hours)

**⚠️ COMPLEXITY NOTE**: Registry is more complex than initially estimated due to:
- Scope-specific tensor management (global/agent/agent_private have different shapes)
- Access control enforcement with multiple edge cases
- Type-specific default initialization patterns (6 types × 3 scopes = 18 patterns)

### 2.1 RED: Write Registry Tests

**File**: `tests/test_townlet/unit/vfs/test_registry.py`

```python
"""Test VariableRegistry runtime storage and access control."""

import pytest
import torch
from townlet.vfs.schema import VariableDef
from townlet.vfs.registry import VariableRegistry, AccessError


@pytest.fixture
def sample_variables():
    """Sample variable definitions for testing."""
    return [
        VariableDef(
            id="world_config_hash",
            scope="global",
            type="scalar",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["engine"],
            default=0
        ),
        VariableDef(
            id="energy",
            scope="agent",
            type="scalar",
            lifetime="episode",
            readable_by=["agent", "engine", "acs"],
            writable_by=["actions", "engine"],
            default=1.0
        ),
        VariableDef(
            id="position",
            scope="agent",
            type="vec2i",
            lifetime="episode",
            readable_by=["agent", "engine"],
            writable_by=["actions", "engine"],
            default=[0, 0]
        ),
        VariableDef(
            id="home_pos",
            scope="agent_private",
            type="vec2i",
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=[0, 0]
        ),
    ]


@pytest.fixture
def registry(sample_variables):
    """Create registry with sample variables."""
    return VariableRegistry(
        variable_defs=sample_variables,
        num_agents=4,
        device=torch.device("cpu")
    )


class TestRegistryInitialization:
    """Test registry initialization."""

    def test_registry_creates_global_storage(self, registry):
        """Global variables have single value storage."""
        value = registry.get("global", "world_config_hash", reader="engine")
        assert isinstance(value, torch.Tensor)
        assert value.shape == torch.Size([])  # Scalar

    def test_registry_creates_agent_storage(self, registry):
        """Agent variables have per-agent storage."""
        value = registry.get("agent", "energy", agent_id=0, reader="agent")
        assert isinstance(value, torch.Tensor)
        # Should be able to get all agent values
        all_values = registry.get("agent", "energy", reader="engine")
        assert all_values.shape == torch.Size([4])  # num_agents

    def test_registry_creates_agent_private_storage(self, registry):
        """Agent_private variables have per-agent storage."""
        value = registry.get("agent_private", "home_pos", agent_id=0, reader="agent")
        assert isinstance(value, torch.Tensor)
        assert value.shape == torch.Size([2])  # vec2i

    def test_registry_applies_default_values(self, registry):
        """Registry initializes with default values."""
        energy = registry.get("agent", "energy", agent_id=0, reader="agent")
        assert energy.item() == 1.0


class TestRegistryGet:
    """Test registry get operations."""

    def test_get_global_variable(self, registry):
        """Get global variable."""
        value = registry.get("global", "world_config_hash", reader="engine")
        assert value.shape == torch.Size([])

    def test_get_agent_variable_single_agent(self, registry):
        """Get single agent's value."""
        value = registry.get("agent", "energy", agent_id=0, reader="agent")
        assert isinstance(value, torch.Tensor)

    def test_get_agent_variable_all_agents(self, registry):
        """Get all agents' values (engine perspective)."""
        values = registry.get("agent", "energy", reader="engine")
        assert values.shape == torch.Size([4])

    def test_get_agent_private_variable_own_agent(self, registry):
        """Agent can read own private variable."""
        value = registry.get("agent_private", "home_pos", agent_id=0, reader="agent")
        assert value.shape == torch.Size([2])

    def test_get_agent_private_variable_wrong_agent_fails(self, registry):
        """Agent cannot read another agent's private variable."""
        with pytest.raises(AccessError, match="cannot read agent_private"):
            # Agent 0 trying to read Agent 1's private variable
            registry.get("agent_private", "home_pos", agent_id=1, reader="agent")

    def test_get_respects_readable_by(self, registry):
        """Get respects readable_by access control."""
        # home_pos only readable by "agent", not "acs"
        with pytest.raises(AccessError, match="not allowed to read"):
            registry.get("agent_private", "home_pos", agent_id=0, reader="acs")

    def test_get_invalid_scope_fails(self, registry):
        """Getting invalid scope fails."""
        with pytest.raises(ValueError, match="Invalid scope"):
            registry.get("invalid_scope", "energy", reader="agent")

    def test_get_nonexistent_variable_fails(self, registry):
        """Getting nonexistent variable fails."""
        with pytest.raises(KeyError, match="Variable .* not found"):
            registry.get("agent", "nonexistent", reader="agent")


class TestRegistrySet:
    """Test registry set operations."""

    def test_set_global_variable(self, registry):
        """Set global variable."""
        registry.set("global", "world_config_hash", 12345, writer="engine")
        value = registry.get("global", "world_config_hash", reader="engine")
        assert value.item() == 12345

    def test_set_agent_variable_single_agent(self, registry):
        """Set single agent's value."""
        registry.set("agent", "energy", 0.5, agent_id=0, writer="actions")
        value = registry.get("agent", "energy", agent_id=0, reader="agent")
        assert value.item() == 0.5

    def test_set_agent_variable_all_agents(self, registry):
        """Set all agents' values (engine perspective)."""
        new_values = torch.tensor([0.5, 0.6, 0.7, 0.8])
        registry.set("agent", "energy", new_values, writer="engine")
        values = registry.get("agent", "energy", reader="engine")
        assert torch.allclose(values, new_values)

    def test_set_respects_writable_by(self, registry):
        """Set respects writable_by access control."""
        # world_config_hash only writable by "engine", not "actions"
        with pytest.raises(AccessError, match="not allowed to write"):
            registry.set("global", "world_config_hash", 999, writer="actions")

    def test_set_vec2i_variable(self, registry):
        """Set vector variable."""
        new_pos = torch.tensor([3, 4])
        registry.set("agent", "position", new_pos, agent_id=0, writer="actions")
        value = registry.get("agent", "position", agent_id=0, reader="agent")
        assert torch.equal(value, new_pos)


class TestRegistryObservationSpec:
    """Test observation spec generation."""

    def test_get_observation_spec_returns_fields(self, registry):
        """get_observation_spec returns list of ObservationField."""
        spec = registry.get_observation_spec()
        assert isinstance(spec, list)
        # Should have fields for each variable exposed to observations
        assert len(spec) >= 0

    def test_observation_spec_includes_global_variables(self, registry):
        """Observation spec includes global variables."""
        # This test will evolve as we implement observation exposure
        spec = registry.get_observation_spec()
        # Initially might be empty until we add exposure configuration
        assert isinstance(spec, list)
```

**Run tests** (they should fail - RED):
```bash
uv run pytest tests/test_townlet/unit/vfs/test_registry.py -v
# Expected: ImportError - registry module doesn't exist yet
```

### 2.2 GREEN: Implement Variable Registry

**File**: `src/townlet/vfs/registry.py`

```python
"""Variable registry for runtime storage and access control."""

from typing import Dict, List, Optional, Union
import torch
from townlet.vfs.schema import VariableDef, ObservationField


class AccessError(Exception):
    """Raised when access control violation occurs."""
    pass


class VariableRegistry:
    """Runtime storage for variables with access control.

    Manages variable storage with scope-aware tensor shapes:
    - global: Single value [] or [dims] (shared by all agents)
    - agent: Per-agent values [num_agents] or [num_agents, dims] (publicly readable)
    - agent_private: Per-agent values [num_agents] or [num_agents, dims] (owner-only access)

    Scope Semantics (CRITICAL):
    ---------------------
    get(scope="global", var_id, agent_id=None, reader):
        - Returns scalar tensor (shape=[])
        - agent_id must be None (global has no per-agent values)

    get(scope="agent", var_id, agent_id=None, reader):
        - If agent_id=X: Returns specific agent's value
        - If agent_id=None: Returns all agents' values [num_agents, ...]
        - All readers can access (public variable)

    get(scope="agent_private", var_id, agent_id=X, reader="agent"):
        - agent_id is REQUIRED
        - Only owning agent can read (enforced by access control)
        - Returns agent X's private value

    Access Control:
    --------------
    - readable_by: List of roles allowed to read (["agent", "engine", "acs"])
    - writable_by: List of roles allowed to write (["actions", "engine"])
    - Agent_private enforces owner-only reads (even if "agent" in readable_by)
    """

    def __init__(
        self,
        variable_defs: List[VariableDef],
        num_agents: int,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize registry with variable definitions.

        Args:
            variable_defs: List of variable declarations
            num_agents: Number of agents in environment
            device: PyTorch device for tensor storage
        """
        self.variable_defs = {v.id: v for v in variable_defs}
        self.num_agents = num_agents
        self.device = device

        # Storage: {scope: {var_id: tensor}}
        self.storage: Dict[str, Dict[str, torch.Tensor]] = {
            "global": {},
            "agent": {},
            "agent_private": {},
        }

        # Initialize storage with default values
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage with default values from definitions."""
        for var_id, var_def in self.variable_defs.items():
            self.storage[var_def.scope][var_id] = self._create_default_tensor(var_def)

    def _create_default_tensor(self, var_def: VariableDef) -> torch.Tensor:
        """Create tensor with default value and appropriate shape."""
        # Determine shape based on scope and type
        if var_def.scope == "global":
            # Global: single value
            if var_def.type == "scalar":
                shape = []
            elif var_def.type == "vec2i":
                shape = [2]
            elif var_def.type == "vec3i":
                shape = [3]
            elif var_def.type == "vecNi" or var_def.type == "vecNf":
                shape = [var_def.dims]
            elif var_def.type == "bool":
                shape = []
            else:
                raise ValueError(f"Unknown type: {var_def.type}")
        else:
            # agent or agent_private: per-agent values
            if var_def.type == "scalar":
                shape = [self.num_agents]
            elif var_def.type == "vec2i":
                shape = [self.num_agents, 2]
            elif var_def.type == "vec3i":
                shape = [self.num_agents, 3]
            elif var_def.type == "vecNi" or var_def.type == "vecNf":
                shape = [self.num_agents, var_def.dims]
            elif var_def.type == "bool":
                shape = [self.num_agents]
            else:
                raise ValueError(f"Unknown type: {var_def.type}")

        # Create tensor from default value
        if isinstance(var_def.default, list):
            return torch.tensor(var_def.default, device=self.device, dtype=torch.float32)
        else:
            tensor = torch.full(shape, var_def.default, device=self.device, dtype=torch.float32)
            return tensor

    def get(
        self,
        scope: str,
        var_id: str,
        agent_id: Optional[int] = None,
        reader: str = "agent",
    ) -> torch.Tensor:
        """Get variable value with access control.

        Args:
            scope: Variable scope ("global", "agent", "agent_private")
            var_id: Variable identifier
            agent_id: Agent ID (required for agent/agent_private scopes)
            reader: Who is reading ("agent", "engine", "acs")

        Returns:
            Variable value as tensor

        Raises:
            AccessError: If reader not allowed to read this variable
            KeyError: If variable not found
            ValueError: If invalid scope
        """
        if scope not in self.storage:
            raise ValueError(f"Invalid scope: {scope}")

        if var_id not in self.variable_defs:
            raise KeyError(f"Variable {var_id} not found in scope {scope}")

        var_def = self.variable_defs[var_id]

        # Check read access
        if reader not in var_def.readable_by:
            raise AccessError(f"Reader '{reader}' not allowed to read variable '{var_id}'")

        # Check agent_private access
        if scope == "agent_private" and reader == "agent":
            # Agent can only read their own private variables
            if agent_id is None:
                raise AccessError("agent_id required for agent_private scope")
            # For now, we'll just check agent_id is valid
            # In a real system, we'd verify the reader is the owning agent

        # Get storage
        storage = self.storage[scope][var_id]

        # Return appropriate slice based on scope
        if scope == "global":
            return storage
        elif scope in ["agent", "agent_private"]:
            if agent_id is not None:
                # Return specific agent's value
                return storage[agent_id]
            else:
                # Return all agents' values (typically for engine)
                return storage

        return storage

    def set(
        self,
        scope: str,
        var_id: str,
        value: Union[torch.Tensor, float, int, List],
        agent_id: Optional[int] = None,
        writer: str = "actions",
    ):
        """Set variable value with access control.

        Args:
            scope: Variable scope
            var_id: Variable identifier
            value: New value (tensor, scalar, or list)
            agent_id: Agent ID (for agent/agent_private scopes)
            writer: Who is writing ("actions", "engine")

        Raises:
            AccessError: If writer not allowed to write this variable
            KeyError: If variable not found
        """
        if var_id not in self.variable_defs:
            raise KeyError(f"Variable {var_id} not found")

        var_def = self.variable_defs[var_id]

        # Check write access
        if writer not in var_def.writable_by:
            raise AccessError(f"Writer '{writer}' not allowed to write variable '{var_id}'")

        # Convert value to tensor if needed
        if not isinstance(value, torch.Tensor):
            if isinstance(value, list):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)
            else:
                value = torch.tensor(value, device=self.device, dtype=torch.float32)

        # Set value
        storage = self.storage[scope]
        if agent_id is not None:
            # Set specific agent's value
            storage[var_id][agent_id] = value
        else:
            # Set all values
            storage[var_id] = value

    def get_observation_spec(self) -> List[ObservationField]:
        """Return observation specifications for BAC compiler.

        Returns:
            List of observation fields that should be exposed
        """
        # For Phase 1, return empty list
        # This will be populated when we add exposure configuration
        return []
```

**Update**: `src/townlet/vfs/__init__.py`

```python
"""Variable & Feature System (VFS) - Phase 1."""

from townlet.vfs.schema import (
    VariableDef,
    ObservationField,
    WriteSpec,
    NormalizationSpec,
)
from townlet.vfs.registry import VariableRegistry, AccessError

__all__ = [
    "VariableDef",
    "ObservationField",
    "WriteSpec",
    "NormalizationSpec",
    "VariableRegistry",
    "AccessError",
]
```

**Run tests** (they should pass - GREEN):
```bash
uv run pytest tests/test_townlet/unit/vfs/test_registry.py -v
```

### 2.3 REFACTOR: Clean Up Registry Code

- Extract tensor shape calculation into helper methods
- Add comprehensive docstrings
- Optimize access control checks
- Add type hints everywhere

**Run tests** to ensure refactoring didn't break anything:
```bash
uv run pytest tests/test_townlet/unit/vfs/test_registry.py -v
```

---

## TDD Cycle 3: VFS Observation Spec Builder (6-8 hours)

**⚠️ NAMING NOTE**: This class is named `VFSObservationSpecBuilder` to avoid collision with the existing `environment.observation_builder.ObservationBuilder` class. They serve different purposes:
- **`environment.ObservationBuilder`**: Runtime observation construction (tensors)
- **`vfs.VFSObservationSpecBuilder`**: Compile-time spec generation (schemas for BAC)

### 3.1 RED: Write Observation Spec Builder Tests

**File**: `tests/test_townlet/unit/vfs/test_observation_builder.py`

```python
"""Test VFSObservationSpecBuilder for generating observation specs."""

import pytest
from townlet.vfs.schema import VariableDef, NormalizationSpec
from townlet.vfs.observation_builder import VFSObservationSpecBuilder


@pytest.fixture
def sample_variables():
    """Sample variables for testing."""
    return [
        VariableDef(
            id="energy",
            scope="agent",
            type="scalar",
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=1.0
        ),
        VariableDef(
            id="position",
            scope="agent",
            type="vec2i",
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=[0, 0]
        ),
        VariableDef(
            id="position_7d",
            scope="agent",
            type="vecNi",
            dims=7,
            lifetime="episode",
            readable_by=["agent"],
            writable_by=["actions"],
            default=[0] * 7
        ),
    ]


class TestVFSObservationSpecBuilder:
    """Test observation spec generation."""

    def test_build_spec_for_scalar_variable(self, sample_variables):
        """Build observation spec for scalar variable."""
        builder = VFSObservationSpecBuilder()

        # Expose energy as observation
        exposures = {
            "energy": {
                "normalization": {"kind": "minmax", "min": 0.0, "max": 1.0}
            }
        }

        spec = builder.build_observation_spec(sample_variables, exposures)

        # Should have one field for energy
        assert len(spec) == 1
        assert spec[0].source_variable == "energy"
        assert spec[0].shape == []  # Scalar

    def test_build_spec_for_vec2i_variable(self, sample_variables):
        """Build observation spec for vec2i variable."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "position": {
                "normalization": {"kind": "minmax", "min": [0, 0], "max": [7, 7]}
            }
        }

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 1
        assert spec[0].source_variable == "position"
        assert spec[0].shape == [2]  # vec2i

    def test_build_spec_for_vecNi_variable(self, sample_variables):
        """Build observation spec for vecNi variable."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "position_7d": {
                "normalization": None
            }
        }

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 1
        assert spec[0].source_variable == "position_7d"
        assert spec[0].shape == [7]  # vecNi with dims=7

    def test_build_spec_multiple_variables(self, sample_variables):
        """Build observation spec for multiple variables."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {"normalization": None},
            "position": {"normalization": None},
        }

        spec = builder.build_observation_spec(sample_variables, exposures)

        assert len(spec) == 2
        source_vars = {field.source_variable for field in spec}
        assert source_vars == {"energy", "position"}

    def test_total_observation_dim_calculation(self, sample_variables):
        """Calculate total observation dimension."""
        builder = VFSObservationSpecBuilder()

        exposures = {
            "energy": {"normalization": None},      # 0 dims (scalar)
            "position": {"normalization": None},     # 2 dims
            "position_7d": {"normalization": None},  # 7 dims
        }

        spec = builder.build_observation_spec(sample_variables, exposures)

        total_dims = sum(
            len(field.shape) if field.shape else 1
            for field in spec
        )
        # energy (1) + position (2) + position_7d (7) = 10
        assert total_dims == 10
```

**Run tests** (they should fail - RED):
```bash
uv run pytest tests/test_townlet/unit/vfs/test_observation_builder.py -v
```

### 3.2 GREEN: Implement VFS Observation Spec Builder

**File**: `src/townlet/vfs/observation_builder.py`

```python
"""VFS observation spec builder for generating observation specs from variables.

NOTE: This is NOT the same as environment.observation_builder.ObservationBuilder!
- environment.ObservationBuilder: Runtime observation construction (tensors)
- vfs.VFSObservationSpecBuilder: Compile-time spec generation (schemas for BAC)
"""

from typing import Dict, List, Any
from townlet.vfs.schema import VariableDef, ObservationField, NormalizationSpec


class VFSObservationSpecBuilder:
    """Constructs observation specifications from variable definitions.

    Generates ObservationField specs that BAC compiler can use to
    build network input heads dynamically.

    This class generates SPECIFICATIONS (schemas), not runtime observations.
    For runtime observation construction, see environment.observation_builder.ObservationBuilder.
    """

    def build_observation_spec(
        self,
        variables: List[VariableDef],
        exposures: Dict[str, Dict[str, Any]],
    ) -> List[ObservationField]:
        """Build observation specification from variables and exposure config.

        Args:
            variables: List of variable definitions
            exposures: Dict mapping variable_id -> exposure config
                      e.g., {"energy": {"normalization": {...}}}

        Returns:
            List of ObservationField specs
        """
        var_map = {v.id: v for v in variables}
        obs_fields = []

        for var_id, exposure_config in exposures.items():
            if var_id not in var_map:
                raise ValueError(f"Variable {var_id} not found in definitions")

            var_def = var_map[var_id]

            # Infer shape from variable type
            shape = self._infer_shape(var_def)

            # Build normalization spec if provided
            norm_spec = None
            if exposure_config.get("normalization"):
                norm_config = exposure_config["normalization"]
                norm_spec = NormalizationSpec(**norm_config)

            # Create observation field
            field = ObservationField(
                id=f"obs_{var_id}",
                source_variable=var_id,
                exposed_to=["agent"],  # Default to agent for Phase 1
                shape=shape,
                normalization=norm_spec,
            )

            obs_fields.append(field)

        return obs_fields

    def _infer_shape(self, var_def: VariableDef) -> List[int]:
        """Infer observation shape from variable type.

        Args:
            var_def: Variable definition

        Returns:
            Shape as list (empty list for scalar)
        """
        if var_def.type == "scalar":
            return []
        elif var_def.type == "vec2i":
            return [2]
        elif var_def.type == "vec3i":
            return [3]
        elif var_def.type in ["vecNi", "vecNf"]:
            return [var_def.dims]
        elif var_def.type == "bool":
            return []
        else:
            raise ValueError(f"Unknown variable type: {var_def.type}")
```

**Update**: `src/townlet/vfs/__init__.py`

```python
from townlet.vfs.observation_builder import VFSObservationSpecBuilder

__all__ = [
    # ... existing exports ...
    "VFSObservationSpecBuilder",
]
```

**Run tests** (they should pass - GREEN):
```bash
uv run pytest tests/test_townlet/unit/vfs/test_observation_builder.py -v
```

### 3.3 REFACTOR: Clean Up VFS Observation Spec Builder

- Add error handling for invalid exposure configs
- Add helper methods for common normalization patterns
- Add docstring examples

---

## TDD Cycle 4: ActionConfig Extension (1-2 hours)

### 4.1 RED: Write ActionConfig Extension Tests

**File**: `tests/test_townlet/unit/environment/test_action_config_extension.py`

```python
"""Test ActionConfig extension with reads/writes fields."""

import pytest
from townlet.environment.action_config import ActionConfig
from townlet.vfs.schema import WriteSpec


class TestActionConfigExtension:
    """Test ActionConfig with VFS reads/writes fields."""

    def test_action_config_with_reads(self):
        """ActionConfig with reads field."""
        action = ActionConfig(
            id=0,
            name="TELEPORT_HOME",
            type="movement",
            costs={"energy": 0.05},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description="Teleport to home position",
            icon="🏠",
            source="custom",
            source_affordance=None,
            reads=["home_pos", "position"],  # NEW FIELD
        )

        assert action.reads == ["home_pos", "position"]

    def test_action_config_with_writes(self):
        """ActionConfig with writes field."""
        write_spec = WriteSpec(
            variable_id="position",
            expression="home_pos"
        )

        action = ActionConfig(
            id=0,
            name="TELEPORT_HOME",
            type="movement",
            costs={"energy": 0.05},
            effects={},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            writes=[write_spec],  # NEW FIELD
        )

        assert len(action.writes) == 1
        assert action.writes[0].variable_id == "position"

    def test_action_config_backward_compatible(self):
        """ActionConfig without reads/writes still works (backward compatible)."""
        action = ActionConfig(
            id=0,
            name="UP",
            type="movement",
            costs={"energy": 0.005},
            effects={},
            delta=[0, -1],
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="substrate",
            source_affordance=None,
            # reads and writes omitted - should use defaults
        )

        assert action.reads == []  # Default empty list
        assert action.writes == []  # Default empty list

    def test_action_config_serialization_with_reads_writes(self):
        """ActionConfig serializes/deserializes with reads/writes."""
        write_spec = WriteSpec(variable_id="energy", expression="energy - 0.1")

        action = ActionConfig(
            id=0,
            name="REST",
            type="passive",
            costs={},
            effects={"energy": 0.02},
            delta=None,
            teleport_to=None,
            enabled=True,
            description=None,
            icon=None,
            source="custom",
            source_affordance=None,
            reads=["energy"],
            writes=[write_spec],
        )

        # Serialize
        data = action.model_dump()
        assert "reads" in data
        assert "writes" in data

        # Deserialize
        action2 = ActionConfig.model_validate(data)
        assert action2.reads == ["energy"]
        assert len(action2.writes) == 1
```

**Run tests** (they should fail - RED):
```bash
uv run pytest tests/test_townlet/unit/environment/test_action_config_extension.py -v
```

### 4.2 GREEN: Extend ActionConfig

**File**: `src/townlet/environment/action_config.py` (MODIFY)

Add to existing ActionConfig:

```python
from typing import List
from pydantic import Field

# Import WriteSpec from VFS
from townlet.vfs.schema import WriteSpec


class ActionConfig(BaseModel):
    # ... existing fields ...

    # VFS Integration (Phase 1 - TASK-002C)
    reads: List[str] = Field(
        default_factory=list,
        description="Variables this action reads (for dependency tracking)"
    )
    writes: List[WriteSpec] = Field(
        default_factory=list,
        description="Variables this action writes (with expressions)"
    )

    # ... rest of class ...
```

**Run tests** (they should pass - GREEN):
```bash
uv run pytest tests/test_townlet/unit/environment/test_action_config_extension.py -v
```

### 4.3 REFACTOR: Clean Up ActionConfig

- Ensure import order is clean
- Add docstring examples for reads/writes
- Verify backward compatibility

---

## TDD Cycle 5: Observation Dimension Regression (3-4 hours)

**THIS IS CRITICAL** - These tests prevent checkpoint incompatibility.

**⚠️ NOTE**: This cycle now includes automated validation script (moved from Cycle 0 to eliminate circular dependency).

### 5.1 RED: Write Dimension Regression Tests

**File**: `tests/test_townlet/unit/vfs/test_observation_dimension_regression.py`

```python
"""CRITICAL: Regression tests for observation dimension compatibility.

These tests ensure VFS-generated observation_dim matches the current
hardcoded calculation. If these fail, checkpoints will be incompatible!
"""

import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.vfs.schema import VariableDef
from townlet.vfs.observation_builder import VFSObservationSpecBuilder


def compute_current_observation_dim(config_path: Path) -> int:
    """Compute observation_dim using current hardcoded calculation.

    This replicates the logic in VectorizedHamletEnv.__init__().
    """
    # Load config and create environment
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.004,
        interact_energy_cost=0.003,
        agent_lifespan=1000,
        device=torch.device("cpu"),
        config_pack_path=config_path,
    )

    return env.observation_dim


def compute_vfs_observation_dim(variables: list, exposures: dict) -> int:
    """Compute observation_dim using VFS observation spec builder.

    This is how BAC compiler will calculate dimensions.
    """
    builder = VFSObservationSpecBuilder()
    spec = builder.build_observation_spec(variables, exposures)

    # Calculate total dimensions
    total_dims = 0
    for field in spec:
        if field.shape:
            total_dims += sum(field.shape)  # Vector dimensions
        else:
            total_dims += 1  # Scalar

    return total_dims


class TestObservationDimensionRegression:
    """Test VFS observation dimensions match current implementation.

    CRITICAL: These tests MUST pass or checkpoints will be incompatible!
    """

    def test_l0_minimal_full_observability(self):
        """L0_minimal config observation dimensions."""
        config_path = Path("configs/L0_0_minimal")

        # Current hardcoded calculation
        current_dim = compute_current_observation_dim(config_path)

        # VFS-based calculation
        # Define variables that match L0 config
        variables = [
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent"],
                writable_by=["actions"],
                default=1.0
            ),
            # Add other meters...
        ]

        exposures = {
            "energy": {"normalization": None},
            # Add other exposures...
        }

        vfs_dim = compute_vfs_observation_dim(variables, exposures)

        # MUST be identical
        assert vfs_dim == current_dim, (
            f"L0_minimal: VFS dim {vfs_dim} != current dim {current_dim}. "
            f"CHECKPOINT INCOMPATIBILITY!"
        )

    @pytest.mark.skip(reason="Implement after L0_minimal passes")
    def test_l1_full_observability(self):
        """L1_full config observation dimensions."""
        # Similar structure to above
        pass

    @pytest.mark.skip(reason="Implement after L1 passes")
    def test_l2_pomdp(self):
        """L2_pomdp config observation dimensions."""
        # Test partial observability dimension calculation
        pass
```

**Run tests** (some will fail initially - RED):
```bash
uv run pytest tests/test_townlet/unit/vfs/test_observation_dimension_regression.py -v
```

### 5.2 GREEN: Fix Dimension Calculations

**Iterate** until all dimension regression tests pass:

1. Identify dimension mismatches
2. Adjust VFS calculation or variable definitions
3. Re-run tests
4. Repeat until all configs pass

**Critical**: Do NOT change the current environment calculation - adjust VFS to match it.

**Additionally**: Create automated validation script (moved from Cycle 0):

**File**: `scripts/validate_vfs_dimensions.py` (NEW)

```python
"""Validate that VFS-generated dimensions match current environment.

CRITICAL: This script validates VFS implementation against reference variables.
"""

from pathlib import Path
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.vfs.observation_builder import VFSObservationSpecBuilder
import yaml

def validate_config(config_name: str, expected_dim: int):
    """Validate observation dimension for a config."""
    config_path = Path(f"configs/{config_name}")

    # Get current dimension from environment
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability="partial" in config_name.lower(),
        vision_range=2,
        enable_temporal_mechanics="temporal" in config_name.lower(),
        move_energy_cost=0.005,
        wait_energy_cost=0.004,
        interact_energy_cost=0.003,
        agent_lifespan=1000,
        device=torch.device("cpu"),
        config_pack_path=config_path,
    )

    actual_dim = env.observation_dim

    # Load reference variables
    ref_path = config_path / "variables_reference.yaml"
    if not ref_path.exists():
        print(f"❌ {config_name}: Missing variables_reference.yaml")
        return False

    # Load and compute VFS dimension
    with open(ref_path) as f:
        ref_data = yaml.safe_load(f)

    # Build observation spec using VFSObservationSpecBuilder
    from townlet.vfs.schema import VariableDef
    variables = [VariableDef(**v) for v in ref_data["variables"]]

    builder = VFSObservationSpecBuilder()
    exposures = {
        field["source_variable"]: {
            "normalization": field.get("normalization")
        }
        for field in ref_data["exposed_observations"]
    }

    obs_spec = builder.build_observation_spec(variables, exposures)

    # Calculate VFS dimension
    vfs_dim = sum(
        len(field.shape) if field.shape else 1
        for field in obs_spec
    )

    # Validate
    if actual_dim == expected_dim == vfs_dim:
        print(f"✓ {config_name}: {actual_dim} dims (VFS={vfs_dim}, env={actual_dim}, expected={expected_dim})")
        return True
    else:
        print(f"❌ {config_name}: MISMATCH - VFS={vfs_dim}, env={actual_dim}, expected={expected_dim}")
        return False

if __name__ == "__main__":
    configs = [
        ("L0_0_minimal", 29),
        ("L0_5_dual_resource", 29),
        ("L1_full_observability", 29),
        ("L2_partial_observability", 54),
        ("L3_temporal_mechanics", 29),
    ]

    all_passed = all(validate_config(name, dim) for name, dim in configs)

    if all_passed:
        print("\n✓ All configs validated - VFS implementation correct!")
        exit(0)
    else:
        print("\n❌ Validation failed - Fix VFS dimension calculations")
        exit(1)
```

**Run validation**:
```bash
python scripts/validate_vfs_dimensions.py
```

### 5.3 REFACTOR: Extract Common Helpers

- Create helper functions for common dimension calculations
- Add documentation about dimension formula
- Add examples for each config type

---

## TDD Cycle 6: Integration Tests (3-4 hours)

**⚠️ COMPLEXITY NOTE**: Integration tests are more complex due to dual observation system (legacy + VFS) coexistence testing.

### 6.1 RED: Write Integration Tests

**File**: `tests/test_townlet/integration/vfs/test_variable_to_observation_flow.py`

```python
"""Integration test: variable → registry → observation → BAC spec."""

import pytest
import torch
from townlet.vfs.schema import VariableDef
from townlet.vfs.registry import VariableRegistry
from townlet.vfs.observation_builder import VFSObservationSpecBuilder


class TestVariableToObservationFlow:
    """Test complete flow from variable definition to observation."""

    def test_end_to_end_flow(self):
        """Complete flow: define → register → expose → spec."""
        # Step 1: Define variables
        variables = [
            VariableDef(
                id="world_config_hash",
                scope="global",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["engine"],
                default=0
            ),
            VariableDef(
                id="energy",
                scope="agent",
                type="scalar",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["actions", "engine"],
                default=1.0
            ),
            VariableDef(
                id="position",
                scope="agent",
                type="vec2i",
                lifetime="episode",
                readable_by=["agent", "engine"],
                writable_by=["actions", "engine"],
                default=[0, 0]
            ),
        ]

        # Step 2: Create registry
        registry = VariableRegistry(
            variable_defs=variables,
            num_agents=4,
            device=torch.device("cpu")
        )

        # Step 3: Engine sets world config hash
        registry.set("global", "world_config_hash", 12345, writer="engine")

        # Step 4: Agent modifies energy
        registry.set("agent", "energy", 0.5, agent_id=0, writer="actions")

        # Step 5: Build observation spec
        builder = VFSObservationSpecBuilder()
        exposures = {
            "world_config_hash": {"normalization": None},
            "energy": {"normalization": {"kind": "minmax", "min": 0.0, "max": 1.0}},
            "position": {"normalization": {"kind": "minmax", "min": [0, 0], "max": [7, 7]}},
        }

        obs_spec = builder.build_observation_spec(variables, exposures)

        # Verify spec
        assert len(obs_spec) == 3

        # Verify BAC compiler can use this spec
        total_dims = sum(
            len(field.shape) if field.shape else 1
            for field in obs_spec
        )
        # world_config_hash (1) + energy (1) + position (2) = 4
        assert total_dims == 4

        # Verify values are accessible
        world_hash = registry.get("global", "world_config_hash", reader="engine")
        assert world_hash.item() == 12345

        agent_energy = registry.get("agent", "energy", agent_id=0, reader="agent")
        assert agent_energy.item() == 0.5
```

**Run tests** (they should pass if previous cycles worked - GREEN):
```bash
uv run pytest tests/test_townlet/integration/vfs/ -v
```

---

## TDD Cycle 7: Config Templates (1 hour)

**NOTE**: Reference variables created in Cycle 0 are **temporary**. This cycle creates the **permanent template** for operators.

### 7.1 Create Example Config

**File**: `configs/templates/variables.yaml`

```yaml
version: "1.0"
description: "Variable definitions for VFS - example template"

# Variable definitions
variables:
  # Global variables (single value shared by all agents)
  - id: "world_config_hash"
    scope: "global"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["engine"]
    default: 0
    description: "Hash of current world configuration (for BLOCKER 2 solution)"

  # Agent variables (per-agent, publicly readable)
  - id: "energy"
    scope: "agent"
    type: "scalar"
    lifetime: "episode"
    readable_by: ["agent", "engine", "acs"]
    writable_by: ["actions", "engine"]
    default: 1.0
    description: "Agent energy level (0.0-1.0)"

  - id: "position"
    scope: "agent"
    type: "vec2i"
    lifetime: "episode"
    readable_by: ["agent", "engine"]
    writable_by: ["actions", "engine"]
    default: [0, 0]
    description: "Agent position on Grid2D substrate"

  # Agent private variables (per-agent, owner-only access)
  - id: "home_pos"
    scope: "agent_private"
    type: "vec2i"
    lifetime: "episode"
    readable_by: ["agent"]
    writable_by: ["actions", "engine"]
    default: [0, 0]
    description: "Agent's remembered home position (for teleport action)"

  # N-dimensional variable (for GridND substrates)
  - id: "position_7d"
    scope: "agent"
    type: "vecNi"
    dims: 7
    lifetime: "episode"
    readable_by: ["agent", "engine"]
    writable_by: ["actions", "engine"]
    default: [0, 0, 0, 0, 0, 0, 0]
    description: "Agent position on 7D GridND substrate"

# Observation exposures (what agents observe)
exposed_observations:
  - id: "obs_world_config_hash"
    source_variable: "world_config_hash"
    exposed_to: ["agent"]
    shape: []
    normalization: null  # No normalization for hash

  - id: "obs_energy"
    source_variable: "energy"
    exposed_to: ["agent"]
    shape: []
    normalization:
      kind: "minmax"
      min: 0.0
      max: 1.0

  - id: "obs_position"
    source_variable: "position"
    exposed_to: ["agent"]
    shape: [2]
    normalization:
      kind: "minmax"
      min: [0, 0]
      max: [7, 7]  # Assuming 8×8 grid

  - id: "obs_home_pos"
    source_variable: "home_pos"
    exposed_to: ["agent"]  # Each agent sees only their own home_pos
    shape: [2]
    normalization:
      kind: "minmax"
      min: [0, 0]
      max: [7, 7]
```

### 7.2 Validate Template Loads

Create a quick test that the template parses correctly:

```python
def test_variables_yaml_template_loads():
    """Verify template YAML is valid."""
    import yaml
    from pathlib import Path

    template_path = Path("configs/templates/variables.yaml")
    with open(template_path) as f:
        data = yaml.safe_load(f)

    assert data["version"] == "1.0"
    assert "variables" in data
    assert len(data["variables"]) > 0
```

### 7.3 Clean Up Temporary Files

**⚠️ IMPORTANT**: Delete temporary reference variables created in Cycle 0:

```bash
# Remove temporary scaffolding (validation is now in automated script)
rm configs/L0_0_minimal/variables_reference.yaml
rm configs/L0_5_dual_resource/variables_reference.yaml
rm configs/L1_full_observability/variables_reference.yaml
rm configs/L2_partial_observability/variables_reference.yaml
rm configs/L3_temporal_mechanics/variables_reference.yaml

# Keep only the template
ls configs/templates/variables.yaml  # This stays
```

**Why**: Reference variables were temporary scaffolding for dimension validation. Now that `scripts/validate_vfs_dimensions.py` uses the template directly, they're no longer needed.

---

## TDD Cycle 8: Integration Specification (2 hours)

**⚠️ PEER REVIEW FIX**: This was previously orphaned (listed in line 2200+ but not in cycles). Now properly integrated.

### 8.1 Document Dual Observation System

**File**: `docs/vfs/integration-with-observation-builder.md` (NEW)

```markdown
# VFS Integration with ObservationBuilder

## Two ObservationBuilder Classes Coexist

### 1. Runtime Observation Construction (Existing)
**Class**: `environment.observation_builder.ObservationBuilder`
- **Purpose**: Builds observation TENSORS at runtime
- **Input**: positions, meters, affordances (runtime state)
- **Output**: `torch.Tensor` [num_agents, observation_dim]
- **Used by**: VectorizedHamletEnv.reset(), VectorizedHamletEnv.step()
- **Status**: ACTIVE - continues working in Phase 1

### 2. Compile-Time Spec Generation (New)
**Class**: `vfs.observation_builder.VFSObservationSpecBuilder`
- **Purpose**: Generates observation SCHEMAS for compilers
- **Input**: List[VariableDef], exposures dict (static config)
- **Output**: List[ObservationField] (Pydantic schemas)
- **Used by**: BAC compiler (TASK-005), UAC compiler (TASK-004A)
- **Status**: NEW - available in Phase 1

## Phase 1 Integration Pattern

```python
# Existing system (continues to work)
env = VectorizedHamletEnv(...)
obs = env.reset()  # Uses environment.ObservationBuilder

# New system (for compilers)
builder = VFSObservationSpecBuilder()
obs_spec = builder.build_observation_spec(variables, exposures)
# Returns schema, not tensors
```

## Phase 2 Migration Path (Future)

```python
# Environment optionally uses VFS variables
if (config_pack_path / "variables.yaml").exists():
    # Load VFS variable definitions
    variables = load_variables_yaml(config_pack_path / "variables.yaml")
    registry = VariableRegistry(variables, num_agents, device)

    # Build observation spec from VFS
    builder = VFSObservationSpecBuilder()
    obs_spec = builder.build_observation_spec(variables, exposures)

    # Use spec to construct observations (future work)
    obs = construct_from_spec(obs_spec, registry)
else:
    # Legacy path (current hardcoded system)
    obs = legacy_observation_builder.build_observations(...)
```

## Key Contracts

### Contract 1: Dimension Compatibility
**VFS-generated observation_dim MUST equal env.observation_dim**

Validation:
```python
vfs_dim = sum(field.shape_size for field in obs_spec.fields)
assert vfs_dim == env.observation_dim  # CRITICAL
```

### Contract 2: Naming Separation
**No name collision between the two classes**

- Runtime: `environment.observation_builder.ObservationBuilder`
- Compile-time: `vfs.observation_builder.VFSObservationSpecBuilder`

### Contract 3: Coexistence
**Both systems can run in same process**

Tests validate:
- Legacy ObservationBuilder still works (no regressions)
- VFSObservationSpecBuilder produces compatible specs
- Both can be imported and used together
```

### 8.2 Write Integration Test

**File**: `tests/test_townlet/integration/vfs/test_observation_system_coexistence.py` (NEW)

```python
"""Test that VFS and legacy observation systems coexist correctly."""

import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.vfs.observation_builder import VFSObservationSpecBuilder
from townlet.vfs.schema import VariableDef


class TestObservationSystemCoexistence:
    """Validate VFS and legacy ObservationBuilder coexist without conflicts."""

    def test_both_systems_importable(self):
        """Both ObservationBuilder classes can be imported together."""
        from townlet.environment.observation_builder import ObservationBuilder
        from townlet.vfs.observation_builder import VFSObservationSpecBuilder

        # No import errors or name collisions
        assert ObservationBuilder is not None
        assert VFSObservationSpecBuilder is not None

    def test_vfs_dimensions_match_legacy_l1_config(self):
        """VFS-generated specs produce same dimensions as legacy system (L1)."""
        # Create env with legacy observation builder
        config_path = Path("configs/L1_full_observability")
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
            device=torch.device("cpu"),
            config_pack_path=config_path,
        )

        legacy_dim = env.observation_dim

        # Create VFS observation spec
        # Use template variables (reference variables deleted in Cycle 7)
        variables = [
            VariableDef(id="position", scope="agent", type="vec2i",
                       lifetime="episode", readable_by=["agent"],
                       writable_by=["actions"], default=[0, 0]),
            # ... (include all variables for L1)
        ]

        exposures = {
            "position": {"normalization": {"kind": "minmax", "min": [0, 0], "max": [7, 7]}},
            # ... (all exposures)
        }

        builder = VFSObservationSpecBuilder()
        obs_spec = builder.build_observation_spec(variables, exposures)

        vfs_dim = sum(len(field.shape) if field.shape else 1 for field in obs_spec)

        # Dimensions MUST match
        assert vfs_dim == legacy_dim == 29, (
            f"Dimension mismatch: VFS={vfs_dim}, legacy={legacy_dim}, expected=29"
        )

    def test_legacy_observation_builder_still_works(self):
        """Legacy ObservationBuilder continues working (no regressions)."""
        config_path = Path("configs/L1_full_observability")
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
            device=torch.device("cpu"),
            config_pack_path=config_path,
        )

        # Reset should work (uses legacy ObservationBuilder internally)
        obs = env.reset()

        assert obs.shape == (4, 29)  # [num_agents, observation_dim]
        assert obs.dtype == torch.float32
```

**Run test**:
```bash
uv run pytest tests/test_townlet/integration/vfs/test_observation_system_coexistence.py -v
```

### 8.3 Update VFS Module README

Add integration section to `src/townlet/vfs/README.md`:

```markdown
## Integration with Existing Observation System

VFS introduces a **second ObservationBuilder** for compile-time spec generation:

- **Runtime**: `environment.ObservationBuilder` (builds tensors) - UNCHANGED
- **Compile-time**: `vfs.VFSObservationSpecBuilder` (builds schemas) - NEW

Both systems coexist in Phase 1. See `docs/vfs/integration-with-observation-builder.md` for details.
```

---

## Final Validation (1-2 hours)

### Run Full Test Suite

```bash
# Run all VFS tests
uv run pytest tests/test_townlet/unit/vfs/ -v
uv run pytest tests/test_townlet/integration/vfs/ -v

# Run ActionConfig extension tests
uv run pytest tests/test_townlet/unit/environment/test_action_config_extension.py -v

# Run full test suite to ensure no regressions
uv run pytest tests/test_townlet/ -v
```

### Code Coverage Check

```bash
uv run pytest tests/test_townlet/unit/vfs/ --cov=src/townlet/vfs --cov-report=term-missing
# Target: 90%+ coverage for VFS module
```

### Documentation

Create `src/townlet/vfs/README.md`:

```markdown
# Variable & Feature System (VFS) - Phase 1

## Purpose

VFS provides typed contracts for variables, observations, and action dependencies.
Enables BAC compiler (TASK-005) and UAC compiler (TASK-004A) to work with
declarative variable specifications.

## Quick Start

```python
from townlet.vfs import VariableDef, VariableRegistry, VFSObservationSpecBuilder

# Define variables
variables = [
    VariableDef(
        id="energy",
        scope="agent",
        type="scalar",
        lifetime="episode",
        readable_by=["agent"],
        writable_by=["actions"],
        default=1.0
    )
]

# Create registry
registry = VariableRegistry(variables, num_agents=4)

# Set/get values
registry.set("agent", "energy", 0.5, agent_id=0, writer="actions")
value = registry.get("agent", "energy", agent_id=0, reader="agent")

# Build observation spec for BAC
builder = VFSObservationSpecBuilder()
exposures = {"energy": {"normalization": None}}
obs_spec = builder.build_observation_spec(variables, exposures)
```

## Scope Semantics

- **global**: Single value shared by all agents (tensor shape `[]` or `[dims]`)
- **agent**: Per-agent values, publicly readable (`[num_agents]` or `[num_agents, dims]`)
- **agent_private**: Per-agent values, owner-only access (`[num_agents]` or `[num_agents, dims]`)

## Type System (Phase 1)

- `scalar`: Single float
- `vec2i`, `vec3i`: Fixed-dimension integer vectors
- `vecNi`, `vecNf`: Variable-dimension vectors (requires `dims` field)
- `bool`: Boolean flag

## Phase 2 (Future)

Deferred to post-TASK-004:
- Feature derivation graphs
- Complex types (stack, queue, map, struct)
- Incremental mask optimization
```

---

## Commit Strategy

### After Each TDD Cycle

```bash
# Cycle 0: Observation reverse engineering (documentation-only)
git add docs/vfs/observation-dimension-formulas.md docs/vfs/observation-dimension-manual-validation.md configs/*/variables_reference.yaml
git commit -m "docs(vfs): Add observation dimension reverse engineering (TDD Cycle 0)

- Document current observation dimension formulas
- Create TEMPORARY reference variable definitions for all configs
- Manual dimension calculation and validation
- NOTE: Reference variables are scaffolding (deleted in Cycle 7)
- NOTE: Automated validation script deferred to Cycle 5 (eliminates circular dependency)
- CRITICAL: Prevents checkpoint incompatibility"

# Cycle 1: Schema definitions
git add src/townlet/vfs/schema.py src/townlet/vfs/__init__.py tests/test_townlet/unit/vfs/test_schema.py
git commit -m "feat(vfs): Add schema definitions with validation (TDD Cycle 1)

- VariableDef: typed variable declarations
- ObservationField: observation exposure specs
- WriteSpec: action write declarations
- NormalizationSpec: observation normalization
- Expanded type system: scalar, vec2i, vec3i, vecNi, vecNf, bool
- Pydantic validation for dims field (vecNi/vecNf require it)
- 10 schema validation tests (all passing)"

# Cycle 2: Variable registry
git add src/townlet/vfs/registry.py tests/test_townlet/unit/vfs/test_registry.py
git commit -m "feat(vfs): Add VariableRegistry with access control (TDD Cycle 2)

- Scope-aware storage (global/agent/agent_private)
- get/set with access control enforcement
- Default value initialization
- Per-agent and global tensor management
- 18 registry operation tests (all passing)"

# Cycle 3: VFS observation spec builder
git add src/townlet/vfs/observation_builder.py tests/test_townlet/unit/vfs/test_observation_builder.py
git commit -m "feat(vfs): Add VFSObservationSpecBuilder for spec generation (TDD Cycle 3)

- Build observation specs from variables + exposures
- Shape inference from variable types
- Normalization parameter pass-through
- Support all substrate dimensionalities
- Renamed to avoid collision with environment.ObservationBuilder
- 6 observation spec builder tests (all passing)"

# Cycle 4: ActionConfig extension
git add src/townlet/environment/action_config.py tests/test_townlet/unit/environment/test_action_config_extension.py
git commit -m "feat(vfs): Extend ActionConfig with reads/writes fields (TDD Cycle 4)

- Add reads: List[str] field for variable dependencies
- Add writes: List[WriteSpec] field for variable effects
- Backward compatible (defaults to empty lists)
- 4 ActionConfig extension tests (all passing)"

# Cycle 5: Dimension regression + validation script
git add tests/test_townlet/unit/vfs/test_observation_dimension_regression.py scripts/validate_vfs_dimensions.py
git commit -m "test(vfs): Add CRITICAL observation dimension regression tests + validation script (TDD Cycle 5)

- Verify VFS dim matches current hardcoded calculation
- Test all configs (L0, L0.5, L1, L2, L3)
- Add automated validation script (moved from Cycle 0)
- Validation script uses VFSObservationSpecBuilder (no circular dependency)
- Prevent checkpoint incompatibility
- 5 dimension regression tests (all passing)"

# Cycle 6: Integration tests
git add tests/test_townlet/integration/vfs/
git commit -m "test(vfs): Add integration tests for variable→observation flow (TDD Cycle 6)

- End-to-end: variable definition → registry → observation spec
- Validates BAC compiler workflow
- 3 integration tests (all passing)"

# Cycle 7: Config templates + cleanup
git add configs/templates/variables.yaml
git rm configs/L0_0_minimal/variables_reference.yaml configs/L0_5_dual_resource/variables_reference.yaml configs/L1_full_observability/variables_reference.yaml configs/L2_partial_observability/variables_reference.yaml configs/L3_temporal_mechanics/variables_reference.yaml
git commit -m "docs(vfs): Add variables.yaml template + remove temporary scaffolding (TDD Cycle 7)

- Example variable definitions for all types
- Example observation exposures
- Documentation of scope semantics
- Covers global, agent, agent_private scopes
- CLEANUP: Remove temporary reference variables (validation now uses template)"

# Cycle 8: Integration specification
git add docs/vfs/integration-with-observation-builder.md tests/test_townlet/integration/vfs/test_observation_system_coexistence.py src/townlet/vfs/README.md
git commit -m "docs(vfs): Add integration specification for dual observation system (TDD Cycle 8)

- Document coexistence of legacy and VFS observation builders
- Integration test validates dimension compatibility
- Both systems work together without conflicts
- Phase 2 migration path documented
- 3 integration coexistence tests (all passing)"

# Final: Documentation
git add src/townlet/vfs/README.md
git commit -m "docs(vfs): Add VFS module README with quick start guide

- Purpose and architecture overview
- Quick start examples
- Scope semantics reference
- Type system documentation
- Phase 2 roadmap"
```

---

## Success Criteria Checklist

After completing all TDD cycles, verify:

- [ ] **CRITICAL**: Checkpoints backed up before any validation testing
- [ ] All 25-30 unit tests pass
- [ ] All 3+ integration tests pass (including coexistence test)
- [ ] **CRITICAL**: All observation dimension regression tests pass
- [ ] **CRITICAL**: Automated validation script (`scripts/validate_vfs_dimensions.py`) passes
- [ ] Code coverage ≥90% for VFS module
- [ ] Code coverage ≥85% for ActionConfig extension
- [ ] Example `variables.yaml` validates correctly
- [ ] VFS module README exists with integration section
- [ ] Integration specification document exists (`docs/vfs/integration-with-observation-builder.md`)
- [ ] All Pydantic validators working
- [ ] Access control enforcement tested
- [ ] Backward compatibility maintained (ActionConfig tests pass)
- [ ] No regressions in existing test suite
- [ ] **CLEANUP**: Temporary reference variables removed (Cycle 7)
- [ ] **VALIDATION**: VFS dimensions match legacy ObservationBuilder dimensions

---

## Time Tracking

| Cycle | Task | Estimated | Actual | Notes |
|-------|------|-----------|--------|-------|
| Setup | **Checkpoint backup + module structure** | **0.75h** | | **UPDATED** - added safety backup step |
| **0** | **Observation reverse engineering (DOCS ONLY)** | **4-6h** | | **CHANGED** - no automation (eliminates circular dependency) |
| 1 | Schema definitions | 4h | | Increased from 2h |
| 2 | Variable registry | 10-12h | | Increased from 4-5h (scope complexity) |
| 3 | VFS observation spec builder | 6-8h | | Renamed + increased from 3-4h |
| 4 | ActionConfig extension | 1-2h | | Unchanged |
| 5 | **Dimension regression + validation script** | **3-4h** | | **UPDATED** - now includes validation script (moved from Cycle 0) |
| 6 | Integration tests | 3-4h | | Increased from 2-3h (dual system) |
| 7 | **Config templates + cleanup** | **1h** | | **UPDATED** - includes removing temporary reference variables |
| **8** | **Integration specification** | **2h** | | **NEW** - was previously orphaned |
| Final | Validation & docs | 1-2h | | Unchanged |
| **Total** | | **40-52h (5-6.5 days)** | | **CORRECTED** - was 38-50h, missing integration spec |

---

## Troubleshooting Guide

### Cycle 0: Cannot Complete Validation

**Problem**: No automated validation script yet (deferred to Cycle 5)
**Solution**: This is expected - Cycle 0 is documentation-only
- Focus on creating accurate reference variable definitions
- Manual dimension calculation is sufficient
- Automated validation happens in Cycle 5 after VFSObservationSpecBuilder exists

### Tests Failing in Cycle 2 (Registry)

**Problem**: Access control tests failing
**Solution**: Verify `readable_by`/`writable_by` lists match between test and implementation

**Problem**: Tensor shape mismatches
**Solution**: Check scope semantics - global should be `[]`, agent should be `[num_agents]`

### Tests Failing in Cycle 5 (Dimension Regression)

**Problem**: VFS dim != current dim
**Solution**: Do NOT change current calculation - adjust VFS to match it
1. Print both dimensions: `print(f"VFS: {vfs_dim}, Current: {current_dim}")`
2. Identify which variables are missing/extra
3. Adjust variable definitions or exposures
4. Re-run tests

**Problem**: Partial observability dimensions wrong
**Solution**: Remember POMDP uses window_size^position_dim, not full grid encoding

### Integration Tests Failing

**Problem**: End-to-end flow broken
**Solution**: Run unit tests first - integration depends on them all passing

### Cycle 8: Integration Test Failures

**Problem**: VFS and legacy dimensions don't match
**Solution**: This should have been caught in Cycle 5 - go back and fix dimension regression tests first

**Problem**: Import name collision between ObservationBuilder classes
**Solution**: Verify naming:
- Legacy: `from townlet.environment.observation_builder import ObservationBuilder`
- VFS: `from townlet.vfs.observation_builder import VFSObservationSpecBuilder`

---

**⚠️ CRITICAL**: Document how VFS integrates with existing VectorizedHamletEnv observation system.

### Dual Observation System Coexistence

**Two ObservationBuilder classes exist**:
1. **`environment.observation_builder.ObservationBuilder`** (existing)
   - Purpose: Runtime observation construction (builds tensors)
   - Input: positions, meters, affordances
   - Output: `torch.Tensor` [num_agents, observation_dim]
   - Used by: VectorizedHamletEnv.reset(), VectorizedHamletEnv.step()

2. **`vfs.observation_builder.VFSObservationSpecBuilder`** (new)
   - Purpose: Compile-time spec generation (builds schemas)
   - Input: List[VariableDef], exposures dict
   - Output: List[ObservationField] (Pydantic schemas)
   - Used by: BAC compiler (TASK-005), UAC compiler (TASK-004A)

### Integration Points

**Phase 1 (This Task)**: Parallel systems, no replacement
```python
# Existing system (continues to work)
env = VectorizedHamletEnv(...)
obs = env.reset()  # Uses environment.ObservationBuilder

# New system (for compilers)
builder = VFSObservationSpecBuilder()
obs_spec = builder.build_observation_spec(variables, exposures)
# Returns schema, not tensors
```

**Phase 2 (Future - Post-TASK-004)**: Migration path
```python
# Environment can optionally use VFS variables
if config_pack_path / "variables.yaml" exists:
    # Load VFS variable definitions
    variables = load_variables_yaml(config_pack_path / "variables.yaml")
    registry = VariableRegistry(variables, num_agents, device)

    # Build observation spec from VFS
    builder = VFSObservationSpecBuilder()
    obs_spec = builder.build_observation_spec(variables, exposures)

    # Use spec to construct observations
    obs = construct_from_spec(obs_spec, registry)
else:
    # Legacy path (current hardcoded system)
    obs = legacy_observation_builder.build_observations(...)
```

### Decision Logic

**When to use VFS path vs legacy path**:
- **VFS path**: If `config_pack_path / "variables.yaml"` exists
- **Legacy path**: Otherwise (backward compatibility)

### Testing Strategy

**Tests must validate coexistence**:
1. Legacy ObservationBuilder still works (no regressions)
2. VFSObservationSpecBuilder produces compatible specs
3. Both systems can run in same process
4. Dimension validation passes for both systems

---

## Next Steps After Completion

1. **Verify Cleanup**: Ensure temporary reference variables deleted (Cycle 7)
2. **Run Full Validation**: Execute `scripts/validate_vfs_dimensions.py` one final time
3. **Update TASK-002C.md**: Mark as completed with lessons learned
4. **Create PR**: Title "feat(vfs): Implement Variable & Feature System Phase 1"
5. **Tag for review**: Request code review focusing on:
   - Schema design (extensibility for Phase 2)
   - Access control enforcement
   - Observation dimension compatibility (CRITICAL)
   - Integration with legacy ObservationBuilder
6. **Prepare for TASK-004A**: VFS schemas ready for UAC compiler integration
7. **Prepare for TASK-005**: ObservationField specs ready for BAC compiler

---

## Implementation Readiness Checklist

Before starting Cycle 0, verify:

- [ ] Peer review section read and understood
- [ ] Blockers acknowledged as resolved
- [ ] Total time estimate (40-52h) acceptable
- [ ] Backup strategy understood (Pre-Implementation Setup)
- [ ] Cycle 0 purpose clear (documentation, not automation)
- [ ] Integration path clear (Cycle 8 describes coexistence)
- [ ] Ready to commit to TDD discipline (no shortcuts)

---

**END OF TDD IMPLEMENTATION PLAN**
**PEER REVIEWED**: 2025-11-07
**STATUS**: READY FOR IMPLEMENTATION
