# Implementation Plan: TASK-001 Variable-Size Meter System (TDD-Ready)

**Task**: TASK-001 Variable-Size Meter System
**Priority**: CRITICAL
**Effort**: 15-21 hours (updated for test infrastructure integration)
**Status**: Ready for TDD Implementation
**Created**: 2025-11-04
**Updated**: 2025-11-04 (integrated with new test infrastructure)
**Method**: Research → Plan → Review Loop (see `docs/methods/RESEARCH-PLAN-REVIEW-LOOP.md`)

---

## Executive Summary

This plan details the TDD implementation approach for removing the hardcoded 8-meter constraint from HAMLET/Townlet. The system currently validates that all universes have exactly 8 meters with indices [0-7]. This artificial constraint blocks:
- 4-meter pedagogical tutorials (L0: energy, health, money, mood)
- 12-meter complex simulations (add reputation, skill, spirituality, community_trust)
- 16-meter research experiments
- Domain-specific universes (factory sims, trading bots, ecosystems)

**Key Insight**: Meter count is **metadata**, not a fixed constant. The fix unblocks entire design space for 13-19 hours of effort.

**Implementation Strategy**: Test-Driven Development (TDD) with RED-GREEN-REFACTOR cycle applied to each phase.

---

## Problem Statement

### Current Constraint

**File**: `src/townlet/environment/cascade_config.py:70-76`

```python
@field_validator("bars")
def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
    if len(v) != 8:  # ❌ HARDCODED
        raise ValueError(f"Expected 8 bars, got {len(v)}")

    indices = {bar.index for bar in v}
    if indices != {0, 1, 2, 3, 4, 5, 6, 7}:  # ❌ HARDCODED
        raise ValueError(f"Bar indices must be 0-7, got {sorted(indices)}")
```

**From UNIVERSE_AS_CODE.md**:
> "Those indices are wired everywhere (policy nets, replay buffers, cascade maths, affordance effects). Changing them casually will break everything. So we treat them as stable ABI."

### Why This Is Technical Debt

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: **More expressive**
- ✅ Enables 4-meter tutorials (L0: just energy + health)
- ✅ Enables 12-meter complex universes (add reputation, skill, spirituality)
- ✅ Enables domain-specific universes (factory, trading bot, ecosystem)
- ❌ Does NOT make system more fragile (obs_dim already computed dynamically in places)

**Conclusion**: 8-bar limit is **technical debt masquerading as a design constraint**.

---

## Codebase Analysis Summary

### Hardcoded 8-Meter Locations (From Subagent Analysis)

| File | Lines | Issue | Change Required |
|------|-------|-------|-----------------|
| **cascade_config.py** | 27, 70-71, 75, 117, 119 | Validation `le=7`, `len != 8`, `indices != {0..7}` | Use `len(bars)` dynamic |
| **vectorized_env.py** | 119-120, 123-124 | obs_dim calculation `+ 8` | Use `meter_count` from config |
| **vectorized_env.py** | 161 | `torch.zeros((num_agents, 8))` | Use `len(bars)` |
| **vectorized_env.py** | 290, 470, 550 | Hardcoded indices `[0]`, `[3]`, `[6]` | Use `meter_name_to_index` lookup |
| **vectorized_env.py** | 400-410, 419-429 | 8-element cost arrays | Build dynamically from config |
| **affordance_config.py** | 27-36, 198-207 | METER_NAME_TO_IDX dict (duplicate!) | Load from bars.yaml |
| **cascade_engine.py** | 72, 321 | `torch.zeros(8, ...)` | Use `len(bars)` |
| **cascade_engine.py** | 53, 67, 70, 149, etc. | 13 comments referencing "[8]" | Update to "[meter_count]" |

---

## Solution Architecture

### Design Principles

1. **Meter count is metadata**: Loaded from `bars.yaml`, not hardcoded
2. **Dynamic tensor sizing**: All tensors sized `[num_agents, meter_count]`
3. **Name-based access**: Use `meter_name_to_index` dict instead of hardcoded indices
4. **Backward compatibility**: Existing 8-meter configs still work
5. **Fail-fast validation**: Checkpoint loading validates meter count matches

### Architecture Changes

**Layer 1: Config (bars.yaml)**
- Remove validation requiring exactly 8 meters
- Add `meter_count` property to BarsConfig
- Add `meter_name_to_index` property for name lookups
- Validate indices are contiguous from 0

**Layer 2: Engine (VectorizedHamletEnv, CascadeEngine)**
- Size all tensors dynamically: `torch.zeros((num_agents, meter_count))`
- Replace hardcoded indices with name lookups
- Build cost arrays dynamically from config

**Layer 3: Network (SimpleQNetwork, RecurrentSpatialQNetwork)**
- Compute `obs_dim` dynamically from meter_count
- Networks already take `obs_dim` as parameter (no changes needed)

**Layer 4: Checkpoint**
- Store `universe_metadata` with meter_count
- Validate meter_count matches on load
- Handle legacy checkpoints (assume 8 meters)

---

## Test Infrastructure Integration

### Overview of New Test Infrastructure (November 2025)

HAMLET/Townlet has a comprehensive test infrastructure with **560 tests** (67% coverage):
- **Unit tests** (`tests/test_townlet/unit/`) - Isolated component testing (426 tests)
- **Integration tests** (`tests/test_townlet/integration/`) - Cross-component interactions (114 tests)
- **Property tests** (`tests/test_townlet/properties/`) - Hypothesis-based fuzzing (20 tests)
- **Shared fixtures** (`conftest.py`) - Eliminate test duplication

**TASK-001 tests will integrate into this structure**, not create parallel test files.

### Key Test Infrastructure Patterns

#### 1. Always Use `cpu_device` Fixture
```python
def test_my_feature(cpu_device):
    """Always use CPU for deterministic tests."""
    env = VectorizedHamletEnv(..., device=cpu_device)  # ✅ Not device="cuda"
```

**Why**: Prevents GPU randomness, ensures reproducible results.

#### 2. Fixture Composition Pattern
```python
# conftest.py provides composed fixtures
@pytest.fixture
def config_4meter(tmp_path, test_config_pack_path):
    """4-meter config pack (composes base fixtures)."""
    # ...

@pytest.fixture
def env_4meter(cpu_device, config_4meter):
    """4-meter environment (composes config + device)."""
    return VectorizedHamletEnv(..., device=cpu_device, config_pack_path=config_4meter)
```

**Why**: Reusable, composable, eliminates duplication.

#### 3. Behavioral Assertions (Not Exact Values)
```python
# ✅ Good: Behavioral
assert late_survival.mean() > early_survival.mean(), "Agents should improve"

# ❌ Bad: Exact value (fragile)
assert late_survival.mean() == 123.45, "Must be exactly 123.45"
```

**Why**: Tests should verify behavior, not implementation details.

### Test File Organization for TASK-001

| Phase | Test File | Type |
|-------|-----------|------|
| **Phase 1** | `unit/environment/test_variable_meter_config.py` | Unit (NEW) |
| **Phase 2** | `unit/environment/test_variable_meter_engine.py` | Unit (NEW) |
| **Phase 3** | `unit/agent/test_variable_meter_networks.py` | Unit (NEW) |
| **Phase 4** | `integration/test_checkpointing.py` | Integration (EXTEND) |
| **Phase 5** | `integration/test_variable_meter_integration.py` | Integration (NEW) |

**Rationale**: Mirrors existing structure - unit tests in component directories (`environment/`, `agent/`), integration tests in `integration/`.

### Required Conftest.py Fixtures

Before starting Phase 1, add these fixtures to `tests/test_townlet/conftest.py`:

```python
# =============================================================================
# TASK-001: VARIABLE METER CONFIG FIXTURES
# =============================================================================
# Required imports (add to top of conftest.py if not present):
# import copy
# import shutil
# import yaml

@pytest.fixture
def config_4meter(tmp_path, test_config_pack_path):
    """Create temporary 4-meter config pack for testing.

    Meters: energy, health, money, mood
    Use for: TASK-001 variable meter config/engine tests
    """
    config_4m = tmp_path / "config_4m"
    shutil.copytree(test_config_pack_path, config_4m)

    # Create 4-meter bars.yaml
    bars_config = {
        "version": "2.0",
        "description": "4-meter test universe",
        "bars": [
            {"name": "energy", "index": 0, "tier": "pivotal",
             "range": [0.0, 1.0], "initial": 1.0, "base_depletion": 0.005},
            {"name": "health", "index": 1, "tier": "pivotal",
             "range": [0.0, 1.0], "initial": 1.0, "base_depletion": 0.0},
            {"name": "money", "index": 2, "tier": "resource",
             "range": [0.0, 1.0], "initial": 0.5, "base_depletion": 0.0},
            {"name": "mood", "index": 3, "tier": "secondary",
             "range": [0.0, 1.0], "initial": 0.7, "base_depletion": 0.001},
        ],
        "terminal_conditions": [
            {"meter": "energy", "operator": "<=", "value": 0.0},
            {"meter": "health", "operator": "<=", "value": 0.0},
        ],
    }

    with open(config_4m / "bars.yaml", 'w') as f:
        yaml.safe_dump(bars_config, f)

    # Simplify cascades.yaml
    cascades_config = {
        "version": "2.0",
        "modulations": [],
        "cascades": [
            {"name": "low_mood_hits_energy", "category": "secondary_to_pivotal",
             "source": "mood", "source_index": 3, "target": "energy",
             "target_index": 0, "threshold": 0.2, "strength": 0.01}
        ],
        "execution_order": ["secondary_to_pivotal"],
    }

    with open(config_4m / "cascades.yaml", 'w') as f:
        yaml.safe_dump(cascades_config, f)

    return config_4m


@pytest.fixture
def config_12meter(tmp_path, test_config_pack_path):
    """Create temporary 12-meter config pack for testing.

    Meters: 8 standard + reputation, skill, spirituality, community_trust
    Use for: TASK-001 variable meter scaling tests
    """
    config_12m = tmp_path / "config_12m"
    shutil.copytree(test_config_pack_path, config_12m)

    # Load existing 8-meter bars
    with open(test_config_pack_path / "bars.yaml", 'r') as f:
        bars_8m = yaml.safe_load(f)

    # Add 4 new meters
    extra_meters = [
        {"name": "reputation", "index": 8, "tier": "secondary",
         "range": [0.0, 1.0], "initial": 0.5, "base_depletion": 0.002},
        {"name": "skill", "index": 9, "tier": "secondary",
         "range": [0.0, 1.0], "initial": 0.3, "base_depletion": 0.001},
        {"name": "spirituality", "index": 10, "tier": "secondary",
         "range": [0.0, 1.0], "initial": 0.6, "base_depletion": 0.002},
        {"name": "community_trust", "index": 11, "tier": "secondary",
         "range": [0.0, 1.0], "initial": 0.7, "base_depletion": 0.001},
    ]

    bars_12m = copy.deepcopy(bars_8m)  # Deep copy to avoid modifying original
    bars_12m["bars"].extend(extra_meters)

    with open(config_12m / "bars.yaml", 'w') as f:
        yaml.safe_dump(bars_12m, f)

    return config_12m


@pytest.fixture
def env_4meter(cpu_device, config_4meter):
    """4-meter environment for TASK-001 testing."""
    return VectorizedHamletEnv(
        num_agents=1, grid_size=8, partial_observability=False,
        device=cpu_device, config_pack_path=config_4meter,
    )


@pytest.fixture
def env_12meter(cpu_device, config_12meter):
    """12-meter environment for TASK-001 testing."""
    return VectorizedHamletEnv(
        num_agents=1, grid_size=8, partial_observability=False,
        device=cpu_device, config_pack_path=config_12meter,
    )
```

**Location**: Add after existing training component fixtures in `conftest.py`.

### Fixture Usage Example

```python
# OLD (from original plan - hardcoded paths)
def test_4_meter_env():
    env = VectorizedHamletEnv(
        num_agents=2,
        config_pack_path=Path("configs/L0_4meter_tutorial")  # ❌ Hardcoded
    )

# NEW (using fixtures)
def test_4_meter_env(cpu_device, config_4meter):
    env = VectorizedHamletEnv(
        num_agents=2,
        device=cpu_device,  # ✅ CPU for determinism
        config_pack_path=config_4meter,  # ✅ Fixture
    )
```

---

## TDD Implementation Plan

### TDD Approach: RED-GREEN-REFACTOR

Each phase follows strict TDD:

1. **RED**: Write failing test first
2. **GREEN**: Write minimal code to pass test
3. **REFACTOR**: Clean up code while keeping tests green

**Critical Rule**: Never write implementation code before writing the test.

---

## Phase 0: Setup Test Fixtures (1 hour)

### Goal
Prepare test infrastructure with fixtures for 4-meter and 12-meter config packs before starting TDD.

### 0.1: Add Fixtures to Conftest.py

**File**: `tests/test_townlet/conftest.py`

Add the fixtures documented in the "Test Infrastructure Integration" section (lines 176-289) to conftest.py.

**Required imports** (add to top of conftest.py if not present):
```python
import copy
import shutil
import yaml
```

**Fixtures to add**:
- `config_4meter`: Creates temporary 4-meter config pack
- `config_12meter`: Creates temporary 12-meter config pack
- `env_4meter`: 4-meter environment fixture
- `env_12meter`: 12-meter environment fixture

### 0.2: Verify Fixtures Load Correctly

**Smoke test** to verify fixtures work:

```bash
# Test fixture imports
python -c "import copy, shutil, yaml; print('✓ All imports available')"

# Collect tests to verify no import errors
pytest --collect-only tests/test_townlet/conftest.py

# Verify config_4meter fixture can be instantiated
pytest tests/test_townlet/unit/test_configuration.py -k "test_4_meter" --collect-only
```

### 0.3: Create Minimal Test Configs (Optional)

If 4-meter or 12-meter config packs don't exist yet, the fixtures will create them dynamically using `tmp_path`. No manual config creation needed.

### Phase 0 Success Criteria
- [ ] conftest.py has copy, shutil, yaml imports
- [ ] config_4meter fixture added
- [ ] config_12meter fixture added
- [ ] env_4meter fixture added
- [ ] env_12meter fixture added
- [ ] `pytest --collect-only` runs without import errors
- [ ] Fixtures compose correctly (tmp_path → config → env)

**Estimated Time**: 1 hour

---

## Phase 1: Config Schema Refactor (3-4 hours)

### Goal
Make `BarsConfig` accept variable-size meter lists (1-32 meters).

### 1.1: Write Tests for Variable Meter Count Validation (RED)

**Test File**: `tests/test_townlet/unit/test_configuration.py` (EXTEND existing file)

```python
"""Unit tests for variable-size meter configuration (TASK-001 Phase 1).

Tests that BarsConfig accepts variable meter counts (1-32) instead of
hardcoded 8 meters.
"""

from pathlib import Path

import pytest
import yaml

from townlet.environment.cascade_config import (
    BarConfig,
    BarsConfig,
    load_bars_config,
)


class TestVariableMeterConfigValidation:
    """Test that BarsConfig accepts variable meter counts."""

    def test_4_meter_config_validates(self, tmp_path):
        """4-meter config should validate successfully."""
        config = BarsConfig(
            version="2.0",
            description="4-meter tutorial",
            bars=[
                BarConfig(name="energy", index=0, tier="pivotal",
                         range=[0.0, 1.0], initial=1.0, base_depletion=0.005),
                BarConfig(name="health", index=1, tier="pivotal",
                         range=[0.0, 1.0], initial=1.0, base_depletion=0.0),
                BarConfig(name="money", index=2, tier="resource",
                         range=[0.0, 1.0], initial=0.5, base_depletion=0.0),
                BarConfig(name="mood", index=3, tier="secondary",
                         range=[0.0, 1.0], initial=0.7, base_depletion=0.001),
            ],
            terminal_conditions=[]
        )

        # Should NOT raise
        assert config.meter_count == 4
        assert config.meter_names == ["energy", "health", "money", "mood"]

    def test_12_meter_config_validates(self, tmp_path):
        """12-meter config should validate successfully."""
        bars = [
            BarConfig(name=f"meter_{i}", index=i, tier="secondary",
                     range=[0.0, 1.0], initial=0.5, base_depletion=0.001)
            for i in range(12)
        ]

        config = BarsConfig(
            version="2.0",
            description="12-meter complex",
            bars=bars,
            terminal_conditions=[]
        )

        assert config.meter_count == 12
        assert len(config.meter_names) == 12

    def test_existing_8_meter_config_still_validates(self, test_config_pack_path):
        """Backward compatibility: 8-meter configs still work."""
        config = load_bars_config(test_config_pack_path / "bars.yaml")
        assert config.meter_count == 8

    def test_0_meters_rejected(self):
        """Must have at least 1 meter."""
        with pytest.raises(ValueError, match="Must have at least 1 meter"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=[],  # Empty!
                terminal_conditions=[]
            )

    def test_33_meters_rejected(self):
        """Must not exceed 32 meters."""
        bars = [
            BarConfig(name=f"meter_{i}", index=i, tier="secondary",
                     range=[0.0, 1.0], initial=0.5, base_depletion=0.001)
            for i in range(33)
        ]

        with pytest.raises(ValueError, match="Too many meters.*Max 32"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=bars,
                terminal_conditions=[]
            )

    def test_non_contiguous_indices_rejected(self):
        """Indices must be contiguous from 0."""
        with pytest.raises(ValueError, match="must be contiguous"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=[
                    BarConfig(name="energy", index=0, tier="pivotal",
                             range=[0.0, 1.0], initial=1.0, base_depletion=0.005),
                    BarConfig(name="health", index=2, tier="pivotal",  # Gap! Missing 1
                             range=[0.0, 1.0], initial=1.0, base_depletion=0.0),
                ],
                terminal_conditions=[]
            )

    def test_duplicate_indices_rejected(self):
        """Indices must be unique."""
        with pytest.raises(ValueError, match="duplicate"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=[
                    BarConfig(name="energy", index=0, tier="pivotal",
                             range=[0.0, 1.0], initial=1.0, base_depletion=0.005),
                    BarConfig(name="health", index=0, tier="pivotal",  # Duplicate 0!
                             range=[0.0, 1.0], initial=1.0, base_depletion=0.0),
                ],
                terminal_conditions=[]
            )

    def test_duplicate_names_rejected(self):
        """Meter names must be unique."""
        with pytest.raises(ValueError, match="Duplicate bar names"):
            config = BarsConfig(
                version="2.0",
                description="Invalid",
                bars=[
                    BarConfig(name="energy", index=0, tier="pivotal",
                             range=[0.0, 1.0], initial=1.0, base_depletion=0.005),
                    BarConfig(name="energy", index=1, tier="pivotal",  # Duplicate name!
                             range=[0.0, 1.0], initial=1.0, base_depletion=0.0),
                ],
                terminal_conditions=[]
            )
```

**Run Tests**: `pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation -v`

**Expected**: 🔴 **ALL TESTS FAIL** (RED phase)
- Current code validates len(v) == 8
- Current code has `le=7` in Field validators
- meter_count property doesn't exist

### 1.2: Implement Variable Meter Count Validation (GREEN)

**File**: `src/townlet/environment/cascade_config.py`

```python
# CHANGE 1: Remove hardcoded le=7 constraint from BarConfig
class BarConfig(BaseModel):
    """Single meter (bar) configuration."""

    name: str = Field(description="Meter name (e.g., 'energy', 'health')")
    index: int = Field(ge=0, description="Meter index in tensor [0, meter_count-1]")  # ✅ Removed le=7
    tier: str = Field(description="Tier: 'pivotal', 'resource', 'secondary'")
    range: list[float] = Field(description="[min, max] value range")
    initial: float = Field(description="Initial value at episode start")
    base_depletion: float = Field(description="Base depletion rate per step")
    description: str | None = None
    key_insight: str | None = None


# CHANGE 2: Replace hardcoded 8 validation with dynamic 1-32 validation
class BarsConfig(BaseModel):
    """Complete bars.yaml configuration."""

    version: str = Field(description="Config version")
    description: str = Field(description="Config description")
    bars: list[BarConfig] = Field(description="List of meter configurations")
    terminal_conditions: list[TerminalCondition] = Field(description="Death conditions")
    notes: list[str] | None = None

    # NEW: Computed properties for meter metadata
    @property
    def meter_count(self) -> int:
        """Number of meters in this universe."""
        return len(self.bars)

    @property
    def meter_names(self) -> list[str]:
        """List of meter names in index order."""
        sorted_bars = sorted(self.bars, key=lambda b: b.index)
        return [bar.name for bar in sorted_bars]

    @property
    def meter_name_to_index(self) -> dict[str, int]:
        """Map meter names to indices."""
        return {bar.name: bar.index for bar in self.bars}

    @field_validator("bars")
    @classmethod
    def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
        """Validate bar list (variable size)."""
        meter_count = len(v)

        # CHANGE: Accept 1-32 meters instead of exactly 8
        if meter_count < 1:
            raise ValueError("Must have at least 1 meter")

        if meter_count > 32:  # Reasonable upper limit
            raise ValueError(f"Too many meters: {meter_count}. Max 32 supported.")

        # CHANGE: Validate indices are contiguous from 0 to meter_count-1
        indices = {bar.index for bar in v}
        expected_indices = set(range(meter_count))
        if indices != expected_indices:
            raise ValueError(
                f"Bar indices must be contiguous from 0 to {meter_count-1}, "
                f"got {sorted(indices)}"
            )

        # Check for duplicate indices (should be caught above, but explicit)
        if len(indices) != meter_count:
            raise ValueError(f"Duplicate bar indices found: {sorted(bar.index for bar in v)}")

        # Check names are unique
        names = [bar.name for bar in v]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate bar names found: {names}")

        return v


# CHANGE 3: Remove le=7 constraints from CascadeConfig
class CascadeConfig(BaseModel):
    """Configuration for a single cascade effect."""

    name: str
    source: str  # Source meter name
    source_index: int = Field(ge=0, description="Source meter index")  # ✅ Removed le=7
    target: str  # Target meter name
    target_index: int = Field(ge=0, description="Target meter index")  # ✅ Removed le=7
    threshold: float
    strength: float
    category: str
```

**Run Tests**: `pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation -v`

**Expected**: 🟢 **ALL TESTS PASS** (GREEN phase)

### 1.3: Refactor (REFACTOR)

- Extract constants: `MIN_METERS = 1`, `MAX_METERS = 32`
- Add docstrings to new properties
- Check for code duplication

```python
# At module level
MIN_METERS = 1
MAX_METERS = 32

class BarsConfig(BaseModel):
    # ...

    @field_validator("bars")
    @classmethod
    def validate_bars(cls, v: list[BarConfig]) -> list[BarConfig]:
        """Validate bar list accepts variable meter counts (1-32)."""
        meter_count = len(v)

        if meter_count < MIN_METERS:
            raise ValueError(f"Must have at least {MIN_METERS} meter")

        if meter_count > MAX_METERS:
            raise ValueError(f"Too many meters: {meter_count}. Max {MAX_METERS} supported.")

        # ... rest of validation
```

**Run Tests**: `pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation -v`

**Expected**: 🟢 **STILL PASS** after refactoring

### 1.4: Create Example Configs

**File**: `configs/L0_4meter_tutorial/bars.yaml` (NEW)

```yaml
version: "2.0"
description: "Simplified 4-meter tutorial universe"

bars:
  - name: "energy"
    index: 0
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.005
    description: "Ability to act and move"

  - name: "health"
    index: 1
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.0
    description: "General condition; death if zero"

  - name: "money"
    index: 2
    tier: "resource"
    range: [0.0, 1.0]
    initial: 0.5
    base_depletion: 0.0
    description: "Budget for affordances"

  - name: "mood"
    index: 3
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.7
    base_depletion: 0.001
    description: "Mental wellbeing"

terminal_conditions:
  - meter: "energy"
    operator: "<="
    value: 0.0
    description: "Death by exhaustion"

  - meter: "health"
    operator: "<="
    value: 0.0
    description: "Death by health failure"

notes:
  - "Simplified 4-meter universe for L0 pedagogy"
  - "Teaches basic resource management without complexity"
```

**Test**: Add test to load this config

```python
def test_4_meter_tutorial_config_loads():
    """Example 4-meter tutorial config should load."""
    config = load_bars_config(Path("configs/L0_4meter_tutorial/bars.yaml"))
    assert config.meter_count == 4
    assert config.meter_names == ["energy", "health", "money", "mood"]
    assert config.meter_name_to_index == {
        "energy": 0, "health": 1, "money": 2, "mood": 3
    }
```

### Phase 1 Success Criteria
- [ ] All validation tests pass
- [ ] BarsConfig accepts 1-32 meters
- [ ] meter_count, meter_names, meter_name_to_index properties work
- [ ] Example 4-meter config loads successfully
- [ ] Existing 8-meter configs still validate (backward compatible)

**Estimated Time**: 3-4 hours

---

## Phase 2: Engine Layer Refactor (4-6 hours)

### Goal
Make all tensor operations use dynamic meter count, not hardcoded 8.

### 2.1: Write Tests for Dynamic Tensor Sizing (RED)

**Test File**: `tests/test_townlet/unit/test_configuration.py` (EXTEND existing file with new test classes)

```python
"""Unit tests for variable-size meter engine dynamics (TASK-001 Phase 2).

Tests that engine layer creates tensors of correct size for variable meters.
"""

import pytest
import torch
from pathlib import Path

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.environment.cascade_engine import CascadeEngine


class TestVariableMeterEngineDynamics:
    """Test that engine creates tensors of correct size for variable meters."""

    def test_4_meter_env_creates_correct_tensor_shape(self, cpu_device, config_4meter):
        """4-meter env should create [num_agents, 4] tensor."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        # Meters tensor should be [2, 4]
        assert env.meters.shape == (2, 4)
        assert env.meter_count == 4

    def test_8_meter_env_still_works(self, cpu_device, test_config_pack_path):
        """Backward compat: 8-meter env still creates [num_agents, 8] tensor."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=cpu_device,
            config_pack_path=test_config_pack_path,
        )

        assert env.meters.shape == (2, 8)
        assert env.meter_count == 8

    def test_12_meter_env_creates_correct_tensor_shape(self, cpu_device, config_12meter):
        """12-meter env should create [num_agents, 12] tensor."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_12meter,
        )

        assert env.meters.shape == (2, 12)
        assert env.meter_count == 12

    def test_meters_initialized_with_config_values(self, cpu_device, config_4meter):
        """Meters should be initialized with values from bars.yaml."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        # Check initial values match bars.yaml
        # energy: 1.0, health: 1.0, money: 0.5, mood: 0.7
        assert env.meters[0, 0].item() == pytest.approx(1.0)  # energy
        assert env.meters[0, 1].item() == pytest.approx(1.0)  # health
        assert env.meters[0, 2].item() == pytest.approx(0.5)  # money
        assert env.meters[0, 3].item() == pytest.approx(0.7)  # mood

    def test_engine_uses_name_based_lookups_not_hardcoded_indices(
        self, cpu_device, config_4meter
    ):
        """Engine should use _get_meter_index() instead of hardcoded [0], [3], [6]."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        # Verify _get_meter_index() method exists and works
        energy_idx = env.bars_config.meter_name_to_index["energy"]
        health_idx = env.bars_config.meter_name_to_index["health"]
        money_idx = env.bars_config.meter_name_to_index["money"]
        mood_idx = env.bars_config.meter_name_to_index["mood"]

        assert energy_idx == 0
        assert health_idx == 1
        assert money_idx == 2
        assert mood_idx == 3

        # Verify all meter names can be looked up
        for meter_name in env.bars_config.meter_names:
            idx = env.bars_config.meter_name_to_index[meter_name]
            assert 0 <= idx < env.meter_count

    def test_observation_dim_scales_with_meter_count(self, cpu_device, config_4meter, test_config_pack_path):
        """obs_dim should scale correctly with meter count."""
        env_4 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        env_8 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=test_config_pack_path,
        )

        # obs_dim should differ by exactly 4 (meter count difference)
        # obs_dim = grid_size² + meter_count + affordances + extras
        assert env_8.observation_dim == env_4.observation_dim + 4

    def test_cascade_engine_uses_dynamic_meter_count(self, cpu_device, config_4meter):
        """CascadeEngine should build tensors of size [meter_count]."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        # Base depletion tensor should be [4]
        depletions = env.cascade_engine.base_depletions
        assert depletions.shape == (4,)

        # Apply depletions should work
        initial_meters = env.meters.clone()
        env.cascade_engine.apply_base_depletions(env.meters)

        # Energy should have depleted
        assert env.meters[0, 0] < initial_meters[0, 0]
```

**Run Tests**: `pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine tests/test_townlet/unit/test_configuration.py::TestCascadeEngineVariableMeters tests/test_townlet/unit/test_configuration.py::TestObservationBuilderVariableMeters -v`

**Expected**: 🔴 **ALL TESTS FAIL** (RED phase)
- Current code hardcodes `torch.zeros((num_agents, 8))`
- Current code hardcodes obs_dim calculation with `+ 8`

### 2.2: Implement Dynamic Tensor Sizing (GREEN)

**File**: `src/townlet/environment/vectorized_env.py`

```python
class VectorizedHamletEnv:
    def __init__(self, ...):
        # ... existing init code ...

        # Load bars config
        self.bars_config = load_bars_config(config_pack_path / "bars.yaml")

        # CHANGE: Store meter count from config
        self.meter_count = self.bars_config.meter_count

        # CHANGE: Create meters tensor with dynamic size
        self.meters = torch.zeros(
            (self.num_agents, self.meter_count),  # ✅ Dynamic!
            dtype=torch.float32,
            device=self.device
        )

        # CHANGE: Initialize with values from bars.yaml
        for bar in self.bars_config.bars:
            self.meters[:, bar.index] = bar.initial

        # ... rest of init ...

        # CHANGE: Compute obs_dim dynamically
        if self.partial_observability:
            self.observation_dim = (
                self.vision_window_size * self.vision_window_size +  # Local grid
                2 +  # Position (x, y)
                self.meter_count +  # ✅ Dynamic meters
                (self.num_affordance_types + 1) +  # Affordance one-hot
                4  # Temporal extras
            )
        else:
            self.observation_dim = (
                self.grid_size * self.grid_size +  # Full grid
                self.meter_count +  # ✅ Dynamic meters
                (self.num_affordance_types + 1) +  # Affordance one-hot
                4  # Temporal extras
            )

    def _get_meter_index(self, meter_name: str) -> int:
        """Get meter index by name."""
        return self.bars_config.meter_name_to_index[meter_name]

    def _check_terminal_conditions(self) -> torch.Tensor:
        """Check which agents hit terminal conditions."""
        # CHANGE: Use meter name lookup instead of hardcoded indices
        energy_idx = self._get_meter_index("energy")
        health_idx = self._get_meter_index("health")

        # Check terminal conditions from bars config
        dead = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        for condition in self.bars_config.terminal_conditions:
            meter_idx = self._get_meter_index(condition.meter)

            if condition.operator == "<=":
                dead |= self.meters[:, meter_idx] <= condition.value
            elif condition.operator == ">=":
                dead |= self.meters[:, meter_idx] >= condition.value

        return dead

    def _apply_movement_costs(self, moving_agents: torch.Tensor):
        """Apply movement costs to meters."""
        # CHANGE: Build cost tensor dynamically from config
        # For now, load from hardcoded dict (will move to actions.yaml in TASK-003)
        movement_costs = torch.zeros(self.meter_count, device=self.device)

        # Map meter names to costs
        cost_map = {
            "energy": self.move_energy_cost,
            "hygiene": 0.003,
            "satiation": 0.004,
        }

        for meter_name, cost in cost_map.items():
            if meter_name in self.bars_config.meter_name_to_index:
                meter_idx = self._get_meter_index(meter_name)
                movement_costs[meter_idx] = cost

        self.meters[moving_agents] -= movement_costs
```

**File**: `src/townlet/environment/cascade_engine.py`

```python
class CascadeEngine:
    def __init__(self, config: HamletConfig, device: torch.device):
        self.config = config
        self.device = device

        # CHANGE: Get meter count from config
        self.meter_count = len(config.bars.bars)

        # CHANGE: Build base depletion tensor dynamically
        self.base_depletions = self._build_base_depletion_tensor()

    def _build_base_depletion_tensor(self) -> torch.Tensor:
        """Build tensor of base depletion rates [meter_count]."""
        # CHANGE: Size by meter_count, not hardcoded 8
        depletions = torch.zeros(self.meter_count, device=self.device)

        for bar in self.config.bars.bars:
            depletions[bar.index] = bar.base_depletion

        return depletions

    def apply_base_depletions(self, meters: torch.Tensor) -> None:
        """Apply base depletions to meters.

        Args:
            meters: [num_agents, meter_count] current meter values
        """
        meters -= self.base_depletions  # Broadcasting: [N, M] -= [M]
        meters.clamp_(min=0.0)
```

**Run Tests**: `pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine tests/test_townlet/unit/test_configuration.py::TestCascadeEngineVariableMeters tests/test_townlet/unit/test_configuration.py::TestObservationBuilderVariableMeters -v`

**Expected**: 🟢 **ALL TESTS PASS** (GREEN phase)

### 2.3: Refactor (REFACTOR)

- Extract `_build_movement_costs()` method
- Add type hints to `_get_meter_index()`
- Consolidate meter lookups

**Run Tests**: `pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine tests/test_townlet/unit/test_configuration.py::TestCascadeEngineVariableMeters tests/test_townlet/unit/test_configuration.py::TestObservationBuilderVariableMeters -v`

**Expected**: 🟢 **STILL PASS**

### Phase 2 Success Criteria
- [ ] VectorizedHamletEnv creates tensors sized `[num_agents, meter_count]`
- [ ] CascadeEngine builds tensors sized `[meter_count]`
- [ ] obs_dim computed dynamically from meter_count
- [ ] Meter access uses name-based lookups, not hardcoded indices
- [ ] All tensor operations use dynamic sizes

**Estimated Time**: 4-6 hours

---

## Phase 3: Network Layer Updates (2-3 hours)

### Goal
Networks receive correct observation dimension based on meter count.

### 3.1: Write Tests for Network obs_dim (RED)

**Test File**: `tests/test_townlet/unit/agent/test_variable_meter_networks.py` (NEW)

```python
"""Unit tests for networks with variable meter obs_dim (TASK-001 Phase 3)."""

import pytest
import torch
from pathlib import Path

from townlet.agent.networks import SimpleQNetwork, RecurrentSpatialQNetwork
from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestVariableMeterNetworkCompatibility:
    """Test that networks handle variable meter counts correctly."""

    def test_network_with_4_meter_obs_dim(self, cpu_device, config_4meter):
        """Network should work with 4-meter obs_dim."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        network = SimpleQNetwork(obs_dim=env.observation_dim, action_dim=5).to(cpu_device)

        # Forward pass should work
        obs = env.reset()
        q_values = network(obs)

        assert q_values.shape == (1, 5)  # [batch, actions]

    def test_network_with_8_meter_obs_dim(self, cpu_device, test_config_pack_path):
        """Network should work with 8-meter obs_dim (backward compat)."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=test_config_pack_path,
        )

        network = SimpleQNetwork(obs_dim=env.observation_dim, action_dim=5).to(cpu_device)
        obs = env.reset()
        q_values = network(obs)

        assert q_values.shape == (1, 5)

    def test_network_with_12_meter_obs_dim(self, cpu_device, config_12meter):
        """Network should work with 12-meter obs_dim."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_12meter,
        )

        network = SimpleQNetwork(obs_dim=env.observation_dim, action_dim=5).to(cpu_device)
        obs = env.reset()
        q_values = network(obs)

        assert q_values.shape == (1, 5)

    def test_obs_dim_difference_equals_meter_count_difference(
        self, cpu_device, config_4meter, config_12meter
    ):
        """obs_dim should scale linearly with meter_count."""
        env_4 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        env_12 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_12meter,
        )

        # Difference should be exactly meter count difference
        obs_dim_diff = env_12.observation_dim - env_4.observation_dim
        meter_count_diff = env_12.meter_count - env_4.meter_count

        assert obs_dim_diff == meter_count_diff  # 8 meter difference
```

**Run Tests**: `pytest tests/test_townlet/unit/agent/test_variable_meter_networks.py -v`

**Expected**: 🔴 **MAY FAIL** if obs_dim not computed correctly in Phase 2
- If Phase 2 done correctly, these should pass immediately
- Networks already take obs_dim as parameter, so no changes needed

### 3.2: Verify Network Code (GREEN)

**File**: `src/townlet/agent/networks.py`

```python
class SimpleQNetwork(nn.Module):
    """
    Simple MLP Q-network for full observability.

    Observation space scales with meter count:
    - 4 meters: obs_dim = grid_size² + 4 + affordances + extras
    - 8 meters: obs_dim = grid_size² + 8 + affordances + extras
    - 12 meters: obs_dim = grid_size² + 12 + affordances + extras
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # obs_dim is computed from config (includes dynamic meter_count)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: [batch, obs_dim] observations

        Returns:
            q_values: [batch, action_dim] Q-values for each action
        """
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        q_values = self.q_head(x)
        return q_values
```

**No changes needed** - network already takes `obs_dim` as parameter.

**Run Tests**: `pytest tests/test_townlet/unit/agent/test_variable_meter_networks.py -v`

**Expected**: 🟢 **ALL TESTS PASS** (GREEN phase)

### 3.3: Update Runner Logging (REFACTOR)

**File**: `src/townlet/demo/runner.py`

```python
def main():
    # ... env setup ...

    # Log meter count for debugging
    logger.info(f"Universe configuration:")
    logger.info(f"  Meter count: {env.meter_count}")
    logger.info(f"  Meter names: {env.bars_config.meter_names}")
    logger.info(f"  Observation dimension: {env.observation_dim}")
    logger.info(f"  Action dimension: {env.action_dim}")

    # Create network
    network = SimpleQNetwork(obs_dim=env.observation_dim, action_dim=env.action_dim)
    logger.info(f"Network created with obs_dim={env.observation_dim}")
```

### Phase 3 Success Criteria
- [ ] Network creation uses dynamically computed obs_dim
- [ ] Networks work with 4, 8, and 12-meter universes
- [ ] Forward pass works for all meter counts
- [ ] Logging shows meter count and obs_dim

**Estimated Time**: 2-3 hours

---

## Phase 4: Checkpoint Compatibility (2-3 hours)

### Goal
Store meter_count in checkpoint metadata; validate on load.

### 4.1: Write Tests for Checkpoint Metadata (RED)

**Test File**: `tests/test_townlet/integration/test_checkpointing.py` (EXTEND existing)

Add the following test class to the existing checkpoint test file:

```python
"""
Checkpoint tests for variable-size meter system (TASK-001).

Extends existing integration/test_checkpointing.py with tests for:
- Checkpoint metadata includes meter count
- Loading validates meter count matches
- Legacy checkpoints load with backward compatibility
"""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.training.state import save_checkpoint, load_checkpoint


class TestVariableMeterCheckpoints:
    """Test checkpoint saving/loading with variable meters (TASK-001)."""

    def test_checkpoint_includes_meter_metadata(
        self, cpu_device, config_4meter, tmp_path
    ):
        """Saved checkpoint should include meter count and names."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save checkpoint
        save_checkpoint(
            population=env.population,  # Assuming population exists
            env=env,
            episode=100,
            path=checkpoint_path
        )

        # Load and check metadata
        checkpoint_data = torch.load(checkpoint_path)

        assert "universe_metadata" in checkpoint_data
        assert checkpoint_data["universe_metadata"]["meter_count"] == 4
        assert checkpoint_data["universe_metadata"]["meter_names"] == [
            "energy", "health", "money", "mood"
        ]
        assert "version" in checkpoint_data["universe_metadata"]
        assert "obs_dim" in checkpoint_data["universe_metadata"]

    def test_loading_checkpoint_validates_meter_count(
        self, cpu_device, config_4meter, test_config_pack_path, tmp_path
    ):
        """Loading checkpoint should fail if meter counts don't match."""
        env_4 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        env_8 = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=test_config_pack_path,  # 8-meter baseline
        )

        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save from 4-meter env
        save_checkpoint(
            population=env_4.population,
            env=env_4,
            episode=100,
            path=checkpoint_path
        )

        # Try to load into 8-meter env (should fail)
        with pytest.raises(ValueError, match="meter count mismatch"):
            load_checkpoint(checkpoint_path, env_8)

    def test_loading_checkpoint_with_matching_meters_succeeds(
        self, cpu_device, config_4meter, tmp_path
    ):
        """Loading checkpoint should succeed if meter counts match."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save and load with same env type
        save_checkpoint(
            population=env.population,
            env=env,
            episode=100,
            path=checkpoint_path
        )

        checkpoint = load_checkpoint(checkpoint_path, env)

        # Should NOT raise
        assert checkpoint.episode == 100
        assert checkpoint.universe_metadata["meter_count"] == 4

    def test_legacy_checkpoint_loads_with_warning(
        self, cpu_device, test_config_pack_path, tmp_path
    ):
        """Legacy checkpoints (no metadata) should load with warning."""
        # Create fake legacy checkpoint (no universe_metadata)
        legacy_data = {
            "episode": 50,
            "q_network_state": {},
            "optimizer_state": {},
            "timestamp": "2025-01-01T00:00:00"
        }

        checkpoint_path = tmp_path / "legacy_checkpoint.pt"
        torch.save(legacy_data, checkpoint_path)

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=test_config_pack_path,  # 8-meter baseline
        )

        # Load should work (assumes 8 meters)
        with pytest.warns(UserWarning, match="legacy checkpoint"):
            checkpoint = load_checkpoint(checkpoint_path, env)

        assert checkpoint.episode == 50
```

**Run Tests**: `pytest tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints -v`

**Expected**: 🔴 **ALL TESTS FAIL** (RED phase)
- `universe_metadata` doesn't exist in current checkpoint format
- Validation on load doesn't exist

### 4.2: Implement Checkpoint Metadata (GREEN)

**File**: `src/townlet/training/state.py`

```python
from pydantic import BaseModel, Field, model_validator
from datetime import datetime


class PopulationCheckpoint(BaseModel):
    """Checkpoint for population training state with universe metadata."""

    # NEW: Universe metadata for compatibility validation
    universe_metadata: dict = Field(
        description="Universe configuration metadata (meter_count, names, version)"
    )

    # Existing fields
    episode: int = Field(description="Episode number at checkpoint")
    q_network_state: dict = Field(description="Q-network state dict")
    optimizer_state: dict = Field(description="Optimizer state dict")
    exploration_state: dict = Field(description="Exploration strategy state")
    timestamp: str = Field(description="ISO timestamp of checkpoint")

    @model_validator(mode="after")
    def validate_metadata(self) -> "PopulationCheckpoint":
        """Ensure required metadata exists."""
        required_keys = ["meter_count", "meter_names", "version", "obs_dim"]
        for key in required_keys:
            if key not in self.universe_metadata:
                raise ValueError(f"Missing required metadata: {key}")
        return self


def save_checkpoint(population, env, episode: int, path: Path) -> None:
    """Save training checkpoint with universe metadata.

    Args:
        population: VectorizedPopulation with Q-network and optimizer
        env: VectorizedHamletEnv with universe configuration
        episode: Current episode number
        path: Path to save checkpoint
    """
    checkpoint = PopulationCheckpoint(
        episode=episode,
        q_network_state=population.q_network.state_dict(),
        optimizer_state=population.optimizer.state_dict(),
        exploration_state=population.exploration.get_state(),
        timestamp=datetime.now().isoformat(),
        universe_metadata={
            "meter_count": env.meter_count,
            "meter_names": env.bars_config.meter_names,
            "version": env.bars_config.version,
            "obs_dim": env.observation_dim,
            "action_dim": env.action_dim,
        }
    )

    with open(path, "wb") as f:
        torch.save(checkpoint.model_dump(), f)

    logger.info(f"Saved checkpoint: episode {episode}, meter_count={env.meter_count}")


def load_checkpoint(path: Path, current_env) -> PopulationCheckpoint:
    """Load training checkpoint and validate compatibility.

    Args:
        path: Path to checkpoint file
        current_env: Current VectorizedHamletEnv to validate against

    Returns:
        PopulationCheckpoint with validated metadata

    Raises:
        ValueError: If meter counts don't match
    """
    with open(path, "rb") as f:
        checkpoint_data = torch.load(f)

    # Handle legacy checkpoints (no universe_metadata)
    if "universe_metadata" not in checkpoint_data:
        logger.warning(
            f"Loading legacy checkpoint (no universe_metadata). "
            f"Assuming 8-meter universe."
        )
        checkpoint_data["universe_metadata"] = {
            "meter_count": 8,
            "meter_names": ["energy", "hygiene", "satiation", "money",
                           "mood", "social", "health", "fitness"],
            "version": "1.0",
            "obs_dim": current_env.observation_dim,  # Trust current
            "action_dim": current_env.action_dim,
        }

    checkpoint = PopulationCheckpoint(**checkpoint_data)

    # VALIDATE: Meter count must match
    if checkpoint.universe_metadata["meter_count"] != current_env.meter_count:
        raise ValueError(
            f"Checkpoint meter count mismatch: "
            f"checkpoint has {checkpoint.universe_metadata['meter_count']} meters, "
            f"current environment has {current_env.meter_count} meters. "
            f"Cannot load checkpoint from different universe."
        )

    # VALIDATE: Meter names should match (log warning if different)
    checkpoint_meters = checkpoint.universe_metadata["meter_names"]
    current_meters = current_env.bars_config.meter_names
    if checkpoint_meters != current_meters:
        logger.warning(
            f"Meter names differ between checkpoint and current environment:\n"
            f"  Checkpoint: {checkpoint_meters}\n"
            f"  Current: {current_meters}\n"
            f"Proceeding with load, but this may cause issues."
        )

    logger.info(
        f"Loaded checkpoint: episode {checkpoint.episode}, "
        f"meter_count={checkpoint.universe_metadata['meter_count']}"
    )

    return checkpoint
```

**Run Tests**: `pytest tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints -v`

**Expected**: 🟢 **ALL TESTS PASS** (GREEN phase)

### 4.3: Refactor (REFACTOR)

- Extract validation logic to separate method
- Add helper to inspect checkpoint metadata without loading

```python
def inspect_checkpoint_metadata(path: Path) -> dict:
    """Inspect checkpoint metadata without loading full checkpoint.

    Args:
        path: Path to checkpoint file

    Returns:
        Universe metadata dict
    """
    with open(path, "rb") as f:
        checkpoint_data = torch.load(f)

    if "universe_metadata" not in checkpoint_data:
        return {
            "meter_count": 8,
            "meter_names": ["energy", "hygiene", "satiation", "money",
                           "mood", "social", "health", "fitness"],
            "version": "1.0 (legacy)",
        }

    return checkpoint_data["universe_metadata"]
```

### Phase 4 Success Criteria
- [ ] Checkpoints include universe_metadata with meter_count
- [ ] Loading validates meter_count matches current environment
- [ ] Loading fails clearly if meter counts don't match
- [ ] Legacy checkpoints load with warning (assume 8 meters)
- [ ] Can inspect checkpoint metadata without loading full file

**Estimated Time**: 2-3 hours

---

## Phase 5: Integration Testing (2-3 hours)

### Goal
Ensure end-to-end training works with variable meters.

### 5.1: Write Integration Tests (RED)

**Test File**: `tests/test_townlet/integration/test_variable_meter_integration.py` (NEW)

```python
"""
Integration tests for variable-size meter system (TASK-001).

Tests end-to-end training flows with 4, 8, and 12-meter universes:
- Full episode rollouts
- Training with checkpointing
- Cascade effects with variable meters
"""
import pytest
import torch
from pathlib import Path
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.vectorized import VectorizedPopulation
from townlet.training.state import save_checkpoint, load_checkpoint


class TestVariableMeterIntegration:
    """End-to-end integration tests for variable meter system (TASK-001)."""

    def test_full_training_episode_4_meters(self, cpu_device, config_4meter):
        """Full training episode with 4-meter universe should complete."""
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        # Reset and run episode
        obs = env.reset()
        assert obs.shape == (4, env.observation_dim)

        done = torch.zeros(4, dtype=torch.bool, device=cpu_device)
        steps = 0
        max_steps = 500

        while not done.all() and steps < max_steps:
            # Random policy for testing
            actions = torch.randint(0, 5, (4,), device=cpu_device)
            obs, rewards, done, info = env.step(actions)
            steps += 1

        # Should complete without errors (at least some agents die)
        assert steps < max_steps

    def test_full_training_episode_8_meters(self, cpu_device, test_config_pack_path):
        """Full training episode with 8-meter universe (backward compat)."""
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=cpu_device,
            config_pack_path=test_config_pack_path,  # 8-meter baseline
        )

        obs = env.reset()
        done = torch.zeros(4, dtype=torch.bool, device=cpu_device)
        steps = 0

        while not done.all() and steps < 500:
            actions = torch.randint(0, 5, (4,), device=cpu_device)
            obs, rewards, done, info = env.step(actions)
            steps += 1

        # Should complete without errors
        assert steps < 500

    def test_full_training_episode_12_meters(self, cpu_device, config_12meter):
        """Full training episode with 12-meter universe should complete."""
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_12meter,
        )

        obs = env.reset()
        done = torch.zeros(4, dtype=torch.bool, device=cpu_device)
        steps = 0

        while not done.all() and steps < 500:
            actions = torch.randint(0, 5, (4,), device=cpu_device)
            obs, rewards, done, info = env.step(actions)
            steps += 1

        # Should complete without errors
        assert steps < 500

    def test_training_with_checkpointing_4_meters(
        self, cpu_device, config_4meter, tmp_path
    ):
        """Training with save/load checkpoints should work."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        population = VectorizedPopulation(
            env=env,
            learning_rate=0.00025,
            gamma=0.99,
            device=cpu_device,
        )

        # Train for 10 episodes
        for episode in range(10):
            obs = env.reset()
            done = torch.zeros(2, dtype=torch.bool, device=cpu_device)

            while not done.all():
                actions = population.select_actions(obs, epsilon=0.1)
                next_obs, rewards, done, info = env.step(actions)

                # Store transition, update Q-network
                population.store_transitions(obs, actions, rewards, next_obs, done)
                population.update()

                obs = next_obs

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint_4m.pt"
        save_checkpoint(population, env, episode=10, path=checkpoint_path)

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, env)
        assert checkpoint.episode == 10
        assert checkpoint.universe_metadata["meter_count"] == 4

    def test_cascade_effects_work_with_variable_meters(
        self, cpu_device, config_4meter
    ):
        """Cascade effects should apply correctly with any meter count."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            config_pack_path=config_4meter,
        )

        # Set up specific meter values to trigger cascades
        env.meters[0, 3] = 0.1  # Low mood (index 3)

        initial_energy = env.meters[0, 0].item()

        # Apply cascades
        env.cascade_engine.apply_cascades(env.meters)

        # Energy should be affected by low mood cascade (if configured)
        # This depends on cascades.yaml for 4-meter config
        final_energy = env.meters[0, 0].item()

        # At minimum, should not crash
        assert 0.0 <= final_energy <= 1.0
```

**Run Tests**: `pytest tests/test_townlet/integration/test_variable_meter_integration.py -v`

**Expected**: 🔴 **MAY FAIL** if Phases 1-4 incomplete
- If all previous phases passed, integration tests should pass

### 5.2: Fix Any Integration Issues (GREEN)

Debug and fix any issues revealed by integration tests:
- Missing meter references in configs
- Cascade configuration issues
- Population/training loop issues

**Run Tests**: `pytest tests/test_townlet/integration/test_variable_meter_integration.py -v`

**Expected**: 🟢 **ALL TESTS PASS** (GREEN phase)

### 5.3: Add Performance Benchmarks (REFACTOR)

```python
def test_performance_comparison_meter_counts(benchmark):
    """Benchmark training speed with different meter counts."""
    def run_episode(meter_count: int):
        if meter_count == 4:
            config_path = Path("configs/L0_4meter_tutorial")
        elif meter_count == 8:
            config_path = Path("configs/L1_full_observability")
        elif meter_count == 12:
            config_path = Path("configs/L2_12meter_complex")

        env = VectorizedHamletEnv(num_agents=4, config_pack_path=config_path)
        obs = env.reset()
        done = torch.zeros(4, dtype=torch.bool)

        while not done.all():
            actions = torch.randint(0, 5, (4,), device=env.device)
            obs, rewards, done, info = env.step(actions)

    # Benchmark each meter count
    for meter_count in [4, 8, 12]:
        result = benchmark(run_episode, meter_count)
        print(f"{meter_count} meters: {result.mean:.4f}s per episode")
```

### Phase 5 Success Criteria
- [ ] Full training episode completes with 4, 8, and 12 meters
- [ ] Checkpoint save/load works across meter counts
- [ ] Cascade effects apply correctly with variable meters
- [ ] No crashes or errors in end-to-end training
- [ ] Performance acceptable for all meter counts

**Estimated Time**: 2-3 hours

---

## Configuration Examples

### Example 1: 4-Meter Minimal Tutorial

**Location**: `configs/L0_4meter_tutorial/`

**bars.yaml**: (Already shown in Phase 1)

**cascades.yaml**:
```yaml
version: "2.0"
description: "Simplified cascades for 4-meter universe"

modulations: []

cascades:
  - name: "low_mood_hits_energy"
    source: "mood"
    source_index: 3
    target: "energy"
    target_index: 0
    threshold: 0.2
    strength: 0.010
    category: "secondary_to_pivotal"

execution_order:
  - "secondary_to_pivotal"
```

**affordances.yaml**: (Simplified with 4-meter effects)

**training.yaml**:
```yaml
environment:
  grid_size: 3  # Tiny grid for simple tutorial
  enabled_affordances: ["Bed"]

population:
  num_agents: 1
  learning_rate: 0.001
  gamma: 0.99

curriculum:
  max_steps_per_episode: 100  # Short episodes

training:
  max_episodes: 500
  epsilon_decay: 0.99  # Fast learning
```

### Example 2: 12-Meter Complex Simulation

**Location**: `configs/L2_12meter_complex/`

**bars.yaml**: (Extended with reputation, skill, spirituality, community_trust)

**Pedagogical Value**:
- **4-meter**: Simplest resource management (energy, health, money, mood)
- **8-meter**: Standard curriculum (existing complexity)
- **12-meter**: Research-level sociological modeling

---

## Success Criteria (Overall)

### Config Layer
- [ ] BarsConfig accepts 1-32 meters
- [ ] Validation checks contiguous indices from 0
- [ ] meter_count, meter_names, meter_name_to_index properties work
- [ ] Example 4-meter and 12-meter configs validate
- [ ] Existing 8-meter configs still work (backward compatible)

### Engine Layer
- [ ] VectorizedHamletEnv creates tensors `[num_agents, meter_count]`
- [ ] CascadeEngine builds tensors `[meter_count]`
- [ ] obs_dim computed dynamically from meter_count
- [ ] Meter access uses name-based lookups
- [ ] No remaining hardcoded `8` in meter code

### Network Layer
- [ ] Networks receive correct obs_dim for meter_count
- [ ] Forward pass works for 4, 8, 12 meters
- [ ] Training works with variable meter counts

### Checkpoint Layer
- [ ] Checkpoints include universe_metadata
- [ ] Loading validates meter_count matches
- [ ] Loading fails clearly if mismatch
- [ ] Legacy checkpoints load with warning

### Integration
- [ ] Full training episodes complete for 4, 8, 12 meters
- [ ] Checkpoint save/load works
- [ ] Cascade effects apply correctly
- [ ] No crashes or errors

---

## Risks & Mitigations

### Risk 1: Breaking Existing Checkpoints
**Likelihood**: High (expected)
**Impact**: Low (handled gracefully)
**Mitigation**: Fallback loader assumes 8 meters for legacy checkpoints

### Risk 2: Performance Degradation
**Likelihood**: Very low
**Impact**: Negligible
**Mitigation**: GPU tensor ops scale linearly; pre-allocation prevents per-step overhead

### Risk 3: Missing Meter References in Configs
**Likelihood**: Medium (for new configs)
**Impact**: Medium (crashes at runtime)
**Mitigation**:
- TASK-004 (Universe Compiler) will catch these at compile time
- For now, add validation in AffordanceEngine

### Risk 4: Test Data Dependency
**Likelihood**: Medium (tests need 12-meter config)
**Impact**: Low (can create minimal test configs)
**Mitigation**: Create minimal 12-meter config just for testing if full config doesn't exist

---

## Estimated Effort

| Phase | Description | TDD Time | Implementation Time | Total |
|-------|-------------|----------|---------------------|-------|
| **Phase 0** | Setup test fixtures | 0h | 1h | 1h |
| **Phase 1** | Config schema refactor | 1-1.5h | 2-2.5h | 3-4h |
| **Phase 2** | Engine layer refactor | 1.5-2h | 2.5-4h | 4-6h |
| **Phase 3** | Network layer updates | 0.5-1h | 1.5-2h | 2-3h |
| **Phase 4** | Checkpoint compatibility | 1-1.5h | 1-1.5h | 2-3h |
| **Phase 5** | Integration testing | 1-1.5h | 1-1.5h | 2-3h |
| **Total** | | **4.5-7.5h** (TDD) | **9.5-12.5h** (Impl) | **15-21h** |

**TDD Overhead**: ~40% of time spent writing tests first (RED phase)
**Value**: Prevents regressions, documents expected behavior, enables confident refactoring
**Note**: Phase 0 is setup/infrastructure, not TDD (no tests written)

---

## Follow-Up Work (Post-Implementation)

1. **Create Additional Teaching Packs**:
   - L0_4meter_minimal
   - L2_12meter_social
   - L3_16meter_research

2. **Update Documentation**:
   - UNIVERSE_AS_CODE.md: Remove "exactly 8 bars" language
   - CLAUDE.md: Document variable meter system
   - Add examples to README

3. **Performance Benchmarking**:
   - Measure training speed for 4, 8, 12, 16 meters
   - Document tradeoffs

4. **Pedagogical Materials**:
   - Write lesson on "Designing Meter Systems"
   - Show meter count affects agent behavior

---

## Running TASK-001 Tests

This section provides pytest commands for running TASK-001 tests at various granularities.

### Run All TASK-001 Tests

```bash
# Run all variable meter tests (unit + integration)
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation \
       tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine \
       tests/test_townlet/unit/test_configuration.py::TestCascadeEngineVariableMeters \
       tests/test_townlet/unit/test_configuration.py::TestObservationBuilderVariableMeters \
       tests/test_townlet/unit/agent/test_networks.py::TestVariableMeterNetworks \
       tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints \
       tests/test_townlet/integration/test_variable_meter_integration.py \
       -v
```

### Run Tests By Phase

**Phase 1: Config Schema**
```bash
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation -v
```

**Phase 2: Engine Layer**
```bash
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine \
       tests/test_townlet/unit/test_configuration.py::TestCascadeEngineVariableMeters \
       tests/test_townlet/unit/test_configuration.py::TestObservationBuilderVariableMeters -v
```

**Phase 3: Network Layer**
```bash
pytest tests/test_townlet/unit/agent/test_networks.py::TestVariableMeterNetworks -v
```

**Phase 4: Checkpoint Compatibility**
```bash
pytest tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints -v
```

**Phase 5: Integration**
```bash
pytest tests/test_townlet/integration/test_variable_meter_integration.py -v
```

### Run with Coverage

```bash
# Coverage for all TASK-001 tests
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation \
       tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine \
       tests/test_townlet/unit/test_configuration.py::TestCascadeEngineVariableMeters \
       tests/test_townlet/unit/test_configuration.py::TestObservationBuilderVariableMeters \
       tests/test_townlet/unit/agent/test_networks.py::TestVariableMeterNetworks \
       tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints \
       tests/test_townlet/integration/test_variable_meter_integration.py \
       --cov=townlet.environment.cascade_config \
       --cov=townlet.environment.vectorized_env \
       --cov=townlet.environment.cascade_engine \
       --cov=townlet.environment.observation_builder \
       --cov=townlet.agent.networks \
       --cov=townlet.training.state \
       --cov-report=term-missing \
       -v
```

### Run Specific Test Classes

```bash
# Config tests
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation -v

# Engine tests
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine -v
pytest tests/test_townlet/unit/test_configuration.py::TestCascadeEngineVariableMeters -v
pytest tests/test_townlet/unit/test_configuration.py::TestObservationBuilderVariableMeters -v

# Network tests
pytest tests/test_townlet/unit/agent/test_networks.py::TestVariableMeterNetworks -v

# Checkpoint tests
pytest tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints -v

# Integration tests
pytest tests/test_townlet/integration/test_variable_meter_integration.py::TestVariableMeterIntegration -v
```

### Watch Mode (Continuous Testing During Development)

```bash
# Watch and rerun tests on file changes (requires pytest-watch)
ptw tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation -- -v
```

### Quick Smoke Test

Run one test from each phase to verify basic functionality:

```bash
pytest \
  tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation::test_4_meter_config_validates \
  tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine::test_4_meter_env_creates_correct_tensor_shape \
  tests/test_townlet/unit/agent/test_networks.py::TestVariableMeterNetworks::test_network_with_4_meter_obs_dim \
  tests/test_townlet/integration/test_checkpointing.py::TestVariableMeterCheckpoints::test_checkpoint_includes_meter_metadata \
  tests/test_townlet/integration/test_variable_meter_integration.py::TestVariableMeterIntegration::test_full_training_episode_4_meters \
  -v
```

### Fixture Verification

Test that required fixtures are working correctly:

```bash
# Verify cpu_device fixture
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterConfigValidation::test_4_meter_config_validates -v -s

# Verify config_4meter fixture creates correct config
pytest tests/test_townlet/unit/test_configuration.py::TestVariableMeterEngine::test_4_meter_env_creates_correct_tensor_shape -v -s
```

### Expected Test Counts

- **Phase 0 (Fixtures)**: ~0 tests (setup only, verified with --collect-only)
- **Phase 1 (Config)**: ~6 tests
- **Phase 2 (Engine)**: ~13 tests (5 env + 4 cascade + 4 observation)
- **Phase 3 (Network)**: ~4 tests
- **Phase 4 (Checkpoint)**: ~4 tests
- **Phase 5 (Integration)**: ~5 tests
- **Total**: ~32 tests for TASK-001

---

## Conclusion

This TDD-ready plan provides a systematic approach to removing the 8-meter constraint. Each phase follows strict RED-GREEN-REFACTOR:

1. **Write failing test first** (documents expected behavior)
2. **Write minimal code to pass** (simplest solution)
3. **Refactor while keeping tests green** (clean code)

**Total effort**: 13-19 hours (1-2 days)
**Risk**: Low (straightforward refactoring with test coverage)
**Priority**: CRITICAL (highest-leverage infrastructure change)
**Impact**: Unblocks entire design space (4-32 meters)

**Next Step**: Begin Phase 1 - Config Schema Refactor (3-4 hours)

**Slogan**: "From 'exactly 8 bars' to 'as many bars as your universe needs.'"
