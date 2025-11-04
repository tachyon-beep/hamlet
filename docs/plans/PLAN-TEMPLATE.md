# Implementation Plan: TASK-XXX [Descriptive Title] (TDD-Ready)

**Task**: TASK-XXX [Descriptive Title]
**Priority**: [CRITICAL | HIGH | MEDIUM | LOW]
**Effort**: [X-Y hours] (includes TDD, audit, testing)
**Status**: [Ready for TDD | In Progress | Review | Complete]
**Created**: YYYY-MM-DD
**Updated**: YYYY-MM-DD (describe major revisions)
**Method**: Research → Plan → Review Loop (see `docs/methods/RESEARCH-PLAN-REVIEW-LOOP.md`)

**Keywords**: [5-8 searchable keywords for AI discovery: TDD, refactoring, schema, etc.]
**Test Strategy**: [TDD | Retrofit | Hybrid]
**Breaking Changes**: [Yes/No - if yes, describe impact]

---

## AI-Friendly Summary (Skim This First!)

**What**: [One sentence describing the implementation]
**Why**: [One sentence explaining the strategic value]
**How**: [One sentence describing the TDD approach]

**Quick Assessment**:

- **Implementation Approach**: [TDD | Bottom-up | Top-down | Incremental]
- **Test Coverage Goal**: [X% unit, Y% integration]
- **Phases**: [X phases over Y-Z hours]
- **Risk Level**: [Low | Medium | High]

**Decision Point**: If you're not planning or implementing TASK-XXX, STOP READING HERE.

---

## Executive Summary

[2-3 paragraph summary including:]

- What hardcoded constraint/limitation is being removed
- Why this unblocks significant capability (the "leverage point")
- What the TDD approach brings to the implementation
- Key insight that makes the solution work

**Key Insight**: [The fundamental realization - e.g., "Meter count is metadata, not a constant"]

**Implementation Strategy**: Test-Driven Development (TDD) with RED-GREEN-REFACTOR cycle applied to each phase.

---

## Review-Driven Updates (YYYY-MM-DD)

[Add this section after plan review - document what changed and why]

This plan was reviewed by [research agent | human reviewer] and updated to address [N] critical issues:

### ✅ Issue 1: [Issue Name] RESOLVED

- **Problem**: [What was wrong with original plan]
- **Fix**: [How it was addressed]
- **Time Impact**: [+/- X hours]

### ✅ Issue 2: [Issue Name] RESOLVED

- **Problem**: [What was wrong]
- **Fix**: [How it was addressed]

### Additional Improvements

- ✅ **[Improvement 1]**: [Brief description]
- ✅ **[Improvement 2]**: [Brief description]

**Review Score**: [Before]/10 → [After]/10 after revisions (Risk: [Low|Medium|High])

**See**: `docs/reviews/REVIEW-TASK-XXX-PLAN.md` for full review details

---

## Problem Statement

### Current Constraint

**File**: `src/path/to/file.py:line-range`

```python
# Current problematic code showing hardcoded constraint
def example():
    if count != 8:  # ❌ HARDCODED
        raise ValueError("Expected exactly 8")
```

**From Documentation**:
> "[Quote from CLAUDE.md, UNIVERSE_AS_CODE.md, etc. explaining why this exists]"

### Why This Is Technical Debt

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: **More expressive**

- ✅ Enables [capability 1]
- ✅ Enables [capability 2]
- ✅ Enables [capability 3]
- ❌ Does NOT make system more fragile (explain why)

**Conclusion**: [X-constraint] is **technical debt masquerading as a design constraint**.

---

## Codebase Analysis Summary

### Hardcoded Locations (From Research/Audit)

| File | Lines | Issue | Change Required |
|------|-------|-------|-----------------|
| **file1.py** | 27, 70-71 | Validation constraint | Use dynamic value |
| **file2.py** | 119-120 | Hardcoded dimension | Compute from config |
| **file3.py** | 161 | Hardcoded tensor size | Use `len(collection)` |
| **file4.py** | 290, 470 | Hardcoded indices | Use name-based lookup |

**Total**: ~[N] locations across [M] files

---

## Solution Architecture

### Design Principles

1. **[Principle 1]**: [One sentence - e.g., "Meter count is metadata"]
2. **[Principle 2]**: [One sentence - e.g., "Dynamic tensor sizing"]
3. **[Principle 3]**: [One sentence - e.g., "Name-based access"]
4. **[Principle 4]**: [One sentence - e.g., "Backward compatibility"]
5. **[Principle 5]**: [One sentence - e.g., "Fail-fast validation"]

### Architecture Changes

**Layer 1: [Config Layer Name]**

- Change 1
- Change 2

**Layer 2: [Engine Layer Name]**

- Change 1
- Change 2

**Layer 3: [Network Layer Name]**

- Change 1
- Change 2

**Layer 4: [Checkpoint/Storage Layer Name]**

- Change 1
- Change 2

---

## Test Infrastructure Integration

### Overview of Test Infrastructure

[Project Name] has a comprehensive test infrastructure with **[N] tests** ([X]% coverage):

- **Unit tests** (`tests/test_[project]/unit/`) - Isolated component testing ([N] tests)
- **Integration tests** (`tests/test_[project]/integration/`) - Cross-component interactions ([N] tests)
- **Property tests** (`tests/test_[project]/properties/`) - Hypothesis-based fuzzing ([N] tests)
- **Shared fixtures** (`conftest.py`) - Eliminate test duplication

**TASK-XXX tests will integrate into this structure**, not create parallel test files.

### Key Test Infrastructure Patterns

#### 1. Always Use `[device_fixture]` Fixture

```python
def test_my_feature([device_fixture]):
    """Always use [device] for deterministic tests."""
    component = Component(..., device=[device_fixture])  # ✅ Not device="cuda"
```

**Why**: [Prevents GPU randomness, ensures reproducible results]

#### 2. Fixture Composition Pattern

```python
# conftest.py provides composed fixtures
@pytest.fixture
def config_variant(tmp_path, base_config):
    """Variant config (composes base fixtures)."""
    # ...

@pytest.fixture
def env_variant([device_fixture], config_variant):
    """Variant environment (composes config + device)."""
    return Environment(..., device=[device_fixture], config=config_variant)
```

**Why**: Reusable, composable, eliminates duplication.

#### 3. Behavioral Assertions (Not Exact Values)

```python
# ✅ Good: Behavioral
assert improved_metric > baseline_metric, "Should improve over baseline"

# ❌ Bad: Exact value (fragile)
assert metric == 123.45, "Must be exactly 123.45"
```

**Why**: Tests should verify behavior, not implementation details.

### Test File Organization for TASK-XXX

**IMPORTANT**: [Describe any conflicts with existing test structure]

| Phase | Test File | Type | Rationale |
|-------|-----------|------|-----------|
| **Phase 1** | `unit/[component]/test_[feature].py` | Unit (NEW) | [Why this location] |
| **Phase 2** | `unit/[component]/test_[feature].py` | Unit (EXTEND) | [Why extend existing] |
| **Phase 3** | `unit/[other]/test_[other].py` | Unit (EXTEND) | [Why this location] |
| **Phase 4** | `integration/test_[feature].py` | Integration (EXTEND) | [Why extend existing] |
| **Phase 5** | `integration/test_[feature]_integration.py` | Integration (NEW) | [Why new file] |

**Decision**: [State the decision clearly - e.g., "Create test_variable_meters.py (NEW)"]

- **Pro**: [Reason 1]
- **Pro**: [Reason 2]
- **Con**: [Tradeoff if any]

### Required Conftest.py Fixtures

Before starting Phase 1, add these fixtures to `tests/test_[project]/conftest.py`:

```python
# =============================================================================
# TASK-XXX: [FIXTURE GROUP NAME]
# =============================================================================
# Required imports (add to top of conftest.py if not present):
# import [module1]
# import [module2]

@pytest.fixture
def taskXXX_config_variant(tmp_path, base_config):
    """Create temporary variant config for TASK-XXX testing.

    [Description of what this fixture provides]
    Use ONLY for: TASK-XXX [feature] tests
    Do NOT use for: [Other purposes - use separate fixtures]
    """
    # Implementation
    return config_path


@pytest.fixture
def taskXXX_env_variant([device_fixture], taskXXX_config_variant):
    """Variant environment for TASK-XXX testing."""
    return Environment(
        device=[device_fixture],
        config_pack_path=taskXXX_config_variant,
    )
```

**Location**: Add after existing [category] fixtures in `conftest.py`.

### Fixture Usage Example

```python
# OLD (hardcoded paths - anti-pattern)
def test_feature():
    env = Environment(
        config_pack_path=Path("configs/specific_config")  # ❌ Hardcoded
    )

# NEW (using fixtures)
def test_feature([device_fixture], taskXXX_config_variant):
    env = Environment(
        device=[device_fixture],  # ✅ Deterministic
        config_pack_path=taskXXX_config_variant,  # ✅ Fixture with task prefix
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

## Phase 0: Setup Test Fixtures (X hours)

### Goal

Prepare test infrastructure with fixtures before starting TDD.

### 0.1: Add Fixtures to Conftest.py

**File**: `tests/test_[project]/conftest.py`

Add the fixtures documented in the "Test Infrastructure Integration" section.

**Required imports** (add to top of conftest.py if not present):

```python
import [module1]
import [module2]
```

**Fixtures to add**:

- `taskXXX_config_variant`: [Description]
- `taskXXX_env_variant`: [Description]

### 0.2: Verify Fixtures Load Correctly

**Smoke test** to verify fixtures work:

```bash
# Test fixture imports
python -c "import [modules]; print('✓ All imports available')"

# Collect tests to verify no import errors
pytest --collect-only tests/test_[project]/conftest.py

# Verify fixture can be instantiated
pytest tests/test_[project]/unit/test_[file].py -k "test_fixture" --collect-only
```

### Phase 0 Success Criteria

- [ ] conftest.py has required imports
- [ ] taskXXX_config_variant fixture added
- [ ] taskXXX_env_variant fixture added
- [ ] `pytest --collect-only` runs without import errors
- [ ] Fixtures compose correctly (dependencies work)

**Estimated Time**: X hours

---

## Phase 0.5: Comprehensive Hardcoded Audit (X hours)

### Goal

Identify ALL locations with hardcoded assumptions before starting implementation.

### 0.5.1: Run Comprehensive Audit

**Location**: Project root

Execute search commands to find all hardcoded references:

```bash
# Find validation constraints
grep -rn "[pattern1]" src/[project]/**/*.py

# Find hardcoded sizes
grep -rn "[pattern2]" src/[project]/**/*.py

# Find hardcoded indices
grep -rn "[pattern3]" src/[project]/**/*.py

# Find comments referencing constraint
grep -rn "[pattern4]" src/[project]/**/*.py
```

### 0.5.2: Document All Findings

Create a checklist of ALL files and line numbers that need modification:

```
[ ] file1.py:27 - [Description of issue]
[ ] file1.py:70 - [Description of issue]
[ ] file2.py:119 - [Description of issue]
[ ] file2.py:161 - [Description of issue]
[ ] (Add all findings from grep)
```

### 0.5.3: Verify No Instances Missed

Run additional searches for edge cases:

```bash
# Find edge case patterns
grep -rn "[edge_pattern]" src/[project]/**/*.py

# Find related hardcoded values
grep -rn "[related_pattern]" src/[project]/**/*.py
```

### Phase 0.5 Success Criteria

- [ ] All grep commands executed and results documented
- [ ] Checklist created with every file:line_number that needs changes
- [ ] No false positives in checklist
- [ ] Audit results saved to `docs/reviews/TASK-XXX-AUDIT-RESULTS.md`
- [ ] Estimated [N] locations identified across [M] files

**Estimated Time**: X hours

**Deliverable**: Complete audit checklist documenting all hardcoded locations

---

## Phase 1: [Layer Name] Refactor (X hours)

### Goal

[What this phase achieves - e.g., "Make BarsConfig accept variable-size lists"]

### 1.1: Write Tests for [Feature] (RED)

**Test File**: `tests/test_[project]/unit/[component]/test_[feature].py` (NEW)

```python
"""Unit tests for [feature] (TASK-XXX Phase 1).

[Description of what these tests verify]
"""

from pathlib import Path
import pytest

from [project].[module] import (
    [Class1],
    [Class2],
)


class Test[FeatureName]:
    """Test that [component] [does something]."""

    def test_minimum_boundary_validates(self, tmp_path):
        """Minimum valid: [N]-[unit] should validate successfully."""
        # Test minimum boundary condition
        assert result.property == expected_minimum

    def test_nominal_case_validates(self, tmp_path):
        """Nominal case: [description] should validate successfully."""
        # Test normal/expected case
        assert result.property == expected

    def test_maximum_boundary_validates(self, tmp_path):
        """Maximum valid: [N]-[unit] should validate successfully."""
        # Test maximum boundary condition
        assert result.property == expected_maximum

    def test_existing_configs_still_validate(self, base_config):
        """Backward compatibility: existing configs still work."""
        # Test backward compatibility
        assert result.property == legacy_value

    def test_invalid_below_minimum_rejected(self):
        """Must have at least [minimum]."""
        with pytest.raises(ValueError, match="[error pattern]"):
            # Test validation rejects below minimum

    def test_invalid_above_maximum_rejected(self):
        """Must not exceed [maximum]."""
        with pytest.raises(ValueError, match="[error pattern]"):
            # Test validation rejects above maximum

    def test_invalid_constraint_rejected(self):
        """[Constraint] must be [requirement]."""
        with pytest.raises(ValueError, match="[error pattern]"):
            # Test validation enforces constraint
```

**Run Tests**: `pytest tests/test_[project]/unit/[component]/test_[feature].py::Test[FeatureName] -v`

**Expected**: 🔴 **ALL TESTS FAIL** (RED phase)

- Current code has hardcoded constraint
- Properties don't exist yet
- Validation doesn't exist

### 1.2: Implement [Feature] Validation (GREEN)

**File**: `src/[project]/[component]/[module].py`

```python
# BEFORE (hardcoded constraint)
def validate_thing(cls, v: list[Thing]) -> list[Thing]:
    if len(v) != 8:  # ❌ HARDCODED
        raise ValueError(f"Expected 8, got {len(v)}")
    # ...

# AFTER (dynamic validation)
def validate_thing(cls, v: list[Thing]) -> list[Thing]:
    """Validate thing list (variable size)."""
    count = len(v)

    if count < MIN_COUNT:
        raise ValueError(f"Must have at least {MIN_COUNT}")

    if count > MAX_COUNT:
        raise ValueError(f"Too many: {count}. Max {MAX_COUNT} supported.")

    # Validate additional constraints
    # ...

    return v


# NEW: Add computed properties
@property
def thing_count(self) -> int:
    """Number of things in this [context]."""
    return len(self.things)

@property
def thing_names(self) -> list[str]:
    """List of thing names in order."""
    sorted_things = sorted(self.things, key=lambda t: t.index)
    return [thing.name for thing in sorted_things]
```

**Run Tests**: `pytest tests/test_[project]/unit/[component]/test_[feature].py::Test[FeatureName] -v`

**Expected**: 🟢 **ALL TESTS PASS** (GREEN phase)

### 1.3: Refactor (REFACTOR)

- Extract constants: `MIN_COUNT`, `MAX_COUNT`
- Add docstrings to new properties
- Check for code duplication

```python
# At module level
MIN_COUNT = 1
MAX_COUNT = 32

class ConfigClass(BaseModel):
    # ... existing code ...

    @field_validator("things")
    @classmethod
    def validate_things(cls, v: list[Thing]) -> list[Thing]:
        """Validate thing list accepts variable counts (1-32)."""
        count = len(v)

        if count < MIN_COUNT:
            raise ValueError(f"Must have at least {MIN_COUNT}")

        if count > MAX_COUNT:
            raise ValueError(f"Too many: {count}. Max {MAX_COUNT} supported.")

        # ... rest of validation
```

**Run Tests**: `pytest tests/test_[project]/unit/[component]/test_[feature].py::Test[FeatureName] -v`

**Expected**: 🟢 **STILL PASS** after refactoring

### 1.4: Create Example Configs (Optional)

**File**: `configs/[variant_name]/config.yaml` (NEW)

```yaml
# Example configuration showing new capability
version: "2.0"
description: "[Description of variant]"

things:
  - name: "thing1"
    # ... properties
  - name: "thing2"
    # ... properties
```

**Test**: Add test to load this config

```python
def test_example_config_loads():
    """Example [variant] config should load."""
    config = load_config(Path("configs/[variant_name]/config.yaml"))
    assert config.thing_count == [expected_count]
    assert config.thing_names == [expected_names]
```

### Phase 1 Success Criteria

- [ ] All validation tests pass
- [ ] [Component] accepts [range] of [things]
- [ ] Properties (thing_count, thing_names, etc.) work correctly
- [ ] Example [variant] config loads successfully
- [ ] Existing configs still validate (backward compatible)

**Estimated Time**: X hours

---

## Phase 2: [Layer Name] Refactor (X hours)

### Goal

[What this phase achieves - e.g., "Make all tensor operations use dynamic sizing"]

### 2.1: Write Tests for [Feature] (RED)

**Test File**: [Extend existing or new file]

```python
"""Unit tests for [feature] (TASK-XXX Phase 2)."""

import pytest
import torch  # or appropriate library

from [project].[module] import [Classes]


class Test[FeatureName]:
    """Test that [component] [does something] correctly."""

    def test_variant1_creates_correct_shape(self, [device_fixture], taskXXX_config_variant1):
        """[Variant1] should create correct tensor shape."""
        component = Component(
            device=[device_fixture],
            config=taskXXX_config_variant1,
        )

        # Verify dynamic sizing
        assert component.data.shape == expected_shape
        assert component.count == expected_count

    def test_variant2_creates_correct_shape(self, [device_fixture], taskXXX_config_variant2):
        """[Variant2] should create correct tensor shape."""
        # Similar test for different variant
        assert component.data.shape == expected_shape

    def test_initialized_with_config_values(self, [device_fixture], taskXXX_config_variant1):
        """Data should be initialized with values from config."""
        component = Component(
            device=[device_fixture],
            config=taskXXX_config_variant1,
        )

        # Check initial values match config
        assert component.data[0, 0].item() == pytest.approx(expected_value1)
        assert component.data[0, 1].item() == pytest.approx(expected_value2)

    def test_uses_name_based_lookups_not_hardcoded_indices(
        self, [device_fixture], taskXXX_config_variant1
    ):
        """Should use name-based lookups instead of hardcoded indices."""
        component = Component(
            device=[device_fixture],
            config=taskXXX_config_variant1,
        )

        # Verify lookup method exists and works
        idx1 = component.config.name_to_index["name1"]
        idx2 = component.config.name_to_index["name2"]

        assert idx1 == expected_idx1
        assert idx2 == expected_idx2

        # Verify all names can be looked up
        for name in component.config.names:
            idx = component.config.name_to_index[name]
            assert 0 <= idx < component.count
```

**Run Tests**: `pytest tests/test_[project]/unit/[component]/test_[feature].py -v`

**Expected**: 🔴 **ALL TESTS FAIL** (RED phase)

### 2.2: Implement [Feature] (GREEN)

**File**: `src/[project]/[component]/[module].py`

```python
class Component:
    def __init__(self, ...):
        # Load config
        self.config = load_config(config_path)

        # CHANGE: Store dynamic count from config
        self.count = self.config.thing_count

        # CHANGE: Create data structures with dynamic size
        self.data = torch.zeros(
            (self.num_items, self.count),  # ✅ Dynamic!
            dtype=torch.float32,
            device=self.device
        )

        # CHANGE: Initialize with values from config
        for thing in self.config.things:
            self.data[:, thing.index] = thing.initial_value

    def _get_thing_index(self, thing_name: str) -> int:
        """Get thing index by name."""
        return self.config.name_to_index[thing_name]

    def process(self):
        """Process using name-based access."""
        # CHANGE: Use name lookup instead of hardcoded indices
        thing1_idx = self._get_thing_index("thing1")
        thing2_idx = self._get_thing_index("thing2")

        # Use dynamic indices
        result = self.data[:, thing1_idx] + self.data[:, thing2_idx]
        return result
```

**Run Tests**: `pytest tests/test_[project]/unit/[component]/test_[feature].py -v`

**Expected**: 🟢 **ALL TESTS PASS** (GREEN phase)

### 2.3: Refactor (REFACTOR)

- Extract helper methods
- Add type hints
- Consolidate lookups

**Run Tests**: `pytest tests/test_[project]/unit/[component]/test_[feature].py -v`

**Expected**: 🟢 **STILL PASS**

### Phase 2 Success Criteria

- [ ] Component creates data structures sized by config
- [ ] All operations use dynamic sizes
- [ ] Name-based access works correctly
- [ ] No remaining hardcoded values in [scope]

**Estimated Time**: X hours

---

## Phase 3: [Layer Name] Updates (X hours)

### Goal

[What this phase achieves - e.g., "Networks receive correct dimensions"]

### 3.1: Write Tests for [Feature] (RED)

[Follow same RED-GREEN-REFACTOR pattern]

### 3.2: Implement [Feature] (GREEN)

[Implementation details]

### 3.3: Refactor (REFACTOR)

[Cleanup details]

### Phase 3 Success Criteria

- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

**Estimated Time**: X hours

---

## Phase 4: [Layer Name] Compatibility (X hours)

### Goal

[What this phase achieves - e.g., "Store metadata in checkpoints"]

### 4.1: Write Tests for [Feature] (RED)

**Test File**: `tests/test_[project]/integration/test_[feature].py` (EXTEND existing)

```python
"""Checkpoint tests for [feature] (TASK-XXX)."""

import pytest
import torch
from pathlib import Path

from [project].[module] import save_checkpoint, load_checkpoint


class Test[Feature]Checkpoints:
    """Test checkpoint saving/loading with [feature]."""

    def test_checkpoint_includes_metadata(
        self, [device_fixture], taskXXX_config_variant, tmp_path
    ):
        """Saved checkpoint should include [metadata]."""
        component = Component(
            device=[device_fixture],
            config=taskXXX_config_variant,
        )

        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save checkpoint
        save_checkpoint(
            component=component,
            episode=100,
            path=checkpoint_path
        )

        # Load and check metadata
        checkpoint_data = torch.load(checkpoint_path)

        assert "metadata" in checkpoint_data
        assert checkpoint_data["metadata"]["thing_count"] == expected_count
        assert checkpoint_data["metadata"]["thing_names"] == expected_names

    def test_loading_checkpoint_validates_compatibility(
        self, [device_fixture], taskXXX_config_v1, taskXXX_config_v2, tmp_path
    ):
        """Loading checkpoint should validate compatibility."""
        comp_v1 = Component(device=[device_fixture], config=taskXXX_config_v1)
        comp_v2 = Component(device=[device_fixture], config=taskXXX_config_v2)

        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save from v1
        save_checkpoint(comp_v1, episode=100, path=checkpoint_path)

        # Try to load into v2 (should fail if incompatible)
        with pytest.raises(ValueError, match="compatibility mismatch"):
            load_checkpoint(checkpoint_path, comp_v2)

    def test_legacy_checkpoint_loads_with_warning(
        self, [device_fixture], base_config, tmp_path
    ):
        """Legacy checkpoints (no metadata) should load with warning."""
        # Create fake legacy checkpoint
        legacy_data = {
            "episode": 50,
            "state": {},
            "timestamp": "2025-01-01T00:00:00"
        }

        checkpoint_path = tmp_path / "legacy_checkpoint.pt"
        torch.save(legacy_data, checkpoint_path)

        component = Component(device=[device_fixture], config=base_config)

        # Load should work (with defaults)
        with pytest.warns(UserWarning, match="legacy checkpoint"):
            checkpoint = load_checkpoint(checkpoint_path, component)

        assert checkpoint.episode == 50
```

### 4.2: Implement Checkpoint Metadata (GREEN)

**File**: `src/[project]/training/checkpoints.py`

```python
from pydantic import BaseModel, Field, model_validator


class Checkpoint(BaseModel):
    """Checkpoint with metadata."""

    # NEW: Metadata for compatibility validation
    metadata: dict = Field(description="Configuration metadata")

    # Existing fields
    episode: int = Field(description="Episode number")
    state: dict = Field(description="Component state")
    timestamp: str = Field(description="ISO timestamp")

    @model_validator(mode="after")
    def validate_metadata(self) -> "Checkpoint":
        """Ensure required metadata exists."""
        required_keys = ["thing_count", "thing_names", "version"]
        for key in required_keys:
            if key not in self.metadata:
                raise ValueError(f"Missing required metadata: {key}")
        return self


def save_checkpoint(component, episode: int, path: Path) -> None:
    """Save checkpoint with metadata."""
    checkpoint = Checkpoint(
        episode=episode,
        state=component.get_state(),
        timestamp=datetime.now().isoformat(),
        metadata={
            "thing_count": component.count,
            "thing_names": component.config.thing_names,
            "version": component.config.version,
        }
    )

    with open(path, "wb") as f:
        torch.save(checkpoint.model_dump(), f)


def load_checkpoint(path: Path, current_component) -> Checkpoint:
    """Load checkpoint and validate compatibility."""
    with open(path, "rb") as f:
        checkpoint_data = torch.load(f)

    # Handle legacy checkpoints
    if "metadata" not in checkpoint_data:
        logger.warning("Loading legacy checkpoint (no metadata). Assuming defaults.")
        checkpoint_data["metadata"] = {
            "thing_count": DEFAULT_COUNT,
            "thing_names": DEFAULT_NAMES,
            "version": "1.0",
        }

    checkpoint = Checkpoint(**checkpoint_data)

    # VALIDATE: Compatibility
    if checkpoint.metadata["thing_count"] != current_component.count:
        raise ValueError(
            f"Checkpoint compatibility mismatch: "
            f"checkpoint has {checkpoint.metadata['thing_count']}, "
            f"current has {current_component.count}. "
            f"Cannot load checkpoint from incompatible configuration."
        )

    return checkpoint
```

### Phase 4 Success Criteria

- [ ] Checkpoints include metadata
- [ ] Loading validates compatibility
- [ ] Loading fails clearly if incompatible
- [ ] Legacy checkpoints load with warning

**Estimated Time**: X hours

---

## Phase 5: Integration Testing (X hours)

### Goal

Ensure end-to-end functionality works with all variants.

### 5.1: Write Integration Tests (RED)

**Test File**: `tests/test_[project]/integration/test_[feature]_integration.py` (NEW)

```python
"""Integration tests for [feature] (TASK-XXX)."""

import pytest
import torch
from pathlib import Path

from [project].[component1] import Component1
from [project].[component2] import Component2


class Test[Feature]Integration:
    """End-to-end integration tests for [feature]."""

    def test_full_workflow_variant1(self, [device_fixture], taskXXX_config_variant1):
        """Full workflow with [variant1] should complete."""
        component = Component1(
            device=[device_fixture],
            config=taskXXX_config_variant1,
        )

        # Run workflow
        result = component.run_workflow()

        # Should complete without errors
        assert result.success
        assert result.metric > baseline_value

    def test_full_workflow_variant2(self, [device_fixture], taskXXX_config_variant2):
        """Full workflow with [variant2] should complete."""
        # Similar test for variant2
        assert result.success

    def test_integration_with_other_component(
        self, [device_fixture], taskXXX_config_variant1
    ):
        """[Feature] should integrate with [other component]."""
        comp1 = Component1(device=[device_fixture], config=taskXXX_config_variant1)
        comp2 = Component2(device=[device_fixture], config=taskXXX_config_variant1)

        # Test interaction
        result = comp1.interact_with(comp2)

        # Should work correctly
        assert result.compatible
        assert result.output.shape == expected_shape

    def test_with_checkpointing(
        self, [device_fixture], taskXXX_config_variant1, tmp_path
    ):
        """Workflow with save/load checkpoints should work."""
        component = Component1(
            device=[device_fixture],
            config=taskXXX_config_variant1,
        )

        # Run for N iterations
        for i in range(10):
            component.step()

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(component, episode=10, path=checkpoint_path)

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, component)
        assert checkpoint.episode == 10
        assert checkpoint.metadata["thing_count"] == component.count
```

### 5.2: Fix Any Integration Issues (GREEN)

Debug and fix any issues revealed by integration tests.

### 5.3: Add Performance Benchmarks (REFACTOR)

```python
def test_performance_comparison_variants(benchmark):
    """Benchmark performance with different variants."""
    def run_workflow(variant_config):
        component = Component(config=variant_config)
        return component.run_workflow()

    # Benchmark each variant
    for variant in [variant1, variant2, variant3]:
        result = benchmark(run_workflow, variant)
        print(f"{variant}: {result.mean:.4f}s per workflow")
```

### Phase 5 Success Criteria

- [ ] Full workflow completes for all variants
- [ ] Integration with other components works
- [ ] Checkpoint save/load works in integration
- [ ] No crashes or errors in end-to-end testing
- [ ] Performance acceptable for all variants

**Estimated Time**: X hours

---

## Configuration Examples

### Example 1: [Variant Name] (Minimal)

**Use Case**: [When to use this variant]

**Files**: `configs/[variant_name]/`

```yaml
# config.yaml
version: "2.0"
description: "[Description]"

things:
  - name: "thing1"
    property1: value1
  - name: "thing2"
    property2: value2
```

**Pedagogical Value** (if applicable):

- [What students learn from this variant]
- [What concepts it demonstrates]

### Example 2: [Variant Name] (Complex)

**Use Case**: [When to use this variant]

**Files**: `configs/[variant_name]/`

[More complex configuration example]

---

## Success Criteria (Overall)

### Config Layer

- [ ] Config accepts [range] of [things]
- [ ] Validation checks [constraints]
- [ ] Properties (count, names, mapping) work
- [ ] Example configs validate
- [ ] Existing configs still work (backward compatible)

### Engine Layer

- [ ] Component creates data structures sized by config
- [ ] All operations use dynamic sizes
- [ ] Name-based access works
- [ ] No remaining hardcoded values

### Network Layer

- [ ] Networks receive correct dimensions
- [ ] Forward pass works for all variants
- [ ] Training works with variable configurations

### Checkpoint Layer

- [ ] Checkpoints include metadata
- [ ] Loading validates compatibility
- [ ] Loading fails clearly if mismatch
- [ ] Legacy checkpoints load with warning

### Integration

- [ ] Full workflow completes for all variants
- [ ] Checkpoint save/load works
- [ ] All components interact correctly
- [ ] No crashes or errors

---

## Risks & Mitigations

### Risk 1: Breaking Existing Functionality

**Likelihood**: [High/Medium/Low]
**Impact**: [High/Medium/Low]
**Mitigation**:

- Comprehensive audit (Phase 0.5) finds all dependencies
- TDD ensures existing tests still pass
- Backward compatibility for legacy configs

### Risk 2: Missing Hardcoded Assumptions

**Likelihood**: [High/Medium/Low]
**Impact**: [High/Medium/Low]
**Mitigation**:

- Phase 0.5 comprehensive audit with multiple search patterns
- Boundary tests catch edge cases
- Integration tests catch cross-component issues

### Risk 3: Performance Regression

**Likelihood**: [High/Medium/Low]
**Impact**: [High/Medium/Low]
**Mitigation**:

- Performance benchmarks in Phase 5
- Dynamic sizing happens at initialization, not per-step
- Caching of computed properties

### Risk 4: [Project-Specific Risk]

**Likelihood**: [High/Medium/Low]
**Impact**: [High/Medium/Low]
**Mitigation**: [Specific mitigation strategy]

---

## Estimated Effort

| Phase | Description | TDD Time | Implementation Time | Total |
|-------|-------------|----------|---------------------|-------|
| **Phase 0** | Setup test fixtures | 0h | Xh | Xh |
| **Phase 0.5** | Comprehensive audit | 0h | Xh | Xh |
| **Phase 1** | [Layer 1] refactor | Xh | Xh | Xh |
| **Phase 2** | [Layer 2] refactor | Xh | Xh | Xh |
| **Phase 3** | [Layer 3] updates | Xh | Xh | Xh |
| **Phase 4** | [Layer 4] compatibility | Xh | Xh | Xh |
| **Phase 5** | Integration testing | Xh | Xh | Xh |
| **Total** | | **Xh** (TDD) | **Xh** (Impl) | **X-Yh** |

**TDD Overhead**: ~40% of time spent writing tests first (RED phase)
**Value**: Prevents regressions, documents expected behavior, enables confident refactoring
**Note**: Phase 0 is setup/infrastructure, not TDD (no tests written)

---

## Follow-Up Work (Post-Implementation)

1. **Create Additional [Variants]**:
   - [Variant 1]
   - [Variant 2]
   - [Variant 3]

2. **Update Documentation**:
   - [Doc 1]: Remove "[old constraint]" language
   - [Doc 2]: Document [new capability]
   - Add examples to README

3. **Performance Benchmarking**:
   - Measure performance for different variants
   - Document tradeoffs

4. **Pedagogical Materials** (if applicable):
   - Write lesson on "[concept]"
   - Show how [parameter] affects behavior

---

## Running TASK-XXX Tests

This section provides pytest commands for running TASK-XXX tests at various granularities.

### Run All TASK-XXX Tests

```bash
# Run all [feature] tests (unit + integration)
pytest tests/test_[project]/unit/[component]/test_[feature].py \
       tests/test_[project]/integration/test_[feature].py \
       -v
```

### Run Tests By Phase

**Phase 1: [Layer 1]**

```bash
pytest tests/test_[project]/unit/[component]/test_[feature].py::Test[Phase1] -v
```

**Phase 2: [Layer 2]**

```bash
pytest tests/test_[project]/unit/[component]/test_[feature].py::Test[Phase2] -v
```

**Phase 5: Integration**

```bash
pytest tests/test_[project]/integration/test_[feature]_integration.py -v
```

### Run with Coverage

```bash
pytest tests/test_[project]/unit/[component]/test_[feature].py \
       --cov=[project].[module1] \
       --cov=[project].[module2] \
       --cov-report=term-missing \
       -v
```

### Quick Smoke Test

Run one test from each phase to verify basic functionality:

```bash
pytest \
  tests/test_[project]/unit/.../::Test[Phase1]::test_nominal_case \
  tests/test_[project]/unit/.../::Test[Phase2]::test_variant1 \
  tests/test_[project]/integration/...::test_full_workflow \
  -v
```

### Expected Test Counts

- **Phase 0 (Fixtures)**: ~0 tests (setup only, verified with --collect-only)
- **Phase 1 ([Layer])**: ~[N] tests
- **Phase 2 ([Layer])**: ~[N] tests
- **Phase 3 ([Layer])**: ~[N] tests
- **Phase 4 ([Layer])**: ~[N] tests
- **Phase 5 (Integration)**: ~[N] tests
- **Total**: ~[N] tests for TASK-XXX

---

## Conclusion

This TDD-ready plan provides a systematic approach to [implementing feature]. Each phase follows strict RED-GREEN-REFACTOR:

1. **Write failing test first** (documents expected behavior)
2. **Write minimal code to pass** (simplest solution)
3. **Refactor while keeping tests green** (clean code)

**Total effort**: X-Y hours ([N-M] days) - revised after review
**Risk**: [Low|Medium|High] (reduced by comprehensive audit and TDD)
**Priority**: [CRITICAL|HIGH|MEDIUM|LOW] ([description of leverage])
**Impact**: [Description of what this unblocks]

**Next Step**: Begin Phase 0.5 - Comprehensive Audit (Xh), then Phase 1 - [Layer 1] Refactor (Xh)

**Slogan**: "[Memorable phrase capturing the transformation]"
