# HAMLET Test Suite: First Principles Assessment

**Date**: 2025-11-07
**Context**: Post-Sprint 11 comprehensive audit
**Total Test Files**: 103 (54 unit, 32 integration)
**Total Test Count**: 1558 tests

---

## Executive Summary

The HAMLET test suite has **strong foundational structure** but suffers from **fragmentation** and **lack of centralized scaffolding**. Recent sprint work (8-11) added high-quality coverage but **replicated existing structural problems** rather than fixing them.

### Key Metrics
- **Overall coverage**: ~14% (artificially low due to integration-heavy modules)
- **Unit test coverage**: Strong for substrate (67-97%), config (60-96%)
- **Zero-coverage modules**: 8 critical modules (action_labels, criteria, replay, video_renderer, demo suite)
- **Magic number occurrences**: 113+ instances of hardcoded values

---

## CRITICAL STRUCTURAL PROBLEMS

### 1. **Magic Numbers Epidemic** 🔴 CRITICAL

**Problem**: Hardcoded values scattered across 100+ test files create brittle tests that break when schemas evolve.

**Evidence**:
```python
# Appears 13+ times across test files
meters = (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)  # What do these mean?

# Grid size "8" appears 163 times
grid_size = 8  # Why 8? What if we change to 10?

# Observation dim "29" appears 21 times
obs_dim = 29  # How is this calculated?

# Action dim "8" appears 96 times
action_dim = 8  # Grid2D specific, but hardcoded everywhere
```

**Impact**:
- **Changing meter count** (8→9) would break ~13 tests
- **Changing grid size** (8→10) would break ~163 test assertions
- **Changing observation encoding** would break ~21 tests
- **No single source of truth** for these values

**Example from recent work** (test_recorder.py:56):
```python
meters = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85])
```
What are these meters? Energy, health, satiation, money, mood, social, fitness, hygiene. **Not documented anywhere in the test.**

---

### 2. **Repetitive Pydantic Boilerplate** 🔴 CRITICAL

**Problem**: Every test manually constructs BarsConfig, BarConfig, AffordanceConfig from scratch.

**Evidence**:
- **BarsConfig**: 20 instantiations (each 20-30 lines)
- **BarConfig**: 36 instantiations (each 7-10 lines)
- **AffordanceConfig**: 31 instantiations (each 10-15 lines)

**Example from test_affordance_config.py (repeated 6 times)**:
```python
bars_config = BarsConfig(
    version="1.0",
    description="Test bars",
    bars=[
        BarConfig(
            name="energy",
            index=0,
            tier="pivotal",
            initial=1.0,
            base_depletion=0.01,
            description="Energy meter",
        ),
    ],
    terminal_conditions=[
        TerminalCondition(
            meter="energy",
            operator="<=",
            value=0.0,
            description="Death by energy depletion",
        ),
    ],
)
```

**Impact**:
- **600+ lines of repetitive boilerplate** across test suite
- **Schema changes require updating 20+ test files**
- **No consistency** in test data values
- **Cognitive load**: Developers must remember exact required fields

---

### 3. **No Centralized Test Data Builders** 🔴 CRITICAL

**Problem**: No factory functions or builders for common test entities.

**Missing builders**:
- `make_test_bars_config()` - Would eliminate 20 boilerplate instantiations
- `make_test_bar()` - Would eliminate 36 boilerplate instantiations
- `make_test_affordance()` - Would eliminate 31 boilerplate instantiations
- `make_test_meters()` - Would centralize the 8-meter magic tuple
- `make_test_episode_metadata()` - Would centralize episode test data
- `make_test_recorded_step()` - Would eliminate 13+ identical instantiations

**What we should have**:
```python
# In tests/test_townlet/builders.py (DOES NOT EXIST)

def make_test_bars_config(
    num_meters: int = 8,
    include_terminal: bool = True
) -> BarsConfig:
    """Create a minimal valid BarsConfig for testing."""
    # Single source of truth for test configs

def make_test_meters() -> tuple[float, ...]:
    """Return standard 8-meter test values.

    Returns: (energy, health, satiation, money, mood, social, fitness, hygiene)
    """
    return (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)
```

---

### 4. **Tempfile Anti-Pattern** 🟡 MODERATE

**Problem**: Every test manually creates `tempfile.TemporaryDirectory()` context managers.

**Evidence**: 113 occurrences of identical tempfile patterns

**Current pattern** (repeated 113 times):
```python
with tempfile.TemporaryDirectory() as tmpdir:
    config_path = Path(tmpdir) / "affordances.yaml"
    # Test logic
```

**Better pattern** (not implemented):
```python
@pytest.fixture
def temp_test_dir() -> Path:
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# Usage
def test_something(temp_test_dir):
    config_path = temp_test_dir / "affordances.yaml"
```

---

### 5. **Observation/Action Dimension Confusion** 🟡 MODERATE

**Problem**: Tests hardcode observation and action dimensions without documenting **why** those values are correct.

**Evidence**:
```python
# From conftest.py
FULL_OBS_DIM_8X8 = 93  # What's the calculation?

# Actually (from CLAUDE.md):
# Grid2D: 2 coords + 8 meters + 15 affordances + 4 temporal = 29 dims
# But tests use 93? Why the discrepancy?

# Action space
action_dim = 8  # Grid2D specific, but not documented
```

**What we need**:
```python
class TestDimensions:
    """Centralized dimension calculations for test validation."""

    @staticmethod
    def obs_dim_grid2d(
        position_encoding: str = "relative",
        num_meters: int = 8,
        num_affordances: int = 14
    ) -> int:
        """Calculate observation dim for Grid2D substrate.

        Formula: 2 (coords) + 8 (meters) + 15 (affordances) + 4 (temporal) = 29
        """
        return 2 + num_meters + (num_affordances + 1) + 4

    @staticmethod
    def action_dim_grid2d(num_custom: int = 2) -> int:
        """Calculate action dim for Grid2D substrate.

        Formula: 6 (substrate) + 2 (custom) = 8
        """
        return 6 + num_custom
```

---

### 6. **Integration vs Unit Test Blurring** 🟡 MODERATE

**Problem**: Some "unit" tests are actually integration tests, and vice versa.

**Examples**:

**Unit test doing integration work**:
```python
# tests/test_townlet/unit/environment/test_vectorized_env.py
def test_full_environment_reset():
    env = VectorizedHamletEnv(...)  # Loads full config, substrate, affordances
    env.reset()  # Tests entire stack
```
This should be in `integration/` not `unit/`.

**Missing unit tests for integration-heavy code**:
```python
# src/townlet/environment/vectorized_env.py: 6% coverage
# src/townlet/population/vectorized.py: 9% coverage
```
Core logic **is** unit-testable with proper mocking, but tests are only integration.

---

### 7. **Zero Coverage Modules** 🔴 CRITICAL

**Completely untested modules**:

| Module | LOC | Coverage | Risk Level |
|--------|-----|----------|------------|
| `recording/criteria.py` | 103 | **0%** | HIGH - Recording decisions |
| `recording/replay.py` | 82 | **0%** | HIGH - Episode replay |
| `recording/video_renderer.py` | 137 | **0%** | MEDIUM - Visualization |
| `action_labels.py` | 68 | **0%** | LOW - UI labels |
| `demo/database.py` | 89 | **0%** | HIGH - Data persistence |
| `demo/runner.py` | 351 | **0%** | CRITICAL - Main training loop |
| `demo/live_inference.py` | 487 | **0%** | MEDIUM - Inference server |

**Impact**:
- `demo/runner.py` is the **main training entry point** - zero coverage!
- `recording/criteria.py` decides **what to record** - zero coverage!
- `demo/database.py` handles **all persistence** - zero coverage!

---

## STRUCTURAL GAPS

### 8. **No Test Utilities Module** 🟡 MODERATE

**Missing**: `tests/test_townlet/utils.py` or `builders.py` for shared test helpers.

**What we need**:
```python
# tests/test_townlet/builders.py (CREATE THIS)

"""Centralized test data builders and factories.

Provides single source of truth for test entity construction.
Eliminates magic numbers and boilerplate Pydantic instantiation.
"""

from dataclasses import dataclass
from typing import Literal

@dataclass
class TestDimensions:
    """Canonical dimension calculations for all substrates."""

    # Standard test grid
    GRID_SIZE: int = 8
    NUM_METERS: int = 8
    NUM_AFFORDANCES: int = 14

    # Grid2D
    GRID2D_OBS_DIM: int = 29  # 2 + 8 + 15 + 4
    GRID2D_ACTION_DIM: int = 8  # 6 substrate + 2 custom

    # POMDP
    POMDP_VISION_RANGE: int = 2
    POMDP_WINDOW_SIZE: int = 5  # 2 * vision_range + 1
    POMDP_OBS_DIM: int = 54  # 25 + 2 + 8 + 15 + 4

def make_test_meters() -> tuple[float, ...]:
    """Standard 8-meter test values.

    Returns: (energy, health, satiation, money, mood, social, fitness, hygiene)
    """
    return (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)

def make_test_bars_config(
    num_meters: int = 8,
    include_terminal: bool = True
) -> BarsConfig:
    """Build minimal BarsConfig for testing."""
    # Implementation

def make_test_affordance(
    id: str = "Bed",
    interaction_type: Literal["instant", "multi_tick"] = "instant",
    **kwargs
) -> AffordanceConfig:
    """Build minimal AffordanceConfig for testing."""
    # Implementation
```

---

### 9. **No Test Documentation Standards** 🟡 MODERATE

**Problem**: Test docstrings inconsistent, often missing **why** the test exists.

**Current state**:
```python
# ❌ BAD: What does this test?
def test_export_success():
    ...

# ⚠️ OKAY: Describes what, but not why
def test_export_success_with_explicit_grid_size():
    """Should successfully export episode with explicit grid size."""
    ...

# ✅ GOOD: Describes what, why, and regression context
def test_export_success_with_explicit_grid_size():
    """Should successfully export episode with explicit grid size.

    Covers: export_episode_video() happy path
    Regression: Previously failed when grid_size was explicitly passed
    """
    ...
```

---

## POSITIVE FINDINGS ✅

### What's Working Well

1. **Test Organization** ✅
   - Clear separation: `unit/`, `integration/`, `validation/`
   - Logical module grouping matches source structure
   - Naming conventions consistent (`test_*.py`)

2. **Substrate Test Coverage** ✅
   - Grid2D: 67% → 87% (Sprint 1-7)
   - Grid3D: 80% → 95% (Sprint 4)
   - GridND: 88% → 97% (Sprint 7)
   - Excellent comprehensive test suites

3. **Recent Sprint Quality** ✅
   - Sprints 8-11 produced high-quality tests
   - Good use of AAA pattern
   - Comprehensive edge case coverage
   - Clear test class organization

4. **Config System Tests** ✅
   - affordance_config: 96% coverage (Sprint 11)
   - cascade_config: 60% coverage
   - Good Pydantic validator coverage

5. **Fixtures Available** ✅
   - `conftest.py` provides:
     - Config path fixtures
     - Device fixtures (CPU/CUDA)
     - Environment fixtures (basic, POMDP, temporal)
     - Good session-scoped optimization

---

## RECOMMENDED ACTION PLAN

### Phase 1: Structural Fixes (High Priority)

#### 1.1 Create Test Builders Module (CRITICAL)
```bash
# Create tests/test_townlet/builders.py
- TestDimensions dataclass (canonical dimensions)
- make_test_meters() (eliminate magic tuple)
- make_test_bars_config() (eliminate 20 boilerplate instantiations)
- make_test_bar() (eliminate 36 boilerplate instantiations)
- make_test_affordance() (eliminate 31 boilerplate instantiations)
- make_test_episode_metadata()
- make_test_recorded_step()
```

**Impact**: Eliminates 600+ lines of boilerplate, centralizes magic numbers

#### 1.2 Refactor Existing Tests to Use Builders (CRITICAL)
```bash
# Start with high-duplication tests
- test_affordance_config.py (6 BarsConfig duplicates)
- test_recorder.py (13 RecordedStep duplicates)
- test_video_export.py (metadata duplicates)
```

**Impact**: Immediate reduction in brittleness, faster test writing

#### 1.3 Add Tempfile Fixtures (MODERATE)
```python
# In conftest.py
@pytest.fixture
def temp_test_dir() -> Path:
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def temp_yaml_file(temp_test_dir) -> Path:
    """Temporary YAML file path."""
    return temp_test_dir / "test.yaml"
```

**Impact**: Eliminates 113 repetitive tempfile patterns

---

### Phase 2: Coverage Gaps (High Priority)

#### 2.1 Zero-Coverage Critical Modules
**Priority order**:
1. **demo/runner.py** (351 LOC) - Main training entry point
2. **demo/database.py** (89 LOC) - Data persistence
3. **recording/criteria.py** (103 LOC) - Recording decisions
4. **recording/replay.py** (82 LOC) - Episode replay

**Target**: 70%+ coverage on each

#### 2.2 Low-Coverage Core Modules
**Priority order**:
1. **vectorized_env.py** (6% → 50%) - Core environment logic
2. **vectorized.py** (9% → 50%) - Population training
3. **affordance_engine.py** (9% → 60%) - Interaction logic
4. **observation_builder.py** (13% → 70%) - Observation encoding

**Strategy**: Extract unit-testable logic, add mocked unit tests

---

### Phase 3: Test Quality (Medium Priority)

#### 3.1 Documentation Standards
Create `tests/TEST_WRITING_GUIDE.md`:
```markdown
# Test Writing Standards

## Docstring Format
def test_something():
    """Should [expected behavior] when [condition].

    Covers: [module.function]
    Regression: [if fixing a bug]
    Related: [related test names]
    """
```

#### 3.2 Integration vs Unit Separation
- Audit `unit/environment/test_vectorized_env.py`
- Move integration-style tests to `integration/`
- Add true unit tests with mocking

---

### Phase 4: Architectural Improvements (Low Priority)

#### 4.1 Hypothesis Property-Based Testing
```python
# For substrate tests (already has properties/)
from hypothesis import given, strategies as st

@given(grid_size=st.integers(min_value=3, max_value=100))
def test_grid2d_position_bounds(grid_size):
    """Position clamping works for any grid size."""
    substrate = Grid2DSubstrate(...)
    # Property: all positions must be in [0, grid_size)
```

#### 4.2 Test Parametrization
```python
# Reduce duplication via parametrization
@pytest.mark.parametrize("substrate_type,expected_action_dim", [
    ("grid2d", 8),
    ("grid3d", 10),
    ("gridnd", 16),
])
def test_action_space_dimensions(substrate_type, expected_action_dim):
    ...
```

---

## METRICS TO TRACK

### Before Refactoring
- Magic number instances: **113+**
- Boilerplate Pydantic lines: **600+**
- Zero-coverage modules: **8**
- Average test file length: **~200 lines**

### After Phase 1 (Structural Fixes)
- Magic number instances: **<20** (builders only)
- Boilerplate Pydantic lines: **<100** (builders only)
- Zero-coverage modules: **8** (no change yet)
- Average test file length: **~120 lines** (40% reduction)

### After Phase 2 (Coverage Gaps)
- Zero-coverage modules: **0** ✅
- Overall coverage: **14% → 35%+**
- Critical module coverage: **70%+** each

---

## CONCRETE NEXT STEPS

### Immediate (Sprint 12)
1. **Create `tests/test_townlet/builders.py`** with:
   - `TestDimensions` dataclass
   - `make_test_meters()`
   - `make_test_bars_config()`
   - `make_test_bar()`

2. **Refactor 3 highest-duplication test files**:
   - `test_affordance_config.py` (6 BarsConfig duplicates)
   - `test_recorder.py` (13 RecordedStep duplicates)
   - Pick one more with 5+ duplicates

3. **Add tempfile fixture** to conftest.py

### Short-term (Sprints 13-14)
4. **Test `demo/runner.py`** (0% → 70%)
5. **Test `recording/criteria.py`** (0% → 70%)
6. **Refactor 5 more high-duplication files**

### Medium-term (Sprints 15-17)
7. **Test `demo/database.py`** (0% → 70%)
8. **Test `vectorized_env.py`** (6% → 50%)
9. **Create TEST_WRITING_GUIDE.md**

---

## APPENDIX: Test Anti-Patterns Observed

### A1. Magic Number Anti-Pattern
```python
# ❌ BAD
meters = (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)

# ✅ GOOD
from tests.test_townlet.builders import make_test_meters
meters = make_test_meters()
```

### A2. Boilerplate Pydantic Anti-Pattern
```python
# ❌ BAD (20+ lines repeated)
bars_config = BarsConfig(
    version="1.0",
    description="Test bars",
    bars=[
        BarConfig(
            name="energy",
            index=0,
            tier="pivotal",
            initial=1.0,
            base_depletion=0.01,
            description="Energy meter",
        ),
    ],
    terminal_conditions=[...],
)

# ✅ GOOD (1 line)
from tests.test_townlet.builders import make_test_bars_config
bars_config = make_test_bars_config()
```

### A3. Tempfile Anti-Pattern
```python
# ❌ BAD (repeated 113 times)
with tempfile.TemporaryDirectory() as tmpdir:
    config_path = Path(tmpdir) / "test.yaml"
    # ...

# ✅ GOOD (fixture)
def test_something(temp_test_dir):
    config_path = temp_test_dir / "test.yaml"
    # ...
```

---

## CONCLUSION

The HAMLET test suite has **solid foundations** but needs **structural refactoring** before continuing coverage expansion. The primary issues are:

1. **Magic numbers everywhere** (113+ instances)
2. **Boilerplate duplication** (600+ lines)
3. **Zero coverage on critical modules** (8 modules, including main training loop)

**Recommendation**: **Pause coverage expansion** (Sprints 12-14) to:
1. Create centralized test builders (`builders.py`)
2. Refactor existing tests to use builders
3. Add tempfile fixtures

**Then resume** with:
4. Zero-coverage critical modules (`runner.py`, `database.py`, `criteria.py`)
5. Low-coverage core modules (`vectorized_env.py`, etc.)

This approach will make the test suite **maintainable** rather than just **comprehensive**.

---

**End of Assessment**
