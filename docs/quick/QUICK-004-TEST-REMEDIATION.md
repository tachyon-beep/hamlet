# QUICK-004: Test Suite Structural Remediation

**Status**: 🟡 IN PROGRESS
**Created**: 2025-11-07
**Priority**: CRITICAL
**Effort**: 3-4 sprints
**Impact**: Foundation for maintainable test suite

---

## Problem Statement

The HAMLET test suite has **113+ magic number instances** and **600+ lines of duplicated Pydantic boilerplate** across 103 test files. This creates:

- **Brittle tests**: Changing meter count (8→9) breaks ~13 tests
- **Maintenance burden**: Schema changes require updating 20+ test files
- **Cognitive load**: Developers must memorize exact Pydantic field requirements
- **Zero coverage**: 8 critical modules (including main training loop) untested

**Example of the problem**:
```python
# Repeated 13+ times across test files
meters = (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)  # What are these?

# Repeated 6 times in test_affordance_config.py alone
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

---

## Solution: Four-Phase Remediation

### Phase 1: Structural Fixes (Sprints 12-14) 🔴 CRITICAL

**Goal**: Centralize magic numbers and eliminate boilerplate duplication.

#### Sprint 12: Create Test Builders Infrastructure
- [ ] Create `tests/test_townlet/builders.py` with:
  - `TestDimensions` dataclass (canonical dimensions)
  - `make_test_meters()` - Eliminate 8-meter magic tuple
  - `make_test_bars_config()` - Eliminate 20 BarsConfig duplicates
  - `make_test_bar()` - Eliminate 36 BarConfig duplicates
  - `make_test_affordance()` - Eliminate 31 AffordanceConfig duplicates
  - `make_test_terminal_condition()`
  - `make_test_episode_metadata()`
  - `make_test_recorded_step()`

- [ ] Add tempfile fixtures to `conftest.py`:
  - `temp_test_dir` - Eliminate 113 tempfile patterns
  - `temp_yaml_file` - Common YAML test file path

- [ ] Refactor 3 highest-duplication test files:
  - `test_affordance_config.py` (6 BarsConfig duplicates)
  - `test_recorder.py` (13 RecordedStep duplicates)
  - `test_video_export.py` or similar (5+ duplicates)

**Success Criteria**:
- `builders.py` provides single source of truth for test data
- Magic number instances: 113+ → ~30 (builders only)
- Boilerplate Pydantic lines: 600+ → ~300 (50% reduction)
- 3 test files refactored, all tests passing

#### Sprint 13: Refactor Recording Tests
- [ ] Refactor `test_video_export.py` to use builders
- [ ] Refactor remaining recording tests
- [ ] Update integration tests using hardcoded metadata

**Success Criteria**:
- All recording tests use builders
- Magic number instances: ~30 → ~20
- Boilerplate lines: ~300 → ~200

#### Sprint 14: Refactor Environment/Substrate Tests
- [ ] Refactor substrate tests to use `TestDimensions`
- [ ] Refactor environment tests to use builders
- [ ] Create `TEST_WRITING_GUIDE.md` documenting builder usage

**Success Criteria**:
- Magic number instances: ~20 (builders + constants only)
- Boilerplate lines: <100 (builders only)
- Test writing guide published

---

### Phase 2: Critical Coverage Gaps (Sprints 15-17) 🔴 CRITICAL

**Goal**: Test zero-coverage critical modules.

#### Sprint 15: Demo Runner (Main Training Loop)
- [ ] Test `demo/runner.py` (351 LOC, 0% → 70%)
  - DemoRunner initialization
  - Checkpoint loading/saving
  - Episode execution loop
  - TensorBoard integration
  - Database insertion

**Priority**: CRITICAL - This is the main training entry point!

#### Sprint 16: Recording Criteria
- [ ] Test `recording/criteria.py` (103 LOC, 0% → 70%)
  - Periodic criterion
  - Survival threshold criterion
  - Stage transition criterion
  - Custom criteria composition

**Priority**: HIGH - Decides what episodes to record

#### Sprint 17: Database Persistence
- [ ] Test `demo/database.py` (89 LOC, 0% → 70%)
  - Schema creation
  - Training history insertion
  - Recording metadata insertion
  - Query methods

**Priority**: HIGH - All data persistence

---

### Phase 3: Core Module Coverage (Sprints 18-21) 🟡 MODERATE

**Goal**: Improve low-coverage core modules.

#### Sprint 18: Vectorized Environment
- [ ] Test `vectorized_env.py` (6% → 50%)
  - Extract unit-testable logic
  - Mock substrate/affordance dependencies
  - Test reset, step, observation encoding

#### Sprint 19: Vectorized Population
- [ ] Test `population/vectorized.py` (9% → 50%)
  - Mock network dependencies
  - Test batch learning logic
  - Test checkpoint management

#### Sprint 20: Affordance Engine
- [ ] Test `affordance_engine.py` (9% → 60%)
  - Interaction validation
  - Effect application
  - Multi-tick interaction logic

#### Sprint 21: Observation Builder
- [ ] Test `observation_builder.py` (13% → 70%)
  - Position encoding (relative/scaled/absolute)
  - Affordance encoding
  - Temporal feature encoding

---

### Phase 4: Quality & Architecture (Sprints 22-24) 🟢 LOW PRIORITY

**Goal**: Improve test quality and maintainability.

#### Sprint 22: Remaining Zero-Coverage Modules
- [ ] `recording/replay.py` (0% → 70%)
- [ ] `recording/video_renderer.py` (0% → 70%)
- [ ] `action_labels.py` (0% → 70%)

#### Sprint 23: Test Documentation
- [ ] Create `TEST_WRITING_GUIDE.md`
- [ ] Document builder usage patterns
- [ ] Document fixture usage
- [ ] Add property-based testing examples

#### Sprint 24: Hypothesis Property Testing
- [ ] Add property tests for substrates
- [ ] Add property tests for observation encoding
- [ ] Add property tests for action space composition

---

## Metrics Tracking

### Before Remediation (Sprint 11 Baseline)
```
Magic Numbers:          113+
Boilerplate Lines:      600+
Zero-Coverage Modules:  8
Overall Coverage:       14%
Test Files:             103
Total Tests:            1558
```

### After Phase 1 (Sprint 14)
```
Magic Numbers:          <20 (builders + constants)
Boilerplate Lines:      <100 (builders only)
Zero-Coverage Modules:  8 (no change yet)
Overall Coverage:       14% (no change yet)
Refactored Test Files:  ~15 (high-duplication files)
```

### After Phase 2 (Sprint 17)
```
Zero-Coverage Modules:  5 (runner, criteria, database tested)
Overall Coverage:       14% → 25%
Critical Module Cov:    70%+ each
```

### After Phase 3 (Sprint 21)
```
Overall Coverage:       25% → 35%+
Core Module Coverage:   50-70% each
```

### After Phase 4 (Sprint 24)
```
Zero-Coverage Modules:  0 ✅
Overall Coverage:       35-40%
All Critical Modules:   70%+ ✅
```

---

## Success Criteria (Phase 1 - Sprint 12)

**Must Have**:
- [x] `builders.py` created with all core builders
- [x] `TestDimensions` dataclass with canonical values
- [x] Tempfile fixtures added to conftest.py
- [x] 3 test files refactored to use builders
- [x] All existing tests still pass
- [x] Ruff compliance

**Nice to Have**:
- [ ] 4-5 test files refactored (overachieve)
- [ ] Property-based testing examples
- [ ] Integration test builder support

**Metrics**:
- Magic numbers: 113+ → ~30 (70% reduction)
- Boilerplate: 600+ → ~300 (50% reduction)
- 0 test regressions

---

## Design: Test Builders Module

### File: `tests/test_townlet/builders.py`

```python
"""Centralized test data builders and factories.

Provides single source of truth for test entity construction.
Eliminates magic numbers and boilerplate Pydantic instantiation.

Usage:
    from tests.test_townlet.builders import (
        TestDimensions,
        make_test_meters,
        make_test_bars_config,
    )

    # Use canonical dimensions
    obs_dim = TestDimensions.GRID2D_OBS_DIM  # 29

    # Use standard test meters
    meters = make_test_meters()  # (1.0, 0.9, 0.8, ...)

    # Create minimal config
    bars = make_test_bars_config(num_meters=8)
"""

from dataclasses import dataclass
from typing import Literal

from townlet.environment.affordance_config import (
    AffordanceConfig,
    AffordanceCost,
    AffordanceEffect,
)
from townlet.environment.cascade_config import (
    BarConfig,
    BarsConfig,
    TerminalCondition,
)
from townlet.recording.data_structures import (
    EpisodeMetadata,
    RecordedStep,
)


@dataclass
class TestDimensions:
    """Canonical dimension calculations for all substrates.

    These are SINGLE SOURCE OF TRUTH for test dimension expectations.
    Any changes to substrate dimensions should update these values.
    """

    # Standard test grid
    GRID_SIZE: int = 8
    NUM_METERS: int = 8
    NUM_AFFORDANCES: int = 14  # Vocabulary size

    # Grid2D substrate (relative encoding)
    GRID2D_POSITION_DIM: int = 2  # x, y normalized
    GRID2D_METER_DIM: int = 8
    GRID2D_AFFORDANCE_DIM: int = 15  # 14 affordances + 1 "none"
    GRID2D_TEMPORAL_DIM: int = 4  # sin, cos, progress, lifetime
    GRID2D_OBS_DIM: int = 29  # 2 + 8 + 15 + 4
    GRID2D_ACTION_DIM: int = 8  # 6 substrate + 2 custom

    # POMDP (5x5 window)
    POMDP_VISION_RANGE: int = 2
    POMDP_WINDOW_SIZE: int = 5  # 2 * 2 + 1
    POMDP_WINDOW_CELLS: int = 25  # 5 * 5
    POMDP_OBS_DIM: int = 54  # 25 + 2 + 8 + 15 + 4

    # Grid3D substrate
    GRID3D_POSITION_DIM: int = 3  # x, y, z
    GRID3D_ACTION_DIM: int = 10  # 8 substrate + 2 custom

    # GridND (7D example)
    GRIDND_7D_POSITION_DIM: int = 7
    GRIDND_7D_ACTION_DIM: int = 16  # 14 substrate + 2 custom


def make_test_meters() -> tuple[float, ...]:
    """Create standard 8-meter test values.

    Returns normalized meter values for:
    (energy, health, satiation, money, mood, social, fitness, hygiene)

    Each value chosen to be distinct for debugging.
    """
    return (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)


def make_test_bar(
    name: str = "energy",
    index: int = 0,
    tier: Literal["pivotal", "primary", "secondary", "resource"] = "pivotal",
    initial: float = 1.0,
    base_depletion: float = 0.01,
    description: str | None = None,
) -> BarConfig:
    """Create minimal BarConfig for testing.

    Args:
        name: Meter name
        index: Meter index in tensor
        tier: Cascade tier
        initial: Initial normalized value [0, 1]
        base_depletion: Passive decay per step
        description: Human-readable description

    Returns:
        Valid BarConfig ready for testing
    """
    if description is None:
        description = f"{name.capitalize()} meter"

    return BarConfig(
        name=name,
        index=index,
        tier=tier,
        initial=initial,
        base_depletion=base_depletion,
        description=description,
    )


def make_test_terminal_condition(
    meter: str = "energy",
    operator: Literal["<=", ">=", "<", ">", "=="] = "<=",
    value: float = 0.0,
    description: str | None = None,
) -> TerminalCondition:
    """Create minimal TerminalCondition for testing."""
    if description is None:
        description = f"Death by {meter} {operator} {value}"

    return TerminalCondition(
        meter=meter,
        operator=operator,
        value=value,
        description=description,
    )


def make_test_bars_config(
    num_meters: int = 8,
    include_terminal: bool = True,
) -> BarsConfig:
    """Create minimal BarsConfig for testing.

    Args:
        num_meters: Number of meters to include (1-8)
        include_terminal: Include energy depletion terminal condition

    Returns:
        Valid BarsConfig with standard test meters
    """
    meter_names = ["energy", "health", "satiation", "money", "mood", "social", "fitness", "hygiene"]
    tiers = ["pivotal", "pivotal", "primary", "resource", "secondary", "secondary", "secondary", "secondary"]

    if num_meters > 8:
        raise ValueError(f"make_test_bars_config supports up to 8 meters, got {num_meters}")

    bars = [
        make_test_bar(
            name=meter_names[i],
            index=i,
            tier=tiers[i],
            initial=1.0,
            base_depletion=0.01 if i < 2 else 0.005,
        )
        for i in range(num_meters)
    ]

    terminal_conditions = []
    if include_terminal:
        terminal_conditions.append(
            make_test_terminal_condition(
                meter="energy",
                operator="<=",
                value=0.0,
            )
        )

    return BarsConfig(
        version="1.0",
        description="Test bars configuration",
        bars=bars,
        terminal_conditions=terminal_conditions,
    )


def make_test_affordance(
    id: str = "Bed",
    name: str | None = None,
    category: str = "energy_restoration",
    interaction_type: Literal["instant", "multi_tick", "continuous", "dual"] = "instant",
    required_ticks: int | None = None,
    effects: list[tuple[str, float]] | None = None,
    operating_hours: tuple[int, int] = (0, 24),
) -> AffordanceConfig:
    """Create minimal AffordanceConfig for testing.

    Args:
        id: Affordance ID
        name: Human-readable name (defaults to id)
        category: Affordance category
        interaction_type: Type of interaction
        required_ticks: Required ticks (for multi_tick/dual)
        effects: List of (meter, amount) tuples
        operating_hours: (open, close) tuple

    Returns:
        Valid AffordanceConfig ready for testing
    """
    if name is None:
        name = id

    # Auto-set required_ticks for multi_tick/dual
    if interaction_type in ["multi_tick", "dual"] and required_ticks is None:
        required_ticks = 5

    # Default effects
    effect_list = []
    if effects:
        effect_list = [
            AffordanceEffect(meter=meter, amount=amount)
            for meter, amount in effects
        ]

    return AffordanceConfig(
        id=id,
        name=name,
        category=category,
        interaction_type=interaction_type,
        required_ticks=required_ticks,
        effects=effect_list,
        operating_hours=list(operating_hours),
    )


def make_test_episode_metadata(
    episode_id: int = 100,
    survival_steps: int = 10,
    total_reward: float = 10.0,
    curriculum_stage: int = 1,
) -> EpisodeMetadata:
    """Create minimal EpisodeMetadata for testing."""
    return EpisodeMetadata(
        episode_id=episode_id,
        survival_steps=survival_steps,
        total_reward=total_reward,
        extrinsic_reward=total_reward,
        intrinsic_reward=0.0,
        curriculum_stage=curriculum_stage,
        epsilon=0.5,
        intrinsic_weight=0.0,
        timestamp=1234567890.0,
        affordance_layout={"Bed": (2, 3)},
        affordance_visits={"Bed": 1},
    )


def make_test_recorded_step(
    step: int = 0,
    position: tuple[int, ...] = (3, 5),
    meters: tuple[float, ...] | None = None,
    action: int = 2,
    reward: float = 1.0,
    done: bool = False,
) -> RecordedStep:
    """Create minimal RecordedStep for testing."""
    if meters is None:
        meters = make_test_meters()

    return RecordedStep(
        step=step,
        position=position,
        meters=meters,
        action=action,
        reward=reward,
        intrinsic_reward=0.1,
        done=done,
        q_values=None,
    )
```

---

## Rollout Plan

### Sprint 12 (Current)
1. Create `builders.py` module
2. Add tempfile fixtures to conftest.py
3. Refactor `test_affordance_config.py`
4. Refactor `test_recorder.py`
5. Refactor `test_video_export.py` or similar
6. Run full test suite (must pass)
7. Run ruff (must pass)
8. Commit and push

### Sprint 13
1. Refactor remaining recording tests
2. Refactor integration tests using metadata
3. Update any tests using hardcoded dimensions

### Sprint 14
1. Refactor substrate tests
2. Refactor environment tests
3. Create `TEST_WRITING_GUIDE.md`

---

## Risk Mitigation

**Risk**: Refactoring breaks existing tests
**Mitigation**: Run full test suite after each file refactor

**Risk**: Builders become too complex
**Mitigation**: Keep builders simple, focus on eliminating duplication only

**Risk**: Magic numbers still proliferate
**Mitigation**: Add ruff rule to detect common magic numbers (future)

---

## References

- Original assessment: `TEST_SUITE_ASSESSMENT.md`
- Test fixtures: `tests/test_townlet/conftest.py`
- Example high-duplication test: `tests/test_townlet/unit/environment/test_affordance_config.py`

---

**Status**: Ready for Sprint 12 execution
**Next Action**: Create builders.py and refactor 3 test files
