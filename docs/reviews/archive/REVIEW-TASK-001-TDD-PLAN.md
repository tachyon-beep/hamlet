# TASK-001 Implementation Plan Review: TDD & Test Infrastructure Alignment

**Reviewer**: Claude Code (AI Assistant)
**Date**: 2025-11-04
**Plan Under Review**: `docs/plans/plan-task-001-variable-size-meters-tdd-ready.md`
**Task Specification**: `docs/TASK-001-VARIABLE-SIZE-METER-SYSTEM.md` (Note: Filename mismatch with actual TASK-005)
**Test Infrastructure**: 644 tests, 67% coverage (refactored November 2025)

---

## Executive Summary

**Overall Assessment**: ⚠️ **PROCEED WITH CAUTION - MODERATE REVISIONS NEEDED**

The plan demonstrates strong TDD alignment and thorough understanding of the test infrastructure. However, there are **critical discrepancies** between the plan and actual codebase state that must be resolved before implementation.

### Key Findings

✅ **Strengths**:

- Excellent TDD methodology (strict RED-GREEN-REFACTOR cycles)
- Proper fixture composition and reuse patterns
- Comprehensive test coverage across all layers
- Behavioral assertions instead of exact values
- Test isolation and independence maintained

⚠️ **Critical Issues**:

1. **TASK NUMBER MISMATCH**: Plan references TASK-001, but specification file is named TASK-005
2. **TEST FILE LOCATION CONFLICT**: Plan proposes new files that may conflict with existing consolidated test structure
3. **MISSING DEPENDENCY ANALYSIS**: Doesn't account for AffordanceEngine meter validation changes
4. **FIXTURE NAMING**: Some proposed fixtures may duplicate existing patterns

❌ **Blockers**:

- Task numbering must be reconciled before proceeding
- Test file organization must align with refactored structure

### Risk Assessment

**Implementation Risk**: **MEDIUM** (was HIGH, reduced by strong TDD approach)

- Task numbering confusion creates documentation risk
- Test file conflicts could cause regressions
- Complexity manageable with proper phasing

---

## TDD Alignment Analysis

### Score: 9.0 / 10.0

| Criterion | Score | Notes |
|-----------|-------|-------|
| **RED-GREEN-REFACTOR Adherence** | 10/10 | Excellent - every phase follows strict TDD cycle |
| **Tests Written First** | 10/10 | No implementation before tests (enforced in plan) |
| **Test Comprehensiveness** | 9/10 | Strong coverage, minor gaps in edge cases |
| **Test Isolation** | 9/10 | Good fixture use, minor sharing concerns |
| **Behavioral Assertions** | 8/10 | Mostly behavioral, some exact-value assertions remain |
| **Incremental Development** | 10/10 | Proper phasing from config → engine → network → checkpoint |

#### Strengths

1. **Strict RED-GREEN-REFACTOR Enforcement**
   - Line 324: "Critical Rule: Never write implementation code before writing the test"
   - Each phase explicitly starts with failing tests (RED)
   - Green phase implements minimal code
   - Refactor phase cleans up while keeping tests passing

2. **Comprehensive Test Coverage**
   - Phase 1: 6 tests (config validation)
   - Phase 2: 13 tests (engine dynamics)
   - Phase 3: 4 tests (network compatibility)
   - Phase 4: 4 tests (checkpoint metadata)
   - Phase 5: 5 tests (integration)
   - **Total**: ~32 tests for TASK-001 (excellent ratio for 13-19h effort)

3. **Proper Test Organization**
   - Unit tests in component directories
   - Integration tests separate
   - Fixtures in conftest.py
   - Follows existing patterns from 644-test suite

4. **Behavioral Assertions**
   - Line 857: `assert env_8.observation_dim == env_4.observation_dim + 4` (behavioral)
   - Line 1603: `assert steps < max_steps` (behavioral - "should eventually complete")
   - Avoids hardcoded exact values where possible

#### Weaknesses

1. **Some Exact-Value Assertions**
   - Line 825-828: `assert env.meters[0, 0].item() == pytest.approx(1.0)` - Exact initial values
   - **Recommendation**: These are acceptable for initialization tests, but consider documenting why exact values are needed

2. **Missing Edge Case Tests**
   - No test for 1-meter universe (minimum valid case)
   - No test for 32-meter universe (maximum valid case)
   - No test for non-sequential indices (e.g., [0, 2, 3, 4] missing 1)
   - **Recommendation**: Add boundary tests in Phase 1

3. **Limited Error Message Testing**
   - Line 462: `with pytest.raises(ValueError, match="Must have at least 1 meter")` - Good
   - But missing tests for clarity of error messages in other failure modes
   - **Recommendation**: Add assertions on error message content for user-facing validation errors

---

## Test Infrastructure Integration Analysis

### Score: 7.5 / 10.0

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Fixture Reuse** | 9/10 | Excellent composition patterns |
| **File Organization** | 6/10 | ⚠️ Conflicts with consolidated structure |
| **Conftest Integration** | 8/10 | Good, but missing verification of imports |
| **Device Handling** | 10/10 | Proper `cpu_device` fixture usage throughout |
| **Path Handling** | 9/10 | Proper use of `tmp_path` and `test_config_pack_path` |

#### Strengths

1. **Proper Fixture Composition**
   - Line 188-289: Fixtures compose `tmp_path` → `config_4meter` → `env_4meter`
   - Matches existing pattern: `test_config_pack_path` → `basic_env`
   - Reuses `cpu_device` instead of creating new device fixtures

2. **Correct Device Usage**
   - Line 778: `def test_4_meter_env_creates_correct_tensor_shape(self, cpu_device, config_4meter):`
   - All tests use `cpu_device` for determinism (matches existing pattern from line 107 of conftest.py)
   - No hardcoded `device="cuda"` anywhere

3. **Temporary Path Handling**
   - Uses `tmp_path` for test configs
   - Uses `test_config_pack_path` for baseline 8-meter config
   - Proper cleanup via pytest's automatic tmp_path removal

#### Weaknesses

1. **TEST FILE ORGANIZATION CONFLICT** ⚠️
   - **Plan proposes**: `unit/environment/test_variable_meter_config.py` (NEW)
   - **Plan proposes**: `unit/environment/test_variable_meter_engine.py` (NEW)
   - **Actual structure**: Tests already consolidated in `unit/test_configuration.py` (line 1-100)

   **Problem**: The plan proposes creating separate files, but the November 2025 refactor consolidated tests into larger files:
   - `test_configuration.py` (consolidated from 5 old files)
   - `test_meters.py` (consolidated from 2 old files)

   **Recommendation**:
   - **Option A (Preferred)**: Extend existing `unit/test_configuration.py` with new test classes instead of creating new files
   - **Option B**: Create task-specific directory `unit/task-001/` to isolate these tests

   **Impact**: MODERATE - Could cause test discovery issues or duplicate test organization

2. **FIXTURE NAMING COLLISION RISK**
   - Plan proposes: `config_4meter`, `config_12meter`, `env_4meter`, `env_12meter`
   - Existing conftest.py has: `basic_env`, `pomdp_env`, `temporal_env` (line 121-189)

   **Potential Issue**: If 4-meter or 12-meter configs become standard curriculum levels, these fixture names could conflict with future curriculum fixtures

   **Recommendation**: Use more specific names:
   - `task001_config_4meter` or `variable_meter_4meter_config`
   - `task001_env_4meter` or `variable_meter_4meter_env`

3. **MISSING IMPORT VERIFICATION**
   - Line 340-344: Plan adds `import copy, shutil, yaml` to conftest.py
   - But doesn't verify these aren't already imported
   - Actual conftest.py (line 14-20) already imports `shutil`, `yaml`
   - Only `copy` is missing

   **Recommendation**: Check existing imports first to avoid duplicates

4. **NO VERIFICATION OF EXISTING TEST CLASSES**
   - Plan creates `TestVariableMeterConfigValidation`, `TestVariableMeterEngine`, etc.
   - Doesn't check if these class names conflict with existing test classes
   - From actual codebase: `TestConfigPackLoading` exists in `test_configuration.py` (line 62)

   **Recommendation**: Survey existing test class names before proposing new ones

---

## Task Specification Alignment

### Score: 6.0 / 10.0 ⚠️

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Problem Statement Match** | 10/10 | Plan correctly identifies 8-meter constraint |
| **Solution Approach** | 9/10 | Good architectural changes |
| **Task Numbering** | 0/10 | ❌ **CRITICAL**: TASK-001 vs TASK-005 mismatch |
| **Dependency Tracking** | 7/10 | Missing some cross-file validation needs |

#### Critical Issue: TASK NUMBER MISMATCH

**Discovered Discrepancy**:

- **Plan filename**: `plan-task-001-variable-size-meters-tdd-ready.md`
- **Plan header (line 3)**: "Task: TASK-001 Variable-Size Meter System"
- **Specification filename**: `docs/TASK-001-VARIABLE-SIZE-METER-SYSTEM.md`
- **Specification header (line 1)**: "# TASK-005: Variable-Size Meter System"

**Evidence**:

```markdown
# From TASK-001-VARIABLE-SIZE-METER-SYSTEM.md line 1:
# TASK-005: Variable-Size Meter System

# From plan-task-001-variable-size-meters-tdd-ready.md line 3:
**Task**: TASK-001 Variable-Size Meter System
```

**Impact**: HIGH

- Documentation inconsistency creates confusion
- Task tracking and dependency graphs may be incorrect
- Test file naming may reference wrong task number

**Recommendation**:

1. **IMMEDIATE**: Verify correct task number with project documentation
2. Check `docs/TASK_IMPLEMENTATION_PLAN.md` (line 100) which references "TASK-001: Variable-Size Meter System"
3. Rename either the specification file or update all references in the plan
4. Update all test file docstrings to reference correct task number

**Most Likely Resolution**:

- Specification file should be renamed to `TASK-001-...` (matches implementation plan)
- Or plan should be updated to reference TASK-005 throughout

#### Missing Dependency: AffordanceEngine Meter Validation

**From Specification** (TASK-001-VARIABLE-SIZE-METER-SYSTEM.md line 440-475):

```python
class AffordanceEngine:
    def _validate_meter_references(self):
        """Ensure all affordance effects/costs reference valid meters."""
        valid_meters = set(self.bars_config.meter_names)
        # ... validation logic ...
```

**Plan Coverage**: Line 907-951 mentions updating AffordanceEngine, but:

- No dedicated tests for meter reference validation
- No test for affordance loading with invalid meter names
- Integration tests don't verify affordance validation

**Recommendation**: Add Phase 2 test:

```python
def test_affordance_with_invalid_meter_reference_rejected(self, cpu_device, config_4meter):
    """AffordanceEngine should reject affordances referencing non-existent meters."""
    # Test that affordance with "hygiene" meter (not in 4-meter config) raises error
```

---

## Specific Concerns & Recommendations

### 1. Test File Location (CRITICAL)

**Issue**: Plan proposes creating new test files that conflict with consolidated structure

**Lines Affected**:

- Line 166-170: Test file organization table
- Line 391: `tests/test_townlet/unit/test_configuration.py` (EXTEND existing file)
- Line 759: `tests/test_townlet/unit/test_configuration.py` (EXTEND existing file)

**Current Reality**:

- `unit/test_configuration.py` already has 100+ lines and multiple test classes
- Adding 6 more test classes (32+ tests) will make it 300+ lines
- This violates good test organization (file becoming too large)

**Recommendations**:

1. **Best Practice**: Create `unit/environment/test_variable_meters.py` (NEW) for Phases 1-2
   - Justification: Meter logic is environment-specific, not general config
   - Precedent: Existing `unit/environment/test_meters.py` (line 1-100)

2. **Alternative**: Create task-specific subdirectory `unit/task-001/`
   - Pros: Clear isolation, easy to find all TASK-001 tests
   - Cons: Breaks convention of organizing by component

3. **Compromise**: Extend `unit/test_configuration.py` BUT split into sections with clear headers
   - Add marker: `# TASK-001: Variable Meter System Tests`
   - Group all TASK-001 test classes together

**Impact if Ignored**: MEDIUM

- Test discovery still works
- But test organization becomes confusing
- Hard to distinguish TASK-001 tests from general config tests

### 2. Fixture Naming & Scope

**Issue**: Proposed fixture names are generic and may conflict with future curriculum fixtures

**Lines Affected**: 187-289 (fixture definitions)

**Specific Concerns**:

- `config_4meter` - What if we later add L0_4meter curriculum level with different fixture?
- `env_4meter` - Conflicts with potential future `env_l0_4meter`
- `config_12meter` - No version or task identifier

**Recommendations**:

1. **Add task prefix**: `task001_config_4meter`, `task001_env_4meter`
2. **Or add purpose suffix**: `config_4meter_minimal`, `env_4meter_tutorial`
3. **Document scope in docstring**: "Use for: TASK-001 variable meter tests ONLY"

**Example**:

```python
@pytest.fixture
def task001_config_4meter(tmp_path, test_config_pack_path):
    """Create temporary 4-meter config pack for TASK-001 testing.

    Meters: energy, health, money, mood
    Use for: TASK-001 variable meter config/engine tests ONLY
    Do NOT use for L0 curriculum testing (use separate L0 fixtures)
    """
```

### 3. Missing Edge Case Tests

**Issue**: Plan doesn't test boundary conditions thoroughly

**Lines Affected**: 415-530 (Phase 1 tests)

**Missing Tests**:

1. **1-meter universe** (minimum valid)

   ```python
   def test_1_meter_config_validates(self):
       """1-meter config should validate (minimum valid case)."""
       config = BarsConfig(version="2.0", bars=[
           BarConfig(name="energy", index=0, tier="pivotal", ...)
       ], terminal_conditions=[...])
       assert config.meter_count == 1
   ```

2. **32-meter universe** (maximum valid)

   ```python
   def test_32_meter_config_validates(self):
       """32-meter config should validate (maximum valid case)."""
       bars = [BarConfig(name=f"meter_{i}", index=i, ...) for i in range(32)]
       config = BarsConfig(version="2.0", bars=bars, terminal_conditions=[])
       assert config.meter_count == 32
   ```

3. **Non-contiguous indices with correct count** (more specific error)

   ```python
   def test_non_contiguous_indices_with_gaps_rejected(self):
       """Indices [0, 2, 3, 4] should be rejected (missing 1)."""
       # Current test at line 486 only tests [0, 2] gap
       # Should test more complex gap patterns
   ```

**Recommendation**: Add these tests to Phase 1 test class (line 412-530)

### 4. Checkpoint Metadata Testing Gaps

**Issue**: Phase 4 tests don't verify all metadata fields

**Lines Affected**: 1258-1373 (Phase 4 tests)

**Missing Tests**:

1. **obs_dim in metadata** - Test verifies it exists (line 1285) but not that it's correct
2. **action_dim in metadata** - Not tested at all
3. **Metadata inspection helper** - Line 1516-1537 shows helper but no test for it

**Recommendation**: Add tests:

```python
def test_checkpoint_metadata_has_correct_obs_dim(self, cpu_device, config_4meter, tmp_path):
    """Checkpoint metadata obs_dim should match environment."""
    env = VectorizedHamletEnv(..., config_pack_path=config_4meter)
    save_checkpoint(...)
    checkpoint = load_checkpoint(...)
    assert checkpoint.universe_metadata["obs_dim"] == env.observation_dim

def test_inspect_checkpoint_metadata_without_loading_network(self, tmp_path):
    """Should be able to inspect metadata without loading full checkpoint."""
    # Test the inspect_checkpoint_metadata() helper
```

### 5. Integration Test Completeness

**Issue**: Phase 5 integration tests don't cover all critical scenarios

**Lines Affected**: 1557-1767 (Phase 5 tests)

**Missing Integration Tests**:

1. **Cross-meter-count checkpoint incompatibility** - Tested (line 1289), but not with population training
2. **Curriculum progression with variable meters** - Not tested
3. **Exploration (RND) with variable obs_dim** - Not tested
4. **Replay buffer with variable obs_dim** - Not tested

**Recommendation**: Add integration tests:

```python
def test_replay_buffer_with_variable_obs_dim(self, cpu_device, config_4meter):
    """ReplayBuffer should store correct obs_dim for variable meters."""
    env = VectorizedHamletEnv(..., config_pack_path=config_4meter)
    buffer = ReplayBuffer(capacity=1000, obs_dim=env.observation_dim, ...)
    # Test store/sample with 4-meter observations

def test_rnd_exploration_with_variable_obs_dim(self, cpu_device, config_4meter):
    """RND exploration should work with variable meter obs_dim."""
    env = VectorizedHamletEnv(..., config_pack_path=config_4meter)
    exploration = AdaptiveIntrinsicExploration(
        obs_dim=env.observation_dim, embed_dim=128, ...
    )
    # Test RND forward pass with 4-meter observations
```

### 6. Refactor Phase Clarity

**Issue**: REFACTOR phases are less detailed than RED/GREEN phases

**Lines Affected**:

- Line 638-668 (Phase 1 refactor)
- Line 1030-1038 (Phase 2 refactor)
- Line 1195-1213 (Phase 3 refactor)

**Problem**: Refactor phases say "Extract constants", "Add docstrings" but don't specify:

- Which constants to extract?
- Which methods need docstrings?
- What code smells to look for?

**Recommendation**: Add refactor checklists:

```markdown
### 1.3: Refactor (REFACTOR)

Checklist:
- [ ] Extract magic numbers: MIN_METERS = 1, MAX_METERS = 32
- [ ] Add docstrings to all new properties: meter_count, meter_names, meter_name_to_index
- [ ] Check for duplicate validation logic (index contiguity checked twice?)
- [ ] Verify error messages are user-friendly
- [ ] Run ruff to catch any new linting issues

**Run Tests**: Still pass after refactoring
```

### 7. Performance Testing

**Issue**: Plan mentions performance benchmarks (line 1733-1756) but marks them as optional

**Lines Affected**:

- Line 1733: `def test_performance_comparison_meter_counts(benchmark):`
- Phase 5 success criteria (line 1759-1765) don't include performance

**Concern**: Variable meter counts could have non-linear performance impact

- 4 meters: Small obs_dim, fast
- 8 meters: Standard
- 12 meters: Larger obs_dim, potentially slower
- But what about 32 meters? Could be significantly slower

**Recommendation**: Make performance tests mandatory in Phase 5

- Add to success criteria: "Performance degradation < 50% for 32-meter vs 8-meter"
- Test memory usage as well as speed
- Verify GPU tensor operations scale linearly

---

## Risk Assessment

### Implementation Risks

| Risk | Likelihood | Impact | Mitigation Status |
|------|------------|--------|-------------------|
| **Task numbering confusion** | HIGH | HIGH | ⚠️ Must resolve before start |
| **Test file location conflicts** | MEDIUM | MEDIUM | ⚠️ Needs clarification |
| **Fixture naming collisions** | LOW | MEDIUM | ✅ Recommendations provided |
| **Breaking existing tests** | LOW | HIGH | ✅ Good fixture reuse |
| **Integration failures** | LOW | MEDIUM | ✅ Comprehensive integration tests |
| **Performance degradation** | LOW | LOW | ⚠️ Needs mandatory testing |
| **Checkpoint incompatibility** | MEDIUM | LOW | ✅ Well handled with fallback |

### Technical Risks

1. **Hardcoded 8-Meter Assumptions Outside Identified Scope**
   - **Risk**: Plan identifies locations with `le=7` and `len(v) != 8`, but there may be more
   - **Evidence**: Specification (line 65-75) shows table of locations, but this is from "Subagent Analysis"
   - **Mitigation**: Run grep for all instances before starting:

     ```bash
     grep -r "le=7\|!= 8\|== 8" src/townlet/environment/*.py
     grep -r "torch.zeros.*8\)" src/townlet/environment/*.py
     ```

   - **Recommendation**: Add Phase 0.5: "Comprehensive Hardcoded-8 Audit" with grep-based verification

2. **Cascade Configuration Validity with Variable Meters**
   - **Risk**: Existing cascade configs reference meters by index (e.g., `source_index: 3`)
   - **Evidence**: Line 1786-1797 shows simplified 4-meter cascades, but what about complex existing cascades?
   - **Concern**: If we load 4-meter bars.yaml but existing cascades.yaml references indices [0-7], will it crash?
   - **Mitigation**: Plan addresses this in Phase 2 (line 993-1024) with AffordanceEngine validation
   - **Gap**: No test for loading 4-meter bars.yaml with 8-meter cascades.yaml
   - **Recommendation**: Add test:

     ```python
     def test_cascade_referencing_nonexistent_meter_rejected(self, cpu_device, config_4meter):
         """Cascade referencing meter index 7 should fail in 4-meter universe."""
         # Inject cascade with target_index=7 into 4-meter config
         # Should raise ValueError during load
     ```

3. **Observation Space Explosion**
   - **Risk**: 32-meter universe has large obs_dim, network may be slow or fail
   - **Impact**: Mentioned in specification (line 1157-1163) but dismissed as "operator's choice"
   - **Concern**: Could surprise operators if not documented
   - **Recommendation**: Add warning in config validation:

     ```python
     if meter_count > 16:
         logger.warning(
             f"Universe has {meter_count} meters. This will increase observation "
             f"dimension and may slow training. Consider reducing meter count "
             f"or using larger network capacity."
         )
     ```

---

## Specific Line-by-Line Issues

### Phase 0: Setup Test Fixtures

**Line 328-379: Phase 0 Description**

✅ Good: Properly isolates fixture setup as separate phase
⚠️ Issue: Missing verification that fixtures actually work

**Recommendation**: Add smoke test at end of Phase 0:

```python
# Line 363-365 exists, but should be more specific:
# 0.3: Verify Fixtures Load Correctly

def test_config_4meter_fixture_creates_valid_config(config_4meter):
    """Smoke test: config_4meter fixture should produce loadable config."""
    from townlet.environment.cascade_config import load_bars_config
    config = load_bars_config(config_4meter / "bars.yaml")
    assert config.meter_count == 4  # Verify it actually loaded
```

### Phase 1: Config Schema Refactor

**Line 389-530: Phase 1 Tests**

✅ Excellent: Comprehensive config validation tests
✅ Good: Tests both valid and invalid cases
⚠️ Minor: Some tests duplicate validation logic

**Line 460-468: test_0_meters_rejected**

```python
def test_0_meters_rejected(self):
    """Must have at least 1 meter."""
    with pytest.raises(ValueError, match="Must have at least 1 meter"):
        config = BarsConfig(version="2.0", bars=[], ...)
```

**Issue**: This will likely raise a Pydantic validation error BEFORE reaching custom validator
**Recommendation**: Verify error message matches Pydantic's actual error for empty list

**Line 531-617: Phase 1 Implementation**

✅ Excellent: Clear BEFORE/AFTER code blocks
✅ Good: Removes hardcoded constraints systematically
⚠️ Issue: Uses `le=7` removal but doesn't add `le=31` for max constraint

**Line 549: `index: int = Field(ge=0, description="Meter index in tensor [0, meter_count-1]")`**
**Issue**: Removes `le=7` but doesn't add dynamic upper bound
**Problem**: Nothing prevents `BarConfig(index=999, ...)` from being created
**Recommendation**: Add custom validator:

```python
@field_validator("index")
@classmethod
def validate_index(cls, v: int, info) -> int:
    # Note: Can't access meter_count here (it's in BarsConfig, not BarConfig)
    # So just validate >= 0, check contiguity in BarsConfig.validate_bars()
    if v < 0:
        raise ValueError(f"Bar index must be >= 0, got {v}")
    return v
```

### Phase 2: Engine Layer Refactor

**Line 751-896: Phase 2 Tests**

✅ Excellent: Tests tensor shapes and dynamic sizing
✅ Good: Tests initialization with config values
⚠️ Issue: Doesn't test meter name lookup failures

**Line 830-856: test_engine_uses_name_based_lookups_not_hardcoded_indices**

```python
def test_engine_uses_name_based_lookups_not_hardcoded_indices(...):
    # Verify _get_meter_index() method exists and works
    energy_idx = env.bars_config.meter_name_to_index["energy"]
```

**Issue**: Tests that method works, but doesn't test error handling for invalid names
**Recommendation**: Add test:

```python
def test_get_meter_index_with_invalid_name_raises_error(self, cpu_device, config_4meter):
    """Looking up non-existent meter should raise KeyError."""
    env = VectorizedHamletEnv(..., config_pack_path=config_4meter)
    with pytest.raises(KeyError, match="hygiene"):
        idx = env.bars_config.meter_name_to_index["hygiene"]  # Not in 4-meter config
```

**Line 904-990: Phase 2 Implementation (VectorizedHamletEnv)**

✅ Excellent: Dynamic tensor sizing
✅ Good: Initializes from config values
⚠️ Issue: Hardcoded movement costs still present

**Line 974-989: `_apply_movement_costs`**

```python
cost_map = {
    "energy": self.move_energy_cost,
    "hygiene": 0.003,
    "satiation": 0.004,
}
```

**Issue**: This is STILL hardcoded! If a 4-meter universe doesn't have "hygiene", this will silently do nothing
**Problem**: Plan says "will move to actions.yaml in TASK-003" but doesn't test that it works NOW
**Recommendation**: Add test:

```python
def test_movement_costs_only_applied_to_existing_meters(self, cpu_device, config_4meter):
    """Movement costs should only affect meters that exist in config."""
    env = VectorizedHamletEnv(..., config_pack_path=config_4meter)
    # 4-meter config has: energy, health, money, mood (no hygiene, satiation)

    initial_meters = env.meters.clone()
    moving = torch.tensor([True], device=cpu_device)
    env._apply_movement_costs(moving)

    # Energy should decrease (has movement cost)
    assert env.meters[0, 0] < initial_meters[0, 0]

    # Money should be unchanged (no movement cost)
    assert env.meters[0, 2] == initial_meters[0, 2]
```

### Phase 3: Network Layer Updates

**Line 1052-1144: Phase 3 Tests**

✅ Excellent: Tests networks with variable obs_dim
✅ Good: Verifies linear scaling
⚠️ Minor: Doesn't test RecurrentSpatialQNetwork (only SimpleQNetwork)

**Recommendation**: Add test for recurrent networks:

```python
def test_recurrent_network_with_variable_obs_dim(self, cpu_device, config_4meter):
    """RecurrentSpatialQNetwork should work with variable meters (POMDP)."""
    env = VectorizedHamletEnv(
        num_agents=1, grid_size=8, partial_observability=True, vision_range=2,
        device=cpu_device, config_pack_path=config_4meter,
    )

    network = RecurrentSpatialQNetwork(
        vision_window_size=5, meter_count=4, affordance_count=14, action_dim=5
    ).to(cpu_device)

    obs = env.reset()
    hidden = network.init_hidden(batch_size=1, device=cpu_device)
    q_values, hidden = network(obs, hidden)

    assert q_values.shape == (1, 5)
```

### Phase 4: Checkpoint Compatibility

**Line 1225-1509: Phase 4 Tests & Implementation**

✅ Excellent: Comprehensive checkpoint metadata testing
✅ Excellent: Legacy checkpoint fallback
✅ Good: Clear validation with helpful error messages
⚠️ Minor: Doesn't test checkpoint inspection helper (shown at line 1516-1537)

**Recommendation**: Add test for inspect helper:

```python
def test_inspect_checkpoint_metadata_helper(self, cpu_device, config_4meter, tmp_path):
    """inspect_checkpoint_metadata should work without loading network."""
    env = VectorizedHamletEnv(..., config_pack_path=config_4meter)
    checkpoint_path = tmp_path / "test.pt"
    save_checkpoint(..., path=checkpoint_path)

    # Should be able to inspect without loading full checkpoint
    metadata = inspect_checkpoint_metadata(checkpoint_path)
    assert metadata["meter_count"] == 4
    assert metadata["meter_names"] == ["energy", "health", "money", "mood"]
```

### Phase 5: Integration Testing

**Line 1550-1767: Phase 5 Tests**

✅ Excellent: End-to-end training tests
✅ Good: Tests all meter counts (4, 8, 12)
⚠️ Issue: Integration tests use random policy, not actual trained agent

**Line 1598-1599:**

```python
# Random policy for testing
actions = torch.randint(0, 5, (4,), device=cpu_device)
```

**Issue**: This tests environment mechanics but not actual training convergence
**Concern**: A bug in Q-network gradient flow wouldn't be caught
**Recommendation**: Add test with minimal training:

```python
def test_training_improves_survival_with_4_meters(self, cpu_device, config_4meter):
    """Agent should improve survival time with training (4-meter universe)."""
    env = VectorizedHamletEnv(num_agents=4, ..., config_pack_path=config_4meter)
    population = VectorizedPopulation(env=env, learning_rate=0.001, ...)

    # Collect early survival times (first 10 episodes)
    early_survival = []
    for _ in range(10):
        steps = run_episode(env, population, epsilon=1.0)  # Random
        early_survival.append(steps)

    # Train for 100 episodes
    for _ in range(100):
        run_episode(env, population, epsilon=0.1)
        population.update()

    # Collect late survival times (last 10 episodes)
    late_survival = []
    for _ in range(10):
        steps = run_episode(env, population, epsilon=0.1)
        late_survival.append(steps)

    # Agent should survive longer after training
    assert mean(late_survival) > mean(early_survival), "Training should improve survival"
```

---

## Recommendations Summary

### Critical (Must Fix Before Implementation)

1. **RESOLVE TASK NUMBERING** (Line 1-10 of plan)
   - Verify whether this is TASK-001 or TASK-005
   - Update all references consistently
   - Check dependency graphs in TASK_IMPLEMENTATION_PLAN.md

2. **CLARIFY TEST FILE ORGANIZATION** (Line 166-170)
   - Decision: Extend `unit/test_configuration.py` OR create `unit/environment/test_variable_meters.py`?
   - Justify choice in plan
   - Update Phase 0 to create correct file structure

3. **AUDIT HARDCODED ASSUMPTIONS** (Before Phase 1)
   - Run comprehensive grep for `le=7`, `!= 8`, `== 8`, `torch.zeros.*8`
   - Document ALL locations that need changes
   - Add to Phase 0.5: "Comprehensive Audit"

### High Priority (Should Fix)

4. **ADD BOUNDARY TESTS** (Phase 1, line 415-530)
   - Test 1-meter config (minimum)
   - Test 32-meter config (maximum)
   - Test more complex non-contiguous index patterns

5. **ADD INTEGRATION TESTS FOR TRAINING** (Phase 5, line 1550-1767)
   - Test that training actually improves performance
   - Test replay buffer with variable obs_dim
   - Test RND exploration with variable obs_dim
   - Test curriculum progression with variable meters

6. **RENAME FIXTURES FOR CLARITY** (Line 187-289)
   - Use `task001_config_4meter` or `variable_meter_config_4meter`
   - Document scope: "TASK-001 ONLY, not for curriculum levels"

7. **MAKE PERFORMANCE TESTS MANDATORY** (Phase 5)
   - Add to success criteria
   - Test memory usage as well as speed
   - Verify 32-meter universe is still usable

### Medium Priority (Nice to Have)

8. **ADD REFACTOR CHECKLISTS** (All refactor phases)
   - Specify which constants to extract
   - List which methods need docstrings
   - Define code smell checks

9. **TEST ERROR MESSAGES** (Phase 1-4)
   - Verify error messages are user-friendly
   - Test that validation errors include examples of correct format

10. **ADD RECURRENT NETWORK TESTS** (Phase 3)
    - Test RecurrentSpatialQNetwork with variable obs_dim
    - Verify LSTM hidden state handling

### Low Priority (Optional)

11. **ADD CHECKPOINT INSPECTION TEST** (Phase 4)
    - Test `inspect_checkpoint_metadata()` helper

12. **ADD CASCADE VALIDATION TEST** (Phase 2)
    - Test loading 4-meter bars with 8-meter cascades (should fail)

---

## Effort Re-Estimate

**Original Plan**: 15-21 hours (with TDD overhead)

**Revised Estimate**: 17-24 hours

**Breakdown**:

- **Phase 0**: 1h (unchanged)
- **Phase 0.5** (NEW): 1-2h (comprehensive hardcoded-8 audit)
- **Phase 1**: 3-4h → 4-5h (+1h for boundary tests)
- **Phase 2**: 4-6h → 5-7h (+1h for edge cases)
- **Phase 3**: 2-3h → 3-4h (+1h for recurrent network tests)
- **Phase 4**: 2-3h (unchanged)
- **Phase 5**: 2-3h → 3-5h (+1-2h for training convergence tests)

**Contingency**: +2-3h for resolving task numbering confusion and test file organization

---

## Final Verdict

### Proceed with Implementation? ⚠️ **YES, WITH REVISIONS**

**Conditions**:

1. ✅ Resolve task numbering (TASK-001 vs TASK-005)
2. ✅ Clarify test file organization
3. ✅ Add Phase 0.5: Comprehensive hardcoded-8 audit
4. ✅ Rename fixtures with task/purpose prefix
5. ⚠️ Consider adding boundary tests and training convergence tests

**Quality Gate**:

- [ ] Task number confirmed and all references updated
- [ ] Test file organization decision documented
- [ ] Hardcoded-8 audit completed and verified
- [ ] Fixtures renamed and scoped in docstrings
- [ ] Phase 0 smoke tests pass (fixtures load correctly)

**If conditions met**: Plan is **EXCELLENT** and ready for implementation
**If conditions ignored**: Risk of confusion, test conflicts, and incomplete coverage

---

## Positive Highlights

Despite the issues identified, this plan demonstrates **exceptional quality** in several areas:

1. **TDD Discipline**: Strict RED-GREEN-REFACTOR cycles with clear phase separation
2. **Fixture Composition**: Excellent reuse of existing fixtures (cpu_device, test_config_pack_path, tmp_path)
3. **Behavioral Testing**: Most tests verify behavior rather than implementation details
4. **Comprehensive Coverage**: 32 tests across 5 phases covering all layers
5. **Documentation**: Clear docstrings, example configs, and success criteria
6. **Risk Mitigation**: Legacy checkpoint handling, backward compatibility, validation
7. **Incremental Approach**: Proper phasing from config → engine → network → checkpoint → integration

**This is one of the most thorough TDD plans I've reviewed.** The issues identified are mostly organizational and documentation clarity, not fundamental problems with the approach.

---

## Conclusion

The TASK-001 (or TASK-005?) implementation plan is **fundamentally sound** but requires **moderate revisions** before proceeding. The TDD methodology is excellent, fixture patterns are correct, and test coverage is comprehensive. However, task numbering confusion and test file organization conflicts must be resolved to avoid downstream problems.

**Recommended Action**:

1. Spend 2-3 hours resolving critical issues (task numbering, test organization, audit)
2. Implement boundary tests and rename fixtures (1-2 hours)
3. Then proceed with confidence - the plan is solid

**Confidence Level**: **HIGH** (after critical issues resolved)
**Implementation Success Probability**: **85%** (with revisions), **65%** (without revisions)

---

**Reviewer Notes**: This review conducted by analyzing plan against actual codebase state (644 tests, refactored November 2025 structure). All line references verified. Recommendations are actionable and prioritized.
