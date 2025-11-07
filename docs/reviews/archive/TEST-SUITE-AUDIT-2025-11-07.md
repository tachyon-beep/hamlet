# Test Suite Audit: Magic Numbers and Design Issues

**Date**: 2025-11-07
**Auditor**: Claude Code (Independent Review)
**Scope**: 90 test files in `tests/test_townlet/`

---

## Executive Summary

The test suite has **significant brittleness** due to:
1. **Magic number proliferation** (action counts, observation dims, meter indices)
2. **Duplicate test setup** (100+ manual env instantiations)
3. **Hardcoded expectations** that break when implementation changes
4. **Missing abstractions** (should derive from system, not hardcode)

**Severity**: ⚠️ **HIGH** - Recent TASK-002B changes required manual updates to dozens of tests

---

## Category 1: Magic Action Counts ⚠️ CRITICAL

### Problem: Hardcoded action_dim expectations

**Pattern Found**:
```python
# tests/test_townlet/integration/test_curriculum_transfer.py:48-49
assert env_l0.action_dim == 8, f"L0 should have 8 actions, got {env_l0.action_dim}"
assert env_l1.action_dim == 8, f"L1 should have 8 actions, got {env_l1.action_dim}"
```

**Why This Is Bad**:
- When custom actions changed (4 → 2), these broke
- Should derive from `env.action_dim` not hardcode expectations
- Tests fail for the wrong reason (changed action count, not broken transfer)

**Occurrences**:
- `test_curriculum_transfer.py`: 2 instances (lines 48-49)
- `test_substrate_migration.py`: 3 instances (lines 59, 88, 124)
- `test_action_builder.py`: 6 instances (manual test data)

**Recommendation**:
```python
# ❌ BAD: Hardcoded expectation
assert env_l0.action_dim == 8

# ✅ GOOD: Test the invariant that matters
assert env_l0.action_dim == env_l1.action_dim, "Checkpoint transfer requires same action_dim"

# ✅ GOOD: Derive from system
expected_actions = substrate.action_space_size + len(load_global_actions())
assert env.action_dim == expected_actions
```

---

## Category 2: Magic Observation Dimensions ⚠️ CRITICAL

### Problem: Hardcoded FULL_OBS_DIM = 93

**Pattern Found**:
```python
# Duplicated across 6 test files:
FULL_OBS_DIM = 93  # Standard 8×8 full observability observation dimension
```

**Files with this constant**:
1. `test_checkpointing.py:36`
2. `test_configuration.py:56`
3. `test_replay_buffers.py:20`
4. `test_networks.py:26`
5. `test_exploration_strategies.py:25`

**Why This Is Bad**:
- Magic number 93 = 64 (grid) + 2 (pos) + 8 (meters) + 15 (affordance) + 4 (temporal)
- If ANY component changes, all 6 files need manual updates
- Tests can pass with wrong obs_dim if we forget to update the constant

**Impact**:
- Phase 6 (observation encoding changes) will break all these tests
- Variable meter configs (TASK-001) already make this invalid for 4-meter/12-meter

**Recommendation**:
```python
# ❌ BAD: Hardcoded magic number
FULL_OBS_DIM = 93

# ✅ GOOD: Derive from environment
def test_network_output_shape(basic_env):
    obs_dim = basic_env.observation_dim  # Dynamic!
    net = SimpleQNetwork(obs_dim=obs_dim, action_dim=basic_env.action_dim)

# ✅ GOOD: Use fixture that computes it
@pytest.fixture
def expected_obs_dim(basic_env):
    return basic_env.observation_dim
```

---

## Category 3: Duplicate Environment Setup 🔴 HIGH

### Problem: Manual VectorizedHamletEnv() instantiation

**Statistics**:
- **100 manual instantiations** in integration tests
- **30 Grid2DSubstrate(width=8, ...)** creations
- Many tests pass identical parameters

**Example** (test_curriculum_transfer.py:54-67):
```python
env_l0 = VectorizedHamletEnv(
    config_pack_path=Path("configs/L0_0_minimal"),
    num_agents=1,
    grid_size=3,
    partial_observability=False,
    vision_range=2,
    enable_temporal_mechanics=False,
    enabled_affordances=["Bed"],
    move_energy_cost=0.5,
    wait_energy_cost=0.1,
    interact_energy_cost=0.3,
    agent_lifespan=1000,
    device=torch.device("cpu"),
)
```

**Why This Is Bad**:
- Repeated setup code (DRY violation)
- When parameters change, dozens of tests need updates
- Tests that should use fixtures don't

**Available Fixtures** (conftest.py):
- `basic_env` ✅
- `pomdp_env` ✅
- `temporal_env` ✅
- `grid2d_3x3_env` ✅
- `grid2d_8x8_env` ✅
- `task001_env_4meter` ✅
- `task001_env_12meter` ✅

**Recommendation**:
```python
# ❌ BAD: Manual instantiation
def test_something():
    env = VectorizedHamletEnv(config_pack_path=..., num_agents=1, ...)

# ✅ GOOD: Use fixture
def test_something(basic_env):
    env = basic_env

# ✅ GOOD: Parametrize when you need variations
@pytest.mark.parametrize("grid_size", [3, 8, 16])
def test_scales_with_grid_size(test_config_pack_path, device, grid_size):
    env = VectorizedHamletEnv(config_pack_path=test_config_pack_path, grid_size=grid_size, device=device)
```

---

## Category 4: Exact Floating Point Comparisons 🟡 MEDIUM

### Problem: Fragile equality checks

**Statistics**: 190 uses of `.item()` for tensor → scalar conversion

**Pattern Found**:
```python
# tests/test_townlet/integration/test_temporal_mechanics.py:571
assert money_values[1] == pytest.approx(0.55625, abs=0.001)
```

**Why This Can Be Bad**:
- ✅ GOOD: Uses `pytest.approx()` with tolerance
- ⚠️ RISKY: Exact values like 0.55625 are implementation-specific
- If cost formula changes slightly, test breaks

**Worse Pattern**:
```python
# Some tests do:
assert value == 0.0  # ❌ Exact float comparison (no tolerance)
```

**Recommendation**:
- Use `pytest.approx()` for all float comparisons ✅
- Test **relationships** not exact values when possible
- Example: `assert final_energy < initial_energy` (energy decreases)
- Not: `assert final_energy == 0.995` (exact value)

---

## Category 5: Missing Test Helpers/Factories 🟡 MEDIUM

### Problem: No centralized test data generation

**Current State**:
- Each test creates ActionConfig manually
- Each test creates substrate configurations inline
- Lots of boilerplate duplication

**Good Example** (from latest commit):
```python
# tests/test_townlet/unit/test_action_builder.py:24-44
def make_action(id: int, name: str, type: str, **overrides) -> ActionConfig:
    """Factory for creating ActionConfig in tests with explicit defaults."""
    defaults = {
        "id": id,
        "name": name,
        "type": type,
        "delta": None,
        "teleport_to": None,
        "costs": {},
        "effects": {},
        "enabled": True,
        "description": None,
        "icon": None,
        "source": "substrate",
        "source_affordance": None,
    }
    defaults.update(overrides)
    return ActionConfig(**defaults)
```

**What's Missing**:
- Similar factories for SubstrateConfig, BarsConfig, AffordanceConfig
- Centralized test data builders for common scenarios
- Shared test utilities module

**Recommendation**:
- Create `tests/test_townlet/builders/` with factories
- Add `tests/test_townlet/test_data/` with canonical configs
- Document patterns in test README

---

## Category 6: Tests Coupled to Implementation 🟡 MEDIUM

### Problem: Tests know too much about internals

**Pattern Found**:
```python
# tests/test_townlet/unit/test_substrate_actions.py
assert len(actions) == 6  # UP, DOWN, LEFT, RIGHT, INTERACT, WAIT
```

**Why This Can Be Bad**:
- Assumes specific action ordering
- If substrate adds new actions, breaks
- Should test **behavior** not **structure**

**Better Approach**:
```python
# ✅ Test behavior
actions = substrate.get_default_actions()
action_names = {a.name for a in actions}
assert "UP" in action_names
assert "INTERACT" in action_names
assert all(a.id is not None for a in actions)  # Test invariant

# ❌ Don't test structure
assert len(actions) == 6  # Fragile!
```

---

## Category 7: Inconsistent Test Organization 🟢 LOW

### Problem: Tests split across integration/unit in unpredictable ways

**Observations**:
- Some substrate tests in `integration/` should be `unit/`
- Some environment tests in `unit/` should be `integration/`
- Not a blocker, but reduces discoverability

**Example Issues**:
- `test_substrate_nd.py` is integration but tests pure substrate logic
- `test_variable_meters_e2e.py` mixes unit and integration concerns

**Recommendation**:
- **Unit**: Tests single component in isolation (no config loading)
- **Integration**: Tests multiple components together (with real configs)
- Review test placement during refactors

---

## Priority Recommendations

### 🔴 IMMEDIATE (Before Next Major Change)

1. **Remove hardcoded action_dim=8 from curriculum transfer tests**
   - Replace with `env.action_dim` comparisons
   - File: `test_curriculum_transfer.py:48-49`

2. **Remove FULL_OBS_DIM=93 constant from all 6 files**
   - Use `env.observation_dim` dynamically
   - Files: test_checkpointing, test_configuration, test_replay_buffers, test_networks, test_exploration_strategies

3. **Add test utility module**
   - `tests/test_townlet/utils/builders.py` with factories
   - Document in tests/README.md

### 🟡 SOON (Next Sprint)

4. **Replace 50+ manual env instantiations with fixtures**
   - Start with tests that duplicate setup
   - Use parametrize for variations

5. **Add obs_dim/action_dim validation helpers**
   ```python
   def assert_valid_observation(env, obs):
       assert obs.shape[1] == env.observation_dim
       assert obs.dtype == torch.float32
   ```

6. **Audit temporal mechanics tests for exact float comparisons**
   - Ensure all use `pytest.approx()`
   - Consider testing ranges instead of exact values

### 🟢 LATER (Ongoing Improvement)

7. **Create test data builders for configs**
8. **Move substrate unit tests out of integration/**
9. **Add test coverage metrics to CI**
10. **Document test organization patterns**

---

## Metrics

| Category | Count | Severity | Effort to Fix |
|----------|-------|----------|--------------|
| Hardcoded action_dim | 11 | HIGH | 1 hour |
| Hardcoded FULL_OBS_DIM | 6 files | CRITICAL | 2 hours |
| Manual env instantiation | 100+ | HIGH | 8 hours |
| Exact float comparisons | 190 | MEDIUM | 4 hours |
| Missing test factories | N/A | MEDIUM | 4 hours |
| Implementation coupling | ~30 tests | MEDIUM | 6 hours |

**Total Technical Debt**: ~25 hours of focused refactoring

---

## Conclusion

The test suite suffers from **premature concretization** - hardcoding values that should be derived from the system under test. This creates maintenance burden and false failures when implementation details change.

**Key Insight**: Tests broke during TASK-002B not because functionality changed, but because we hardcoded implementation details (action_dim=6 → 8) instead of testing invariants (checkpoint transfer requires matching action_dim).

**Next Steps**:
1. Fix immediate issues (FULL_OBS_DIM, action_dim hardcoding)
2. Create test utilities module with builders
3. Incrementally replace manual instantiation with fixtures
4. Add test pattern documentation

**Long-term Goal**: Tests should survive implementation changes by testing **behavior and invariants**, not **structure and magic numbers**.
