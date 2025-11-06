# TASK-002A Phase 8: Testing & Verification - Research Document

**Date**: 2025-11-05
**Status**: Research Complete
**Phase**: 8 of 8 (Final Verification)
**Estimated Effort**: 6-8 hours

---

## Executive Summary

Phase 8 is the **final verification phase** for TASK-002A substrate abstraction. It ensures all changes (Phases 0-7) work correctly across Grid2D, Aspatial, and future substrates.

**Key Finding**: ~40 test files need updates for substrate parameterization. Most are **mechanical changes** (replace hardcoded `8` with `env.substrate.width`).

**Scope**:
1. **Unit Tests**: Update hardcoded position/grid assumptions
2. **Integration Tests**: Add parameterized tests for Grid2D + Aspatial
3. **Property-Based Tests**: Verify substrate contracts hold
4. **Regression Tests**: Ensure backward compatibility
5. **Performance Benchmarks**: Verify no slowdown from abstraction

**Critical Discovery**: Tests currently assume 8×8 grid with 2D positions. Must parameterize ALL grid size and position dimension assumptions.

---

## Research Questions

### Q1: Which test files have hardcoded position assumptions?

**Answer**: 15+ test files hardcode `(x, y)` positions or `grid_size=8`

**Files Requiring Updates**:

**Environment Tests** (`tests/test_townlet/unit/environment/`):
1. `test_observations.py` - Hardcodes grid_size=8, assumes 2D positions
2. `test_meters.py` - Uses fixed meter indices (8 meters)
3. `test_affordances.py` - Assumes affordances have x, y attributes
4. `test_action_masking.py` - Checks boundary conditions for 8×8 grid
5. `test_engine_dynamic_sizing.py` - Already parameterized for meters (TASK-001)

**Population Tests** (`tests/test_townlet/unit/population/`):
6. `test_action_selection.py` - Hardcodes agent positions

**Agent Tests** (`tests/test_townlet/unit/agent/`):
7. `test_networks.py` - Hardcodes obs_dim calculations

**Integration Tests** (`tests/test_townlet/integration/`):
8. `test_episode_execution.py` - Assumes 2D movement
9. `test_data_flows.py` - Checks position tensor shapes
10. `test_training_loop.py` - Hardcodes grid_size
11. `test_temporal_mechanics.py` - Uses fixed grid dimensions
12. `test_checkpointing.py` - Validates checkpoint format (BREAKING CHANGE)
13. `test_curriculum_signal_purity.py` - Assumes spatial exploration
14. `test_intrinsic_exploration.py` - Uses position-based novelty
15. `test_recording_recorder.py` - Records (x, y) positions

**Property-Based Tests** (`tests/test_townlet/properties/`):
16. `test_environment_properties.py` - Grid boundary properties
17. `test_exploration_properties.py` - Position-based novelty properties

**Evidence**:
```python
# test_observations.py (BEFORE - hardcoded)
def test_full_obs_dimension():
    env = VectorizedHamletEnv(grid_size=8, ...)
    obs = env.reset()
    # Hardcoded: 8×8=64 grid + 8 meters + 15 affordances + 4 temporal = 91
    assert obs.shape[1] == 91

# test_observations.py (AFTER - parameterized)
def test_full_obs_dimension(basic_env):
    obs = basic_env.reset()
    # Computed from substrate
    expected_dim = (basic_env.substrate.width * basic_env.substrate.height
                    + 8  # meters
                    + 15  # affordances
                    + 4)  # temporal
    assert obs.shape[1] == expected_dim
```

---

### Q2: What integration tests need substrate parameterization?

**Answer**: All integration tests that create environments need substrate fixtures

**Current Pattern** (BEFORE):
```python
def test_training_loop():
    """Test full training loop."""
    env = VectorizedHamletEnv(
        num_agents=4,
        grid_size=8,  # ← Hardcoded
        device=device
    )
    # ...
```

**New Pattern** (AFTER):
```python
@pytest.mark.parametrize("substrate_fixture", ["grid2d_env", "aspatial_env"])
def test_training_loop(substrate_fixture, request):
    """Test full training loop with different substrates."""
    env = request.getfixturevalue(substrate_fixture)
    # Works for both Grid2D and Aspatial
    # ...
```

**Integration Tests Requiring Parameterization**:

1. **test_episode_execution.py**:
   - Test episode lifecycle (reset → step → done)
   - Must work for Grid2D (agents move) and Aspatial (no movement)

2. **test_data_flows.py**:
   - Test observation/action flow through system
   - Must validate position_dim from substrate (2 for Grid2D, 0 for Aspatial)

3. **test_training_loop.py**:
   - Test full Q-learning loop
   - Must work with both spatial and aspatial exploration

4. **test_temporal_mechanics.py**:
   - Test time-based affordances
   - Must work without positions (aspatial)

5. **test_checkpointing.py** (BREAKING CHANGE):
   - Test checkpoint save/load
   - Must validate new checkpoint format (Version 3 with substrate)

6. **test_curriculum_signal_purity.py**:
   - Test curriculum doesn't leak position info
   - Must work for aspatial (no positions to leak)

7. **test_intrinsic_exploration.py**:
   - Test RND novelty rewards
   - Must work for aspatial (observation-based novelty, not position-based)

8. **test_recording_recorder.py**:
   - Test episode recording
   - Must handle variable position_dim (2 for Grid2D, 0 for Aspatial)

**Finding**: All 8 integration tests can be parameterized with `@pytest.mark.parametrize("substrate_type", ["grid2d", "aspatial"])`.

---

### Q3: How to add property-based tests for substrate abstraction?

**Answer**: Use Hypothesis to generate random substrate configs and verify contracts

**Property 1: Position validity**
```python
from hypothesis import given, strategies as st

@given(
    width=st.integers(min_value=2, max_value=20),
    height=st.integers(min_value=2, max_value=20),
    x=st.integers(min_value=0, max_value=19),
    y=st.integers(min_value=0, max_value=19)
)
def test_position_validation_property(width, height, x, y):
    """Any position within bounds should be valid."""
    substrate = Grid2DSubstrate(width=width, height=height)

    position = torch.tensor([x, y], dtype=torch.long)

    if x < width and y < height:
        # Position within bounds → should be valid
        assert substrate.is_valid_position(position)
    else:
        # Position outside bounds → should be invalid
        assert not substrate.is_valid_position(position)
```

**Property 2: Distance symmetry**
```python
@given(
    width=st.integers(min_value=3, max_value=10),
    height=st.integers(min_value=3, max_value=10),
    x1=st.integers(min_value=0, max_value=9),
    y1=st.integers(min_value=0, max_value=9),
    x2=st.integers(min_value=0, max_value=9),
    y2=st.integers(min_value=0, max_value=9)
)
def test_distance_symmetry_property(width, height, x1, y1, x2, y2):
    """Distance from A to B should equal distance from B to A."""
    substrate = Grid2DSubstrate(width=width, height=height, distance_metric="manhattan")

    # Clamp positions to grid
    x1, y1 = min(x1, width-1), min(y1, height-1)
    x2, y2 = min(x2, width-1), min(y2, height-1)

    pos1 = torch.tensor([[x1, y1]], dtype=torch.long)
    pos2 = torch.tensor([[x2, y2]], dtype=torch.long)

    dist_ab = substrate.compute_distances(pos1, pos2)
    dist_ba = substrate.compute_distances(pos2, pos1)

    assert torch.allclose(dist_ab, dist_ba), "Distance should be symmetric"
```

**Property 3: Movement validity**
```python
@given(
    width=st.integers(min_value=3, max_value=10),
    height=st.integers(min_value=3, max_value=10),
    x=st.integers(min_value=0, max_value=9),
    y=st.integers(min_value=0, max_value=9),
    action=st.integers(min_value=0, max_value=3)  # UP, DOWN, LEFT, RIGHT
)
def test_movement_stays_in_bounds(width, height, x, y, action):
    """Moving from valid position should stay in bounds (with clamping)."""
    substrate = Grid2DSubstrate(width=width, height=height, boundary="clamp")

    # Clamp start position to grid
    x, y = min(x, width-1), min(y, height-1)

    positions = torch.tensor([[x, y]], dtype=torch.long)
    actions = torch.tensor([action], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, actions)

    # New position must be valid
    assert substrate.is_valid_position(new_positions[0])

    # New position must be within bounds
    assert 0 <= new_positions[0, 0] < width
    assert 0 <= new_positions[0, 1] < height
```

**Property 4: Observation dimension consistency**
```python
@given(
    width=st.integers(min_value=3, max_value=10),
    height=st.integers(min_value=3, max_value=10),
    num_agents=st.integers(min_value=1, max_value=5)
)
def test_observation_dim_consistency(width, height, num_agents, test_config_pack_path, cpu_device):
    """Observation dimension should match substrate + meters + affordances + temporal."""
    env = VectorizedHamletEnv(
        num_agents=num_agents,
        grid_size=width,  # Assumes square grid
        device=cpu_device,
        config_pack_path=test_config_pack_path
    )

    obs = env.reset()

    # Expected dimension (full observability)
    grid_dim = width * height  # From substrate
    meter_dim = 8  # From config
    affordance_dim = 15  # 14 affordances + "none"
    temporal_dim = 4  # time_of_day, retirement_age, interaction_progress, interaction_ticks

    expected_dim = grid_dim + meter_dim + affordance_dim + temporal_dim

    assert obs.shape == (num_agents, expected_dim), f"Observation shape mismatch: {obs.shape} vs ({num_agents}, {expected_dim})"
```

**Finding**: Property-based tests catch edge cases (e.g., 2×2 grid, single agent) that manual tests miss.

---

### Q4: What regression tests are needed?

**Answer**: Ensure backward compatibility and behavioral equivalence

**Regression Test 1: Grid2D behaves like legacy hardcoded grid**
```python
def test_grid2d_equivalent_to_legacy_hardcoded_grid(test_config_pack_path, cpu_device):
    """Grid2D substrate should produce identical behavior to legacy hardcoded grid."""
    # Create environment with Grid2D substrate (new)
    env_new = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=cpu_device,
        config_pack_path=test_config_pack_path
    )

    # Set identical random seeds
    torch.manual_seed(42)
    obs_new = env_new.reset()

    # Verify substrate is Grid2D
    assert env_new.substrate.type == "grid2d"

    # Verify observation dimension matches legacy (91 for 8×8 grid + 8 meters + 15 affordances + 4 temporal)
    assert obs_new.shape[1] == 91, f"Observation dim changed: {obs_new.shape[1]} vs 91"

    # Run 10 steps with fixed actions
    actions = [0, 1, 2, 3, 4] * 2  # UP, DOWN, LEFT, RIGHT, INTERACT × 2
    for action in actions:
        obs, reward, done, info = env_new.step(torch.tensor([action]))

        # Verify position within bounds
        assert 0 <= env_new.positions[0, 0] < 8
        assert 0 <= env_new.positions[0, 1] < 8
```

**Regression Test 2: Checkpoints load correctly**
```python
def test_checkpoint_format_version_3(tmp_path, basic_env, adversarial_curriculum, epsilon_greedy_exploration):
    """New checkpoints (Version 3) should include substrate metadata."""
    from townlet.population.vectorized import VectorizedPopulation

    pop = VectorizedPopulation(
        env=basic_env,
        curriculum=adversarial_curriculum,
        exploration=epsilon_greedy_exploration,
        device=basic_env.device
    )

    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint_v3.pt"
    pop.save_checkpoint(checkpoint_path, episode=100)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Verify Version 3 format
    assert checkpoint["version"] == 3, "Checkpoint should be Version 3"

    # Verify substrate metadata present
    assert "substrate_config" in checkpoint, "Checkpoint should include substrate config"

    substrate_config = checkpoint["substrate_config"]
    assert substrate_config["type"] == "grid2d"
    assert substrate_config["width"] == 8
    assert substrate_config["height"] == 8
```

**Regression Test 3: Observation dimensions unchanged**
```python
@pytest.mark.parametrize(
    "grid_size,expected_obs_dim",
    [
        (3, 9 + 8 + 15 + 4),   # L0_0_minimal: 36
        (7, 49 + 8 + 15 + 4),  # L0_5_dual_resource: 76
        (8, 64 + 8 + 15 + 4),  # L1_full_observability: 91
    ]
)
def test_observation_dims_unchanged_from_legacy(grid_size, expected_obs_dim, test_config_pack_path, cpu_device):
    """Observation dimensions should match legacy values (behavioral equivalence)."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=grid_size,
        device=cpu_device,
        config_pack_path=test_config_pack_path,
        partial_observability=False
    )

    obs = env.reset()

    assert obs.shape[1] == expected_obs_dim, (
        f"Observation dimension changed for {grid_size}×{grid_size} grid: "
        f"{obs.shape[1]} vs {expected_obs_dim}"
    )
```

**Finding**: Regression tests ensure substrate abstraction doesn't change existing behavior.

---

## Critical Findings

### Finding 1: conftest.py Needs Substrate Fixtures

**Issue**: Current fixtures hardcode `grid_size=8` and `partial_observability=False`

**Evidence** (`tests/test_townlet/conftest.py`, line 124):
```python
@pytest.fixture
def basic_env(test_config_pack_path: Path, device: torch.device) -> VectorizedHamletEnv:
    """Create a basic environment with default test config."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,  # ← Hardcoded
        partial_observability=False,
        # ...
    )
```

**Fix**: Add substrate-parameterized fixtures:
```python
@pytest.fixture
def grid2d_3x3_env(test_config_pack_path, device):
    """Small 3×3 Grid2D environment for fast tests."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=3,
        device=device,
        config_pack_path=test_config_pack_path
    )

@pytest.fixture
def grid2d_8x8_env(test_config_pack_path, device):
    """Standard 8×8 Grid2D environment (legacy compatibility)."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=device,
        config_pack_path=test_config_pack_path
    )

@pytest.fixture
def aspatial_env(test_config_pack_path, device):
    """Aspatial environment (no grid, meters only)."""
    # Requires aspatial config pack (created in Phase 8)
    aspatial_config_path = test_config_pack_path.parent / "aspatial_test"
    return VectorizedHamletEnv(
        num_agents=1,
        device=device,
        config_pack_path=aspatial_config_path
    )
```

**Impact**: All tests can use `@pytest.mark.parametrize("env_fixture", ["grid2d_3x3_env", "grid2d_8x8_env", "aspatial_env"])` to test multiple substrates.

---

### Finding 2: Checkpoint Format Breaking Change

**Issue**: Phase 4 introduced Version 3 checkpoint format with substrate metadata

**Current Format** (Version 2 - legacy):
```python
{
    "version": 2,
    "episode": 100,
    "population_state": {
        "q_network": {...},
        "optimizer": {...}
    },
    "epsilon": 0.5
}
```

**New Format** (Version 3 - with substrate):
```python
{
    "version": 3,
    "episode": 100,
    "substrate_config": {  # ← NEW
        "type": "grid2d",
        "width": 8,
        "height": 8,
        "topology": "square",
        "boundary": "clamp"
    },
    "population_state": {
        "q_network": {...},
        "optimizer": {...}
    },
    "epsilon": 0.5
}
```

**Test Requirements**:
1. **test_checkpoint_save_includes_substrate**: Verify Version 3 checkpoints have substrate_config
2. **test_checkpoint_load_version3**: Verify loading Version 3 works
3. **test_checkpoint_load_version2_fails**: Verify loading Version 2 raises clear error

**Finding**: Checkpoint tests MUST validate new format and reject old format.

---

### Finding 3: Aspatial Config Pack Required

**Issue**: Aspatial tests need aspatial config pack (doesn't exist yet)

**Solution**: Create `configs/aspatial_test/` config pack in Phase 8

**Required Files**:
```
configs/aspatial_test/
├── substrate.yaml    # type: aspatial
├── bars.yaml         # 8 meters (same as test config)
├── cascades.yaml     # Empty (no cascades for simple test)
├── affordances.yaml  # 4 affordances (Bed, Hospital, HomeMeal, Job)
├── cues.yaml         # Minimal UI cues
└── training.yaml     # Test training config
```

**substrate.yaml**:
```yaml
version: "1.0"
type: "aspatial"
description: "Aspatial test substrate (no spatial grid, meters only)"

aspatial:
  # No additional configuration needed
  # Aspatial substrates have no positions, distances, or movement
```

**Finding**: Creating aspatial config pack is **prerequisite** for aspatial tests.

---

### Finding 4: Position Tensor Shape Validation

**Issue**: Many tests assume `positions.shape == (num_agents, 2)` (2D positions)

**Evidence**:
```python
# test_data_flows.py (BEFORE - hardcoded)
def test_position_tensor_shape():
    env = VectorizedHamletEnv(num_agents=4, grid_size=8)
    assert env.positions.shape == (4, 2)  # ← Assumes 2D

# test_data_flows.py (AFTER - parameterized)
def test_position_tensor_shape(basic_env):
    num_agents = basic_env.num_agents
    position_dim = basic_env.substrate.position_dim
    assert basic_env.positions.shape == (num_agents, position_dim)
```

**Finding**: ALL position shape assertions must use `substrate.position_dim` instead of hardcoded `2`.

---

### Finding 5: Test Execution Speed

**Current Test Suite**: ~300 tests, ~45 seconds runtime (on CPU)

**Estimated Impact of Parameterization**:
- Add 2x tests (Grid2D + Aspatial parameterization) → ~600 tests
- Add property-based tests (~50 examples each) → ~200 additional test cases
- **Estimated new runtime**: ~90 seconds (2x slowdown)

**Mitigation**:
- Use `pytest-xdist` for parallel test execution: `pytest -n auto`
- Mark slow tests with `@pytest.mark.slow`
- Run property-based tests with fewer examples in CI: `@settings(max_examples=10)`

**Finding**: Test suite will double in size but remain under 2 minutes (acceptable).

---

## Test Organization Strategy

### Test File Structure

**Unit Tests** (fast, isolated):
```
tests/test_townlet/unit/
├── substrate/
│   ├── test_grid2d.py           # Grid2D substrate tests
│   ├── test_aspatial.py         # Aspatial substrate tests
│   └── test_substrate_config.py # Config loading tests
├── environment/
│   ├── test_observations.py     # Observation builder tests (PARAMETERIZE)
│   ├── test_meters.py           # Meter tests (already parameterized)
│   └── test_affordances.py      # Affordance tests (PARAMETERIZE)
└── ...
```

**Integration Tests** (slower, full system):
```
tests/test_townlet/integration/
├── test_episode_execution.py         # PARAMETERIZE (Grid2D + Aspatial)
├── test_training_loop.py             # PARAMETERIZE (Grid2D + Aspatial)
├── test_checkpointing.py             # Version 3 checkpoint format
└── test_substrate_migration.py       # NEW: Behavioral equivalence tests
```

**Property-Based Tests** (randomized):
```
tests/test_townlet/properties/
├── test_substrate_properties.py      # NEW: Substrate contract properties
├── test_environment_properties.py    # PARAMETERIZE (existing)
└── test_exploration_properties.py    # PARAMETERIZE (existing)
```

---

## Performance Benchmarking

**Benchmark 1: Environment Reset Speed**
```python
def test_benchmark_reset_speed_grid2d(benchmark, basic_env):
    """Benchmark reset() with Grid2D substrate."""
    benchmark(basic_env.reset)
    # Target: <1ms per reset

def test_benchmark_reset_speed_aspatial(benchmark, aspatial_env):
    """Benchmark reset() with Aspatial substrate."""
    benchmark(aspatial_env.reset)
    # Expected: Faster than Grid2D (no position randomization)
```

**Benchmark 2: Step Throughput**
```python
def test_benchmark_step_throughput_grid2d(benchmark, basic_env):
    """Benchmark step() throughput with Grid2D substrate."""
    basic_env.reset()
    actions = torch.zeros(basic_env.num_agents, dtype=torch.long, device=basic_env.device)

    benchmark(basic_env.step, actions, depletion_multiplier=1.0)
    # Target: >1000 steps/sec on GPU, >100 steps/sec on CPU
```

**Benchmark 3: Observation Builder Speed**
```python
def test_benchmark_obs_builder_grid2d(benchmark, basic_env):
    """Benchmark observation construction with Grid2D substrate."""
    benchmark(basic_env.observation_builder.build_observations,
              basic_env.positions, basic_env.meters, basic_env.affordances,
              basic_env.time_of_day, basic_env.interaction_progress,
              basic_env.interaction_ticks, basic_env.step_counts)
    # Target: <0.5ms per observation batch
```

**Acceptance Criteria**:
- Grid2D performance: Within 5% of legacy hardcoded performance
- Aspatial performance: At least 10% faster than Grid2D (no position ops)

---

## Test Coverage Goals

**Coverage Targets**:
- Unit tests: >90% line coverage
- Integration tests: >85% line coverage
- Overall: >88% line coverage (maintain current level)

**Critical Paths** (must have 100% coverage):
- `substrate/base.py`: Abstract interface
- `substrate/grid2d.py`: Position validation, movement, distance
- `substrate/aspatial.py`: No-op implementations
- `substrate/config.py`: Config loading and validation
- `environment/vectorized_env.py`: Substrate integration

**Coverage Verification**:
```bash
cd /home/john/hamlet
uv run pytest --cov=townlet --cov-report=html --cov-report=term-missing
# Open htmlcov/index.html to view coverage report
```

---

## Testing Strategy Matrix

| Test Type | Grid2D | Aspatial | 3D Grid (Future) |
|-----------|--------|----------|------------------|
| **Unit Tests** | ✅ Yes | ✅ Yes | 🎯 Future |
| **Integration Tests** | ✅ Yes | ✅ Yes | 🎯 Future |
| **Property-Based** | ✅ Yes | ✅ Yes | 🎯 Future |
| **Regression** | ✅ Yes (vs legacy) | ❌ N/A (new) | 🎯 Future |
| **Performance** | ✅ Yes (vs legacy) | ✅ Yes (vs Grid2D) | 🎯 Future |

---

## Risk Assessment

### Risk 1: Test Suite Runtime Doubles

**Scenario**: Parameterized tests (Grid2D + Aspatial) double test count → 2x runtime

**Likelihood**: HIGH (inevitable with parameterization)

**Mitigation**:
1. Use `pytest-xdist` for parallel execution: `pytest -n auto`
2. Mark slow tests: `@pytest.mark.slow`
3. Run slow tests only in CI, not locally
4. Use smaller grids in tests (3×3 instead of 8×8 where possible)

**Acceptance**: 2x runtime is acceptable if tests remain under 2 minutes

---

### Risk 2: Aspatial Tests Reveal Design Flaws

**Scenario**: Aspatial tests fail due to hardcoded spatial assumptions in core logic

**Likelihood**: MEDIUM (Phase 4-6 should have fixed most issues)

**Mitigation**:
1. Run aspatial tests early in Phase 8 (catch issues fast)
2. Add debug logging to identify where spatial assumptions leak
3. Use property-based tests to find edge cases

**Contingency**: If major issues found, create TASK-002B for fixes

---

### Risk 3: Property-Based Tests Find Unexpected Bugs

**Scenario**: Hypothesis generates edge case that breaks substrate abstraction

**Likelihood**: MEDIUM (property-based tests are designed to find bugs)

**Mitigation**:
1. Start with small search space (e.g., `max_examples=10`)
2. Gradually increase examples as bugs fixed
3. Add regression test for each bug found

**Acceptance**: Finding bugs is GOOD - that's the point of testing!

---

## Open Questions

### Q1: Should we test all substrate combinations?

**Context**: Future may have Grid2D + Aspatial + Graph3D + Toroidal, etc.

**Options**:
1. Test all combinations exhaustively (slow but thorough)
2. Test each substrate independently (fast but may miss interaction bugs)
3. Test "representative" combinations (e.g., Grid2D + Aspatial only)

**Recommendation**: Option 3 for Phase 8 (Grid2D + Aspatial). Add more substrates in future phases as they're implemented.

---

### Q2: Should property-based tests run in CI?

**Context**: Property-based tests can be slow (100+ examples)

**Options**:
1. Run full property tests in CI (thorough but slow)
2. Run reduced property tests in CI (fast but less coverage)
3. Run property tests only on release branches (compromise)

**Recommendation**: Option 2 (10 examples in CI, 100 examples manually before release)

---

### Q3: Should we keep legacy checkpoint loading?

**Context**: Version 2 checkpoints are incompatible with Version 3 (substrate metadata missing)

**Options**:
1. Support loading Version 2 (backward compatible but complex)
2. Reject Version 2 with clear error (breaking change but simpler)
3. Provide migration script (one-time conversion)

**Recommendation**: Option 2 (reject Version 2 with clear error message). Breaking changes authorized per TASK-002A scope.

---

## Conclusion

Phase 8 is **comprehensive but mechanical**:

1. **Scope is clear**: 40 test files need updates (mostly mechanical parameterization)
2. **Patterns are established**: Use fixtures + parameterization for multi-substrate tests
3. **New tests are focused**: Property-based tests for substrate contracts, regression tests for backward compatibility
4. **Performance is monitored**: Benchmarks ensure no slowdown from abstraction

**Critical Success Factors**:
- Create aspatial config pack (prerequisite for aspatial tests)
- Update conftest.py with substrate fixtures
- Parameterize all hardcoded grid size / position dim assumptions
- Verify Version 3 checkpoint format
- Maintain >88% test coverage

**Estimated Effort**: 6-8 hours (on high end due to 40 files requiring updates)

**Readiness**: Ready for implementation. All patterns documented, fixtures designed, test structure defined.
