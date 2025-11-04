# Townlet Test Suite

**Status**: ✅ Fully Refactored (November 2025)
**Total Tests**: 560 tests (426 unit + 114 integration + 20 properties)
**Coverage**: 67% of townlet codebase
**Runtime**: ~66 seconds for complete suite

---

## Overview

This test suite provides comprehensive coverage of the HAMLET Townlet deep RL training system using a three-tier architecture:

1. **Unit Tests** - Component-level testing with isolation
2. **Integration Tests** - Cross-component interactions and data flows
3. **Property-Based Tests** - Hypothesis fuzzing for edge cases

### Design Philosophy

- **Behavioral assertions** over exact values (tests survive config changes)
- **Real components** in integration tests (minimal mocking)
- **CPU device** for determinism (eliminates GPU randomness)
- **Fixture composition** for reusability (see `conftest.py`)

---

## Directory Structure

```
tests/test_townlet/
├── conftest.py                    # Shared fixtures for all tests
├── fixtures/
│   └── mock_config.yaml          # Frozen config for deterministic tests
│
├── unit/                          # Unit tests (426 tests)
│   ├── agent/
│   │   └── test_networks.py           # SimpleQNetwork, RecurrentSpatialQNetwork (19 tests)
│   ├── environment/
│   │   ├── test_action_masking.py     # Action masking logic (37 tests)
│   │   ├── test_observations.py       # Observation builder (32 tests)
│   │   ├── test_affordances.py        # All 14 affordances (39 tests)
│   │   └── test_meters.py             # Meter dynamics, cascades (41 tests)
│   ├── population/
│   │   └── test_action_selection.py   # Q-value selection, epsilon-greedy (12 tests)
│   ├── curriculum/
│   │   └── test_curriculums.py        # Adversarial and static curriculum (33 tests)
│   ├── exploration/
│   │   └── test_exploration_strategies.py  # Epsilon-greedy, RND, adaptive (64 tests)
│   ├── training/
│   │   └── test_replay_buffers.py     # ReplayBuffer, SequentialReplayBuffer (52 tests)
│   ├── recording/                     # Episode recording unit tests (45 tests)
│   │   ├── test_data_structures.py    # Serialization, deserialization (18 tests)
│   │   ├── test_database.py           # Database operations (13 tests)
│   │   └── test_criteria.py           # Recording criteria logic (14 tests)
│   └── test_configuration.py      # Config loading, validation (52 tests)
│
├── integration/                   # Integration tests (114 tests)
│   ├── test_checkpointing.py          # Save/load state, round-trip (15 tests)
│   ├── test_curriculum_signal_purity.py   # Curriculum sees survival time, not rewards (11 tests)
│   ├── test_runner_integration.py     # DemoRunner orchestration, database logging (5 tests)
│   ├── test_temporal_mechanics.py     # Time-based affordances, operating hours (10 tests, 5 XFAIL)
│   ├── test_recurrent_networks.py     # LSTM hidden state management (8 tests, 5 XFAIL)
│   ├── test_intrinsic_exploration.py  # RND + adaptive annealing (6 tests)
│   ├── test_episode_execution.py      # Full episode lifecycle (6 tests)
│   ├── test_training_loop.py          # Multi-episode training, masked loss (8 tests, 5 XFAIL)
│   ├── test_data_flows.py             # Observation, reward, action pipelines (8 tests)
│   ├── test_recording_recorder.py     # Episode recorder integration (12 tests)
│   ├── test_recording_playback.py     # Playback system integration (10 tests)
│   ├── test_recording_replay_manager.py  # Replay management (13 tests)
│   └── test_recording_video_export.py # Video rendering pipeline (9 tests, 1 XFAIL)
│
└── properties/                    # Property-based tests (20 properties)
    ├── test_environment_properties.py  # Invariants: grid bounds, meter bounds (6 properties)
    ├── test_exploration_properties.py  # Epsilon decay, action selection (8 properties)
    └── test_replay_buffer_properties.py  # Capacity, FIFO, sampling (6 properties)
```

---

## Running Tests

### Run All Tests

```bash
pytest tests/test_townlet/
```

### Run by Category

```bash
# Unit tests only (fast, ~30s)
pytest tests/test_townlet/unit/ -v

# Integration tests only (~25s)
pytest tests/test_townlet/integration/ -v

# Property-based tests only (~5s)
pytest tests/test_townlet/properties/ -v
```

### Run Specific Component

```bash
# Environment tests
pytest tests/test_townlet/unit/environment/ -v

# Checkpointing integration
pytest tests/test_townlet/integration/test_checkpointing.py -v

# Replay buffer properties
pytest tests/test_townlet/properties/test_replay_buffer_properties.py -v
```

### Skip Expected Failures

```bash
# Skip XFAIL tests (temporal mechanics not fully implemented)
pytest tests/test_townlet/ --runxfail=False -v
```

### Coverage Report

```bash
# Terminal report
pytest tests/test_townlet/ --cov=townlet --cov-report=term-missing

# HTML report (opens in browser)
pytest tests/test_townlet/ --cov=townlet --cov-report=html
open htmlcov/index.html
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/test_townlet/ -n auto
```

---

## Shared Fixtures (conftest.py)

### Configuration Fixtures

- `mock_config_path` - Path to frozen mock config (for exact-value assertions)
- `test_config_pack_path` - Path to test config pack
- `temp_config_pack(mock_config)` - Temporary config directory for isolated tests
- `mock_config` - Loaded YAML configuration dict

### Device Fixtures

- `device` - CUDA if available, else CPU
- `cpu_device` - Force CPU (use for deterministic tests)

### Environment Fixtures

- `basic_env(cpu_device, test_config_pack_path)` - 1 agent, full obs, no temporal
- `pomdp_env(cpu_device, test_config_pack_path)` - 1 agent, 5×5 vision window
- `temporal_env(cpu_device, test_config_pack_path)` - 1 agent, full obs, temporal mechanics
- `multi_agent_env(cpu_device, test_config_pack_path)` - 4 agents, full obs

### Network Fixtures

- `simple_qnetwork(cpu_device)` - MLP Q-network for full observability
- `recurrent_qnetwork(cpu_device)` - CNN+LSTM Q-network for POMDP

### Training Component Fixtures

- `replay_buffer(cpu_device)` - Standard replay buffer
- `adversarial_curriculum(cpu_device)` - Adaptive curriculum
- `static_curriculum(cpu_device)` - Fixed difficulty curriculum
- `epsilon_greedy_exploration(cpu_device)` - Epsilon-greedy strategy
- `adaptive_intrinsic_exploration(cpu_device)` - RND + annealing
- `vectorized_population(basic_env, cpu_device, test_config_pack_path)` - Full population with training

### Utility Fixtures

- `sample_observations(cpu_device)` - Generate sample observation tensors
- `sample_actions(cpu_device)` - Generate sample action tensors

---

## Writing New Tests

### Unit Test Pattern

```python
import pytest
import torch

def test_component_behavior(cpu_device, test_config_pack_path):
    """Test specific behavior with CPU device for determinism."""
    # Arrange: Create component
    component = MyComponent(device=cpu_device)

    # Act: Perform operation
    result = component.do_something()

    # Assert: Verify behavior (not exact values)
    assert result > 0, "Result should be positive"
    assert result < 1, "Result should be less than 1"
```

### Integration Test Pattern

```python
def test_cross_component_interaction(cpu_device, test_config_pack_path):
    """Test data flow from ComponentA → ComponentB."""
    # Arrange: Create real components (no mocks)
    env = VectorizedHamletEnv(..., device=cpu_device)
    population = VectorizedPopulation(env=env, ...)

    # Act: Execute pipeline
    population.reset()
    state = population.step_population(env)

    # Assert: Verify data transformation
    assert state.observations.shape == (num_agents, obs_dim)
    assert state.rewards.shape == (num_agents,)
```

### Property-Based Test Pattern

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=4))
def test_universal_property(action):
    """Test property holds for all valid actions."""
    component = MyComponent()
    result = component.process(action)

    # Assert invariant that must always hold
    assert 0 <= result <= 1, "Result must be in [0, 1]"
```

---

## Best Practices

### 1. Always Use CPU Device for Determinism

```python
def test_movement(cpu_device, test_config_pack_path):
    env = VectorizedHamletEnv(..., device=cpu_device)  # ✅ Deterministic
    # NOT: device=torch.device("cuda")  # ❌ Non-deterministic
```

### 2. Control Agent Positions for Movement Tests

```python
def test_movement(cpu_device, test_config_pack_path):
    env = VectorizedHamletEnv(..., device=cpu_device)
    env.reset()
    env.positions[0] = torch.tensor([4, 4], device=cpu_device)  # ✅ Center, all moves valid
    # NOT: Use random spawn position  # ❌ May spawn at edge, blocking movement
```

### 3. Use Behavioral Assertions

```python
# ✅ Good: Behavioral
assert late_survival.mean() > early_survival.mean(), "Agents should improve"

# ❌ Bad: Exact value
assert late_survival.mean() == 123.45, "Must be exactly 123.45"
```

### 4. Leverage Fixture Composition

```python
@pytest.fixture
def trained_population(basic_env, vectorized_population):
    """Population after 10 episodes of training."""
    for _ in range(10):
        vectorized_population.step_population(basic_env)
    return vectorized_population

def test_uses_trained_population(trained_population):
    # Use composed fixture
    assert len(trained_population.replay_buffer) > 0
```

### 5. Use pytest.approx for Floating Point

```python
# ✅ Good: Tolerance
assert result == pytest.approx(0.5, rel=0.01)

# ❌ Bad: Exact float
assert result == 0.5000000001
```

---

## Pytest Markers

Tests are marked for selective execution:

```python
@pytest.mark.slow  # Mark tests that take >5 seconds
@pytest.mark.gpu   # Mark tests requiring CUDA
@pytest.mark.integration  # Mark integration tests
@pytest.mark.e2e   # Mark end-to-end tests (if any)
```

Run marked tests:

```bash
# Skip slow tests
pytest tests/test_townlet/ -m "not slow"

# Run only GPU tests
pytest tests/test_townlet/ -m gpu

# Run only integration tests
pytest tests/test_townlet/ -m integration
```

---

## Common Gotchas

### 1. CUDA Non-Determinism

**Problem**: Tests fail intermittently due to GPU randomness.
**Solution**: Use `cpu_device` fixture.

### 2. Random Agent Spawning

**Problem**: Movement tests fail when agents spawn at grid edges.
**Solution**: Force agent position to center after `env.reset()`.

### 3. Grid Size for Affordances

**Problem**: Small grids (3×3) can't fit 14 affordances.
**Solution**: Use minimum 5×5 grid for tests with multiple affordances.

### 4. Exact Value Assertions

**Problem**: Tests break when tuning hyperparameters.
**Solution**: Use behavioral assertions or `pytest.approx()`.

### 5. Mixing Unit and Integration Concerns

**Problem**: Environment tests mixed with population tests.
**Solution**: Unit tests isolate component, integration tests test interactions.

---

## Test Metrics (as of 2025-11-04)

### Test Count by Category

| Category | Tests | Files | Runtime | Status |
|----------|-------|-------|---------|--------|
| Unit | 426 | 15 | ~35s | ✅ All passing |
| Integration | 114 | 13 | ~26s | ✅ 99 passing, 14 XFAIL, 1 XPASS |
| Properties | 20 | 3 | ~5s | ✅ All passing |
| **Total** | **560** | **31** | **~66s** | **✅ 550 passing** |

### Coverage by Module

| Module | Statements | Coverage |
|--------|-----------|----------|
| agent/networks.py | 69 | 97% |
| environment/observation_builder.py | 77 | 100% |
| environment/meter_dynamics.py | 24 | 100% |
| environment/vectorized_env.py | 250 | 95% |
| exploration/epsilon_greedy.py | 25 | 100% |
| exploration/adaptive_intrinsic.py | 55 | 100% |
| training/replay_buffer.py | 100 | 99% |
| population/vectorized.py | 335 | 91% |
| recording/data_structures.py | 40 | 100% |
| recording/criteria.py | 105 | 96% |
| recording/replay.py | 83 | 87% |
| recording/video_renderer.py | 137 | 96% |
| **Overall** | **3687** | **67%** |

---

## Troubleshooting

### Tests Fail Intermittently

1. Check if using `cpu_device` fixture (not `device`)
2. Verify agent positions are controlled (not random)
3. Check for race conditions in multi-threaded tests

### Import Errors

```bash
# Ensure PYTHONPATH includes src/
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
pytest tests/test_townlet/
```

### Coverage Not Generated

```bash
# Install coverage plugin
uv pip install pytest-cov

# Run with coverage
pytest tests/test_townlet/ --cov=townlet
```

### Property Tests Fail

- Check Hypothesis example count (default 100)
- Increase timeout if needed: `@settings(deadline=1000)`  # 1 second
- Verify strategies generate valid inputs

---

## Resources

- **Methodology**: `docs/methods/TEST_REFACTORING_METHODOLOGY.md` - Test refactoring process
- **Research**: `tests/RESEARCH-INTEGRATION-TEST-STRATEGY.md` - Integration test strategy
- **Planning**: `tests/PLAN-INTEGRATION-TESTS.md` - Detailed task breakdown
- **Status**: `tests/test_townlet/TEST_REFACTORING_STATUS.md` - Progress tracking

---

## Migration Notes (for reference)

This test suite was created Nov 2025 by consolidating 57 unstructured test files into an organized architecture using the "parallel build + clean cutover" methodology:

1. **Tasks 1-10**: Unit tests (381 tests across 12 files)
2. **Tasks 11-13**: Integration tests (87 tests across 9 files)
3. **Task 14**: Property-based tests (20 properties across 3 files)
4. **Tasks 15-18**: Validation, documentation, cutover
5. **Post-cutover**: Recording test integration (72 tests from test_recording/ subdirectory)

**Key improvements**:
- ✅ Organized unit/integration/properties structure
- ✅ Shared fixtures eliminate duplication
- ✅ CPU device ensures determinism
- ✅ Behavioral assertions (not exact values)
- ✅ Comprehensive coverage (67% of codebase, up from 54%)
- ✅ Fast runtime (~66 seconds for 560 tests)
- ✅ Episode recording tests integrated (was separate subdirectory)

**Old test files**: Deleted after validation (commits: bc777c2, a77ad0e)

---

**For questions or issues, see**: `tests/test_townlet/TEST_REFACTORING_STATUS.md`
