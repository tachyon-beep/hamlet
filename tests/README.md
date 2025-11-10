# Townlet Test Suite Documentation

This document describes the organization, fixtures, utilities, and best practices for the Townlet test suite.

## Table of Contents

- [Test Organization](#test-organization)
- [Available Fixtures](#available-fixtures)
- [Test Utilities](#test-utilities)
- [Writing Tests](#writing-tests)
- [Running Tests](#running-tests)
- [Test Markers](#test-markers)

---

## Test Organization

The test suite is organized by component area:

```
tests/
├── test_townlet/
│   ├── conftest.py               # Shared fixtures
│   ├── fixtures/                 # Frozen test data
│   ├── utils/                    # Test utilities (builders, assertions)
│   ├── unit/                     # Unit tests (isolated components)
│   │   ├── substrate/
│   │   ├── environment/
│   │   ├── agent/
│   │   ├── training/
│   │   └── ...
│   ├── integration/              # Integration tests (multiple components)
│   │   ├── test_training_loop.py
│   │   ├── test_custom_actions.py
│   │   └── ...
│   ├── slow/                     # Opt-in end-to-end (smoke + slow variants)
│   │   └── test_training_levels.py
│   └── properties/               # Property-based tests (hypothesis)
│       ├── test_substrate_properties.py
│       ├── test_replay_buffer_properties.py
│       └── ...
└── test_integration/             # Legacy directory (kept for backwards compat)
```

**Principles:**
- **Unit tests**: Test single components in isolation
- **Integration tests**: Test interactions between components
- **Property-based tests**: Test invariants across random inputs
- **End-to-end tests**: Live in `tests/test_townlet/slow/`; smoke variants (≤10 episodes) run by default, while the 200-episode suites are marked with `@pytest.mark.slow`.

---

## Available Fixtures

Fixtures are defined in `tests/test_townlet/conftest.py` and automatically available to all tests.

### Configuration Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `mock_config_path` | session | Path to frozen mock config (locked, don't modify) |
| `test_config_pack_path` | session | Path to test config pack (configs/test) |
| `temp_config_pack` | function | Writable copy of test config pack |
| `mock_config` | function | Loaded mock config as dict |

**Example:**
```python
def test_something(test_config_pack_path):
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        config_pack_path=test_config_pack_path,
        device=torch.device("cpu"),
    )
```

### Device Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `device` | session | CUDA if available, else CPU |
| `cpu_device` | function | Force CPU (for deterministic tests) |

**Example:**
```python
def test_something(device):
    tensor = torch.zeros(10, device=device)
```

### Environment Fixtures

| Fixture | Description | Grid Size | Observability | Agents |
|---------|-------------|-----------|---------------|--------|
| `basic_env` | Standard full observability | 8×8 | Full | 1 |
| `pomdp_env` | Partial observability (LSTM) | 8×8 | 5×5 window | 1 |
| `temporal_env` | Temporal mechanics enabled | 8×8 | Full | 1 |
| `multi_agent_env` | Multiple agents | 8×8 | Full | 4 |
| `grid2d_3x3_env` | Small grid for fast tests | 3×3 | Full | 1 |
| `grid2d_8x8_env` | Explicit 8×8 grid | 8×8 | Full | 1 |
| `aspatial_env` | No spatial substrate | N/A | N/A | 1 |

**Example:**
```python
def test_environment_reset(basic_env):
    obs = basic_env.reset()
    assert obs.shape == (1, basic_env.observation_builder.get_observation_dim())
```

### Network Fixtures

| Fixture | Network Type | Use Case |
|---------|-------------|----------|
| `simple_qnetwork` | MLP | Full observability |
| `recurrent_qnetwork` | LSTM | Partial observability |

**Example:**
```python
def test_network_forward(simple_qnetwork, basic_env):
    obs = basic_env.reset()
    q_values = simple_qnetwork(obs)
    assert q_values.shape == (1, basic_env.action_dim)
```

### Training Component Fixtures

| Fixture | Description |
|---------|-------------|
| `replay_buffer` | ReplayBuffer (capacity=1000) |
| `adversarial_curriculum` | AdversarialCurriculum (test parameters) |
| `static_curriculum` | StaticCurriculum (fixed difficulty) |
| `epsilon_greedy_exploration` | Epsilon-greedy exploration |
| `adaptive_intrinsic_exploration` | RND + adaptive annealing |
| `vectorized_population` | VectorizedPopulation (full training setup) |

**Example:**
```python
def test_replay_buffer_sample(replay_buffer):
    # Add transitions
    for i in range(100):
        replay_buffer.add(obs, action, reward, next_obs, done)

    # Sample batch
    batch = replay_buffer.sample(batch_size=32)
    assert batch["observations"].shape == (32, obs_dim)
```

### Variable Meter Fixtures (TASK-001)

| Fixture | Meters | Description |
|---------|--------|-------------|
| `task001_config_4meter` | 4 | Config pack with 4 meters |
| `task001_config_12meter` | 12 | Config pack with 12 meters |
| `task001_env_4meter` | 4 | Environment with 4 meters |
| `task001_env_4meter_pomdp` | 4 | POMDP environment with 4 meters |
| `task001_env_12meter` | 12 | Environment with 12 meters |

**Example:**
```python
def test_variable_meters(task001_env_4meter):
    assert task001_env_4meter.meter_count == 4
    obs = task001_env_4meter.reset()
    # Observation includes 4 meters, not 8
```

---

## Test Utilities

The `tests/test_townlet/utils/` module provides reusable builders and assertions.

### Builder Functions (`utils/builders.py`)

**Substrate Builders:**

```python
from tests.test_townlet.utils.builders import make_grid2d_substrate, make_grid3d_substrate

def test_substrate():
    # Create Grid2D with defaults (8×8, clamp, manhattan)
    substrate = make_grid2d_substrate()

    # Create small Grid2D for fast tests
    substrate = make_grid2d_substrate(width=3, height=3)

    # Create Grid3D
    substrate = make_grid3d_substrate(width=5, height=5, depth=3)
```

**Bars Config Builders:**

```python
from tests.test_townlet.utils.builders import make_bars_config, make_standard_8meter_config

def test_bars_config():
    # Create minimal config with 4 meters (meter_0, meter_1, meter_2, meter_3)
    config = make_bars_config(meter_count=4)

    # Create standard 8-meter config (energy, health, satiation, etc.)
    config = make_standard_8meter_config()
```

**Position Builders:**

```python
from tests.test_townlet.utils.builders import make_positions

def test_positions():
    # Create positions for 4 agents in 2D grid (all at origin)
    positions = make_positions(num_agents=4, position_dim=2, value=0)
    # Shape: (4, 2), all values = 0
```

### Assertion Functions (`utils/assertions.py`)

**Observation Assertions:**

```python
from tests.test_townlet.utils.assertions import assert_valid_observation

def test_observation(basic_env):
    obs = basic_env.reset()
    assert_valid_observation(basic_env, obs)
    # Validates: shape, dtype, no NaN/Inf
```

**Action Mask Assertions:**

```python
from tests.test_townlet.utils.assertions import assert_valid_action_mask

def test_action_mask(basic_env):
    mask = basic_env.get_action_masks()
    assert_valid_action_mask(basic_env, mask)
    # Validates: shape, dtype, at least one valid action per agent
```

**Meter Assertions:**

```python
from tests.test_townlet.utils.assertions import assert_meters_in_range

def test_meters(basic_env):
    basic_env.step(actions)
    assert_meters_in_range(basic_env)
    # Validates: meters in [0, 1], no NaN/Inf
```

**Position Assertions:**

```python
from tests.test_townlet.utils.assertions import assert_positions_in_bounds

def test_positions(basic_env):
    positions = basic_env.substrate.positions
    assert_positions_in_bounds(positions, width=8, height=8)
    # Validates: positions within grid bounds
```

**Reward/Done Assertions:**

```python
from tests.test_townlet.utils.assertions import assert_valid_rewards, assert_valid_dones

def test_step(basic_env):
    _, rewards, dones, _, _ = basic_env.step(actions)
    assert_valid_rewards(rewards, basic_env.num_agents)
    assert_valid_dones(dones, basic_env.num_agents)
```

---

## Writing Tests

### Test Structure

**Good test structure:**
```python
def test_feature_behavior():
    """Test that feature X behaves correctly under condition Y."""
    # Arrange: Set up test objects
    substrate = make_grid2d_substrate(width=3, height=3)
    positions = make_positions(num_agents=1, position_dim=2)

    # Act: Perform action
    new_positions = substrate.apply_movement(positions, deltas)

    # Assert: Verify results
    assert new_positions[0, 0] == 1  # Specific assertion
    assert_positions_in_bounds(new_positions, width=3, height=3)  # Helper
```

### Use Fixtures for Common Setup

**Bad: Create objects in every test**
```python
def test_something():
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, ...)
    obs = env.reset()
    assert obs.shape == (1, env.observation_builder.get_observation_dim())

def test_something_else():
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, ...)  # Duplicate!
    obs = env.reset()
    assert obs.dtype == torch.float32
```

**Good: Use fixtures**
```python
def test_something(basic_env):
    obs = basic_env.reset()
    assert obs.shape == (1, basic_env.observation_builder.get_observation_dim())

def test_something_else(basic_env):
    obs = basic_env.reset()
    assert obs.dtype == torch.float32
```

### Use Builders for Custom Objects

**Bad: Verbose construction**
```python
def test_custom_substrate():
    substrate = Grid2DSubstrate(
        width=5,
        height=5,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    # 5 lines of boilerplate
```

**Good: Use builders**
```python
from tests.test_townlet.utils.builders import make_grid2d_substrate

def test_custom_substrate():
    substrate = make_grid2d_substrate(width=5, height=5)
    # 1 line, same result
```

### Use Assertion Helpers

**Bad: Manual validation**
```python
def test_observation(basic_env):
    obs = basic_env.reset()
    assert obs.shape == (1, basic_env.observation_builder.get_observation_dim())
    assert obs.dtype == torch.float32
    assert not torch.isnan(obs).any()
    assert not torch.isinf(obs).any()
    # 4 assertions, easy to forget one
```

**Good: Use assertion helpers**
```python
from tests.test_townlet.utils.assertions import assert_valid_observation

def test_observation(basic_env):
    obs = basic_env.reset()
    assert_valid_observation(basic_env, obs)
    # 1 line, all validations included
```

---

## Running Tests

### Run All Tests
```bash
uv run pytest
```

### Run Specific Test File
```bash
uv run pytest tests/test_townlet/unit/substrate/test_grid2d.py
```

### Run Specific Test Function
```bash
uv run pytest tests/test_townlet/unit/substrate/test_grid2d.py::test_grid2d_relative_encoding_dimensions
```

### Run Tests by Marker
```bash
# Run only integration tests
uv run pytest -m integration

# Run only slow tests
uv run pytest -m slow

# Skip slow tests
uv run pytest -m "not slow"

# Run only GPU tests (skipped if no CUDA)
uv run pytest -m gpu
```

### Training-Level Pipelines
- Smoke coverage (≤10 episodes, default marker): `uv run pytest tests/test_townlet/slow/test_training_levels.py -k smoke`
- Full curriculum (200 episodes, opt-in): `uv run pytest -m slow tests/test_townlet/slow/test_training_levels.py`
- Smoke configs reside in `configs/test/training_level_{1,2,3}_smoke.yaml`; their slow counterparts keep the longer `training_level_{1,2,3}.yaml` settings.

### Slow Integration Suites

Several integration files (e.g., `test_training_loop.py`, `test_recurrent_networks.py`, `test_temporal_mechanics.py`) are marked `@pytest.mark.slow` because they require dozens of episodes or long sequential replay runs. These suites are skipped by default; opt in with:

```bash
uv run pytest -m slow tests/test_townlet/integration
```

CI only runs these periodically, so include the same marker when validating major changes to training/runtime orchestration.

### Run with Coverage
```bash
uv run pytest --cov=townlet --cov-report=term-missing
```

### Run with Verbose Output
```bash
uv run pytest -v
```

### Run with Output Capture Disabled (see prints)
```bash
uv run pytest -s
```

---

## Test Markers

Custom markers are defined in `conftest.py` and can be used to categorize tests:

| Marker | Description | Usage |
|--------|-------------|-------|
| `@pytest.mark.slow` | Long-running test (opt-in) | `pytest -m slow` |
| `@pytest.mark.gpu` | Requires GPU | Skipped automatically if no CUDA |
| `@pytest.mark.integration` | Integration test | `pytest -m integration` |
| `@pytest.mark.e2e` | End-to-end test | `pytest -m e2e` |

**Example:**
```python
import pytest

@pytest.mark.slow
def test_full_training_loop():
    # This test takes 30+ seconds
    pass

@pytest.mark.gpu
def test_gpu_training():
    # This test requires CUDA
    assert torch.cuda.is_available()
```

---

## Best Practices

### DO:
- ✅ Use fixtures for common setup (environments, networks, curricula)
- ✅ Use builder functions for custom test objects
- ✅ Use assertion helpers to validate invariants
- ✅ Write descriptive test names (`test_feature_behavior_under_condition`)
- ✅ Add docstrings explaining what the test validates
- ✅ Use markers to categorize tests (`@pytest.mark.slow`, `@pytest.mark.gpu`)
- ✅ Test edge cases (empty inputs, boundary conditions)
- ✅ Test error handling (invalid inputs should raise errors)

### DON'T:
- ❌ Create environments manually in every test (use fixtures)
- ❌ Repeat validation logic (use assertion helpers)
- ❌ Test multiple unrelated behaviors in one test
- ❌ Use hardcoded constants (use `env.action_dim`, not `10`)
- ❌ Modify fixtures in place (create copies if needed)
- ❌ Test implementation details (test observable behavior)

---

## Example Test

Here's a complete example combining fixtures, builders, and assertions:

```python
import pytest
import torch

from tests.test_townlet.utils.builders import make_grid2d_substrate, make_positions
from tests.test_townlet.utils.assertions import (
    assert_valid_observation,
    assert_meters_in_range,
    assert_valid_action_mask,
)


def test_environment_reset_and_step(basic_env):
    """Test environment reset and step maintain invariants."""
    # Reset environment
    obs = basic_env.reset()

    # Validate observation
    assert_valid_observation(basic_env, obs)
    assert_meters_in_range(basic_env)

    # Take random actions
    actions = torch.randint(0, basic_env.action_dim, (basic_env.num_agents,))
    next_obs, rewards, dones, _, _ = basic_env.step(actions)

    # Validate step outputs
    assert_valid_observation(basic_env, next_obs)
    assert_meters_in_range(basic_env)
    assert rewards.shape == (basic_env.num_agents,)
    assert dones.shape == (basic_env.num_agents,)


def test_custom_substrate_movement():
    """Test custom substrate boundary behavior."""
    # Use builder for custom substrate
    substrate = make_grid2d_substrate(width=3, height=3, boundary="clamp")

    # Use builder for positions
    positions = make_positions(num_agents=1, position_dim=2, value=0)

    # Move right (delta = [1, 0])
    deltas = torch.tensor([[1, 0]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Validate result
    assert new_positions[0, 0] == 1  # X moved right
    assert new_positions[0, 1] == 0  # Y unchanged


@pytest.mark.slow
def test_full_training_episode(vectorized_population):
    """Test full training episode converges (slow test)."""
    for episode in range(100):
        vectorized_population.run_episode()

    # Check that training happened
    assert vectorized_population.episode_count == 100
```

---

## Questions?

If you need additional fixtures, builders, or assertion helpers, please add them to the appropriate module:
- New fixtures → `tests/test_townlet/conftest.py`
- New builders → `tests/test_townlet/utils/builders.py`
- New assertions → `tests/test_townlet/utils/assertions.py`

Keep this README updated when adding new utilities!
