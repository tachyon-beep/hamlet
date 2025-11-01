# Integration Tests - Training Levels

**Location:** `tests/test_integration/test_training_levels.py`  
**Purpose:** End-to-end validation that each training level works completely

---

## Quick Start

```bash
# Run all integration tests (Level 1, 2, 3 - takes ~20 minutes total)
uv run pytest tests/test_integration/ -v

# Run only config validation (fast - <1 second)
uv run pytest tests/test_integration/test_training_levels.py::test_all_configs_valid -v

# Run specific level
uv run pytest tests/test_integration/test_training_levels.py::test_level_1_full_observability_integration -v
uv run pytest tests/test_integration/test_training_levels.py::test_level_2_pomdp_integration -v
uv run pytest tests/test_integration/test_training_levels.py::test_level_3_temporal_integration -v

# Run checkpoint resume test
uv run pytest tests/test_integration/test_training_levels.py::test_checkpoint_resume -v
```

---

## Test Suite

### 1. Config Validation Test ⚡ Fast (<1 second)

```bash
uv run pytest tests/test_integration/ -k "test_all_configs_valid" -v
```

**Validates:**
- All config files are valid YAML
- Required sections present (environment, population, curriculum, exploration, training)
- Required fields present in each section

**Configs tested:**
- `level_1_1_integration_test.yaml`
- `level_2_1_integration_test.yaml`
- `level_3_1_integration_test.yaml`
- `level_1_full_observability.yaml`
- `level_2_pomdp.yaml`
- `level_3_temporal.yaml`

---

### 2. Level 1 Integration Test 🟢 Moderate (~5 minutes)

```bash
uv run pytest tests/test_integration/ -k "test_level_1" -v
```

**Validates:**
- SimpleQNetwork trains without errors
- Full observability works
- Checkpoints can be saved and loaded
- Database logging works
- Agent shows learning progress

**Config:** `configs/level_1_1_integration_test.yaml`
- 200 episodes (~5 minutes)
- Full observability
- MLP Q-Network
- Reduced buffer (2000)
- Max 200 steps/episode

**Expected outcome:**
- Completes 200 episodes
- Saves 2 checkpoints (ep 100, ep 200)
- Average survival improves over time
- No crashes or errors

---

### 3. Level 2 Integration Test 🟡 Moderate (~8 minutes)

```bash
uv run pytest tests/test_integration/ -k "test_level_2" -v
```

**Validates:**
- RecurrentSpatialQNetwork trains without errors
- LSTM hidden state management works
- Target network for LSTM works (ACTION #9)
- Partial observability works (5×5 window)
- Sequential replay buffer works

**Config:** `configs/level_2_1_integration_test.yaml`
- 200 episodes (~8 minutes)
- Partial observability (5×5 window)
- LSTM with target network
- Reduced buffer (2000)
- Max 200 steps/episode

**Expected outcome:**
- Completes 200 episodes
- Saves 2 checkpoints
- Target network present in checkpoint
- LSTM memory used correctly
- No crashes or errors

---

### 4. Level 3 Integration Test 🟡 Moderate (~8 minutes)

```bash
uv run pytest tests/test_integration/ -k "test_level_3" -v
```

**Validates:**
- Temporal mechanics work (24-tick day/night cycles)
- Multi-tick interactions work (progressive benefits)
- Operating hours masking works
- LSTM learns temporal patterns
- Time-of-day observation included

**Config:** `configs/level_3_1_integration_test.yaml`
- 200 episodes (~8 minutes)
- Partial observability + temporal mechanics
- LSTM with time awareness
- Reduced buffer (2000)
- Max 200 steps/episode

**Expected outcome:**
- Completes 200 episodes
- Temporal mechanics don't cause errors
- Multi-tick interactions work
- No crashes or errors

---

### 5. Checkpoint Resume Test 🟢 Moderate (~5 minutes)

```bash
uv run pytest tests/test_integration/ -k "test_checkpoint_resume" -v
```

**Validates:**
- Training can be paused and resumed
- Checkpoints save Q-network and optimizer state
- Checkpoint structure is correct

**Process:**
1. Train for 100 episodes
2. Save checkpoint
3. Verify checkpoint can be loaded
4. Verify checkpoint has required fields

---

## Lite Configs vs Full Configs

### Lite Configs (for integration tests)

**Purpose:** Fast validation that systems work end-to-end

**Characteristics:**
- 200 episodes (~5-8 minutes)
- Reduced replay buffer (2000 vs 10000)
- Reduced max steps (200 vs 500)
- Reduced min_steps_at_stage (50 vs 1000)
- Reduced survival_window (50 vs 100)
- Higher epsilon_min (0.1 vs 0.01)

**Files:**
- `configs/level_1_1_integration_test.yaml`
- `configs/level_2_1_integration_test.yaml`
- `configs/level_3_1_integration_test.yaml`

### Full Configs (for production training)

**Purpose:** Achieve best performance over extended training

**Characteristics:**
- 5000-10000 episodes (hours/days)
- Full replay buffer (10000)
- Full episode length (500 steps)
- Full curriculum parameters
- Full exploration annealing

**Files:**
- `configs/level_1_full_observability.yaml` (5000 episodes)
- `configs/level_2_pomdp.yaml` (10000 episodes)
- `configs/level_3_temporal.yaml` (10000 episodes)

---

## Running Integration Tests in CI/CD

### Quick Validation (for every commit)

```bash
# Just config validation (<1 second)
uv run pytest tests/test_integration/ -k "test_all_configs_valid" -v
```

### Full Validation (nightly or pre-release)

```bash
# All integration tests (~20 minutes total)
uv run pytest tests/test_integration/ -v -m "integration"
```

### Skip Slow Tests

```bash
# Run only fast unit tests (skip integration)
uv run pytest tests/ -v -m "not slow and not integration"
```

---

## Markers

Integration tests use pytest markers:

```python
@pytest.mark.integration  # Marks as integration test
@pytest.mark.slow         # Marks as slow test (>1 minute)
```

**Filter by markers:**
```bash
# Only integration tests
uv run pytest -m "integration" -v

# Only slow tests
uv run pytest -m "slow" -v

# Skip slow tests
uv run pytest -m "not slow" -v

# Skip integration tests
uv run pytest -m "not integration" -v
```

---

## Test Output Example

```
tests/test_integration/test_training_levels.py::test_all_configs_valid PASSED
✅ configs/level_1_1_integration_test.yaml is valid
✅ configs/level_2_1_integration_test.yaml is valid
✅ configs/level_3_1_integration_test.yaml is valid
✅ configs/level_1_full_observability.yaml is valid
✅ configs/level_2_pomdp.yaml is valid
✅ configs/level_3_temporal.yaml is valid

✅ All Config Validation Test PASSED

tests/test_integration/test_training_levels.py::test_level_1_full_observability_integration PASSED

✅ Level 1 Integration Test PASSED
   Episodes: 200
   Avg survival (last 20): 42.3 steps
   Checkpoints saved: 2

tests/test_integration/test_training_levels.py::test_level_2_pomdp_integration PASSED
   ✅ Target network present (ACTION #9 validated)

✅ Level 2 Integration Test PASSED
   Episodes: 200
   Avg survival (last 20): 38.1 steps
   Checkpoints saved: 2

tests/test_integration/test_training_levels.py::test_level_3_temporal_integration PASSED

✅ Level 3 Integration Test PASSED
   Episodes: 200
   Avg survival (last 20): 36.7 steps
   Checkpoints saved: 2

========================= 5 passed in 21.45s =========================
```

---

## Troubleshooting

### Tests fail with "Config file not found"

Make sure you're running from project root:
```bash
cd /home/john/hamlet
uv run pytest tests/test_integration/ -v
```

### Tests fail with CUDA out of memory

Edit lite configs to use CPU:
```yaml
training:
  device: cpu  # Change from cuda
```

### Tests timeout

Integration tests are marked `@pytest.mark.slow`. Default timeout is sufficient for lite configs (200 episodes).

If needed, increase timeout:
```bash
uv run pytest tests/test_integration/ -v --timeout=600  # 10 minutes per test
```

---

## Adding New Integration Tests

When adding a new level:

1. **Create lite config:** `configs/level_X_Y_integration_test.yaml`
   - 200 episodes
   - Reduced parameters (buffer, steps, thresholds)

2. **Add test function:** `tests/test_integration/test_training_levels.py`
   ```python
   @pytest.mark.integration
   @pytest.mark.slow
   def test_level_X_integration(temp_run_dir):
       config_path = Path("configs/level_X_Y_integration_test.yaml")
       # ... run pipeline
   ```

3. **Add to config validation test:**
   ```python
   config_files = [
       # ... existing configs
       "configs/level_X_Y_integration_test.yaml",
   ]
   ```

4. **Update this documentation**

---

## Summary

**Fast validation (<1 sec):**
```bash
uv run pytest tests/test_integration/ -k "test_all_configs_valid"
```

**Full integration (~20 min):**
```bash
uv run pytest tests/test_integration/ -v
```

**Best practice:**
- Run config validation on every commit
- Run full integration tests before merging major changes
- Use lite configs for CI/CD, full configs for production training
