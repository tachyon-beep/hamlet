# HAMLET Test Writing Guide

**For Contributors**: How to write clean, maintainable tests using the builders infrastructure.

**Last Updated**: 2025-11-07 (Sprint 13)

---

## Quick Start

### ❌ Old Way (Avoid)
```python
def test_something():
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
    # Test logic...
```

**Problems**: 20+ lines of boilerplate, hardcoded values, brittle to schema changes.

### ✅ New Way (Use Builders)
```python
from tests.test_townlet.utils.builders import make_test_bars_config

def test_something():
    bars_config = make_test_bars_config(num_meters=1)
    # Test logic...
```

**Benefits**: 1 line, uses canonical values, resilient to schema changes.

---

## The Builders Module

**Location**: `tests/test_townlet/builders.py`

**Purpose**: Single source of truth for test data construction.

### What's Available

| Builder | What It Creates | Default Behavior |
|---------|----------------|------------------|
| `make_test_meters()` | 8-meter tuple | `(1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)` |
| `make_test_bar()` | Single BarConfig | Energy meter, index=0 |
| `make_test_bars_config()` | Complete BarsConfig | 8 meters + terminal condition |
| `make_test_affordance()` | AffordanceConfig | Bed affordance, instant type |
| `make_test_episode_metadata()` | EpisodeMetadata | episode_id=100, survival_steps=10 |
| `make_test_recorded_step()` | RecordedStep | step=0, standard meters |
| `make_test_terminal_condition()` | TerminalCondition | Energy <= 0.0 death |

### Canonical Dimensions

```python
from tests.test_townlet.utils.builders import TestDimensions

# Use these instead of magic numbers!
obs_dim = TestDimensions.GRID2D_OBS_DIM  # 29
action_dim = TestDimensions.GRID2D_ACTION_DIM  # 8
grid_size = TestDimensions.GRID_SIZE  # 8
```

---

## Common Patterns

### Pattern 1: Creating Test Config

```python
from tests.test_townlet.utils.builders import make_test_bars_config

# Minimal config (1 meter)
bars = make_test_bars_config(num_meters=1)

# Standard config (8 meters)
bars = make_test_bars_config()

# Without terminal condition
bars = make_test_bars_config(num_meters=4, include_terminal=False)
```

### Pattern 2: Creating Test Affordance

```python
from tests.test_townlet.utils.builders import make_test_affordance

# Default bed affordance
bed = make_test_affordance()

# Custom affordance
job = make_test_affordance(
    id="Job",
    category="income",
    effects=[("money", 100.0)],
    operating_hours=(9, 17)  # 9am-5pm
)

# Multi-tick affordance (auto-sets required_ticks=5)
sleep = make_test_affordance(
    id="Sleep",
    interaction_type="multi_tick"
)
```

### Pattern 3: Creating Episode Data

```python
from tests.test_townlet.utils.builders import make_test_episode_metadata, make_test_recorded_step

# Default episode
metadata = make_test_episode_metadata()

# Custom episode ID
metadata = make_test_episode_metadata(episode_id=42)

# Default recorded step
step = make_test_recorded_step()

# Custom step
step = make_test_recorded_step(
    step=5,
    position=(7, 2),
    action=1,
    done=True
)
```

### Pattern 4: Using Fixtures

```python
def test_yaml_loading(temp_test_dir):
    """Use temp_test_dir instead of tempfile.TemporaryDirectory()"""
    config_path = temp_test_dir / "test.yaml"

    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)

    # Test logic...

def test_yaml_simple(temp_yaml_file):
    """Or use temp_yaml_file for common case"""
    with open(temp_yaml_file, 'w') as f:
        yaml.dump(config_data, f)

    # Test logic...
```

---

## Migration Guide

### Step 1: Identify Duplication

Look for:
- ❌ Hardcoded `(1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)` tuples
- ❌ Manual `BarsConfig(...)` with 20+ lines
- ❌ `tempfile.TemporaryDirectory()` context managers
- ❌ Magic numbers like `29` (observation dim) or `8` (action dim)

### Step 2: Add Imports

```python
from tests.test_townlet.utils.builders import (
    TestDimensions,          # For canonical dimensions
    make_test_meters,         # For meter tuples
    make_test_bars_config,    # For BarsConfig
    make_test_episode_metadata,  # For episode data
    make_test_recorded_step,  # For recorded steps
)
```

### Step 3: Replace Patterns

**Before**:
```python
meters = (1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85)
```

**After**:
```python
meters = make_test_meters()
```

---

**Before**:
```python
bars_config = BarsConfig(
    version="1.0",
    description="Test bars",
    bars=[...],  # 15 lines
    terminal_conditions=[...],  # 7 lines
)
```

**After**:
```python
bars_config = make_test_bars_config(num_meters=1)
```

---

**Before**:
```python
with tempfile.TemporaryDirectory() as tmpdir:
    config_path = Path(tmpdir) / "test.yaml"
    # ...
```

**After**:
```python
def test_something(temp_test_dir):  # Add parameter
    config_path = temp_test_dir / "test.yaml"
    # ...
```

---

## Advanced: Customizing Builders

### When to Customize

Customize builders when your test needs **specific values** different from defaults:

```python
# Custom bar with different tier
energy_bar = make_test_bar(
    name="energy",
    tier="secondary",  # Not default "pivotal"
    base_depletion=0.05  # Higher depletion
)

# Custom episode with high survival
long_episode = make_test_episode_metadata(
    episode_id=200,
    survival_steps=500,  # Not default 10
    total_reward=1000.0
)
```

### When NOT to Customize

If you're using the default values, **don't specify them**:

```python
# ❌ BAD: Redundant parameters
step = make_test_recorded_step(
    step=0,  # This is the default!
    position=(3, 5),  # This is the default!
    action=2  # This is the default!
)

# ✅ GOOD: Use defaults
step = make_test_recorded_step()
```

---

## Examples from Real Tests

### Example 1: test_recorder.py Refactoring

**Before** (11 lines):
```python
step = RecordedStep(
    step=0,
    position=(3, 5),
    meters=(1.0, 0.9, 0.8, 0.5, 0.7, 0.6, 0.95, 0.85),
    action=2,
    reward=1.0,
    intrinsic_reward=0.1,
    done=False,
    q_values=None,
)
```

**After** (1 line):
```python
step = make_test_recorded_step()
```

**Savings**: 10 lines per instance × 13 instances = **130 lines eliminated**

### Example 2: Dimension Validation

**Before** (magic number):
```python
def test_observation_size():
    obs = env.reset()
    assert obs.shape[-1] == 29  # Where does 29 come from?
```

**After** (documented):
```python
from tests.test_townlet.utils.builders import TestDimensions

def test_observation_size():
    obs = env.reset()
    assert obs.shape[-1] == TestDimensions.GRID2D_OBS_DIM  # 29 = 2+8+15+4
```

**Benefits**: Self-documenting, survives schema changes.

---

## Best Practices

### ✅ DO

1. **Use builders for all new tests**
   ```python
   # New test
   def test_new_feature():
       config = make_test_bars_config()  # ✅
   ```

2. **Use TestDimensions for assertions**
   ```python
   assert obs_dim == TestDimensions.GRID2D_OBS_DIM  # ✅
   ```

3. **Use temp_test_dir fixture**
   ```python
   def test_file_ops(temp_test_dir):  # ✅
       path = temp_test_dir / "file.txt"
   ```

4. **Customize only when needed**
   ```python
   # Only override episode_id
   metadata = make_test_episode_metadata(episode_id=42)  # ✅
   ```

### ❌ DON'T

1. **Don't hardcode magic numbers**
   ```python
   assert obs.shape[-1] == 29  # ❌ Use TestDimensions.GRID2D_OBS_DIM
   ```

2. **Don't manually create Pydantic configs**
   ```python
   bars = BarsConfig(...)  # ❌ Use make_test_bars_config()
   ```

3. **Don't use tempfile.TemporaryDirectory**
   ```python
   with tempfile.TemporaryDirectory() as tmpdir:  # ❌ Use temp_test_dir fixture
   ```

4. **Don't specify default values**
   ```python
   step = make_test_recorded_step(step=0)  # ❌ Redundant
   step = make_test_recorded_step()  # ✅ Cleaner
   ```

---

## FAQ

### Q: Can I still use raw Pydantic for complex tests?

**A**: Yes, but prefer builders + customization:

```python
# ❌ Avoid if possible
config = AffordanceConfig(
    id="ComplexThing",
    # 20 lines of config...
)

# ✅ Better: Start with builder, customize
config = make_test_affordance(
    id="ComplexThing",
    interaction_type="multi_tick",
    required_ticks=10
)
```

### Q: What if I need a value that's not the default?

**A**: Pass it as a parameter:

```python
# Need episode_id=55 instead of default 100
metadata = make_test_episode_metadata(episode_id=55)
```

### Q: How do I know what defaults a builder has?

**A**: Check the builder docstring or `builders.py` source:

```python
from tests.test_townlet.utils.builders import make_test_bar
help(make_test_bar)  # Shows all parameters and defaults
```

### Q: Can I use builders in integration tests?

**A**: Absolutely! Builders work everywhere:

```python
# Integration test
def test_full_training_loop():
    config = make_test_bars_config()
    # ...
```

---

## Rationale

### Why Builders?

1. **Single Source of Truth**
   - Change schema once in builders, not in 100 test files

2. **Reduces Boilerplate**
   - 20 lines → 1 line (95% reduction)

3. **Self-Documenting**
   - `TestDimensions.GRID2D_OBS_DIM` tells you what 29 means

4. **Consistency**
   - All tests use same canonical values

5. **Maintainability**
   - Schema evolution doesn't break tests

### Metrics

- **Before builders**: 113+ magic numbers, 600+ lines of boilerplate
- **After builders**: 0 magic numbers (in new tests), ~100 lines total
- **Impact**: 83% reduction in boilerplate when fully adopted

---

## Further Reading

- **builders.py source**: `tests/test_townlet/builders.py`
- **Example refactoring**: `tests/test_townlet/unit/recording/test_recorder.py`
- **Remediation plan**: `docs/quick/QUICK-004-TEST-REMEDIATION.md`
- **Test suite assessment**: `TEST_SUITE_ASSESSMENT.md`

---

**Last Updated**: Sprint 13 (2025-11-07)
**Next**: Continue gradual adoption across test suite (Sprints 14+)
