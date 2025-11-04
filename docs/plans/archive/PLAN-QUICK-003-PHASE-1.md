# QUICK-003 Phase 1 Execution Plan: Config Loading Refactoring

**Status**: In Progress
**Created**: 2025-11-05
**Estimated Time**: 2-3 days (200+ LOC)

---

## Overview

Remove UAC/BAC defaults from `runner.py` and `unified_server.py`, add fail-fast validation, update test configs to be 100% complete.

## Findings from Code Analysis

### runner.py Analysis

**UAC/BAC Defaults Found** (❌ = Must Remove):

| Line | Parameter | Default | Type | Status |
|------|-----------|---------|------|--------|
| 84 | `max_episodes` | `10000` | BAC | ❌ Remove |
| 260 | `device` | `"cuda"` | Infrastructure | ✅ Keep (whitelisted) |
| 270 | `num_agents` | `1` | UAC | ❌ Remove |
| 271 | `grid_size` | `8` | UAC | ❌ Remove |
| 272 | `partial_observability` | `False` | UAC | ❌ Remove |
| 273 | `vision_range` | `2` | UAC | ❌ Remove |
| 274 | `enabled_affordances` | `None` | UAC | ❌ Remove |
| 275-286 | Energy costs | `0.005, 0.001, 0.0` | UAC | ❌ Remove |
| 308-312 | Curriculum params | Various | UAC | ❌ Remove |
| 322-329 | Exploration params | Various | BAC | ❌ Remove |
| 333-336 | Population params | Various | BAC | ❌ Remove |
| 340-344 | Training hyperparams | Various | BAC | ❌ Remove |
| 403-409 | Hparams logging | `.get()` calls | N/A | ❌ Remove (use direct access) |

**Total**: 30+ `.get()` calls with defaults to remove

### Config Completeness Analysis

**L0_0_minimal/training.yaml**: ✅ COMPLETE
- Has all environment, population, curriculum, exploration, training params
- Only missing: `enabled_affordances` is present but L1 needs it too

**L1_full_observability/training.yaml**: ⚠️ MISSING `enabled_affordances`
- Has all other params
- Needs to explicitly set `enabled_affordances: null` or list all affordances

### unified_server.py Analysis

**Minimal Changes Needed**:
- Line 400: `step_delay=0.2` - Infrastructure (visualization), ✅ Keep
- Mostly pass-through to runner.py, no UAC/BAC defaults

---

## Implementation Strategy

### Step 1: Create Validation Helper Function

Add helper function at top of `runner.py` to validate required config params:

```python
def _validate_required_params(config: dict, section: str, required_params: list[str]) -> None:
    """
    Validate that all required parameters are present in config section.

    Args:
        config: Full config dict
        section: Section name (e.g., "environment", "population")
        required_params: List of required parameter names

    Raises:
        ValueError: If any required parameter is missing
    """
    section_config = config.get(section, {})
    missing_params = [param for param in required_params if param not in section_config]

    if missing_params:
        # Create helpful error message with example
        example = "\n".join(f"  {param}: [value]" for param in missing_params)
        raise ValueError(
            f"Missing required parameters in '{section}' section of config:\n"
            f"{example}\n\n"
            f"Add these parameters to your training.yaml under '{section}:'."
        )
```

### Step 2: Refactor runner.py __init__

**Current (lines 79-84)**:
```python
# Set max_episodes: use provided value, otherwise read from config, otherwise default to 10000
if max_episodes is not None:
    self.max_episodes = max_episodes
else:
    training_cfg = self.config.get("training", {})
    self.max_episodes = training_cfg.get("max_episodes", 10000)
```

**After (fail-fast validation)**:
```python
# Set max_episodes: use provided value, otherwise read from config (required)
if max_episodes is not None:
    self.max_episodes = max_episodes
else:
    training_cfg = self.config.get("training", {})
    if "max_episodes" not in training_cfg:
        raise ValueError(
            "Missing required parameter 'max_episodes' in training section.\n"
            "Add to your training.yaml:\n"
            "training:\n"
            "  max_episodes: 5000"
        )
    self.max_episodes = training_cfg["max_episodes"]
```

### Step 3: Refactor runner.py run() method

**Lines 260-344** - Replace all `.get()` with defaults:

**Before**:
```python
device_str = self.config.get("training", {}).get("device", "cuda")
num_agents = population_cfg.get("num_agents", 1)
grid_size = environment_cfg.get("grid_size", 8)
# ... 30+ more
```

**After**:
```python
# Validate all required params first
_validate_required_params(self.config, "environment", [
    "grid_size", "partial_observability", "vision_range",
    "energy_move_depletion", "energy_wait_depletion", "energy_interact_depletion"
])
_validate_required_params(self.config, "population", [
    "num_agents", "learning_rate", "gamma",
    "replay_buffer_capacity", "network_type"
])
_validate_required_params(self.config, "curriculum", [
    "max_steps_per_episode", "survival_advance_threshold",
    "survival_retreat_threshold", "entropy_gate", "min_steps_at_stage"
])
_validate_required_params(self.config, "exploration", [
    "embed_dim", "initial_intrinsic_weight",
    "variance_threshold", "survival_window"
])
_validate_required_params(self.config, "training", [
    "device", "train_frequency", "target_update_frequency",
    "max_grad_norm", "epsilon_start", "epsilon_decay", "epsilon_min"
])

# Now use direct access (no defaults)
device_str = self.config["training"]["device"]
num_agents = population_cfg["num_agents"]
grid_size = environment_cfg["grid_size"]
# ... etc
```

**Special case - device fallback to CPU**:
```python
# Device: use config value, fallback to CPU if CUDA unavailable (infrastructure)
device_str = self.config["training"]["device"]
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
if device_str == "cuda" and not torch.cuda.is_available():
    logger.warning("CUDA requested but not available, falling back to CPU")
```

**Special case - enabled_affordances**:
```python
# enabled_affordances: None = all affordances, otherwise explicit list
enabled_affordances = environment_cfg.get("enabled_affordances", None)
# This is OK because None has semantic meaning ("all affordances")
# But we should still require it to be explicit in configs
```

### Step 4: Update Test Configs

**Add to L1_full_observability/training.yaml** (line 29, after vision_range):
```yaml
  enabled_affordances: null  # null = all 14 affordances enabled
```

**Verify L0_5, L2, L3** configs have all required params.

### Step 5: Update Hparams Logging

**Lines 398-412** - Replace `.get()` calls:

**Before**:
```python
self.hparams = {
    "learning_rate": learning_rate,
    "gamma": gamma,
    "network_type": network_type,
    "replay_buffer_capacity": replay_buffer_capacity,
    "grid_size": environment_cfg.get("grid_size", 8),  # ❌
    "partial_observability": environment_cfg.get("partial_observability", False),  # ❌
    # ...
}
```

**After**:
```python
self.hparams = {
    "learning_rate": learning_rate,
    "gamma": gamma,
    "network_type": network_type,
    "replay_buffer_capacity": replay_buffer_capacity,
    "grid_size": environment_cfg["grid_size"],  # ✅
    "partial_observability": environment_cfg["partial_observability"],  # ✅
    # ...
}
```

### Step 6: unified_server.py (Minimal Changes)

No UAC/BAC defaults found. Only pass-through orchestration.

**Keep as-is**: Line 400 `step_delay=0.2` is infrastructure (visualization speed).

---

## File-by-File Changes

### File 1: src/townlet/demo/runner.py

**Lines to change**: 20-40, 79-84, 260-344, 398-412, 454

**Changes**:
1. Add `_validate_required_params()` helper function (lines 21-40)
2. Refactor `__init__` max_episodes logic (lines 79-84)
3. Add validation calls at start of `run()` (lines 260-270)
4. Replace all `.get(..., default)` with direct access `[...]` (lines 270-344)
5. Update hparams logging to use direct access (lines 398-412)
6. Update episode step limit access (line 454)

**Expected LOC**: 150+ lines changed

### File 2: configs/L1_full_observability/training.yaml

**Lines to change**: 29 (after vision_range)

**Changes**:
1. Add `enabled_affordances: null  # All 14 affordances` (line 29)

**Expected LOC**: 1 line added

### File 3: configs/L0_5_dual_resource/training.yaml

**Lines to verify**: Check if `enabled_affordances` is present

### File 4: configs/L2_partial_observability/training.yaml

**Lines to verify**: Check if `enabled_affordances` is present

### File 5: configs/L3_temporal_mechanics/training.yaml

**Lines to verify**: Check if `enabled_affordances` is present

---

## Testing Plan

### Phase 1: Linter Validation
```bash
# Run linter on demo/ files
python scripts/no_defaults_lint.py src/townlet/demo/ --whitelist .defaults-whitelist.txt

# Expected: 0 violations in demo/ files (after refactoring)
```

### Phase 2: Config Loading Tests
```bash
# Test each config loads without errors
for config in configs/L0_0_minimal configs/L0_5_dual_resource configs/L1_full_observability configs/L2_partial_observability configs/L3_temporal_mechanics; do
  echo "Testing $config..."
  python -c "
from pathlib import Path
import yaml
import sys
sys.path.insert(0, 'src')
from townlet.demo.runner import DemoRunner

try:
    runner = DemoRunner(
        config_dir='$config',
        db_path='test.db',
        checkpoint_dir='test_checkpoints',
        max_episodes=1
    )
    print('✅ $config loaded successfully')
except Exception as e:
    print(f'❌ $config failed: {e}')
    sys.exit(1)
  "
done
```

### Phase 3: Missing Param Validation
```bash
# Test fail-fast validation by removing a required param
cp configs/L0_0_minimal/training.yaml configs/L0_0_minimal/training.yaml.bak
sed -i '/grid_size/d' configs/L0_0_minimal/training.yaml

# Try to load - should fail with clear error
python -c "
from townlet.demo.runner import DemoRunner
try:
    runner = DemoRunner(
        config_dir='configs/L0_0_minimal',
        db_path='test.db',
        checkpoint_dir='test_checkpoints',
        max_episodes=1
    )
    print('❌ Should have failed!')
    exit(1)
except ValueError as e:
    print(f'✅ Fail-fast validation working: {e}')
"

# Restore backup
mv configs/L0_0_minimal/training.yaml.bak configs/L0_0_minimal/training.yaml
```

### Phase 4: Full Test Suite
```bash
# Run full test suite to verify no regressions
uv run pytest tests/ -v

# Expected: All tests pass
```

---

## Risks and Mitigations

**Risk 1**: Missing params in configs cause immediate failures
- **Mitigation**: Test all 5 configs before committing
- **Mitigation**: Clear error messages guide users to fix

**Risk 2**: Test suite depends on default values
- **Mitigation**: Update test fixtures to have complete configs
- **Mitigation**: Tests should use minimal configs explicitly

**Risk 3**: Breaking changes for users with custom configs
- **Mitigation**: All example configs updated
- **Mitigation**: Migration guide in QUICK-003 task doc

---

## Success Criteria

- [ ] ✅ All `.get(..., default)` removed for UAC/BAC params in runner.py
- [ ] ✅ Fail-fast validation with clear error messages added
- [ ] ✅ All 5 test configs load successfully
- [ ] ✅ Missing param triggers clear error with example
- [ ] ✅ Linter passes for demo/ files (0 UAC/BAC violations)
- [ ] ✅ Full test suite passes (no regressions)
- [ ] ✅ `enabled_affordances` explicitly set in all configs

---

## Estimated Timeline

- **Step 1-2**: Validation helper + __init__ refactor - 2 hours
- **Step 3**: run() method refactor (30+ params) - 3 hours
- **Step 4**: Update configs - 1 hour
- **Step 5**: Hparams logging - 1 hour
- **Step 6**: unified_server.py verification - 30 min
- **Testing**: All 4 test phases - 2 hours

**Total**: ~10 hours (1.5 days)

---

**Created**: 2025-11-05
**Author**: Claude Code + PDR-002 Compliance
