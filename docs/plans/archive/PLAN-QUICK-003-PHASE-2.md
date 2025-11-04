# QUICK-003 Phase 2 Execution Plan: Environment Layer

**Status**: Planning
**Created**: 2025-11-05
**Scope**: Remove UAC defaults from environment layer (8 files, 29 violations)

## Overview

Phase 2 removes UAC parameter defaults from the environment layer. Unlike Phase 1 (config loading), this phase focuses on class/function parameter defaults that should be required, not config dictionary `.get()` calls.

**Key Principle**: Parameters that control universe behavior (grid size, energy costs, observability) MUST be explicitly provided by callers. No hidden defaults allowed.

## Violation Analysis

### Summary

- **Total Violations**: 29 across 8 files
- **Linter Output**: `python scripts/no_defaults_lint.py src/townlet/environment/`

### Breakdown by File

**vectorized_env.py (12 violations)**:
- Lines 36-46: DEF001 - `__init__` has 10 UAC parameter defaults
  - `grid_size=8` ❌ (UAC)
  - `device=torch.device("cpu")` ⚠️ (infrastructure - keep with docs)
  - `partial_observability=False` ❌ (UAC)
  - `vision_range=2` ❌ (UAC)
  - `enable_temporal_mechanics=False` ❌ (UAC)
  - `enabled_affordances=None` ✅ (semantic - None means "all")
  - `move_energy_cost=0.005` ❌ (UAC)
  - `wait_energy_cost=0.001` ❌ (UAC)
  - `interact_energy_cost=0.0` ❌ (UAC)
  - `agent_lifespan=1000` ❌ (UAC)
  - `config_pack_path=None` ⚠️ (infrastructure - fallback to test config)
- Lines 153-157: CALL001 - `get(..., default)` for meter indices (semantic - optional meters)
- Line 67: ASG002 - Ternary default for config_pack_path (infrastructure)
- Line 389: ASG002 - Ternary default (need to check)
- Line 709: CALL001 - `get(..., default)` (need to check)
- Lines 320, 611: DEF001 - Function parameter defaults (need to check)

**affordance_engine.py (6 violations)**:
- Line 215: ASG001 - Logical OR default
- Line 342: ASG002 - Ternary default
- Lines 114, 166, 249, 326, 416: DEF001 - Function parameter defaults

**affordance_config.py (5 violations)**:
- Lines 65, 66, 69, 70, 73: CALL003 - Dataclass `default_factory=` (likely field defaults)

**cascade_engine.py (1 violation)**:
- Line 145: DEF001 - Function parameter default

**cascade_config.py (1 violation)**:
- Line 29: CALL003 - `default=` in config

**meter_dynamics.py (2 violations)**:
- Lines 37, 65: DEF001 - Function parameter defaults

**observation_builder.py (1 violation)**:
- Line 52: DEF001 - Function parameter default

**reward_strategy.py (1 violation)**:
- Line 42: DEF001 - Function parameter default

## Implementation Strategy

### Approach

1. **Read each file** to understand defaults
2. **Classify defaults** as UAC (must remove), infrastructure (can keep), or semantic (None="all")
3. **Remove UAC defaults** from parameters
4. **Keep infrastructure defaults** with explicit documentation
5. **Verify tests don't break** (they already pass all params explicitly)

### Decision Matrix

| Default Type | Action | Example |
|--------------|--------|---------|
| **UAC Behavioral** | ❌ Remove | `grid_size=8`, `energy_cost=0.005` |
| **Infrastructure** | ⚠️ Keep with docs | `device=cpu`, `config_pack_path=None` |
| **Semantic** | ✅ Keep | `enabled_affordances=None` (means "all") |
| **Optional Meters** | ✅ Keep | `hygiene_idx=None` (meter might not exist) |

## File-by-File Changes

### 1. vectorized_env.py (12 violations)

**Changes**:

**A. Remove UAC parameter defaults from `__init__` (lines 33-47)**:

```python
# ❌ BEFORE
def __init__(
    self,
    num_agents: int,
    grid_size: int = 8,  # ❌ Remove
    device: torch.device = torch.device("cpu"),  # ⚠️ Keep (infrastructure)
    partial_observability: bool = False,  # ❌ Remove
    vision_range: int = 2,  # ❌ Remove
    enable_temporal_mechanics: bool = False,  # ❌ Remove
    enabled_affordances: list[str] | None = None,  # ✅ Keep (semantic)
    move_energy_cost: float = 0.005,  # ❌ Remove
    wait_energy_cost: float = 0.001,  # ❌ Remove
    interact_energy_cost: float = 0.0,  # ❌ Remove
    agent_lifespan: int = 1000,  # ❌ Remove
    config_pack_path: Path | None = None,  # ⚠️ Keep (infrastructure fallback)
):

# ✅ AFTER
def __init__(
    self,
    num_agents: int,
    grid_size: int,  # ✅ Required
    device: torch.device = torch.device("cpu"),  # ⚠️ Infrastructure default (documented)
    partial_observability: bool,  # ✅ Required
    vision_range: int,  # ✅ Required
    enable_temporal_mechanics: bool,  # ✅ Required
    enabled_affordances: list[str] | None = None,  # ✅ Semantic (None = "all affordances")
    move_energy_cost: float,  # ✅ Required
    wait_energy_cost: float,  # ✅ Required
    interact_energy_cost: float,  # ✅ Required
    agent_lifespan: int,  # ✅ Required
    config_pack_path: Path | None = None,  # ⚠️ Infrastructure fallback (documented)
):
    """
    Initialize vectorized environment.

    Args:
        num_agents: Number of parallel agents
        grid_size: Grid dimension (grid_size × grid_size)
        device: PyTorch device (default: cpu). Infrastructure default - can be overridden.
        partial_observability: If True, agent sees only local window (POMDP)
        vision_range: Radius of vision window (2 = 5×5 window)
        enable_temporal_mechanics: Enable time-based mechanics and multi-tick interactions
        enabled_affordances: List of affordance names to enable (None = all affordances)
        move_energy_cost: Energy cost per movement action
        wait_energy_cost: Energy cost per WAIT action
        interact_energy_cost: Energy cost per INTERACT action
        agent_lifespan: Maximum lifetime in steps (provides retirement incentive)
        config_pack_path: Path to config pack (default: configs/test). Infrastructure default.

    Note:
        - device and config_pack_path have infrastructure defaults (PDR-002 exemption)
        - enabled_affordances=None is a semantic default (None means "all affordances enabled")
        - All other parameters are UAC behavioral parameters and MUST be explicitly provided
    """
```

**B. Keep meter index lookups (lines 153-157)** - These are semantic defaults (optional meters):

```python
# ✅ KEEP (meters might not exist in bars.yaml)
self.energy_idx = meter_name_to_index.get("energy", 0)  # Fallback to first meter
self.health_idx = meter_name_to_index.get("health", min(6, meter_count - 1))  # Fallback to meter 6 or last
self.hygiene_idx = meter_name_to_index.get("hygiene", None)  # Optional meter
self.satiation_idx = meter_name_to_index.get("satiation", None)  # Optional meter
self.money_idx = meter_name_to_index.get("money", None)  # Optional meter
```

**C. Check line 389, 709, 320, 611** - Need to read and evaluate

**Expected Impact**: 8 UAC defaults removed, 0 tests broken (tests already pass all params)

---

### 2. affordance_engine.py (6 violations)

**Read lines 114, 166, 215, 249, 326, 342, 416** to identify what defaults exist and classify them.

**Expected Changes**: Remove UAC defaults from function parameters, keep infrastructure/semantic defaults.

---

### 3. affordance_config.py (5 violations)

**Read lines 65, 66, 69, 70, 73** to check dataclass field defaults.

**Expected Changes**:
- If dataclass fields have UAC defaults (e.g., `cost: float = 5.0`), these are likely **schema defaults** that should come from YAML
- May need to make fields required or add validation

---

### 4. Other Files (8 violations total)

- **cascade_config.py** (line 29): Check dataclass/dict default
- **cascade_engine.py** (line 145): Check function parameter default
- **meter_dynamics.py** (lines 37, 65): Check function parameter defaults
- **observation_builder.py** (line 52): Check function parameter default
- **reward_strategy.py** (line 42): Check function parameter default

**Expected Changes**: Remove UAC defaults from function parameters.

---

## Testing Strategy

### No Breaking Changes Expected

**Reason**: Test fixtures (conftest.py) already pass all parameters explicitly. Removing parameter defaults should not break any tests.

**Verification**:
1. Run linter on environment/ directory
2. Run environment unit tests: `uv run pytest tests/test_townlet/unit/environment/ -v`
3. Run integration tests: `uv run pytest tests/test_townlet/integration/ -v`
4. Verify all 5 production configs still load

### Test Files That Create VectorizedHamletEnv

- `tests/test_townlet/conftest.py` - Fixtures (already explicit)
- `tests/test_townlet/unit/environment/test_*.py` - Unit tests (already explicit)
- `tests/test_townlet/integration/test_*.py` - Integration tests (already explicit)

**Action**: Read a few test files to confirm they pass all params, then proceed with refactor.

---

## Execution Checklist

### Phase 2.1: vectorized_env.py (Primary File)

- [ ] Read vectorized_env.py __init__ (lines 33-80)
- [ ] Remove 8 UAC parameter defaults (grid_size, partial_observability, etc.)
- [ ] Keep infrastructure defaults (device, config_pack_path) with documentation
- [ ] Keep semantic default (enabled_affordances=None)
- [ ] Keep meter index lookups (lines 153-157)
- [ ] Check lines 389, 709, 320, 611 for other defaults
- [ ] Run linter on vectorized_env.py
- [ ] Run environment tests

### Phase 2.2: affordance_engine.py

- [ ] Read affordance_engine.py (lines 114, 166, 215, 249, 326, 342, 416)
- [ ] Classify each default (UAC vs infrastructure vs semantic)
- [ ] Remove UAC defaults
- [ ] Run linter on affordance_engine.py
- [ ] Run affordance tests

### Phase 2.3: affordance_config.py

- [ ] Read affordance_config.py (lines 65-73)
- [ ] Check dataclass field defaults
- [ ] Remove UAC defaults from dataclass fields
- [ ] Run linter on affordance_config.py
- [ ] Run config tests

### Phase 2.4: Remaining Files

- [ ] cascade_config.py (line 29)
- [ ] cascade_engine.py (line 145)
- [ ] meter_dynamics.py (lines 37, 65)
- [ ] observation_builder.py (line 52)
- [ ] reward_strategy.py (line 42)
- [ ] Run linter on all remaining files
- [ ] Run full environment test suite

### Phase 2.5: Validation

- [ ] Run linter on environment/: `python scripts/no_defaults_lint.py src/townlet/environment/`
- [ ] Expected: 0 violations (or only infrastructure/semantic defaults whitelisted)
- [ ] Run environment tests: `uv run pytest tests/test_townlet/unit/environment/ -v`
- [ ] Run integration tests: `uv run pytest tests/test_townlet/integration/ -v`
- [ ] Verify all 5 production configs still load
- [ ] Update whitelist to remove environment/ entries

---

## Success Criteria

1. **0 UAC defaults** in environment layer (infrastructure/semantic defaults documented and whitelisted)
2. **All tests pass** (unit + integration)
3. **All production configs load** (L0_0, L0_5, L1, L2, L3)
4. **Linter passes** for environment/ directory
5. **Documentation updated** for infrastructure defaults (device, config_pack_path)

---

## Next Steps After Phase 2

**Phase 3**: Networks Layer (src/townlet/agent/networks.py)
- Remove BAC defaults (hidden_dim, activation functions)
- Add network config schema (BRAIN_AS_CODE)

**Phase 4**: Remaining Systems (curriculum, exploration, population, training)
- Remove BAC/UAC defaults from all subsystems
- Add config schemas for all modules
