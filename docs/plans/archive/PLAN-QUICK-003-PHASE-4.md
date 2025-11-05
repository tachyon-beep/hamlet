# QUICK-003 Phase 4 Execution Plan: Complete Defaults Cleanup

**Status**: Planning
**Created**: 2025-11-05
**Scope**: Remove ALL remaining UAC/BAC defaults (5 subsystems, 100 violations)

## Overview

Phase 4 completes the PDR-002 no-defaults cleanup by addressing ALL remaining violations across:
1. **Environment** (29 violations) - **PRIORITY** (brought forward per user request)
2. **Curriculum** (11 violations)
3. **Exploration** (11 violations)
4. **Population** (13 violations)
5. **Training** (36 violations)

**Total**: 100 violations to review and address

**Key Principle**: Each default must be classified as:
- ❌ **UAC/BAC** → Remove (behavioral parameter)
- ✅ **Infrastructure** → Keep with documentation (device, port, logging)
- ✅ **Semantic** → Keep (None="all", empty list, optional)
- ✅ **Feature Toggle** → Keep (enable/disable features)

## Violation Breakdown

### 1. Environment Layer (29 violations) - PRIORITY

**affordance_config.py (5 violations)**:
```python
# Lines 65-73: Pydantic Field defaults
costs: list[AffordanceCost] = Field(default_factory=list)  # ✅ Schema default (empty list OK)
costs_per_tick: list[AffordanceCost] = Field(default_factory=list)  # ✅ Schema default
effects: list[AffordanceEffect] = Field(default_factory=list)  # ✅ Schema default
effects_per_tick: list[AffordanceEffect] = Field(default_factory=list)  # ✅ Schema default
completion_bonus: list[AffordanceEffect] = Field(default_factory=list)  # ✅ Schema default
```
**Classification**: ✅ **All acceptable** - Pydantic schema defaults for optional collections
**Action**: Whitelist these patterns

**affordance_engine.py (6 violations)**:
```python
# Line 114: apply_instant_interaction
def apply_instant_interaction(..., check_affordability: bool = False):  # ✅ Feature toggle

# Line 166: apply_multi_tick_interaction
def apply_multi_tick_interaction(..., check_affordability: bool = False):  # ✅ Feature toggle

# Line 215: Logical OR default
required_ticks = affordance.required_ticks or 1  # ✅ Semantic (None=instant=1)

# Line 249: get_action_masks
def get_action_masks(..., check_affordability: bool = True, check_hours: bool = True):  # ✅ Feature toggles

# Line 326: get_affordance_cost
def get_affordance_cost(..., cost_mode: str = "instant"):  # ✅ Feature toggle (instant vs per_tick)

# Line 342: Ternary default
costs = affordance.costs if cost_mode == "instant" else affordance.costs_per_tick  # ✅ OK (not a default)

# Line 416: create_affordance_engine
def create_affordance_engine(config_pack_path: Path | None = None, num_agents: int = 1, device: torch.device = torch.device("cpu")):  # ⚠️ Mixed
```
**Classification**:
- Lines 114, 166, 249, 326: ✅ **Feature toggles** (acceptable)
- Line 215: ✅ **Semantic** (None=instant=1 tick)
- Line 342: ✅ **Not a default** (ternary expression in logic)
- Line 416: **Mixed** - `config_pack_path=None` ✅ (infra), `num_agents=1` ❌ (UAC), `device=cpu` ✅ (infra)

**Action**: Remove `num_agents=1` default from `create_affordance_engine`, whitelist rest

**cascade_config.py (1 violation)**:
```python
# Line 29: Pydantic field default
type: str | None = None  # ✅ Schema default (optional field)
```
**Classification**: ✅ **Schema default** (optional field)
**Action**: Whitelist

**cascade_engine.py (1 violation)**:
```python
# Line 145: apply_base_depletions
def apply_base_depletions(self, meters: torch.Tensor, depletion_multiplier: float = 1.0):  # ✅ Curriculum parameter
```
**Classification**: ✅ **Curriculum parameter** (1.0 = no scaling, acceptable default)
**Action**: Whitelist

**meter_dynamics.py (2 violations)**:
```python
# Line 37: __init__
def __init__(self, ..., cascade_config_dir: Path | None = None):  # ⚠️ Check if UAC or infra

# Line 65: deplete_meters
def deplete_meters(self, ..., depletion_multiplier: float = 1.0):  # ✅ Curriculum parameter
```
**Classification**:
- Line 37: Need to check - likely infrastructure fallback
- Line 65: ✅ **Curriculum parameter** (acceptable)

**Action**: Review line 37, likely whitelist both

**observation_builder.py (1 violation)**:
```python
# Line 52: build_observations
def build_observations(self, ..., time_of_day: int = 0):  # ❌ UAC default (time should be explicit)
```
**Classification**: ❌ **UAC default** (time_of_day is universe state)
**Action**: Remove default, make required parameter

**reward_strategy.py (1 violation)**:
```python
# Line 42: __init__
def __init__(self, ..., baseline_steps: float = 100.0):  # ❌ UAC default (baseline is universe parameter)
```
**Classification**: ❌ **UAC default** (baseline survival is universe-specific)
**Action**: Remove default, pass explicit value

**vectorized_env.py (12 violations)**:
```python
# Line 33: __init__ - Already has infrastructure defaults ✅
# Line 73: Ternary for config_pack_path ✅ Infrastructure
# Line 159-163: Meter index lookups with fallbacks ✅ Semantic (optional meters)
# Line 326: step(depletion_multiplier=1.0) ✅ Curriculum
# Line 395: old_positions ternary ✅ Not a default
# Line 617: calculate_baseline_survival(depletion_multiplier=1.0) ✅ Curriculum
# Line 715: checkpoint.get("ordering", self.affordance_names) ✅ Backwards compatibility
```
**Classification**: ✅ **All acceptable** - infrastructure, semantic, curriculum
**Action**: Whitelist all

**Environment Summary**:
- **Total**: 29 violations
- **Acceptable**: 27 violations (infrastructure/semantic/feature toggles)
- **Remove**: 2-3 violations (observation_builder.py:52, reward_strategy.py:42, affordance_engine.py:416)

---

### 2. Curriculum Layer (11 violations)

**adversarial.py (9 violations)**:
```python
# Lines 171-182: __init__ loads from checkpoint dict with .get() defaults
warmup_episodes = checkpoint.get("warmup_episodes", 100)  # ❌ UAC default
current_stage = checkpoint.get("current_stage", 0)  # ✅ Checkpoint default (0=stage 0)
# ... 7 more

# Line 139: __init__
def __init__(self, max_steps_per_episode: int = 500, ...):  # ❌ UAC defaults (5+ params)

# Line 434: get_stage_info
def get_stage_info(self, stage_idx: int | None = None):  # ✅ Feature (None=current stage)
```
**Classification**:
- Lines 171-182: Mixed - some checkpoint defaults ✅, some UAC ❌
- Line 139: ❌ **UAC defaults** (curriculum config should be explicit)
- Line 434: ✅ **Feature** (None=current)

**Action**: Remove UAC defaults from __init__, review checkpoint loading

**static.py (2 violations)**:
```python
# Line 24: __init__
def __init__(self, max_steps_per_episode: int = 500):  # ❌ UAC default

# Line 42: Logical OR
self.max_steps = max_steps_per_episode or 500  # ✅ Already validated above
```
**Classification**:
- Line 24: ❌ **UAC default**
- Line 42: ✅ **OK** (redundant with line 24, but not a separate default)

**Action**: Remove line 24 default

---

### 3. Exploration Layer (11 violations)

**action_selection.py (1 violation)**:
```python
# Line 11: epsilon_greedy_action_selection
def epsilon_greedy_action_selection(epsilon: float = 0.1, ...):  # ❌ BAC default
```
**Action**: Remove default

**adaptive_intrinsic.py (2 violations)**:
```python
# Line 21: __init__
def __init__(self, ..., survival_window: int = 100, ...):  # ❌ BAC defaults (multiple params)

# Line 77: select_actions
def select_actions(..., training: bool = True):  # ✅ Feature toggle
```
**Action**: Remove __init__ defaults, keep select_actions

**base.py (1 violation)**:
```python
# Line 25: select_actions
def select_actions(..., training: bool = True):  # ✅ Feature toggle
```
**Action**: Whitelist

**epsilon_greedy.py (2 violations)**:
```python
# Line 25: __init__
def __init__(self, epsilon_start: float = 1.0, ...):  # ❌ BAC defaults

# Line 43: select_actions
def select_actions(..., training: bool = True):  # ✅ Feature toggle
```
**Action**: Remove __init__ defaults, keep select_actions

**rnd.py (5 violations)**:
```python
# Lines 23, 53: __init__ methods
def __init__(self, ..., hidden_dim: int = 128, ...):  # ❌ BAC defaults

# Line 101: select_actions
def select_actions(..., training: bool = True):  # ✅ Feature toggle

# Line 191: get_novelty_map
def get_novelty_map(..., normalize: bool = False):  # ✅ Feature toggle
```
**Action**: Remove __init__ defaults, keep select_actions and get_novelty_map

---

### 4. Population Layer (13 violations)

**vectorized.py (13 violations)**:
```python
# Line 42: __init__
def __init__(self, ..., batch_size: int = 32, target_update_frequency: int = 100, ...):  # ❌ BAC defaults

# Lines 161, 269, 281, 392, 424, 534, 547, 550: Ternary expressions
# Various ternary defaults for LSTM state handling ✅ Feature-specific

# Lines 880, 886: checkpoint.get() calls
# Backwards compatibility for checkpoint loading ✅

# Line 261: _sync_curriculum_metrics
def _sync_curriculum_metrics(self, force: bool = False):  # ✅ Feature toggle

# Line 719: build_telemetry_snapshot
def build_telemetry_snapshot(self, include_q_values: bool = False):  # ✅ Feature toggle
```
**Action**: Remove __init__ BAC defaults, whitelist rest

---

### 5. Training Layer (36 violations)

**replay_buffer.py (1 violation)**:
```python
# Line 17: __init__
def __init__(self, capacity: int = 10000):  # ❌ BAC default
```
**Action**: Remove default

**sequential_replay_buffer.py (1 violation)**:
```python
# Line 96: sample_sequences
def sample_sequences(..., burn_in_length: int = 0):  # ✅ Feature (0=no burn-in)
```
**Action**: Whitelist

**state.py (10 violations)**:
```python
# Lines 46-50, 66-69: Pydantic Field defaults
episode: int | None = None  # ✅ Schema defaults (optional fields)
training_step: int | None = None  # ✅
hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None  # ✅
curriculum_stage: int = 0  # ✅
last_checkpoint_episode: int = 0  # ✅
intrinsic_weight_history: list[float] = Field(default_factory=list)  # ✅
survival_history: list[float] = Field(default_factory=list)  # ✅
stage_history: list[int] = Field(default_factory=list)  # ✅
checkpoint_episodes: list[int] = Field(default_factory=list)  # ✅

# Line 94: __init__
def __init__(self, ..., episode: int = 0, ...):  # ⚠️ Check if metadata or UAC

# Line 122: Ternary
timestamp = checkpoint.get("timestamp", datetime.now().isoformat())  # ✅ Metadata fallback
```
**Action**: Review line 94, likely whitelist all (metadata/schema defaults)

**tensorboard_logger.py (28 violations)**:
```python
# Line 47: __init__
def __init__(self, log_dir: str | None = None, ...):  # ⚠️ Infrastructure (logging)

# Lines 74, 161, 196, 219, 261, 279: Method defaults
# Various enabled: bool = True parameters ✅ Feature toggles

# Lines 99-294: Ternary expressions and .get() calls
# Telemetry data extraction with fallbacks ✅ Telemetry (acceptable)
```
**Action**: Whitelist all (telemetry/logging infrastructure)

---

## Implementation Strategy

### Phase 4.1: Environment Cleanup (PRIORITY)

**High Priority Removals**:
1. `observation_builder.py:52` - Remove `time_of_day=0` default
2. `reward_strategy.py:42` - Remove `baseline_steps=100.0` default
3. `affordance_engine.py:416` - Remove `num_agents=1` default

**Steps**:
- [ ] Read observation_builder.py, identify all `build_observations()` callers
- [ ] Remove `time_of_day=0`, ensure all callers pass explicit time
- [ ] Read reward_strategy.py, identify all `RewardStrategy()` instantiations
- [ ] Remove `baseline_steps=100.0`, ensure vectorized_env.py passes explicit value
- [ ] Read affordance_engine.py, update `create_affordance_engine()`
- [ ] Remove `num_agents=1`, update any callers
- [ ] Whitelist remaining 26 violations (document as infrastructure/semantic)

### Phase 4.2: Curriculum Cleanup

**Removals**:
1. `adversarial.py:139` - Remove all __init__ defaults
2. `static.py:24` - Remove `max_steps_per_episode=500` default

**Steps**:
- [ ] Read adversarial.py __init__, list all parameters with defaults
- [ ] Remove defaults, ensure runner.py passes all curriculum config
- [ ] Read static.py __init__, remove default
- [ ] Whitelist checkpoint loading .get() calls (backwards compatibility)

### Phase 4.3: Exploration Cleanup

**Removals**:
1. `action_selection.py:11` - Remove `epsilon=0.1` default
2. `adaptive_intrinsic.py:21` - Remove all __init__ defaults
3. `epsilon_greedy.py:25` - Remove all __init__ defaults
4. `rnd.py:23,53` - Remove all __init__ defaults

**Steps**:
- [ ] Read each exploration class __init__
- [ ] Remove BAC defaults (epsilon, survival_window, hidden_dim, etc.)
- [ ] Ensure runner.py passes all exploration config
- [ ] Whitelist feature toggles (`training=True`, `normalize=False`)

### Phase 4.4: Population Cleanup

**Removals**:
1. `vectorized.py:42` - Remove __init__ defaults (batch_size, target_update_frequency, etc.)

**Steps**:
- [ ] Read vectorized.py __init__, list all BAC defaults
- [ ] Remove defaults, ensure runner.py passes all population config
- [ ] Whitelist ternary expressions (LSTM state logic)
- [ ] Whitelist checkpoint .get() calls (backwards compatibility)

### Phase 4.5: Training Cleanup

**Removals**:
1. `replay_buffer.py:17` - Remove `capacity=10000` default

**Steps**:
- [ ] Remove capacity default from ReplayBuffer
- [ ] Ensure vectorized.py passes explicit capacity
- [ ] Whitelist state.py (all Pydantic schema defaults)
- [ ] Whitelist tensorboard_logger.py (all telemetry infrastructure)

### Phase 4.6: Validation

- [ ] Run linter on all 5 subsystems
- [ ] Expected: Only whitelisted violations remain
- [ ] Run full test suite: `uv run pytest`
- [ ] Verify all 5 production configs load (L0_0, L0_5, L1, L2, L3)
- [ ] Document all whitelisted patterns in `.defaults-whitelist.txt`

---

## Classification Summary

| Subsystem | Total | Remove | Whitelist | Notes |
|-----------|-------|--------|-----------|-------|
| **Environment** | 29 | 2-3 | 26-27 | Mostly infrastructure/semantic |
| **Curriculum** | 11 | 6-7 | 4-5 | Remove __init__ defaults |
| **Exploration** | 11 | 6-7 | 4-5 | Remove __init__ BAC defaults |
| **Population** | 13 | 3-4 | 9-10 | Remove __init__ BAC defaults |
| **Training** | 36 | 1 | 35 | Mostly telemetry/schema |
| **Total** | 100 | 18-22 | 78-82 | ~80% are acceptable |

---

## Expected Removals (Detailed List)

### Must Remove (UAC/BAC Behavioral Defaults):

1. `observation_builder.py:52` - `time_of_day=0`
2. `reward_strategy.py:42` - `baseline_steps=100.0`
3. `affordance_engine.py:416` - `num_agents=1`
4. `adversarial.py:139` - all __init__ defaults (5+ params)
5. `static.py:24` - `max_steps_per_episode=500`
6. `action_selection.py:11` - `epsilon=0.1`
7. `adaptive_intrinsic.py:21` - all __init__ defaults (5+ params)
8. `epsilon_greedy.py:25` - all __init__ defaults (3+ params)
9. `rnd.py:23,53` - all __init__ defaults (3+ params each)
10. `vectorized.py:42` - all __init__ BAC defaults (batch_size, target_update_frequency, etc.)
11. `replay_buffer.py:17` - `capacity=10000`

**Estimated Total Removals**: ~18-22 defaults across 11 locations

---

## Success Criteria

1. **18-22 UAC/BAC defaults removed** from behavioral parameters
2. **78-82 defaults whitelisted** with documentation (infrastructure/semantic/schema)
3. **All tests pass** - full test suite
4. **All production configs load** - L0_0, L0_5, L1, L2, L3
5. **Linter passes** - only whitelisted violations remain
6. **Whitelist documented** - clear classification for each pattern

---

## Estimated Effort

- **Environment** (priority): 2-3 hours (2-3 removals + documentation)
- **Curriculum**: 1-2 hours (2 files, 6-7 removals)
- **Exploration**: 2-3 hours (4 files, 6-7 removals)
- **Population**: 1-2 hours (1 file, 3-4 removals)
- **Training**: 0.5-1 hour (1 removal + whitelist documentation)
- **Testing**: 1-2 hours (verify all systems)

**Total**: 8-13 hours (1-2 days)

---

## Next Steps

1. **Execute Phase 4.1** - Environment cleanup (PRIORITY)
2. **Execute Phase 4.2-4.5** - Other subsystems
3. **Deploy compliant whitelist** - Document all acceptable patterns
4. **Final validation** - Linter + tests + configs

Phase 4 completes QUICK-003 PDR-002 cleanup across entire townlet codebase.
