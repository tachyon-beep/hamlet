# TASK-001 Hardcoded-8 Audit Results

**Date**: 2025-11-04
**Task**: TASK-001 Variable-Size Meter System
**Purpose**: Identify ALL locations with hardcoded 8-meter assumptions

---

## Summary

**Total Locations Found**: 13 unique lines across 3 files
**Files Affected**: 3 (cascade_config.py, cascade_engine.py, vectorized_env.py)
**Duplicate Code**: METER_NAME_TO_IDX defined TWICE in affordance_config.py

---

## Detailed Findings

### File: src/townlet/environment/cascade_config.py

| Line | Issue | Change Required | Priority |
|------|-------|-----------------|----------|
| 27 | `index: int = Field(ge=0, le=7, ...)` | Change to `le=meter_count-1` | **HIGH** |
| 70 | `if len(v) != 8:` | Change to `if len(v) < 1 or len(v) > 32:` | **CRITICAL** |
| 75 | `if indices != {0, 1, 2, 3, 4, 5, 6, 7}:` | Change to `if indices != set(range(len(v))):` | **CRITICAL** |
| 117 | `source_index: int = Field(ge=0, le=7, ...)` | Change to `le=meter_count-1` | **HIGH** |
| 119 | `target_index: int = Field(ge=0, le=7, ...)` | Change to `le=meter_count-1` | **HIGH** |

**Actions**:

- Add `meter_count` property to BarsConfig
- Update validation to use dynamic `len(v)`
- Add validation for contiguous indices from 0

---

### File: src/townlet/environment/cascade_engine.py

| Line | Issue | Change Required | Priority |
|------|-------|-----------------|----------|
| 53 | Comment: `# Pre-compute base depletion tensor [8]` | Update to `[meter_count]` | LOW |
| 67 | Comment: `Build tensor of base depletion rates [8]` | Update to `[meter_count]` | LOW |
| 70 | Comment: `Tensor of shape [8] with depletion rates` | Update to `[meter_count]` | LOW |
| 72 | `depletions = torch.zeros(8, device=self.device)` | Change to `torch.zeros(meter_count, ...)` | **CRITICAL** |
| 319 | Comment: `Tensor of shape [8] with initial values` | Update to `[meter_count]` | LOW |
| 321 | `initial_values = torch.zeros(8, device=self.device)` | Change to `torch.zeros(meter_count, ...)` | **CRITICAL** |

**Actions**:

- Get meter_count from `self.config.bars.meter_count`
- Build tensors dynamically
- Update all comments for clarity

---

### File: src/townlet/environment/vectorized_env.py

| Line | Issue | Change Required | Priority |
|------|-------|-----------------|----------|
| 116 | Comment: `# Grid + position + 8 meters + affordance` | Update to `meter_count meters` | LOW |
| 117 | `self.observation_dim = ... + 2 + 8 + ...` | Change to `+ 2 + meter_count + ...` | **CRITICAL** |
| 120 | Comment: `# Grid one-hot + 8 meters + affordance` | Update to `meter_count meters` | LOW |
| 121 | `self.observation_dim = ... + 8 + ...` | Change to `+ meter_count + ...` | **CRITICAL** |
| 158 | `self.meters = torch.zeros((self.num_agents, 8), ...)` | Change to `(num_agents, meter_count), ...` | **CRITICAL** |

**Actions**:

- Get meter_count from loaded bars config
- Compute observation_dim dynamically
- Size meters tensor based on meter_count

---

### File: src/townlet/environment/affordance_config.py

| Line | Issue | Change Required | Priority |
|------|-------|-----------------|----------|
| 27 | `METER_NAME_TO_IDX: dict[str, int] = { ... }` | **DUPLICATE** - Remove entirely | **HIGH** |
| 49 | `if self.meter not in METER_NAME_TO_IDX:` | Use bars_config.meter_name_to_index | **HIGH** |
| 50 | Error message uses `METER_NAME_TO_IDX.keys()` | Use bars_config.meter_names | **HIGH** |
| 63 | `if self.meter not in METER_NAME_TO_IDX:` | Use bars_config.meter_name_to_index | **HIGH** |
| 64 | Error message uses `METER_NAME_TO_IDX.keys()` | Use bars_config.meter_names | **HIGH** |
| 198 | `METER_NAME_TO_IDX = { ... }` | **DUPLICATE** - Remove entirely | **HIGH** |

**Actions**:

- **CRITICAL**: Remove BOTH METER_NAME_TO_IDX dictionaries (duplicated code!)
- Load meter mapping from bars.yaml at runtime
- Add `meter_name_to_index` property to BarsConfig
- Validate meter references against BarsConfig in AffordanceEngine

---

### File: src/townlet/environment/affordance_engine.py

| Line | Issue | Change Required | Priority |
|------|-------|-----------------|----------|
| 27 | `from .affordance_config import METER_NAME_TO_IDX` | Remove import, use bars_config | **HIGH** |
| 151 | `meter_idx = METER_NAME_TO_IDX[cost.meter]` | Use `self.meter_name_to_idx` | **HIGH** |
| 156 | `meter_idx = METER_NAME_TO_IDX[effect.meter]` | Use `self.meter_name_to_idx` | **HIGH** |
| 205 | `meter_idx = METER_NAME_TO_IDX[cost.meter]` | Use `self.meter_name_to_idx` | **HIGH** |
| 210 | `meter_idx = METER_NAME_TO_IDX[effect.meter]` | Use `self.meter_name_to_idx` | **HIGH** |
| 219 | `meter_idx = METER_NAME_TO_IDX[effect.meter]` | Use `self.meter_name_to_idx` | **HIGH** |
| 242 | `meter_idx = METER_NAME_TO_IDX[cost.meter]` | Use `self.meter_name_to_idx` | **HIGH** |
| 399 | `meter_idx = METER_NAME_TO_IDX[effect.meter]` | Use `self.meter_name_to_idx` | **HIGH** |
| 408 | `meter_idx = METER_NAME_TO_IDX[cost.meter]` | Use `self.meter_name_to_idx` | **HIGH** |

**Actions**:

- Add `bars_config` parameter to AffordanceEngine `__init__`
- Build `self.meter_name_to_idx` from bars_config
- Replace all METER_NAME_TO_IDX lookups with self.meter_name_to_idx

---

## Implementation Checklist

### Phase 1: Config Layer (cascade_config.py)

- [ ] Line 27: Update BarConfig.index constraint to dynamic
- [ ] Line 70: Update validation to accept 1-32 meters
- [ ] Line 75: Update validation to check contiguous indices
- [ ] Line 117: Update CascadeConfig.source_index constraint
- [ ] Line 119: Update CascadeConfig.target_index constraint
- [ ] Add `meter_count` property to BarsConfig
- [ ] Add `meter_names` property to BarsConfig
- [ ] Add `meter_name_to_index` property to BarsConfig

### Phase 2: Engine Layer (cascade_engine.py, vectorized_env.py)

- [ ] cascade_engine.py:72 - Dynamic depletions tensor
- [ ] cascade_engine.py:321 - Dynamic initial_values tensor
- [ ] vectorized_env.py:117 - Dynamic obs_dim (POMDP)
- [ ] vectorized_env.py:121 - Dynamic obs_dim (full obs)
- [ ] vectorized_env.py:158 - Dynamic meters tensor
- [ ] Update all comments from [8] to [meter_count]

### Phase 3: Affordance Config (affordance_config.py, affordance_engine.py)

- [ ] affordance_config.py:27 - Remove METER_NAME_TO_IDX dict (first instance)
- [ ] affordance_config.py:198 - Remove METER_NAME_TO_IDX dict (second instance)
- [ ] affordance_config.py:49,50,63,64 - Remove validation (move to engine)
- [ ] affordance_engine.py:27 - Remove import of METER_NAME_TO_IDX
- [ ] affordance_engine.py - Add bars_config parameter to **init**
- [ ] affordance_engine.py - Build self.meter_name_to_idx from bars_config
- [ ] affordance_engine.py - Replace all METER_NAME_TO_IDX with self.meter_name_to_idx (9 locations)

---

## Risk Assessment

### High Risk Items

1. **METER_NAME_TO_IDX removal**: Affects 9 locations in affordance_engine.py
2. **obs_dim calculation**: Affects network initialization
3. **Tensor sizing**: Must be done at initialization, not per-step

### Medium Risk Items

1. **Cascade index validation**: Must validate against dynamic meter_count
2. **Comments**: Should be updated for clarity but won't break code

### Low Risk Items

1. **Comment updates**: Documentation only, no functional impact

---

## Verification Plan

### After Phase 1

- [ ] Test 4-meter config validates successfully
- [ ] Test 12-meter config validates successfully
- [ ] Test 1-meter config validates (boundary)
- [ ] Test 32-meter config validates (boundary)
- [ ] Test 33-meter config fails validation
- [ ] Test non-contiguous indices fail validation

### After Phase 2

- [ ] Test 4-meter environment creates 4-element tensors
- [ ] Test 12-meter environment creates 12-element tensors
- [ ] Test obs_dim scales correctly (4-meter < 8-meter < 12-meter)
- [ ] Test cascade engine works with variable meters

### After Phase 3

- [ ] Test affordance engine validates meter references
- [ ] Test affordance costs use correct meter indices
- [ ] Test affordance effects use correct meter indices

---

## Conclusion

**Total Changes Required**: ~35 locations across 5 files

**Files to Modify**:

1. `cascade_config.py` - 5 validation changes + 3 new properties
2. `cascade_engine.py` - 2 tensor sizing changes + 4 comment updates
3. `vectorized_env.py` - 3 obs_dim/tensor changes + 3 comment updates
4. `affordance_config.py` - Remove 2 duplicate dictionaries + 4 validations
5. `affordance_engine.py` - Remove import + add bars_config + 9 lookups

**Confidence Level**: HIGH

- Audit found all hardcoded-8 assumptions
- No false positives (grid_size=8 excluded correctly)
- Clear implementation path for each location

**Next Step**: Begin Phase 1 (Config Schema Refactor) with TDD approach
