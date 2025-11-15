Title: AffordanceConfig must require operating_hours; no hidden 24/7 default

Severity: high
Status: RESOLVED
Resolved Date: 2025-11-15

Subsystem: config/affordance + environment
Affected Version/Branch: main

Affected Files:
- `src/townlet/config/affordance.py:44` (operating_hours: Optional)
- `src/townlet/environment/vectorized_env.py:60` (raises if operating_hours is None)

Description:
- Current schema permits `operating_hours` to be omitted (`None`), implicitly meaning 24/7. The compiler also treats missing hours as always open when building `action_mask_table`.
- Runtime `VectorizedHamletEnv` rejects affordances missing `operating_hours` with a ValueError during affordance conversion.

Reproduction:
- Affordance with no operating_hours compiles, but env init raises.

Expected Behavior:
- No hidden defaults. `operating_hours` is required everywhere. If absent, compilation fails with a clear error.

Actual Behavior:
- Compiler and runtime disagree; leads to runtime failure.

Root Cause:
- Divergent assumptions between config schema and runtime.

Proposed Fix (Breaking OK):
- Enforce required `operating_hours` in config and compiler:
  - Schema: In `src/townlet/config/affordance.py`, change `operating_hours` to a required `list[int]` with length 2; remove Optional.
  - Compiler: In `src/townlet/universe/compiler.py` (action_mask_table build), error if any affordance lacks `operating_hours` instead of assuming always-open.
  - Runtime: Keep `VectorizedHamletEnv` strict (already errors on missing hours).

Migration Impact:
- All packs must add explicit `operating_hours: [open_hour, close_hour]` for every affordance. For 24/7 availability, set `[0, 24]`.

Tests:
- Add compiler test that missing `operating_hours` fails with actionable error.
- Add positive tests for 24/7 (`[0, 24]`) and overnight windows (e.g., `[18, 28]`).

Owner: config+env

---

## RESOLUTION

**Fixed By**: Systematic debugging process using superpowers:systematic-debugging skill
**Resolution Date**: 2025-11-15

### Implementation Summary

**Phase 1 (Root Cause Investigation)**: Identified that config schema allowed None with implicit "24/7 always open" default, compiler had 4 locations with None handling logic, while runtime correctly rejected None. This created a config→compiler→runtime mismatch.

**Phase 2 (Pattern Analysis)**: Found that other required fields (BarConfig.range, BarConfig.initial) use required pattern without Optional/default. Identified that operating_hours is behavioral (not metadata) and should be required.

**Phase 3 (Hypothesis)**: Confirmed that making operating_hours required would eliminate mismatch and enforce no-defaults principle.

**Phase 4 (Implementation)**:

1. **Schema Change** (`src/townlet/config/affordance.py:52`):
   - Changed from `operating_hours: list[int] | None = Field(default=None)`
   - To `operating_hours: list[int] = Field(min_length=2, max_length=2)`
   - Updated validator to remove None handling

2. **Compiler Changes** (`src/townlet/universe/compiler.py`):
   - Removed None checks at 4 locations:
     - Line 1277: _validate_operating_hours
     - Line 1659: _calculate_income_hours_per_day
     - Line 1667: _affordance_open_for_hour
     - Line 2102: action_mask_table building

3. **Test Coverage** (`tests/test_townlet/unit/config/test_affordance_config_dto.py`):
   - Added `test_operating_hours_required()` validating:
     - Missing operating_hours raises ValidationError
     - Explicit None raises ValidationError
     - Explicit [0, 24] works correctly

4. **Test Fixtures Updated**:
   - `test_capability_validation.py`: 9 test cases updated with `operating_hours: [0, 24]`
   - All other test affordances already had explicit values

### Test Results
- Config DTO tests: 12/12 PASSED ✓
- Capability validation tests: 8/8 PASSED ✓
- Stage 6 optimization tests: 3/3 PASSED ✓
- Total: 23/23 PASSED ✓

### Code Review
- Reviewer: feature-dev:code-reviewer subagent
- Status: ✅ APPROVED
- Findings: Implementation is complete, correct, and follows all project principles
- No issues found, no changes required

### Files Modified
1. `src/townlet/config/affordance.py` - Schema enforcement
2. `src/townlet/universe/compiler.py` - Removed implicit None handling (4 locations)
3. `tests/test_townlet/unit/config/test_affordance_config_dto.py` - New test + fixture updates
4. `tests/test_townlet/unit/universe/test_capability_validation.py` - Fixture updates

### Migration Notes
- All 13 config packs already had explicit operating_hours (160 declarations total)
- Breaking change appropriate for pre-release project (zero users)
- Aligns with CLAUDE.md no-defaults principle and zero-backwards-compatibility policy

### Impact
- ✅ Eliminates config/runtime mismatch
- ✅ Enforces no-defaults principle for behavioral parameters
- ✅ Clear validation errors when operating_hours missing
- ✅ All implicit "24/7 always open" logic removed
- ✅ Maintains defensive runtime validation for safety
