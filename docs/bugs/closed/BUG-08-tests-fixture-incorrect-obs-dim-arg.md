Title: Test fixture passes unsupported `obs_dim` to ReplayBuffer constructor

Severity: medium
Status: RESOLVED
Resolved Date: 2025-11-15

Subsystem: tests-fixtures
Affected Version/Branch: main

Affected Files:
- `tests/test_townlet/_fixtures/training.py:36`

Description:
- `ReplayBuffer` does not accept an `obs_dim` parameter, but the fixture passes it.
- This fixture would throw if invoked; likely dead code but confusing.

Reproduction:
- Import the fixture or call it directly; Python will error on unexpected keyword argument.

Expected Behavior:
- Fixture matches the buffer's constructor, or the class supports early preallocation via `obs_dim`.

Actual Behavior:
- Mismatch between fixture and API.

Root Cause:
- API drift or leftover from an earlier design.

Proposed Fix (Breaking OK):
- Either: remove `obs_dim=...` from fixture, or
- Add optional `obs_dim` to `ReplayBuffer.__init__` to preallocate (and document), updating code accordingly.

Migration Impact:
- If modifying ReplayBuffer, update docs and tests to reflect preallocation option.

Alternatives Considered:
- Keep as-is; leaves confusing/unusable fixture.

Tests:
- Ensure fixture composes and the buffer can be used immediately in tests.

Owner: tests

---

## RESOLUTION

**Fixed By**: Systematic debugging process using superpowers:systematic-debugging skill
**Resolution Date**: 2025-11-15

### Implementation Summary

**Phase 1 (Root Cause Investigation)**: Identified that ReplayBuffer uses lazy initialization, inferring obs_dim from data on first push(). The fixture passed an unsupported `obs_dim=FULL_OBS_DIM_8X8` parameter that would cause TypeError on construction.

**Phase 2 (Pattern Analysis)**: Found 30+ working examples across the codebase - ALL create ReplayBuffer with only `capacity` and `device` parameters. No code passes `obs_dim`. The fixture was never actually used by any test (dead code).

**Phase 3 (Hypothesis)**: Confirmed that removing the `obs_dim` parameter would fix the fixture to match current API.

**Phase 4 (Implementation)**:

1. **Fixture Fix** (`tests/test_townlet/_fixtures/training.py:28`):
   ```python
   # BEFORE:
   return ReplayBuffer(capacity=1000, obs_dim=FULL_OBS_DIM_8X8, device=device)

   # AFTER:
   return ReplayBuffer(capacity=1000, device=device)
   ```

2. **Import Cleanup**:
   - Removed unused `FULL_OBS_DIM_8X8` import
   - Follows no-defaults principle: delete dead code immediately

### Test Results
- Fixture functionality verified: Successfully creates buffer, accepts data, can sample
- Import test: No errors
- Unit tests: 3/3 PASSED
- Property-based tests: 6/6 PASSED

### Code Review
- Reviewer: feature-dev:code-reviewer subagent
- Status: ✅ APPROVED
- Score: 100/100 (Excellent)
- Findings:
  - Fixture now matches actual ReplayBuffer API
  - Unused import properly removed
  - Zero impact (fixture never used in tests)
  - Supports lazy initialization correctly

### Files Modified
1. `tests/test_townlet/_fixtures/training.py` - Removed obs_dim parameter and unused import

### Migration Notes
- Zero migration required - fixture was never used
- All tests create ReplayBuffer instances inline
- No breaking changes

### Impact
- ✅ Fixture now works correctly with ReplayBuffer API
- ✅ Supports lazy initialization pattern (obs_dim inferred from data)
- ✅ Removed confusing dead code
- ✅ Follows codebase patterns (30+ examples verified)
- ✅ Ready for use if needed in future tests
