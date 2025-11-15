Title: step() recomputes observations twice per step; time-of-day inconsistency

Severity: medium
Status: open

Subsystem: environment/vectorized
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/vectorized_env.py:1244`
- `src/townlet/environment/vectorized_env.py:947`

Description:
- `_calculate_shaped_rewards` calls `_get_observations()` to compute intrinsic rewards, then `step()` recomputes observations for return.
- `time_of_day` increments after rewards; observations used for intrinsic are computed with previous time, while returned observations use the incremented time.

Reproduction:
- Step environment with temporal features enabled and an exploration module that uses observations for intrinsic reward; log `time_sin/cos` differences.

Expected Behavior:
- Single observation per step or at least consistent temporal encoding between intrinsic computation and returned observation.

Actual Behavior:
- Two encodes; intrinsic sees t, returned obs sees t+1.

Root Cause:
- `_calculate_shaped_rewards` calls `_get_observations()` before `time_of_day` increments; `step()` increments time and calls `_get_observations()` again.

Proposed Fix (Breaking OK):
- Compute observation once at the end of the step and pass it (or the required features) to both DAC/intrinsic and as the returned value, or
- Increment `time_of_day` before both reward calc and final observation to keep alignment.

Migration Impact:
- Intrinsic computation semantics change slightly; tests relying on the old time reference should be updated.

Alternatives Considered:
- Cache obs pre-step and post-step separately; increases complexity and still duplicates work.

Tests:
- Add a test asserting `time_sin/cos` consistency between intrinsic eval and returned obs.

Owner: environment
