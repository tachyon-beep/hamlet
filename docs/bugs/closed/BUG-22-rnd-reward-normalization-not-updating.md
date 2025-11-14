Title: RND reward normalization (RunningMeanStd) never updates during env rollout

Severity: high
Status: fixed

Subsystem: exploration/RND + environment
Affected Version/Branch: main

Affected Files:
- `src/townlet/exploration/rnd.py` (compute_intrinsic_rewards)
- `src/townlet/environment/vectorized_env.py:1244` (calls with update_stats=False)

Description:
- RND computes intrinsic rewards as normalized MSE using a running mean/std (RunningMeanStd).
- The environment calls `compute_intrinsic_rewards(..., update_stats=False)` every step, so running stats never update.
- Intrinsic rewards remain scaled by the initial variance (≈1), defeating normalization.

Reproduction:
1) Run training; log `reward_rms.var` across steps → remains at the initial value.
2) Intrinsic reward magnitudes do not adapt to observation novelty distribution.

Expected Behavior:
- During training rollouts, update the running stats so normalization tracks the distribution.

Actual Behavior:
- Stats not updated; normalization is effectively static.

Root Cause:
- Env invokes `compute_intrinsic_rewards` with `update_stats=False`.

Proposed Fix (Breaking OK):
- Change env to call with `update_stats=True` during training paths, or
- Split API: `compute_intrinsic_rewards()` always updates, add `compute_intrinsic_rewards_eval()` for eval.

Migration Impact:
- Intrinsic reward magnitudes will change; DAC weight calibration may need adjustment.

Alternatives Considered:
- Update stats inside `RNDExploration.update` based on the same observations buffer; harder to match per-step distribution.

Tests:
- Add test that `reward_rms.var` changes after multiple steps with update enabled.

Owner: exploration

## Fix Applied (2025-11-15)

**Root Cause Confirmed:**
- Environment called `compute_intrinsic_rewards(..., update_stats=False)` during training rollouts
- Population separately called `compute_intrinsic_rewards(old_obs, update_stats=True)` but with stale observations (step N-1)
- Result: RND normalization stats lagged 2 steps behind the observations being normalized

**Fix:**
1. Changed `src/townlet/environment/vectorized_env.py:1250` to call with `update_stats=True`
2. Changed `src/townlet/population/vectorized.py:583` to call with `update_stats=False` (stats now updated in env)

**Verification:**
- Created diagnostic scripts demonstrating the 2-step lag
- Created regression tests in `tests/test_townlet/unit/exploration/test_rnd_stats_update_timing.py`
- All existing RND tests pass (47 tests)
- All DAC integration tests pass (8 tests)

**Impact:**
- RND intrinsic rewards now properly normalized using current observation distribution
- No changes to public APIs or config files
- Checkpoints remain compatible (stats already tracked)
