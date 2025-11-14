Title: RND reward normalization (RunningMeanStd) never updates during env rollout

Severity: high
Status: open

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
