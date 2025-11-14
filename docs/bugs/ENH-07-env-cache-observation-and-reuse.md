Title: Cache observation within step and reuse for intrinsic and return

Severity: medium
Status: open

Subsystem: environment/vectorized
Affected Version/Branch: main

Description:
- `_calculate_shaped_rewards` recomputes observations purely to feed intrinsic reward.
- `step()` also recomputes observations for the return value.
- Caching the observation once per step avoids redundant grid encoding and VFS writes.

Proposed Enhancement:
- In `step()`, compute observations once (at a consistent time reference), pass them into `_calculate_shaped_rewards` (or directly to the exploration module), and return the same tensor.
- Optionally thread a flag to skip re-encoding inside `_calculate_shaped_rewards`.

Migration Impact:
- Minor behavior change if we change the timing of `time_of_day` increment (see BUG-14) but net positive.

Tests:
- Perf microbenchmarks optional; functional tests should remain intact.

Owner: environment
