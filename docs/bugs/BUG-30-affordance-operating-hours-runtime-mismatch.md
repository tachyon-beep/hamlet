Title: AffordanceConfig must require operating_hours; no hidden 24/7 default

Severity: high
Status: open

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
