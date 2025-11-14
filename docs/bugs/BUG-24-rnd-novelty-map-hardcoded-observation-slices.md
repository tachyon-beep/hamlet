Title: RND.get_novelty_map hardcodes observation slices (64:70) and grid flattening

Severity: medium
Status: open

Subsystem: exploration/RND
Affected Version/Branch: main

Affected Files:
- `src/townlet/exploration/rnd.py:281` (meters slice 64:70)

Description:
- `get_novelty_map` constructs a fake observation by setting a one-hot grid cell and meters slice to 0.5 using hardcoded indices.
- This breaks when observation layout/size differs from the assumed 70-dim layout.

Reproduction:
- Use a universe with different obs_dim or meter count; call `get_novelty_map` → out-of-bounds or nonsense placements.

Expected Behavior:
- Build observations based on the compiler’s observation spec (field slices), not hardcoded indices.

Actual Behavior:
- Hardcoded indices; function becomes invalid outside legacy layouts.

Root Cause:
- Legacy assumptions baked into debug visualization helper.

Proposed Fix (Breaking OK):
- Accept `ObservationSpec` or `ObservationActivity` and construct the observation using group slices to place the agent and meters appropriately.

Migration Impact:
- Callers must provide spec/activity; signature change.

Alternatives Considered:
- Remove `get_novelty_map` from core RND class and move to a debug utility that can import the env/spec.

Tests:
- Add smoke test for novelty map generation using a compiled universe.

Owner: exploration
