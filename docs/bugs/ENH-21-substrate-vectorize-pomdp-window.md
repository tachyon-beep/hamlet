Title: Vectorize Grid2D encode_partial_observation to avoid Python loops

Severity: medium
Status: open

Subsystem: substrate/grid2d
Affected Version/Branch: main

Affected Files:
- `src/townlet/substrate/grid2d.py:224` (encode_partial_observation)

Description:
- The POMDP local window is built with nested Python loops: per-agent × per-affordance.
- This is O(A×F) Python work per step and can become costly with many agents/affordances.

Proposed Enhancement:
- Vectorize using tensor operations (e.g., compute relative positions for all affordances vs each agent, clamp, then scatter into preallocated windows), or use unfold/strided conv on a sparse occupancy grid representation.

Migration Impact:
- Behavior unchanged; perf improvement only.

Tests:
- Existing tests suffice; add micro-benchmark notes if desired.

Owner: substrate
