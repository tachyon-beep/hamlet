Title: Grid/ND substrate config uses defaults for behavior-critical fields

Severity: medium
Status: fixed

Subsystem: config/substrate
Affected Version/Branch: main

Affected Files:
- `src/townlet/substrate/config.py:23` (GridConfig.distance_metric default)
- `src/townlet/substrate/config.py:28` (GridConfig.observation_encoding default)
- `src/townlet/substrate/config.py:69` (GridNDConfig.distance_metric default)
- `src/townlet/substrate/config.py:75` (GridNDConfig.observation_encoding default)
- `src/townlet/substrate/config.py:81` (GridNDConfig.topology default)

Description:
- The No-Defaults principle states behavior-affecting parameters must be explicit. GridConfig and GridNDConfig set defaults for `distance_metric`, `observation_encoding`, and `topology`.

Reproduction:
- Omit those fields in `substrate.yaml`; Pydantic will fill defaults instead of forcing explicit operator choice.

Expected Behavior:
- Require explicit values for these fields; validation fails when omitted.

Actual Behavior:
- Implicit defaults applied.

Root Cause:
- Convenience defaults in schema.

Proposed Fix (Breaking OK):
- Remove defaults and require operators to specify values; provide migration guidance in error messages and templates.

Migration Impact:
- Existing packs must add explicit entries; aligns with policy.

Tests:
- Negative tests: missing fields raise errors; positive tests with explicit values pass.

Owner: config

## Fix Applied (2025-11-15)

**Root Cause Confirmed:**
Multiple substrate config fields had Pydantic defaults violating the "no hidden defaults" principle:
- `GridConfig.distance_metric` (default="manhattan")
- `GridConfig.observation_encoding` (default="relative")
- `GridNDConfig.distance_metric` (default="manhattan")
- `GridNDConfig.observation_encoding` (default="relative")
- `GridNDConfig.topology` (default="hypercube")
- `ContinuousConfig.distance_metric` (default="euclidean")
- `ContinuousConfig.observation_encoding` (default="relative")

**Fix:**
1. Removed all `default=` parameters from Field() definitions in `src/townlet/substrate/config.py`
2. Added explicit values to 4 config packs missing `observation_encoding`:
   - L1_continuous_2D/substrate.yaml
   - L1_continuous_3D/substrate.yaml
   - L2_partial_observability/substrate.yaml
   - L3_temporal_mechanics/substrate.yaml
3. Updated 38+ test cases across 4 test files to add explicit field values
4. Updated 3 example config files

**Verification:**
- All 409 substrate config tests pass
- All config packs load successfully
- Schema now correctly enforces required fields

**Impact:**
- Enforces "no hidden defaults" principle consistently
- Operators must now explicitly specify all behavioral parameters
- More reproducible configs - no implicit fallbacks
- Pydantic validation errors clearly state missing required fields

**Note on `action_discretization`:**
This field intentionally remains `Optional` with `default=None` because None has explicit semantics (use legacy 4-way actions). This is a nullable field, not a hidden default.
