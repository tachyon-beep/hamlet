Title: Grid/ND substrate config uses defaults for behavior-critical fields

Severity: medium
Status: open

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
