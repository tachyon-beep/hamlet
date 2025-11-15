Title: Schema-level validation for mandatory actions per substrate

Severity: low
Status: open

Subsystem: environment/validation
Affected Version/Branch: main

Description:
- Align config validation with env requirements by declaring mandatory action names per substrate topology (e.g., INTERACT, WAIT, 4/6 movement deltas).

Proposed Enhancement:
- Extend `SubstrateActionValidator` to promote `INTERACT`/`WAIT` to errors when required by substrate; encode a policy table.
- Optionally enrich error messages with suggested fixes.

Migration Impact:
- Configs missing mandatory actions will fail fast during validation.

Tests:
- Unit tests for grid2d/cubic/aspatial combinations.

Owner: environment
