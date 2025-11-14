Title: Expose RNG seeding for affordance randomization and substrate placement

Severity: low
Status: open

Subsystem: environment/vectorized
Affected Version/Branch: main

Description:
- Env uses Python `random.shuffle` and substrate-provided random initializers without an explicit seed path.
- Tests and experiments benefit from reproducible layouts.

Proposed Enhancement:
- Add optional `rng_seed: int | None` to env constructor/reset to seed Python `random` and torch RNG used by substrate.
- Document that higher-level training runners can set seeds.

Migration Impact:
- Backwards compatible; only new optional parameter.

Tests:
- Determinism test: two resets with the same seed yield identical affordance positions and spawns.

Owner: environment
