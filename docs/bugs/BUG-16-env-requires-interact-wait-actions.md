Title: Env hard-requires INTERACT and WAIT actions; validator only warns

Severity: medium
Status: open

Subsystem: environment/vectorized + validation
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/vectorized_env.py:400` (get_action_by_name("INTERACT"/"WAIT"))
- `src/townlet/environment/substrate_action_validator.py:77` (INTERACT only warns)

Description:
- `VectorizedHamletEnv` looks up `INTERACT` and `WAIT` actions by name during init and will raise if missing.
- `SubstrateActionValidator` only emits a warning when INTERACT is not present.
- This mismatch can lead to runtime errors after passing validation.

Reproduction:
- Configure a global action space without INTERACT; validator yields a warning, environment init raises `ValueError`.

Expected Behavior:
- Validation enforces the same requirements as the environment (error if mandatory actions are missing), or environment tolerates absence gracefully (mask/omit usage).

Actual Behavior:
- Inconsistent: warns vs raises.

Root Cause:
- Divergent assumptions between validation and environment about mandatory actions.

Proposed Fix (Breaking OK):
- Make validator error (not warn) when required actions for a substrate are missing; document mandatory set per topology, including `WAIT` and `INTERACT`.
- Alternatively, relax environment to make these actions optional, guarding all callsites (more complexity and less clarity).

Migration Impact:
- Config packs missing these actions must be updated to include them or tests must expect errors earlier in validation.

Alternatives Considered:
- Auto-inject missing actions; violates no-defaults principle.

Tests:
- Add validation test ensuring missing INTERACT/WAIT triggers error; env should not be instantiable with an invalid action set.

Owner: environment
