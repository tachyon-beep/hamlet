Title: Deprecate AffordanceEngine.get_action_masks in favor of env-level masking

Severity: low
Status: open

Subsystem: environment/affordances
Affected Version/Branch: main

Description:
- Having two masking implementations (one in env, one in AffordanceEngine) risks drift and duplication.
- AffordanceEngine’s mask is already incorrect (BUG-12) and not integrated with dynamic action spaces.

Proposed Enhancement:
- Mark `AffordanceEngine.get_action_masks` as deprecated; remove in next release.
- Keep affordance-hours and affordability helpers in AffordanceEngine; mask composition remains in env.

Migration Impact:
- Callers should use `VectorizedHamletEnv.get_action_masks`.

Tests:
- Ensure no tests rely on AffordanceEngine masks; update any that do.

Owner: environment
