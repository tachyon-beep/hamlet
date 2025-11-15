Title: AffordanceEngine.get_action_masks uses hardcoded action counts (4 + 15)

Severity: high
Status: open

Subsystem: environment/affordances
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/affordance_engine.py:281`

Description:
- `get_action_masks` assumes exactly 4 movement actions and 15 affordances, building a fixed-size mask (19 columns).
- Townlet now composes a dynamic action space from substrate + custom actions with a variable number of affordances.
- This function becomes wrong when action space differs, producing incorrect shapes and indices.

Reproduction:
- Call `AffordanceEngine.get_action_masks` in any universe where `num_affordances != 15` or movement actions are not 4.
- The returned mask will not match `env.action_dim` and indices for affordances are wrong.

Expected Behavior:
- Mask dimensions match the composed action space and align with the environment’s indices.

Actual Behavior:
- Hardcoded 19-wide mask with misaligned affordance indices.

Root Cause:
- Legacy assumption carried over after action space became dynamic.

Proposed Fix (Breaking OK):
- Remove `get_action_masks` from AffordanceEngine or rewrite it to accept a layout map from `ComposedActionSpace`.
- Source of truth for masks should be in `VectorizedHamletEnv.get_action_masks` (current integration point).

Migration Impact:
- Downstream code calling `AffordanceEngine.get_action_masks` must switch to `VectorizedHamletEnv.get_action_masks`.

Alternatives Considered:
- Dynamically mirroring environment action space inside AffordanceEngine; adds coupling and duplication.

Tests:
- Add regression test ensuring mask width equals `env.action_dim` and indices map through `ComposedActionSpace`.

Owner: environment
