Title: Expose substrate-level valid movement mask utility

Severity: low
Status: open

Subsystem: substrate
Affected Version/Branch: main

Description:
- Environments compute movement validity masks based on positions and grid bounds.
- Substrates already encode all boundary semantics; a utility here would centralize and de-duplicate logic.

Proposed Enhancement:
- Add method `get_valid_movement_mask(positions) -> Tensor[batch, 2N]` indicating valid ± moves per dimension, leaving INTERACT/WAIT to env.
- Env can then combine this with affordance and death masks.

Migration Impact:
- Optional utility; env can adopt incrementally.

Tests:
- Unit tests across boundary modes (clamp/wrap/bounce/sticky).

Owner: substrate
