Title: Affordance randomization can collide with agent spawn positions

Severity: medium
Status: open

Subsystem: environment/vectorized
Affected Version/Branch: main

Affected Files:
- `src/townlet/environment/vectorized_env.py:1400` (randomize_affordance_positions)
- `src/townlet/environment/vectorized_env.py:616` (positions initialized after randomization)

Description:
- `randomize_affordance_positions` shuffles all substrate positions and assigns the first N to affordances.
- It only checks that `len(positions) >= affordances + num_agents` but does not prevent agent spawn positions from overlapping with chosen affordances.

Reproduction:
- Use a small grid and many affordances; random collisions between spawn and affordances are possible.

Expected Behavior:
- Agents spawn on empty tiles distinct from affordances, or at least a deterministic policy handles collisions.

Actual Behavior:
- No guarantee of separation; `initialize_positions` may pick positions already allocated to affordances.

Root Cause:
- Capacity check without actual reservation of non-overlapping slots for agent spawns.

Proposed Fix (Breaking OK):
- Reserve `num_agents` positions for agent spawns explicitly when selecting affordance positions (e.g., skip the first `num_agents` shuffled positions for agents), or pass the reserved set into `initialize_positions`.

Migration Impact:
- Slightly different random layouts; tests relying on exact positions must adjust or set RNG seeds.

Alternatives Considered:
- Allow overlaps and resolve at runtime by moving agents; adds complexity during step.

Tests:
- Add property test on small grids ensuring no overlap between spawn positions and affordances after reset.

Owner: environment
