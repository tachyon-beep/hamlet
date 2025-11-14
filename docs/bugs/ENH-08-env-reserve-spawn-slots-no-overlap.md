Title: Reserve agent spawn slots when randomizing affordances

Severity: medium
Status: open

Subsystem: environment/vectorized
Affected Version/Branch: main

Description:
- Ensure `randomize_affordance_positions` selects positions disjoint from the set reserved for agents.

Proposed Enhancement:
- Shuffle all substrate positions; reserve the first `num_agents` as spawn slots and the next `len(affordances)` as affordance slots (or vice versa) to avoid overlaps.
- Pass reserved spawn positions into `initialize_positions` if substrate supports it; otherwise, set positions explicitly.

Migration Impact:
- Changes randomization distribution; document in README and tests set RNG seeds.

Tests:
- Add test to ensure no overlap after reset.

Owner: environment
