Title: Spatial feasibility check assumes only 1 agent; ignores `population.num_agents`

Severity: medium
Status: open

Subsystem: universe/compiler
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py:1189` (required_cells = required + 1)

Description:
- The grid capacity check uses `required_cells = required + 1` (+1 agent), but does not account for multi-agent populations.
- For `num_agents > 1`, agents may exceed available cells even when affordances fit, if distinct spawn cells are required.

Reproduction:
1) Set `population.num_agents = 8` on a 3×3 grid (9 cells) with 2+ affordances.
2) Compiler passes feasibility check, but environment may collide at spawn or rely on overlapping positions.

Expected Behavior:
- When unique spawn positions are required, compiler should check `required_cells = required_affordances + num_agents`.

Actual Behavior:
- Always adds only 1 agent in feasibility calculation.

Root Cause:
- Hard-coded `+ 1` rather than reading `population.num_agents`.

Proposed Fix (Breaking OK):
- Use `num_agents = max(1, raw_configs.population.num_agents)` and compute `required_cells = required_affordances + num_agents`.
- If overlapping spawns are acceptable, make this a config switch and document behavior.

Migration Impact:
- Packs that relied on overlaps must update configs or set policy explicitly.

Alternatives Considered:
- Treat as warning; less safe and drifts from env behavior where overlaps are discouraged.

Tests:
- Add compiler test that fails when `required_cells` exceeds capacity for multi-agent cases.

Owner: compiler
