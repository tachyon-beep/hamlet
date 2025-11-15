Title: allow_unfeasible_universe can downgrade fundamentally unwinnable universes to warnings

Severity: medium
Status: open

Ticket Type: JANK
Subsystem: universe/compiler (Stage 4 cross-validation)
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py:1080`
- `src/townlet/config/training.py:91`

Description:
- `training.allow_unfeasible_universe` controls whether certain Stage 4 feasibility issues are treated as errors or warnings.
- `_record_feasibility_issue` uses this flag for:
  - Economic balance and income hours (`_validate_economic_balance`),
  - Meter sustainability (`_validate_meter_sustainability`),
  - Capacity constraints (`_validate_capacity_and_sustainability`).
- With the flag set to `true`, universes that are logically unwinnable can still compile, with only warnings.

Reproduction:
1) Configure a universe where a critical meter (e.g., health) has `base_depletion > 0` and no affordance/effect pipeline ever increases it.
2) Set `training.allow_unfeasible_universe: true` in `training.yaml`.
3) Compile:
   - Compilation succeeds.
   - Warnings about unsustainable meters are emitted, but there is no hard failure.

Expected Behavior:
- Debug/test configs should be able to bypass softer economic checks, but:
  - Universes that are structurally impossible to survive should still fail compilation, or
  - At minimum, a strong, single diagnostic should flag “Universe is unsustainable even with allow_unfeasible_universe=true”.

Actual Behavior:
- There is no distinction between “economically stressed” and “mathematically unwinnable” worlds; both are subject to the same downgrade.
- From the outside, “compile succeeded” looks the same, regardless of whether survival is possible.

Root Cause:
- `allow_unfeasible_universe` is applied uniformly to all feasibility checks, with no severity taxonomy.
- Stage 4 treats all `UAC-VAL-005` / `UAC-VAL-002` issues as optional when the flag is set.

Risk:
- Students and operators may train agents for long periods on impossible universes, misattributing failure to the RL algorithm instead of the configuration.
- CI and tooling that only look at success/failure status can’t distinguish “green but unsustainable” universes from truly healthy ones.

Proposed Directions:
- Introduce severity tiers for feasibility issues:
  - Hard failures (e.g., critical meter with no restoring affordances) must always be errors.
  - Softer economic concerns (e.g., income < costs, but still some restorative path) may honor `allow_unfeasible_universe`.
- When the flag downgrades any feasibility issue, emit a single high-visibility summary line or code indicating “Universe compiled in unfeasible mode”.

Tests:
- Unit: critical meter with positive depletion and zero restoration should raise an error even with `allow_unfeasible_universe=true`.
- Integration: highlight difference between a pack that is merely “harsh” vs one that is literally unwinnable.

Owner: compiler
Links:
- `src/townlet/universe/compiler.py:1080–1160`, `1680–1800`
