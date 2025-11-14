Title: Action costs/effects with unknown meters are silently trimmed during action space composition

Severity: medium
Status: open

Ticket Type: JANK
Subsystem: universe/compiler_inputs (action space composition)
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler_inputs.py:190`

Description:
- `_compose_action_space` builds a combined action space from substrate defaults plus `global_actions.yaml`.
- The helper `_trim_meter_payload` removes entries in `costs` / `effects` whose meter names are not listed in `hamlet_config.bars`.
- When this happens, the compiler only emits a generic hint once and continues; no per-action diagnostic is emitted.

Reproduction:
1) Define a custom action in `global_actions.yaml` with a typo’d meter name (e.g., `"hygeine"` instead of `"hygiene"`).
2) Compile the pack.
3) Observe:
   - Compilation succeeds.
   - A single hint about “Action costs/effects referencing meters absent from bars.yaml were ignored…” may appear.
   - The action executes without the intended cost/effect, unbalancing the environment.

Expected Behavior:
- For standard universes, any action referencing an unknown meter should produce a targeted compile error at `global_actions.yaml:<action.name>` naming the offending meter.
- Only when explicitly configured (e.g., variable-meter experiments) should trimming be allowed, with clear per-action warnings.

Actual Behavior:
- Unknown meter references are silently removed from costs/effects.
- Only a generic hint is emitted, and it does not identify which actions or meters were affected.

Root Cause:
- `_trim_meter_payload` was designed as a defensive measure for variable-meter universes, but its behavior is unconditional.
- There is no strict mode vs permissive mode distinction for action meter references.

Risk:
- Config bugs (typos or stale meter names) quietly shift the game’s economics.
- Operators and students may debug “weird” training dynamics without realizing that some action costs or effects are being ignored.

Proposed Directions:
- Introduce strict vs permissive behavior:
  - Default: unknown meters in action costs/effects → `UAC-ACT-002` errors with precise locations.
  - Optional flag (e.g., `training.allow_unknown_action_meters`) to enable trimming, with per-action warnings.
- Make the existing hint more specific: include action name and meter names, and emit once per action rather than once globally.

Tests:
- Unit: unknown meter in action costs → compile-time error in strict mode.
- Unit: in permissive mode, the same config yields a warning and a trimmed payload, and the warning points directly at `global_actions.yaml`.

Owner: compiler
Links:
- `src/townlet/universe/compiler_inputs.py:190–260`
