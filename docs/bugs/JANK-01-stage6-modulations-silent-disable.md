Title: Stage 6 silently disables cascade modulations on cascades.yaml parse/validation errors

Severity: high
Status: open

Ticket Type: JANK
Subsystem: universe/compiler (Stage 6 optimization)
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/compiler.py:2040`
- `src/townlet/environment/cascade_config.py:214`

Description:
- Stage 6 (`_stage_6_optimize`) re-parses `cascades.yaml` via `load_full_cascades_config` to obtain modulations.
- The call is wrapped in a broad `except Exception: cascades_config = None`.
- If any error occurs (YAML, schema, or other) during this re-parse, all modulations are silently dropped from `modulation_data`.

Reproduction:
1) Create a pack whose `bars.yaml`/`cascades.yaml` load via `HamletConfig` (Stage 1) but fail `EnvironmentConfig`’s stricter validation.
2) Compile the pack.
3) Inspect `compiled.optimization_data.modulation_data` → empty, with no structured diagnostic pointing at `cascades.yaml`.

Expected Behavior:
- Either Stage 6 reuses the already-validated cascades config from Stage 1/4, or:
- Any failure in the cascades config used by Stage 6 surfaces as a `CompilationMessage` tied to `cascades.yaml`, not as a silent “no modulations”.

Actual Behavior:
- Any exception in `load_full_cascades_config` is swallowed.
- `modulation_data` ends up empty even though `cascades.yaml` defines modulations, changing meter dynamics semantics while compilation still “succeeds”.

Root Cause:
- Dual schema stack for cascades (config.cascade vs environment.cascade_config) plus a defensive `except Exception` in Stage 6.
- Stage 6 treats modulations as optional optimization data and does not integrate with the main `CompilationErrorCollector`.

Risk:
- Fitness/mood modulations can be defined but never applied at runtime with no clear signal to operators.
- Behavior diverges from the YAML specification, and the only symptom is missing modulation effects in training.

Proposed Directions:
- Narrow the exception handling in `_stage_6_optimize`:
  - Treat `ValidationError` / `YAMLError` from `load_full_cascades_config` as structured compiler errors against `cascades.yaml`.
  - Introduce a dedicated code (e.g., `UAC-OPT-001`) when modulations are disabled due to config issues.
- Longer term: unify cascades schema so Stage 6 operates on the same config object used in Stage 1/4 instead of re-parsing from disk.

Tests:
- Unit: craft a cascades config that passes `HamletConfig` but fails `EnvironmentConfig`; expect a clear diagnostic and no silent drop of modulations.
- Regression: valid cascades with modulations produce non-empty `optimization_data.modulation_data` and drive `MeterDynamics` as expected.

Owner: compiler
Links:
- `src/townlet/universe/compiler.py:2034–2108`
- `src/townlet/environment/cascade_config.py`
