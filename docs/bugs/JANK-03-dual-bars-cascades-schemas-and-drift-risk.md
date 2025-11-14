Title: Bars/cascades modeled twice (config.cascade vs environment.cascade_config) with drift risk

Severity: medium
Status: open

Ticket Type: JANK
Subsystem: universe/compiler + environment/cascade_config
Affected Version/Branch: main

Affected Files:
- `src/townlet/config/cascade.py`
- `src/townlet/environment/cascade_config.py`
- `src/townlet/universe/compiler.py:2034`

Description:
- Bars and cascades are represented by two schema stacks:
  - `townlet.config.cascade` used via `HamletConfig` for Stage 1/2/4 validation and metadata.
  - `townlet.environment.cascade_config` used for environment-facing behavior and re-parsed again in Stage 6 to extract modulations.
- Stage 6 uses both:
  - `raw_configs.cascades` to build `cascade_data` for meter dynamics, and
  - `load_full_cascades_config` for modulations (see JANK-01).

Reproduction:
1) Introduce a subtle schema difference between `config.cascade` and `environment.cascade_config` (e.g., new field validated only in one).
2) Configure `cascades.yaml` to rely on that field.
3) Compile the pack; observe that:
   - Stage 1/4 run using `HamletConfig`’s schema.
   - Stage 6 may behave differently depending on how `EnvironmentConfig` interprets the same YAML.

Expected Behavior:
- There should be a single canonical representation for bars/cascades.
- All compiler stages (including Stage 6) should operate on the same in-memory config object rather than re-parsing the same files under a different schema.

Actual Behavior:
- The same YAML is parsed by two different stacks, with potentially different validation rules.
- Stage 6 needs a second parse via `EnvironmentConfig`, and failures are currently handled via a broad `except` (see JANK-01).

Root Cause:
- Historical layering: initial config work used `townlet.config.cascade`, while later runtime-oriented cascade work introduced `environment.cascade_config`.
- Stage 6 optimization ended up bridging the gap by reloading `cascades.yaml` instead of reusing the already-loaded DTOs.

Risk:
- Schema or validation changes in one stack may not be mirrored in the other, leading to:
  - Valid configurations that behave differently in optimization vs cross-validation.
  - Silent feature loss (e.g., missing modulations) or inconsistent behavior between different compiler stages.

Proposed Directions:
- Consolidate to a single cascade schema:
  - Option A: Make `environment.cascade_config.EnvironmentConfig` the canonical model and embed it directly in `HamletConfig`.
  - Option B: Promote `townlet.config.cascade.CascadesConfig` and have `EnvironmentConfig` adapt from it.
- Refactor Stage 6 to use the same in-memory object created in Stage 1/2 instead of reloading `cascades.yaml` from disk.

Tests:
- Regression: create a pack with a modulation and a cascade relying on newer fields; confirm behavior is identical across all stages.
- Failure path: deliberately break cascades.yaml and assert that both Stage 4 and Stage 6 report compatible diagnostics.

Owner: compiler
Links:
- `src/townlet/universe/compiler.py:2034–2108`
- `src/townlet/config/cascade.py`
- `src/townlet/environment/cascade_config.py`
