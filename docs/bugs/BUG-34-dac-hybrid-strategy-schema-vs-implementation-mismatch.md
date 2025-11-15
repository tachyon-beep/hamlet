Title: DAC hybrid extrinsic strategy semantics diverge between schema and implementation

Severity: medium
Status: open

Subsystem: environment/dac_engine + config/drive_as_code
Affected Version/Branch: main

Affected Files:
- `docs/config-schemas/drive_as_code.md:517`
- `src/townlet/config/drive_as_code.py:167`
- `src/townlet/environment/dac_engine.py:400`
- `tests/test_townlet/unit/environment/test_dac_engine.py:955`

Description:
- The `drive_as_code.yaml` schema and docs describe the `hybrid` extrinsic strategy as:
  - `Formula: reward = Σ(weight_i × strategy_i)`
  - With future fields `strategies` (sub-strategies) and `weights` for combining different strategy outputs.
- The current implementation in `DACEngine._compile_extrinsic()` implements a completely different "simplified hybrid" behavior:
  - It uses `ExtrinsicStrategyConfig.bar_bonuses` and interprets them as a mixture of linear and shaped bar terms (using `scale` as weight and `center` as a pivot).
  - There is no support for composing multiple sub‑strategies or applying weights to their outputs; it's effectively another bar-based strategy similar to `weighted_sum` / `constant_base_with_shaped_bonus`.
- The unit test `test_hybrid_strategy` asserts this simplified behavior, which conflicts with the more general semantics described in the config schema and may confuse operators expecting strategy composition.

Reproduction:
- Consult `docs/config-schemas/drive_as_code.md` under "9. hybrid":
  - It documents `hybrid` as "weighted combination of multiple strategies" with `strategies` and `weights` fields (marked "NOT YET IMPLEMENTED").
- Inspect `src/townlet/config/drive_as_code.py`:
  - `ExtrinsicStrategyConfig` does not include `strategies` or `weights` fields; `hybrid` uses the same union as other strategies (primarily `bar_bonuses`, `base`, etc.).
- Inspect `src/townlet/environment/dac_engine.py`:
  - For `strategy.type == "hybrid"`, the code uses only `base` and `bar_bonuses`, treating `center`≈0 as linear and `center≠0` as shaped offsets.
- Result: the strategy that operators can actually use today is a "hybrid bar-bonus" variation, not a hybrid of multiple sub‑strategies as documented.

Expected Behavior:
- Either:
  - The `hybrid` strategy should align with the documented semantics (compose multiple sub‑strategies), or
  - The documentation and schema should clearly state that current `hybrid` is a simplified bar-based variant, with true strategy composition deferred to a later version.

Actual Behavior:
- Schema and docs suggest a powerful "compose strategies" mechanism that does not exist in code.
- Implementation provides a narrower behavior (weighted bar terms) that overlaps heavily with existing strategies and is not clearly distinguished in user-facing docs.

Root Cause:
- The original design envisioned a general strategy-composition `hybrid` mode, but the first iteration implemented a simplified version reusing `BarBonusConfig` to avoid adding new DTO types.
- Documentation and config-schema text were not updated to reflect the simplified implementation, leading to divergence.

Proposed Fix (Breaking OK):
- Option A (align implementation to docs; more work):
  - Extend `ExtrinsicStrategyConfig` to support a `strategies: list[ExtrinsicStrategyConfig]` field and a `weights: list[float]`.
  - In `_compile_extrinsic`, for `type="hybrid"`, compile each sub-strategy into a closure and combine their outputs with the specified weights.
  - Deprecate the current bar-bonus hybrid behavior or factor it into a separate strategy name (e.g., `hybrid_bars`).
- Option B (align docs to implementation; minimal):
  - Update `docs/config-schemas/drive_as_code.md` to describe the current "simplified hybrid" semantics, remove or clearly mark the multi-strategy composition as future work.
  - Explicitly state that `hybrid` uses `bar_bonuses` fields and does not accept nested strategies yet.

Migration Impact:
- If Option A is chosen:
  - Existing configs using `type: hybrid` with `bar_bonuses` will need migration, either to a renamed strategy or to a sub-strategy list form.
  - Tests like `test_hybrid_strategy` will need to be updated to reflect new semantics.
- If Option B is chosen:
  - No code changes, but users lose the expectation of a general hybrid composition until a future version.

Alternatives Considered:
- Remove `hybrid` entirely until full semantics are implemented:
  - Simplifies mental model but drops a partially useful strategy; may not be worth the churn pre‑v1.0.

Tests:
- If implementing full hybrid:
  - Add unit tests where `hybrid` combines `multiplicative` and `constant_base_with_shaped_bonus` sub-strategies with different weights and verify the weighted sum behavior.

Owner: DAC engine
Links:
- `docs/config-schemas/drive_as_code.md:517`
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md` (DAC section)
