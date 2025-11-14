Title: DAC composition.normalize and clip fields are ignored at runtime

Severity: medium
Status: open

Subsystem: environment/dac_engine + config/drive_as_code
Affected Version/Branch: main

Affected Files:
- `src/townlet/config/drive_as_code.py:314`
- `src/townlet/environment/dac_engine.py:843`
- `docs/config-schemas/drive_as_code.md:234`
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md:860`

Description:
- The Drive As Code schema exposes `composition.normalize` (tanh normalization) and `composition.clip` (min/max clipping) as first-class fields in `CompositionConfig`, and the config-schema docs describe them as active features.
- However, `DACEngine.calculate_rewards()` never reads or applies these settings: it always returns `total_reward = extrinsic + intrinsic + shaping` without optional normalization or clipping.
- This creates a misleading contract: configuration and documentation suggest that operators can bound reward ranges declaratively, but the engine ignores those settings, so extreme rewards are unbounded and downstream logs/checkpoints do not reflect the intended shaping.

Reproduction:
- Create or modify a `drive_as_code.yaml` to enable composition controls, e.g.:
  - `composition.normalize: true`
  - `composition.clip: {min: -10.0, max: 10.0}`
- Run a training session where extrinsic/shaping can produce rewards outside the [-10, 10] range (e.g., deliberately high bar bonuses).
- Inspect rewards as logged by `VectorizedPopulation` (e.g., via TensorBoard or direct prints):
  - Observed reward magnitudes will exceed the configured clip bounds.
  - The distribution will not be squashed into [-1, 1] as implied by `normalize: true`.

Expected Behavior:
- DAC composition should honor the configuration:
  - If `normalize` is true, apply a non-linear normalization (e.g., `torch.tanh`) to `total_reward` before returning it.
  - If `clip` is not null, clamp `total_reward` into `[clip["min"], clip["max"]]` after normalization (or in a clearly documented order).
- Documentation and runtime behavior should agree on whether composition supports normalization/clipping.

Actual Behavior:
- `CompositionConfig` is fully defined and validated, and tests assert default values.
- `DACEngine.__init__` only reads `log_components` and `log_modifiers` from `composition` and ignores the rest.
- `calculate_rewards()` computes `total_reward = extrinsic + intrinsic + shaping` and returns it without any normalization or clipping, regardless of `composition.normalize` or `composition.clip`.

Root Cause:
- CompositionConfig was designed to include normalization/clipping, but the runtime implementation in `DACEngine` only wired up the logging fields and never added the corresponding operations in `calculate_rewards()`.
- The schema docs and architecture report (subsystem catalog) assumed these features would be implemented, but the implementation stopped at DTO definition.

Proposed Fix (Breaking OK):
- Implement composition controls in `DACEngine.calculate_rewards()`:
  - After computing `total_reward`, read `composition.normalize` and `composition.clip` from `self.dac_config.composition`.
  - If `normalize` is true, apply `torch.tanh(total_reward)` (or another documented normalization) and optionally scale to a known range if needed.
  - If `clip` is not None, apply `torch.clamp(total_reward, min=clip["min"], max=clip["max"])`.
  - Ensure operations respect `dones` (dead agents should still have zero total reward).
- Update docs in `docs/config-schemas/drive_as_code.md` to describe the exact order of operations (e.g., "compose → normalize → clip") and any interactions with downstream scaling (e.g., if Population later changes reward handling).

Migration Impact:
- Existing configs that set `normalize` or `clip` currently have no effect; after the fix, their semantics become active.
  - For existing curriculum packs, default `normalize=false` and `clip=null` means behavior remains unchanged.
  - For any experiments that set these fields expecting behavior that never materialized, rewards will change and training curves may look different; this is acceptable pre‑v1.0 but should be called out in release notes.

Alternatives Considered:
- Remove `normalize` and `clip` from the schema and docs entirely:
  - Simplifies DAC but contradicts the design direction and reduces control over reward scales.
- Implement composition controls outside DAC (e.g., in Population or training loop):
  - Increases coupling and makes rewards less declarative; DAC is the natural place to apply them.

Tests:
- Extend `tests/test_townlet/unit/environment/test_dac_engine.py`:
  - Add a test where `composition.clip` is set and extrinsic/shaping clearly exceed bounds; assert that returned `total_reward` values are clamped.
  - Add a test for `composition.normalize: true` where raw total rewards are large, and assert output is within [-1, 1] (or whatever normalization is documented).

Owner: DAC engine
Links:
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md:860` (Missing Reward Normalization / Missing Reward Clipping)
- `docs/config-schemas/drive_as_code.md`
