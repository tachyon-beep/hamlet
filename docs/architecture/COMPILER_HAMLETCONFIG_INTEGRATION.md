# Universe Compiler + HamletConfig Integration

**Scope**: Describe how the Universe Compiler (Task-004A) consumes the already-implemented HamletConfig (Task-003) and the new shared loaders so we avoid duplicating validation logic.

## Data Flow

```
config_pack/
├─ training.yaml  ─┐
├─ bars.yaml      │
├─ cascades.yaml  │
├─ affordances.yaml│
├─ cues.yaml       │  HamletConfig.load(config_dir)
├─ substrate.yaml  │        ↓
├─ variables_reference.yaml (VFS loader)
└─ configs/global_actions.yaml (global loader)
```

1. `HamletConfig.load(config_dir)` remains the entry point for training/environment/population/curriculum/exploration/bars/cascades/affordances/substrate/cues. Stage 1 should never re-read those files directly—reuse the DTO outputs.
2. Two additional loaders fill in what HamletConfig does not cover:
   - `load_variables_reference_config(config_dir)` in `townlet.vfs.schema` returns `list[VariableDef]` for Stage 2’s VFS compiler.
   - `load_global_actions_config()` in `townlet.environment.action_config` returns an `ActionSpaceConfig` wrapper for the action composer.
3. Stage 1 collects these into `RawConfigs`:

```python
@dataclass
class RawConfigs:
    hamlet_config: HamletConfig
    variables_reference: list[VariableDef]
    global_actions: ActionSpaceConfig
```

## Responsibilities

- **HamletConfig (Task-003)**: structural + cross-config validation, exposes typed DTOs reused by downstream stages. No compiler stage should open YAML files it already parsed.
- **Compiler Stage 1**: orchestrates `HamletConfig.load`, `load_variables_reference_config`, and `load_global_actions_config`, populating `RawConfigs` for later stages.
- **Later Stages**: read sections via properties (`raw_configs.bars`, `raw_configs.cascades`, etc.) to ensure there is exactly one source of truth.

## Migration Guidance

Existing runtime code (e.g., `VectorizedHamletEnv`) still loads configs independently. During Task-004A implementation we:
1. Keep HamletConfig-based compilation path for compiler/testing work.
2. Gradually migrate runtime consumers to accept `CompiledUniverse` and drop direct YAML loading once the compiler is stable.
3. Eventually delete the legacy loaders in runtime modules after all consumers use compiled artifacts.

## Strict DTO Enforcement

- All compiler-facing DTOs now specify `ConfigDict(extra="forbid")`. Any unknown YAML key is rejected during Stage 1 instead of being silently ignored later in the pipeline.
- Operators should run `uv run pytest tests/test_townlet/unit/config/test_training_config_dto.py -k extra` (or equivalent) after adding new fields to confirm schemas were updated everywhere.
- The compiler’s Stage 4 guard rails (`UAC-VAL-006`) treat over-sized packs as a security violation, so prefer reusing existing DTOs over inventing ad-hoc sections.

## Training-Driven Action Masking

- `training.enabled_actions` is consumed directly by Stage 1 via HamletConfig. The field must list canonical action names from `configs/global_actions.yaml` and/or substrate defaults.
- Stage 1 threads the list into `RawConfigs.global_actions`, so `_stage_5_build_rich_metadata` and `VectorizedHamletEnv` both read the same enabled flags. No runtime recomputation is required.
- Config packs should check in explicit lists. See `configs/L0_0_minimal/training.yaml` (movement + REST only) and `configs/L0_5_dual_resource/training.yaml` (movement + REST/MEDITATE) for templates.

## Operator Migration Notes

1. **Clean up stray YAML keys** – `extra="forbid"` will now flag typos like `grid_sie` or `trainig`. Fix the source YAML rather than suppressing the error.
2. **Adopt `enabled_actions`** – add the field under the `training:` block, copying one of the reference packs, to control curriculum action unlocks.
3. **Expect guard-rail errors** – hitting limits such as `MAX_AFFORDANCES` now surfaces `UAC-VAL-006`. Trim the pack or split it into multiple scenarios before re-running the compiler.
