# Universe Compiler Guide

**Document Type**: Architecture Guide  \\
**Status**: Draft  \\
**Version**: 0.1  \\
**Audience**: RL / systems engineers, infra maintainers  \\
**Technical Level**: Advanced

## AI-Friendly Summary (Skim This First!)

**What**: End-to-end reference for the seven-stage UniverseCompiler pipeline located in `src/townlet/universe/compiler.py`, including inputs, validation rules, emitted artifacts, and caching semantics.
**Why**: TASK-004A promotes “compile once, execute many” so environments stop re-loading YAML at runtime; this doc is the single place that ties the architecture spec in `docs/architecture/COMPILER_ARCHITECTURE.md` to the actual implementation.
**Who Must Read**: Anyone modifying config loaders, validation rules, or runtime consumers.
**Reading Strategy**: Section 2 for a quick refresher on the pipeline, Sections 4–6 before changing validation/metadata logic, Section 7 for cache/provenance questions.

## 1. Scope & Responsibilities

- **Inputs**: Full config packs (`training.yaml`, `bars.yaml`, `cascades.yaml`, `affordances.yaml`, `cues.yaml`, `substrate.yaml`, `variables_reference.yaml`) plus shared `configs/global_actions.yaml`.
- **Outputs**: Immutable `CompiledUniverse` artifacts (`src/townlet/universe/compiled.py`) containing the canonical DTOs, observation spec, metadata, optimization tensors, and runtime views.
- **Consumers**: `VectorizedHamletEnv` (runtime), `DemoRunner` / training pipeline, checkpoint utilities, and testing tools (`scripts/validate_substrate_runtime.py`).
- **Out of Scope**: BAC/Brain compiler, substrate implementations, or curriculum logic beyond the metadata needed to initialize them.

## 2. Seven-Stage Pipeline (Implementation View)

| Stage | Location | Purpose |
| --- | --- | --- |
| 1. Parse YAML | `RawConfigs.from_config_dir` | Load every required file via shared loaders, attach `SourceMap` entries, and enforce “no defaults” validation. |
| 2. Build Symbol Tables | `_stage_2_build_symbol_tables` | Register meters, cascades, variables, actions, affordances, cues; fail fast on duplicates. |
| 3. Resolve References | `_stage_3_resolve_references` | Walk every cross-file reference (affordance effects, action costs, training overrides) with UAC error codes. |
| 4. Cross-Validation | `_stage_4_cross_validate` + helpers + `CuesCompiler` | Enforce spatial feasibility, economic balance, cascade cycles, temporal rules, substrate/action alignment, capability semantics, and cue integrity. |
| 5. Metadata & ObservationSpec | `_stage_5_compute_metadata`, `_stage_5_build_rich_metadata` | Use `VFSObservationSpecBuilder` + `vfs_to_observation_spec` to compute dims, populate meter/action/affordance metadata, and derive hash/provenance. |
| 6. Optimization Data | `_stage_6_optimize` | Pre-compute tensors (base depletions, cascade/modulation tables, hourly action masks, position maps) in deterministic order. |
| 7. Emit Artifact | `_stage_7_emit_compiled_universe` | Construct frozen `CompiledUniverse`, serialize/cache via MessagePack, expose runtime DTOs.

The compiler is intentionally pure and deterministic: given the same YAML content and compiler version, the emitted artifact (including `.metadata.config_hash` and `.metadata.provenance_id`) is stable, which unlocks cache hits and checkpoint validation.

## 3. Key Data Structures

- `RawConfigs`: staged DTO bundle exposing convenient properties (`.bars`, `.affordances`, `.cues`, `.substrate`, etc.) while preserving source-map metadata.
- `UniverseSymbolTable`: central registry for Stage 2→4 (meters, affordances by id/name, cascades, cues, variables, actions).
- `CompiledUniverse`: frozen dataclass with DTO copies, observation spec, rich metadata, optimization tensors, and helper methods (`create_environment`, `to_runtime`, `check_checkpoint_compatibility`).
- `ObservationSpec` & `ObservationField`: deterministic UUID-backed observation definitions derived from VFS exposures.
- `OptimizationData`: tensors/lookup tables consumed by runtime (action mask, cascade data, modulation data, position maps).

## 4. Validation & Errors

- **Error Catalog**: Stage 3+ emit structured `CompilationMessage`s with codes (`UAC-RES-*`, `UAC-VAL-*`, `UAC-ACT-*`), file:line info via `SourceMap`, and actionable hints.
- **CuesCompiler**: dedicated helper (`src/townlet/universe/cues_compiler.py`) validates `cues.yaml` (meter references, [0,1] thresholds, domain coverage & non-overlap for visual cues) and keeps that logic modular for future expansions.
- **Security Limits**: Hard caps (`MAX_METERS`, `MAX_AFFORDANCES`, etc.) guard against malicious or accidental config explosion.
- **Feasibility Guardrails**: Stage 4 models spatial feasibility, income/cost balance, cascade cycles, temporal operating hours, availability ranges, effect-pipeline compatibility, substrate/action constraints, and sustainability (critical meters, capacity vs. agents).

Any collector with accumulated issues uses `CompilationErrorCollector.check_and_raise()` so contributors see all failures at once; warnings (e.g., economic stress) accompany raised errors so operators retain full context.

## 5. Metadata, ObservationSpec, & Runtime Contract

- ObservationSpec generation relies on the VFS schema (`variables_reference.yaml` + explicit `exposed_observations`). Full observability strips `local_window`, POMDP strips `grid_encoding` per training config.
- `UniverseMetadata` tracks: counts (meters, affordances, actions), dims (observation, inferred grid cells), economic stats, tick cadence, compiler/config versions, `config_hash`, `provenance_id`, Git SHA, and Python/Torch/Pydantic versions.
- Rich metadata surfaces affordable introspection for training logs (`ActionSpaceMetadata`, `MeterMetadata`, `AffordanceMetadata`). This information is consumed by `VectorizedHamletEnv`, `DemoRunner`, and analytics/telemetry hooks.
- Runtime integration happens via `CompiledUniverse.to_runtime()` which emits read-only DTO proxies for environment/curriculum/exploration constructors—no YAML reads happen past compile time (see `tests/test_townlet/unit/environment/test_vectorized_env_runtime.py`).

## 6. Caching & Provenance

- Cache artifacts live in `<config_dir>/.compiled/universe.msgpack`. The compiler normalizes YAML (sorted keys) before hashing, then folds in file names to avoid collisions.
- `_build_cache_fingerprint()` compares both config hash and provenance id (compiler version + git SHA + Python/Torch/Pydantic versions). Any change in YAML content *or* toolchain invalidates the cache.
- `CompiledUniverse.save_to_cache/load_from_cache` perform MessagePack serialization with defensive fallbacks (corrupt cache triggers full recompilation + warning).
- Checkpoints store `config_hash`, obs/action dims, meter counts, and observation-field UUIDs (`townlet/training/checkpoint_utils.py`). Loading mismatched checkpoints yields friendly warnings/errors.

## 7. Usage Patterns

```python
from pathlib import Path
from townlet.universe.compiler import UniverseCompiler

compiler = UniverseCompiler()
compiled = compiler.compile(Path("configs/L1_full_observability"))

env = compiled.create_environment(num_agents=4)
runtime = compiled.to_runtime()
print(runtime.metadata.observation_dim)
```

- **Cache warm-up**: The first compile of a pack (~50–100 ms) writes `.compiled/universe.msgpack`; subsequent loads drop to single-digit milliseconds.
- **Headless validation**: Use `scripts/validate_substrate_runtime.py` to iterate over all packs and smoke-test runtime integration with compiled universes.
- **Training integration**: `DemoRunner` compiles once per run and reuses the runtime view for curriculum/exploration/population setup.

## 8. Testing Checklist

Run the targeted universe compiler suite before landing changes:

- `uv run pytest tests/test_townlet/unit/universe/test_universe_compiler_stage1.py`
- `uv run pytest tests/test_townlet/unit/universe/test_stage2_symbol_table.py`
- `uv run pytest tests/test_townlet/unit/universe/test_stage3_resolve.py`
- `uv run pytest tests/test_townlet/unit/universe/test_stage4_cross_validate.py`
- `uv run pytest tests/test_townlet/unit/universe/test_stage5_metadata.py`
- `uv run pytest tests/test_townlet/unit/universe/test_stage6_optimize.py`
- `uv run pytest tests/test_townlet/unit/universe/test_cues_compiler.py`
- `uv run pytest tests/test_townlet/unit/universe/test_compiled_universe.py`
- `uv run pytest tests/test_townlet/unit/environment/test_vectorized_env_runtime.py`

End-to-end sanity check: `uv run scripts/validate_substrate_runtime.py --config configs/L1_full_observability` (ensures compiled metadata matches runtime behavior).

## 9. Related Documents

- `docs/architecture/COMPILER_ARCHITECTURE.md` – canonical design rationale and diagrams.
- `docs/tasks/TASK-004A-COMPILER-IMPLEMENTATION.md` – task checklist and acceptance criteria for this workstream.
- `docs/tasks/STREAM-001-UAC-BAC-FOUNDATION.md` – broader roadmap covering downstream compilers (Brain/BAC).
- `docs/vfs/vfs-integration-guide.md` – explains how VFS variables map into observation specs.

## 10. Future Enhancements

- Incremental compilation (only re-run stages for changed YAML) once config packs grow further.
- Extended cue processing (e.g., derived cues, behavior cues) by expanding `CuesCompiler` with specialized passes.
- Shared cache registry for remote training clusters (content-addressed store instead of per-pack directories).

## 11. CLI Tooling

The repository now ships a lightweight CLI so ops/dev workflow can run compiler operations without writing ad-hoc scripts:

```bash
# Compile a pack (writes .compiled/universe.msgpack by default)
python -m townlet.compiler compile configs/L1_full_observability

# Inspect an existing artifact
python -m townlet.compiler inspect configs/L1_full_observability/.compiled/universe.msgpack

# Inspect as JSON (for automation/CI logs)
python -m townlet.compiler inspect --format json configs/L1_full_observability/.compiled/universe.msgpack

# Validate config packs without touching cache (useful for CI lint checks)
python -m townlet.compiler validate configs/L1_full_observability

# Validate every config pack via CLI (script wraps the commands above)
python scripts/validate_compiler_cli.py
```

Flags:

- `compile --no-cache`: Skip cache usage entirely (always rebuild, don’t write artifacts).
- `inspect <artifact>`: Prints the metadata summary (universe name, counts, config hash) for debugging/deployment notes.
- `inspect --format json`: Emit a JSON payload with artifact path + metadata hashes for machine-readable auditing.
- `validate`: Equivalent to `compile --no-cache` but discards the result immediately, so `.compiled/` never appears—handy for CI. The helper script `scripts/validate_compiler_cli.py` iterates over every pack in `configs/`.
