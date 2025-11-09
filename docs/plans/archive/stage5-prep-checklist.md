## Stage 5 Prep Checklist (Draft)

- **UniverseMetadata coverage (19 fields):**
  - Need fixture values for name, schema version, compiled timestamp, config hash, observation/action dims, counts for meters/affordances/actions, position dim, substrate type, agent slots, population size, curriculum id, environment hash, cues hash, affordance hash, bars hash, cascades hash.
  - Derive from `configs/L0_0_minimal` pack for deterministic baseline.

- **ObservationSpec wrapper tests:**
  - Validate `total_dims`, `get_field_by_name`, `get_fields_by_semantic_type`, and invalid index/dim guards.
  - Build synthetic `ObservationField` list (position, meters, affordance occupancy, temporal features) matching VFS builder outputs.

- **ObservationSpec generation (VFS integration):**
  - Using `VFSObservationSpecBuilder` with `variables_reference` from minimal pack, assert field ordering + dims.
  - Include exposures for scalar, vec2i, vecNf to exercise `_infer_shape`/normalization logic.

- **Rich metadata DTOs:**
  - `ActionSpaceMetadata`: ensure enabled mask/hints align with Stage 1 composed action space.
  - `MeterMetadata`: confirm index ordering matches `bars.yaml` indices and boolean flags propagate.
  - `AffordanceMetadata`: compute aggregate cost/effects from effect pipeline migration; include disabled affordance sample.

- **Config hash prep:**
  - Snapshot list of files/sections to hash (training.yaml, bars.yaml, cascades.yaml, affordances.yaml, substrate.yaml, cues.yaml, global_actions.yaml, variables_reference.yaml).
  - Decide whether to sort keys or rely on canonical serialization before Stage 5 implementation.
