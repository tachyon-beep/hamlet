Title: VFS semantic_type and curriculum_active metadata partially used and inconsistent between layers

Severity: medium
Status: open

Subsystem: vfs/observation_builder + universe/adapters + agent/networks
Affected Version/Branch: main

Affected Files:
- `src/townlet/vfs/schema.py:147`
- `src/townlet/vfs/observation_builder.py:40`
- `src/townlet/universe/adapters/vfs_adapter.py:31`
- `src/townlet/agent/networks.py:418`
- `docs/plans/2025-11-11-quick-05-structured-obs-masking.md`

Description:
- VFS Phase 1.5 introduced `ObservationField.semantic_type` and `curriculum_active` to support structured encoders and curriculum masking, and `VFSAdapter.build_observation_activity()` does use these fields to produce `ObservationActivity` for StructuredQNetwork.
- However:
  - `vfs_to_observation_spec()` ignores `semantic_type` and infers semantics from field names (BUG-28 already covers this partial issue).
  - There is no single, authoritative path for semantic/curriculum metadata: some code relies on VFS fields, some reconstructs semantics via name heuristics, and some only sees flattened slices without knowing which dims are padding vs active.
- The result is that the VFS metadata “kind of works” for StructuredQNetwork and RND, but is not consistently threaded through all relevant DTOs and consumers, making it easy for future changes to break masking/grouping semantics without test failures.

Reproduction:
- Create or modify VFS observation fields with non-standard IDs but explicit `semantic_type` and `curriculum_active` values:
  - At compile time, `VFSObservationSpecBuilder` and `VFSAdapter.build_observation_activity()` will respect these fields.
  - The adapter used to construct compiler observation specs (`vfs_to_observation_spec`) will still infer `semantic_type` from names (`_semantic_from_name`), potentially diverging from the explicit settings.
- StructuredQNetwork:
  - Consumes `ObservationActivity.group_slices`, not the original VFS ObservationField definitions, so any mismatch between semantic inference and explicit metadata may surface only indirectly (e.g., wrong grouping).

Expected Behavior:
- `semantic_type` and `curriculum_active` should be the single source of truth for observation grouping and active/padding dims:
  - All adapters/DTO builders should prefer and propagate these fields rather than re-deriving semantics from names.
  - Name-based heuristics should only act as a fallback when explicit metadata is absent.

Actual Behavior:
- Metadata usage is split:
  - `ObservationField.semantic_type` and `curriculum_active` are used by `VFSAdapter.build_observation_activity`.
  - `vfs_to_observation_spec` uses `_semantic_from_name(field.id)` and does not read `field.semantic_type`.
  - The architecture docs and quick-05 plan describe a consistent story, but the implementation is half evolved, increasing the risk of drift.

Root Cause:
- Semantic/curriculum metadata were added incrementally for structured encoders without refactoring all existing adapters and DTOs to treat them as canonical.
- Name-based heuristics were left in place for backward compatibility and convenience.

Proposed Fix (Breaking OK):
- Make `semantic_type` and `curriculum_active` first-class across the whole pipeline:
  - Update `vfs_to_observation_spec` to read `field.semantic_type` and only fall back to `_semantic_from_name` if it is absent (as in BUG-28).
  - Ensure `ObservationSpec` records semantic groups and, if needed, curriculum activity in a way that can be consumed by any downstream components (not only `ObservationActivity`).
  - Add invariants/tests that `ObservationActivity` group slices align with `ObservationSpec` semantics and `curriculum_active` flags.

Migration Impact:
- Observations may be grouped differently (but more correctly) when explicit `semantic_type` is provided; structured networks will see updated group_slices.
  - For configs that do not use explicit semantics, behavior should remain unchanged.

Alternatives Considered:
- Leave name-based heuristics as the primary mechanism and treat VFS `semantic_type` as advisory only:
  - Rejected: future VFS2 work will depend on reliable semantic labels, especially for automated network head construction.

Tests:
- Extend adapter tests:
  - Ensure that when `semantic_type` is set on VFS ObservationFields, both `ObservationSpec` and `ObservationActivity` reflect that semantic grouping.
  - Verify that `curriculum_active` is honored consistently in mask and slices.

Owner: VFS + compiler/adapters + networks
Links:
- `docs/plans/2025-11-11-quick-05-structured-obs-masking.md`
- `docs/arch-analysis-2025-11-13-1532/02-subsystem-catalog.md:111`
