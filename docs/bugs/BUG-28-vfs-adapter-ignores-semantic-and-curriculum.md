Title: vfs_to_observation_spec ignores semantic_type and curriculum_active from VFS fields

Severity: medium
Status: open

Subsystem: universe/adapters (VFS → DTO)
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/adapters/vfs_adapter.py:75` (semantic from name)

Description:
- VFSObservationField includes explicit `semantic_type` and `curriculum_active` metadata, but `vfs_to_observation_spec` infers semantics by name heuristics and discards curriculum activity (the latter is used later by ObservationActivity but initial spec could carry semantics directly).

Reproduction:
- Provide fields with custom names but explicit semantic_type (e.g., spatial alias). The adapter assigns None or wrong semantics.

Expected Behavior:
- Use `field.semantic_type` when available; fall back to name heuristics only if missing. Carry forward any relevant flags.

Actual Behavior:
- Heuristic-only; risks misclassification and downstream grouping errors.

Root Cause:
- Simplicity in adapter implementation; didn’t thread VFS metadata.

Proposed Fix (Breaking OK):
- Read `getattr(field, 'semantic_type', None)` and prefer it; only call `_semantic_from_name` as fallback.
- Ensure ObservationActivity builder and group_slices align with explicit semantics.

Migration Impact:
- Group slices may change to the correct semantics. Structured encoders benefit.

Tests:
- Adapter test with explicit semantic_type overriding name heuristics.

Owner: compiler/adapters
