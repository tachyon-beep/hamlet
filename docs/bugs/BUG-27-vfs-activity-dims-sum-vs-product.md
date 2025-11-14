Title: VFSAdapter.build_observation_activity uses sum of shape dims instead of product

Severity: high
Status: open

Subsystem: universe/adapters (VFS → DTO)
Affected Version/Branch: main

Affected Files:
- `src/townlet/universe/adapters/vfs_adapter.py:154` (field_dims = int(sum(field.shape)))

Description:
- ObservationActivity’s active_mask is flattened by repeating per-field according to `field_dims`.
- The adapter computes `field_dims` as the sum of shape entries, not the product. For non-scalar vector fields, this undercounts (e.g., shape [4] → 4 OK; shape [2,2] → incorrectly 4 vs intended 4? For general case, using sum breaks if shapes were multi-dim; in current VFS we mostly use vecN so shapes are 1-D, but the method contradicts `_flatten_dims` used elsewhere and is fragile.)

Reproduction:
- Add a field with shape [2, 3] (future-proofing) or consider conceptual mismatch with `_flatten_dims` which uses product. The activity mask length can diverge from ObservationSpec.total_dims.

Expected Behavior:
- Use product of shape dimensions to match flattened observation length.

Actual Behavior:
- Sums shape entries; inconsistent with ObservationSpec flattening.

Root Cause:
- Implementation uses `sum(shape)` instead of multiplicative flattening.

Proposed Fix (Breaking OK):
- Replace `field_dims = int(sum(field.shape))` with a product-based computation (same helper as `_flatten_dims`). Ensure alignment with ObservationSpec.

Migration Impact:
- Active masks will change for any fields with multi-dim shapes; safer and correct.

Tests:
- Add test that ObservationActivity.total_dims equals ObservationSpec.total_dims for a mixed set of fields.

Owner: compiler/adapters
