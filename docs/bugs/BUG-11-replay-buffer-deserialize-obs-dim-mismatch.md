Title: ReplayBuffer.load_from_serialized does not reinitialize on obs_dim mismatch

Severity: medium
Status: open

Subsystem: training/replay-buffer (standard)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/replay_buffer.py:192`

Description:
- When storage already exists and the serialized `obs_dim` differs from current storage, the code does not reinitialize.
- This can cause shape errors or silent truncation when restoring from a checkpoint created with a different observation spec.

Reproduction:
1) Create buffer; push with `obs_dim = N`; serialize.
2) Create buffer; perform a first push with `obs_dim = M != N` to allocate storage; then call `load_from_serialized`.
3) Tensors retain the current allocation (M) but state expects N.

Expected Behavior:
- On load, if `obs_dim` or `capacity` differ, reinitialize storage or raise with remediation guidance.

Actual Behavior:
- Reinitializes only if storage is `None`.

Root Cause:
- `if self.observations is None:` guards reallocation; no shape validation otherwise.

Proposed Fix (Breaking OK):
- Validate serialized shapes vs current; if mismatch, reinitialize storages to serialized capacity and shapes, or raise.

Migration Impact:
- Callers must not “pre-warm” buffers with different shapes before loading, or accept that load will reinitialize.

Alternatives Considered:
- Always reinitialize on load; safest and simplest.

Tests:
- Negative test: mismatch in `obs_dim` raises.
- Positive test: clean reinit path restores successfully.

Owner: training
Links:
- N/A
