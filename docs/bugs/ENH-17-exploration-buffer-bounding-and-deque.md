Title: Use deque and explicit maxlen for RND.obs_buffer; bound memory

Severity: low
Status: open

Subsystem: exploration/RND
Affected Version/Branch: main

Description:
- `obs_buffer` is a Python list trimmed in chunks; use `collections.deque(maxlen=training_batch_size*2)` for O(1) pops and bounded memory.

Proposed Enhancement:
- Replace list with `deque`; bound to a small multiple of `training_batch_size` (e.g., 2×).
- Minor performance/footprint improvement under heavy training.

Migration Impact:
- Internal change; API stable.

Tests:
- None needed beyond existing coverage.

Owner: exploration
