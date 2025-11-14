Title: Add `clear()` and `stats()` API to buffers for manageability

Severity: low
Status: open

Subsystem: training/replay-buffer (standard, sequential, PER)
Affected Version/Branch: main

Description:
- No simple way to reset buffers mid-run or inspect occupancy/memory usage without internal poking.

Proposed Enhancement:
- Add `clear()` method to reset size/position and drop stored data.
- Add `stats()` returning dict: `{size, capacity, occupancy_ratio, memory_bytes (approx), device}`.
- Sequential: include `num_episodes` and `num_transitions`.

Migration Impact:
- Backwards compatible; new API only.

Tests:
- Unit tests for `clear()` idempotence and `stats()` content.

Owner: training
