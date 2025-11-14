Title: ReplayBuffer.push writes items in a Python loop (non-vectorized)

Severity: medium
Status: open

Subsystem: training/replay-buffer (standard)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/replay_buffer.py:78`

Description:
- `push` iterates per transition and assigns into storage tensors index-by-index.
- This leaves perf on the table; vectorized slice writes are both simpler and faster.

Reproduction:
- Profile pushing large batches vs a vectorized two-slice implementation.

Expected Behavior:
- Use one or two slice assignments depending on wrap-around.

Actual Behavior:
- Python loop over `batch_size` with repeated device-bound assignments.

Root Cause:
- Simplicity choice in initial implementation.

Proposed Fix (Breaking OK):
- Compute `end = (position + batch) % capacity`; perform either
  - single contiguous slice write, or
  - two-slice write (tail then head) per tensor.
- No API change; purely performance.

Migration Impact:
- None (behavior preserved).

Alternatives Considered:
- Keep loop; simpler but slower.

Tests:
- Ensure functional tests still pass; optional microbenchmarks in dev docs.

Owner: training
Links:
- N/A
