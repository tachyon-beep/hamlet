Title: SequentialReplayBuffer eviction uses list pop(0) (O(n))

Severity: medium
Status: open

Subsystem: training/replay-buffer (sequential)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/sequential_replay_buffer.py:92`

Description:
- Evicting oldest episodes uses `self.episodes.pop(0)` which is O(n) per eviction.
- This can become costly under frequent eviction (long training, tight capacity).

Reproduction:
- Create many small episodes with capacity close to batch needs; observe CPU time in eviction loop.

Expected Behavior:
- O(1) eviction for the oldest episode.

Actual Behavior:
- O(n) per eviction due to list head removal.

Root Cause:
- Data structure choice (list) for a queue pattern.

Proposed Fix (Breaking OK):
- Replace list with `collections.deque` for `episodes` to enable O(1) popleft.
- Update indexing/append logic accordingly.

Migration Impact:
- Internal only; no API changes.

Alternatives Considered:
- Maintain head/tail indices; more complex than using `deque`.

Tests:
- Performance-focused (optional) or regression check that behavior/serialization remains identical.

Owner: training
Links:
- N/A
