Title: Implement segment tree (sum-tree) for PER sampling and updates

Severity: medium
Status: open

Subsystem: training/replay-buffer (PER)
Affected Version/Branch: main

Description:
- Current PER sampling computes probabilities over all priorities and calls `np.random.choice` each time (O(n)).
- Priority updates also cost O(1) per index but resampling recomputes full probs.
- A segment tree (sum-tree) enables O(log n) sampling and O(log n) updates, improving scalability.

Proposed Enhancement:
- Introduce a lightweight segment tree structure storing `priority ** alpha`.
- Support `sample(batch_size)` returning indices and normalized importance weights.
- Keep API identical to existing PER buffer.

Migration Impact:
- None at API level; internal performance improvement.

Tests:
- Deterministic unit tests against a fixed priority array (mock RNG) to validate sampling distribution and updates.

Owner: training
