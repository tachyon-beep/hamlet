Title: PER probability calc can degenerate with zero-sum priorities

Severity: low
Status: open

Subsystem: training/replay-buffer (PER)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/prioritized_replay_buffer.py:109`

Description:
- `probs = priorities ** alpha; probs /= probs.sum()` assumes a positive sum.
- With all-zero priorities (unlikely but possible after manual resets), this will NaN/raise.

Reproduction:
- Construct buffer; explicitly zero out `priorities` and call `sample`.

Expected Behavior:
- Fallback to uniform distribution when the sum is zero.

Actual Behavior:
- Division by zero potential.

Root Cause:
- No guard around sum of probabilities.

Proposed Fix (Breaking OK):
- If `probs.sum() == 0`, set uniform probabilities over `size_current`.

Migration Impact:
- None.

Alternatives Considered:
- Add small epsilon before normalization; still fails for exact zeros.

Tests:
- Unit test: zero priorities → uniform sampling (no crash).

Owner: training
Links:
- N/A
