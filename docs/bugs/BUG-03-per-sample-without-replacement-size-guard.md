Title: PrioritizedReplayBuffer.sample uses replace=False without size guard

Severity: high
Status: open

Subsystem: training/replay-buffer (PER)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/prioritized_replay_buffer.py:112`

Description:
- `np.random.choice(..., replace=False)` requires `batch_size <= size_current`.
- The buffer lacks a precondition check and can throw a NumPy error if called early or externally.

Reproduction:
1) Fresh `PrioritizedReplayBuffer(capacity=100)`.
2) Call `.sample(batch_size=16)` before pushing 16 items.
3) NumPy raises due to sampling without replacement over a smaller set.

Expected Behavior:
- Raise a clear `ValueError` when `batch_size > size_current`.
- Optionally support `replace=True` when explicitly requested.

Actual Behavior:
- Uncaught NumPy exception; unclear error surface.

Root Cause:
- No guard in `sample`; relies on integration layer to avoid early calls.

Proposed Fix (Breaking OK):
- Add explicit guard: if `batch_size > self.size_current`, raise `ValueError`.
- Optionally add parameter `with_replacement: bool = False` and branch accordingly.

Migration Impact:
- External callers must satisfy the precondition or opt into replacement sampling.

Alternatives Considered:
- Always sample with replacement; changes learning dynamics and distribution.

Tests:
- New unit test: sampling before sufficient size raises ValueError.

Owner: training
Links:
- N/A
