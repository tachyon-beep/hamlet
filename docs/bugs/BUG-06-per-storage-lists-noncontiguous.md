Title: PrioritizedReplayBuffer stores lists of CPU tensors (non-contiguous, device churn)

Severity: medium
Status: open

Subsystem: training/replay-buffer (PER)
Affected Version/Branch: main

Affected Files:
- `src/townlet/training/prioritized_replay_buffer.py:41`

Description:
- PER uses Python lists of CPU tensors for observations/actions/rewards/etc.
- Sampling stacks and moves to device each time; adds overhead and fragmentation.

Reproduction:
- Profile training with PER vs standard buffer on GPU; observe extra host→device copies and Python overhead.

Expected Behavior:
- Use preallocated contiguous tensors on target device (like standard buffer) for efficient sampling.

Actual Behavior:
- List-backed storage; per-sample `.cpu()`/`.to(self.device)` churn.

Root Cause:
- Simpler storage approach chosen initially.

Proposed Fix (Breaking OK):
- Rework PER to mirror `ReplayBuffer`: preallocate tensors, keep `position`, `size_current`, and `priorities` tensor.
- Maintain priorities as Torch/NumPy array; ensure vectorized gather at sample.

Migration Impact:
- None at API level; internal behavior/perf changes only.

Alternatives Considered:
- Keep lists and cache device tensors; still complex and error-prone.

Tests:
- Existing PER tests must continue to pass; add perf notes.

Owner: training
Links:
- N/A
