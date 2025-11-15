Title: Add optional `obs_dim`, `dtype`, and `pin_memory` to ReplayBuffer init for preallocation

Severity: medium
Status: open

Subsystem: training/replay-buffer (standard)
Affected Version/Branch: main

Description:
- ReplayBuffer allocates lazily on first push; tests/fixtures expect (or try to use) preallocation.
- No control over dtype or pinned CPU memory which can help GPU throughput.

Proposed Enhancement:
- Extend `__init__` with optional `obs_dim: int | None = None`, `dtype: torch.dtype = torch.float32`, and `pin_memory: bool = False`.
- If `obs_dim` provided, preallocate all storages immediately with chosen dtype and pinning when on CPU.

Migration Impact:
- Backwards compatible; enables cleaner fixtures and performance tuning.

Tests:
- Unit tests for preallocation path, dtype correctness, and pin_memory flag effect on CPU tensors.

Owner: training
