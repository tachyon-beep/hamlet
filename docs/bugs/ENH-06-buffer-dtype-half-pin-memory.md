Title: Support dtype selection and pinned CPU memory to improve throughput

Severity: low
Status: open

Subsystem: training/replay-buffer (standard, PER)
Affected Version/Branch: main

Description:
- Training loops may benefit from half precision (`float16`/`bfloat16`) and pinned CPU memory for faster H2D copies.
- Buffers currently hardcode `float32` CPU tensors.

Proposed Enhancement:
- Add buffer-level `dtype` and `pin_memory` options; allocate storages accordingly.
- Ensure device casting in sample paths preserves dtype; add asserts in tests.

Migration Impact:
- Backwards compatible defaults remain `float32` and non-pinned.

Tests:
- Sampling correctness for half precision; verify `.is_pinned()` on CPU tensors when enabled.

Owner: training
