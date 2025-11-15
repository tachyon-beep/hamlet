Title: Accept `torch.Generator` or seed for reproducible sampling

Severity: low
Status: open

Subsystem: training/replay-buffer (standard, sequential, PER)
Affected Version/Branch: main

Description:
- Sampling uses global RNG (torch/numpy/random); no hook for deterministic sampling per run or test.

Proposed Enhancement:
- Add optional `generator: torch.Generator | None` to sampling APIs and use it for `randperm`/`randint`.
- For PER, thread either a NumPy `Generator` or bridge via seed.

Migration Impact:
- Backwards compatible; default remains current randomness.

Tests:
- Reproducibility test where fixed generator yields identical index sets.

Owner: training
