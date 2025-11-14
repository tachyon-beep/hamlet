Title: Expose weight initialization scheme in BrainConfig and apply in factories

Severity: low
Status: open

Subsystem: agent
Affected Version/Branch: main

Description:
- Networks rely on default PyTorch initialization; allow configuring init (Kaiming, Xavier, orthogonal) for MLP/CNN/LSTM layers.

Proposed Enhancement:
- Extend BrainConfig with `init` block and apply in NetworkFactory after construction.

Migration Impact:
- Default remains PyTorch init; opt-in for experiments.

Tests:
- Smoke tests for each init mode; ensure no shape errors.

Owner: agent
