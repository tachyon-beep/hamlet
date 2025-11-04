---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Environment

- **OS**: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- **Python Version**: [e.g., 3.13.0]
- **CUDA Version**: [e.g., 12.1 or "CPU only"]
- **GPU**: [e.g., NVIDIA RTX 3090 or "None"]
- **HAMLET Version**: [e.g., commit hash or branch name]
- **Installation Method**: [e.g., `uv sync`, `pip install`]

## Steps to Reproduce

1. Go to '...'
2. Run command '...'
3. Observe '...'
4. See error

## Minimal Reproducible Example

```python
# Paste minimal code that reproduces the bug
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv

# Your code here...
```

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

A clear description of what actually happened.

## Error Messages / Traceback

```
Paste full error message and traceback here
```

## Additional Context

- Configuration files used (if applicable)
- Training logs (if applicable)
- Screenshots or videos (if helpful)
- Any other context about the problem

## Possible Solution

If you have ideas about what might be causing this or how to fix it, please share!

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproducible example
- [ ] I have included the full error traceback
- [ ] I have specified my environment details
