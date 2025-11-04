# Contributing to HAMLET

Thank you for your interest in contributing to HAMLET! This document provides guidelines and instructions for contributing to this pedagogical Deep Reinforcement Learning project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- **Python 3.13+** required
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
- **CUDA-capable GPU** optional but recommended for training
- **Node.js 18+** required for frontend development

### First-Time Setup

```bash
# Clone the repository
git clone https://github.com/tachyon-beep/hamlet
cd hamlet

# Install dependencies
uv sync

# Install development dependencies
uv sync --extra dev

# Install recording system dependencies (optional)
uv sync --extra recording

# Run tests to verify setup
uv run pytest
```

## Development Setup

### Python Environment

We use `uv` for fast, reliable dependency management:

```bash
# Create/sync virtual environment
uv sync

# Add a new dependency
uv pip install <package>

# Update pyproject.toml to reflect new dependency
# (edit [project.dependencies] or [project.optional-dependencies])
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev  # Start development server on http://localhost:5173
```

### IDE Configuration

**VS Code**: Recommended extensions:

- Python (Microsoft)
- Pylance
- Ruff
- Vetur or Volar (for Vue)

**PyCharm**: Configure pytest as test runner, enable Ruff integration

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clear, focused commits
- Follow code style guidelines (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run full test suite
uv run pytest

# Run specific test file
uv run pytest tests/test_townlet/test_your_feature.py

# Run with coverage
uv run pytest --cov=townlet --cov-report=term-missing

# Skip slow tests during development
uv run pytest -m "not slow"
```

### 4. Submit Pull Request

See [Pull Request Process](#pull-request-process) below.

## Testing

### Test Organization

- **Unit tests**: `tests/test_townlet/unit/` - Test individual components
- **Integration tests**: `tests/test_townlet/integration/` - Test component interactions
- **Recording tests**: `tests/test_townlet/recording/` - Test episode recording system

### Writing Tests

```python
import pytest
import torch

def test_your_feature():
    """Test description following numpy docstring format."""
    # Arrange
    input_data = torch.tensor([1.0, 2.0, 3.0])

    # Act
    result = your_function(input_data)

    # Assert
    assert result.shape == (3,)
    assert torch.allclose(result, expected_output)

# Mark slow tests (>5 seconds)
@pytest.mark.slow
def test_training_loop():
    ...

# Mark integration tests
@pytest.mark.integration
def test_full_training_cycle():
    ...

# Skip tests requiring GPU when unavailable
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_acceleration():
    ...
```

### Test Markers

- `@pytest.mark.slow` - Tests taking >5 seconds
- `@pytest.mark.gpu` - Tests requiring CUDA/GPU
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests

### Running Marked Tests

```bash
# Skip slow tests (default behavior)
uv run pytest -m "not slow"

# Run only integration tests
uv run pytest -m integration

# Run GPU tests only
uv run pytest -m gpu
```

## Code Style

### Python Code Style

We enforce code style with **Ruff** and **Black**:

```bash
# Format code with Black
uv run black src/ tests/

# Lint with Ruff
uv run ruff check src/

# Type check with mypy
uv run mypy src/townlet
```

### Style Guidelines

- **Line length**: 140 characters (configured in pyproject.toml)
- **Imports**: Sorted with isort (via Ruff)
- **Type hints**: Encouraged but not required (mypy lenient mode)
- **Docstrings**: NumPy style for public APIs
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: `_leading_underscore`

### Example Code Style

```python
"""Module docstring explaining purpose."""

import torch
import torch.nn as nn
from typing import Optional

class SimpleQNetwork(nn.Module):
    """Q-network for full observability environments.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension
    action_dim : int
        Action space dimension
    hidden_dim : int, optional
        Hidden layer size, by default 256
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network.

        Parameters
        ----------
        x : torch.Tensor
            Observation tensor of shape (batch_size, obs_dim)

        Returns
        -------
        torch.Tensor
            Q-values of shape (batch_size, action_dim)
        """
        return self.network(x)
```

## Documentation

### Adding Documentation

- **Code**: Add docstrings to public classes, functions, and modules
- **Architecture**: Add design docs to `docs/architecture/`
- **User guides**: Add how-to docs to `docs/manual/`
- **Research**: Add design exploration to `docs/research/`
- **Teachable moments**: Add pedagogical insights to `docs/teachable_moments/`

### Documentation Style

- Use **Markdown** for all documentation
- Include **code examples** where appropriate
- Keep docs **concise and focused**
- Update the relevant README when adding new docs

### Updating README

When adding major features, update:

- Main `README.md` with high-level overview
- `docs/README.md` to reference new documentation
- `CHANGELOG.md` to document changes

## Pull Request Process

### Before Submitting

1. **Run full test suite**: `uv run pytest`
2. **Check code style**: `uv run black src/ tests/ && uv run ruff check src/`
3. **Run type checker**: `uv run mypy src/townlet`
4. **Update CHANGELOG.md**: Add entry under `[Unreleased]`
5. **Update documentation**: If adding features or changing APIs
6. **Rebase on main**: `git pull --rebase origin main`

### PR Title Format

Use conventional commit style:

- `feat: Add recurrent network support for POMDP`
- `fix: Correct reward calculation in temporal mechanics`
- `docs: Update training level specifications`
- `refactor: Simplify observation encoding logic`
- `test: Add integration tests for curriculum progression`
- `chore: Update dependencies in pyproject.toml`

### PR Description Template

```markdown
## Summary
Brief description of changes (1-2 sentences)

## Motivation
Why are these changes needed? Link to issue if applicable.

## Changes
- Bullet list of specific changes
- Keep focused and atomic

## Testing
- [ ] Added unit tests
- [ ] Added integration tests (if applicable)
- [ ] All tests pass locally
- [ ] Tested manually (describe if relevant)

## Documentation
- [ ] Updated relevant documentation
- [ ] Updated CHANGELOG.md
- [ ] Added docstrings to new code

## Checklist
- [ ] Code follows style guidelines (Black + Ruff)
- [ ] Type hints added where appropriate
- [ ] No breaking changes (or documented if unavoidable)
- [ ] Commits are clean and well-described
```

### Review Process

1. **Automated checks** run on all PRs (lint, tests, type checking)
2. **Code review** by maintainer(s)
3. **Discussion** and iteration if needed
4. **Approval** and merge once checks pass and review approved

## Reporting Issues

### Bug Reports

Use the bug report template (`.github/ISSUE_TEMPLATE/bug_report.md`) and include:

- **Environment**: Python version, OS, CUDA version
- **Steps to reproduce**: Minimal example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full traceback if applicable

### Feature Requests

Use the feature request template (`.github/ISSUE_TEMPLATE/feature_request.md`) and include:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches
- **Pedagogical value**: How does this help students learn RL? (if applicable)

### Questions

For questions about usage or architecture:

- Check existing documentation in `docs/`
- Search existing issues and discussions
- Open a discussion (not an issue) for general questions

## Development Philosophy

HAMLET follows a **research-driven pedagogical** approach:

### Core Principles

1. **UNIVERSE_AS_CODE**: All game mechanics configurable via YAML
2. **Teachable Moments**: Preserve "interesting failures" as learning opportunities
3. **Research-Driven**: Explore design space before implementing
4. **Progressive Complexity**: Build from simple (L0) to complex (L3)
5. **No Black Boxes**: Students should understand every component

### When to Research vs. TDD

- **Research first** when:
  - Multiple viable approaches with unclear tradeoffs
  - Architectural decisions with long-term impact
  - Novel RL techniques or curriculum design

- **TDD directly** when:
  - Requirements are clear
  - Single obvious approach
  - Bug fixes or incremental improvements

See `docs/methods/RESEARCH-PLAN-REVIEW-LOOP.md` for detailed methodology.

## Getting Help

- **Documentation**: Start with `docs/README.md`
- **Architecture**: Read `docs/architecture/TOWNLET_HLD.md`
- **Issues**: Search existing issues or open a new one
- **Discussions**: Use GitHub Discussions for open-ended questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see [LICENSE](LICENSE)).

---

Thank you for contributing to HAMLET! Your efforts help students learn graduate-level RL in an intuitive, hands-on way.
