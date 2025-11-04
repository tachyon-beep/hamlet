# Linting and Code Quality

This project enforces strict linting standards to maintain code quality and consistency.

## Quick Start

```bash
# Check for lint violations
uv run ruff check .

# Auto-fix violations
uv run ruff check . --fix

# Format code with black
uv run black src tests

# Type check
uv run mypy src/townlet
```

## Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically check code before committing:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run hooks manually on all files
pre-commit run --all-files
```

Once installed, the hooks will automatically run on every commit and prevent commits with lint violations.

## CI/CD Enforcement

All pull requests and pushes to main are checked by GitHub Actions:

- **Ruff Linter**: Enforces zero warnings (blocks merge if violations found)
- **Black Formatter**: Checks code formatting
- **MyPy**: Type checking on `src/townlet/`
- **No-Defaults Linter**: Ensures no unauthorized default parameters
- **Markdownlint**: Checks markdown documentation

**The build will fail if any lint violations are found.**

## Linting Rules

We enforce the following rule sets (configured in `pyproject.toml`):

- **F** - Pyflakes (undefined names, unused imports)
- **E/W** - pycodestyle (style errors and warnings)
- **I** - isort (import sorting)
- **N** - pep8-naming (naming conventions)
- **UP** - pyupgrade (modern Python syntax)
- **S** - flake8-bandit (security)
- **B** - flake8-bugbear (bug patterns)
- **PERF** - perflint (performance)
- **ARG** - flake8-unused-arguments
- And many more...

## Line Length

Maximum line length is **140 characters** (not 79 or 88).

## Common Fixes

### Unused Imports (F401)
```python
# ❌ Bad
import json
import os  # Unused

# ✅ Good
import json
```

### Modern Type Hints (UP006, UP045)
```python
# ❌ Bad
from typing import List, Optional
def process(items: List[int]) -> Optional[str]:
    ...

# ✅ Good
def process(items: list[int]) -> str | None:
    ...
```

### Unused Variables (F841)
```python
# ❌ Bad
result = some_function()  # Never used

# ✅ Good - Remove or prefix with _
_result = some_function()  # Intentionally unused
```

### Line Too Long (E501)
```python
# ❌ Bad
logger.info(f"Very long message with {many} {variables} that exceeds the 140 character limit")

# ✅ Good
logger.info(
    f"Very long message with {many} {variables} "
    f"that exceeds the 140 character limit"
)
```

## Bypassing Checks (Discouraged)

**We do NOT disable lint warnings.** If you encounter a false positive, discuss it with the team before adding any `# noqa` comments.

The only exception is for third-party code you cannot modify.

## Local Development Workflow

1. Write code
2. Run `uv run ruff check . --fix` to auto-fix
3. Run `uv run black src tests` to format
4. Run `uv run mypy src/townlet` for type checking
5. Commit (pre-commit hooks will run automatically)
6. Push (CI will verify everything)

## Baseline Statistics

As of 2025-11-04, the codebase has **0 lint warnings**.

```bash
# Check current status
uv run ruff check . --statistics
```

## Troubleshooting

**Problem**: Pre-commit hooks fail on commit

**Solution**: Run `uv run ruff check . --fix` and commit again

**Problem**: CI fails with "Lint violations found"

**Solution**: Pull latest changes, run linter locally, fix violations, push

**Problem**: Black and Ruff disagree on formatting

**Solution**: Black takes precedence - run `uv run black src tests` after ruff

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
