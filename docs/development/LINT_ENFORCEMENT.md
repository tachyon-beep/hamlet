# Lint Enforcement Setup

## Overview

All lint checks are **required to pass** before code can be merged. Failures will block commits both locally (via pre-commit hooks) and in CI/CD (via GitHub Actions).

## Enforcement Layers

### 1. Pre-Commit Hooks (Local) ⚡

Pre-commit hooks run automatically before each `git commit` and will **block the commit** if any checks fail.

**Installation** (one-time setup):

```bash
uv pip install pre-commit
uv run pre-commit install
```

**Checks that run on commit**:

- ✓ Ruff (linting with auto-fix)
- ✓ Ruff formatter (formatting)
- ✓ Black (formatting)
- ✓ Trailing whitespace
- ✓ End-of-file fixer
- ✓ YAML syntax
- ✓ Large files check
- ✓ Merge conflict markers
- ✓ Line ending fixes

**Note**: Mypy type checking runs in CI only (not in pre-commit hooks) to avoid slow dependency installation.

**Bypassing pre-commit** (NOT recommended):

```bash
git commit --no-verify  # Skip pre-commit hooks (use sparingly!)
```

### 2. GitHub Actions CI/CD (Remote) 🚨

The `Lint` workflow runs on every push and pull request. **All checks must pass** before PRs can be merged.

**Checks that run in CI**:

1. **Ruff (lint)** - Zero warnings enforced
2. **Black (format)** - All code must be formatted
3. **Mypy (type check)** - Zero type errors enforced
4. **No-defaults linter** - No unauthorized defaults

**Location**: `.github/workflows/lint.yml`

**Status**: All lint checks are **required status checks** - PRs cannot merge until they pass.

## Local Development Workflow

### Quick Fix Commands

```bash
# Auto-fix most lint issues
uv run ruff check . --fix          # Fix linting violations
uv run black src tests             # Format code
uv run mypy src/townlet            # Check types (no auto-fix)

# Run all checks manually (same as CI)
uv run ruff check .
uv run black --check src tests
uv run mypy src/townlet --show-error-codes
python scripts/no_defaults_lint.py src/townlet/ --whitelist .defaults-whitelist.txt
markdownlint-cli2 "**/*.md" "!node_modules" "!.venv" "!frontend/node_modules"
```

### Pre-Commit Manual Run

```bash
# Run all pre-commit hooks manually (without committing)
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files
uv run pre-commit run mypy --all-files
```

### Typical Workflow

1. Write code
2. Run `uv run ruff check . --fix` to auto-fix lint issues
3. Run `uv run black src tests` to format code
4. Run `uv run mypy src/townlet` to check types
5. `git add` your changes
6. `git commit` - pre-commit hooks run automatically
7. If hooks fail, fix issues and try again
8. `git push` - CI runs and must pass

## Type Checking with Mypy

### Current Status ✅

**All mypy type errors are resolved!** (61/61 errors fixed)

```bash
$ uv run mypy src/townlet
Success: no issues found in 44 source files
```

### Maintaining Type Safety

- **All new code** must be fully typed (no `Any` or `# type: ignore` unless absolutely necessary)
- **Type errors block commits** - both locally and in CI
- See `docs/development/MYPY_REMAINING_ERRORS.md` for fix patterns

### Common Type Patterns

```python
# Union type narrowing
if isinstance(self.replay_buffer, SequentialReplayBuffer):
    batch = self.replay_buffer.sample_sequences(batch_size)

# Non-None assertions
assert self.env is not None, "Environment must be initialized"

# Explicit Tensor typing
loss: torch.Tensor = F.mse_loss(q_pred, q_target)
loss_value: float = loss.item()

# Type annotations for inferred collections
self.affordance_interactions: dict[str, int] = {}
self.config_path: Path | None = None
```

## No-Defaults Linter

Enforces the **no-defaults principle** - all behavioral parameters must be explicitly specified in config files.

**Exemptions** (whitelisted in `.defaults-whitelist.txt`):

- Truly optional features (e.g., `cues.yaml` for visualization)
- Metadata fields (descriptions, display names)
- Computed values (observation dimensions)

**Adding exemptions**:

```bash
# Add to .defaults-whitelist.txt
echo "src/townlet/path/file.py:function_name:param_name" >> .defaults-whitelist.txt
```

## Markdownlint

All markdown files must follow style guide (`.markdownlint.yaml`).

**Common fixes**:

```bash
# Auto-fix most issues
markdownlint-cli2 "**/*.md" "!node_modules" "!.venv" --fix

# Check specific file
markdownlint-cli2 docs/README.md
```

## CI/CD Configuration

### Workflow File

**File**: `.github/workflows/lint.yml`

**Key features**:

- Runs on all pushes to `main` and all PRs
- Each check explicitly fails with `exit 1` on violations
- Clear error messages with fix instructions
- Summary output showing all passing checks

### Making Lint Required for Merging

**In GitHub repository settings** (requires admin access):

1. Go to **Settings** → **Branches**
2. Add branch protection rule for `main`:
   - Check "Require status checks to pass before merging"
   - Select "Lint" as required status check
   - Check "Require branches to be up to date before merging"
3. Save changes

**Result**: PRs cannot be merged until the `Lint` workflow passes.

## Troubleshooting

### Pre-commit hooks not running?

```bash
# Reinstall hooks
uv run pre-commit install

# Check hook is installed
ls -la .git/hooks/pre-commit

# Run manually to test
uv run pre-commit run --all-files
```

### Mypy errors on clean checkout?

```bash
# Ensure dependencies are installed
uv sync --all-extras

# Clear mypy cache and re-run
rm -rf .mypy_cache
uv run mypy src/townlet
```

### Ruff/Black conflicts?

Ruff formatter and Black should be compatible. If conflicts occur:

```bash
# Format with both in order
uv run ruff check . --fix
uv run black src tests
```

### CI passing but local pre-commit failing?

```bash
# Update pre-commit hooks to latest versions
uv run pre-commit autoupdate

# Clear cache and re-run
uv run pre-commit clean
uv run pre-commit run --all-files
```

## Summary

**Zero-tolerance policy**: All lint violations must be fixed before merging.

**Enforcement**:

- ✅ Pre-commit hooks (local)
- ✅ GitHub Actions CI (remote)
- ✅ Required status checks (repository settings)

**Quick reference**:

```bash
# Auto-fix everything possible
uv run ruff check . --fix && uv run black src tests

# Check everything (same as CI)
uv run pre-commit run --all-files

# Install hooks (one-time)
uv run pre-commit install
```

**All type errors resolved**: See `MYPY_REMAINING_ERRORS.md` for completion status.
