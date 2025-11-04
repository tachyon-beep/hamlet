# PDR-001: Mandatory Lint Enforcement

**Status**: Accepted ✅
**Type**: Development Process
**Priority**: CRITICAL
**Date Proposed**: 2025-11-04
**Date Adopted**: 2025-11-04
**Decision Makers**: Development Team
**Affects**: All contributors to HAMLET codebase
**Enforcement**: Pre-commit hooks (local) + GitHub Actions (CI)

**Keywords**: lint, mypy, type-safety, pre-commit, CI/CD, code-quality, ruff, black, no-defaults
**Subsystems**: All (development process)
**Breaking Changes**: No (developers must install pre-commit hooks)
**Supersedes**: N/A

---

## AI-Friendly Summary (Skim This First!)

**What**: All lint violations (ruff, black, mypy, no-defaults, markdownlint) must be fixed before code can be merged. Enforcement at two layers: pre-commit hooks (local) and GitHub Actions CI (remote).

**Why**: Historical mypy errors accumulated from 0 → 60+ violations. Type checking discovered **RND loss tracking bug** where predictor loss was computed but discarded without monitoring. Zero-tolerance policy prevents regression and ensures type safety.

**Scope**: **Mandatory** for all developers. Pre-commit hooks must be installed (`uv run pre-commit install`). CI lint workflow must pass before PRs merge.

**Quick Assessment**:

- **Current Status**: ✅ All 61 mypy errors fixed (100% coverage across 44 files)
- **Enforcement**: Pre-commit hooks installed + CI checks updated with explicit `exit 1` on failures
- **Impact**: +3-5 seconds per commit (hooks), but catches bugs before CI
- **Bug Prevented**: RND loss tracking bug discovered via mypy variable shadowing warning

**Decision Point**: If you're not committing code or reviewing PRs, STOP READING HERE. If you're fixing lint errors, see `docs/development/LINT_ENFORCEMENT.md` for quick fix commands.

---

## Context

### Current Situation

Prior to this policy, lint warnings and type errors could accumulate in the codebase without blocking commits or merges.

**Pain Points**:

- Type safety regressions (mypy errors increasing from 0 → 60+ over time)
- Inconsistent code formatting (black, ruff violations)
- Configuration bugs (unauthorized defaults causing non-reproducible behavior)
- Markdown documentation inconsistencies
- Pre-commit hooks configured but not enforced (developers could skip installation)
- CI lint checks ran but didn't block PR merges

**Historical Evidence**:

- **2025-10-30**: Mypy errors reached 60+ violations across 7 files
- **2025-11-04**: RND loss bug discovered via mypy type checking - predictor loss was computed but discarded without tracking
- **Ongoing**: Style inconsistencies requiring manual review feedback

### Why Now?

**Catalyzing Event**:

The comprehensive mypy delinting effort (2025-11-04) fixed all 61 type errors and **discovered a real bug** in RND loss tracking. This demonstrated that:

1. Type checking provides **tangible bug discovery** (not just style enforcement)
2. Allowing violations to accumulate makes fixing them expensive (~2 hours for 61 errors)
3. Without enforcement, the same violations will accumulate again

This policy prevents regression and establishes type safety as a **maintained property** of the codebase.

---

## Policy Decision

### Core Requirement

**All lint violations must be fixed before code can be merged.**

### Detailed Requirements

**Layer 1: Pre-Commit Hooks (Local)**

Pre-commit hooks are **mandatory** for all developers and automatically run on `git commit`.

**Checks/Actions**:

1. Ruff (linting with auto-fix) - catches code quality issues
2. Black (code formatting) - enforces consistent style
3. Mypy (type checking on `src/townlet/`) - prevents type errors
4. Standard checks (trailing whitespace, YAML syntax, etc.) - basic hygiene
5. Markdownlint (markdown formatting) - documentation quality

**Behavior**: Hooks **block commits** if any check fails. Developer must fix violations before commit succeeds.

**Installation**: `uv run pre-commit install` (documented in `CONTRIBUTING.md`)

**Layer 2: GitHub Actions CI/CD (Remote)**

The `Lint` workflow runs on all pushes and pull requests, providing a final enforcement layer.

**Checks/Actions**:

1. Ruff (zero warnings enforced)
2. Black (zero format violations enforced)
3. Mypy (zero type errors enforced)
4. No-defaults linter (no unauthorized defaults)
5. Markdownlint (zero violations enforced)

**Behavior**: Each check explicitly fails with `exit 1` on violations. PRs **cannot merge** until all checks pass.

**Configuration**: `.github/workflows/lint.yml` + required status check in repository settings

### Exceptions

**When exceptions are allowed**:

- `# type: ignore[error-code]` - Only for untyped third-party libraries, with justification comment explaining why
- `# noqa: error-code` - Only for false positives in ruff/black, with justification comment
- `.defaults-whitelist.txt` entries - Only for truly optional parameters (metadata, visualization settings), with team approval

**How to request exception**:

1. Add justification comment explaining why exception is necessary
2. For no-defaults whitelist: Add entry to `.defaults-whitelist.txt` with comment explaining exemption
3. For systemic issues: Raise in team discussion/GitHub issue

**Zero-tolerance for**:

- Bypassing hooks with `git commit --no-verify` (strongly discouraged)
- Merging PRs with failing lint checks
- Adding violations "temporarily" with intent to fix later

## Rationale

### Type Safety Benefits

**From recent mypy delinting effort** (2025-11-04):

- Fixed 61 type errors across 7 files (100% coverage)
- **Discovered RND loss tracking bug** - predictor loss was computed but discarded
  - Bug found via variable shadowing warning (line 536 vs 604)
  - Fixed using Test-Driven Development (RED-GREEN-REFACTOR)
  - Now properly tracked in `population.last_rnd_loss` for TensorBoard monitoring
- Prevented potential runtime errors (None access, Union type mismatches)
- Improved code documentation through type annotations

**Prevented bugs**:

```python
# BEFORE (type error hidden, bug present)
loss = rnd.update_predictor()  # Line 536 - computed but discarded
# ... 70 lines later ...
loss = F.mse_loss(...)  # Line 604 - overwrites without using rnd_loss

# AFTER (mypy caught variable shadowing → discovered bug)
rnd_loss = rnd.update_predictor()  # Line 536 - explicitly tracked
self.last_rnd_loss = rnd_loss  # Line 539 - saved for monitoring
# ... 70 lines later ...
loss: torch.Tensor = F.mse_loss(...)  # Line 604 - clear distinction
```

### Code Quality Benefits

1. **Consistency**: All code follows same style (Black, Ruff)
2. **Maintainability**: Types make refactoring safer
3. **Onboarding**: New contributors see clean, well-formatted code
4. **Documentation**: Type hints serve as inline documentation

### UNIVERSE_AS_CODE Alignment

The no-defaults linter enforces the **no-defaults principle**:

- All behavioral parameters must be explicit in config files
- No hidden defaults in code
- Operators know exactly what values are being used
- Old configs remain reproducible when code defaults change

## Consequences

### Positive

1. **Type safety guaranteed** - mypy errors can't accumulate
2. **Bug discovery** - type checking finds logic bugs (like RND loss tracking)
3. **Consistent style** - all code formatted the same way
4. **Faster reviews** - no need to comment on style/formatting
5. **Better docs** - type hints provide inline documentation
6. **Config reproducibility** - no-defaults ensures behavioral transparency

### Negative

1. **Initial friction** - developers must install pre-commit hooks
2. **Slower commits** - hooks add ~3-5 seconds per commit
3. **Learning curve** - mypy type errors require understanding Union types, assertions, etc.
4. **Occasional false positives** - may need `# type: ignore` or `# noqa` for edge cases

### Mitigation Strategies

**For initial friction**:

- Clear setup instructions in `CONTRIBUTING.md`
- Auto-fix commands provided (`ruff check --fix`, `black src tests`)
- Comprehensive error messages in CI with fix instructions

**For learning curve**:

- `docs/development/MYPY_REMAINING_ERRORS.md` documents fix patterns
- `docs/development/LINT_ENFORCEMENT.md` provides troubleshooting guide
- Pre-commit hooks auto-fix most issues (ruff, black, markdownlint)

**For false positives**:

- `# type: ignore[error-code]` allowed with justification comment
- `.defaults-whitelist.txt` for legitimate no-defaults exemptions
- Team discussion for ambiguous cases

## Implementation

### Phase 1: Pre-Commit Hooks (Completed)

- ✅ Created `.pre-commit-config.yaml` with all checks
- ✅ Installed hooks locally: `uv run pre-commit install`
- ✅ Verified hooks run on commit and block violations

### Phase 2: CI/CD Enforcement (Completed)

- ✅ Updated `.github/workflows/lint.yml`:
  - All checks explicitly fail with `exit 1` on violations
  - Clear error messages with fix instructions
  - Summary output showing all passing checks
- ✅ Mypy now runs with `--show-error-codes` for better debugging

### Phase 3: Documentation (Completed)

- ✅ Created `docs/development/LINT_ENFORCEMENT.md`
- ✅ Created `docs/decisions/PDR-001-LINT-ENFORCEMENT.md` (this document)
- ✅ Updated `docs/development/MYPY_REMAINING_ERRORS.md` with completion status

### Phase 4: Repository Settings (Manual - Requires Admin)

**To make Lint workflow a required status check**:

1. Go to repository **Settings** → **Branches**
2. Add/edit branch protection rule for `main`:
   - ☐ Check "Require status checks to pass before merging"
   - ☐ Select "Lint" as required status check
   - ☐ Check "Require branches to be up to date before merging"
   - ☐ Optional: Check "Require linear history" (prevents merge commits)
3. Save changes

**Result**: PRs cannot merge until `Lint` workflow passes.

---

## Compliance

### How to Comply

**Quick Start** (one-time setup):

```bash
# Install pre-commit hooks
uv pip install pre-commit
uv run pre-commit install

# Verify installation
ls -la .git/hooks/pre-commit
```

**Daily Workflow**:

1. Write code as normal
2. Run auto-fix commands before committing:

   ```bash
   uv run ruff check . --fix    # Fix most lint issues
   uv run black src tests       # Format code
   ```

3. Stage changes: `git add .`
4. Commit: `git commit -m "message"` (hooks run automatically)
5. If hooks fail, fix violations and try again
6. Push: `git push` (CI runs, must pass)

**Quick Fix Commands**:

```bash
# Auto-fix everything possible
uv run ruff check . --fix && uv run black src tests

# Check types (no auto-fix, must fix manually)
uv run mypy src/townlet --show-error-codes

# Run all pre-commit hooks manually (without committing)
uv run pre-commit run --all-files
```

### Verification

**How to verify compliance**:

```bash
# Check if pre-commit hooks are installed
uv run pre-commit run --all-files

# Check mypy
uv run mypy src/townlet

# Check ruff
uv run ruff check .

# Check black
uv run black --check src tests
```

**Expected output** (all passing):

```
✅ All pre-commit hooks passed
Success: no issues found in 44 source files (mypy)
All checks passed! (ruff)
All done! ✨ (black)
```

### Troubleshooting

**Problem 1: Pre-commit hooks not running**

**Symptoms**: Commits succeed without running checks

**Solution**:

```bash
# Reinstall hooks
uv run pre-commit install

# Verify installation
ls -la .git/hooks/pre-commit

# Test manually
uv run pre-commit run --all-files
```

**Problem 2: Mypy errors on clean checkout**

**Symptoms**: `ModuleNotFoundError` or similar type errors

**Solution**:

```bash
# Install all dependencies
uv sync --all-extras

# Clear mypy cache
rm -rf .mypy_cache

# Re-run
uv run mypy src/townlet
```

**Problem 3: Ruff/Black conflicts**

**Symptoms**: Ruff and Black disagree on formatting

**Solution**:

```bash
# Run in order (ruff first, then black)
uv run ruff check . --fix
uv run black src tests
```

**Problem 4: CI passing but local pre-commit failing**

**Symptoms**: Pre-commit hooks fail locally but CI passes

**Solution**:

```bash
# Update pre-commit hooks to match CI versions
uv run pre-commit autoupdate

# Clear cache and re-run
uv run pre-commit clean
uv run pre-commit run --all-files
```

**Problem 5: Need to bypass hooks for emergency fix**

**When allowed**: Only for critical production hotfixes

**Solution**:

```bash
# Bypass hooks (use sparingly!)
git commit --no-verify -m "hotfix: critical bug"

# MUST fix violations in follow-up commit
git commit -m "fix: address lint violations from hotfix"
```

**Note**: Using `--no-verify` for non-emergency commits is a policy violation.

---

## Alternatives Considered

### Alternative 1: Warnings Only (No Blocking)

**Description**: Run lint checks but only warn, don't block commits or merges.

**Pros**:

- ✅ No friction for developers
- ✅ No commit delays
- ✅ No learning curve for fixing violations

**Cons**:

- ❌ Historical evidence: violations accumulate (0 → 60+ mypy errors)
- ❌ RND loss bug went undetected despite warning
- ❌ No incentive to fix issues
- ❌ Technical debt grows over time

**Why Rejected**: Evidence shows warnings are ignored until they become a crisis. The RND loss bug proves type errors can hide real bugs.

### Alternative 2: CI Only (No Pre-Commit Hooks)

**Description**: Enforce lint checks in CI but not locally.

**Pros**:

- ✅ No local setup required
- ✅ Consistent environment (CI)
- ✅ Catches all violations eventually

**Cons**:

- ❌ Slower feedback loop (wait for CI)
- ❌ Wastes CI minutes on preventable failures
- ❌ Developers push broken code → CI fails → fix → push again
- ❌ Poor developer experience

**Why Rejected**: Pre-commit hooks provide immediate feedback (3-5 seconds) vs waiting for CI (2-5 minutes). Faster feedback improves productivity.

### Alternative 3: Manual Code Review for Style

**Description**: Rely on code reviewers to catch and comment on lint violations.

**Pros**:

- ✅ No tooling overhead
- ✅ Human judgment for edge cases
- ✅ Flexible to project needs

**Cons**:

- ❌ Doesn't scale (wastes reviewer time)
- ❌ Inconsistent enforcement across reviewers
- ❌ Bikeshedding in PR discussions
- ❌ Reviewers should focus on logic, not formatting
- ❌ Delays PR merge for trivial issues

**Why Rejected**: Automated tools are better at style enforcement than humans. Reviewers should focus on architecture and correctness, not whitespace.

### Alternative 4: Pre-Commit Only (No CI)

**Description**: Rely solely on pre-commit hooks, no CI enforcement.

**Pros**:

- ✅ Fast feedback (local)
- ✅ Catches issues before push
- ✅ No CI overhead

**Cons**:

- ❌ Can be bypassed with `git commit --no-verify`
- ❌ New contributors might not install hooks
- ❌ No enforcement on direct pushes to main (if allowed)
- ❌ Relies on developer discipline

**Why Rejected**: Defense-in-depth requires both local AND remote enforcement. CI provides safety net for skipped hooks or uninstalled pre-commit.

## Success Metrics

### Quantitative

**Metric 1: Mypy Error Count**

- **Baseline**: 61 errors (before policy)
- **Target**: 0 errors (maintained)
- **Measurement**: `uv run mypy src/townlet | grep "error"`
- **Current**: ✅ 0 errors in 44 files

**Metric 2: Ruff Warnings**

- **Baseline**: Unknown (not tracked before)
- **Target**: 0 warnings
- **Measurement**: `uv run ruff check . | grep "warning"`
- **Current**: ✅ 0 warnings

**Metric 3: Black Violations**

- **Baseline**: Unknown (not tracked before)
- **Target**: 0 violations
- **Measurement**: `uv run black --check src tests; echo $?` (exit code 0 = pass)
- **Current**: ✅ 0 violations

**Metric 4: CI Pass Rate**

- **Baseline**: N/A (lint checks didn't block before)
- **Target**: >95% of pushes pass lint checks (after initial adoption period)
- **Measurement**: GitHub Actions success rate for `Lint` workflow
- **Current**: Establishing baseline

### Qualitative

**Goal 1**: Developers report faster onboarding (no style questions in onboarding docs)

**Goal 2**: PRs focus on logic/architecture, not formatting (code review comments measure)

**Goal 3**: Fewer runtime type errors in production (incident reports)

**Goal 4**: Config files are self-documenting (no hidden defaults causing confusion)

## Review Schedule

**Frequency**: Quarterly (every 3 months)

**Next Review**: 2026-02-04

**Review Criteria**:

- Evaluate false positive rate (how often `# type: ignore` or `# noqa` needed)
- Review pre-commit hook performance (commit time overhead increasing?)
- Check CI pass rate metric (>95% target being met?)
- Assess qualitative goals (onboarding feedback, PR review focus)
- Update linter configurations as ecosystem evolves
- Consider adding new checks (e.g., import sorting, docstring coverage, complexity metrics)

**Owner**: Development Team

---

## References

### Documentation

- **Implementation Guide**: `docs/development/LINT_ENFORCEMENT.md` (detailed how-to, troubleshooting)
- **Related Policies**: None (first policy decision)
- **Architecture Docs**: `docs/architecture/UNIVERSE_AS_CODE.md` (no-defaults principle)
- **Fix Patterns**: `docs/development/MYPY_REMAINING_ERRORS.md` (type safety patterns)

### Tools & Automation

- **Config Files**:
  - `.pre-commit-config.yaml` (pre-commit hook configuration)
  - `.github/workflows/lint.yml` (CI/CD lint workflow)
  - `.defaults-whitelist.txt` (no-defaults exemptions)
  - `pyproject.toml` (ruff, black, mypy configuration)
- **CI/CD Workflows**: `.github/workflows/lint.yml`
- **Scripts**: `scripts/no_defaults_lint.py` (custom no-defaults linter)

### External References

- **Pre-commit Framework**: https://pre-commit.com
- **Ruff Linter**: https://docs.astral.sh/ruff/
- **Black Formatter**: https://black.readthedocs.io
- **Mypy Type Checker**: https://mypy.readthedocs.io
- **Industry Best Practices**: Google Engineering Practices (code review guidelines)

---

## Appendix

### Case Study: Type Safety Bug Discovery

**RND Loss Tracking Bug** (Discovered 2025-11-04)

**Context**: During comprehensive mypy delinting effort (fixing 61 errors), type checker discovered a real bug via variable shadowing warning.

**Symptom**: Mypy reported variable shadowing error:

```
src/townlet/population/vectorized.py:604: error: Name "loss" already defined on line 536 [no-redef]
```

**Root Cause**: RND predictor loss was computed but immediately discarded without tracking:

```python
# Line 536 (BAD - computed but unused)
loss = rnd.update_predictor()  # Returns predictor loss

# ... 70 lines of code later ...

# Line 604 (Overwrites without using rnd_loss)
loss = F.mse_loss(q_pred, q_target)  # Shadows previous loss variable
```

**Impact**:

- RND predictor loss was not tracked or logged to TensorBoard
- Impossible to monitor intrinsic exploration effectiveness during training
- No visibility into whether RND predictor was learning correctly
- Potential training issues could go undetected

**Fix** (via Test-Driven Development):

1. **RED**: Wrote failing test `test_rnd_loss_tracked_during_training()`
   - Asserted `population.last_rnd_loss` exists and is finite
   - Test failed as expected (attribute didn't exist)
2. **GREEN**: Implemented minimal fix
   - Added `self.last_rnd_loss = 0.0` initialization in `VectorizedPopulation.__init__`
   - Added `self.last_rnd_loss = rnd_loss` tracking after predictor update (line 539)
   - Renamed variable to avoid shadowing: `loss` → `rnd_loss` (line 536)
3. **REFACTOR**: Not needed (code already clean)

**Result**:

- RND loss now properly tracked and available for TensorBoard logging
- Monitoring of exploration behavior enabled
- Tests verify contract (loss must be tracked and finite)

**Lesson**:
Type checking catches not just type errors, but also **logic bugs** via:

- Variable shadowing detection (name reused incorrectly)
- Unused variable detection (computed but not used)
- Union type narrowing (forces explicit handling of None cases)
- Other static analysis warnings

**Broader Implication**:
This case study demonstrates the **tangible value** of enforced type checking beyond style:

- **Bug discovered**: Real logic bug, not just style violation
- **Prevention**: Would have been caught immediately with pre-commit hooks
- **Cost**: 2 hours to fix 61 accumulated errors vs instant feedback on each commit
- **ROI**: One bug prevented justifies entire policy

### FAQ

**Q: Why can't I just use `--no-verify` for "quick fixes"?**

A: `--no-verify` bypasses all pre-commit hooks, including security checks. It should only be used for emergency hotfixes that need immediate deployment. Using it for convenience undermines the entire policy and creates tech debt.

**Q: What if I disagree with a lint rule?**

A: Raise it in team discussion or GitHub issue. If there's consensus that a rule is problematic, we can update the configuration (e.g., disable specific ruff rules in `pyproject.toml`). Don't bypass enforcement individually.

**Q: How do I handle third-party library type stubs?**

A: Use `# type: ignore[import-untyped]` with a comment explaining which library lacks stubs. Example:

```python
from untyped_library import foo  # type: ignore[import-untyped] - no stubs available
```

**Q: What if pre-commit hooks slow down my workflow?**

A: Hooks add ~3-5 seconds per commit. If it's significantly slower:

1. Check if you're running on large file sets (hooks should only check staged files)
2. Consider running `uv run ruff check . --fix && uv run black src tests` manually before committing
3. Raise performance issues in team discussion - we can optimize hook configuration

**Q: Can I fix violations in a follow-up PR?**

A: No. The policy requires violations to be fixed before merge. If you encounter violations in existing code while working on a feature, fix them in the same PR or in a separate PR first.

---

**Status**: Accepted ✅
**Effective Date**: 2025-11-04
**Supersedes**: N/A (First policy decision)
