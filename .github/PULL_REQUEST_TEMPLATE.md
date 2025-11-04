## Summary

Brief description of changes (1-2 sentences).

## Motivation

Why are these changes needed? Link to issue if applicable (e.g., "Closes #123").

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Test coverage improvement

## Changes

- Bullet list of specific changes
- Keep focused and atomic
- One logical change per PR when possible

## Testing

### Tests Added/Updated

- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Added/updated docstrings
- [ ] No tests needed (explain why)

### Test Results

```bash
# Paste test output showing all tests pass
uv run pytest
```

- [ ] All tests pass locally
- [ ] No new warnings introduced
- [ ] Test coverage maintained or improved

### Manual Testing (if applicable)

Describe any manual testing performed:

- Configuration used
- Training runs completed
- Frontend testing done
- Edge cases verified

## Documentation

- [ ] Updated relevant documentation in `docs/`
- [ ] Updated `CHANGELOG.md` under `[Unreleased]`
- [ ] Updated `README.md` (if user-facing changes)
- [ ] Added code comments for complex logic
- [ ] Docstrings added to new public APIs

## Code Quality

- [ ] Code follows style guidelines (Black + Ruff)
- [ ] Ran `uv run black src/ tests/`
- [ ] Ran `uv run ruff check src/`
- [ ] Ran `uv run mypy src/townlet` (or confirmed type issues are acceptable)
- [ ] No `print()` statements (use `logging` instead)
- [ ] No TODO comments (create issues instead)

## Breaking Changes

Does this PR introduce any breaking changes?

- [ ] No breaking changes
- [ ] Yes, breaking changes (describe below)

**If yes, describe:**

- What breaks and why
- Migration path for users
- Version bump required (major/minor)

## Additional Context

- Screenshots (if UI changes)
- Performance benchmarks (if performance-related)
- Links to research or design docs
- Pedagogical implications (if relevant to teaching RL)

## Checklist

- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
- [ ] My commits are clean and well-described
- [ ] I have self-reviewed my code
- [ ] I have commented my code where needed
- [ ] My changes generate no new warnings
- [ ] I have updated documentation accordingly
- [ ] My changes maintain or improve test coverage
- [ ] All CI checks pass

## Related Issues/PRs

- Closes #
- Related to #
- Depends on #

---

**For Reviewers:**

- Focus areas for review: [Specify what you'd like reviewers to pay attention to]
- Known limitations: [Any known issues or technical debt introduced]
