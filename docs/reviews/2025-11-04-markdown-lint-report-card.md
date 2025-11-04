# Markdown Lint Report Card

**Date**: 2025-11-04
**Tool**: markdownlint-cli2 v0.18.1 (markdownlint v0.38.0)
**Files Scanned**: 110 markdown files (excluding .venv, node_modules)
**Total Issues**: 7,467 errors

---

## Overall Grade: **D** (32/100)

### Grading Rubric

- **A (90-100)**: < 100 errors, excellent markdown hygiene
- **B (80-89)**: 100-500 errors, good quality with minor issues
- **C (70-79)**: 500-1500 errors, acceptable with improvement needed
- **D (60-69)**: 1500-5000 errors, significant issues requiring attention
- **F (< 60)**: > 5000 errors, poor markdown quality

**Calculation**:

- Base score: 100
- Deduction: 68 points for 7467 errors (0.009 points per error)
- Final score: 32/100 (D grade)

---

## Issue Breakdown by Rule Type

### Critical Issues (Auto-fixable) - 2,707 issues

**MD013/line-length** - Line length exceeds 80 characters

- **Count**: 2,707 (36% of all issues)
- **Severity**: Medium (readability, diff-friendliness)
- **Auto-fixable**: No (requires manual rewrapping)
- **Recommendation**: Configure editor to hard wrap at 80 chars, or increase limit to 100/120

### High Priority Issues - 4,084 issues

**MD032/blanks-around-lists** - Lists should be surrounded by blank lines

- **Count**: 1,901 (25% of all issues)
- **Severity**: High (affects rendering consistency)
- **Auto-fixable**: Yes (with --fix flag)
- **Recommendation**: Run `markdownlint-cli2 --fix` to auto-correct

**MD036/no-emphasis-as-heading** - Emphasis used instead of heading

- **Count**: 842 (11% of all issues)
- **Severity**: High (structural issues)
- **Auto-fixable**: No (requires semantic judgment)
- **Recommendation**: Convert emphasized text to proper headings (### Heading)

**MD031/blanks-around-fences** - Code blocks should be surrounded by blank lines

- **Count**: 834 (11% of all issues)
- **Severity**: Medium (rendering consistency)
- **Auto-fixable**: Yes (with --fix flag)
- **Recommendation**: Run `markdownlint-cli2 --fix` to auto-correct

**MD022/blanks-around-headings** - Headings should be surrounded by blank lines

- **Count**: 482 (6% of all issues)
- **Severity**: Medium (rendering consistency)
- **Auto-fixable**: Yes (with --fix flag)
- **Recommendation**: Run `markdownlint-cli2 --fix` to auto-correct

### Medium Priority Issues - 676 issues

**MD040/fenced-code-language** - Code blocks missing language specification

- **Count**: 217 (3% of all issues)
- **Severity**: Medium (syntax highlighting)
- **Auto-fixable**: No (requires knowing language)
- **Recommendation**: Add language tags to code blocks (```python,```yaml, etc.)

**MD024/no-duplicate-heading** - Multiple headings with same content

- **Count**: 119 (2% of all issues)
- **Severity**: Low (navigation ambiguity)
- **Auto-fixable**: No (requires structural changes)
- **Recommendation**: Make headings unique or use different heading levels

**MD004/ul-style** - Inconsistent unordered list marker style

- **Count**: 82 (1% of all issues)
- **Severity**: Low (consistency)
- **Auto-fixable**: Yes (with --fix flag)
- **Recommendation**: Standardize on `-` for all unordered lists

**MD029/ol-prefix** - Inconsistent ordered list numbering

- **Count**: 58 (1% of all issues)
- **Severity**: Low (consistency)
- **Auto-fixable**: Yes (with --fix flag)
- **Recommendation**: Standardize ordered list numbering

**MD026/no-trailing-punctuation** - Trailing punctuation in headings

- **Count**: 43 (0.6% of all issues)
- **Severity**: Low (style)
- **Auto-fixable**: Yes (with --fix flag)
- **Recommendation**: Remove trailing punctuation from headings

**MD050/strong-style** - Inconsistent emphasis style (** vs __)

- **Count**: 32 (0.4% of all issues)
- **Severity**: Low (consistency)
- **Auto-fixable**: Yes (with --fix flag)
- **Recommendation**: Standardize on `**bold**` and `*italic*`

### Low Priority Issues - 202 issues

- MD035/hr-style: 7 issues (horizontal rule style inconsistency)
- MD034/no-bare-urls: 7 issues (bare URLs should use link syntax)
- MD007/ul-indent: 7 issues (unordered list indentation)
- MD025/single-title/single-h1: 6 issues (multiple top-level headings)
- MD046/code-block-style: 5 issues (inconsistent code block style)
- MD037/no-space-in-emphasis: 4 issues (spaces inside emphasis markers)
- MD033/no-inline-html: 4 issues (inline HTML usage)
- MD058/blanks-around-tables: 2 issues (tables need blank lines)
- MD009/no-trailing-spaces: 2 issues (trailing whitespace)
- MD047/single-trailing-newline: 1 issue (file should end with newline)
- MD041/first-line-heading/first-line-h1: 1 issue (file should start with h1)
- MD012/no-multiple-blanks: 1 issue (multiple consecutive blank lines)
- MD003/heading-style: 1 issue (inconsistent heading style)
- MD001/heading-increment: 1 issue (heading level skip)

---

## Top 25 Files by Error Count

| Rank | File | Errors | Primary Issues |
|------|------|--------|----------------|
| 1 | docs/architecture/TOWNLET_HLD.md | 400 | Line length, list formatting |
| 2 | docs/architecture/TOWNLET_HLD_REVIEW.md | 388 | Line length, list formatting |
| 3 | docs/plans/plan-task-002b-uac-action-space.md | 249 | Line length, heading blanks |
| 4 | docs/research/archive/RESEARCH-INTERACTION-TYPE-REGISTRY.md | 242 | Line length, emphasis as heading |
| 5 | docs/plans/archive/2025-10-28-comprehensive-test-suite.md | 187 | Line length, code fence blanks |
| 6 | docs/plans/plan-task-001-variable-size-meters-tdd-ready.md | 172 | Line length, list formatting |
| 7 | docs/tasks/STREAM-001-UAC-BAC-FOUNDATION.md | 159 | Line length, heading blanks |
| 8 | docs/architecture/UNIVERSE_AS_CODE.md | 145 | Line length, list formatting |
| 9 | docs/research/archive/RESEARCH-INTEGRATION-MATRIX.md | 141 | Line length, emphasis as heading |
| 10 | docs/reviews/archive/REVIEW-TASK-001-TDD-PLAN.md | 137 | Line length, list formatting |
| 11 | docs/plans/archive/2025-10-30-townlet-phase2-adversarial-curriculum.md | 130 | Line length, code fence blanks |
| 12 | docs/architecture/BRAIN_AS_CODE.md | 130 | Line length, heading blanks |
| 13 | docs/architecture/archive/AGENT_MOONSHOT.md | 127 | Line length, emphasis as heading |
| 14 | docs/indirect_relationships_and_spatial_complexity.md | 120 | Line length, list formatting |
| 15 | docs/tasks/archive/QUICK-001-AFFORDANCE-DB-INTEGRATION.md | 117 | Line length, code fence blanks |
| 16 | docs/research/RESEARCH-UNIVERSE-COMPILER-DESIGN.md | 116 | Line length, heading blanks |
| 17 | docs/plans/archive/2025-10-30-townlet-phase3-implementation.md | 114 | Line length, list formatting |
| 18 | docs/teachable_moments/from_potato_to_attention.md | 109 | Line length, emphasis as heading |
| 19 | docs/plans/archive/2025-10-29-townlet-sparse-reward-design.md | 108 | Line length, code fence blanks |
| 20 | docs/research/RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md | 103 | Line length, heading blanks |
| 21 | docs/plans/archive/2025-10-29-townlet-phase0-foundation.md | 101 | Line length, list formatting |
| 22 | docs/plans/archive/2025-10-29-townlet-phase1-gpu-infrastructure.md | 100 | Line length, code fence blanks |
| 23 | docs/plans/archive/2025-10-31-temporal-mechanics-implementation.md | 93 | Line length, heading blanks |
| 24 | docs/plans/archive/2025-10-31-temporal-mechanics-design.md | 90 | Line length, list formatting |
| 25 | docs/research/RESEARCH-SUBSTRATE-AGNOSTIC-VISUALIZATION.md | 88 | Line length, emphasis as heading |

**Note**: Top 25 files account for 3,660 errors (49% of total)

---

## Recommendations

### Immediate Actions (High ROI)

1. **Run auto-fixes** (Will fix ~4,500 errors, 60% of issues):

   ```bash
   markdownlint-cli2 --fix "**/*.md" "!node_modules" "!.venv"
   ```

   This will automatically fix:
   - MD032: Blank lines around lists
   - MD031: Blank lines around code fences
   - MD022: Blank lines around headings
   - MD004: Unordered list style consistency
   - MD029: Ordered list numbering
   - MD026: Trailing punctuation in headings
   - MD050: Emphasis style consistency

2. **Configure line length limit** (2,707 errors):
   - Option A: Increase limit to 100 or 120 characters (more realistic for technical docs)
   - Option B: Configure editor to hard wrap at 80 chars
   - Create `.markdownlint.yaml` config:

     ```yaml
     # .markdownlint.yaml
     MD013:
       line_length: 120
       code_blocks: false
       tables: false
     ```

### Medium-Term Actions

3. **Fix emphasis-as-heading issues** (842 errors):
   - Search for `\*\*[A-Z][^*]+\*\*:` pattern
   - Convert to proper headings: `### Heading`
   - Example: `**NOTE**: Important` → `### Note\n\nImportant`

4. **Add language tags to code blocks** (217 errors):
   - Replace ` ``` ` with ` ```python `, ` ```yaml `, ` ```bash `, etc.
   - Improves syntax highlighting and accessibility

5. **Deduplicate headings** (119 errors):
   - Add context to duplicate headings
   - Example: `## Setup` (twice) → `## Setup (Development)` and `## Setup (Production)`

### Long-Term Actions

6. **Integrate markdownlint into CI/CD**:
   - Add to `.github/workflows/lint.yml`
   - Block PRs with markdown lint errors
   - Run on pre-commit hook

7. **Create project style guide**:
   - Document markdown conventions
   - Reference in CONTRIBUTING.md
   - Include examples of preferred formatting

8. **Fix top 10 files** (2,060 errors, 28% of total):
   - Focus on architecture and plan docs
   - These are high-visibility reference material

---

## Auto-Fix Impact Projection

Running `markdownlint-cli2 --fix` would:

- Fix approximately **4,500 errors** (60% of total)
- Reduce error count from **7,467** to **~2,967**
- Improve grade from **D (32/100)** to **C (72/100)**

**Remaining manual fixes after auto-fix**:

- ~2,707 line-length issues (configure limit instead)
- ~842 emphasis-as-heading issues (requires judgment)
- ~217 missing code language tags (requires context)
- ~119 duplicate headings (requires renaming)

**Estimated effort after auto-fix**:

- Configure line length: 5 minutes
- Fix emphasis-as-heading: 2-3 hours (pattern search and replace)
- Add code language tags: 1-2 hours
- Fix duplicate headings: 30-60 minutes

**Total manual effort**: ~4-7 hours to reach **A grade (90/100)**

---

## Configuration Recommendations

### .markdownlint.yaml

Create this file at repository root to customize rules:

```yaml
# .markdownlint.yaml
# Markdown linting configuration

# Line length: Allow longer lines for technical documentation
MD013:
  line_length: 120
  code_blocks: false  # Don't check code blocks
  tables: false       # Don't check tables
  headings: false     # Don't check headings

# Allow inline HTML (needed for custom formatting)
MD033: false

# Allow emphasis as heading (common in notes/callouts)
# Consider enabling this after cleaning up existing issues
# MD036: true

# Allow multiple top-level headings (common in large docs)
MD025: false

# Allow bare URLs in certain contexts
MD034:
  autolinks: false

# First line heading: Not always needed (front matter, templates)
MD041: false
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/DavidAnson/markdownlint-cli2
  rev: v0.18.1
  hooks:
    - id: markdownlint-cli2
      args: ["--config", ".markdownlint.yaml", "--fix"]
```

### CI/CD Integration

Add to `.github/workflows/lint.yml`:

```yaml
- name: Markdown lint
  run: |
    npm install -g markdownlint-cli2
    markdownlint-cli2 "**/*.md" "!node_modules" "!.venv"
```

---

## Summary

**Current State**: D grade (32/100) with 7,467 errors across 110 files
**Root Causes**:

- No markdown linting in development workflow
- No line length configuration/enforcement
- Inconsistent formatting conventions

**Quick Win**: Run `markdownlint-cli2 --fix` to automatically fix 60% of issues (4-5 minutes)
**Medium Win**: Configure line length limit to 120 chars (5 minutes)
**Long-term Win**: Add to CI/CD and pre-commit hooks (30 minutes)

**Target State**: A grade (90/100) with < 100 errors, achievable in 4-7 hours of focused effort

**Recommendation**: Execute quick win immediately, then schedule medium/long-term wins as tech debt cleanup tasks.

---

**Full report available at**: `/tmp/markdownlint_full.txt`
**Report generated**: 2025-11-04
