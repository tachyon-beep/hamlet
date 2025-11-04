# BUG-XXX: [Concise Bug Description]

**Status**: [Reported | Investigating | In Progress | Fixed | Closed | Won't Fix]
**Priority**: [P0-Critical | P1-High | P2-Medium | P3-Low]
**Severity**: [Blocker | Major | Minor | Trivial]
**Estimated Effort**: [X hours] (investigation + fix + testing)
**Reported**: YYYY-MM-DD
**Fixed**: YYYY-MM-DD (when resolved)
**Reporter**: [Name/System]

**Keywords**: [Add 3-5 searchable keywords for AI/human discovery]
**Subsystems**: [Which components are affected]
**Regression**: [Yes/No - did this work before?]
**Reproducibility**: [Always | Sometimes | Rare]

---

## AI-Friendly Summary (Skim This First!)

**What Breaks**: [One sentence: Observable failure]
**Expected Behavior**: [One sentence: What should happen]
**Actual Behavior**: [One sentence: What actually happens]
**Impact**: [One sentence: Who/what is affected]

**Quick Triage**:

- **Severity**: [How bad is this?]
- **Urgency**: [How soon must this be fixed?]
- **Workaround**: [Yes/No - if yes, noted below]

**Decision Point**: If this bug doesn't affect your current work, STOP READING HERE.

---

## Bug Report

### Environment

**System**:

- OS: [Linux/macOS/Windows version]
- Python: [Version]
- HAMLET Version: [Commit hash or tag]
- Config: [Which config pack: L0_0_minimal, L1_full_observability, etc.]

**Hardware** (if relevant):

- GPU: [Model or "CPU only"]
- RAM: [Amount]

### Steps to Reproduce

**Minimal Reproducible Example**:

```bash
# Commands to reproduce the bug
cd /path/to/hamlet
export PYTHONPATH=...
python -m townlet.demo.runner --config configs/L0_0_minimal
```

**Reproduction Rate**: [Always | 8/10 runs | Rare]

**Prerequisites**:

- [Required state/setup 1]
- [Required state/setup 2]

### Expected Behavior

**What Should Happen**:
[Clear description of correct behavior]

**Example** (if helpful):

```python
# Expected output or behavior
```

### Actual Behavior

**What Actually Happens**:
[Clear description of failure]

**Error Message** (if applicable):

```
[Full error traceback or log output]
```

**Symptoms**:

- [Observable symptom 1]
- [Observable symptom 2]

### Screenshots/Logs (if applicable)

**Visual Evidence**:
[Attach screenshots, graphs, or logs]

---

## Impact Assessment

### Who Is Affected

**User Impact**:

- ✅ **None**: [Only affects specific edge case]
- ⚠️ **Some**: [Affects X% of users/runs]
- ❌ **All**: [Blocks all users/runs]

**Affected Use Cases**:

- [Use case 1 that breaks]
- [Use case 2 that breaks]

### Blast Radius

**Direct Impact**:

- [Subsystem 1]: [How it's affected]
- [Subsystem 2]: [How it's affected]

**Indirect Impact**:

- [Downstream system 1]: [Potential cascade effect]

**Data Corruption Risk**: [Yes/No]

- [If yes, describe what data could be lost/corrupted]

---

## Root Cause Analysis

### Investigation Status

- [ ] Bug reproduced locally
- [ ] Failure mode understood
- [ ] Root cause identified
- [ ] Fix approach determined

### Hypothesis

**Initial Theory**:
[What we think is causing this]

**Evidence**:

- [Observation 1 supporting theory]
- [Observation 2 supporting theory]

**Alternative Theories**:

- [Theory 2]: [Why considered/rejected]
- [Theory 3]: [Why considered/rejected]

### Root Cause (Once Identified)

**File**: `path/to/file.py:line_number`

**Code**:

```python
# Problematic code
def buggy_function():
    # The issue is here...
```

**Why This Fails**:
[Detailed explanation of the failure mechanism]

**Introduced In**:

- Commit: [Hash]
- Date: [YYYY-MM-DD]
- Related PR/Task: [If applicable]

---

## Fix Strategy

### Proposed Solution

**Approach**:
[Describe how to fix this]

**Changes Required**:

1. **File**: `path/to/file.py:line`
   - Change: [What to modify]
   - Reason: [Why this fixes it]

2. **File**: `path/to/other.py:line`
   - Change: [What to modify]
   - Reason: [Why this is needed]

**Code Example**:

```python
# Fixed code
def corrected_function():
    # Fixed implementation
```

### Alternative Approaches Considered

**Option 1: [Description]**

- ✅ Pros: [Benefit 1, Benefit 2]
- ❌ Cons: [Drawback 1, Drawback 2]
- **Rejected because**: [Reason]

**Option 2: [Description]**

- ✅ Pros: [Benefit 1, Benefit 2]
- ❌ Cons: [Drawback 1, Drawback 2]
- **Selected because**: [Reason]

---

## Testing Plan

### Regression Test

**New Test** (to prevent recurrence):

```python
# tests/test_path/test_bugfix.py
def test_bug_xxx_does_not_recur():
    """Regression test for BUG-XXX: [Description]."""
    # Test that verifies bug is fixed
    assert expected_behavior()
```

**Test Categories**:

- [ ] Unit test for specific function
- [ ] Integration test for end-to-end flow
- [ ] Property test for edge cases

### Validation Steps

**Before Fix**:

- [ ] Reproduce bug consistently
- [ ] Confirm failure mode
- [ ] Document current behavior

**After Fix**:

- [ ] Bug no longer reproduces
- [ ] All existing tests pass
- [ ] New regression test passes
- [ ] No new bugs introduced

**Manual Testing** (if applicable):

- [ ] [Manual test 1]
- [ ] [Manual test 2]

---

## Workaround (Temporary)

**Available**: [Yes/No]

**If Yes**:

```bash
# Temporary workaround steps
```

**Limitations**:

- [Limitation 1 of workaround]
- [Limitation 2 of workaround]

**Recommended**: [Yes/No - is workaround safe to use?]

---

## Fix Implementation

### Test-Driven Development Approach

**Phase 1: RED (Write Failing Test)**

- [ ] Write regression test that fails with current code
- [ ] Verify test fails for the right reason
- [ ] Commit test to prevent future regressions

**Phase 2: GREEN (Minimal Fix)**

- [ ] Implement minimal fix to make test pass
- [ ] Verify all tests pass
- [ ] No new failures introduced

**Phase 3: REFACTOR (Clean Up)**

- [ ] Improve code clarity if needed
- [ ] Tests remain green
- [ ] Update documentation

### Verification Checklist

- [ ] Bug no longer reproduces
- [ ] Regression test added and passing
- [ ] All existing tests pass
- [ ] No performance regression
- [ ] Edge cases handled
- [ ] Documentation updated (if needed)

---

## Related Issues

### Duplicate Reports

- BUG-XXX: [Related or duplicate bug]
- BUG-YYY: [Related or duplicate bug]

### Related Tasks

- TASK-XXX: [Task that might address root cause]
- QUICK-XXX: [Related cleanup work]

### Similar Bugs

- BUG-AAA: [Similar failure mode]
- BUG-BBB: [Similar subsystem]

---

## Prevention Strategy

### How to Prevent Recurrence

**Short Term**:

- [Immediate fix to prevent this specific bug]

**Long Term**:

- [Architectural/process changes to prevent similar bugs]

**Testing Improvements**:

- [Additional test coverage needed]
- [Property tests to catch edge cases]

**Code Quality**:

- [Static analysis rules to add]
- [Lint rules to enforce]

---

## Post-Fix Notes

### Lessons Learned

**What We Learned**:

- [Lesson 1]
- [Lesson 2]

**Process Improvements**:

- [How to catch this earlier next time]

### Follow-up Work

**Created Tasks**:

- TASK-XXX: [Architectural fix to prevent class of bugs]
- QUICK-XXX: [Cleanup work identified during fix]

**Technical Debt Incurred**:

- [Any shortcuts taken that need future cleanup]

---

## Communication

### Stakeholder Notification

**Who to Notify**:

- [Person/team 1]: [What they need to know]
- [Person/team 2]: [What they need to know]

**User Communication** (if customer-facing):

- [ ] Bug acknowledged
- [ ] Workaround provided
- [ ] Fix timeline communicated
- [ ] Fix deployed and verified

### Release Notes Entry

**For Next Release**:

```markdown
### Bug Fixes
- **BUG-XXX**: Fixed [concise description]. [Impact statement].
```

---

## References

### Code Files

- `src/path/to/buggy_file.py:line` - [Where bug occurs]
- `tests/path/to/test.py` - [Regression test]

### Related Documentation

- [Link to architecture doc]
- [Link to design decision]
- [Link to user guide]

### External References

- GitHub Issue: [Link if applicable]
- Stack Overflow: [Link if researched there]
- Research Paper: [Link if academic bug]

---

**END OF BUG REPORT**
