# TASK-XXX: [Descriptive Title - Major System Change]

**Status**: [Planned | In Progress | Blocked | Completed]
**Priority**: [Low | Medium | High | CRITICAL]
**Estimated Effort**: [X-Y hours or days] (multi-phase tasks, potential refactors)
**Dependencies**: [None | TASK-XXX, QUICK-XXX]
**Enables**: [TASK-XXX, TASK-YYY] (what this unblocks)
**Created**: YYYY-MM-DD
**Completed**: YYYY-MM-DD (when done)

**Keywords**: [Add 5-8 searchable keywords for AI/human discovery]
**Subsystems**: [List all affected subsystems]
**Architecture Impact**: [None | Minor | Major | Breaking]
**Breaking Changes**: [Yes/No - if yes, list what breaks]

---

## AI-Friendly Summary (Skim This First!)

**What**: [One sentence: Core system change being implemented]
**Why**: [One sentence: Strategic value - what this unblocks]
**Scope**: [One sentence: What's included and explicitly excluded]

**Quick Assessment**:
- **Current Limitation**: [What can't be done today]
- **After Implementation**: [What becomes possible]
- **Unblocks**: [What downstream work this enables]
- **Impact Radius**: [How many subsystems affected]

**Decision Point**: If this task is not relevant to your current work, STOP READING HERE.

---

## Problem Statement

### Current Constraint

[Detailed description of what's hardcoded, inflexible, or broken]

**Example** (code/config showing the limitation):
```python
# Current problematic code
```

### Why This Is Technical Debt, Not Design

**Test**: "If removed, would the system be more expressive or more fragile?"

**Answer**: [More expressive / More fragile]
- ✅ Enables: [Capability 1]
- ✅ Enables: [Capability 2]
- ❌ Does NOT: [What it doesn't break]

**Conclusion**: [Technical debt / Intentional constraint]

### Impact of Current Constraint

**Cannot Create**:
- [Use case 1 that's blocked]
- [Use case 2 that's blocked]
- [Use case 3 that's blocked]

**Pedagogical Cost** (if applicable):
- [How this limits teaching value]

**Research Cost** (if applicable):
- [How this limits experimentation]

**From Analysis**: [Summary of why this is high-leverage]

---

## Solution Overview

### Design Principle

**Core Philosophy**: [One sentence capturing the design intent]

**Key Insight**: [The fundamental realization that makes this possible]

### Architecture Changes

**1. [Layer 1 Name]**: [What changes here]

**2. [Layer 2 Name]**: [What changes here]

**3. [Layer 3 Name]**: [What changes here]

**4. [Layer 4 Name]**: [What changes here]

### Compatibility Strategy

**Backward Compatibility**:
- [How existing configs/checkpoints are handled]

**Migration Path**:
- [How users upgrade from old system]

**Versioning**:
- [How we detect version mismatches]

---

## Detailed Design

### Phase 1: [Foundation Name] (X hours)

**Objective**: [What this phase establishes]

**Changes**:
- File: `path/to/file.py`
  - [Change 1]
  - [Change 2]
- File: `path/to/other.py`
  - [Change 1]

**Tests**:
- [ ] Unit tests for new functionality
- [ ] Existing tests still pass

**Success Criteria**: [How to know this phase is complete]

### Phase 2: [Integration Name] (Y hours)

**Objective**: [What this phase connects]

**Changes**:
[Similar structure to Phase 1]

**Migration**:
- [ ] Old configs validated
- [ ] New configs validated
- [ ] Clear error messages

**Success Criteria**: [How to know this phase is complete]

### Phase 3: [Validation Name] (Z hours)

**Objective**: [What this phase proves]

**Testing**:
- [ ] Full regression suite
- [ ] Integration tests with real configs
- [ ] Performance benchmarks (if applicable)

**Documentation**:
- [ ] Update CLAUDE.md
- [ ] Update config examples
- [ ] Update architecture docs

**Success Criteria**: [How to know task is complete]

---

## Testing Strategy

### Test Coverage Requirements

**Unit Tests**:
- [Subsystem 1]: X% coverage
- [Subsystem 2]: Y% coverage

**Integration Tests**:
- [ ] Test 1: [Description]
- [ ] Test 2: [Description]

**Property-Based Tests** (if applicable):
- [ ] Property 1: [Description]
- [ ] Property 2: [Description]

### Regression Testing

**Critical Paths**:
- [ ] Existing training runs still work
- [ ] Existing checkpoints load correctly
- [ ] Existing configs validate

**Performance Testing**:
- [ ] No significant performance regression
- [ ] Benchmark: [Specific metric] remains within [X%]

---

## Migration Guide

### For Existing Configs

**Before** (old format):
```yaml
# Old config structure
```

**After** (new format):
```yaml
# New config structure
```

**Migration Script** (if needed):
```bash
# Command to migrate configs
```

### For Existing Checkpoints

**Compatibility**: [Full / Partial / None]

**Migration Process**:
1. [Step 1]
2. [Step 2]

---

## Examples

### Example 1: [Simple Use Case]

**Config**:
```yaml
# Example configuration
```

**Usage**:
```python
# Example code showing usage
```

**Output**: [What this produces]

### Example 2: [Complex Use Case]

**Config**:
```yaml
# More complex configuration
```

**Usage**:
```python
# Example code showing advanced usage
```

**Output**: [What this produces]

---

## Acceptance Criteria

### Must Have (Blocking)

- [ ] [Criterion 1 - specific and testable]
- [ ] [Criterion 2 - specific and testable]
- [ ] All tests pass (unit + integration)
- [ ] No regressions in existing functionality
- [ ] Documentation updated

### Should Have (Important)

- [ ] [Criterion 1 - valuable but not blocking]
- [ ] [Criterion 2 - valuable but not blocking]

### Could Have (Future)

- [ ] [Enhancement 1 - deferred to follow-up task]
- [ ] [Enhancement 2 - deferred to follow-up task]

---

## Risk Assessment

### Technical Risks

**Risk 1: [Description]**
- **Severity**: [High/Medium/Low]
- **Mitigation**: [How to address]
- **Contingency**: [Fallback plan]

**Risk 2: [Description]**
- **Severity**: [High/Medium/Low]
- **Mitigation**: [How to address]
- **Contingency**: [Fallback plan]

### Blocking Dependencies

- ✅ **NONE**: [All prerequisites exist]
- ⚠️ **PARTIAL**: [What needs to be done first]
- ❌ **BLOCKED**: [What's blocking this entirely]

### Impact Radius

**Files Modified**: [Estimated number]
**Tests Added**: [Estimated number]
**Breaking Changes**: [List if any]

**Blast Radius**: [Small/Medium/Large]
- [Description of potential impact]

---

## Effort Breakdown

### Detailed Estimates

**Phase 1**: X hours
- [Subtask 1]: X hours
- [Subtask 2]: Y hours

**Phase 2**: Y hours
- [Subtask 1]: X hours
- [Subtask 2]: Y hours

**Phase 3**: Z hours
- [Subtask 1]: X hours
- [Subtask 2]: Y hours

**Total**: [X-Y hours range]

**Confidence**: [High/Medium/Low]

### Assumptions

- [Assumption 1 that affects estimate]
- [Assumption 2 that affects estimate]

---

## Future Work (Explicitly Out of Scope)

### Not Included in This Task

1. **[Enhancement 1]**
   - **Why Deferred**: [Reason]
   - **Follow-up Task**: [TASK-XXX if created]

2. **[Enhancement 2]**
   - **Why Deferred**: [Reason]
   - **Follow-up Task**: [TASK-YYY if created]

### Enables Future Tasks

- **TASK-XXX**: [What becomes possible]
- **TASK-YYY**: [What becomes possible]

---

## References

### Related Documentation

- **Design Docs**: [Link to architecture/design docs]
- **Prior Discussion**: [Link to GitHub issues, conversations]
- **Research**: [Link to experiments, benchmarks]

### Related Tasks

- **Prerequisites**: [TASK-XXX, TASK-YYY]
- **Parallel Work**: [TASK-ZZZ]
- **Follow-up**: [TASK-AAA]

### Code References

- `src/path/to/file.py:line` - [What this file does]
- `tests/path/to/test.py` - [What this tests]

---

## Notes for Implementer

### Before Starting

- [ ] Read related tasks (prerequisites and follow-ups)
- [ ] Understand the "Why" section - this drives design decisions
- [ ] Review examples to understand desired end state
- [ ] Check for blocking dependencies

### During Implementation

- [ ] Follow TDD where specified
- [ ] Update tests as you modify code
- [ ] Keep commit messages clear (reference task number)
- [ ] Document non-obvious decisions in code comments

### Before Marking Complete

- [ ] All acceptance criteria met
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Task file updated with completion date and summary

---

**END OF TASK SPECIFICATION**
