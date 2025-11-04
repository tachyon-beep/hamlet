# ADR-XXX: [Architecture Decision Title - System Design Choice]

**Status**: [Proposed | Accepted | Deprecated | Superseded]
**Type**: [Architecture | Design Pattern | Technology Choice | Data Model]
**Priority**: [Low | Medium | High | CRITICAL]
**Date Proposed**: YYYY-MM-DD
**Date Adopted**: YYYY-MM-DD (when accepted)
**Deciders**: [Architecture Team | Development Team | Tech Lead]
**Affects**: [List specific subsystems or "All systems"]

**Keywords**: [Add 5-8 searchable keywords for AI/human discovery]
**Subsystems**: [List all affected subsystems]
**Architecture Impact**: [None | Minor | Major | Breaking]
**Breaking Changes**: [Yes/No - if yes, describe migration path]
**Supersedes**: [ADR-XXX | N/A]
**Superseded By**: [ADR-YYY | N/A] (if deprecated)

---

## AI-Friendly Summary (Skim This First!)

**What**: [One sentence: Core architectural decision being made]
**Why**: [One sentence: Problem this solves or value it provides]
**Impact**: [One sentence: Major implications or changes to system design]

**Quick Assessment**:

- **Problem**: [What architectural constraint/limitation are we solving]
- **Solution**: [High-level approach taken]
- **Trade-offs**: [Key trade-offs accepted (performance vs simplicity, etc.)]
- **Alternatives**: [Brief mention of rejected alternatives]

**Decision Point**: If this doesn't affect the subsystems you work on, STOP READING HERE. If you're implementing or extending this architecture, continue reading for design rationale and constraints.

---

## Context

### Problem Statement

[Describe the architectural problem or constraint that necessitates a decision]

**Current Limitation**:

- [Limitation 1]
- [Limitation 2]
- [Limitation 3]

**Why This Matters**:

- [Impact 1 - what can't be built without this]
- [Impact 2 - what breaks or scales poorly]
- [Impact 3 - what becomes unmaintainable]

### Forces at Play

**Technical Forces**:

- [Force 1: e.g., "Need for horizontal scalability"]
- [Force 2: e.g., "Memory constraints on GPU"]
- [Force 3: e.g., "Type safety requirements"]

**Business Forces**:

- [Force 1: e.g., "Pedagogical value - students need to understand X"]
- [Force 2: e.g., "Research flexibility - must support experimentation"]

**Constraints**:

- [Constraint 1: e.g., "Must be GPU-native"]
- [Constraint 2: e.g., "Must support YAML configuration"]
- [Constraint 3: e.g., "Backward compatibility required"]

### Scope

**In Scope**:

- [What this decision covers]
- [Systems affected]

**Out of Scope**:

- [What this explicitly does NOT cover]
- [Future work deferred to other ADRs]

---

## Decision

### Core Decision

**[One sentence statement of the architectural decision]**

### Architecture Overview

**Design Principle**: [One sentence capturing the fundamental design philosophy]

**Key Insight**: [The core realization that makes this architecture work]

### Components

**1. [Component 1 Name]**

**Responsibility**: [What this component does]

**Interface**:

```python
# Key interface or contract
class ComponentName:
    def key_method(self, args) -> ReturnType:
        """Contract description"""
```

**Dependencies**: [What this depends on]

**2. [Component 2 Name]**

**Responsibility**: [What this component does]

**Interface**:

```python
# Key interface or contract
```

**Dependencies**: [What this depends on]

### Data Flow

```
[Component A] --data--> [Component B] --transformed--> [Component C]
     ↓                        ↑
     └────────feedback────────┘
```

**Flow Description**:

1. [Step 1 in data flow]
2. [Step 2 in data flow]
3. [Step 3 in data flow]

### Key Design Decisions

**Decision 1: [Specific design choice]**

**Rationale**: [Why this choice was made]

**Alternative Considered**: [What else was considered]

**Trade-off Accepted**: [What we gave up for this benefit]

**Decision 2: [Specific design choice]**

**Rationale**: [Why this choice was made]

**Alternative Considered**: [What else was considered]

**Trade-off Accepted**: [What we gave up for this benefit]

---

## Rationale

### Why This Approach?

**Advantage 1: [Key benefit]**

[Explanation of how this architecture provides this benefit]

**Example**:

```python
# Code example demonstrating the benefit
```

**Advantage 2: [Key benefit]**

[Explanation]

**Advantage 3: [Key benefit]**

[Explanation]

### Why Not Alternatives?

**Alternative 1 Rejected**: [Name of alternative]

**What it proposed**: [Brief description]

**Why rejected**: [Technical/business reason]

**Trade-off**: [What we would have gained vs lost]

**Alternative 2 Rejected**: [Name of alternative]

**What it proposed**: [Brief description]

**Why rejected**: [Technical/business reason]

---

## Consequences

### Positive

**1. [Positive consequence 1]**

[Detailed explanation]

**Enables**:

- [Capability 1 this unlocks]
- [Capability 2 this unlocks]

**2. [Positive consequence 2]**

[Detailed explanation]

### Negative

**1. [Negative consequence 1]**

[Detailed explanation]

**Mitigation**:

- [How we reduce this impact]

**2. [Negative consequence 2]**

[Detailed explanation]

**Mitigation**:

- [How we reduce this impact]

### Neutral

**1. [Neutral consequence 1]**

[Something that changes but isn't clearly positive or negative]

---

## Implementation

### Migration Path

**For Existing Systems**:

**Phase 1: [Compatibility Layer]**

1. [Step 1]
2. [Step 2]

**Phase 2: [Gradual Migration]**

1. [Step 1]
2. [Step 2]

**Phase 3: [Cleanup]**

1. [Step 1]
2. [Step 2]

**Backward Compatibility**:

- [How old code continues to work]
- [Deprecation timeline if applicable]

### Code Changes

**Files to Create**:

- `src/path/to/new_component.py` - [Purpose]
- `src/path/to/interface.py` - [Purpose]

**Files to Modify**:

- `src/path/to/existing.py` - [What changes]
- `configs/schema.yaml` - [What changes]

**Files to Deprecate** (if applicable):

- `src/path/to/old_component.py` - [Replacement]

### Testing Strategy

**Unit Tests**:

- [ ] Test component 1 in isolation
- [ ] Test component 2 in isolation
- [ ] Test interfaces/contracts

**Integration Tests**:

- [ ] Test data flow end-to-end
- [ ] Test backward compatibility
- [ ] Test migration path

**Performance Tests** (if applicable):

- [ ] Benchmark critical path
- [ ] Compare against baseline
- [ ] Verify scalability claims

---

## Examples

### Example 1: [Simple Use Case]

**Scenario**: [Description of use case]

**Before** (old architecture):

```python
# Code showing old approach
```

**After** (new architecture):

```python
# Code showing new approach
```

**Benefits Demonstrated**:

- [How this example shows the advantage]

### Example 2: [Complex Use Case]

**Scenario**: [Description of use case]

**Architecture Diagram**:

```
[Component interactions for this use case]
```

**Code**:

```python
# Implementation example
```

**Benefits Demonstrated**:

- [How this example shows the advantage]

---

## Risks and Assumptions

### Technical Risks

**Risk 1: [Description]**

- **Severity**: [High/Medium/Low]
- **Likelihood**: [High/Medium/Low]
- **Mitigation**: [How to address]
- **Contingency**: [Fallback plan]

**Risk 2: [Description]**

- **Severity**: [High/Medium/Low]
- **Likelihood**: [High/Medium/Low]
- **Mitigation**: [How to address]

### Assumptions

**Assumption 1**: [What we're assuming is true]

- **Impact if wrong**: [What breaks if this assumption is violated]
- **Validation**: [How we verify this assumption]

**Assumption 2**: [What we're assuming is true]

- **Impact if wrong**: [What breaks if this assumption is violated]

---

## Alternatives Considered

### Alternative 1: [Name]

**Description**: [What this architecture proposed]

**Pros**:

- ✅ [Advantage 1]
- ✅ [Advantage 2]
- ✅ [Advantage 3]

**Cons**:

- ❌ [Disadvantage 1]
- ❌ [Disadvantage 2]
- ❌ [Disadvantage 3]

**Why Rejected**: [Primary reason for not choosing this]

**Trade-off Analysis**:

| Criterion | Chosen Architecture | This Alternative |
|-----------|---------------------|------------------|
| Performance | [Rating] | [Rating] |
| Maintainability | [Rating] | [Rating] |
| Scalability | [Rating] | [Rating] |
| Complexity | [Rating] | [Rating] |

### Alternative 2: [Name]

**Description**: [What this architecture proposed]

**Pros**:

- ✅ [Advantage 1]
- ✅ [Advantage 2]

**Cons**:

- ❌ [Disadvantage 1]
- ❌ [Disadvantage 2]

**Why Rejected**: [Primary reason for not choosing this]

### Alternative 3: Do Nothing

**Description**: Keep current architecture as-is

**Pros**:

- ✅ No migration cost
- ✅ No new complexity

**Cons**:

- ❌ [Problem 1 remains unsolved]
- ❌ [Problem 2 remains unsolved]

**Why Rejected**: [Why status quo is unacceptable]

---

## Compatibility

### Backward Compatibility

**Existing Code**: [Compatible | Breaks | Deprecation Required]

**Migration Required**: [Yes/No]

**Deprecation Timeline**:

- [Date]: [What becomes deprecated]
- [Date]: [What becomes unsupported]

### Forward Compatibility

**Future Extensions**:

- [Extension 1 that this architecture supports]
- [Extension 2 that this architecture supports]

**Locked-In Decisions** (hard to change later):

- [Decision 1 that future changes must respect]
- [Decision 2 that future changes must respect]

---

## Success Metrics

### Quantitative

**Metric 1: [Measurable outcome]**

- **Baseline**: [Current value]
- **Target**: [Goal after implementation]
- **Measurement**: [How to measure]

**Metric 2: [Measurable outcome]**

- **Baseline**: [Current value]
- **Target**: [Goal after implementation]
- **Measurement**: [How to measure]

### Qualitative

**Goal 1**: [Desired outcome]
**Goal 2**: [Desired outcome]
**Goal 3**: [Desired outcome]

---

## Review and Evolution

### Review Schedule

**Frequency**: [Quarterly | After Major Release | As Needed]

**Next Review**: YYYY-MM-DD

**Review Criteria**:

- [When to revisit this decision]
- [What metrics to check]
- [What assumptions to validate]

### Evolution Path

**Likely Future Changes**:

- [Change 1 that this architecture anticipates]
- [Change 2 that this architecture anticipates]

**Unlikely But Possible**:

- [Change 1 that would require major rework]
- [Change 2 that would require major rework]

---

## References

### Related Decisions

- **Prerequisites**: [ADR-XXX, ADR-YYY]
- **Related**: [ADR-ZZZ]
- **Supersedes**: [ADR-AAA (if applicable)]
- **Policies**: [PDR-XXX (if applicable)]

### Documentation

- **Design Docs**: [Link to detailed design]
- **Implementation Guide**: [Link to how-to]
- **API Docs**: [Link to API reference]

### Code References

- `src/path/to/component.py:LineRange` - [Core implementation]
- `tests/path/to/test.py` - [Test suite]
- `configs/example.yaml` - [Example configuration]

### External References

- **Research Papers**: [Link if based on published research]
- **Industry Standards**: [Link to standards followed]
- **Prior Art**: [Link to similar architectures in other projects]
- **Benchmarks**: [Link to performance comparisons]

---

## Appendix

### Glossary

**Term 1**: [Definition used in this ADR]
**Term 2**: [Definition used in this ADR]

### Diagrams

**Component Diagram**:

```
[ASCII art or link to diagram showing component relationships]
```

**Sequence Diagram**:

```
[ASCII art or link to diagram showing interaction flow]
```

**Deployment Diagram** (if applicable):

```
[ASCII art or link to diagram showing runtime deployment]
```

---

**Status**: [Proposed | Accepted | Deprecated | Superseded] ✅
**Effective Date**: YYYY-MM-DD
**Supersedes**: [ADR-XXX | N/A]
**Superseded By**: [ADR-YYY | N/A]
