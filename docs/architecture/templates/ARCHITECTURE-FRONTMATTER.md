# [Section Number]. [Document Title]

**Document Type**: [Overview | Component Spec | Design Rationale | Interface Spec | Requirements | Implementation Guide | Success Criteria]
**Status**: [Draft | In Review | Approved | Superseded | Deprecated]
**Version**: X.Y
**Last Updated**: YYYY-MM-DD
**Owner**: [Role or team responsible for this section]
**Parent Document**: [Name of larger architecture document, if applicable]

**Audience**: [Primary audience - e.g., "Engineers, Researchers" or "Governance, Policy" or "Instructors, Students"]
**Technical Level**: [Executive | Intermediate | Deep Technical]
**Estimated Reading Time**: [X minutes for skim | Y minutes for deep read]

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
[1-2 sentences: What architectural aspect/component/decision does this document cover?]

**Why This Document Exists**:
[1-2 sentences: What problem does this architecture solve? Why was this design approach chosen?]

**Who Should Read This**:
- **Must Read**: [Roles that absolutely need this - e.g., "Backend engineers implementing the runtime engine"]
- **Should Read**: [Roles that benefit from understanding - e.g., "Anyone working on agent training"]
- **Optional**: [Roles that might find it interesting - e.g., "Students learning RL concepts"]

**Reading Strategy**:
- **Quick Scan** (2 min): Read sections [X, Y] for the core concept
- **Partial Read** (10 min): Read sections [X, Y, Z] for working knowledge
- **Deep Study** (30 min): Read entire document for complete understanding

---

## Document Scope

**In Scope**:
- [What architectural topics/components are covered]
- [What design decisions are documented]
- [What level of detail is provided]

**Out of Scope**:
- [What related topics are NOT covered here]
- [What details are deferred to other documents]
- [What implementation specifics are excluded]

**Boundaries**:
[1-2 sentences clarifying where this document's responsibility ends and other documents begin]

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [Documents to read BEFORE this one - e.g., "01-executive-summary.md"]
- **Builds On**: [Documents this one extends/refines - e.g., "02-brain-as-code.md"]
- **Related**: [Peer documents on similar topics - e.g., "08-declarative-goals.md"]
- **Next**: [Logical next document to read - e.g., "07-telemetry-ui-surfacing.md"]

**Section Number**: [X] / [Total]
**Architecture Layer**: [Conceptual | Logical | Physical | Implementation]

---

## Keywords for Discovery

**Primary Keywords**: [3-5 keywords for what this document is ABOUT]
**Secondary Keywords**: [3-5 keywords for concepts/patterns used]
**Subsystems**: [List relevant subsystems - e.g., agent, environment, curriculum, exploration]
**Design Patterns**: [Relevant patterns - e.g., Factory, Observer, Strategy, Interpreter]

**Quick Search Hints**:
- Looking for [X]? → See section [Y]
- Looking for [A]? → See section [B]
- Looking for [C]? → See related doc [D]

---

## Version History

**Current Version**: X.Y - YYYY-MM-DD
[Brief summary of current version changes]

**Previous Versions**:
- **X.Y-1** (YYYY-MM-DD): [What changed and why]
- **X.Y-2** (YYYY-MM-DD): [What changed and why]

**Deprecation Notice** (if applicable):
[If this document is superseded, explain what replaces it and migration path]

---

## Document Type Specifics

<!-- REMOVE sections not applicable to your document type -->

### For Overview Documents (Type: Overview)

**Strategic Question This Answers**:
[The high-level "why" or "what" question this overview addresses]

**Key Takeaways** (3-5 bullets):
- [Most important concept #1]
- [Most important concept #2]
- [Most important concept #3]

**Mental Models**:
[1-2 sentences describing the conceptual framework or analogy that helps understand this system]

---

### For Component Specification Documents (Type: Component Spec)

**Component Name**: [Official component name]
**Component Type**: [Module | Service | Engine | Interface | Filter]
**Location in Codebase**: [e.g., `src/townlet/agent/factory.py`]

**Interface Contract**:
- **Inputs**: [What this component receives]
- **Outputs**: [What this component produces]
- **Dependencies**: [What this component requires]
- **Guarantees**: [What this component promises]

**Critical Properties**:
- [Essential property #1 - e.g., "Idempotent"]
- [Essential property #2 - e.g., "Deterministic"]
- [Essential property #3 - e.g., "Stateless"]

---

### For Design Rationale Documents (Type: Design Rationale)

**Design Question Addressed**:
[The specific design question or trade-off this document resolves]

**Alternatives Considered**:
1. **[Alternative A]**: [Why rejected/deferred]
2. **[Alternative B]**: [Why rejected/deferred]
3. **[Chosen approach]**: [Why selected]

**Key Trade-offs**:
- **Chosen**: [What we optimized for - e.g., "Reproducibility, accountability"]
- **Sacrificed**: [What we accepted as cost - e.g., "Runtime overhead, complexity"]

**Decision Drivers**:
- [Driver #1 - e.g., "Regulatory compliance requirements"]
- [Driver #2 - e.g., "Pedagogical transparency"]
- [Driver #3 - e.g., "Research reproducibility"]

---

### For Interface Specification Documents (Type: Interface Spec)

**Interface Name**: [Name of the interface/protocol/API]
**Interface Type**: [API | Protocol | Data Format | Event Stream]
**Stability**: [Experimental | Stable | Frozen | Deprecated]

**Contract Guarantees**:
- **Backward Compatibility**: [Yes/No and policy]
- **Forward Compatibility**: [Yes/No and policy]
- **Versioning Scheme**: [How versions are managed]

**Implementation Status**:
- ✅ **Implemented**: [What's done]
- 🚧 **In Progress**: [What's underway]
- ❌ **Not Started**: [What's planned]

---

### For Requirements Documents (Type: Requirements)

**Requirement Category**: [Functional | Non-Functional | Quality Attribute]
**Priority**: [Must Have | Should Have | Nice to Have]

**Success Criteria**:
- [Measurable criterion #1]
- [Measurable criterion #2]
- [Measurable criterion #3]

**Validation Method**:
[How we verify these requirements are met - e.g., "Unit tests", "Integration tests", "Manual verification"]

---

### For Implementation Guide Documents (Type: Implementation Guide)

**Implementation Phase**: [Phase number/name]
**Estimated Effort**: [Effort estimate]
**Prerequisite Phases**: [What must be done first]

**Implementation Checkpoints**:
- [ ] Checkpoint #1
- [ ] Checkpoint #2
- [ ] Checkpoint #3

**Validation Steps**:
- [ ] Validation #1
- [ ] Validation #2

---

### For Success Criteria Documents (Type: Success Criteria)

**What "Success" Means**:
[1-2 sentences defining success for this architecture]

**Measurable Outcomes**:
- [Metric #1 with target value]
- [Metric #2 with target value]
- [Metric #3 with target value]

**Evaluation Method**:
[How these outcomes will be measured and validated]

---

## Diagrams and Visuals

**Diagrams Included**:
- [Diagram type #1 - e.g., "Component interaction diagram (§6.3)"]
- [Diagram type #2 - e.g., "Data flow diagram (§6.5)"]

**External Visual References**:
- [Link to Mermaid diagram source, if applicable]
- [Link to architecture diagrams repository, if applicable]

---

## Open Questions and Future Work

**Unresolved Design Questions**:
- [Question #1 requiring future decision]
- [Question #2 requiring future decision]

**Planned Enhancements**:
- [Enhancement #1 (see TASK-XXX)]
- [Enhancement #2 (see TASK-XXX)]

**Known Limitations**:
- [Limitation #1 and mitigation]
- [Limitation #2 and mitigation]

---

## References

**Related Architecture Docs**:
- [Document name]: [Brief description]
- [Document name]: [Brief description]

**Related Tasks**:
- TASK-XXX: [Task name and relevance]
- QUICK-XXX: [Quick task name and relevance]

**Code References**:
- `path/to/file.py` - [What this implements from this doc]
- `path/to/file.py` - [What this implements from this doc]

**External References**:
- [Paper/article citation]: [Relevance]
- [Design pattern documentation]: [Relevance]

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT BELOW**

---

<!-- ARCHITECTURE DOCUMENT CONTENT STARTS HERE -->
