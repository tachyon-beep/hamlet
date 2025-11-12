# Task: Analyze Config DTOs Subsystem

## Context
- **Workspace:** `archeologist_archive/`
- **Read:** `01-discovery-findings.md` (holistic assessment)
- **Write to:** `02-subsystem-catalog.md` (append your section)

## Scope
Analyze the **Config DTOs** subsystem located in `src/townlet/config/`

## Expected Output
Follow the EXACT contract specified in the `analyzing-unknown-codebases` skill:

```markdown
## [Subsystem Name]

**Location:** `path/to/subsystem/`

**Responsibility:** [One sentence describing what this subsystem does]

**Key Components:**
- `file1.ext` - [Brief description]
- `file2.ext` - [Brief description]
- `file3.ext` - [Brief description]

**Dependencies:**
- Inbound: [Subsystems that depend on this one]
- Outbound: [Subsystems this one depends on]

**Patterns Observed:**
- [Pattern 1]
- [Pattern 2]

**Concerns:**
- [Any issues, gaps, or technical debt observed]

**Confidence:** [High/Medium/Low] - [Brief reasoning]

---
```

## Validation Criteria
- [ ] All contract sections complete
- [ ] Confidence level marked with reasoning
- [ ] Dependencies bidirectional (if A depends on B, B shows A as inbound)
- [ ] Output appended to `02-subsystem-catalog.md` (not separate file)
- [ ] NO extra sections added
- [ ] Sections in exact order as contract

## Analysis Focus
- Identify all Pydantic DTO models (BarConfig, AffordanceConfig, etc.)
- Map which runtime systems consume these DTOs
- Document validation patterns (no-defaults principle)
- Identify cross-DTO relationships
