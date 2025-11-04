# HAMLET Task Management System

This directory contains task specifications, templates, and archived completed work.

## Directory Structure

```
docs/tasks/
├── README.md              # This file
├── templates/             # Task templates
│   ├── QUICK-TEMPLATE.md  # Template for quick one-shot tasks (0.5-4 hours)
│   ├── TASK-TEMPLATE.md   # Template for major programming tasks (multi-day)
│   └── BUG-TEMPLATE.md    # Template for bug reports and fixes
├── archive/               # Completed tasks (moved here when done)
│   └── QUICK-001-*.md
└── [Active tasks live here at root level]
```

## Task Types

### QUICK-XXX: Quick One-Shot Tasks (0.5-4 hours)

**Use For**:

- Small feature additions
- Database method implementations
- Configuration updates
- Documentation improvements
- Simple refactors

**Example**: `QUICK-001-AFFORDANCE-DB-INTEGRATION.md`

- Implementing a TODO stub
- Adding tracking to existing system
- Single subsystem changes

**Characteristics**:

- Self-contained (few dependencies)
- Can be completed in one sitting
- Limited scope (3-5 files)
- Minimal architectural impact

### TASK-XXX: Major Programming Tasks (Multi-day)

**Use For**:

- Architectural changes
- Cross-subsystem refactors
- New subsystem implementation
- Breaking changes
- Multi-phase work

**Example**: `TASK-001-VARIABLE-SIZE-METER-SYSTEM.md`

- Removing hardcoded constraints
- Enabling new use cases
- Foundational infrastructure

**Characteristics**:

- Multiple phases (3-5 phases common)
- Significant scope (10+ files)
- Enables downstream work
- May have breaking changes

### BUG-XXX: Bug Reports and Fixes

**Use For**:

- Bug reports
- Regression tracking
- Hotfixes
- Production issues

**Example**: `BUG-027-CHECKPOINT-COMPATIBILITY.md`

- Something that worked before broke
- Unexpected behavior
- Error conditions

**Characteristics**:

- Clear reproduction steps
- Root cause analysis
- Regression test required
- May reveal deeper issues

## Implementation Methodology: QUICK vs TASK

### QUICK Tasks → Direct TDD

**QUICK tasks go straight to implementation using Test-Driven Development:**

```
QUICK-XXX Task Created
    ↓
Write Failing Tests (RED)
    ↓
Implement Minimal Code (GREEN)
    ↓
Refactor & Clean Up
    ↓
Task Complete → Archive
```

**Why Direct TDD**:

- Scope is well-understood (single subsystem)
- Changes are localized (3-5 files)
- Requirements are clear
- Low architectural risk

**Example**: QUICK-001-AFFORDANCE-DB-INTEGRATION

- Task spec defined what to build
- Went straight to writing failing tests
- Implemented method to make tests pass
- Completed in 2.5 hours

### TASK Tasks → Research-Plan-Review Loop

**TASK tasks require research and planning BEFORE implementation:**

```
TASK-XXX Task Created
    ↓
Research Phase (use docs/methods/RESEARCH-PLAN-REVIEW-LOOP.md)
    ↓
Create Implementation Plan (docs/plans/PLAN-XXX.md)
    ↓
Review Plan (identify risks, dependencies)
    ↓
Execute Plan Using TDD (implement each phase)
    ↓
Task Complete (keep at root - reference material)
```

**Why Research First**:

- Scope is complex (10+ files)
- Architectural impact unclear
- Cross-subsystem dependencies
- Breaking changes possible
- Need to explore design space

**Methodology**: `docs/methods/RESEARCH-PLAN-REVIEW-LOOP.md`

1. **Research**: Understand current system, constraints, alternatives
2. **Plan**: Create detailed implementation plan with phases
3. **Review**: Validate plan, identify risks, get feedback
4. **Execute**: Implement plan using TDD for each component

**Plan Location**: `docs/plans/PLAN-XXX-[DESCRIPTIVE-NAME].md`

- Contains phase-by-phase implementation steps
- Exact file paths and line numbers
- Code examples for each change
- Test requirements per phase
- Risk mitigation strategies

**Example**: TASK-001-VARIABLE-SIZE-METER-SYSTEM

- Research: Understand where 8-bar assumption is hardcoded
- Plan: Create migration strategy, compatibility approach
- Review: Validate no breaking changes
- Execute: Implement each phase with TDD

### BUG Tasks → Reproduce-Fix-Prevent

**BUG tasks follow a diagnostic workflow:**

```
BUG-XXX Reported
    ↓
Reproduce Bug Locally
    ↓
Write Failing Regression Test (RED)
    ↓
Identify Root Cause
    ↓
Implement Minimal Fix (GREEN)
    ↓
Verify All Tests Pass
    ↓
Document Prevention Strategy
```

**Why This Flow**:

- Must reproduce to understand
- Regression test prevents recurrence
- Root cause may reveal deeper issues
- Fix must not break other things

---

## Numbering Convention

### QUICK Tasks

- **QUICK-001** through **QUICK-999**
- Sequential numbering
- Retire to archive when complete

### TASK Tasks

- **TASK-000** through **TASK-999**
- May use sub-numbering for related work:
  - TASK-002A, TASK-002B (parallel sub-tasks)
  - TASK-002-PHASE1, TASK-002-PHASE2 (sequential phases)

### BUG Tasks

- **BUG-001** through **BUG-999**
- Sequential by discovery date
- Keep even when fixed (audit trail)

## AI-Friendly Front Matter

**Every task includes rich metadata at the top for AI/human skimming**:

```markdown
**Status**: [Current state]
**Priority**: [Urgency level]
**Estimated Effort**: [Time estimate]
**Keywords**: [Searchable terms]
**Subsystems**: [What's affected]

## AI-Friendly Summary
**What**: [One sentence]
**Why**: [One sentence]
**Scope**: [One sentence]
```

**Purpose**: An AI (or human) can quickly determine if a task is relevant without reading the entire document.

## Workflow

### Creating a Task

1. Copy appropriate template:

   ```bash
   cp docs/tasks/templates/QUICK-TEMPLATE.md docs/tasks/QUICK-042-MY-TASK.md
   ```

2. Fill in front matter (especially AI summary)

3. Complete sections relevant to task type

4. Save and commit

### Working on a Task

**For QUICK Tasks** (Direct TDD):

1. Update **Status** to "In Progress"

2. Write failing tests first (RED):
   - Create test file
   - Write tests for desired behavior
   - Watch tests fail correctly

3. Implement minimal code (GREEN):
   - Write simplest code to pass tests
   - Verify all tests pass
   - No regressions

4. Refactor and document:
   - Clean up code
   - Tests stay green
   - Update task with completion notes

**For TASK Tasks** (Research-Plan-Review):

1. Update **Status** to "In Progress"

2. **Research Phase**:
   - Follow `docs/methods/RESEARCH-PLAN-REVIEW-LOOP.md`
   - Understand current system thoroughly
   - Explore design alternatives
   - Identify constraints and risks

3. **Plan Phase**:
   - Create `docs/plans/PLAN-XXX.md` with detailed steps using docs/plans/PLAN-TEMPLATE.md
   - Include exact file paths and line numbers
   - Define phases with acceptance criteria
   - Specify tests for each phase

4. **Review Phase**:
   - Validate plan completeness
   - Check for architectural issues
   - Identify dependencies
   - Get feedback if needed

5. **Execute Phase**:
   - Implement plan phase-by-phase
   - Use TDD for each component
   - Update plan as you learn
   - Check off acceptance criteria

**For BUG Tasks**:

1. Update **Status** to "Investigating"

2. Reproduce bug locally:
   - Follow reproduction steps
   - Confirm failure mode
   - Document exact conditions

3. Write failing regression test (RED)

4. Identify root cause and implement fix (GREEN)

5. Update **Status** to "Fixed" and document prevention

### Completing a Task

**For QUICK Tasks**:

1. Update **Status** to "Completed"

2. Add **Completed** date

3. Add implementation summary:
   - Methodology used (TDD)
   - Phases completed
   - Files modified
   - Tests added
   - Coverage impact
   - Total time

4. Move to archive:

   ```bash
   mv docs/tasks/QUICK-042-*.md docs/tasks/archive/
   ```

5. Update related tasks (if this unblocked downstream work)

**For TASK Tasks**:

1. Update **Status** to "Completed"

2. Add **Completed** date

3. Add implementation summary in task file

4. Update or finalize plan file in `docs/plans/`

5. **Keep task at root** (reference material for future work)

6. Update related/downstream tasks

**For BUG Tasks**:

1. Update **Status** to "Fixed"

2. Add **Fixed** date and root cause summary

3. Document prevention strategy

4. Keep at root (audit trail) or archive if trivial

## Best Practices

### For Task Authors

**Be Specific**:

- Use concrete examples
- Include code snippets
- Reference line numbers
- Link to related tasks

**Be Realistic**:

- Effort estimates should include testing
- Mark dependencies clearly
- Note risks upfront

**Be AI-Friendly**:

- Front matter should enable quick triage
- Keywords aid discovery
- One-sentence summaries for skimming

### For Implementers

**Before Starting**:

- Read "AI-Friendly Summary" first
- Check dependencies are met
- Understand the "Why"

**During Work**:

- Follow TDD where specified
- Update task as you learn
- Document non-obvious decisions

**Before Completing**:

- All acceptance criteria met
- Tests pass
- Documentation updated
- Task file updated with summary

## Task Templates Explained

### templates/QUICK-TEMPLATE.md

- Lightweight structure
- Focus on implementation steps
- TDD checklist included
- Example: QUICK-001

### templates/TASK-TEMPLATE.md

- Comprehensive structure
- Multiple phases
- Risk assessment
- Migration guide
- Example: TASK-001, TASK-002A

### templates/BUG-TEMPLATE.md

- Reproduction steps
- Root cause analysis
- Fix strategy
- Regression test
- Prevention strategy

## Finding Tasks

### By Keyword

```bash
# Search all tasks for a keyword
grep -r "keyword" docs/tasks/*.md
```

### By Subsystem

```bash
# Find tasks affecting database
grep "**Subsystems**.*database" docs/tasks/*.md
```

### By Status

```bash
# Find all planned tasks
grep "**Status**: Planned" docs/tasks/*.md
```

### By Priority

```bash
# Find all critical tasks
grep "**Priority**: CRITICAL" docs/tasks/*.md
```

## Archive Policy

**When to Archive**:

- Task status = "Completed"
- Implementation summary added
- All acceptance criteria met

**What to Archive**:

- **QUICK tasks**: Always archive when complete
  - Self-contained implementation records
  - Historical value only
  - Move to `archive/` to keep root clean

- **TASK tasks**: Keep at root (rarely archive)
  - Architectural reference material
  - Documents design decisions
  - Referenced by future work
  - Contains research and planning insights
  - Only archive if superseded by newer approach

- **BUG tasks**: Keep at root (archive if trivial)
  - Audit trail for production issues
  - Pattern recognition for similar bugs
  - Archive only simple typo fixes

**Archive Structure**:

```
docs/tasks/archive/
├── 2025-11/           # Optional: organize by month
│   └── QUICK-001-AFFORDANCE-DB-INTEGRATION.md
│   └── QUICK-042-MINOR-CLEANUP.md
│   └── BUG-007-TYPO-FIX.md
└── [Or use flat structure without dates]
```

**Why This Policy**:

- **QUICK**: Implementation details don't affect future architecture
- **TASK**: Design rationale remains relevant for months/years
- **BUG**: Major bugs are learning opportunities, trivial ones clutter

**Keeps Root Focused On**:

- Active architectural work (TASK-XXX)
- Ongoing bugs (BUG-XXX)
- Current quick tasks (QUICK-XXX in progress)

## Integration with Git

### Commit Messages

```bash
# Reference task in commits
git commit -m "QUICK-042: Implement feature X

Implements database method as specified in QUICK-042.
- Added insert_foo() method
- Added 3 unit tests
- Coverage: 87%

See docs/tasks/QUICK-042-*.md for details."
```

### Pull Requests

- Link to task in PR description
- Mark task completed when PR merges
- Archive task after merge

## Questions?

**For task creation**: See templates and examples
**For workflow**: Follow steps in this README
**For AI assistance**: Front matter enables smart task discovery

---

**Task Management Philosophy**:
Tasks are living documents. They start as specifications, evolve during implementation, and end as historical records of what was built and why.
