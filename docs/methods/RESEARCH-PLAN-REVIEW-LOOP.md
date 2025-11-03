# The Research → Plan → Review Loop

## Overview

This document captures the **research-driven development workflow** used for HAMLET's complex architectural decisions. It describes when to research deeply vs when to "just implement it," and lessons learned from applying this process.

This is not intended for coding, which follows TDD best practices. Instead, this is for **high-level planning or design decisions** where multiple approaches exist and the right choice isn't obvious.

---

## The Three-Phase Loop

```
┌─────────────────────────────────────────────────────────┐
│                    1. RESEARCH PHASE                    │
│  "What are we actually building? What are the options?" │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                     2. PLAN PHASE                       │
│    "Given research, what's the optimal approach?"       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    3. REVIEW PHASE                      │
│   "Does this achieve the goals? What did we learn?"     │
└────────────────────┬────────────────────────────────────┘
                     │
                     └──────────> (iterate or implement)
```

---

## Phase 1: Research

### Purpose

**Understand the design space** before committing to an approach. Identify constraints, tradeoffs, and options that aren't obvious from requirements alone.

### When to Research (vs Just Implement)

**✅ DO RESEARCH when**:

- Multiple viable approaches exist with unclear tradeoffs
- Domain expertise is needed (coordinate systems, economic models, utility functions)
- Implementation would be costly to change later
- Pedagogical value depends on design choices
- Best practices aren't well-known
- **Risk of "yolo with TDD" is high** (would make wrong architecture choice)

**❌ SKIP RESEARCH when**:

- Problem is purely mechanical (move file A to B)
- One obvious approach exists
- Easy to change later with refactoring
- Well-established patterns apply directly
- "Just try it and see" is fast enough

### Example: When We Researched

**Spatial Substrates** (researched ✅):

- **Why**: Multiple topologies (2D, 3D, hex, continuous, graph, aspatial)
- **Tradeoff**: Simplicity vs expressivity vs pedagogical value
- **Outcome**: Identified substrate as "optional overlay" on meters (key insight)
- **Value**: Prevented hardcoding 2D assumptions everywhere

**UAC Compiler Infrastructure** (researched ✅):

- **Why**: 8-bar constraint appeared to be "design decision" but was actually technical debt
- **Tradeoff**: ABI stability vs design space flexibility
- **Outcome**: Identified variable-size meters as highest-leverage change
- **Value**: Prevents building on broken foundation

**Action Space** (already documented in TASK-002):

- **Why**: Actions must match substrate (4-way for square, 6-way for hex)
- **Dependency**: Needed substrate research first
- **Value**: Avoided defining actions that wouldn't work with future substrates

### Example: When We'd Skip Research

**Move configs from legacy to townlet** (no research needed):

- Just copy files and update imports

**Fix typo in YAML** (no research needed):

- Just fix it

**Add new meter to existing system** (no research needed):

- Well-defined pattern, just follow it

### Research Deliverables

A good research document contains:

1. **Problem Statement**: What are we trying to solve?
2. **Design Space**: What options exist? (taxonomy, spectrum)
3. **Tradeoffs Analysis**: Pros/cons of each option
4. **Recommendation**: Which approach and why
5. **Implementation Sketch**: Rough code/config examples
6. **Effort Estimate**: How long to implement
7. **Priority/Value**: Why this matters (pedagogical, architectural, performance)

**Examples**:

- `docs/research/RESEARCH-SPATIAL-SUBSTRATES.md`
- `docs/research/RESEARCH-UAC-COMPILER-INFRASTRUCTURE.md`
- `docs/research/RESEARCH-OPPORTUNITIES-UAC.md`

---

## Phase 2: Plan

### Purpose

**Translate research into actionable implementation plan** with clear tasks, dependencies, and success criteria.

### Planning Process

1. **Synthesize Research**: Consolidate findings into recommended approach
2. **Break Down Work**: Divide into phases (Phase 1: foundation, Phase 2: features)
3. **Define Dependencies**: What must come first? (substrate → actions → compilation)
4. **Specify Interfaces**: What YAML schemas? What APIs?
5. **Identify Risks**: What could go wrong?
6. **Set Success Criteria**: How do we know it worked?

### Plan Deliverables

A good TASK document contains:

1. **Problem Statement**: Why current state is inadequate
2. **Solution Overview**: High-level approach
3. **Implementation Plan**: Phase-by-phase breakdown with code examples
4. **Configuration Examples**: Concrete YAML showing usage
5. **Benefits**: What this enables
6. **Dependencies**: What must exist first (other TASKs)
7. **Success Criteria**: Checkboxes for "done"
8. **Effort Estimate**: Hours per phase
9. **Risks & Mitigations**: What could break and how to prevent it

**Examples**:

- `docs/TASK-000-CONFIGURABLE-SPATIAL-SUBSTRATES.md`
- `docs/TASK-001-UAC-CONTRACTS.md`
- `docs/TASK-002-UAC-ACTION-SPACE.md`

### Sequencing Tasks

From research, we identified optimal implementation order:

```
TASK-000 (Spatial Substrates)
    ↓
TASK-001 (Schema Validation) ← Can be parallel
    ↓
TASK-002 (Action Space) ← Depends on substrates
    ↓
TASK-003 (Universe Compilation) ← Depends on all configs
    ↓
TASK-004 (BRAIN_AS_CODE) ← Depends on compiled universe
```

**Key Insight**: Substrate is most fundamental (defines coordinate system). Actions depend on substrate (can't have UP/DOWN in aspatial universe). Compilation validates everything works together.

**Documented in**: `docs/TASK-SEQUENCE-RATIONALE.md`

---

## Phase 3: Review

### Purpose

**Validate that implementation achieves goals** and capture lessons for future work.

### Review Questions

**Did it work?**

- Success criteria met?
- Tests pass?
- Configs load and validate correctly?

**What did we learn?**

- Were estimates accurate?
- What was harder than expected?
- What was easier than expected?
- Any surprises?

**What would we do differently?**

- Better research questions?
- Different sequencing?
- Missing considerations?

### Review Deliverables

(This phase happens after implementation - documented here as template)

**Post-Implementation Review Document**:

1. **What We Built**: Summary of implementation
2. **Estimates vs Actuals**: Planned hours vs actual hours
3. **Key Decisions**: What choices were made during implementation
4. **Surprises**: What we didn't anticipate
5. **Lessons Learned**: What to remember for next time
6. **Tech Debt Introduced**: What shortcuts were taken (if any)
7. **Follow-Up Work**: What's next (if anything)

---

## Key Lessons Learned

### Lesson 1: "Bones First, Content Later"

**Observation**: We almost researched reward models and economic tuning before ensuring the UAC compiler was solid.

**Insight**: **Get the infrastructure right first**. No point designing perfect reward functions if the 8-bar constraint limits what universes you can create.

**Pattern**:

- ✅ Foundation layer (substrate, meters, compiler)
- ✅ Configuration layer (actions, affordances, cascades)
- ✅ Content layer (reward models, economic packs, teaching materials)

**Applied**:

- Researched spatial substrates (foundation)
- Researched UAC compiler (foundation)
- Deferred reward model research (content) until foundation is solid

---

### Lesson 2: Technical Debt vs Design Constraints

**Observation**: The 8-bar constraint appeared in docs as "bars are stable ABI" (design decision), but analysis revealed it's **technical debt** masquerading as a constraint.

**Test**: "If we removed this constraint, would the system be **more expressive** or **more fragile**?"

- More expressive → It's technical debt
- More fragile → It's a design constraint

**Applied**: Variable-size meters make system more expressive (4-meter tutorials, 16-meter complex sims) without breaking anything fundamental. Therefore, 8-bar limit is debt, not design.

**Pattern**: Question "hardcoded" assumptions. Are they **necessary** or just **historical**?

---

### Lesson 3: Substrate Thinking

**Observation**: The spatial grid seemed fundamental, but research revealed **meters are the true universe**, grid is just an optional overlay.

**Insight**: **Find the substrate** - the most fundamental layer everything builds on.

**HAMLET substrate hierarchy**:

1. **Meters** (energy, health, money) - true state space
2. **Spatial substrate** (2D grid, 3D grid, graph, or none) - optional positioning layer
3. **Actions** (movement, interaction) - defined by substrate
4. **Affordances** (bed, job, hospital) - placed in substrate
5. **Cascades** (hunger → health) - meter dynamics

**Pattern**: When designing systems, identify the substrate (most fundamental abstraction) and make everything else build on it.

---

### Lesson 4: "Fuck Around and Find Out" Pedagogy

**Observation**: Suggested toroidal boundaries and 3D grids might be "too complex" for students.

**Reframe**: "The bar for this sort of thing should be low if there's any pedagogical value, letting students or hobbyists just 'fuck around and find out' is half the point."

**Insight**: **Experimentation is the pedagogy**. Enable exploration, not just guided lessons.

**Applied**:

- Include 3D cubic grids (high value, low effort)
- Include toroidal boundaries (literally change clamp to modulo)
- Include aspatial substrates (reveals grid is optional)
- Include hexagonal grids (teaches coordinate systems)

**Pattern**: If feature enables experimentation ("What if the world was a sphere?"), include it even if it's not "serious" or "practical."

---

### Lesson 5: Research Identifies Leverage Points

**Observation**: Could spend 20 hours implementing reward models, but 12 hours on variable-size meters unlocks entire design space.

**Insight**: Research reveals **highest-leverage changes** - where small effort enables huge capability.

**Leverage Analysis**:

- **Variable meters** (12h) → Unblocks 4-meter to 16-meter universes
- **Toroidal boundaries** (1h) → Enables topology experimentation
- **Aspatial substrate** (2h) → Reveals meters are fundamental

vs lower-leverage:

- **Reward model tuning** (8h) → Only affects one curriculum level
- **Economic pack variants** (10h) → Requires balanced foundation first

**Pattern**: **Prioritize enabling capabilities over optimizing content**. Capabilities compound, content is linear.

---

### Lesson 6: Dependency Ordering from Research

**Observation**: Initially planned TASK-000 (Action Space), but research revealed substrate must come first.

**Insight**: **Research clarifies dependencies** that requirements don't show.

**Example**:

- Requirement: "Make action space configurable"
- Research insight: "Actions depend on substrate (4-way for square, 6-way for hex, 0-way for aspatial)"
- Conclusion: Substrate (TASK-000) must come before Actions (TASK-002)

**Pattern**: Research phase reveals **true dependency graph**, not just logical task breakdown.

---

### Lesson 7: Exposed vs Hidden Knobs

**Observation**: Many "gameplay parameters" were hardcoded in Python (`move_energy_cost=0.005`).

**Test**: "Does changing this parameter change gameplay meaningfully?"

- Yes → Should be in YAML (exposed knob)
- No → Can stay in code (implementation detail)

**Examples**:

- ✅ Exposed: `move_energy_cost` (affects survival difficulty)
- ✅ Exposed: `boundary_behavior` (affects navigation strategy)
- ✅ Exposed: `distance_metric` (affects spatial reasoning)
- ❌ Hidden: `tensor_dtype` (implementation detail)
- ❌ Hidden: `device_placement` (implementation detail)

**Pattern**: **Expose all parameters that affect the world's physics**. Hide only true implementation details.

---

### Lesson 8: Research Documents as Future Reference

**Observation**: Research documents contain design rationale that won't fit in code comments.

**Value**:

- "Why did we choose hexagonal grids?" → See research doc
- "What other reward models were considered?" → See research doc
- "Why variable-size meters instead of fixed?" → See research doc

**Pattern**: Research docs are **permanent context**. Future maintainers read them to understand **why**, not just **what**.

---

### Lesson 9: When Research Prevents Rework

**Observation**: Spatial substrate research took 6-8 hours, but prevented 40+ hours of hardcoded 2D assumptions throughout codebase.

**Calculation**:

- Research: 8 hours
- Prevented rework: ~40 hours (if we'd hardcoded 2D everywhere, then had to refactor for 3D/hex/graph)
- **Net savings: 32 hours**

**Pattern**: **Research is insurance against costly rework**. When stakes are high (foundational architecture), research pays for itself.

---

### Lesson 10: Distinguishing Research from Analysis Paralysis

**Warning**: Research can become procrastination. How to avoid?

**Good Research** (productive):

- Time-boxed (6-12 hours max for high-priority topics)
- Delivers concrete recommendations
- Identifies highest-leverage changes
- Has clear stopping criteria ("once we've mapped the design space, stop")

**Analysis Paralysis** (unproductive):

- Open-ended ("let's research forever")
- No deliverables or recommendations
- Explores low-value options
- No stopping criteria

**Test**: "Could we start implementing now with acceptable risk?"

- No → Keep researching
- Yes → Stop researching, start planning

**Applied**: Spatial substrate research stopped once we had:

1. Taxonomy of substrate types (2D, 3D, hex, continuous, graph, aspatial)
2. Implementation sketch (SpatialSubstrate interface)
3. Priority ranking (3D + toroidal first, hex second, graph deferred)
4. Effort estimates per phase

At that point, continuing research would be diminishing returns. Start planning/implementing.

---

## Decision Framework

### "Should I Research This?"

Ask these questions:

1. **Is this foundational?** (Affects many things vs one thing)
2. **Are there multiple approaches?** (Design space vs obvious solution)
3. **Is change costly?** (Hard to refactor vs easy to change)
4. **Is domain knowledge needed?** (Coordinate systems, economics, RL theory)
5. **Is there pedagogical value?** (Students learn from experimentation)

**If 3+ yes**: Research first (6-12 hours)
**If 2 yes**: Light research (2-4 hours)
**If 0-1 yes**: Just implement (TDD)

### "Is My Research Done?"

Stop researching when you have:

1. ✅ **Design space mapped** - What options exist?
2. ✅ **Tradeoffs understood** - Pros/cons clear?
3. ✅ **Recommendation made** - Which approach?
4. ✅ **Implementation sketch** - Rough code examples?
5. ✅ **Effort estimated** - How long to build?
6. ✅ **Priorities set** - What order to implement?

**If all checked**: Move to planning phase.

---

## Templates

### Research Document Template

```markdown
# Research: [Topic Name]

## Problem Statement
What are we trying to solve? Why is it hard?

## Design Space
What options exist?
- Option A: [description]
- Option B: [description]
- Option C: [description]

## Tradeoffs Analysis
| Option | Pros | Cons | Effort |
|--------|------|------|--------|
| A      | ... | ... | 8h     |
| B      | ... | ... | 12h    |
| C      | ... | ... | 20h    |

## Recommendation
Which approach and why?

## Implementation Sketch
Rough code/config examples showing how it would work.

## Priority/Value
Why this matters (pedagogical, architectural, performance).

## Estimated Effort
Phase 1: X hours
Phase 2: Y hours
Total: Z hours
```

### TASK Document Template

```markdown
# TASK-NNN: [Feature Name]

## Problem Statement
Why is current state inadequate?

## Solution Overview
High-level approach.

## Implementation Plan
### Phase 1: Foundation
- Step 1
- Step 2

### Phase 2: Features
- Step 3
- Step 4

## Configuration Examples
Concrete YAML showing usage.

## Benefits
What this enables.

## Dependencies
What must exist first?

## Success Criteria
- [ ] Criteria 1
- [ ] Criteria 2

## Effort Estimate
Phase 1: X hours
Phase 2: Y hours
Total: Z hours

## Risks & Mitigations
What could go wrong and how to prevent it?
```

---

## Summary

**The Research → Plan → Review loop prevents architectural mistakes** by understanding the design space before committing to an approach.

**Key Principles**:

1. **Research foundational changes** (substrate, compiler, meter system)
2. **Skip research for mechanical work** (move files, fix typos)
3. **Bones first, content later** (infrastructure before content)
4. **Question "hardcoded" assumptions** (debt vs design)
5. **Find the substrate** (most fundamental abstraction)
6. **Enable experimentation** ("fuck around and find out")
7. **Identify leverage points** (highest-impact changes)
8. **Research reveals dependencies** (true task ordering)
9. **Expose all gameplay knobs** (everything configurable)
10. **Time-box research** (avoid analysis paralysis)

**Workflow**:

- **Research**: 6-12 hours for high-priority topics → Design space mapped
- **Plan**: 2-4 hours → TASK document with implementation phases
- **Implement**: Follow plan → Build it
- **Review**: After implementation → What did we learn?

**Slogan**: "Research to prevent rework, plan to prevent confusion, review to prevent forgetting."
