# TASK-003: UAC Universe Compilation Pipeline - Planning Status

**Status:** 🟨 PLANNING COMPLETE WITH REVISIONS NEEDED
**Date:** 2025-11-04
**Process:** Research → Plan → Review → Iterate

---

## Planning Process Summary

Successfully executed a staged planning process for TASK-003 using three specialized subagents:

### 1. Research Subagent (COMPLETED ✅)

- **Task:** Comprehensive investigation of current action space implementation
- **Findings:**
  - Action space hardcoded across 50+ files (environment, networks, tests, frontend)
  - **4 critical bugs discovered:**
    - Bug #1: RecurrentSpatialQNetwork default `action_dim=5` (missing WAIT)
    - Bug #2: Test fixtures use `action_dim=5`
    - Bug #3: Hardcoded hygiene/satiation costs (0.003, 0.004)
    - Bug #4: L0.5 wait cost too high (0.0049 ≈ 98% of move cost)
  - Complete file inventory with line numbers
  - Dependency graph showing cascading impacts
- **Output:** Comprehensive research findings (see first subagent output)

### 2. Planning Subagent (COMPLETED ✅)

- **Task:** Develop detailed, bite-sized implementation plan
- **Output:** `docs/plans/2025-11-04-uac-action-space.md`
- **Structure:**
  - Phase 0: Pre-migration bug fixes (4 tasks)
  - Phase 1: Schema design and loader (3 tasks)
  - Phase 2: Environment integration (6 sub-tasks)
  - Phase 3: Config migration (3 tasks)
  - Phase 4: Frontend updates (1 task)
  - Phase 5: Cleanup and documentation (3 tasks)
  - Phase 6: Verification and testing (1 task)
- **Methodology:** TDD throughout (test first, implement, verify, commit)
- **Estimated Effort:** 2-3 days full-time work

### 3. Review Subagent (COMPLETED ✅)

- **Task:** Identify inconsistencies, gaps, and technical concerns
- **Findings:** Plan is **80% ready** but has critical gaps
- **Critical Issues Found:**
  - 🚨 **BLOCKER #1:** Research report not saved to file (missing documentation)
  - 🚨 **BLOCKER #2:** Bug #5 discovered - `population/vectorized.py` also has `action_dim=5` default
  - 🚨 **BLOCKER #3:** Performance optimization needed (avoid per-agent Python loops)
  - ⚠️ High Priority: Meter name hardcoding violates UAC
  - ⚠️ High Priority: Multiple interaction actions not handled
  - ⚠️ High Priority: Missing edge case tests
  - ⚠️ High Priority: Frontend changes underdocumented

---

## Current Plan Status

### Strengths ✅

- **Excellent structure:** 6 phases with clear dependencies
- **TDD methodology:** Test-first approach throughout
- **Backward compatibility:** Legacy fallback mode well-designed
- **Comprehensive scope:** Environment, networks, population, frontend, configs
- **Domain agnostic:** Supports villages, factories, trading bots, etc.
- **Detailed code snippets:** Makes implementation straightforward

### Critical Gaps 🚨 (BLOCKERS)

1. **Missing research report file** - First subagent's output not persisted
2. **Bug #5 not in plan** - `population/vectorized.py` also needs fixing
3. **Performance regression risk** - Plan uses per-agent loops instead of vectorized ops

### High-Priority Gaps ⚠️

4. **Meter name hardcoding** - Loads from bars.yaml needed, not hardcoded list
5. **Multiple interaction actions** - Only masks first one, should mask all
6. **Edge case tests missing** - Zero-cost, negative cost, empty costs, etc.
7. **Integration tests missing** - Multi-agent, dead agents, checkpoints
8. **Frontend underdocumented** - WebSocket protocol changes not detailed

---

## Recommendations

### Before Implementation Starts

**MUST FIX (Blockers):**

1. ✅ Save research report to `docs/research/2025-11-04-uac-action-space-research.md`
   - Include complete file inventory
   - Document all 5 bugs (including Bug #5)
   - Provide line numbers for all hardcoded references

2. 📝 Update plan Phase 0 to include Bug #5
   - Add Task 0.5: Fix population/vectorized.py default action_dim
   - Same pattern as Task 0.1 (networks.py fix)

3. 📝 Update plan Phase 2 for performance
   - Pre-build action lookup tables in `__init__`
   - Use vectorized tensor operations instead of Python loops
   - Add Phase 6.2: Performance benchmarking task

**SHOULD FIX (High Priority):**
4. 📝 Fix meter name hardcoding (Task 2.2.4)

- Load meter names from bars.yaml during `__init__`
- Build dynamic meter_name → index mapping
- Validate action costs reference valid meters

5. 📝 Handle multiple interaction actions (Task 2.3)
   - Mask ALL interaction-type actions, not just first
   - Update action masking logic

6. 📝 Add comprehensive test suites
   - Task 1.4: Edge case validation tests
   - Phase 2.6: Integration testing suite
   - Task 2.7: Backward compatibility tests

7. 📝 Expand frontend documentation (Task 4.1)
   - Detail WebSocket protocol changes
   - Add backward compatibility for replay files
   - Include automated E2E test plan

### Implementation Approach

**Recommended:** Use `superpowers:subagent-driven-development`

- Dispatch fresh subagent per task
- Code review between tasks (use `superpowers:code-reviewer`)
- Fast iteration with quality gates
- Stays in this session (no separate worktree needed)

**Alternative:** Use `superpowers:executing-plans` in separate session

- Batch execution with checkpoints
- Less oversight but faster throughput

---

## Next Steps

### Option 1: Iterate on Plan (Recommended)

1. Address all blockers (#1-3)
2. Address high-priority items (#4-8)
3. Update plan with revised tasks
4. Get user approval
5. Execute via subagent-driven-development

### Option 2: Execute Current Plan

1. Accept current plan with known gaps
2. Address blockers during implementation
3. Risk: May discover more issues mid-implementation

### Option 3: New Planning Session

1. Start fresh with lessons learned
2. Incorporate all review findings upfront
3. More time investment but cleaner result

---

## Files Generated

- ✅ `docs/plans/2025-11-04-uac-action-space.md` - Comprehensive implementation plan
- ⏳ `docs/research/2025-11-04-uac-action-space-research.md` - **NEEDS CREATION**
- ✅ `docs/tasks/TASK-003-PLANNING-STATUS.md` - This document

---

## Validation Checklist (Before Implementation)

Use this to verify plan is ready:

- [ ] Research report file exists with 50+ file inventory
- [ ] All 5 bugs documented (including population.py Bug #5)
- [ ] Performance optimization added to plan (vectorized ops, not loops)
- [ ] Phase 6.2 added: Performance benchmarking
- [ ] Task 0.5 added: Fix population/vectorized.py
- [ ] Meter name loading from bars.yaml (not hardcoded)
- [ ] Multiple interaction actions handled in masking
- [ ] Edge case tests added (Task 1.4)
- [ ] Integration tests added (Phase 2.6)
- [ ] Backward compatibility tests added (Task 2.7)
- [ ] Frontend documentation expanded (WebSocket, E2E tests)
- [ ] Boundary mode validation (restrict to "clamp" or implement all)

---

## Recommendation to User

**Verdict:** The planning process has successfully identified a comprehensive, well-structured implementation approach with **known gaps that can be addressed**.

**Suggested Action:**

1. Review this status document and the three subagent outputs
2. Decide whether to:
   - **Iterate:** Fix blockers and high-priority items, then execute
   - **Execute:** Accept current plan and address gaps during implementation
   - **Restart:** New planning session incorporating all findings
3. Choose execution method (subagent-driven vs parallel session)

The plan demonstrates solid software engineering practices and is **80% ready**. With blocker fixes, it will be **production-ready**.

---

**Status:** 🟨 AWAITING USER DECISION
