# Foundation Rebuild Plan

**Date**: November 1, 2025  
**Purpose**: Fix damage, break down god classes, create clean slate for future thinking

---

## 🎯 Mission

**NOT**: Prepare for Level 3-5 implementation (that's later)

**IS**:

- Get Level 2 POMDP actually working
- Eliminate god classes and antipatterns
- Remove 240+ lines of dead/duplicate code
- Create modular, maintainable, testable codebase
- Clear the mental space to think about future architecture

---

## 📊 Current Problems (from ARCHITECTURE.md deep dive)

### God Classes & Antipatterns

1. **vectorized_env.py: 1247 lines** - 8 responsibilities in one file
   - Meter dynamics (150 lines)
   - Reward calculation (280 lines)
   - Observation building (80 lines)
   - Cascade logic (150 lines)
   - Environment stepping
   - Action masking
   - Affordance handling (200-line elif blocks)
   - Agent state management

2. **Code Duplication** - Epsilon-greedy logic copied 3 times
   - exploration/epsilon_greedy.py (100 lines)
   - exploration/rnd.py (100 lines)
   - population/vectorized.py (47 lines)

3. **Dead Code** - 216 lines of disabled experiments
   - COMPLEX_DISABLED reward system (150 lines)
   - Proximity shaping rewards (66 lines)

### Critical Bugs

4. **Level 2 POMDP Broken**
   - Observation dimension mismatch (env produces 53-55, network expects 50)
   - LSTM trained on single transitions with zero hidden state
   - Can't progress to any future features until this works

5. **Hardcoded Configuration**
   - Meter cascades hardcoded (can't experiment)
   - Affordance effects in 200-line elif blocks (can't modify easily)
   - No way to A/B test design changes

---

## 🏗️ The Plan: 4 Phases, 12-14 Weeks

### Phase 1: Fix Critical Damage (2-4 weeks) 🔴

**Week 1: Quick Wins**

- **Day 1** (30 min): ACTION #13 - Delete 216 lines dead code
- **Day 1** (15 min): ACTION #11 - Remove legacy checkpoint methods
- **Days 2-5** (1 week): ACTION #7 - Build sequential replay buffer

**Weeks 2-4: Fix POMDP**

- ACTION #9 (3-4 weeks) - Network architecture redesign
  - Fix observation dimension mismatch
  - Implement proper sequential LSTM training
  - Add memory validation tests

**Outcome**: Level 2 POMDP works, 240 lines clutter removed

---

### Phase 2: Break Down God Classes (3-5 weeks) 🟡

**Goal**: vectorized_env.py 1247 → ~600 lines via extraction

**Week 5-6: Extract Strategies**

- ACTION #2 (3-5 days) - Extract RewardStrategy class
  - Separate milestone, complex, proximity reward systems
  - Enable A/B testing reward configurations
  - Remove 280 lines from environment

**Week 7-8: Extract Dynamics**

- ACTION #3 (1-2 weeks) - Extract MeterDynamics class
  - Centralize meter behavior
  - Enable testing in isolation
  - Remove 150 lines from environment

**Week 8-9: Extract Observations**

- ACTION #4 (2-3 days) - Extract ObservationBuilder class
  - Separate full vs partial observation logic
  - Clean interface for new observation modes
  - Remove 80 lines from environment

**Week 9-11: Extract Cascades**

- ACTION #1 (2-3 weeks) - Configurable cascade engine
  - Data-driven meter relationships (YAML config)
  - Enable pedagogical experiments
  - Remove 150 lines from environment

**Outcome**: Clear module boundaries, each subsystem testable independently

---

### Phase 3: Eliminate Technical Debt (1-2 weeks) 🟢

**Week 12: Code Quality**

- ACTION #10 (1-2 hours) - Deduplicate epsilon-greedy
  - Extract shared utility function
  - Remove 3 copies → 1 implementation
  
- ACTION #14 (3-5 days) - CI/CD pipeline
  - Ruff (linter/formatter)
  - Mypy (type checking)
  - Vulture (dead code detection)
  - Pre-commit hooks + GitHub Actions

**Week 13: Configuration System**

- ACTION #12 (1-2 weeks) - Config-defined affordances
  - Move 200-line elif blocks to YAML
  - Easy to add/modify affordances
  - Enables modding/experimentation

**Outcome**: Professional code quality, zero embarrassing patterns

---

### Phase 4: Polish & Validate (1-2 weeks) 🎨

**Week 14: Final Touches**

- ACTION #8 (1-2 days) - Add WAIT action
  - Fix forced-movement design flaw
  - Enable strategic "rest" behavior
  
- ACTION #5 (1-2 days) - Target network DQN
  - Improve training stability
  - Standard DQN improvement

**Week 15: Validation**

- Multi-day tech demo (Phase 3.5)
  - Run 10K episodes continuously
  - Prove system is stable
  - Validate learning curves
  - Generate teaching materials

**Optional**:

- ACTION #15 (1-2 weeks) - Unified demo server
  - Single command instead of 3 terminals
  - Better UX for demos
  - Can defer if time-constrained

**Outcome**: Solid foundation validated and ready

---

## 📈 Progress Metrics

### Before (Current State)

- vectorized_env.py: 1247 lines
- Dead code: 216 lines
- Code duplication: 3 copies epsilon-greedy
- Level 2 POMDP: Broken
- Configuration: Hardcoded
- Test coverage: 64%
- CI/CD: None

### After Phase 1

- Dead code: 0 lines ✅
- Level 2 POMDP: Working ✅
- LSTM: Learns with memory ✅
- Test coverage: 70%+ ✅

### After Phase 2

- vectorized_env.py: ~600 lines ✅
- Module boundaries: Clear ✅
- RewardStrategy: Extracted ✅
- MeterDynamics: Extracted ✅
- ObservationBuilder: Extracted ✅
- CascadeEngine: Extracted ✅

### After Phase 3

- Code duplication: 0 ✅
- CI/CD: Automated ✅
- Affordances: YAML-defined ✅
- Code quality: Professional ✅

### After Phase 4

- WAIT action: Implemented ✅
- Target network: Working ✅
- 10K episodes: Validated ✅
- Foundation: Solid ✅

---

## 🎯 Why This Solves the Problem

### Addresses God Classes

**Before**: vectorized_env.py does everything (1247 lines)  
**After**: 5 focused modules (each <300 lines)

- RewardStrategy - Calculate rewards
- MeterDynamics - Update meters
- ObservationBuilder - Build observations
- CascadeEngine - Meter relationships
- VectorizedHamletEnv - Orchestration only (~600 lines)

### Eliminates Duplication

**Before**: Epsilon-greedy logic in 3 places (247 lines total)  
**After**: Single shared utility (100 lines)

### Removes Dead Code

**Before**: 216 lines of disabled experiments  
**After**: 0 lines (removed with documentation of what failed)

### Enables Configuration

**Before**: Hardcoded cascades, affordances, penalties  
**After**: YAML configs - change without code edits

### Fixes Critical Bug

**Before**: POMDP broken (observation mismatch, LSTM not learning)  
**After**: Level 2 works (proper sequential training)

---

## 🚦 Critical Path

```
Phase 1 (Fix Damage)
    ↓
ACTION #13, #11 (quick wins)
    ↓
ACTION #7 (sequential buffer)
    ↓
ACTION #9 (fix POMDP) ← CRITICAL BLOCKER
    ↓
Phase 2 (Break God Classes)
    ↓
ACTION #2 → #3 → #4 → #1 (extractions in order)
    ↓
Phase 3 (Eliminate Debt)
    ↓
ACTION #10, #14, #12 (quality improvements)
    ↓
Phase 4 (Polish & Validate)
    ↓
ACTION #8, #5, Validation
    ↓
CLEAN SLATE FOR FUTURE THINKING
```

**Cannot skip phases** - Each depends on previous

---

## 🤔 What Happens After?

### Then (After Phase 4 Complete)

With clean foundation, we can think clearly about:

**Level 4 Architecture**:

- What should multi-agent competition look like?
- How do we model opponent beliefs?
- What's the right game theory abstraction?
- Fresh design, not bolted onto broken code

**Level 5 Architecture**:

- How should emergent communication work?
- What symbols make sense?
- How to ground language in experience?
- Clean slate thinking, not constrained by technical debt

**Research Questions**:

- Can we A/B test cascade designs? (Yes, YAML configs)
- Can we experiment with new reward systems? (Yes, RewardStrategy)
- Can students modify affordances easily? (Yes, YAML affordances)
- Can we profile and optimize? (Yes, modular code)

---

## 📋 Next Week Action Plan

### Monday

- **Morning** (30 min): ACTION #13 - Delete dead code
  - Remove lines 1000-1230 in vectorized_env.py
  - Run tests (all pass, coverage jumps)
  - Commit: "Remove 216 lines disabled reward systems"

- **Afternoon** (15 min): ACTION #11 - Remove legacy checkpoints
  - Delete unused checkpoint methods in curriculum
  - Run tests (all pass)
  - Commit: "Remove legacy checkpoint methods"

### Tuesday-Friday

- **Full time** (1 week): ACTION #7 - Sequential replay buffer
  - Create TrajectoryReplayBuffer class
  - Store episode sequences
  - Sample temporal sequences for training
  - Test with simple network first
  - Commit: "Add sequential replay buffer for LSTM training"

### By Friday EOD

- Infrastructure ready for ACTION #9 (POMDP fix)
- 240 lines dead code removed
- Clear head to tackle network redesign

---

## ⚠️ Principles

**Red-Green-Refactor**: Only refactor with tests green

**One Change at a Time**: Each action keeps tests passing

**No Feature Creep**: NOT preparing for Level 3-5, just cleaning house

**Measure Everything**: Before/after metrics for each phase

**Document Failures**: Keep disabled code as teaching examples (with explanation)

---

## ✅ Success Definition

**Foundation rebuild is complete when**:

1. ✅ Level 2 POMDP works (agent survives 150+ steps with memory)
2. ✅ vectorized_env.py <700 lines (from 1247)
3. ✅ Zero code duplication (single source of truth)
4. ✅ Zero dead code (216 lines removed)
5. ✅ Configuration-driven (cascades & affordances in YAML)
6. ✅ CI/CD automated (ruff, mypy, vulture, pytest)
7. ✅ Test coverage ≥70%
8. ✅ 10K episode validation successful
9. ✅ Clear module boundaries (each subsystem < 300 lines)
10. ✅ **Can think clearly about future architecture** ← MOST IMPORTANT

---

## 🎓 The Real Goal

Not just clean code. **Mental clarity.**

Right now:

- POMDP is broken (distracting)
- God classes are overwhelming (can't see structure)
- Dead code is confusing (what's real?)
- Duplication is embarrassing (what's canonical?)
- Hardcoded values are rigid (can't experiment)

After rebuild:

- POMDP works ✅ (one less thing to worry about)
- Clear modules ✅ (easy to understand)
- No clutter ✅ (focus on what matters)
- Single source of truth ✅ (confidence in code)
- Configuration-driven ✅ (easy to experiment)

**Result**: Clean mental space to design Level 4-5 architecture properly.

Not bolting features onto broken foundation.  
Building features on solid, clean, understandable base.
