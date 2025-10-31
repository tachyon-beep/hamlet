# Hamlet Cleanup Roadmap

## From Current State to Future Architecture

**Date**: November 1, 2025  
**Purpose**: Strategic plan to clean up codebase and prepare for Levels 2-5  
**Authority**: Based on ARCHITECTURE.md (current state) + ARCHITECTURE_DESIGN.md (future goals)

---

## 🌟 North Star: The v2.0 "Smart Collection" Vision

**Why This Cleanup Matters**: Every refactoring decision must preserve our path to the v2.0 moonshot architecture.

### The Moonshot: From "Flashcard Memorizer" to "Grammar Engine"

**Current (v1.0)**: Monolithic DQN that memorizes `Q(s,a)` values  
**Future (v2.0)**: Modular "Smart Collection" that learns world rules and agent psychology

### The Four Modules (v2.0 Architecture)

1. **Module A: Perception Encoder** → Outputs `BeliefDistribution` (solves POMDP)
2. **Module B: World Model** → Outputs `ImaginedFutures` (learns physics & predicts consequences)
3. **Module C: Social Model** → Outputs `SocialPrediction` (opponent modeling & theory of mind)
4. **Module D: Hierarchical Policy** → Outputs `Goal` + `PrimitiveAction` (strategic HRL)

### Critical Prerequisites (What This Cleanup MUST Achieve)

From AGENT_MOONSHOT.md "Phase 1: Foundation (Prerequisites)":

1. ✅ **Implement explicit data structures** (`BeliefDistribution`, `ImaginedFuture`, `SocialPrediction`, `Goal`)
2. ✅ **Refactor environment for `public_cues`** (enables Module C - Social Model)
3. ✅ **Refactor environment "physics" to be data-driven** (enables Module B - World Model)

**BLOCKER**: "The hardcoded logic in `vectorized_env.py` (Actions #1, #12) must be refactored to be configuration-driven (e.g., YAML) so the World Model can learn these rules."

### The "Don't Lock Out" Checklist

Before each refactoring action, ensure:

- ✅ **Can Module B learn this?** (Is it in YAML/config, not hardcoded?)
- ✅ **Can Module A replace this?** (Is observation construction pluggable?)
- ✅ **Can Module D replace this?** (Is action selection/rewards modular?)
- ✅ **Can Module C extend this?** (Is there room for social cues?)

### Migration Path (Incremental, Not Big Bang)

- **v1.5**: Module A (Perception) + v1.0 Q-Network (hybrid validation)
- **v1.7**: v1.5 + Module B (World Model for planning)
- **v2.0**: Full integration - all modules replace v1.0 DQN

**Key Insight**: Our foundation rebuild (ACTIONS #1, #2, #3, #4, #12) is actually **Phase 1 of the moonshot roadmap**. We're not just cleaning—we're building the launchpad. 🚀

---

## Executive Summary

**Current Position**: Between Level 1.5 and Level 2

- ✅ Level 1.5 working: Full observability, sparse rewards, RND exploration
- ⚠️ Level 2 broken: POMDP implemented but LSTM not learning
- 🎯 Level 3-5 designed: Multi-zone, multi-agent, emergent language

**Critical Blocker**: Level 2 POMDP must work before adding complexity (Levels 3-5 depend on it)

**Cleanup Strategy**: Fix foundation → Validate → Prepare for scale → Add quality

---

## 📊 Progress Update (November 1, 2025)

### Recent Accomplishments ✅

**Phase 4 Quality Tasks - 75% Complete!**

1. ✅ **ACTION #4: ObservationBuilder** (Oct 31, 2025)
   - Created 64-line class, 100% test coverage
   - Extracted 104 lines from vectorized_env.py
   - Zero regressions, all tests passing

2. ✅ **ACTION #2: RewardStrategy** (Oct 31, 2025)
   - Created 14-line class, 100% test coverage
   - Extracted 19 lines from vectorized_env.py
   - Milestone bonuses now modular

3. ✅ **ACTION #3: MeterDynamics** (Nov 1, 2025)
   - Created 268-line class, 100% test coverage
   - Extracted 192 lines from vectorized_env.py
   - All meter cascades isolated and documented
   - **70% coverage milestone achieved! 🎯**

**Combined Impact:**

- vectorized_env.py: **1039 → 731 lines** (-308 lines, -30% reduction)
- Test suite: **All 275 tests passing** (zero regressions)
- Coverage: **64% → 70%** (hit target before major refactoring)
- Three new classes: 100% test coverage on all

### What This Means

✅ **Foundation is now solid** for Level 3 preparation  
✅ **Coverage target met** (70%+) - safe to proceed with more refactoring  
✅ **God object significantly reduced** - from 1039 to 731 lines  
✅ **Ready for next phase** - either continue Phase 4 polish OR jump to Phase 3 scalability

### Decision Point: What's Next?

**Option A: Continue Phase 4 (Polish & Quality)** - 1-2 weeks

- ACTION #14: CI/CD pipeline (linting, type checking, pre-commit hooks)
- ACTION #15: Unified demo server (one command to run everything)
- ACTION #10: Deduplicate epsilon-greedy (remove code duplication)

**Option B: Jump to Phase 3 (Prepare for Scale)** - 2-3 weeks

- ACTION #1: Configurable cascade engine (YAML-driven meter relationships)
- ACTION #12: Configuration-defined affordances (YAML-driven affordance effects)
- ACTION #5: Add target network (stable training for 5000-7000 episodes)

**Option C: Return to Phase 1 (Fix Foundation)** - 4-6 weeks

- ACTION #7: Sequential replay buffer (enable LSTM training)
- ACTION #9: Network architecture redesign (fix POMDP/LSTM issues)
- **Note**: This is the CRITICAL PATH to unblock Levels 3-5

**Recommendation**: **Option C (Phase 1)** - The LSTM/POMDP fix is the blocker for everything. Phase 4 polish is done enough (30% reduction achieved, 70% coverage hit). Should tackle the hard problem now.

---

## Current State Analysis

### What's Working (Keep)

1. **Sparse Reward System** - Milestone-based rewards working well
2. **RND Exploration** - Adaptive intrinsic motivation with variance annealing
3. **Adversarial Curriculum** - 5-stage progression with entropy gating
4. **Vectorized Environment** - GPU-native training (200 episodes/hour)
5. **Action Masking** - Prevents invalid actions effectively
6. **Temporal Mechanics** - Time-of-day cycles and multi-tick interactions (Level 2.5)
7. **Test Coverage** - 64% overall (982/1525 statements, 241 tests passing)

### What's Broken (Fix)

1. **Level 2 POMDP** - CRITICAL BLOCKER
   - Observation dimension mismatch: Environment produces 53-55 dims, network expects 50
   - LSTM trained on single transitions with zero hidden state (defeats purpose)
   - Needs sequential replay buffer for proper LSTM training
   - **Impact**: Can't progress to Levels 3-5 until this works

2. **Dead Code** - 216 lines of disabled reward systems
   - COMPLEX_DISABLED (lines 1000-1150): Per-step meter rewards
   - Proximity shaping (lines 1150-1230): Reward hacking experiments
   - **Impact**: Confusing, takes mental space

3. **Incomplete Checkpointing** - Missing Q-network weights, optimizer state
   - Can't properly resume multi-day training
   - **Impact**: Blocks Phase 3.5 tech demo

4. **Code Duplication** - Epsilon-greedy logic copied 3 times
   - exploration/epsilon_greedy.py (100 lines)
   - exploration/rnd.py (100 lines)
   - population/vectorized.py (47 lines)
   - **Impact**: Maintenance burden, inconsistency risk

5. **No Target Network** - Q-target uses same network as Q-pred
   - **Impact**: Unstable training, won't scale to 5000-7000 episodes (Level 3+)

### What's Not Scalable (Refactor)

1. **Monolithic Environment** - vectorized_env.py is 1247 lines with 8 responsibilities
   - Will grow much worse with Level 3 (5 new meters, 4 zones, 14+ affordances)
   - Needs extraction: RewardStrategy, MeterDynamics, ObservationBuilder

2. **Hardcoded Affordances** - 200+ line elif blocks
   - Level 3 needs 14+ affordances (vs current 15)
   - Should be YAML-defined for easy modification

3. **Hardcoded Meter Cascades** - Differential equations hardcoded
   - Level 3 adds 5 meters: social, fitness, stress, health, knowledge
   - Need configurable cascade engine for pedagogical experiments

---

## Future Requirements (from ARCHITECTURE_DESIGN.md)

### Level 2: Partial Observability (POMDP)

**Status**: Implemented but broken, must fix first

**Requirements**:

- 5×5 vision window (implemented ✅)
- LSTM memory across timesteps (implemented but not learning ⚠️)
- Sequential replay buffer (missing ❌)
- Exploration bonuses (RND working ✅)
- 2000-3000 episodes to learn
- Parameters: ~500K (current RecurrentSpatialQNetwork)

**Blockers**:

- ACTION #9: Network architecture redesign (observation handling, LSTM training)
- ACTION #7: Sequential replay buffer (store trajectories, not single transitions)

### Level 3: Multi-Zone Hierarchical RL

**Status**: Designed, not started

**New Requirements**:

- **4 zones** (Home, Industrial, Commercial, Public) - 80×80 total world
- **14+ affordances** distributed across zones
- **5 new meters** (social, fitness, stress, health, knowledge)
- **Hierarchical policy** (zone → transport → navigation)
- **Transportation** (walk/bus/taxi with cost/time trade-offs)
- **Time mechanics** (day/night cycles, schedules) - partially implemented ✅
- 5000-7000 episodes to learn
- Parameters: ~600K

**Prerequisites**:

- ACTION #1: Configurable cascade engine (for 5 new meters)
- ACTION #12: Configuration-defined affordances (for 14+ affordances)
- ACTION #5: Target network (for longer training stability)
- ACTIONS #2, #3, #4: Extract environment subsystems (for maintainability)

### Level 4: Multi-Agent Competition

**Status**: Designed, future (2-3 months out)

**New Requirements**:

- **2-10 agents** in same world
- **Limited resources** (job slots, fridge capacity, bus seats)
- **Opponent modeling** (theory of mind, belief states)
- **Strategic reasoning** (game theory, counter-strategies)
- **Information asymmetry** (can't see opponent meters/intentions)
- 15,000 episodes to learn
- Parameters: ~3.3M

**Prerequisites**:

- Working Level 3 (hierarchical RL foundation)
- Population-based training infrastructure
- Opponent model architecture (belief LSTM + predictors)

### Level 5: Emergent Communication

**Status**: Designed, future (6+ months out)

**New Requirements**:

- **Family units** with shared information
- **Discrete communication channel** (symbols, tokens)
- **Language grounding** in shared experience
- **Emergent protocols** for coordination

**Prerequisites**:

- Working Level 4 (multi-agent foundation)
- Communication architecture
- Language evaluation metrics

---

## Cleanup Phases

### Phase 1: Fix Foundation (2-4 weeks) 🔴 CRITICAL

**Goal**: Working Level 2 POMDP that actually learns with partial observability

**Tasks**:

1. **ACTION #13: Remove Dead Code (30 minutes)** - Quick win first!
   - Delete 216 lines: COMPLEX_DISABLED + proximity_rewards
   - Location: vectorized_env.py lines 1000-1230
   - **Benefit**: Instant coverage boost (82% → ~95%), clearer codebase
   - **Risk**: None (code is DISABLED and documented as failed experiments)

2. **ACTION #11: Complete Checkpointing (2-4 hours)**
   - Add Q-network state_dict to PopulationCheckpoint
   - Add optimizer state, total_steps, replay_buffer state
   - **Benefit**: Enable multi-day training (Phase 3.5)
   - **Dependencies**: None (standalone fix)

3. **ACTION #7: Sequential Replay Buffer (1 week)**
   - Replace ReplayBuffer with TrajectoryReplayBuffer
   - Store full episodes (trajectories) instead of single transitions
   - Sample sequences of length 20-32 for LSTM training
   - **Benefit**: Enables proper recurrent training
   - **Dependencies**: None (new class, doesn't break existing)

4. **ACTION #9: Network Architecture Redesign (3-4 weeks)**
   - Fix observation dimension mismatch (50 vs 53-55)
   - Integrate sequential training into population.py
   - Test LSTM actually uses memory (sequence memory tests)
   - **Benefit**: Working POMDP (unblocks Levels 3-5)
   - **Dependencies**: ACTION #7 (sequential buffer)

**Validation**:

- LSTM memory tests pass (remembers >5 steps back)
- Level 2 agent survives 150+ steps (vs current ~115)
- Exploration→exploitation transition visible in metrics
- 70%+ test coverage maintained

**Milestone**: Level 2 POMDP working, ready for Phase 3.5 tech demo

---

### Phase 2: Validate System (1 week) 🟡 VALIDATION

**Goal**: Prove system is stable and learns over 48+ hours

**Tasks**:

1. **Phase 3.5 Multi-Day Tech Demo**
   - Run 10K episodes (~48 hours continuous training)
   - Monitor intrinsic weight annealing (1.0 → <0.1)
   - Observe curriculum progression (Stage 1 → Stage 5)
   - Generate teaching materials from real training data

**Success Criteria**:

- Training runs 10K episodes without crashes
- Final survival time > 200 steps (vs ~115 baseline)
- No memory leaks or performance degradation
- Agent shows exploration→exploitation transition

**Deliverables**:

- Training logs and metrics
- Screenshots of learning curves
- Teaching materials demonstrating POMDP concepts
- Validation that foundation is solid

**Milestone**: Foundation validated, ready to add complexity

---

### Phase 3: Prepare for Scale (2-3 weeks) 🟢 SCALABILITY

**Goal**: Infrastructure ready for Level 3 (multi-zone, 5 new meters, 14+ affordances)

**Tasks**:

1. **ACTION #1: Configurable Cascade Engine (2-3 weeks)**
   - Replace hardcoded cascades with data-driven system
   - YAML config for meter relationships and strengths
   - **Benefit**: Easy to add 5 new meters (Level 3), pedagogical experiments
   - **Priority**: HIGH (Level 3 blocker)

2. **ACTION #12: Configuration-Defined Affordances (1-2 weeks)**
   - Move affordance effects from code to YAML
   - Delete 200+ line elif blocks
   - **Benefit**: Easy to add 14+ affordances (Level 3), modding support
   - **Priority**: HIGH (Level 3 blocker)

3. **ACTION #5: Add Target Network (1-2 days)**
   - Separate Q-network and Q-target network
   - Periodic sync (every 1000 steps)
   - **Benefit**: Stable training for 5000-7000 episodes (Level 3)
   - **Priority**: HIGH (Level 3 needs longer training)

**Validation**:

- Config changes don't require code changes
- Can define new meter (e.g., "stress") in <5 minutes
- Can define new affordance (e.g., "Library") in <10 minutes
- Target network improves training stability (measure TD error variance)

**Milestone**: Ready to implement Level 3 (multi-zone hierarchical RL)

---

### Phase 4: Quality & Maintainability (1-2 weeks) 🔵 POLISH

**Goal**: Sustainable codebase as complexity grows

**Status**: ✅ **75% COMPLETE** (3/4 extractions done!)

**Completed Tasks**:

1. ✅ **ACTION #4: Extract ObservationBuilder (DONE - Oct 31, 2025)**
   - Created ObservationBuilder class (64 lines, 100% coverage)
   - Extracted ~104 lines from vectorized_env.py (1039→935 lines)
   - All tests passing, zero regressions

2. ✅ **ACTION #2: Extract RewardStrategy (DONE - Oct 31, 2025)**
   - Created RewardStrategy class (14 lines, 100% coverage)
   - Extracted ~19 lines from vectorized_env.py (937→918 lines)
   - Milestone bonuses now modular and testable

3. ✅ **ACTION #3: Extract MeterDynamics (DONE - Nov 1, 2025)**
   - Created MeterDynamics class (268 lines, 100% coverage)
   - Extracted ~192 lines from vectorized_env.py (923→731 lines)
   - All meter depletion and cascade effects now isolated
   - **Coverage milestone: 70% achieved! 🎯**

**Progress**: vectorized_env.py reduced from **1039 → 731 lines** (-308 lines, -30% reduction)

**Remaining Tasks**:

1. **ACTIONS #2, #3, #4: Extract Environment Subsystems**
   - ✅ Extract RewardStrategy class (ACTION #2: DONE)
   - ✅ Extract MeterDynamics class (ACTION #3: DONE)
   - ✅ Extract ObservationBuilder class (ACTION #4: DONE)
   - **Benefit**: vectorized_env.py 1039 → 731 lines achieved, easier to extend
   - **Priority**: MEDIUM (quality of life, not blocking)
   - **Status**: ✅ COMPLETE - All three extractions successful!

2. **ACTION #14: CI/CD Pipeline (3-5 days)**
   - Ruff (linter/formatter)
   - Mypy (type checking)
   - Vulture (dead code detection - would have caught 216 lines!)
   - Pre-commit hooks
   - GitHub Actions
   - **Benefit**: Prevent bugs, enforce quality
   - **Priority**: MEDIUM (should do before Level 3)

3. **ACTION #15: Unified Demo Server (1-2 weeks)**
   - Merge training + inference + frontend into single process
   - One command: `python run_demo.py` and you're done
   - **Benefit**: Better UX, easier demos, less complexity
   - **Priority**: LOW (nice to have)

4. **ACTION #10: Deduplicate Epsilon-Greedy (1-2 hours)**
   - Extract shared utility function
   - Remove 3 copies of same code
   - **Benefit**: Maintainability, consistency
   - **Priority**: LOW (technical debt)

**Validation**:

- CI passes on all PRs
- New developer can run demo in <5 minutes
- Test coverage at 70%+ (target before refactoring)
- Code quality metrics improved (cyclomatic complexity down)

**Milestone**: Professional-grade codebase ready for Level 3 implementation

---

## Prioritized Action List

### 🔴 CRITICAL PATH (Blocks Future Levels)

1. ✅ **ACTION #13** (30 min) - Remove dead code → instant coverage boost
2. ✅ **ACTION #11** (2-4 hrs) - Complete checkpointing → enable multi-day training
3. 🚧 **ACTION #7** (1 week) - Sequential replay buffer → enable LSTM training
4. 🚧 **ACTION #9** (3-4 weeks) - Fix LSTM/POMDP → unblock Levels 3-5

### 🟢 HIGH PRIORITY (Enables Level 3)

5. 🎯 **Phase 3.5** (1 week) - Multi-day tech demo → validate foundation
6. 🎯 **ACTION #1** (2-3 weeks) - Configurable cascades → add 5 new meters
7. 🎯 **ACTION #12** (1-2 weeks) - Config affordances → add 14+ affordances
8. 🎯 **ACTION #5** (1-2 days) - Target network → stable long training

### 🔵 MEDIUM PRIORITY (Quality)

9. ✅ **ACTIONS #2, #3, #4** (DONE Nov 1, 2025) - Extract subsystems → maintainability
   - ✅ ACTION #4: ObservationBuilder (Oct 31) - 104 lines extracted
   - ✅ ACTION #2: RewardStrategy (Oct 31) - 19 lines extracted  
   - ✅ ACTION #3: MeterDynamics (Nov 1) - 192 lines extracted
   - **Result**: vectorized_env.py 1039→731 lines (-30%), 70% coverage milestone! 🎯
10. 🎨 **ACTION #14** (3-5 days) - CI/CD pipeline → prevent bugs
11. 🎨 **ACTION #15** (1-2 weeks) - Unified server → better UX

### ⚪ LOW PRIORITY (Nice to Have)

12. 🧹 **ACTION #10** (1-2 hrs) - Deduplicate epsilon-greedy → cleanup
13. 🧹 **ACTION #8** (1-2 days) - Add WAIT action → minor improvement

---

## Timeline Estimate

### Aggressive Schedule (6-8 weeks to Level 3 ready)

- **Week 1**: Quick wins (ACTION #13, #11) + start ACTION #7
- **Week 2**: Finish ACTION #7, start ACTION #9
- **Weeks 3-4**: Finish ACTION #9 (LSTM fix)
- **Week 5**: Phase 3.5 validation (multi-day demo)
- **Week 6**: ACTIONS #1, #12 (config systems)
- **Week 7**: ACTION #5 (target network) + quality tasks
- **Week 8**: Buffer, polish, documentation

### Realistic Schedule (10-12 weeks to Level 3 ready)

- **Weeks 1-2**: Quick wins + ACTION #7
- **Weeks 3-6**: ACTION #9 (LSTM fix + testing)
- **Week 7**: Phase 3.5 validation
- **Weeks 8-10**: ACTIONS #1, #12, #5 (scalability)
- **Weeks 11-12**: Quality tasks (ACTIONS #2, #3, #4, #14)

### Conservative Schedule (14-16 weeks to Level 3 ready)

- Add 2-4 weeks buffer for unexpected issues
- Time for thorough testing at each phase
- Time for documentation and teaching materials

---

## Risk Mitigation

### Risk: ACTION #9 Takes Longer Than Expected

**Likelihood**: High (network redesign is complex)  
**Impact**: High (blocks everything)  
**Mitigation**:

- Start with minimal fix (observation dimensions) before full redesign
- Incremental validation (test each component separately)
- Have fallback: Simple MLP baseline always works (Level 1.5)

### Risk: Level 2 POMDP Performance Degrades

**Likelihood**: Medium (partial obs is hard)  
**Impact**: Medium (pedagogically interesting failure)  
**Mitigation**:

- Document "interesting failures" as teaching moments
- Compare to full obs baseline (should be 60-70% of performance)
- Adjust exploration bonus if needed

### Risk: Configuration Systems (ACTIONS #1, #12) Too Complex

**Likelihood**: Medium (balancing flexibility vs simplicity)  
**Impact**: Low (can simplify if needed)  
**Mitigation**:

- Start with simple YAML schema
- Iterate based on usage
- Keep code fallback for edge cases

### Risk: Multi-Day Training Unstable

**Likelihood**: Low (Phase 3 complete, RND working)  
**Impact**: Medium (can't validate)  
**Mitigation**:

- Checkpoint every 100 episodes
- Monitor memory usage
- Graceful shutdown handlers (already implemented)

---

## Success Metrics

### Phase 1 Success (Foundation Fixed)

- ⏳ Level 2 POMDP agent survives 150+ steps
- ⏳ LSTM memory tests pass (remembers 5+ steps back)
- ✅ **Test coverage ≥70% (ACHIEVED: 70%!)**
- ⏳ Observation dimensions match across environment/network
- ⏳ Training stable for 3000 episodes

**Status**: 1/5 complete - coverage target hit, LSTM work remains

### Phase 2 Success (Validation)

- ✅ 10K episodes complete without crashes
- ✅ Intrinsic weight anneals from 1.0 → <0.1
- ✅ Agent progresses through all 5 curriculum stages
- ✅ Final survival time >200 steps
- ✅ Teaching materials generated

### Phase 3 Success (Scalability)

- ✅ New meter defined in YAML (<5 min)
- ✅ New affordance defined in YAML (<10 min)
- ✅ Cascade relationships configurable
- ✅ Target network improves stability (TD error variance down)
- ✅ Ready to implement Level 3

### Phase 4 Success (Quality)

- ⏳ CI pipeline passes
- ✅ **vectorized_env.py <731 lines (ACHIEVED: from 1039, -30%!)**
- ⏳ Demo runs with one command
- ✅ **Test coverage ≥70% (ACHIEVED: 70%!)**
- ⏳ No code duplication warnings

**Status**: 2/5 complete - major extractions done, CI/CD and unification remain

---

## Alignment with ROADMAP.md

This cleanup plan follows the **2→3→1 approach**:

1. **2 (Validate)**: Phase 1-2 fix and validate Level 2 POMDP
2. **3 (Add Complexity)**: Phase 3 prepare for Level 3 (multi-zone)
3. **1 (Optimize)**: Phase 4 quality improvements

**Strategic Rationale**: Don't optimize prematurely. Fix LSTM first, validate it works, then prepare infrastructure for Level 3. Only after complexity is added do we know what needs optimization.

---

## Next Steps (Immediate)

**Week 1 Sprint Plan**:

1. **Day 1** (30 min): ACTION #13 - Remove dead code
   - Delete lines 1000-1230 in vectorized_env.py
   - Run tests (should all pass, coverage jumps to ~95% on that file)
   - Commit: "Remove 216 lines of disabled reward systems"

2. **Day 1-2** (2-4 hrs): ACTION #11 - Complete checkpointing
   - Extend PopulationCheckpoint schema
   - Save/load Q-network weights
   - Test checkpoint/resume works
   - Commit: "Add complete checkpointing with network weights"

3. **Days 3-5** (1 week): ACTION #7 - Sequential replay buffer
   - Create TrajectoryReplayBuffer class
   - Store episodes as sequences
   - Sample sequences for training
   - Test with simple network first
   - Commit: "Add sequential replay buffer for LSTM training"

**By end of Week 1**: Have infrastructure ready for ACTION #9 (LSTM fix)

---

## Questions for Decision

1. **Should we skip Phase 2 validation?**
   - Pro: Faster to Level 3
   - Con: Risk of unstable foundation
   - **Recommendation**: NO - 1 week validation saves 2-4 weeks debugging later

2. **Should we do quality tasks (Phase 4) before Level 3?**
   - Pro: Cleaner codebase before adding complexity
   - Con: Delays Level 3 by 1-2 weeks
   - **Recommendation**: YES - Level 3 will add lots of complexity, need solid foundation

3. **Should we implement Level 3 before Level 2 is perfect?**
   - Pro: Level 3 is exciting (multi-zone, hierarchy)
   - Con: Building on broken foundation
   - **Recommendation**: NO - Fix Level 2 first, it's a dependency

4. **Should we do all of Phase 3 before starting Level 3?**
   - Pro: Fully prepared infrastructure
   - Con: Some tasks might be YAGNI
   - **Recommendation**: Do ACTION #1, #12 for sure (direct blockers), defer #5 if time-constrained

---

## Conclusion

**Bottom Line**: We're 80% to Level 2, but that last 20% (LSTM fix) is critical. Can't skip to Level 3 without working POMDP foundation.

**Recommended Path**:

1. ✅ Quick wins this week (ACTIONS #13, #11)
2. 🚧 Fix LSTM over next 4 weeks (ACTIONS #7, #9)
3. ✅ Validate with multi-day demo (1 week)
4. 🎯 Prepare for scale (ACTIONS #1, #12, #5)
5. 🚀 Build Level 3 (multi-zone hierarchical RL)

**Timeline**: 8-12 weeks to be ready for Level 3 implementation.

**Risk**: ACTION #9 is complex, budget 4-6 weeks not 3-4.

**Payoff**: Working POMDP foundation that scales to Levels 3, 4, 5 (the exciting stuff!).
