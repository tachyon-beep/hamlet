# Cleanup Strategy Summary

**Date**: November 1, 2025

## TL;DR

**Current Problem**: Level 2 POMDP is implemented but LSTM doesn't learn (observation mismatch + single-transition training)

**Why It Matters**: Can't build Levels 3-5 (multi-zone, multi-agent, emergent language) on broken foundation

**Solution**: Fix LSTM (4 weeks) → Validate (1 week) → Prepare for scale (3 weeks) → Add quality (2 weeks)

**Timeline**: 8-12 weeks to be ready for Level 3 implementation

---

## Critical Path (Must Do)

### Week 1: Quick Wins

- ✅ **Remove 216 lines dead code** (30 min) - Instant coverage boost
- ✅ **Complete checkpointing** (2-4 hrs) - Enable multi-day training

### Weeks 2-5: Fix LSTM

- 🚧 **Sequential replay buffer** (1 week) - Store trajectories not transitions
- 🚧 **Network architecture redesign** (3-4 weeks) - Fix observation dims, LSTM training

### Week 6: Validate

- ✅ **Multi-day tech demo** (1 week) - 10K episodes, prove it works

### Weeks 7-9: Prepare for Level 3

- 🎯 **Configurable cascades** (2-3 weeks) - Add 5 new meters easily
- 🎯 **Config-defined affordances** (1-2 weeks) - Add 14+ affordances easily  
- 🎯 **Target network** (1-2 days) - Stable training for 5K+ episodes

### Weeks 10-12: Quality (Optional but Recommended)

- 🎨 **Extract environment subsystems** (1-2 weeks) - Maintainability
- 🎨 **CI/CD pipeline** (3-5 days) - Prevent bugs
- 🎨 **Unified demo server** (1-2 weeks) - Better UX

---

## Why This Order?

**Phase 1 (Fix Foundation)**: Level 2 POMDP must work before Levels 3-5

- Level 3 needs POMDP (partial obs in zones)
- Level 4 needs POMDP (can't see opponent meters)
- Level 5 needs POMDP (communication happens through observation)

**Phase 2 (Validate)**: Don't build on untested foundation

- 1 week validation saves 2-4 weeks debugging later
- Generates teaching materials from real data

**Phase 3 (Prepare for Scale)**: Level 3 needs infrastructure

- 5 new meters (need configurable cascades)
- 14+ affordances (need YAML configs)
- 5000-7000 episodes (need target network stability)

**Phase 4 (Quality)**: Add complexity to clean codebase

- Level 3 adds 4 zones, hierarchical policy, transportation
- Better to refactor before adding complexity
- CI/CD prevents bugs as team grows

---

## What Each Level Needs

### Level 1.5 (Current - Working ✅)

- Full observability
- Sparse rewards
- RND exploration
- 1000 episodes
- ~26K parameters

### Level 2 (Broken - Must Fix 🔴)

- Partial observability (5×5 window)
- LSTM memory
- Exploration bonuses
- 2000-3000 episodes
- ~500K parameters
- **Blockers**: ACTION #7 (sequential buffer), ACTION #9 (LSTM fix)

### Level 3 (Future - Needs Infrastructure 🟢)

- 4 zones (80×80 total)
- 14+ affordances
- 5 new meters
- Hierarchical policy
- Transportation
- 5000-7000 episodes
- ~600K parameters
- **Needs**: ACTION #1 (cascades), ACTION #12 (affordances), ACTION #5 (target network)

### Level 4 (Future - 2-3 months 🔵)

- 2-10 agents
- Opponent modeling
- Game theory
- Theory of mind
- 15,000 episodes
- ~3.3M parameters
- **Needs**: Working Level 3

### Level 5 (Future - 6+ months ⚪)

- Family units
- Emergent communication
- Language grounding
- **Needs**: Working Level 4

---

## Risk Assessment

### High Risk

**ACTION #9 (LSTM fix)** - Complex, blocks everything

- **Mitigation**: Incremental approach, test each component
- **Fallback**: Simple MLP works (Level 1.5)
- **Budget**: 4-6 weeks not 3-4

### Medium Risk  

**Level 2 performance degrades** - Partial obs is hard

- **Mitigation**: Document as teaching moment
- **Expectation**: 60-70% of full obs performance

### Low Risk

**Multi-day training unstable** - Phase 3 working well

- **Mitigation**: Checkpoint every 100 episodes
- **Monitoring**: Memory usage, graceful shutdown

---

## Success Criteria

### Phase 1 Success

- Level 2 agent survives 150+ steps (vs ~115 now)
- LSTM remembers 5+ steps back
- Test coverage ≥70%
- Training stable for 3000 episodes

### Phase 2 Success  

- 10K episodes complete
- Intrinsic weight anneals 1.0 → <0.1
- Final survival >200 steps
- Teaching materials generated

### Phase 3 Success

- New meter in <5 min (YAML)
- New affordance in <10 min (YAML)
- Target network improves stability
- Ready for Level 3

### Phase 4 Success

- CI passes
- vectorized_env.py <700 lines (from 1247)
- One-command demo
- Test coverage ≥70%

---

## Decision Points

### Should we skip validation (Phase 2)?

**NO** - 1 week validation saves 2-4 weeks debugging later

### Should we do quality before Level 3?

**YES** - Level 3 adds lots of complexity, need solid foundation

### Should we implement Level 3 before Level 2 is perfect?

**NO** - Fix Level 2 first, it's a dependency

### Should we do all of Phase 3 before starting Level 3?

**MOSTLY** - Do ACTION #1, #12 for sure (direct blockers), can defer #5 if time-constrained

---

## Next Week Sprint

### Day 1 (30 min)

ACTION #13: Remove 216 lines dead code

- Instant coverage boost (82% → ~95%)
- Clearer codebase

### Day 1-2 (2-4 hrs)

ACTION #11: Complete checkpointing  

- Save Q-network weights
- Enable multi-day training

### Days 3-5 (1 week)

ACTION #7: Sequential replay buffer

- Store trajectories not transitions
- Enable LSTM training

**By end of Week 1**: Ready to start ACTION #9 (LSTM fix)

---

## Bottom Line

**We're 80% to Level 2, but that last 20% (LSTM) is critical.**

Can't skip to Level 3 without working POMDP - it's a foundation for:

- Level 3: Partial obs in zones
- Level 4: Can't see opponents  
- Level 5: Communication through observation

**Recommended**: 4 weeks fixing, 1 week validating, 3 weeks preparing = 8 weeks to Level 3 ready

**Risk**: Budget 10-12 weeks to be safe (ACTION #9 is complex)

**Payoff**: Working POMDP foundation that scales to the exciting stuff (multi-agent, game theory, emergent language)!
