# Townlet Development Roadmap

> **Last Updated:** 2025-10-30
>
> **Current Status:** Phase 3 Complete ✅

## Strategic Direction: 2 → 3 → 1

After completing Phase 3 (Intrinsic Exploration), we've adopted a strategic sequence to avoid premature optimization:

1. **Multi-Day Demo** (Phase 3.5) - Validate system works in production
2. **POMDP Extension** (Phase 4) - Add complexity (partial observability + LSTM)
3. **Optimization** (Phase 5) - Profile and optimize the complete system

**Rationale:** Avoid the 1→3→1 trap where we optimize for a system that doesn't exist yet. Instead, add POMDP first, then optimize based on real profiling data from the complete system.

---

## Phase 3: Intrinsic Exploration ✅ COMPLETE

**Status:** All 8 tasks complete, 37/37 tests passing, pushed to origin/main

**Delivered:**
- ReplayBuffer with dual reward storage (extrinsic + intrinsic)
- RND (Random Network Distillation) for novelty detection
- AdaptiveIntrinsicExploration with variance-based annealing
- VectorizedPopulation integration with Q-learning + RND training
- 4 Vue visualization components (novelty heatmap, reward charts, curriculum tracker, survival trends)
- End-to-end sparse learning tests
- Comprehensive documentation

**Key Achievement:** Agents can now learn in sparse reward environments using intrinsic motivation that automatically transitions to exploitation.

**Test Results:**
- Unit tests: 13/13 passing
- Integration tests: 1/1 passing
- End-to-end: 1/1 passing (baseline comparison shows adaptive intrinsic > epsilon-greedy)

**Known Limitations (documented, not blocking):**
- No target network (simplified DQN)
- RND buffer uses CPU transfers (performance trade-off)
- Hardcoded batch size and training frequency

---

## Phase 3.5: Multi-Day Tech Demo 🎯 NEXT

**Goal:** Validate Phase 3 system works in production over extended runtime (days) and generate teaching materials.

**Why This First:**
- Validates the "watchers get more invested over time" hypothesis
- Reveals any stability issues (memory leaks, NaN losses, crashes) before adding POMDP complexity
- Generates real data on exploration→exploitation transition for teaching
- Uses existing code - low risk, high value

**Approach:**
- Run 10K episode training (full `test_sparse_learning_with_intrinsic` test)
- Stream live visualization via WebSocket (port 8765)
- Monitor metrics over hours/days: survival trends, intrinsic weight decay, curriculum progression
- Capture screenshots and data for teaching materials

**Success Criteria:**
- [ ] Training runs for 48+ hours without crashes
- [ ] Intrinsic weight anneals from 1.0 → <0.1
- [ ] Agent progresses through all 5 curriculum stages
- [ ] Final survival time > 200 steps (vs ~115 baseline)
- [ ] Visualization components update in real-time
- [ ] No memory leaks or performance degradation

**Deliverables:**
- Training logs and metrics (JSON or CSV)
- Screenshots showing exploration→exploitation transition
- Performance report (survival over time, reward components, curriculum progression)
- Optional: Teaching guide using the demo data

**Estimated Duration:** 2-3 days runtime + 1 day setup/monitoring

---

## Phase 4: POMDP Extension (Level 2) 📋 PLANNED

**Goal:** Add partial observability and working memory to create more realistic, pedagogically valuable scenarios.

**Why After Multi-Day Demo:**
- Multi-day demo validates Phase 3 foundation is solid
- POMDP will stress the system - better to know current system works first
- Adding complexity before optimization lets us optimize based on real needs

**Core Changes:**
- **Partial Observability:** Agents see limited field of view (e.g., 3×3 window around position)
- **LSTM Memory:** Temporal state integration for sequential reasoning
- **Modified Observation:** Add "fog of war" - unknown cells have null/masked values
- **New Reward Shaping:** Encourage exploration to discover hidden affordances

**Architecture:**
- Modify `VectorizedHamletEnv` to provide partial observations
- Update Q-networks to include LSTM layer
- Extend `BatchedAgentState` with hidden state management
- Modify RND to work with partial observations (or full state for novelty)

**Teaching Value:**
- Demonstrates working memory requirements
- Shows how partial observability affects exploration
- Creates "interesting failures" when agents forget affordance locations
- Introduces recurrent architectures

**Deliverables:**
- Partial observability implementation
- LSTM-based Q-networks
- Tests validating hidden state propagation
- Documentation of POMDP-specific behaviors

**Estimated Duration:** 5-7 days implementation

**Note:** Don't optimize during this phase. Accept "good enough" performance. Focus on getting POMDP working correctly. Document what's slow or unstable for Phase 5.

---

## Phase 5: Informed Optimization 📋 PLANNED

**Goal:** Profile the complete system (Phase 3 + POMDP) and optimize based on real bottlenecks.

**Why Last:**
- Avoids premature optimization (the 1→3→1 trap)
- Optimization targets are informed by POMDP requirements
- We know which issues actually matter vs. theoretical concerns
- One clean optimization pass instead of multiple revisions

**Optimization Candidates (from Phase 3 code reviews):**
- **Target Network:** Add to DQN for training stability (especially important with LSTM?)
- **Double DQN:** Reduce overestimation bias
- **Dueling Architecture:** Separate value/advantage streams
- **GPU Optimization:** Fix RND CPU transfers, batch LSTM operations
- **Gradient Clipping:** Tune for LSTM gradient flow
- **Buffer Management:** Optimize replay buffer for LSTM sequences
- **Hyperparameter Tuning:** Learning rates, batch sizes, update frequencies

**Process:**
1. **Profile:** Use cProfile, PyTorch profiler, memory profiler on POMDP system
2. **Identify Bottlenecks:** Find actual slow points (not guesses)
3. **Prioritize:** Fix issues by impact (biggest speedup first)
4. **Validate:** Ensure optimizations don't break POMDP correctness
5. **Document:** Record performance improvements

**Success Criteria:**
- [ ] Training speed: 2-5x faster (target: 100 episodes/hour → 200-500 episodes/hour)
- [ ] Memory stable over 10K episodes (no leaks)
- [ ] No NaN losses during LSTM training
- [ ] All tests still pass after optimization
- [ ] Profiling report showing improvements

**Deliverables:**
- Optimized codebase
- Performance benchmarks (before/after)
- Profiling reports
- Documentation of optimization strategies

**Estimated Duration:** 3-5 days

---

## Phase 6: Multi-Agent Competition (Level 4) 📋 FUTURE

**Goal:** Add competitive multi-agent scenarios with theory of mind.

**Concepts:**
- Agents compete for limited resources (only one can use Job at a time)
- Theory of mind: predict other agents' actions
- Emergent cooperation or competition
- Communication (implicit via observation)

**Teaching Value:**
- Game theory applications
- Social intelligence
- Emergent behavior
- Multi-agent RL challenges

**Status:** Exploratory - depends on Phase 4/5 success

---

## Phase 7: Emergent Communication (Level 5) 📋 FUTURE

**Goal:** Family units with emergent language and communication.

**Concepts:**
- Family members can share information
- Communication channel (discrete symbols)
- Emergent protocols for coordination
- Language grounding in shared experience

**Teaching Value:**
- Language emergence
- Communication protocols
- Coordination problems
- Symbolic reasoning

**Status:** Aspirational - long-term vision

---

## Decision Log

### 2025-10-30: Adopted 2→3→1 Strategy

**Decision:** Do multi-day demo → POMDP → optimization (not demo → optimization → POMDP)

**Rationale:**
- Avoids 1→3→1 trap (optimizing twice)
- POMDP reveals real optimization needs
- Current system is proven stable (37/37 tests)
- Optimization targets are concrete not theoretical

**Trade-offs Considered:**
- **Pro (1 first):** Fix known issues while fresh, cleaner foundation
- **Pro (3 first):** One optimization pass, informed by real needs, no wasted work
- **Conclusion:** Real profiling data > theoretical concerns

### 2025-10-28: Completed Phase 2 (Adversarial Curriculum)

**Achievement:** 5-stage curriculum with entropy-gated progression, 13/13 tests passing

### 2025-10-27: Completed Phase 1 (Vectorized Environment)

**Achievement:** GPU-native 8×8 grid environment with 4 affordances, 4 meters

---

## Next Actions

1. **Immediate:** Plan/brainstorm multi-day demo (if needed)
2. **This Week:** Execute multi-day demo run
3. **Next Week:** Analyze demo results, document findings
4. **Following:** Start POMDP planning/implementation

---

## References

- Phase 3 Implementation: `docs/plans/2025-10-30-townlet-phase3-implementation.md`
- Phase 3 Design: `docs/plans/2025-10-30-townlet-phase3-intrinsic-exploration.md`
- Phase 2 Design: `docs/plans/2025-10-30-townlet-phase2-adversarial-curriculum.md`
- Verification: `docs/townlet/PHASE3_VERIFICATION.md`
- Architecture: `docs/ARCHITECTURE_DESIGN.md`
