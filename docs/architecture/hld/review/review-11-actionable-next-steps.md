---
title: "Section 11: Actionable Next Steps"
document_type: "Implementation Guide"
status: "Draft"
version: "2.5"
---

## SECTION 11: ACTIONABLE NEXT STEPS

**Note**: This section describes **planned work**, not the current implementation. The current system (Townlet) implements L0-L3 from CLAUDE.md. This section outlines the roadmap to implement the full L0-L8 curriculum from the HLD.

### 11.1 The Critical Path: First 30 Days

**Goal**: Fix blockers, complete specifications, achieve L0-L5 stability (HLD curriculum L0-L8)

**Assumption**: One full-time developer, already familiar with codebase

---

#### Week 1: Fix Critical Blockers

**Objective**: Address all three blockers from Section 6

**Monday-Tuesday: EthicsFilter Refactor**

```python
# Tasks:
[ ] Remove EthicsFilter from modules dict in agent_architecture.yaml
[ ] Implement EthicsFilter as pure Python class (no nn.Module)
[ ] Update execution_graph.yaml (ethics is controller in @controllers namespace, not module)
[ ] Remove EthicsFilter from weights.pt saving/loading
[ ] Write test: assert not hasattr(ethics_filter, 'parameters')
[ ] Update Brain as Code doc (Section 6.4)
[ ] Update Checkpoint doc (Section 4.1)

# Validation:
python -m pytest tests/test_ethics_filter.py
# Expected: EthicsFilter has no learnable parameters
# Expected: Vetoes are deterministic (same input → same output)
```

**Estimated time**: 2 days
**Risk**: Low (straightforward refactor)

---

**Wednesday-Thursday: Checkpoint Signatures**

```python
# Tasks:
[ ] Implement SecureCheckpointWriter class (from Section 6.3)
[ ] Generate signing key (~/.townlet/signing_key.bin)
[ ] Update checkpoint writer to compute manifest + signature
[ ] Update checkpoint loader to verify before loading
[ ] Write test: tamper with file, verify detection
[ ] Add CheckpointTamperedError exception
[ ] Update Checkpoint doc (Section 4.8)

# Validation:
python -m pytest tests/test_checkpoint_security.py
# Expected: Signature verification passes on valid checkpoint
# Expected: Tampering raises CheckpointTamperedError
```

**Estimated time**: 2 days
**Risk**: Low (standard cryptographic pattern)

---

**Friday: World Config Hash**

```python
# Tasks:
[ ] Implement compute_world_config_hash() function
[ ] Add world_config_hash to observation tensor (Section 4)
[ ] Update ObservationBuilder to include hash
[ ] Update Module B to accept world_config_hash as input
[ ] Write test: different configs → different hashes
[ ] Update High-Level Design doc (Section 13.2)

# Validation:
python -m pytest tests/test_world_config_hash.py
# Expected: Hash changes when affordance costs change
# Expected: Hash stable across runs with same config
```

**Estimated time**: 1 day
**Risk**: Low (hash computation is straightforward)

---

**Week 1 Deliverable**: All three blockers fixed, tests passing

**Checkpoint**: Can you answer these questions?

- ✅ "Is EthicsFilter learned?" → NO (deterministic rules)
- ✅ "Can checkpoints be tampered with?" → NO (signatures detect tampering)
- ✅ "Can world model adapt to curriculum changes?" → YES (hash conditioning)

---

#### Week 2: Complete Missing Specifications

**Objective**: Document and implement all underspecified behaviors from Section 7

**Monday-Tuesday: Multi-Agent Contention**

```python
# Tasks:
[ ] Implement resolve_contention() (distance-first, then agent_id)
[ ] Add action_failed flag to observation
[ ] Write contention resolution tests
[ ] Document in docs/multi_agent_mechanics.md
[ ] Update affordances.yaml schema to include capacity field
[ ] Add telemetry logging for contention events

# Validation:
python -m pytest tests/test_contention.py
# Expected: Same positions → deterministic winner
# Expected: Closer agent wins
# Expected: Loser receives action_failed=True
```

**Estimated time**: 2 days
**Risk**: Low (clear specification)

---

**Wednesday-Thursday: Family Lifecycle**

```python
# Tasks:
[ ] Implement Family class with full state machine
[ ] Implement all state transitions (from Section 7.2)
[ ] Add family_state to Agent class
[ ] Write transition tests (all edge cases)
[ ] Document in docs/family_lifecycle.md
[ ] Create population_genetics.yaml schema

# Validation:
python -m pytest tests/test_family_lifecycle.py
# Expected: All transitions work correctly
# Expected: Child maturity triggers parent eligibility
# Expected: Death updates family state correctly
```

**Estimated time**: 2 days
**Risk**: Medium (complex state machine, many edge cases)

---

**Friday: Child Initialization**

```python
# Tasks:
[ ] Implement initialize_child() with all modes
[ ] Implement crossover_dna() and mutate_dna()
[ ] Implement average_weights() for crossover mode
[ ] Write child initialization tests
[ ] Document in docs/population_genetics.md
[ ] Add genetics config to population_genetics.yaml

# Validation:
python -m pytest tests/test_child_initialization.py
# Expected: Crossover produces intermediate DNA
# Expected: Mutation perturbs genes slightly
# Expected: Weight inheritance works for all modes
```

**Estimated time**: 1 day
**Risk**: Low (clear algorithms)

---

**Week 2 Deliverable**: All missing specs documented and implemented

**Checkpoint**: Can you answer these questions?

- ✅ "What happens when 2 agents try to use Job?" → Closest wins, loser gets action_failed
- ✅ "What happens when a parent dies?" → Family transitions to SINGLE_PARENT
- ✅ "How does a child inherit weights?" → Configurable (crossover/clone/pretrained/random)

---

#### Week 3: Social Observability (L6-7)

**Objective**: Implement observation space for multi-agent levels

**Monday-Tuesday: Cue Engine**

```python
# Tasks:
[ ] Implement CueEngine class
[ ] Load cues from cues.yaml
[ ] Implement compute_cues() (evaluate triggers, emit top-k)
[ ] Write cue computation tests
[ ] Create baseline cue pack (12 cues from Section 5.4)
[ ] Validate cues.yaml schema

# Validation:
python -m pytest tests/test_cue_engine.py
# Expected: Triggers evaluate correctly
# Expected: Max 3 cues emitted per agent
# Expected: Priority ordering works
```

**Estimated time**: 2 days
**Risk**: Low (clear specification in Section 5)

---

**Wednesday-Thursday: Social Observation Builder**

```python
# Tasks:
[ ] Add _get_visible_agent_positions() to ObservationBuilder
[ ] Add _get_visible_agent_cues() to ObservationBuilder
[ ] Update build_observations() for curriculum_level >= 6
[ ] Write observation tests (L6-7)
[ ] Verify observation dimensions match Section 4.5

# Implementation (from Section 4.8):
def _get_visible_agent_positions(self, observer_positions, all_positions):
    # Return [num_agents, max_visible * 2]
    # Relative positions, sorted by distance, padded

def _get_visible_agent_cues(self, observer_positions, all_positions, all_cues):
    # Return [num_agents, max_visible * num_cue_types]
    # Binary matrix of active cues

# Validation:
python -m pytest tests/test_social_observations.py
# Expected: Visible agents sorted by distance
# Expected: Cues encoded as binary vectors
# Expected: Padding works correctly
```

**Estimated time**: 2 days
**Risk**: Medium (careful indexing, padding edge cases)

---

**Friday: Module C: Social Model Integration**

```python
# Tasks:
[ ] Update Module C: Social Model to accept cue inputs
[ ] Implement cue embedding layer
[ ] Test Module C forward pass with cues
[ ] Verify Module C: Social Model can predict state from cues (placeholder training)

# Validation:
# Manual test: feed cues ['looks_tired', 'at_job']
# Expected: Module C: Social Model outputs prediction (even if untrained)
```

**Estimated time**: 1 day
**Risk**: Low (architecture already defined, just connecting)

---

**Week 3 Deliverable**: L6-7 observation space working

**Checkpoint**: Can you run a multi-agent episode with social observations?

```bash
python scripts/run_episode.py --level L6 --num_agents 5  # HLD Level 6
# Expected: Episode completes, observations include other_agents_in_window
# Expected: Cues appear in telemetry logs
```

---

#### Week 4: Communication Channel (L8)

**Objective**: Implement family communication system (HLD Level 8)

**Monday-Tuesday: Family Communication Infrastructure**

```python
# Tasks:
[ ] Add SET_COMM_CHANNEL action to action space
[ ] Add current_signal field to Agent class
[ ] Add family_id and family_members fields
[ ] Process SET_COMM_CHANNEL in environment step
[ ] Write action processing tests

# Validation:
python -m pytest tests/test_communication_action.py
# Expected: SET_COMM_CHANNEL updates agent.current_signal
# Expected: Signal persists across ticks until changed
```

**Estimated time**: 2 days
**Risk**: Low (straightforward state management)

---

**Wednesday-Thursday: Family Communication Observations**

```python
# Tasks:
[ ] Add _get_family_comm_channel() to ObservationBuilder
[ ] Update build_observations() for curriculum_level >= 8
[ ] Normalize signals [0, 999] → [0, 1]
[ ] Handle agents not in families (all zeros)
[ ] Write observation tests (L8)

# Implementation (from Section 4.8):
def _get_family_comm_channel(self, agent_ids, family_data):
    # Return [num_agents, max_family_size]
    # Normalized signals from family members

# Validation:
python -m pytest tests/test_family_observations.py
# Expected: Family members' signals appear in observation
# Expected: Non-family agents receive zeros
```

**Estimated time**: 2 days
**Risk**: Low (similar to social observations)

---

**Friday: End-to-End L8 Test**

```python
# Tasks:
[ ] Create test scenario: 2-agent family
[ ] Parent sets signal, verify child receives it
[ ] Track signal over 100 ticks
[ ] Verify observation dimensions match Section 4.6

# Validation:
python scripts/run_episode.py --level L8 --num_agents 3 --families 1  # HLD Level 8
# Expected: Episode completes
# Expected: Family channel in observations
# Expected: SET_COMM_CHANNEL actions in telemetry
```

**Estimated time**: 1 day
**Risk**: Low (integration test)

---

**Week 4 Deliverable**: L8 communication working end-to-end

**Checkpoint**: Can you run a family episode with communication?

```bash
python scripts/run_family_episode.py
# Expected: Parents and child can exchange signals
# Expected: Signals visible in telemetry
# Expected: No semantic meaning (agents must learn)
```

---

### 11.2 Months 2-3: Training & Validation

**Goal**: Train agents L0-L8 (HLD curriculum), validate curriculum progression, run pilot experiments

---

#### Month 2, Week 1: L0-L3 Training (HLD Curriculum)

```python
# Tasks:
[ ] Train baseline agents on L0-L3 (HLD levels)
[ ] Verify learning curves (from Section 3.2)
[ ] Measure graduation criteria:
    - L0 (HLD): Learns "Bed fixes energy" by episode 20
    - L1 (HLD): Establishes Job-Bed loop by episode 100
    - L2 (HLD): Reaches retirement by episode 200
    - L3 (HLD): Terminal score > 0.65 by episode 500
[ ] Document hyperparameters (learning rate, batch size, etc)
[ ] Save trained checkpoints

# Validation:
- L0 (HLD): 80%+ agents use Bed when energy < 0.3
- L1 (HLD): Stable income from Job by episode 100
- L2 (HLD): 70%+ retirement rate
- L3 (HLD): 80%+ retirement rate, mean score > 0.65
```

**Estimated time**: 1 week (parallel training on GPU)
**Risk**: Medium (may need hyperparameter tuning)

---

#### Month 2, Week 2: L4-L5 Training (HLD - LSTM Required)

```python
# Tasks:
[ ] Implement RecurrentSpatialQNetwork architecture
[ ] Train on L4-L5 (HLD levels) with partial observability
[ ] Verify RND-driven exploration works
[ ] Measure spatial memory performance
[ ] Compare to L3 (HLD) baseline (should be similar after training)

# Validation:
- L4 (HLD): Agents build mental maps by episode 2000
- L4 (HLD): Can navigate to unseen affordances from memory
- L5 (HLD): Respects operating hours (doesn't spam Job at 3am)
- L5 (HLD): Uses WAIT strategically (arrive early, wait for open)
```

**Estimated time**: 1 week
**Risk**: High (LSTM training is finicky, may need architecture tuning)

---

#### Month 2, Week 3: Module C: Social Model Pretraining (CTDE)

```python
# Tasks:
[ ] Collect logged episodes from L4-5 (HLD) training
[ ] Extract (cues, ground_truth_state) pairs
[ ] Train Module C: Social Model via supervised learning
[ ] Evaluate prediction accuracy on held-out set
[ ] Save pretrained Module C: Social Model checkpoint

# Validation:
- Module C: Social Model predicts energy with MAE < 0.15
- Module C: Social Model predicts health with MAE < 0.15
- Module C: Social Model goal prediction accuracy > 60%
- Cue embeddings are learned (not random)
```

**Estimated time**: 1 week
**Risk**: Medium (depends on data quality)

---

#### Month 2, Week 4: L6-L7 Training (HLD - Social Reasoning)

```python
# Tasks:
[ ] Load pretrained Module C: Social Model
[ ] Train full Module A-D stack on L6 (HLD level)
[ ] Measure contention resolution behavior
[ ] Train on L7 (HLD level) with rich cues
[ ] Compare L6 vs L7 performance

# Validation:
- L6 (HLD): Agents choose alternate affordances when contention detected
- L6 (HLD): Strategic resource selection > 60% success rate
- L7 (HLD): Better predictions than L6 (more cues → better inference)
- L7 (HLD): Retirement rate matches L5 (no performance degradation)
```

**Estimated time**: 1 week
**Risk**: High (social reasoning is complex, may need architecture changes)

---

#### Month 3, Week 1-2: L8 Training (HLD - Emergent Communication)

```python
# Tasks:
[ ] Train families on L8 (HLD level) with communication channel
[ ] Track signal usage over 20k episodes
[ ] Measure coordination metrics:
    - Signal diversity
    - Signal stability
    - Coordination gain (family vs solo)
[ ] Analyze emergent protocols
[ ] Document learned signal meanings (post-hoc)

# Validation:
- Families use 3-10 unique signals by episode 20k
- Signal stability > 0.5 (same signal → same context 50%+ of time)
- Family coordination gain > 10% over solo agents
```

**Estimated time**: 2 weeks (long training, extensive analysis)
**Risk**: Very High (emergent communication may not emerge reliably)

---

#### Month 3, Week 3: Curriculum Validation

```python
# Tasks:
[ ] Train agents from scratch through full L0-L8 (HLD) curriculum
[ ] Measure transfer learning (does L7 help L8?)
[ ] Compare to agents trained only on L8 (HLD level, no curriculum)
[ ] Document curriculum benefits
[ ] Write curriculum_results.md

# Validation:
- Curriculum agents outperform scratch agents on L8 (HLD level)
- Each HLD level provides measurable skill (ablation study)
- Training time: curriculum < 2× scratch training
```

**Estimated time**: 1 week
**Risk**: Medium (may not see curriculum benefit)

---

#### Month 3, Week 4: Pilot Experiments

```python
# Tasks:
[ ] Run 3 pilot experiments from Section 8.4:
    1. Q1.1: Does natural selection work? (meritocratic mode)
    2. Q2.1: Emergent communication protocols (L8 HLD families)
    3. Q3.1: Wealth concentration (dynasty mode)
[ ] Collect data, create visualizations
[ ] Write experiment_results.md
[ ] Prepare for publication/demo

# Validation:
- Each experiment has clear hypothesis + result
- Visualizations are publication-ready
- Code + configs are reproducible
```

**Estimated time**: 1 week
**Risk**: Low (exploratory, no specific target)

---

### 11.3 Month 4+: Documentation, Release, Research

**Goal**: Prepare for public release, enable external researchers

---

#### Month 4, Week 1-2: Documentation

```python
# Tasks:
[ ] Write QUICKSTART.md (5-minute tutorial)
[ ] Write TUTORIAL.md (30-minute walkthrough)
[ ] Write API_REFERENCE.md (complete spec)
[ ] Write CONTRIBUTING.md (for contributors)
[ ] Polish all existing docs (Sections 0-9 from this review)
[ ] Create video tutorials (optional)

# Validation:
- New user can run first experiment in 10 minutes
- All public APIs are documented
- Examples cover L0-L8 (HLD curriculum) + all genetics modes
```

**Estimated time**: 2 weeks
**Risk**: Low (tedious but straightforward)

---

#### Month 4, Week 3: Schema Validation & Tooling

```python
# Tasks:
[ ] Write JSON schemas for all YAML configs
[ ] Implement config validator (pre-flight checks)
[ ] Write schema tests (reject invalid configs)
[ ] Create config templates for common scenarios
[ ] Implement townlet validate command

# Example:
townlet validate --config configs/my_world/
# Expected: Reports errors in YAML before launching

# Validation:
- All invalid configs are caught by validator
- Error messages are helpful ("missing required field: cascade.target")
- Templates cover 80% of use cases
```

**Estimated time**: 1 week
**Risk**: Low

---

#### Month 4, Week 4: Testing & CI

```python
# Tasks:
[ ] Achieve 80%+ code coverage
[ ] Set up GitHub Actions (pytest, linting, type checking)
[ ] Write integration tests (L0-L8 smoke tests)
[ ] Write regression tests (prevent performance degradation)
[ ] Document testing strategy in tests/README.md

# Validation:
- pytest tests/ passes on main branch
- CI runs on every PR
- No critical bugs in issue tracker
```

**Estimated time**: 1 week
**Risk**: Medium (achieving good coverage is time-consuming)

---

#### Month 5+: Public Release

```python
# Tasks:
[ ] Create GitHub repo (github.com/tachyon-beep/townlet)
[ ] Write README.md (from Section 11 of this doc)
[ ] Tag v1.0.0 release
[ ] Announce on Twitter, Reddit, HN
[ ] Submit to RL/AI conferences (NeurIPS, ICLR)
[ ] Write blog post (architecture deep dive)
[ ] Create Discord/Slack community (optional)

# Success criteria:
- 100+ GitHub stars in first month
- 5+ external contributors
- 1+ research paper using Townlet (not by you)
- Featured in RL newsletter/podcast
```

**Estimated time**: Ongoing
**Risk**: High (adoption is unpredictable)

---

### 11.4 Validation Criteria & Success Metrics

**How do you know when each stage is "done"?**

#### Week 1 Success: Blockers Fixed

```python
✅ EthicsFilter has no learnable parameters
✅ Checkpoint tampering raises CheckpointTamperedError
✅ World config changes are observable
✅ All tests pass: pytest tests/test_blockers.py
```

#### Week 2 Success: Specifications Complete

```python
✅ Contention resolution is deterministic
✅ All family state transitions work
✅ Child initialization supports 4 modes
✅ All tests pass: pytest tests/test_specifications.py
```

#### Week 3 Success: L6-7 Working

```python
✅ Cue engine emits correct cues
✅ Social observations include positions + cues
✅ Module C: Social Model accepts cue inputs
✅ Multi-agent episode completes without errors
```

#### Week 4 Success: L8 Working

```python
✅ SET_COMM_CHANNEL action works
✅ Family channel in observations
✅ Family episode runs end-to-end
✅ Telemetry logs show signals
```

#### Month 2 Success: L0-L5 Trained (HLD Curriculum)

```python
✅ L0-L3 (HLD): Agents reach retirement reliably
✅ L4-L5 (HLD): LSTM agents navigate from memory
✅ Learning curves match expectations
✅ Saved checkpoints load correctly
```

#### Month 3 Success: L6-L8 Trained (HLD Curriculum)

```python
✅ Module C: Social Model predicts state from cues
✅ L6-L7 (HLD): Social reasoning works
✅ L8 (HLD): Signals are used (even if protocol is unclear)
✅ Pilot experiments produce results
```

#### Month 4 Success: Release-Ready

```python
✅ Documentation is complete
✅ New user can run experiment in 10 minutes
✅ Tests pass, coverage > 80%
✅ No critical bugs
```

---

### 11.5 Resource Requirements

**What do you need to execute this plan?**

#### Human Resources

**Minimum (solo developer)**:

- 1 full-time developer (4 months)
- Familiar with: PyTorch, RL, YAML, testing
- Can learn: Townlet codebase (1 week ramp-up)

**Optimal (small team)**:

- 1 lead engineer (architecture, curriculum L6-L8)
- 1 infrastructure engineer (testing, CI, docs)
- 1 researcher (experiments, analysis, papers)
- 4 months total

**Consulting support**:

- 1 RL expert (review architecture, advise on training)
- 5 hours/month × 4 months = 20 hours

#### Compute Resources

**Week 1-4 (implementation)**:

- Local dev machine (CPU only)
- Total cost: $0

**Month 2-3 (training)**:

- GPU instance (e.g., AWS p3.2xlarge with V100)
- Training time:
  - L0-L3: 24 hours
  - L4-L5: 48 hours (LSTM)
  - L6-L7: 72 hours (Module C + multi-agent)
  - L8: 120 hours (long-horizon communication)
- Total: ~260 GPU-hours
- Cost: ~$3/hour × 260 = **~$780**

**Month 3-4 (experiments)**:

- Same GPU instance
- Pilot experiments: 50 GPU-hours
- Ablations: 30 GPU-hours
- Total: 80 GPU-hours
- Cost: ~$3/hour × 80 = **~$240**

**Total compute cost**: ~$1,020 (very manageable)

**Alternative (free tier)**:

- Google Colab Pro+ ($50/month)
- 4 months × $50 = $200
- Trade-off: Slower, but much cheaper

#### Software Infrastructure

**Required**:

- GitHub (free for public repos)
- PyTorch (free)
- Pytest (free)
- YAML libraries (free)

**Optional**:

- Weights & Biases (experiment tracking, free tier sufficient)
- Discord (community, free)
- Notion/GitHub Wiki (docs, free)

**Total software cost**: $0 (using free tiers)

---

### 11.6 Risk Mitigation

**What could go wrong? How do you handle it?**

#### Risk 1: LSTM Training Fails (L4-L5)

**Probability**: Medium (~40%)
**Impact**: High (blocks L6-L8)

**Mitigation**:

- Week 2 fallback: Use simpler RNN instead of LSTM
- Week 3 fallback: Keep full observability, skip partial obs curriculum
- Month 2 fallback: Use pretrained spatial encoder from other RL work
- Acceptance: L4-L5 may need more tuning than expected (add 2 weeks buffer)

---

#### Risk 2: Emergent Communication Doesn't Emerge (L8)

**Probability**: High (~60%)
**Impact**: Medium (L8 is research frontier, not core platform)

**Mitigation**:

- Month 3 fallback: Use hand-crafted signal semantics as baseline
- Month 3 alternative: Focus on L6-L7 (social reasoning still valuable)
- Month 4 pivot: Reframe as "communication infrastructure" not "emergent protocols"
- Acceptance: L8 is aspirational; failure is publishable result

---

#### Risk 3: Multi-Agent Training is Unstable (L6-L7)

**Probability**: Medium (~50%)
**Impact**: High (blocks social reasoning)

**Mitigation**:

- Month 2 fallback: Train agents against frozen opponents (not co-adapting)
- Month 2 alternative: Use league training (like AlphaStar)
- Month 3 pivot: Focus on single-agent with simulated competitors
- Acceptance: Multi-agent RL is hard; may need architecture research

---

#### Risk 4: Performance Degradation Across Curriculum

**Probability**: Low (~20%)
**Impact**: Medium (curriculum value unclear)

**Mitigation**:

- Month 3 validation: Measure transfer vs scratch
- Month 3 alternative: Offer both curriculum and direct training
- Acceptance: Curriculum may be pedagogical (for humans) not necessary (for agents)

---

#### Risk 5: Adoption Fails (Post-Release)

**Probability**: Medium (~50%)
**Impact**: Low (research value exists regardless)

**Mitigation**:

- Month 4 preparation: Create killer demo (video, interactive)
- Month 5 outreach: Target specific research groups (multi-agent RL, evolutionary)
- Month 6 pivot: Use internally for your own research (still valuable)
- Acceptance: Building good tools is inherently useful, even with small community

---

### 11.7 Checkpoint Meetings (Self-Review)

**Weekly check-ins to stay on track**

#### Week 1 Friday: Blockers Review

```
Questions:
1. Are all three blockers fixed? (Yes/No)
2. Do tests pass? (pytest tests/test_blockers.py)
3. Is documentation updated? (Sections 6.1-6.3)

If NO to any:
- Identify bottleneck (technical? time?)
- Adjust plan (cut scope? extend deadline?)
- Document decision (why, what changes)
```

#### Week 2 Friday: Specifications Review

```
Questions:
1. Is contention deterministic? (run test 100 times)
2. Do all family transitions work? (edge case tests)
3. Can you initialize a child? (test all 4 modes)

If NO to any:
- Extend Week 2 into Week 3 (push social obs back)
- Or: Punt complex specs to Month 4 (focus on core curriculum)
```

#### Week 3 Friday: Social Observations Review

```
Questions:
1. Can you run L6 multi-agent episode? (3+ agents)
2. Are cues in observations? (inspect tensor)
3. Does Module C accept cues? (forward pass test)

If NO to any:
- Debug observation builder (print tensors)
- Simplify cues (use only 3 types, not 12)
- Defer Module C training (use random predictions)
```

#### Week 4 Friday: Communication Review

```
Questions:
1. Can family members exchange signals? (end-to-end test)
2. Is SET_COMM_CHANNEL action working? (telemetry check)
3. Are signals in observations? (inspect tensor)

If NO to any:
- Review family lifecycle (is family_id set?)
- Check observation builder (is family_comm_channel populated?)
- Test with 2 agents only (simplify)
```

#### Month 2 End: Training Review

```
Questions:
1. Did L0-L5 agents train successfully? (learning curves)
2. Are graduation criteria met? (retirement rates)
3. Are checkpoints saved? (can load and resume)

If NO to any:
- Extend training (add 1 week)
- Tune hyperparameters (learning rate, batch size)
- Accept lower performance (update success criteria)
```

#### Month 3 End: Curriculum Review

```
Questions:
1. Did L6-L8 training complete? (even if imperfect)
2. Are pilot experiments done? (3 results)
3. Is anything publishable? (write draft)

If NO to any:
- Focus on best result (L6 or L7, skip L8)
- Simplify experiments (1 strong result > 3 weak)
- Accept partial success (L0-L7 is still valuable)
```

#### Month 4 End: Release Review

```
Questions:
1. Is documentation complete? (QUICKSTART exists)
2. Can external user run experiment? (user testing)
3. Are tests passing? (CI green)

If NO to any:
- Delay release (better to launch well than fast)
- Recruit beta testers (friendly users)
- Cut scope (release L0-L5 first, L6-L8 later)
```

---

### 11.8 Contingency Plans

**If timeline slips, what gets cut?**

#### Priority Tiers

**Tier 0 (Must Have)**:

- Week 1: Blocker fixes
- Week 2-4: L0-L5 implementation
- Month 2: L0-L5 training
- Month 4: Basic documentation

**Tier 1 (Should Have)**:

- Week 3-4: L6-L7 social observations
- Month 3: L6-L7 training
- Month 4: Schema validation
- Month 4: Testing (80% coverage)

**Tier 2 (Nice to Have)**:

- Week 4: L8 communication
- Month 3: L8 training
- Month 3: Pilot experiments
- Month 4: Video tutorials

**Tier 3 (Future Work)**:

- Population genetics experiments
- Alternative inheritance modes
- Dynasty experiments
- External applications (economy, ecosystem)

#### Cut Order (if 4 months becomes 3 months)

**Week 1**: No cuts (blockers are critical)
**Week 2**: No cuts (specs are foundational)
**Week 3**: Cut L7 (keep L6 with sparse cues)
**Week 4**: Cut L8 entirely (defer to future work)
**Month 2**: No cuts (training is essential)
**Month 3**: Cut L8 training, simplify experiments to 1 pilot
**Month 4**: Cut video tutorials, reduce test coverage to 60%

**Result**: 3-month plan focuses on L0-L6, solid foundation for future work

#### Cut Order (if 4 months becomes 2 months)

**Week 1-2**: Combine (fix blockers + critical specs only)
**Week 3-4**: Skip social obs, keep L0-L5 only
**Month 2**: Train L0-L5 only
**Month 3-4**: Documentation + release (L0-L5 only)

**Result**: 2-month plan is "Townlet Core" (single-agent, no social)

---

### 11.9 Success Criteria (Final)

**What does "done" look like?**

#### Minimum Viable Product (3 months)

```python
✅ L0-L5 (HLD curriculum) implemented and working
✅ Agents learn survival strategies from sparse rewards
✅ Partial observability (LSTM) works
✅ All three blockers fixed
✅ Basic documentation (QUICKSTART + API reference)
✅ Tests pass, coverage > 60%
✅ Can demo to external researcher
```

**Outcome**: Research-grade single-agent survival simulator

---

#### Full Product (4 months)

```python
✅ L0-L8 (HLD curriculum) implemented and working
✅ Multi-agent social reasoning (L6-L7 HLD) works
✅ Family communication infrastructure (L8 HLD) exists (even if protocols unclear)
✅ Population genetics system implemented
✅ 3+ pilot experiments completed
✅ Comprehensive documentation (tutorials, videos)
✅ Tests pass, coverage > 80%
✅ Ready for public release
```

**Outcome**: Research platform for multi-agent, social, evolutionary RL

---

#### Stretch Goals (6 months)

```python
✅ All of Full Product, plus:
✅ Emergent communication demonstrated (L8 HLD)
✅ 5+ published experiments (selection, dynasties, protocols)
✅ 10+ external users / contributors
✅ 1+ paper submitted to conference
✅ Alternative applications (economy or ecosystem) implemented
✅ Featured in RL newsletter / podcast
```

**Outcome**: Established research platform, community forming

---

### 11.10 Deliverable Checklist

**Concrete artifacts to produce**

#### Code

- [ ] `ethics_filter.py` (deterministic, no weights)
- [ ] `secure_checkpoint.py` (HMAC signatures)
- [ ] `observation_builder.py` (L6-L8 support)
- [ ] `cue_engine.py` (cue computation)
- [ ] `family.py` (lifecycle state machine)
- [ ] `breeding_selector.py` (pairing logic)
- [ ] `child_initializer.py` (inheritance modes)
- [ ] `population_controller.py` (cap maintenance)

#### Configs

- [ ] `cues.yaml` (baseline cue pack, 12 cues)
- [ ] `population_genetics.yaml` (all inheritance modes)
- [ ] `curriculum_configs/` (L0-L8 HLD directories)
- [ ] Config templates (meritocratic, dynasty, arranged)

#### Tests

- [ ] `test_blockers.py` (Week 1 validation)
- [ ] `test_specifications.py` (Week 2 validation)
- [ ] `test_social_observations.py` (Week 3 validation)
- [ ] `test_family_communication.py` (Week 4 validation)
- [ ] `test_curriculum.py` (integration tests for HLD levels)
- [ ] Coverage report (pytest-cov)

#### Documentation

- [ ] `QUICKSTART.md` (10-minute tutorial)
- [ ] `TUTORIAL.md` (30-minute walkthrough)
- [ ] `API_REFERENCE.md` (complete spec)
- [ ] `CONTRIBUTING.md` (for contributors)
- [ ] `docs/multi_agent_mechanics.md` (contention resolution)
- [ ] `docs/family_lifecycle.md` (state machine)
- [ ] `docs/population_genetics.md` (inheritance modes)
- [ ] `docs/cues.md` (social observability)
- [ ] This master document (Sections 0-11)

#### Trained Models

- [ ] `checkpoints/L0_baseline/` (HLD survival basics)
- [ ] `checkpoints/L3_baseline/` (HLD full small-grid)
- [ ] `checkpoints/L5_baseline/` (HLD LSTM + temporal)
- [ ] `checkpoints/L7_baseline/` (HLD social reasoning)
- [ ] `checkpoints/L8_baseline/` (HLD communication, even if imperfect)
- [ ] `checkpoints/module_c_pretrained/` (Module C: Social Model CTDE)

#### Experimental Results

- [ ] `results/Q1_1_selection_works.md` (natural selection)
- [ ] `results/Q2_1_emergent_comm.md` (HLD L8 protocols, even if negative result)
- [ ] `results/Q3_1_wealth_concentration.md` (dynasties)
- [ ] Learning curves (all L0-L8 HLD levels)
- [ ] Ablation studies (HLD curriculum vs scratch)
- [ ] Visualizations (population dynamics, signal usage)

#### Release Materials

- [ ] `README.md` (from Section 11)
- [ ] GitHub repo setup
- [ ] v1.0.0 release tag
- [ ] Announcement blog post
- [ ] Demo video (5 minutes)
- [ ] Twitter/Reddit posts

---

### 11.11 Final Timeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│ MONTH 1: CORE INFRASTRUCTURE                                │
├─────────────────────────────────────────────────────────────┤
│ Week 1: Fix blockers (ethics, checkpoints, hash)            │
│ Week 2: Complete specs (contention, families, children)     │
│ Week 3: L6-L7 (HLD) social observations                     │
│ Week 4: L8 (HLD) communication channel                      │
│                                                              │
│ Deliverable: L0-L8 (HLD curriculum) implemented (not trained)│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 2: TRAINING (L0-L7 HLD)                               │
├─────────────────────────────────────────────────────────────┤
│ Week 1: L0-L3 (HLD SimpleQNetwork)                          │
│ Week 2: L4-L5 (HLD LSTM)                                    │
│ Week 3: Module C: Social Model pretraining (CTDE)           │
│ Week 4: L6-L7 (HLD social reasoning)                        │
│                                                              │
│ Deliverable: Trained agents L0-L7 (HLD)                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 3: ADVANCED TRAINING & EXPERIMENTS                    │
├─────────────────────────────────────────────────────────────┤
│ Week 1-2: L8 (HLD) training (communication)                 │
│ Week 3: Curriculum validation (HLD L0-L8)                   │
│ Week 4: Pilot experiments (3 results)                       │
│                                                              │
│ Deliverable: Complete HLD curriculum + pilot results        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 4: POLISH & RELEASE                                   │
├─────────────────────────────────────────────────────────────┤
│ Week 1-2: Documentation                                     │
│ Week 3: Schema validation & tooling                         │
│ Week 4: Testing & CI                                        │
│                                                              │
│ Deliverable: v1.0.0 public release                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MONTH 5+: COMMUNITY & RESEARCH                              │
├─────────────────────────────────────────────────────────────┤
│ Public release, user support, paper submissions             │
│ External experiments, community growth                       │
└─────────────────────────────────────────────────────────────┘
```

---

**Critical path**: Blockers → Specs → Training → Release
**Buffer**: 2 weeks (can absorb delays in Month 2-3)
**Total time**: 4 months to v1.0.0
**Total cost**: ~$1,000 compute + 4 months salary

**Next action**: Start Week 1, Monday, with EthicsFilter refactor.

---
