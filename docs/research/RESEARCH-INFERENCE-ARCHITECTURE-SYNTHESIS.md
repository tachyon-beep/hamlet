# Research Synthesis: Dual-Purpose Architecture

## Overview

HAMLET needs to support **two distinct use cases**:

### Use Case 1: Live Training Visualization
**Goal:** Show students "what the agent is learning RIGHT NOW as it trains"
- Audience: Students watching training progress
- Latency: Real-time (<1 second)
- Source: Actual training episodes
- Use: During training run

### Use Case 2: Production Inference
**Goal:** Deploy trained models for external use (student projects, demos, research)
- Audience: Students using trained models in their own code
- Latency: Doesn't matter (load once, use many times)
- Source: Exported model weights
- Use: After training complete

---

## Unified Architecture Vision

Both use cases benefit from the **same foundational improvements**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HamletModel                                   │
│  (Standalone model abstraction - TASK-XXX-MODEL-ABSTRACTION)        │
│                                                                       │
│  • predict(obs, hidden, epsilon) → {actions, q_values, hidden}      │
│  • load_from_checkpoint(path)                                       │
│  • save_inference_checkpoint(path)                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ▲           ▲
                              │           │
          ┌───────────────────┘           └────────────────────┐
          │                                                     │
┌─────────┴───────────┐                           ┌────────────┴─────────┐
│  Use Case 1:        │                           │  Use Case 2:         │
│  Live Viz           │                           │  Production Inference│
├─────────────────────┤                           ├──────────────────────┤
│                     │                           │                      │
│  Training Thread    │                           │  Standalone Script   │
│  ├─> EpisodeRecorder│                           │  ├─> Load model      │
│  ├─> Callback       │                           │  ├─> Run inference   │
│  └─> Queue          │                           │  └─> Use predictions │
│      ↓              │                           │                      │
│  Visualization      │                           │  Custom Application  │
│  ├─> Consume queue  │                           │  (student's code)    │
│  ├─> Use model for  │                           │                      │
│  │    Q-value viz   │                           │                      │
│  └─> Broadcast      │                           │                      │
└─────────────────────┘                           └──────────────────────┘
```

---

## Implementation Strategy: Two Parallel Tracks

### Track 1: Live Training Telemetry (IMMEDIATE)
**Research:** `RESEARCH-LIVE-TRAINING-TELEMETRY.md`
**Effort:** 9 hours
**Priority:** HIGH (enables real-time pedagogy)

**Deliverables:**
1. Episode streaming via callback + queue
2. LiveInferenceServer consumes training episodes
3. <1 second latency from training to frontend

**Implementation:**
- Phase 1: Add callback to EpisodeRecorder (2h)
- Phase 2: Wire queue in UnifiedServer (1h)
- Phase 3: Consume in LiveInferenceServer (2h)
- Phase 4: Update DemoRunner (1h)
- Phase 5: Tests (2h)
- Phase 6: Edge cases (1h)

### Track 2: Model Abstraction (FOUNDATION)
**Research:** `RESEARCH-INFERENCE-ARCHITECTURE.md`
**Effort:** 24-30 hours
**Priority:** MEDIUM (enables external usage)

**Deliverables:**
1. `HamletModel` class (standalone inference)
2. Inference checkpoint format (small, portable)
3. Documentation for external usage
4. Example scripts for student projects

**Implementation:**
- Phase 1: Extract HamletModel class (8h)
- Phase 2: Refactor LiveInferenceServer to use HamletModel (6h)
- Phase 3: Inference checkpoint format (4h)
- Phase 4: Tests + docs (6h)
- Phase 5: Migration path (6h)

---

## How They Work Together

### During Training (Use Case 1)

```python
# In DemoRunner (training loop)
for episode in range(max_episodes):
    # Run training episode
    episode_data = run_episode()

    # TRACK 1: Stream to visualization (IMMEDIATE)
    self.recorder.finish_episode(metadata)  # → callback → queue → viz

    # TRACK 2: Model used for Q-value display (FUTURE)
    # LiveInferenceServer can show Q-values using HamletModel.predict()

    # Checkpoints every 100 episodes (existing)
    if episode % 100 == 0:
        save_checkpoint()  # Full training checkpoint
        save_inference_model()  # TRACK 2: Lightweight model export
```

### After Training (Use Case 2)

```python
# Student's external project
from townlet.inference import HamletModel

# TRACK 2: Load trained model
model = HamletModel.load_from_checkpoint("model_ep10000.pt")

# Run inference in custom environment
obs = custom_env.reset()
while not done:
    result = model.predict(obs, epsilon=0.0)  # Greedy
    action = result["actions"][0]
    obs, reward, done, info = custom_env.step(action)
```

---

## Recommended Sequencing

### Option A: Parallel Development (FASTEST)

**Week 1-2: Track 1 (Live Viz)**
- Implement episode streaming (9 hours)
- Get real-time visualization working
- Students can watch training live

**Week 3-5: Track 2 (Model Abstraction)**
- Extract HamletModel (24-30 hours)
- Enable external usage
- Students can export and use models

**Benefits:**
- ✅ Fastest time-to-value (live viz working in 1-2 weeks)
- ✅ Parallel work (can be done by different people)
- ✅ Independent: Track 1 doesn't depend on Track 2

**Drawbacks:**
- ⚠️ Some rework: LiveInferenceServer refactored twice
  - First: Add episode streaming (Track 1)
  - Second: Use HamletModel (Track 2)

---

### Option B: Foundation First (CLEANEST)

**Week 1-4: Track 2 (Model Abstraction)**
- Extract HamletModel first (24-30 hours)
- Clean architecture from the start

**Week 5: Track 1 (Live Viz)**
- Add episode streaming (9 hours)
- Use HamletModel in LiveInferenceServer
- No rework needed

**Benefits:**
- ✅ Cleanest architecture (only refactor once)
- ✅ Live viz can use HamletModel from day 1
- ✅ Better testing (model tested before viz)

**Drawbacks:**
- ❌ Slower time-to-value (4-5 weeks until live viz)
- ❌ Higher upfront commitment (30 hours before seeing results)

---

### Option C: Hybrid - Quick Win + Foundation (RECOMMENDED)

**Week 1: Track 1 - Quick Implementation (9 hours)**
- Implement episode streaming with existing architecture
- Get live viz working NOW
- Don't worry about model abstraction yet

**Week 2-5: Track 2 - Model Abstraction (24-30 hours)**
- Extract HamletModel
- Refactor LiveInferenceServer to use it
- Some code is touched twice, but that's OK

**Benefits:**
- ✅ Quick win: Live viz in Week 1
- ✅ Clean foundation: Model abstraction by Week 5
- ✅ Incremental value: Students benefit immediately
- ✅ Low risk: Proven approach, can pause after Track 1 if needed

**Drawbacks:**
- ⚠️ Some rework: LiveInferenceServer refactored in Week 3
- But rework is minimal (just swap episode source)

**Recommendation:** **Option C** - Get quick win, then build proper foundation

---

## Detailed Roadmap (Option C)

### Week 1: Live Viz (9 hours)

**Goal:** Students can watch training in real-time

**Tasks:**
1. Add callback to EpisodeRecorder (2h)
2. Wire queue in UnifiedServer (1h)
3. Consume in LiveInferenceServer (2h)
4. Update DemoRunner (1h)
5. Tests + edge cases (3h)

**Deliverable:** `docs/research/RESEARCH-LIVE-TRAINING-TELEMETRY.md` → TASK-XXX-LIVE-VIZ

**Success:** Frontend shows training episodes with <1s latency

---

### Week 2-3: Model Abstraction Core (16 hours)

**Goal:** HamletModel class works, testable

**Tasks:**
1. Create `src/townlet/inference/model.py` (8h)
   - HamletModel class
   - predict() method
   - load_from_training_checkpoint()
   - load_from_inference_checkpoint()
   - save_inference_checkpoint()
2. Unit tests (8h)
   - Test both network types (simple, recurrent)
   - Test hidden state management
   - Test checkpoint loading
   - Test batch prediction

**Deliverable:** Working HamletModel class with tests

**Success:** Can load model and run predictions outside training infrastructure

---

### Week 4: Inference Checkpoints (10 hours)

**Goal:** Export lightweight models

**Tasks:**
1. Add inference checkpoint save to DemoRunner (2h)
2. Define inference checkpoint format (2h)
3. Update HamletModel loading (2h)
4. Migration script for old checkpoints (3h)
5. Tests (1h)

**Deliverable:** Training saves both checkpoint_ep*.pt (training) and model_ep*.pt (inference)

**Success:** model_ep*.pt files are 5-10x smaller than checkpoint_ep*.pt

---

### Week 5: Integration (8 hours)

**Goal:** LiveInferenceServer uses HamletModel

**Tasks:**
1. Refactor LiveInferenceServer (4h)
   - Remove VectorizedPopulation dependency
   - Use HamletModel for Q-value computation
   - Keep episode streaming from Week 1
2. Tests (2h)
3. Documentation (2h)
   - `docs/manual/INFERENCE_USAGE.md`
   - Example scripts

**Deliverable:** Clean architecture, fully documented

**Success:**
- Live viz works with HamletModel
- Students can export and use models externally

---

## Key Design Decisions

### Decision 1: Model Abstraction is Shared

Both use cases benefit from `HamletModel`:

**Live Viz:**
- Can use `model.predict()` to show Q-values in real-time
- Clean interface for policy visualization
- No dependency on VectorizedPopulation

**Production Inference:**
- Students call `model.predict()` in their code
- Portable, self-contained
- Works anywhere (no training dependencies)

**Conclusion:** Model abstraction is foundational for both use cases

---

### Decision 2: Episode Streaming is Orthogonal

Episode streaming (Track 1) is independent of model abstraction (Track 2):

**Can implement episode streaming without model abstraction:**
- Use existing checkpoint loading
- Stream training episodes
- Works fine for live viz

**Can implement model abstraction without episode streaming:**
- Keep checkpoint polling
- HamletModel still useful for external use
- Works fine for production inference

**Conclusion:** Tracks are independent, can be done in any order

---

### Decision 3: Hybrid Sequencing Wins

Why Option C (Hybrid) is best:

1. **Quick pedagogical win:** Live viz in Week 1 (students benefit immediately)
2. **Low risk:** Can stop after Track 1 if needed (still valuable)
3. **Incremental value:** Each week delivers something useful
4. **Foundation eventually:** Model abstraction by Week 5 (clean architecture)
5. **Minimal rework:** Only LiveInferenceServer touched twice (~4 hours rework)

**Alternative considered:** Option B (Foundation First) is cleaner but takes 5 weeks before any visible benefit. Too risky for pedagogical project.

---

## Success Criteria

### Track 1: Live Viz (Week 1)

✅ **Immediate Win:**
- [ ] Training episodes stream to frontend with <1s latency
- [ ] Students can watch "what agent is doing right now"
- [ ] No checkpoint polling delay
- [ ] Episodes show actual training behavior (not reconstructed)

### Track 2: Model Abstraction (Week 2-5)

✅ **Foundation:**
- [ ] HamletModel class exists and is well-tested
- [ ] Can load both network types (simple, recurrent)
- [ ] predict() is pure function (no hidden state in network)
- [ ] Inference checkpoints are 5-10x smaller

✅ **External Usage:**
- [ ] Students can load model in their own scripts
- [ ] Example: Load model, run in custom environment
- [ ] Documentation explains how to export and use models
- [ ] No dependency on training infrastructure

✅ **Integration:**
- [ ] LiveInferenceServer uses HamletModel
- [ ] Live viz still works (episode streaming + model)
- [ ] Backward compatible (old checkpoints load)

---

## Effort Summary

| Track | Description | Hours | Priority |
|-------|-------------|-------|----------|
| **1** | Live Training Telemetry | 9 | HIGH |
| **2** | Model Abstraction | 24-30 | MEDIUM |
| **Total** | Both tracks | 33-39 | - |

**Sequencing (Option C - Recommended):**
- Week 1: Track 1 (9h) → Quick win
- Week 2-5: Track 2 (24-30h) → Foundation
- Total: 5 weeks, 33-39 hours

**Alternative Sequencing:**
- Option A (Parallel): Both tracks simultaneously (fastest calendar time)
- Option B (Foundation First): Track 2 then Track 1 (cleanest, but slow)

---

## Risk Analysis

### Risk 1: Scope Creep

**Risk:** Both tracks expand beyond estimates.

**Mitigation:**
- Track 1 is simple (callback + queue), hard to bloat
- Track 2 can be done in phases (HamletModel first, integration later)
- Each track delivers value independently (can stop early)

### Risk 2: Rework in Option C

**Risk:** LiveInferenceServer refactored twice (Week 1 + Week 5).

**Analysis:**
- Week 1: Add episode streaming (~2h for LiveInferenceServer changes)
- Week 5: Use HamletModel (~4h to refactor)
- Total rework: ~6 hours
- But get live viz working 4 weeks earlier

**Verdict:** Worth the rework for faster time-to-value

### Risk 3: Model Abstraction Complexity

**Risk:** Extracting from VectorizedPopulation is harder than estimated.

**Mitigation:**
- Start with Phase 1 only (HamletModel class)
- Validate effort before committing to full refactor
- Can fall back to inference checkpoints only (simpler, still useful)

---

## Next Steps

1. **Review both research documents:**
   - `RESEARCH-LIVE-TRAINING-TELEMETRY.md` (Track 1)
   - `RESEARCH-INFERENCE-ARCHITECTURE.md` (Track 2)

2. **Decide on sequencing:**
   - Option A: Parallel (fastest)
   - Option B: Foundation first (cleanest)
   - Option C: Hybrid (recommended - quick win + foundation)

3. **Create TASK documents:**
   - `TASK-XXX-LIVE-TRAINING-VIZ.md` (Track 1)
   - `TASK-YYY-MODEL-ABSTRACTION.md` (Track 2)

4. **Implement Track 1 first (if Option C):**
   - 9 hours
   - Students benefit immediately
   - Low risk, high value

---

## Summary

**The Real Need:** Both live visualization AND production inference

**Solution:** Two parallel tracks with shared foundation (HamletModel)

**Recommended Approach:** Hybrid (Option C)
- Week 1: Live viz (quick win)
- Week 2-5: Model abstraction (foundation)
- Total: 33-39 hours over 5 weeks

**Key Insight:** Tracks are independent but synergistic. Model abstraction improves both use cases. Episode streaming enables real-time pedagogy.

**Next:** Review research, confirm sequencing, create TASK documents.
