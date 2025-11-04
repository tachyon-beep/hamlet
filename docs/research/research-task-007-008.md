# Research Summary: TASK-007 & TASK-008

## Overview

**Requested:** Refactor inference system to implement best practices

**Discovered:** Two distinct but complementary use cases:
1. **Live Training Visualization** → TASK-007
2. **Model Abstraction for Inference** → TASK-008

**Implementation Strategy:** Single PR implementing both (unless complexity too high)

**Sequencing:** After TASK-006, before BRAIN_AS_CODE integration affects design

---

## Research Documents Created

1. **RESEARCH-INFERENCE-ARCHITECTURE.md**
   - Original analysis (production inference focus)
   - Model abstraction layer design
   - 24-30 hour estimate

2. **RESEARCH-LIVE-TRAINING-TELEMETRY.md**
   - Reframed analysis (live training visualization)
   - Episode streaming architecture
   - 9 hour estimate

3. **RESEARCH-INFERENCE-ARCHITECTURE-SYNTHESIS.md**
   - How both use cases work together
   - Three implementation options (chose Hybrid - Option C)
   - 33-39 hour total estimate

4. **RESEARCH-EXTENDED-USE-CASES.md** (this review)
   - Extended use cases taxonomy (Tiers 1-4)
   - BRAIN_AS_CODE integration design
   - Single PR implementation strategy

---

## TASK-007: Live Training Visualization

### Problem Statement

**Current:** Visualization lags training by 100+ episodes due to checkpoint polling

**Goal:** Stream training episodes directly to frontend with <1 second latency

### Solution Architecture

**Episode Streaming (Callback + Queue):**
```
Training Thread                 Inference Thread
     ↓                               ↓
Run episode                    LiveInferenceServer
     ↓                               ↓
EpisodeRecorder.finish()      consume queue
     ↓                               ↓
Callback → Queue ──────────→  Broadcast WebSocket
                (<1 second)         ↓
                              Frontend displays
```

**Key Components:**
1. Add `live_stream_callback` to EpisodeRecorder
2. Create thread-safe queue in UnifiedServer
3. LiveInferenceServer consumes queue asynchronously
4. Replay episode step-by-step to WebSocket clients

### Implementation Phases

**Phase 1: Add Callback (2h)**
- Modify EpisodeRecorder to accept callback parameter
- Call callback on finish_episode() with episode data
- Non-blocking, drops episodes if queue full

**Phase 2: Wire Queue (1h)**
- Create queue.Queue in UnifiedServer
- Pass queue to both DemoRunner and LiveInferenceServer
- Training writes, inference reads

**Phase 3: Consumer (2h)**
- Add _stream_training_episodes() to LiveInferenceServer
- Consume from queue in background task
- Broadcast episodes to WebSocket clients

**Phase 4: Integration (2h)**
- Update DemoRunner to accept callback
- Handle mode switching (live vs checkpoint)
- Tests

**Phase 5: Polish (2h)**
- Edge case handling (queue full, errors)
- Performance tuning
- Documentation

**Total: 9 hours**

### Success Criteria

✅ Episodes stream from training to frontend with <1s latency
✅ Shows actual training episodes (not reconstructed)
✅ No checkpoint polling overhead
✅ Bounded memory (queue capacity 10)
✅ Graceful degradation (drops episodes if slow)

### Pedagogical Value

**HIGH:**
- Students watch agent learn in real-time
- See exploration → exploitation transition
- Observe curriculum stage progressions
- Identify interesting behaviors (reward hacking, etc.)

---

## TASK-008: Model Abstraction for Inference

### Problem Statement

**Current:** Tight coupling between inference and training infrastructure

**Issues:**
- LiveInferenceServer recreates full training stack
- No standalone model class
- Can't use models in external projects
- Checkpoints mix training state (optimizer, replay buffer) with model weights

**Goal:** Clean model abstraction for external usage

### Solution Architecture

**HamletModel Class:**
```python
class HamletModel:
    """Standalone inference model."""

    def __init__(self, network: nn.Module):
        self.network = network
        self.is_recurrent = isinstance(network, RecurrentSpatialQNetwork)

    def predict(
        self,
        obs: torch.Tensor,
        hidden: tuple | None = None,
        epsilon: float = 0.0,
    ) -> dict:
        """Pure inference function."""
        with torch.no_grad():
            if self.is_recurrent:
                q_values, new_hidden = self.network(obs, hidden)
            else:
                q_values = self.network(obs)
                new_hidden = None

            # Action selection
            if epsilon > 0:
                actions = epsilon_greedy_selection(q_values, epsilon)
            else:
                actions = torch.argmax(q_values, dim=1)

            return {
                "actions": actions,
                "q_values": q_values,
                "hidden": new_hidden,
            }

    @classmethod
    def load_from_checkpoint(cls, path: Path) -> "HamletModel":
        """Load from inference checkpoint."""
        checkpoint = torch.load(path, weights_only=False)

        # Compile network from brain.yaml spec (BRAIN_AS_CODE integration)
        brain_spec = checkpoint["brain_spec"]
        network = compile_brain(brain_spec)
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()

        return cls(network=network)
```

**Inference Checkpoint Format (with BRAIN_AS_CODE):**
```python
{
    "format_version": "inference_v1",

    # BRAIN_AS_CODE integration (embedded spec)
    "brain_spec": {
        "network": {
            "type": "recurrent_spatial",
            "vision_encoder": {...},
            "lstm": {"hidden_dim": 256},
            "q_head": {...},
        }
    },

    # Model weights
    "state_dict": {...},

    # Metadata
    "metadata": {
        "training_episode": 10000,
        "epsilon": 0.05,
        "curriculum_stage": 5,
        "training_config_path": "configs/L2_partial_observability",
    },

    "created_at": 1699123456.78,
}
```

### Implementation Phases

**Phase 1: HamletModel Core (8h)**
- Create `src/townlet/inference/model.py`
- HamletModel class
- predict() method (functional API)
- Load from training checkpoints (backward compat)

**Phase 2: Inference Checkpoints (4h)**
- Define inference_v1 format
- Add save to DemoRunner
- Integrate BRAIN_AS_CODE (embed brain.yaml spec)

**Phase 3: LiveInferenceServer Refactor (6h)**
- Remove VectorizedPopulation dependency
- Use HamletModel directly
- Keep episode streaming from TASK-007

**Phase 4: Tests (6h)**
- Unit tests for HamletModel
- Load both network types (simple, recurrent)
- Hidden state management
- Checkpoint format compatibility

**Phase 5: Documentation (6h)**
- `docs/manual/INFERENCE_USAGE.md`
- Example: Export and use model
- Example: Compare checkpoints
- Migration guide for old checkpoints

**Total: 30 hours** (revised up from 24-30 for safety)

### Success Criteria

✅ HamletModel class works standalone (no training deps)
✅ Can load from training checkpoints (backward compat)
✅ Inference checkpoints 5-10x smaller (~10MB vs ~50MB)
✅ predict() is pure function (no side effects)
✅ BRAIN_AS_CODE integration (embeds brain.yaml spec)
✅ Students can use models in external projects

### Pedagogical Value

**HIGH:**
- Students extract models for their own projects
- Enables transfer learning experiments
- Policy analysis and visualization
- Research project foundation

---

## Single PR Strategy

### Option A: Parallel Tracks (RECOMMENDED IF COMPLEXITY MANAGEABLE)

**Week 1-2: Both tracks in parallel**
- Track A: Live viz (9h)
- Track B: Model abstraction (30h)
- Total: 39 hours

**Implementation Order:**
1. **Phase 1: Model Abstraction Core (8h)** ← Foundation
2. **Phase 2: Live Viz Streaming (9h)** ← Can use HamletModel
3. **Phase 3: Inference Checkpoints (4h)**
4. **Phase 4: Integration (6h)** ← Refactor LiveInferenceServer
5. **Phase 5: Tests + Docs (12h)** ← Both tracks

**Single PR:** All changes together

**Benefits:**
- ✅ Clean architecture from day 1
- ✅ No rework (LiveInferenceServer refactored once)
- ✅ Both use cases delivered together

**Risks:**
- ⚠️ Large PR (39 hours, ~2000 lines changed)
- ⚠️ Longer review time
- ⚠️ Higher risk of merge conflicts

---

### Option B: Sequential Tasks (IF COMPLEXITY TOO HIGH)

**TASK-007: Live Viz (Week 1)**
- Implement episode streaming (9h)
- Use existing architecture (no model abstraction)
- Merge PR

**TASK-008: Model Abstraction (Week 2-3)**
- Extract HamletModel (30h)
- Refactor LiveInferenceServer (some rework)
- Merge PR

**Benefits:**
- ✅ Smaller PRs (easier to review)
- ✅ Incremental value (live viz working sooner)
- ✅ Lower risk per PR

**Drawbacks:**
- ❌ Rework: LiveInferenceServer touched twice (~4h extra)
- ❌ Temporary: Week 1 code gets refactored in Week 2

---

### Recommendation: Option A (Single PR)

**Rationale:**
- 39 hours is manageable for single PR
- Avoids rework (cleaner architecture)
- Both use cases benefit from shared foundation
- Can pause after Phase 1-2 if needed (still valuable)

**Risk Mitigation:**
- Implement Phase 1 first (8h), validate complexity
- If too complex, split into TASK-007 + TASK-008
- Can always break into smaller PRs during implementation

---

## BRAIN_AS_CODE Integration (Post-TASK-004)

Since TASK-004 will be complete before TASK-007/008:

### Changes to Design

**Inference Checkpoint Format:**
```python
# BEFORE TASK-004: Hardcoded params
{
    "network_type": "recurrent",
    "network_params": {"action_dim": 5, "window_size": 5, ...},
    "state_dict": {...},
}

# AFTER TASK-004: Compiled from spec
{
    "brain_spec": {  # Embedded brain.yaml
        "network": {"type": "recurrent_spatial", ...}
    },
    "brain_spec_hash": "sha256:abc123...",
    "state_dict": {...},
}
```

**HamletModel Loading:**
```python
# Compile network from brain.yaml spec
from townlet.compiler import compile_brain

checkpoint = torch.load(path)
network = compile_brain(checkpoint["brain_spec"])
network.load_state_dict(checkpoint["state_dict"])

model = HamletModel(network=network)
```

### Benefits

1. **Self-Contained:** Checkpoint embeds architecture spec
2. **Reproducible:** brain_spec_hash verifies compatibility
3. **Editable:** Students can modify spec and retrain
4. **Ablations:** Easy to test architecture variants

### Coordination Needed

**Before TASK-007/008 implementation:**
- Review TASK-004 design (brain.yaml schema)
- Confirm compile_brain() API
- Ensure checkpoint format compatibility

---

## Extended Use Cases (Out of Scope)

### Tier 2: Evaluation & Analysis (Future Tasks)

**UC2.1: Batch Inference (6-8h)**
- Evaluate model on 1000+ episodes
- Benchmark different hyperparameters
- Statistical significance testing

**UC2.2: Model Comparison (4-6h)**
- Compare checkpoints across training
- Learning curves on test set
- Overfitting detection

**UC2.3: Curriculum Evaluation (3-4h)**
- Test performance at each stage
- Catastrophic forgetting detection
- Difficulty analysis

**Total: 13-18 hours** (separate tasks)

### Tier 3: Research & Advanced (Enabled by Model Abstraction)

**UC3.1: Transfer Learning**
- Fine-tune pretrained models
- Domain adaptation experiments
- Feature reuse analysis

**UC3.2: Ensemble Methods**
- Combine predictions from multiple models
- Variance reduction
- Uncertainty estimation

**UC3.3: Ablation Studies**
- Test architecture components (enabled by BRAIN_AS_CODE)
- Hyperparameter sensitivity
- Design principle validation

**UC3.4: Policy Visualization (8-12h)**
- Q-value heatmaps
- Trajectory plots
- Attention visualization (LSTM)

**Total: 8-12 hours** (mostly enabled by foundation)

### Tier 4: Production (Not Pedagogical)

**UC4.1: Model Serving API (40-60h)**
- REST/gRPC endpoint
- Batch inference
- Monitoring

**UC4.2: Model Export (6-10h)**
- ONNX, TorchScript
- Cross-framework deployment

**Total: 46-70 hours** (defer indefinitely)

---

## Effort Summary

### In Scope (Single PR)

| Task | Component | Hours | Priority |
|------|-----------|-------|----------|
| TASK-007 | Live viz streaming | 9 | HIGH |
| TASK-008 | Model abstraction | 30 | MEDIUM |
| **Total** | **Both tracks** | **39** | - |

### Out of Scope (Future)

| Tier | Description | Hours | When |
|------|-------------|-------|------|
| Tier 2 | Evaluation tools | 13-18 | After TASK-008 |
| Tier 3 | Research tools | 8-12 | As students request |
| Tier 4 | Production | 46-70 | Never (not pedagogical) |

---

## Task Dependencies

```
TASK-004 (BRAIN_AS_CODE)
    ↓
    Completed before TASK-007/008
    ↓
TASK-007 (Live Viz) ←──┐
    ↓                   │
TASK-008 (Model Abs)    │ Single PR (Option A)
    ↓                   │ or Sequential (Option B)
    Both complete  ←────┘
    ↓
Future: Tier 2 evaluation tools
    ↓
Future: Tier 3 research tools
```

---

## Success Metrics

### TASK-007: Live Viz

**Quantitative:**
- Latency: <1 second from episode completion to frontend display
- Memory: Queue bounded at ~1MB
- Throughput: No dropped episodes under normal load

**Qualitative:**
- Students can watch training in real-time
- Behavior matches actual training (not reconstructed)
- No checkpoint polling delay

### TASK-008: Model Abstraction

**Quantitative:**
- Checkpoint size: 5-10x smaller (~10MB vs ~50MB)
- Load time: <500ms for inference checkpoint
- API simplicity: 1 line to load, 1 line to predict

**Qualitative:**
- Students can export models to external projects
- No dependency on training infrastructure
- Clean, testable architecture

### Combined (Single PR)

**Integration:**
- Live viz uses HamletModel for Q-value display
- Both tracks work together seamlessly
- Backward compatible with old checkpoints

**Pedagogical Impact:**
- Students engage more with training (real-time feedback)
- Students build on trained models (research projects)
- Lower barrier to experimentation

---

## Risk Assessment

### Risk 1: Single PR Too Large

**Probability:** MEDIUM

**Impact:** HIGH (longer review, merge conflicts)

**Mitigation:**
- Start with Phase 1 (8h), validate complexity
- Can split into TASK-007 + TASK-008 if needed
- Use feature flags to isolate changes

### Risk 2: BRAIN_AS_CODE Integration Issues

**Probability:** LOW

**Impact:** MEDIUM (rework checkpoint format)

**Mitigation:**
- Review TASK-004 design before starting
- Coordinate on checkpoint format
- Add migration path if needed

### Risk 3: Hidden State Management Complexity

**Probability:** MEDIUM

**Impact:** MEDIUM (functional API is tricky)

**Mitigation:**
- Design functional API carefully (Phase 1)
- Provide StatefulModel wrapper if needed
- Extensive testing of hidden state

### Risk 4: Effort Underestimate

**Probability:** MEDIUM

**Impact:** MEDIUM (timeline slip)

**Mitigation:**
- 39 hours already includes safety buffer
- Phased implementation (can pause after each phase)
- Tier 2-4 use cases deferred (not in critical path)

---

## Recommendations

### Primary: Single PR (Option A)

**Implement both TASK-007 + TASK-008 in single stream:**
- Week 1-3: 39 hours total
- Phase 1: Model abstraction core (8h)
- Phase 2: Live viz streaming (9h)
- Phase 3: Inference checkpoints (4h)
- Phase 4: Integration (6h)
- Phase 5: Tests + docs (12h)

**Rationale:**
- Avoids rework (cleaner architecture)
- Both use cases benefit from shared foundation
- 39 hours is manageable for single PR
- Can pause/split if complexity too high

### Fallback: Sequential Tasks (Option B)

**If single PR proves too complex:**
- TASK-007 first (9h, quick win)
- TASK-008 second (30h, foundation)
- Accept ~4h rework (LiveInferenceServer refactored twice)

### Defer: Tier 2-4 Use Cases

**Do NOT include in initial implementation:**
- Evaluation tools (Tier 2) - add later as needed
- Research tools (Tier 3) - enabled by foundation
- Production tools (Tier 4) - not pedagogical

---

## Next Steps (Moving to PLAN Phase)

1. **Review Research Documents:**
   - ✅ RESEARCH-INFERENCE-ARCHITECTURE.md
   - ✅ RESEARCH-LIVE-TRAINING-TELEMETRY.md
   - ✅ RESEARCH-INFERENCE-ARCHITECTURE-SYNTHESIS.md
   - ✅ RESEARCH-EXTENDED-USE-CASES.md
   - ✅ RESEARCH-SUMMARY-TASKS-007-008.md (this document)

2. **Coordinate with BRAIN_AS_CODE (TASK-004):**
   - Review brain.yaml schema
   - Confirm compile_brain() API
   - Validate checkpoint format design

3. **Create TASK Documents:**
   - TASK-007-LIVE-TRAINING-VIZ.md (9h)
   - TASK-008-MODEL-ABSTRACTION.md (30h)
   - Combined TASK-007-008.md if single PR

4. **Implementation:**
   - Start with Phase 1 (Model abstraction core)
   - Validate complexity before committing to single PR
   - Can split if needed

---

## Questions Before Moving to PLAN Phase

1. **Single PR Confirmation:** Still comfortable with 39h single PR? Or prefer sequential (TASK-007 then TASK-008)?

2. **Tier 2 Use Cases:** Any evaluation tools (UC2.1-2.3) feel essential for initial release?

3. **BRAIN_AS_CODE Review:** Should I read TASK-004 design docs before creating TASK-007/008?

4. **Documentation Scope:** What examples in initial docs?
   - Watch training live ✅
   - Export and use model ✅
   - Compare checkpoints (Tier 2)
   - Transfer learning (Tier 3)

5. **Testing Coverage:** What level for single PR?
   - Unit tests (model, streaming) ✅
   - Integration tests (training → viz) ✅
   - End-to-end (full workflow)

Ready to create TASK-007-008 document when you are!
