# Research: Extended Use Cases for Model Abstraction

## Context

**Given:**
- BRAIN_AS_CODE (TASK-004) will be complete before this task
- Model architecture defined in `brain.yaml`
- Universe compiler generates model from spec
- This task implements BOTH Track 1 (live viz) + Track 2 (model abstraction) in single PR

**Question:** What other use cases benefit from model abstraction beyond basic inference?

---

## Use Case Taxonomy

### Tier 1: Core Use Cases (IN SCOPE - Single PR)

These are essential for the pedagogical mission and should be included in the initial implementation.

#### UC1.1: Live Training Visualization ⭐ (PRIORITY)
**Goal:** Show students "what the agent is learning RIGHT NOW"

**How it works:**
```python
# Training thread
episode_data = run_training_episode()
live_callback(episode_data)  # → Queue → Viz server

# Visualization server
episode = await live_queue.get()
await broadcast_to_frontend(episode)
```

**Pedagogical Value:** HIGH
- Students see learning in real-time
- Can observe exploration → exploitation transition
- Watch curriculum stage progressions
- Identify interesting behaviors (reward hacking, etc.)

**Enabled by:**
- Episode streaming (callback + queue)
- Model abstraction (for Q-value visualization)

**BRAIN_AS_CODE Integration:**
- Viz can show network architecture diagram (loaded from brain.yaml)
- Display which features the network is attending to

---

#### UC1.2: Export Trained Models for Student Projects ⭐
**Goal:** Students extract models and use in their own code

**How it works:**
```python
# After training completes
# Student exports model
model = HamletModel.load_from_checkpoint("model_ep10000.pt")

# Use in custom environment
class CustomEnvironment:
    """Student's own environment (e.g., different grid, different meters)."""
    pass

env = CustomEnvironment()
obs = env.reset()

while not done:
    result = model.predict(obs, epsilon=0.0)  # Greedy policy
    action = result["actions"][0]
    obs, reward, done, info = env.step(action)
```

**Pedagogical Value:** HIGH
- **Transfer Learning:** "Can agent trained in 8×8 grid work in 16×16?"
- **Domain Adaptation:** "Can HAMLET agent play Atari?"
- **Policy Analysis:** "What does the agent do in novel situations?"
- **Research Projects:** Students can build on trained policies

**Example Student Projects:**
1. **"Does my agent generalize?"**
   - Train on 8×8 grid with 14 affordances
   - Test on 12×12 grid with 20 affordances
   - Measure performance degradation

2. **"Can I fine-tune for a new task?"**
   - Load pretrained model
   - Freeze early layers
   - Fine-tune Q-head on new task (e.g., resource collection)

3. **"What features did the agent learn?"**
   - Load model
   - Run inference on hand-crafted scenarios
   - Visualize Q-values, hidden states (LSTM), attention

**Enabled by:**
- HamletModel class (standalone, no training deps)
- Inference checkpoints (small, portable)
- Clean API: `predict(obs, hidden, epsilon)`

**BRAIN_AS_CODE Integration:**
- Model checkpoint includes `brain.yaml` reference
- Student can see architecture spec
- Can modify brain.yaml and retrain from scratch

---

### Tier 2: Evaluation & Analysis (OUT OF SCOPE - Future Extensions)

These are valuable but can be added later as separate tasks.

#### UC2.1: Batch Inference for Evaluation
**Goal:** Run trained model over large test set

**How it works:**
```python
# Evaluate model on 1000 episodes
model = HamletModel.load_from_checkpoint("model_ep10000.pt")

results = []
for episode in range(1000):
    obs = env.reset()
    hidden = None
    total_reward = 0

    while not done:
        # Batch prediction (multiple envs in parallel)
        result = model.predict(obs, hidden, epsilon=0.0)
        action, hidden = result["actions"], result["hidden"]

        obs, reward, done, info = env.step(action)
        total_reward += reward

    results.append({
        "episode": episode,
        "total_reward": total_reward,
        "survival_steps": info["step_count"],
    })

# Analyze results
mean_reward = np.mean([r["total_reward"] for r in results])
mean_survival = np.mean([r["survival_steps"] for r in results])
```

**Pedagogical Value:** MEDIUM
- **Benchmarking:** Compare models trained with different hyperparameters
- **Statistical Rigor:** "Is this improvement significant?"
- **Curriculum Validation:** "Did agent actually learn each stage?"

**Example Use:**
```python
# Compare epsilon-greedy vs RND exploration
model_eg = HamletModel.load_from_checkpoint("model_eg_ep10000.pt")
model_rnd = HamletModel.load_from_checkpoint("model_rnd_ep10000.pt")

results_eg = evaluate(model_eg, n_episodes=1000)
results_rnd = evaluate(model_rnd, n_episodes=1000)

# Statistical test
p_value = scipy.stats.ttest_ind(results_eg, results_rnd).pvalue
```

**Enabled by:**
- HamletModel.predict() supports batching
- Lightweight checkpoints (fast loading)

**BRAIN_AS_CODE Integration:**
- Can evaluate models with different brain.yaml architectures
- "Does CNN encoder outperform MLP for POMDP?"

**Implementation Effort:** 6-8 hours (separate task)

---

#### UC2.2: Model Comparison Tools
**Goal:** Compare multiple checkpoints across training run

**How it works:**
```python
# Load checkpoints from different episodes
checkpoints = [100, 500, 1000, 2000, 5000, 10000]
models = [HamletModel.load_from_checkpoint(f"model_ep{ep:05d}.pt") for ep in checkpoints]

# Evaluate each on fixed test set
test_scenarios = load_test_scenarios()  # Hand-crafted scenarios

results = []
for ep, model in zip(checkpoints, models):
    survival_rates = []
    for scenario in test_scenarios:
        obs = scenario.reset()
        done = False
        steps = 0

        while not done and steps < 500:
            result = model.predict(obs, epsilon=0.0)
            obs, _, done, _ = scenario.step(result["actions"])
            steps += 1

        survival_rates.append(steps)

    results.append({
        "episode": ep,
        "mean_survival": np.mean(survival_rates),
        "scenario_results": survival_rates,
    })

# Plot learning curve
plt.plot([r["episode"] for r in results], [r["mean_survival"] for r in results])
plt.xlabel("Training Episode")
plt.ylabel("Test Survival (steps)")
plt.title("Learning Curve on Test Scenarios")
```

**Pedagogical Value:** MEDIUM
- **Learning Curves:** Visualize improvement over training
- **Overfitting Detection:** Test set performance vs training performance
- **Checkpoint Selection:** "Which checkpoint generalizes best?"

**Example Student Project:**
"Plot how agent's behavior changes across curriculum stages"

**Implementation Effort:** 4-6 hours (separate task)

---

#### UC2.3: Curriculum Stage Evaluation
**Goal:** Evaluate model's performance at each curriculum stage

**How it works:**
```python
# Load final model
model = HamletModel.load_from_checkpoint("model_ep10000.pt")

# Evaluate at each curriculum stage (different difficulty multipliers)
stages = [
    {"stage": 1, "multiplier": 5.0, "max_steps": 50},
    {"stage": 2, "multiplier": 3.5, "max_steps": 100},
    {"stage": 3, "multiplier": 2.5, "max_steps": 200},
    {"stage": 4, "multiplier": 1.75, "max_steps": 350},
    {"stage": 5, "multiplier": 1.0, "max_steps": 500},
]

for stage_config in stages:
    env = VectorizedHamletEnv(
        depletion_multiplier=stage_config["multiplier"],
        max_steps=stage_config["max_steps"],
    )

    results = evaluate(model, env, n_episodes=100)

    print(f"Stage {stage_config['stage']}: "
          f"Survival {results['mean_survival']:.1f} / {stage_config['max_steps']}")
```

**Pedagogical Value:** HIGH
- **Curriculum Validation:** "Did agent master early stages?"
- **Difficulty Analysis:** "Which stage is hardest?"
- **Catastrophic Forgetting:** "Does agent forget early stages?"

**Example Teaching Moment:**
"Agent survives 480/500 steps at Stage 5 (hardest), but only 45/50 at Stage 1 (easiest). What happened?"

**Answer:** Catastrophic forgetting - agent optimized for hard stages, forgot how to handle easy stages.

**Implementation Effort:** 3-4 hours (separate task)

---

### Tier 3: Research & Advanced (OUT OF SCOPE - Future Research)

These are for advanced students and research projects.

#### UC3.1: Transfer Learning
**Goal:** Fine-tune pretrained model on new task

**How it works:**
```python
# Load pretrained model
base_model = HamletModel.load_from_checkpoint("model_ep10000.pt")

# Freeze early layers (feature extractors)
for param in base_model.network.vision_encoder.parameters():
    param.requires_grad = False
for param in base_model.network.position_encoder.parameters():
    param.requires_grad = False
for param in base_model.network.meter_encoder.parameters():
    param.requires_grad = False

# Only train Q-head on new task
optimizer = torch.optim.Adam(base_model.network.q_head.parameters(), lr=0.001)

# Train on new environment (e.g., different meter dynamics)
new_env = VectorizedHamletEnv(config_pack_path="configs/transfer_task")
train_on_new_task(base_model, new_env, optimizer, episodes=1000)
```

**Pedagogical Value:** HIGH (for advanced students)
- **Feature Reuse:** "What features transfer across tasks?"
- **Sample Efficiency:** "Can we learn faster with transfer?"
- **Domain Adaptation:** "Does indoor agent work outdoors?"

**Example Research Questions:**
1. "Train on 8×8 grid, transfer to 16×16. How many episodes to adapt?"
2. "Train on Bed+Hospital, transfer to full affordances. Performance?"
3. "Train on full observability, transfer to POMDP. Does it work?"

**Implementation Effort:** Model abstraction enables this (no extra code needed!)

---

#### UC3.2: Ensemble Methods
**Goal:** Combine predictions from multiple models

**How it works:**
```python
# Load multiple models (trained with different seeds)
models = [
    HamletModel.load_from_checkpoint(f"model_seed{seed}_ep10000.pt")
    for seed in [42, 123, 456, 789, 1011]
]

# Ensemble prediction (average Q-values)
def ensemble_predict(obs, models, epsilon=0.0):
    q_values_list = []

    for model in models:
        result = model.predict(obs, epsilon=0.0)  # Greedy for all models
        q_values_list.append(result["q_values"])

    # Average Q-values
    avg_q_values = torch.mean(torch.stack(q_values_list), dim=0)

    # Select action from averaged Q-values
    if random.random() < epsilon:
        actions = torch.randint(0, avg_q_values.shape[1], (avg_q_values.shape[0],))
    else:
        actions = torch.argmax(avg_q_values, dim=1)

    return {
        "actions": actions,
        "q_values": avg_q_values,
    }
```

**Pedagogical Value:** MEDIUM
- **Variance Reduction:** "Does ensemble improve stability?"
- **Uncertainty Estimation:** "When do models disagree?"
- **Robustness:** "Does ensemble generalize better?"

**Example Student Project:**
"Train 5 models with different random seeds. Compare single-model vs ensemble performance."

**Implementation Effort:** Model abstraction enables this (no extra code needed!)

---

#### UC3.3: Ablation Studies
**Goal:** Test effect of architecture components

**With BRAIN_AS_CODE (TASK-004 complete):**

```yaml
# brain_baseline.yaml
network:
  type: recurrent_spatial
  vision_encoder:
    type: cnn
    filters: [16, 32]
  lstm:
    hidden_dim: 256

# brain_no_vision.yaml (ablation)
network:
  type: recurrent_spatial
  vision_encoder:
    type: none  # Remove vision encoder
  lstm:
    hidden_dim: 256

# brain_larger_lstm.yaml (ablation)
network:
  type: recurrent_spatial
  vision_encoder:
    type: cnn
    filters: [16, 32]
  lstm:
    hidden_dim: 512  # 2x larger
```

**Train and compare:**
```python
# Train each architecture
configs = ["brain_baseline.yaml", "brain_no_vision.yaml", "brain_larger_lstm.yaml"]

for config in configs:
    runner = DemoRunner(config_dir=f"configs/ablation/{config}")
    runner.run(max_episodes=10000)

# Evaluate each
results = {}
for config in configs:
    model = HamletModel.load_from_checkpoint(f"model_{config}_ep10000.pt")
    results[config] = evaluate(model, n_episodes=1000)

# Compare
for config, result in results.items():
    print(f"{config}: Survival {result['mean_survival']:.1f}")
```

**Pedagogical Value:** HIGH
- **Architecture Understanding:** "Does CNN help for partial observability?"
- **Hyperparameter Tuning:** "Is bigger always better?"
- **Design Principles:** "Which components matter most?"

**Example Research Questions:**
1. "Does vision encoder help if we have full observability?"
2. "Is LSTM necessary for full observability?"
3. "What's the minimum network size that works?"

**Implementation Effort:** Enabled by BRAIN_AS_CODE + Model abstraction (no extra code!)

---

#### UC3.4: Policy Visualization Tools
**Goal:** Visualize what the agent has learned

**How it works:**
```python
# Load trained model
model = HamletModel.load_from_checkpoint("model_ep10000.pt")

# Generate Q-value heatmap for grid positions
q_value_map = np.zeros((grid_size, grid_size, action_dim))

for x in range(grid_size):
    for y in range(grid_size):
        # Create observation with agent at (x, y)
        obs = create_observation(position=(x, y), meters=default_meters)

        # Get Q-values
        result = model.predict(obs)
        q_value_map[x, y, :] = result["q_values"][0].cpu().numpy()

# Plot Q-value heatmap for "INTERACT" action
plt.imshow(q_value_map[:, :, 4], cmap="viridis")
plt.colorbar(label="Q-value for INTERACT")
plt.title("Where does agent want to interact?")

# Overlay affordance positions
for name, pos in affordances.items():
    plt.scatter(pos[0], pos[1], marker="x", color="red", s=100, label=name)
```

**Pedagogical Value:** HIGH
- **Policy Understanding:** "What did the agent learn?"
- **Spatial Strategy:** "Where does agent prefer to go?"
- **Action Preferences:** "When does agent interact vs move?"

**Example Visualizations:**
1. **Q-value heatmaps:** For each action, show Q-value at each position
2. **Trajectory plots:** Show agent's typical paths
3. **Meter response curves:** How do Q-values change with meter values?
4. **Attention maps (LSTM):** What parts of observation does agent attend to?

**Implementation Effort:** 8-12 hours (separate visualization task)

---

### Tier 4: Production Deployment (OUT OF SCOPE - Not Pedagogical)

These are production use cases, not relevant for pedagogical project.

#### UC4.1: Model Serving API
**Goal:** REST/gRPC endpoint for remote inference

**Why NOT in scope:**
- Pedagogical project, not production system
- Students don't need remote inference
- Adds complexity without educational value

**If needed later:** 40-60 hours (see RESEARCH-INFERENCE-ARCHITECTURE.md Option C)

---

#### UC4.2: Model Export (ONNX, TorchScript)
**Goal:** Export model for deployment in other frameworks

**Why NOT in scope:**
- Students use PyTorch (no need for ONNX)
- Adds complexity for minimal benefit
- Can be added later if needed

**If needed later:** 6-10 hours

---

## How BRAIN_AS_CODE Integration Affects Design

Since TASK-004 (BRAIN_AS_CODE) will be complete before this task, the model abstraction design changes:

### Before BRAIN_AS_CODE (Original Design)

```python
# Model checkpoint includes hardcoded params
checkpoint = {
    "network_type": "recurrent",
    "network_params": {
        "action_dim": 5,
        "window_size": 5,
        "num_meters": 8,
        # ... hardcoded architecture
    },
    "state_dict": {...},
}

# Load model
model = HamletModel(
    network_type=checkpoint["network_type"],
    network_params=checkpoint["network_params"],
)
model.network.load_state_dict(checkpoint["state_dict"])
```

### After BRAIN_AS_CODE (New Design)

```python
# Model checkpoint references brain.yaml
checkpoint = {
    "format_version": "inference_v1",
    "brain_spec_path": "configs/L2_partial_observability/brain.yaml",
    "brain_spec_hash": "sha256:abc123...",  # Verify spec didn't change
    "state_dict": {...},
    "metadata": {...},
}

# Load model (compiles from spec)
from townlet.compiler import compile_brain

# Load brain spec from checkpoint
brain_spec = load_brain_spec(checkpoint["brain_spec_path"])

# Compile network from spec (BRAIN_AS_CODE)
network = compile_brain(brain_spec)

# Load weights
model = HamletModel(network=network)
model.network.load_state_dict(checkpoint["state_dict"])
```

### Benefits of BRAIN_AS_CODE Integration

1. **Architecture Portability:** brain.yaml is human-readable, can edit and retrain
2. **Versioning:** brain_spec_hash ensures checkpoint matches architecture
3. **Experimentation:** Students can modify brain.yaml without changing code
4. **Ablations:** Easy to test architecture variants (see UC3.3)

### Design Changes for This Task

**Inference checkpoint format should include:**
```python
{
    "format_version": "inference_v1",
    "brain_spec": {
        # OPTION A: Embed full brain.yaml content
        "network": {...},  # Full spec embedded
    },
    # OR
    # OPTION B: Reference external brain.yaml
    "brain_spec_path": "configs/L2_partial_observability/brain.yaml",
    "brain_spec_hash": "sha256:abc123...",

    "state_dict": {...},
    "metadata": {...},
}
```

**Recommendation:** OPTION A (embed spec)
- Self-contained checkpoint
- No external file dependencies
- Can load model anywhere

---

## Scope for Single PR Implementation

### IN SCOPE (Must Have)

#### Track 1: Live Viz (9 hours)
- [x] Episode streaming (callback + queue)
- [x] LiveInferenceServer consumes queue
- [x] WebSocket broadcast to frontend
- [x] <1s latency from training to frontend

#### Track 2: Model Abstraction (24-30 hours)
- [x] HamletModel class
- [x] predict() method (functional API)
- [x] Inference checkpoint format (with BRAIN_AS_CODE integration)
- [x] Load from training checkpoints (backward compat)
- [x] Documentation + examples

**Total: 33-39 hours**

### OUT OF SCOPE (Future Tasks)

#### Tier 2: Evaluation & Analysis
- [ ] Batch inference tools (UC2.1) - 6-8h
- [ ] Model comparison tools (UC2.2) - 4-6h
- [ ] Curriculum evaluation (UC2.3) - 3-4h

#### Tier 3: Research & Advanced
- [ ] Policy visualization tools (UC3.4) - 8-12h
- [ ] Transfer learning examples (UC3.1) - enabled by model abstraction
- [ ] Ensemble examples (UC3.2) - enabled by model abstraction
- [ ] Ablation examples (UC3.3) - enabled by BRAIN_AS_CODE

#### Tier 4: Production
- [ ] Model serving API (UC4.1) - 40-60h
- [ ] ONNX export (UC4.2) - 6-10h

**Rationale:** Focus on core pedagogical value (live viz + model export). Advanced use cases can be added later as students request them.

---

## Single PR Implementation Strategy

Since you're locking the codebase for a single PR, here's the implementation order:

### Phase 1: Foundation (Week 1)

**Model Abstraction Core** (16 hours)
1. Create `src/townlet/inference/model.py`
2. HamletModel class with predict()
3. Load from training checkpoints
4. Unit tests

**Why first:** Track 2 depends on this, Track 1 can use it

---

### Phase 2: Live Viz (Week 2)

**Episode Streaming** (9 hours)
1. Add callback to EpisodeRecorder
2. Wire queue in UnifiedServer
3. Consume in LiveInferenceServer
4. Tests

**Why second:** Can now use HamletModel for Q-value viz

---

### Phase 3: Integration (Week 3)

**Bring It Together** (14 hours)
1. Inference checkpoint format (with BRAIN_AS_CODE spec)
2. Migration script for old checkpoints
3. Refactor LiveInferenceServer to use HamletModel
4. Documentation + examples
5. Integration tests

**Why last:** Both tracks ready, now polish

---

### Total Timeline

**3 weeks, 39 hours**
- Week 1: Model abstraction foundation (16h)
- Week 2: Live viz + streaming (9h)
- Week 3: Integration + polish (14h)

**Single PR:** All changes together, no broken intermediate states

---

## Example Student Workflows Enabled

### Workflow 1: Watch Training Live
```bash
# Terminal 1: Start training with live viz
python scripts/run_demo.py --config configs/L2_partial_observability --episodes 10000

# Terminal 2: Start frontend
cd frontend && npm run dev

# Browser: http://localhost:5173
# See training episodes in real-time, <1s latency
```

### Workflow 2: Export and Use Model
```python
# After training completes
from townlet.inference import HamletModel

# Load trained model
model = HamletModel.load_from_checkpoint("runs/L2_partial_observability/2025-11-05_123456/checkpoints/model_ep10000.pt")

# Use in custom environment
import gymnasium as gym
env = gym.make("MyCustomEnv-v0")

obs, info = env.reset()
done = False
hidden = None

while not done:
    result = model.predict(obs, hidden, epsilon=0.0)
    action = result["actions"][0]
    hidden = result["hidden"]

    obs, reward, done, truncated, info = env.step(action)
```

### Workflow 3: Compare Models (Future)
```python
# Evaluate multiple checkpoints
from townlet.evaluation import evaluate_checkpoint

checkpoints = [1000, 2000, 5000, 10000]
results = []

for ep in checkpoints:
    model = HamletModel.load_from_checkpoint(f"model_ep{ep:05d}.pt")
    metrics = evaluate_checkpoint(model, n_episodes=100)
    results.append({"episode": ep, **metrics})

# Plot learning curve
import matplotlib.pyplot as plt
plt.plot([r["episode"] for r in results], [r["mean_survival"] for r in results])
```

---

## Summary

### Use Cases Taxonomy

**Tier 1 (IN SCOPE):**
- UC1.1: Live training visualization (priority)
- UC1.2: Export models for student projects

**Tier 2 (OUT OF SCOPE - Future):**
- UC2.1: Batch inference for evaluation
- UC2.2: Model comparison tools
- UC2.3: Curriculum stage evaluation

**Tier 3 (OUT OF SCOPE - Research):**
- UC3.1: Transfer learning
- UC3.2: Ensemble methods
- UC3.3: Ablation studies
- UC3.4: Policy visualization

**Tier 4 (OUT OF SCOPE - Production):**
- UC4.1: Model serving API
- UC4.2: Model export (ONNX)

### BRAIN_AS_CODE Integration

Since TASK-004 completes first:
- Inference checkpoints embed brain.yaml spec
- Model compiled from spec (not hardcoded params)
- Enables easy ablation studies and architecture experiments
- Students can edit brain.yaml and retrain

### Single PR Strategy

**3 weeks, 39 hours:**
1. Week 1: Model abstraction (16h)
2. Week 2: Live viz streaming (9h)
3. Week 3: Integration + polish (14h)

**Focus:** Tier 1 use cases (pedagogical value)
**Defer:** Tier 2-4 (can add later as needed)

---

## Questions for Next Steps

1. **Tier 2 Use Cases:** Any of these feel essential for initial release?
   - Batch evaluation?
   - Model comparison tools?

2. **BRAIN_AS_CODE Coordination:** Should I review TASK-004 design to ensure compatibility?

3. **Documentation:** What examples should be in initial docs?
   - "How to watch training live"
   - "How to export and use model"
   - "How to compare checkpoints"
   - ???

4. **Testing:** What level of coverage for single PR?
   - Unit tests only?
   - Integration tests?
   - End-to-end (training → viz → export)?

Ready to create TASK document when you are!
