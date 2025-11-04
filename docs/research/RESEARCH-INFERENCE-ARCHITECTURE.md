# Research: Inference Architecture and Best Practices

## Problem Statement

The current training and inference architecture in `src/townlet/demo/` has evolved organically but exhibits several antipatterns that hinder maintainability, testing, and future development:

**Core Issues:**
1. **Tight Coupling**: Inference server recreates the entire training stack (environment, population, curriculum, exploration) just to run inference
2. **No Model Abstraction**: There's no standalone "Model" class that encapsulates network + preprocessing + action selection
3. **Complex Checkpoint Format**: Checkpoints mix training state (optimizer, replay buffer, curriculum) with model weights, making them unsuitable for pure inference
4. **Hidden State Management Chaos**: LSTM hidden state is managed in 3+ places (RecurrentSpatialQNetwork, VectorizedPopulation, episode containers)
5. **Mixed Concerns**: LiveInferenceServer (1075 lines) handles model loading, episode execution, WebSocket comms, and replay - violating single responsibility
6. **No Batch Inference**: Inference runs single episodes sequentially, no support for batch prediction
7. **Config Duplication**: Training config is duplicated across checkpoint dict, config_snapshot directory, and re-read by inference server

**Why This Matters:**
- **Pedagogical Value**: Students want to extract trained models and use them in other projects
- **Testing**: Hard to unit test model behavior independently of training infrastructure
- **BRAIN_AS_CODE**: Future work will need clean model abstraction to compile architectures from YAML
- **Production Use**: No clear path from research code to deployment

## Current Architecture Analysis

### Component Breakdown

**src/townlet/demo/**
```
runner.py (750 lines)
├── DemoRunner: Training orchestrator
├── Creates: env, population, curriculum, exploration
├── Checkpoint: Full training state (v2 format)
└── Entry point: python -m townlet.demo.runner

unified_server.py (529 lines)
├── UnifiedServer: Multi-threaded coordinator
├── Training thread: runs DemoRunner
├── Inference thread: runs LiveInferenceServer
└── Frontend subprocess: npm run dev (optional)

live_inference.py (1075 lines)
├── LiveInferenceServer: WebSocket endpoint
├── Recreates: env, population, curriculum, exploration
├── Loads: checkpoint["population_state"]["q_network"]
├── Modes: inference (live) + replay (recorded episodes)
└── Mixed: model logic + comms + episode execution + replay

database.py
└── DemoDatabase: SQLite operations for metrics
```

**src/townlet/agent/**
```
networks.py (245 lines)
├── SimpleQNetwork: MLP for full observability
├── RecurrentSpatialQNetwork: CNN+LSTM for POMDP
└── Hidden state management: self.hidden_state (recurrent only)
```

**src/townlet/population/**
```
vectorized.py (900+ lines)
├── VectorizedPopulation: Training coordinator
├── Creates: q_network, target_network, optimizer
├── Manages: replay_buffer, episode_hidden_states
└── Couples: training logic + model management
```

### Checkpoint Format (v2)

```python
checkpoint = {
    "version": 2,
    "episode": 5000,
    "timestamp": 1699123456.78,

    # Population state (mixed training + model)
    "population_state": {
        "q_network": {...},           # Model weights (WANT)
        "optimizer": {...},           # Training state (DON'T WANT for inference)
        "replay_buffer": {...},       # Training state (DON'T WANT)
        "episode_hidden_states": {...},  # Transient state
    },

    # Curriculum state (training-specific)
    "curriculum_state": {
        "agent_stages": [...],
        "performance_trackers": {...},
    },

    # Environment state
    "affordance_layout": {...},       # Useful for inference
    "agent_ids": [...],               # Metadata

    # Config snapshot
    "training_config": {...},         # Full training.yaml (redundant with config_snapshot/)
    "config_dir": "configs/L2_partial_observability",

    # Inference metadata
    "epsilon": 0.15,                  # Useful for visualization
}
```

**Problems:**
- ~80% of checkpoint is training state not needed for inference
- No versioning for model architecture (network_type hidden in training_config)
- Checkpoint loading requires recreating full training stack
- Can't load model weights without VectorizedPopulation class

### Hidden State Management (Recurrent Networks)

**Problem**: LSTM hidden state is managed in 3 different places with unclear ownership:

1. **RecurrentSpatialQNetwork.hidden_state**
   - Instance variable: `self.hidden_state: tuple[torch.Tensor, torch.Tensor] | None`
   - Set by: `reset_hidden_state()`, `set_hidden_state()`
   - Used by: `forward()` when `hidden` arg is None

2. **VectorizedPopulation.episode_hidden_states**
   - Per-agent tracking: `dict[int, tuple[torch.Tensor, torch.Tensor]]`
   - Updated: After each step, stored in episode container
   - Reset: At episode start

3. **Episode Containers (Recurrent)**
   - Stores: All hidden states for sequence replay
   - Used by: SequentialReplayBuffer during batch training

**Antipattern**: Stateful network makes it hard to:
- Run batch inference with different hidden states per example
- Test network in isolation
- Use network in non-sequential contexts

**Best Practice**: Networks should be pure functions:
```python
# Good (functional)
q_values, new_hidden = network(obs, hidden)

# Bad (stateful)
network.reset_hidden_state()
q_values = network(obs)  # Implicitly uses self.hidden_state
```

## Design Space

### Option A: Extract Model Abstraction Layer ("Model as First-Class Citizen")

**Create a standalone Model class** that encapsulates everything needed for inference:

```python
class HamletModel:
    """
    Self-contained model for HAMLET inference.

    Encapsulates:
    - Network architecture (SimpleQNetwork or RecurrentSpatialQNetwork)
    - Preprocessing (observation normalization, if needed)
    - Action selection (greedy, epsilon-greedy, sampling)
    - Hidden state management (for recurrent models)
    """

    def __init__(self, network_type: str, network_params: dict, device: torch.device):
        self.network_type = network_type
        self.device = device

        # Create network
        if network_type == "simple":
            self.network = SimpleQNetwork(**network_params).to(device)
        elif network_type == "recurrent":
            self.network = RecurrentSpatialQNetwork(**network_params).to(device)

        # Hidden state (recurrent only)
        self.is_recurrent = (network_type == "recurrent")

    def predict(
        self,
        obs: torch.Tensor,
        hidden: tuple | None = None,
        epsilon: float = 0.0,
    ) -> dict:
        """
        Pure inference function.

        Args:
            obs: [batch, obs_dim] observations
            hidden: Optional LSTM hidden state (recurrent only)
            epsilon: Exploration rate (0.0 = greedy)

        Returns:
            {
                "actions": [batch] selected actions,
                "q_values": [batch, action_dim] Q-values,
                "hidden": new hidden state (recurrent only),
            }
        """
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

    def save_inference_checkpoint(self, path: Path, metadata: dict):
        """
        Save inference-only checkpoint (weights + metadata).

        Format:
        {
            "model_type": "hamlet_q_network",
            "network_type": "recurrent",
            "network_params": {...},
            "state_dict": {...},
            "metadata": {
                "training_episode": 5000,
                "epsilon": 0.15,
                "training_config_path": "configs/L2_partial_observability",
            }
        }
        """
        checkpoint = {
            "model_type": "hamlet_q_network",
            "network_type": self.network_type,
            "network_params": self._get_network_params(),
            "state_dict": self.network.state_dict(),
            "metadata": metadata,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_from_checkpoint(cls, path: Path, device: torch.device) -> "HamletModel":
        """Load model from inference checkpoint."""
        checkpoint = torch.load(path, weights_only=False)

        model = cls(
            network_type=checkpoint["network_type"],
            network_params=checkpoint["network_params"],
            device=device,
        )
        model.network.load_state_dict(checkpoint["state_dict"])
        model.network.eval()

        return model

    @classmethod
    def load_from_training_checkpoint(cls, path: Path, device: torch.device) -> "HamletModel":
        """Load model from training checkpoint (v2 format)."""
        checkpoint = torch.load(path, weights_only=False)

        # Extract network config from training config
        training_config = checkpoint["training_config"]
        network_type = training_config["population"]["network_type"]

        # Infer network params from checkpoint
        network_params = cls._infer_network_params(checkpoint, network_type)

        model = cls(network_type=network_type, network_params=network_params, device=device)
        model.network.load_state_dict(checkpoint["population_state"]["q_network"])
        model.network.eval()

        return model
```

**New Inference Server Architecture:**

```python
class LiveInferenceServer:
    """Simplified inference server using HamletModel."""

    def __init__(self, checkpoint_dir: Path, config_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.config_dir = config_dir

        # Load environment config (for grid setup, affordance layout)
        self.env_config = self._load_env_config(config_dir)

        # Model (loaded from checkpoint)
        self.model: HamletModel | None = None

        # Environment (lightweight, for state transitions only)
        self.env = self._create_env(self.env_config)

        # WebSocket clients
        self.clients: set[WebSocket] = set()

    async def _load_latest_checkpoint(self):
        """Load latest checkpoint into model."""
        latest = self._find_latest_checkpoint()

        # Load model (no training infrastructure needed!)
        self.model = HamletModel.load_from_training_checkpoint(
            latest,
            device=self.device,
        )

        logger.info(f"Loaded model from {latest.name}")

    async def _run_episode(self):
        """Run episode using model."""
        self.env.reset()
        hidden = None

        for step in range(500):
            obs = self.env.get_observation()

            # Model inference (clean, no training code)
            result = self.model.predict(obs, hidden, epsilon=self.epsilon)

            actions = result["actions"]
            q_values = result["q_values"]
            hidden = result["hidden"]  # For recurrent models

            # Step environment
            next_obs, reward, done, info = self.env.step(actions)

            # Broadcast state
            await self._broadcast_state(obs, actions, q_values, reward)

            if done:
                break
```

**Benefits:**
- ✅ Clean separation: Model is standalone, no training dependencies
- ✅ Easy testing: `model.predict()` is a pure function
- ✅ Batch inference: Just pass batch of observations
- ✅ Model serving: Easy to wrap in REST/gRPC API
- ✅ BRAIN_AS_CODE ready: Model class can load from compiled spec
- ✅ Checkpoint migration: Supports both v2 training and new inference checkpoints

**Drawbacks:**
- ⚠️ Refactoring effort: Need to extract from VectorizedPopulation
- ⚠️ Backward compatibility: Must support old checkpoints
- ⚠️ Hidden state API: Need to design clean functional API for LSTM

**Effort Estimate:** 24-30 hours
- Phase 1: Extract HamletModel class (8h)
- Phase 2: Refactor LiveInferenceServer (6h)
- Phase 3: Inference checkpoint format (4h)
- Phase 4: Tests + documentation (6h)
- Phase 5: Migration path for old checkpoints (6h)

---

### Option B: Inference-Specific Checkpoints (Minimal Change)

**Keep current architecture** but save separate inference checkpoints:

```python
# In DemoRunner.save_checkpoint()
def save_checkpoint(self):
    # Existing: Full training checkpoint
    torch.save(training_checkpoint, checkpoint_dir / f"checkpoint_ep{ep:05d}.pt")

    # NEW: Inference checkpoint (weights only)
    inference_checkpoint = {
        "model_type": "hamlet_q_network",
        "network_type": self.config["population"]["network_type"],
        "network_params": self._extract_network_params(),
        "state_dict": self.population.q_network.state_dict(),
        "metadata": {
            "training_episode": self.current_episode,
            "epsilon": self.exploration.epsilon,
            "training_config_path": str(self.config_dir),
        }
    }
    torch.save(
        inference_checkpoint,
        checkpoint_dir / f"model_ep{ep:05d}.pt",  # Different filename
    )
```

**Benefits:**
- ✅ Minimal code changes (add save function)
- ✅ Faster checkpoint loading (no training state)
- ✅ Smaller checkpoint files (~10MB vs ~50MB)
- ✅ Clear intent (model_ep*.pt for inference, checkpoint_ep*.pt for training)

**Drawbacks:**
- ❌ Doesn't solve tight coupling issue
- ❌ Doesn't solve hidden state management
- ❌ Inference server still recreates training stack
- ❌ Checkpoint format duplication

**Effort Estimate:** 8-12 hours
- Phase 1: Define inference checkpoint format (2h)
- Phase 2: Add save logic to DemoRunner (2h)
- Phase 3: Update LiveInferenceServer to load inference checkpoints (3h)
- Phase 4: Tests (3h)

---

### Option C: Model Server Architecture (Production-Grade)

**Build a separate model server** with batch inference, API, and monitoring:

```python
# New: src/townlet/serving/

class ModelServer:
    """Production model server with batch inference."""

    def __init__(self, model_path: Path, batch_size: int = 32):
        self.model = HamletModel.load_from_checkpoint(model_path)
        self.batch_size = batch_size

        # Batching queue
        self.request_queue = asyncio.Queue()
        self.batch_processor = asyncio.create_task(self._process_batches())

    async def predict(self, obs: torch.Tensor, timeout: float = 1.0) -> dict:
        """Async prediction with automatic batching."""
        future = asyncio.Future()
        await self.request_queue.put((obs, future))
        return await asyncio.wait_for(future, timeout=timeout)

    async def _process_batches(self):
        """Background task: batch requests and run inference."""
        while True:
            batch = await self._collect_batch()

            # Batch inference
            obs_batch = torch.cat([req[0] for req in batch], dim=0)
            results = self.model.predict(obs_batch)

            # Distribute results
            for i, (obs, future) in enumerate(batch):
                future.set_result({
                    "action": results["actions"][i],
                    "q_values": results["q_values"][i],
                })

# API server (FastAPI)
app = FastAPI()
server = ModelServer("model_ep05000.pt")

@app.post("/predict")
async def predict(request: PredictRequest):
    obs = torch.tensor(request.observation)
    result = await server.predict(obs)
    return {
        "action": result["action"].item(),
        "q_values": result["q_values"].tolist(),
    }

# Metrics
@app.get("/metrics")
def metrics():
    return {
        "model_version": server.model_version,
        "requests_total": server.request_count,
        "latency_p50": server.latency_p50,
        "latency_p99": server.latency_p99,
    }
```

**Benefits:**
- ✅ Production-ready: Supports multiple clients, batch inference, monitoring
- ✅ Scalable: Can run multiple workers, GPU batching
- ✅ Language-agnostic: REST/gRPC API works from any language
- ✅ Model versioning: Can serve multiple models, A/B testing

**Drawbacks:**
- ❌ High complexity: Need API design, batching, monitoring
- ❌ Overkill for pedagogical project
- ❌ Doesn't align with "fuck around and find out" philosophy
- ❌ Students want Python code, not REST APIs

**Effort Estimate:** 40-60 hours
- Phase 1: Model server with batching (16h)
- Phase 2: REST API design (8h)
- Phase 3: Monitoring + metrics (8h)
- Phase 4: Client libraries (8h)
- Phase 5: Deployment docs (6h)
- Phase 6: Tests (14h)

---

### Option D: Wait for BRAIN_AS_CODE (Deferred)

**Defer architecture changes** until TASK-004 (BRAIN_AS_CODE) is complete:

**Rationale:**
- BRAIN_AS_CODE will define model architecture in YAML
- Universe compiler will generate model from spec
- Inference will use compiled model spec
- Checkpoint will reference compiled spec (not hardcoded architecture)

**Example (future):**
```yaml
# configs/L2_partial_observability/brain.yaml
network:
  type: recurrent_spatial
  vision_encoder:
    type: cnn
    filters: [16, 32]
    kernel_size: 3
  position_encoder:
    type: mlp
    hidden_dim: 32
  meter_encoder:
    type: mlp
    hidden_dim: 32
  lstm:
    input_dim: auto  # Computed from encoders
    hidden_dim: 256
  q_head:
    type: mlp
    hidden_dims: [128]
    output_dim: auto  # Computed from action space
```

**Benefits:**
- ✅ Aligns with UAC philosophy (everything configurable)
- ✅ Future-proof (model is just another config file)
- ✅ Enables experimentation (students edit brain.yaml, not networks.py)

**Drawbacks:**
- ❌ Doesn't solve immediate issues (tight coupling, checkpoint format)
- ❌ Blocked on TASK-004 (not yet started)
- ❌ Inference server still needs refactoring
- ❌ Effort estimate TBD (part of TASK-004, probably 40+ hours)

**Recommendation:** Don't wait. Implement Option A now, integrate with BRAIN_AS_CODE later.

---

## Additional Architectural Issues

### Issue 1: Mixed Concerns in LiveInferenceServer

**Current:** LiveInferenceServer (1075 lines) handles:
- Model loading and inference
- Episode execution (environment stepping)
- WebSocket communication
- Replay mode (loading recordings)
- Affordance tracking and stats
- Temporal mechanics visualization

**Best Practice:** Separate concerns via composition:

```python
# NEW: Clear separation

class InferenceEngine:
    """Pure inference logic (no I/O)."""
    def __init__(self, model: HamletModel, env: VectorizedHamletEnv):
        self.model = model
        self.env = env

    def run_episode(self, max_steps: int, epsilon: float) -> EpisodeResults:
        """Run episode, return structured results."""
        ...

class ReplayEngine:
    """Replay recorded episodes (no inference)."""
    def __init__(self, replay_manager: ReplayManager):
        self.replay_manager = replay_manager

    def get_step(self, step_index: int) -> StepData:
        """Get step data from recording."""
        ...

class WebSocketServer:
    """WebSocket communication (no inference logic)."""
    def __init__(self, inference_engine: InferenceEngine, replay_engine: ReplayEngine):
        self.inference = inference_engine
        self.replay = replay_engine

    async def handle_client(self, websocket: WebSocket):
        """Handle client commands, delegate to engines."""
        ...
```

**Benefits:**
- Each class has one responsibility
- Easy to test in isolation
- Can swap implementations (e.g., batch inference engine)

**Effort:** 12-16 hours to refactor

---

### Issue 2: Environment Coupling in Inference

**Current:** Inference server creates full VectorizedHamletEnv:
- Loads all affordance configs
- Initializes cascade engine
- Sets up meter dynamics
- Creates GPU tensors

**Problem:** Just need environment for:
- State transitions (position updates)
- Reward calculation
- Done checking

**Solution:** Lightweight "inference environment":

```python
class InferenceEnv:
    """Lightweight environment for inference (no training features)."""

    def __init__(self, config: EnvConfig, affordance_layout: dict):
        self.grid_size = config.grid_size
        self.affordances = affordance_layout

        # Minimal state
        self.position = None
        self.meters = None

    def reset(self):
        """Reset to initial state."""
        self.position = self._random_position()
        self.meters = torch.ones(8) * 0.5

    def step(self, action: int) -> tuple:
        """Step environment (minimal logic)."""
        # Update position
        new_pos = self._apply_action(self.position, action)

        # Update meters (simple depletion, no cascades)
        self.meters -= 0.01

        # Check done
        done = (self.meters[6] <= 0)  # Health meter

        return self._get_obs(), 1.0 if not done else 0.0, done, {}
```

**Benefits:**
- Faster reset (no full environment initialization)
- Smaller memory footprint
- No dependency on affordance/cascade configs

**Drawbacks:**
- Won't match training exactly (no cascades, simplified depletion)
- Only suitable for visualization, not accurate simulation

**Recommendation:** Keep full environment for now, revisit if performance issues

---

## Tradeoffs Analysis

| Option | Complexity | Effort | Value | Future-Proof | Pedagogical |
|--------|-----------|--------|-------|--------------|-------------|
| **A: Model Abstraction** | Medium | 24-30h | High | ✅ | ✅ |
| **B: Inference Checkpoints** | Low | 8-12h | Medium | ⚠️ | ✅ |
| **C: Model Server** | High | 40-60h | Low* | ✅ | ❌ |
| **D: Wait for BRAIN_AS_CODE** | TBD | TBD | High | ✅ | ✅ |

\* Low value for pedagogical project, high for production

**Key Insights:**
- **Option A** is the sweet spot: Good value, reasonable effort, future-proof
- **Option B** is quick win but doesn't solve architectural issues
- **Option C** is overkill for pedagogical project
- **Option D** shouldn't block near-term improvements

---

## Recommendation

**Primary: Implement Option A (Model Abstraction Layer)**

**Rationale:**
1. **Solves Core Issues**: Decouples inference from training, simplifies checkpoint format, clarifies hidden state management
2. **Enables Future Work**: Clean foundation for BRAIN_AS_CODE integration
3. **Pedagogical Value**: Students can extract models and use them independently
4. **Reasonable Effort**: 24-30 hours is manageable, well-scoped project

**Secondary: Option B as Intermediate Step**

If 24-30 hours feels too large, start with Option B (inference checkpoints) as Phase 1:
- Quick win (8-12h)
- Validates checkpoint format design
- Can incrementally refactor toward full Model abstraction

**Reject: Option C (Model Server)**

Too complex for pedagogical project, doesn't align with "fuck around and find out" philosophy.

**Defer: Option D (BRAIN_AS_CODE)**

Don't block on this. Implement Model abstraction now, integrate with BRAIN_AS_CODE later when TASK-004 is ready.

---

## Implementation Sketch (Option A)

### Phase 1: Extract HamletModel Class (8h)

**File:** `src/townlet/inference/model.py` (NEW)

```python
"""Standalone model for HAMLET inference."""

class HamletModel:
    """Self-contained inference model."""

    def __init__(self, network_type: str, network_params: dict, device: torch.device):
        """Initialize model with network architecture."""
        ...

    def predict(self, obs: torch.Tensor, hidden=None, epsilon: float = 0.0) -> dict:
        """Pure inference function (no side effects)."""
        ...

    @classmethod
    def load_from_training_checkpoint(cls, path: Path, device: torch.device) -> "HamletModel":
        """Load from v2 training checkpoint (backward compat)."""
        ...

    @classmethod
    def load_from_inference_checkpoint(cls, path: Path, device: torch.device) -> "HamletModel":
        """Load from inference checkpoint (new format)."""
        ...

    def save_inference_checkpoint(self, path: Path, metadata: dict):
        """Save inference checkpoint (weights + metadata only)."""
        ...
```

**Tests:**
- Load SimpleQNetwork from training checkpoint
- Load RecurrentSpatialQNetwork from training checkpoint
- Predict with batch of observations
- Hidden state management (recurrent)
- Save/load inference checkpoint round-trip

---

### Phase 2: Refactor LiveInferenceServer (6h)

**File:** `src/townlet/demo/live_inference.py` (REFACTOR)

**Before:**
```python
class LiveInferenceServer:
    def __init__(self, ...):
        self.env = VectorizedHamletEnv(...)  # Full training environment
        self.population = VectorizedPopulation(...)  # Full training stack
        self.curriculum = AdversarialCurriculum(...)
        self.exploration = AdaptiveIntrinsicExploration(...)

    async def _run_episode(self):
        actions = self.population.select_epsilon_greedy_actions(...)  # Hidden in population
```

**After:**
```python
class LiveInferenceServer:
    def __init__(self, ...):
        self.env = VectorizedHamletEnv(...)  # Keep environment (for state transitions)
        self.model: HamletModel | None = None  # Just the model

    async def _load_checkpoint(self):
        self.model = HamletModel.load_from_training_checkpoint(latest, self.device)

    async def _run_episode(self):
        obs = self.env.get_observation()
        result = self.model.predict(obs, hidden, epsilon=self.epsilon)  # Clean!
        actions = result["actions"]
        q_values = result["q_values"]
        hidden = result["hidden"]
```

**Benefits:**
- LiveInferenceServer drops from 1075 lines to ~600 lines
- No dependency on VectorizedPopulation, curriculum, exploration
- Clearer what's needed for inference

---

### Phase 3: Inference Checkpoint Format (4h)

**Add to DemoRunner.save_checkpoint():**

```python
def save_checkpoint(self):
    # Existing: Full training checkpoint (v2)
    training_checkpoint = {...}
    torch.save(training_checkpoint, checkpoint_dir / f"checkpoint_ep{ep:05d}.pt")

    # NEW: Inference checkpoint
    inference_checkpoint = {
        "format_version": "inference_v1",
        "model_type": "hamlet_q_network",
        "network_type": self.config["population"]["network_type"],
        "network_params": self._get_network_params(),
        "state_dict": self.population.q_network.state_dict(),
        "metadata": {
            "training_episode": self.current_episode,
            "training_config": str(self.config_dir),
            "epsilon": self.population._get_current_epsilon_value(),
            "curriculum_stage": self.curriculum.tracker.agent_stages[0].item(),
        },
        "created_at": time.time(),
    }
    torch.save(inference_checkpoint, checkpoint_dir / f"model_ep{ep:05d}.pt")

    logger.info(f"Saved inference checkpoint: model_ep{ep:05d}.pt")
```

**Checkpoint Size Comparison:**
- Training checkpoint: ~50MB (network + optimizer + replay buffer)
- Inference checkpoint: ~10MB (network only)

---

### Phase 4: Tests + Documentation (6h)

**Tests:**
- `tests/test_townlet/test_inference_model.py`:
  - Load SimpleQNetwork from training checkpoint
  - Load RecurrentSpatialQNetwork from training checkpoint
  - Batch prediction
  - Hidden state management
  - Save/load inference checkpoint round-trip
  - Backward compatibility (load old v2 checkpoints)

**Docs:**
- `docs/manual/INFERENCE_USAGE.md`:
  - How to load trained model
  - How to run inference
  - How to use model in custom scripts
  - Example: Extract model for external project

---

### Phase 5: Migration Path (6h)

**Problem:** Existing checkpoints are v2 training format, no inference checkpoints exist.

**Solution:** Add migration script:

```python
# scripts/migrate_checkpoints.py

def migrate_training_to_inference(checkpoint_path: Path) -> Path:
    """Convert v2 training checkpoint to inference checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Extract inference checkpoint
    inference_checkpoint = {
        "format_version": "inference_v1",
        "model_type": "hamlet_q_network",
        "network_type": checkpoint["training_config"]["population"]["network_type"],
        "network_params": extract_network_params(checkpoint),
        "state_dict": checkpoint["population_state"]["q_network"],
        "metadata": {
            "training_episode": checkpoint["episode"],
            "epsilon": checkpoint.get("epsilon", 0.0),
            "migrated_from": str(checkpoint_path),
        },
        "created_at": time.time(),
    }

    # Save as model_ep*.pt
    output_path = checkpoint_path.parent / f"model_ep{checkpoint['episode']:05d}.pt"
    torch.save(inference_checkpoint, output_path)

    return output_path

# Usage:
# python scripts/migrate_checkpoints.py runs/L2_partial_observability/2025-11-02_123456/checkpoints/
```

---

## Priority/Value

**Why This Matters:**

1. **Pedagogical (HIGH):** Students want to extract models and use them in other projects. Current architecture makes this nearly impossible without understanding full training stack.

2. **Architectural (HIGH):** Tight coupling prevents testing, refactoring, and future development. Model abstraction is foundational.

3. **BRAIN_AS_CODE Enabler (HIGH):** TASK-004 will need clean model abstraction. Doing this now unblocks future work.

4. **Performance (MEDIUM):** Smaller checkpoints, faster loading, but not a bottleneck currently.

**Leverage Analysis:**

- **High Leverage:** Model abstraction unlocks:
  - Independent testing (model, inference, training)
  - External usage (students' projects)
  - BRAIN_AS_CODE integration
  - Future optimizations (batch inference, quantization)

- **Moderate Effort:** 24-30 hours total, can be done in phases

- **Net Value:** Very high. This is foundational infrastructure that compounds.

---

## Estimated Effort

### Option A: Model Abstraction Layer

**Total: 24-30 hours**

| Phase | Description | Hours |
|-------|-------------|-------|
| 1 | Extract HamletModel class | 8 |
| 2 | Refactor LiveInferenceServer | 6 |
| 3 | Inference checkpoint format | 4 |
| 4 | Tests + documentation | 6 |
| 5 | Migration path for old checkpoints | 6 |

**Can be parallelized:**
- Phase 1 + 3 can be done concurrently
- Phase 2 depends on Phase 1
- Phase 4 depends on Phases 1-3
- Phase 5 is independent

**Incremental Delivery:**
- After Phase 1: HamletModel works, can be tested
- After Phase 2: Inference server uses new model
- After Phase 3: New checkpoints save both formats
- After Phase 4: Fully tested and documented
- After Phase 5: Old checkpoints migrated

---

## Risks & Mitigations

### Risk 1: Breaking Existing Checkpoints

**Risk:** Refactoring breaks ability to load v2 training checkpoints.

**Mitigation:**
- Keep VectorizedPopulation.load_checkpoint_state() working
- HamletModel.load_from_training_checkpoint() handles v2 format
- Add compatibility tests (load old checkpoint, verify weights match)

### Risk 2: Hidden State API Complexity

**Risk:** Functional hidden state API is awkward to use.

**Mitigation:**
- Provide helper wrapper for stateful usage:
  ```python
  class StatefulModel:
      """Wrapper for models that need hidden state across steps."""
      def __init__(self, model: HamletModel):
          self.model = model
          self.hidden = None

      def predict(self, obs, epsilon=0.0):
          result = self.model.predict(obs, self.hidden, epsilon)
          self.hidden = result["hidden"]
          return result

      def reset(self):
          self.hidden = None
  ```

### Risk 3: Effort Underestimate

**Risk:** 24-30 hours is too optimistic.

**Mitigation:**
- Start with Phase 1 only (HamletModel extraction)
- Validate effort estimate before committing to full refactor
- Can fall back to Option B (inference checkpoints only) if Phase 1 takes too long

### Risk 4: BRAIN_AS_CODE Conflicts

**Risk:** Model abstraction conflicts with future BRAIN_AS_CODE design.

**Mitigation:**
- Design HamletModel to load from dict (easy to populate from YAML)
- Network params are already dict-like, matches config structure
- Coordination: Review design with BRAIN_AS_CODE planning

---

## Dependencies

**Blocks:**
- None (can start immediately)

**Blocked By:**
- None

**Enables:**
- TASK-004 (BRAIN_AS_CODE) - provides clean model abstraction
- Future: Model serving, batch inference, quantization
- Future: Transfer learning (load pretrained model)

---

## Success Criteria

✅ **Phase 1 Complete:**
- [ ] HamletModel class exists
- [ ] Can load SimpleQNetwork from training checkpoint
- [ ] Can load RecurrentSpatialQNetwork from training checkpoint
- [ ] `predict()` method returns actions, q_values, hidden
- [ ] Unit tests pass

✅ **Phase 2 Complete:**
- [ ] LiveInferenceServer uses HamletModel
- [ ] No dependency on VectorizedPopulation, curriculum, exploration
- [ ] Inference runs without errors
- [ ] WebSocket communication works

✅ **Phase 3 Complete:**
- [ ] DemoRunner saves inference checkpoints
- [ ] Inference checkpoints are ~5-10x smaller than training checkpoints
- [ ] HamletModel.load_from_inference_checkpoint() works

✅ **Phase 4 Complete:**
- [ ] Tests pass (>90% coverage of HamletModel)
- [ ] docs/manual/INFERENCE_USAGE.md exists
- [ ] Example script demonstrates model loading

✅ **Phase 5 Complete:**
- [ ] Migration script converts v2 → inference_v1
- [ ] All existing checkpoints migrated
- [ ] Backward compatibility verified

---

## Alternative: Phased Rollout (Option B → Option A)

If 24-30 hours feels too risky, use Option B as Phase 0:

**Phase 0: Inference Checkpoints Only (8-12h)**
- Add inference checkpoint save to DemoRunner
- Update LiveInferenceServer to load inference checkpoints
- Keep existing architecture (VectorizedPopulation, etc.)

**Then Phase 1-5 of Option A:**
- Extract HamletModel class
- Refactor LiveInferenceServer to use HamletModel
- Continue from there

**Benefits:**
- Validates checkpoint format early
- Delivers value incrementally
- Lower initial commitment

**Drawbacks:**
- More total effort (8-12h + 24-30h = 32-42h)
- Potential rework if checkpoint format changes

---

## Summary

**Problem:** Tight coupling between training and inference, no model abstraction, complex checkpoint format.

**Recommendation:** Implement Option A (Model Abstraction Layer) in 5 phases over 24-30 hours.

**Key Benefits:**
- Decouples inference from training
- Enables external usage (students' projects)
- Prepares for BRAIN_AS_CODE
- Smaller checkpoints, faster loading
- Clean architecture, easy testing

**Risk:** Moderate refactoring effort, but well-scoped and incremental.

**Next Step:** Review this research, decide on Option A vs Option B-then-A, create TASK document.
