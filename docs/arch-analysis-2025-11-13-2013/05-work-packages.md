# HAMLET Architecture: Triaged Work Packages

**Generated From:** Architecture Analysis Report (2025-11-13)
**Status:** Planning Phase
**Total Packages:** 15 (4 High Priority, 7 Medium Priority, 4 Low Priority)

---

## Table of Contents

1. [Priority 0: Critical (Blocking)](#priority-0-critical-blocking)
2. [Priority 1: High (Quality/Extensibility)](#priority-1-high-qualityextensibility)
3. [Priority 2: Medium (Technical Debt)](#priority-2-medium-technical-debt)
4. [Priority 3: Low (Nice-to-Have)](#priority-3-low-nice-to-have)
5. [Appendix: Effort Estimation Guide](#appendix-effort-estimation-guide)

---

## Priority 0: Critical (Blocking)

**None identified.** System is production-ready for research and pedagogy.

---

## Priority 1: High (Quality/Extensibility)

### WP-H1: Frontend Derive Configuration from CompiledUniverse

**Issue:** Frontend components hardcode game-specific configuration (action names, meter relationships, heat map colors) instead of deriving from backend metadata.

**Impact:**
- Tight coupling to specific game configurations
- Config changes don't propagate to frontend automatically
- Limited reusability for custom curriculum packs

**Current State:**
```javascript
// AspatialView.vue line 93 - HARDCODED
const ACTION_NAMES = ['Up', 'Down', 'Left', 'Right', 'Interact', 'Wait']

// MeterPanel.vue lines 208-221 - HARDCODED
const TIER_PRIMARY = ['energy', 'health', 'money']
const TIER_SECONDARY = ['mood', 'satiation', 'fitness']
const TIER_TERTIARY = ['hygiene', 'social']

// Grid.vue lines 253-272 - HARDCODED
function getHeatColor(intensity) {
  if (intensity < 0.2) return '#3b82f6'  // blue
  if (intensity < 0.4) return '#06b6d4'  // cyan
  // ... etc
}
```

**Implementation Options:**

**Option A: REST API Metadata Endpoint (Recommended)**
```javascript
// Backend: Add metadata endpoint
@app.get('/api/universe/metadata')
def get_metadata():
    return {
        'action_space': compiled_universe.action_space_metadata.to_dict(),
        'meters': compiled_universe.meter_metadata.to_dict(),
        'affordances': compiled_universe.affordance_metadata.to_dict(),
        'cues': compiled_universe.cues  # Includes colors, icons, tier classifications
    }

// Frontend: Fetch on mount
const simulationStore = useSimulationStore()
await simulationStore.fetchUniverseMetadata()  // Populates reactive state

// Components use reactive metadata
<template>
  <div v-for="action in simulationStore.actionLabels">{{ action }}</div>
  <MeterBar v-for="meter in simulationStore.primaryMeters" :tier="meter.tier" />
</template>
```

**Option B: WebSocket Metadata Message (Alternative)**
```javascript
// First WebSocket message includes metadata
ws.send(JSON.stringify({
  type: 'metadata',
  data: compiled_universe.to_dict()
}))

// Subsequent messages are state updates
ws.send(JSON.stringify({
  type: 'state',
  data: {positions: ..., meters: ...}
}))
```

**Recommendation:** **Option A** - Cleaner separation, REST endpoint reusable for other tools, metadata doesn't need to stream.

**Acceptance Criteria:**
- [ ] Backend `/api/universe/metadata` endpoint returns action labels, meter metadata, cues
- [ ] Frontend Pinia store fetches metadata on init
- [ ] All hardcoded action names replaced with `store.actionLabels`
- [ ] All meter tier classifications derived from `cues.yaml` metadata
- [ ] Heat map colors derived from cues color palette
- [ ] No hardcoded game-specific strings in Vue components
- [ ] Frontend works with custom config packs without code changes

**Effort:** 3-5 days
**Dependencies:** None
**Risk:** Low - Additive change, no breaking modifications

---

### WP-H2: Parameterize Curriculum Stages via YAML

**Issue:** `AdversarialCurriculum` has hardcoded `STAGE_CONFIGS` array in Python instead of YAML configuration.

**Impact:**
- Can't customize curriculum progression without code changes
- Inconsistent with project's declarative configuration philosophy
- Research limited to 5-stage progression only

**Current State:**
```python
# src/townlet/curriculum/adversarial.py lines 50-80 - HARDCODED
STAGE_CONFIGS = [
    StageConfig(
        active_meters=["energy", "health"],
        depletion_multiplier=0.2,
        reward_mode="shaped",
        description="Basic Survival"
    ),
    # ... 4 more stages hardcoded
]
```

**Implementation Options:**

**Option A: Per-Pack Curriculum Config (Recommended)**
```yaml
# configs/L1_full_observability/curriculum.yaml (NEW FILE)
curriculum:
  type: adversarial  # or "static"

  # Adversarial-specific config
  adversarial:
    stages:
      - name: "Basic Survival"
        active_meters: [energy, health]
        depletion_multiplier: 0.2
        reward_mode: shaped
      - name: "Emotional Stability"
        active_meters: [energy, health, mood]
        depletion_multiplier: 0.4
        reward_mode: shaped
      # ... more stages

    advancement_thresholds:
      survival_rate: 0.7
      learning_progress: 0.0
      entropy: 0.5

    retreat_thresholds:
      survival_rate: 0.3
      learning_progress: -0.1

    min_steps_at_stage: 1000

# UAC Stage 1: Parse curriculum.yaml (add to 9 required files)
# AdversarialCurriculum.__init__ receives parsed config
```

**Option B: Global Curriculum Library (Alternative)**
```yaml
# configs/curriculum_library.yaml (GLOBAL)
curriculum_profiles:
  gentle_5stage:
    stages: [...]  # 5 stages with 0.2 → 1.0 progression

  aggressive_3stage:
    stages: [...]  # 3 stages with 0.5 → 1.0 progression

  custom_7stage:
    stages: [...]  # 7 stages for extended training

# configs/L1_full_observability/training.yaml
curriculum:
  type: adversarial
  profile: gentle_5stage  # References library
  overrides:  # Optional per-pack customization
    advancement_thresholds:
      survival_rate: 0.8  # Stricter than library default
```

**Recommendation:** **Option A** - Simpler, self-contained per-pack configs, consistent with current architecture.

**Acceptance Criteria:**
- [ ] UAC parses `curriculum.yaml` (10th required file in config packs)
- [ ] `CurriculumConfig` Pydantic DTO validates structure
- [ ] `STAGE_CONFIGS` module constant deleted
- [ ] `AdversarialCurriculum.__init__` loads stages from config
- [ ] All existing curriculum packs updated with `curriculum.yaml`
- [ ] Advancement/retreat thresholds parameterizable
- [ ] `min_steps_at_stage` configurable
- [ ] Tests verify custom stage progressions (3-stage, 7-stage variants)
- [ ] Breaking change documented in migration guide

**Effort:** 5-8 days (includes UAC integration, config migration, testing)
**Dependencies:** None
**Risk:** Medium - Breaking change requires updating all config packs

**Migration Strategy:**
1. Add `curriculum.yaml` parsing to UAC (default to current hardcoded values for backwards compat)
2. Update all curriculum packs with explicit `curriculum.yaml`
3. Remove hardcoded fallback in next release

---

### WP-H3: Expose Exploration Annealing Thresholds in Config

**Issue:** `AdaptiveIntrinsicExploration` has hardcoded annealing thresholds requiring code edits to tune.

**Impact:**
- Violates No-Defaults Principle
- Requires Python changes to experiment with annealing behavior
- Implicit assumptions about "consistently succeeding" criteria

**Current State:**
```python
# src/townlet/exploration/adaptive_intrinsic.py - HARDCODED
class AdaptiveIntrinsicExploration:
    def __init__(self, ...):
        self.variance_threshold = 100.0  # HARDCODED
        self.min_survival_fraction = 0.4  # HARDCODED
        self.annealing_rate = 0.5  # HARDCODED
        self.min_intrinsic_weight = 0.01  # HARDCODED
```

**Implementation Option (Single Path Forward):**
```yaml
# training.yaml
exploration:
  strategy: adaptive_rnd

  # Base RND config
  rnd:
    predictor_lr: 0.0001
    target_dim: 128
    predictor_hidden_dim: 256

  # Adaptive annealing config (NEW)
  adaptive:
    variance_threshold: 100.0  # Variance below this triggers annealing
    min_survival_fraction: 0.4  # Mean survival must exceed this
    annealing_rate: 0.5  # Multiply intrinsic_weight by this on anneal
    min_intrinsic_weight: 0.01  # Floor for intrinsic_weight
    survival_window: 100  # Episodes to track for variance calculation
```

**Acceptance Criteria:**
- [ ] `ExplorationConfig` DTO includes `adaptive` field with all thresholds
- [ ] `AdaptiveIntrinsicExploration.__init__` loads thresholds from config
- [ ] Hardcoded defaults deleted
- [ ] All curriculum packs updated with explicit adaptive thresholds
- [ ] Docs updated: `docs/config-schemas/training.md` documents adaptive params
- [ ] Tests verify different threshold combinations
- [ ] Config validation ensures thresholds in valid ranges (variance_threshold > 0, etc.)

**Effort:** 2-3 days
**Dependencies:** None
**Risk:** Low - Additive, backwards compatible with defaults during migration

---

### WP-H4: Temporal Features Integration or Removal

**Issue:** `RecurrentSpatialQNetwork` extracts temporal features but documents them as "ignored" - unclear intent.

**Impact:**
- Signal loss for L3 (temporal mechanics curriculum)
- Network inefficiency (extracting unused features)
- Architectural ambiguity

**Current State:**
```python
# src/townlet/agent/networks.py line 213
temporal_features = obs[:, temporal_start:temporal_end]  # Extracted
# Comment: "Temporal features currently ignored"
# No temporal encoder or concatenation
```

**Implementation Options:**

**Option A: Conditional Temporal Encoder (Recommended for L3)**
```python
class RecurrentSpatialQNetwork:
    def __init__(self, obs_dim, action_dim, ..., use_temporal=False):
        # Existing encoders
        self.vision_encoder = ...
        self.position_encoder = ...
        self.meter_encoder = ...
        self.affordance_encoder = ...

        # NEW: Temporal encoder (conditional)
        self.use_temporal = use_temporal
        if use_temporal:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(4, 16),  # 4 temporal features → 16
                nn.LayerNorm(16),
                nn.ReLU()
            )
            lstm_input_dim = 128 + 32 + 32 + 32 + 16  # +16 for temporal
        else:
            lstm_input_dim = 128 + 32 + 32 + 32

        self.lstm = nn.LSTM(lstm_input_dim, 256)

    def forward(self, obs, hidden_state):
        # Extract features
        vision = self.vision_encoder(vision_window)
        position = self.position_encoder(coords)
        meters = self.meter_encoder(meter_vals)
        affordances = self.affordance_encoder(affordance_vals)

        # Conditionally encode temporal
        if self.use_temporal:
            temporal = self.temporal_encoder(temporal_features)
            combined = torch.cat([vision, position, meters, affordances, temporal], dim=1)
        else:
            combined = torch.cat([vision, position, meters, affordances], dim=1)

        lstm_out, hidden = self.lstm(combined, hidden_state)
        q_values = self.q_head(lstm_out)
        return q_values, hidden
```

```yaml
# brain.yaml (L3_temporal_mechanics)
architecture:
  type: recurrent
  use_temporal_encoder: true  # Enable for L3
  hidden_dim: 256
```

**Option B: Remove Temporal Features Entirely**
```python
# Delete temporal extraction code
# Remove temporal features from VFS observation spec
# Update observation dimension calculations
# Faster, simpler if temporal not needed
```

**Recommendation:**
- **Option A** for L3 curriculum (temporal mechanics benefit from time-of-day encoding)
- **Option B** for L0-L2 (no temporal mechanics, cleaner architecture)

**Investigation Tasks** (BEFORE implementation):
1. **Empirical Test:** Train L3 agents with/without temporal encoding, measure performance
2. **Observation Analysis:** Verify temporal features actually vary in L3 (day/night cycles affect observations)
3. **Ablation Study:** Compare: (a) no temporal, (b) temporal MLP encoder, (c) temporal passed to LSTM

**Acceptance Criteria:**
- [ ] Investigation complete: empirical data on temporal feature value
- [ ] Decision documented (Option A or B with justification)
- [ ] If Option A: Temporal encoder implemented, `use_temporal` flag in BrainConfig
- [ ] If Option A: L3 curriculum uses temporal encoder, L0-L2 disable it
- [ ] If Option B: Temporal features removed from observation spec, dims updated
- [ ] "Ignored" comment deleted
- [ ] Tests verify correct observation dimensions

**Effort:** 5-8 days (includes investigation, implementation, ablation study)
**Dependencies:** None
**Risk:** Medium - Requires empirical validation to make correct choice

---

## Priority 2: Medium (Technical Debt)

### WP-M1: Implement Prioritized Replay for Recurrent Networks

**Issue:** `PrioritizedReplayBuffer` raises `NotImplementedError` for recurrent networks (TASK-005 Phase 3).

**Impact:**
- L2 POMDP curriculum can't use prioritized experience replay
- Research limitation for comparing PER vs uniform sampling on POMDP tasks

**Current State:**
```python
# src/townlet/population/vectorized.py line 318
if use_prioritized and network_type == "recurrent":
    raise NotImplementedError("PER not supported for recurrent networks")
```

**Implementation Option (Single Path Forward):**
```python
class PrioritizedSequentialReplayBuffer:
    """Prioritized replay for full episodes (recurrent networks)."""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.episodes = []  # List of episode containers
        self.priorities = []  # TD-error-based priorities per episode
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.max_priority = 1.0

    def push_episode(self, episode_container):
        """Store full episode with initial max priority."""
        self.episodes.append(episode_container)
        self.priorities.append(self.max_priority)

        # Evict oldest if over capacity
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
            self.priorities.pop(0)

    def sample_episodes(self, batch_size):
        """Sample episodes proportional to priority."""
        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample episode indices
        indices = np.random.choice(len(self.episodes), batch_size, p=probs)

        # Compute importance sampling weights
        weights = (len(self.episodes) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Return episodes + weights
        sampled_episodes = [self.episodes[i] for i in indices]
        return sampled_episodes, indices, torch.from_numpy(weights)

    def update_priorities(self, indices, td_errors):
        """Update episode priorities based on mean TD-error."""
        for idx, td_error in zip(indices, td_errors):
            # Use mean absolute TD-error across episode
            priority = abs(td_error.mean().item()) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
```

**Integration with VectorizedPopulation:**
```python
# Training loop modification
if use_prioritized and network_type == "recurrent":
    # Sample episodes with priorities
    episodes, indices, is_weights = replay_buffer.sample_episodes(batch_size)

    # Unroll episodes for training
    batch = unroll_episodes(episodes)

    # Compute TD-errors
    td_errors = compute_td_errors(batch, is_weights)

    # Update priorities
    replay_buffer.update_priorities(indices, td_errors)
```

**Acceptance Criteria:**
- [ ] `PrioritizedSequentialReplayBuffer` class implemented
- [ ] Episode-level priority computation (mean/max TD-error strategy)
- [ ] Importance sampling weights applied to loss
- [ ] `NotImplementedError` removed from Population
- [ ] `brain.yaml` supports `prioritized: true` for recurrent networks
- [ ] L2 curriculum config updated with PER option
- [ ] Tests verify priority updates, IS weights, episode sampling
- [ ] Benchmark: PER vs uniform sampling on L2 POMDP task
- [ ] Docs updated: `docs/config-schemas/brain.md` documents PER for recurrent

**Effort:** 5-8 days
**Dependencies:** None
**Risk:** Medium - Complex interaction between episode storage and priority updates

---

### WP-M2: Network Architecture Plugin Registry

**Issue:** `VectorizedPopulation` uses if/elif chains to select network architecture instead of plugin system.

**Impact:**
- Adding custom network types requires editing Population code
- Violates Open/Closed Principle
- Research friction for experimenting with novel architectures

**Current State:**
```python
# src/townlet/population/vectorized.py lines ~150-180
if brain_config.architecture.type == "feedforward":
    self.q_network = SimpleQNetwork(...)
    self.replay_buffer = ReplayBuffer(...)
elif brain_config.architecture.type == "recurrent":
    self.q_network = RecurrentSpatialQNetwork(...)
    self.replay_buffer = SequentialReplayBuffer(...)
elif brain_config.architecture.type == "dueling":
    self.q_network = DuelingQNetwork(...)
    self.replay_buffer = ReplayBuffer(...)
else:
    raise ValueError(f"Unknown architecture: {brain_config.architecture.type}")
```

**Implementation Options:**

**Option A: Simple Registry Pattern (Recommended)**
```python
# src/townlet/agent/registry.py (NEW FILE)
ARCHITECTURE_REGISTRY = {}

def register_architecture(name, network_cls, buffer_cls):
    """Register a network architecture + compatible replay buffer."""
    ARCHITECTURE_REGISTRY[name] = {
        'network': network_cls,
        'buffer': buffer_cls
    }

# Register built-in architectures
register_architecture("feedforward", SimpleQNetwork, ReplayBuffer)
register_architecture("recurrent", RecurrentSpatialQNetwork, SequentialReplayBuffer)
register_architecture("dueling", DuelingQNetwork, ReplayBuffer)
register_architecture("structured", StructuredQNetwork, ReplayBuffer)

# Population uses registry
def create_networks(brain_config):
    arch = ARCHITECTURE_REGISTRY[brain_config.architecture.type]
    network = arch['network'](obs_dim, action_dim, **brain_config.network)
    buffer = arch['buffer'](capacity, **brain_config.replay)
    return network, buffer
```

**Option B: Factory Class Pattern (More Complex)**
```python
class NetworkFactory:
    """Factory for creating network + buffer pairs."""

    @staticmethod
    def create(arch_type, obs_dim, action_dim, brain_config):
        if arch_type == "feedforward":
            return FeedforwardArchitecture(obs_dim, action_dim, brain_config)
        elif arch_type == "recurrent":
            return RecurrentArchitecture(obs_dim, action_dim, brain_config)
        # ... etc

class ArchitectureBase(ABC):
    @abstractmethod
    def create_network(self): pass

    @abstractmethod
    def create_buffer(self): pass
```

**Recommendation:** **Option A** - Simpler, sufficient for current needs, easy to extend.

**Acceptance Criteria:**
- [ ] `ARCHITECTURE_REGISTRY` dict created in `src/townlet/agent/registry.py`
- [ ] `register_architecture(name, network_cls, buffer_cls)` function
- [ ] All 4 existing architectures registered (feedforward, recurrent, dueling, structured)
- [ ] `VectorizedPopulation` uses registry instead of if/elif chains
- [ ] Custom architecture registration example in docs
- [ ] Tests verify registry lookup, unknown architecture raises clear error
- [ ] No breaking changes to BrainConfig YAML schema

**Effort:** 3-5 days
**Dependencies:** None
**Risk:** Low - Internal refactor, no API changes

---

### WP-M3: Resolve BrainConfig Hidden Dimension TODOs

**Issue:** Network hidden dimensions have `TODO(BRAIN_AS_CODE)` comments indicating incomplete config-driven design.

**Impact:**
- Some architectural parameters still hardcoded
- Incomplete migration to brain.yaml config system
- Inconsistent with No-Defaults Principle

**Current State:**
```python
# src/townlet/agent/networks.py
class RecurrentSpatialQNetwork:
    def __init__(self, obs_dim, action_dim, hidden_dim=256):  # TODO(BRAIN_AS_CODE)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 1 * 1, 128)  # HARDCODED output dim
        )

        self.position_encoder = nn.Linear(2, 32)  # HARDCODED
        self.meter_encoder = nn.Linear(8, 32)  # HARDCODED
        self.affordance_encoder = nn.Linear(15, 32)  # HARDCODED

        self.lstm = nn.LSTM(128 + 32 + 32 + 32, hidden_dim)  # Partial config
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # HARDCODED
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
```

**Implementation Option (Single Path Forward):**
```yaml
# brain.yaml
architecture:
  type: recurrent

  # Vision encoder config
  vision:
    conv_channels: [16, 32]  # Conv layer sizes
    vision_output_dim: 128

  # Encoder output dimensions
  encoders:
    position_dim: 32
    meter_dim: 32
    affordance_dim: 32

  # LSTM config
  lstm:
    hidden_dim: 256

  # Q-head config
  q_head:
    hidden_layers: [128]  # MLP hidden dims before action output
```

```python
# Updated network init
class RecurrentSpatialQNetwork:
    def __init__(self, obs_dim, action_dim, brain_config):
        vision_cfg = brain_config.architecture.vision
        encoder_cfg = brain_config.architecture.encoders
        lstm_cfg = brain_config.architecture.lstm
        qhead_cfg = brain_config.architecture.q_head

        # Vision encoder (configurable)
        self.vision_encoder = self._build_vision_encoder(
            channels=vision_cfg.conv_channels,
            output_dim=vision_cfg.vision_output_dim
        )

        # Other encoders (configurable)
        self.position_encoder = nn.Linear(2, encoder_cfg.position_dim)
        self.meter_encoder = nn.Linear(8, encoder_cfg.meter_dim)
        self.affordance_encoder = nn.Linear(15, encoder_cfg.affordance_dim)

        # LSTM (configurable)
        lstm_input = sum([
            vision_cfg.vision_output_dim,
            encoder_cfg.position_dim,
            encoder_cfg.meter_dim,
            encoder_cfg.affordance_dim
        ])
        self.lstm = nn.LSTM(lstm_input, lstm_cfg.hidden_dim)

        # Q-head (configurable MLP)
        self.q_head = self._build_mlp(
            input_dim=lstm_cfg.hidden_dim,
            hidden_dims=qhead_cfg.hidden_layers,
            output_dim=action_dim
        )
```

**Acceptance Criteria:**
- [ ] `BrainConfig.architecture` schema includes encoder dimensions
- [ ] All hardcoded hidden dims replaced with config params
- [ ] `TODO(BRAIN_AS_CODE)` comments deleted
- [ ] All curriculum brain.yaml files updated with explicit dimensions
- [ ] Config validation ensures dims > 0, reasonable ranges
- [ ] Tests verify network shapes match config specifications
- [ ] Docs updated: `docs/config-schemas/brain.md` documents all architecture params

**Effort:** 5-8 days (includes schema design, network refactor, config migration)
**Dependencies:** None
**Risk:** Medium - Touches critical network code, requires careful testing

---

### WP-M4: Optimize Episode Container GPU Transfer

**Issue:** Recurrent mode uses CPU Python lists for episode containers, GPU transfer at store time.

**Impact:**
- Memory inefficiency (CPU ↔ GPU transfers)
- Potential performance bottleneck for LSTM training

**Current State:**
```python
# Recurrent training loop
episode_containers = [[] for _ in range(num_agents)]  # CPU Python lists

# Accumulate transitions
for step in episode:
    for agent_id in range(num_agents):
        episode_containers[agent_id].append(transition)  # CPU tensors

# Store at episode end (GPU transfer happens here)
if done[agent_id]:
    replay_buffer.push_episode(episode_containers[agent_id])  # Transfer to GPU
```

**Implementation Options:**

**Option A: GPU Episode Buffers (Recommended)**
```python
class GPUEpisodeContainer:
    """Pre-allocated GPU tensor buffers for episode accumulation."""

    def __init__(self, max_episode_length, obs_dim, device):
        self.max_len = max_episode_length
        self.device = device
        self.pos = 0

        # Pre-allocate GPU tensors
        self.observations = torch.zeros(max_len, obs_dim, device=device)
        self.actions = torch.zeros(max_len, dtype=torch.long, device=device)
        self.rewards = torch.zeros(max_len, device=device)
        self.dones = torch.zeros(max_len, dtype=torch.bool, device=device)

    def append(self, obs, action, reward, done):
        """Append transition directly to GPU buffers."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1

    def finalize(self):
        """Return episode sliced to actual length."""
        return {
            'observations': self.observations[:self.pos],
            'actions': self.actions[:self.pos],
            'rewards': self.rewards[:self.pos],
            'dones': self.dones[:self.pos]
        }
```

**Option B: Batched Accumulation (Alternative)**
```python
# Accumulate all agents' transitions in single batched tensor
episode_obs = torch.zeros(num_agents, max_episode_len, obs_dim, device=device)
episode_actions = torch.zeros(num_agents, max_episode_len, dtype=torch.long, device=device)
# ... etc

# Each step, write to [agent_id, step_idx]
episode_obs[agent_ids, step_idx] = observations
episode_actions[agent_ids, step_idx] = actions
```

**Recommendation:** **Option A** - More flexible for variable-length episodes, cleaner API.

**Acceptance Criteria:**
- [ ] `GPUEpisodeContainer` class implemented
- [ ] `VectorizedPopulation` uses GPU containers for recurrent mode
- [ ] CPU Python list accumulation deleted
- [ ] Benchmark: GPU containers vs CPU lists (measure transfer overhead reduction)
- [ ] Memory profiling confirms GPU memory reuse (no leaks)
- [ ] Tests verify episode integrity (no data corruption from buffer reuse)
- [ ] Config param for `max_episode_length` in training.yaml

**Effort:** 3-5 days
**Dependencies:** None
**Risk:** Medium - GPU memory management, potential buffer overflow if episode > max_len

---

### WP-M5: UAC Complexity Refactoring

**Issue:** Universe Compiler is 2,600+ lines with 25+ validation methods in single file.

**Impact:**
- Difficult to navigate and maintain
- Testing individual validation logic requires full compiler setup
- Risk of validation bugs in complex edge cases

**Current State:**
```
src/townlet/universe/compiler.py: 2,600 lines
- Stage 1: Parse (100 lines)
- Stage 2: Symbol Table (200 lines)
- Stage 3: Resolve (300 lines)
- Stage 4: Cross-Validate (800 lines) ← 25+ validation methods
- Stage 5: Metadata (400 lines)
- Stage 6: Optimize (200 lines)
- Stage 7: Emit (100 lines)
- Cache management (200 lines)
- Provenance (100 lines)
- Helpers (200 lines)
```

**Implementation Options:**

**Option A: Split by Stage (Recommended)**
```
src/townlet/universe/
├── compiler.py (300 lines) - Main orchestrator
├── stages/
│   ├── parse.py (150 lines) - Stage 1
│   ├── symbol_table.py (250 lines) - Stage 2
│   ├── resolve.py (350 lines) - Stage 3
│   ├── validate.py (900 lines) - Stage 4 with validation modules
│   │   ├── cascade_validator.py (200 lines)
│   │   ├── affordance_validator.py (200 lines)
│   │   ├── pomdp_validator.py (200 lines)
│   │   ├── economic_validator.py (200 lines)
│   ├── metadata.py (450 lines) - Stage 5
│   ├── optimize.py (250 lines) - Stage 6
│   └── emit.py (150 lines) - Stage 7
├── cache.py (250 lines) - Cache management
└── provenance.py (150 lines) - Hashing, git info
```

**Option B: Split by Domain (Alternative)**
```
src/townlet/universe/
├── compiler.py (main orchestrator)
├── bar_compiler.py (bars + cascades validation)
├── affordance_compiler.py (affordances + interactions)
├── substrate_compiler.py (spatial validation + POMDP)
├── dac_compiler.py (drive_as_code validation)
└── vfs_compiler.py (variables + observation spec)
```

**Recommendation:** **Option A** - Preserves 7-stage pipeline clarity, easier to test individual stages.

**Acceptance Criteria:**
- [ ] Compiler split into 10+ files organized by stage
- [ ] Each stage has isolated unit tests (don't require full compiler)
- [ ] Main `UniverseCompiler` orchestrates stage calls
- [ ] No functional changes (pure refactor)
- [ ] All existing tests pass
- [ ] Code coverage maintained or improved
- [ ] Docs updated: `docs/UNIVERSE-COMPILER.md` reflects new structure

**Effort:** 5-8 days
**Dependencies:** None
**Risk:** Medium - Large refactor, risk of introducing regressions

---

### WP-M6: Frontend WebSocket Error Handling

**Issue:** No error handling for malformed WebSocket messages or connection failures.

**Impact:**
- Frontend crashes on message corruption
- No graceful degradation on connection loss
- Poor UX for network issues

**Current State:**
```javascript
// stores/simulation.js - NO ERROR HANDLING
this.ws.onmessage = (event) => {
  const data = JSON.parse(event.data)  // Can throw on malformed JSON
  this.agents = data.agents  // Can fail if data.agents undefined
  this.meters = data.meters
}
```

**Implementation Option (Single Path Forward):**
```javascript
// stores/simulation.js
export const useSimulationStore = defineStore('simulation', {
  state: () => ({
    connectionStatus: 'disconnected',  // disconnected | connecting | connected | error
    lastError: null,
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    // ... existing state
  }),

  actions: {
    connectWebSocket(url = 'ws://localhost:8766') {
      this.connectionStatus = 'connecting'
      this.ws = new WebSocket(url)

      this.ws.onopen = () => {
        this.connectionStatus = 'connected'
        this.reconnectAttempts = 0
        console.log('[WS] Connected to inference server')
      }

      this.ws.onmessage = (event) => {
        try {
          // Validate JSON
          const data = JSON.parse(event.data)

          // Validate message structure
          if (!this.validateMessage(data)) {
            console.warn('[WS] Invalid message structure:', data)
            return
          }

          // Update state
          this.agents = data.agents ?? this.agents
          this.meters = data.meters ?? this.meters
          this.heatMap = data.heatMap ?? this.heatMap

        } catch (error) {
          console.error('[WS] Message parse error:', error)
          this.lastError = error.message
        }
      }

      this.ws.onerror = (error) => {
        console.error('[WS] Connection error:', error)
        this.connectionStatus = 'error'
        this.lastError = 'WebSocket connection failed'
      }

      this.ws.onclose = (event) => {
        console.log('[WS] Connection closed:', event.code, event.reason)
        this.connectionStatus = 'disconnected'

        // Auto-reconnect with exponential backoff
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 30000)
          console.log(`[WS] Reconnecting in ${delay}ms...`)

          setTimeout(() => {
            this.reconnectAttempts++
            this.connectWebSocket(url)
          }, delay)
        } else {
          console.error('[WS] Max reconnect attempts reached')
          this.lastError = 'Connection lost. Please refresh page.'
        }
      }
    },

    validateMessage(data) {
      // Schema validation
      return (
        data &&
        typeof data === 'object' &&
        Array.isArray(data.agents) &&
        typeof data.meters === 'object'
      )
    }
  }
})
```

```vue
<!-- App.vue - Connection status UI -->
<template>
  <div class="app">
    <!-- Connection status banner -->
    <div v-if="store.connectionStatus !== 'connected'" class="connection-banner">
      <span v-if="store.connectionStatus === 'connecting'">
        Connecting to simulation server...
      </span>
      <span v-else-if="store.connectionStatus === 'error'" class="error">
        {{ store.lastError }}
      </span>
      <span v-else>
        Disconnected. Attempting to reconnect...
      </span>
    </div>

    <!-- Main content -->
    <SimulationView v-if="store.connectionStatus === 'connected'" />
    <EmptyState v-else message="Waiting for connection..." />
  </div>
</template>
```

**Acceptance Criteria:**
- [ ] WebSocket connection status tracked (disconnected/connecting/connected/error)
- [ ] JSON parse errors caught and logged (don't crash app)
- [ ] Message validation before state updates
- [ ] Auto-reconnect with exponential backoff (5 attempts max)
- [ ] UI banner shows connection status
- [ ] Graceful fallback UI when disconnected
- [ ] `console.warn` for invalid messages (debugging)
- [ ] Tests verify error handling paths

**Effort:** 2-3 days
**Dependencies:** None
**Risk:** Low - Error handling is additive

---

### WP-M7: POMDP Validation at Compile Time

**Issue:** POMDP validation happens at environment init (runtime) instead of config compilation (compile time).

**Impact:**
- Config errors discovered late (after UAC passes, during env construction)
- Poor developer experience (slow iteration on POMDP configs)

**Current State:**
```python
# src/townlet/environment/vectorized_env.py.__init__
if use_pomdp:
    if isinstance(substrate, Grid3DSubstrate) and vision_range > 2:
        raise ValueError("Grid3D POMDP vision_range must be ≤ 2")
    if isinstance(substrate, GridNDSubstrate) and substrate.dims >= 4:
        raise ValueError("GridND (N≥4) does not support POMDP")
```

**Implementation Option (Single Path Forward):**
```python
# src/townlet/universe/stages/validate.py (UAC Stage 4)
def validate_pomdp_compatibility(self, compiled):
    """Validate POMDP configurations during compilation."""

    if not compiled.environment_config.use_pomdp:
        return  # No POMDP, skip validation

    substrate_type = compiled.substrate_config.substrate_type
    vision_range = compiled.environment_config.vision_range

    # Grid3D validation
    if substrate_type == "grid3d":
        if vision_range > 2:
            self.errors.append(ValidationError(
                stage="cross_validate",
                category="pomdp",
                message=f"Grid3D POMDP vision_range must be ≤ 2 (got {vision_range})",
                suggestion="Reduce vision_range to 2 or disable POMDP"
            ))

    # GridND validation
    if substrate_type == "gridnd":
        dims = compiled.substrate_config.grid_size
        if len(dims) >= 4:
            window_size = (2 * vision_range + 1) ** len(dims)
            self.errors.append(ValidationError(
                stage="cross_validate",
                category="pomdp",
                message=f"GridND {len(dims)}D POMDP window explodes ({window_size} cells)",
                suggestion="Use Grid2D/Grid3D for POMDP or disable partial observability"
            ))

    # Continuous substrate validation
    if substrate_type.startswith("continuous"):
        self.errors.append(ValidationError(
            stage="cross_validate",
            category="pomdp",
            message="Continuous substrates do not support POMDP (window concept undefined)",
            suggestion="Use grid substrates for POMDP"
        ))
```

**Acceptance Criteria:**
- [ ] UAC Stage 4 includes POMDP validation
- [ ] All POMDP constraints checked at compile time
- [ ] Actionable error messages with suggestions
- [ ] Environment init POMDP checks deleted (redundant)
- [ ] Tests verify UAC rejects invalid POMDP configs
- [ ] Invalid POMDP configs fail during `uv run python -m townlet.compiler validate`
- [ ] CI config validation catches POMDP errors before training

**Effort:** 2-3 days
**Dependencies:** None
**Risk:** Low - Validation logic already exists, just moving to UAC

---

## Priority 3: Low (Nice-to-Have)

### WP-L1: Double DQN Recurrent Mode Optimization

**Issue:** Double DQN in recurrent mode requires redundant forward passes through online network.

**Impact:**
- ~33% slower training for recurrent Double DQN vs vanilla DQN
- GPU compute waste

**Current State:**
```python
# Recurrent Double DQN (3 forward passes)
with torch.no_grad():
    # Pass 1: Online network for action selection
    next_q_online, _ = self.q_network(next_obs, next_hidden)
    next_actions = next_q_online.argmax(dim=1)

    # Pass 2: Target network for value evaluation
    next_q_target, _ = self.target_network(next_obs, next_hidden)
    next_q_values = next_q_target.gather(1, next_actions)

# Pass 3: Online network for current Q-values (training)
current_q, _ = self.q_network(obs, hidden)
```

**Implementation Options:**

**Option A: Cached Forward Pass (Complex)**
```python
# Cache online network forward pass
with torch.no_grad():
    next_q_online, next_hidden_cached = self.q_network(next_obs, next_hidden)
    next_actions = next_q_online.argmax(dim=1)

    next_q_target, _ = self.target_network(next_obs, next_hidden)
    next_q_values = next_q_target.gather(1, next_actions)

# Reuse cached values if obs == next_obs (only at episode boundaries)
if same_episode:
    current_q = next_q_online  # Reuse
else:
    current_q, _ = self.q_network(obs, hidden)
```

**Option B: Accept 3-Pass Overhead (Simple)**
- Document as expected behavior in Double DQN + recurrent
- Benefit: Decoupled networks (cleaner separation of concerns)
- Cost: ~33% more compute

**Recommendation:** **Option B** - Complexity of caching not worth ~10ms speedup per batch. Document as trade-off.

**Acceptance Criteria (Option B):**
- [ ] Docs updated: `docs/config-schemas/training.md` documents Double DQN overhead for recurrent
- [ ] Benchmark added: Measure vanilla vs Double DQN training time (expect ~33% slower)
- [ ] Comment in code explaining 3-pass requirement
- [ ] No code changes (accept current behavior)

**Effort:** 0.5 days (documentation only for Option B)
**Dependencies:** None
**Risk:** None (documentation change)

---

### WP-L2: Curriculum PerformanceTracker Parameter Rename

**Issue:** `update_step()` parameter named "rewards" but receives step counts.

**Impact:**
- Maintainer confusion
- Risk of bugs if actual rewards passed

**Current State:**
```python
# src/townlet/curriculum/adversarial.py lines 88-89
def update_step(self, agent_idx, rewards, done):  # "rewards" is misleading
    self.survival_counts[agent_idx] += rewards  # Actually step counts
```

**Implementation Option (Single Path Forward):**
```python
# Rename parameter
def update_step(self, agent_idx, step_count, done):
    """Update agent performance metrics.

    Args:
        agent_idx: Agent index
        step_count: Number of steps survived this episode
        done: Whether episode terminated
    """
    self.survival_counts[agent_idx] += step_count
```

**Acceptance Criteria:**
- [ ] Parameter renamed to `step_count`
- [ ] Docstring updated
- [ ] All call sites updated
- [ ] Tests verify correct behavior
- [ ] No functional changes

**Effort:** 0.5 days
**Dependencies:** None
**Risk:** None (simple refactor)

---

### WP-L3: DuelingQNetwork Integration or Removal

**Issue:** `DuelingQNetwork` exists but not referenced in factory or training code.

**Impact:**
- Dead code or incomplete feature
- Unclear if intended for future curriculum

**Current State:**
```python
# src/townlet/agent/networks.py
class DuelingQNetwork(nn.Module):
    # 200 lines of implementation
    # Not referenced in NetworkFactory or BrainConfig
```

**Implementation Options:**

**Option A: Integrate into BrainConfig**
```yaml
# brain.yaml
architecture:
  type: dueling  # Enable dueling architecture
  hidden_layers: [256, 128]
  advantage_layers: [128]
  value_layers: [128]
```

```python
# NetworkFactory
ARCHITECTURE_REGISTRY["dueling"] = (DuelingQNetwork, ReplayBuffer)
```

**Option B: Delete if Unused**
- Remove DuelingQNetwork class
- Remove from imports
- Document decision (dueling not beneficial for HAMLET tasks)

**Recommendation:** **Needs investigation** - Benchmark dueling vs SimpleQNetwork on L1. If no improvement, delete.

**Acceptance Criteria (Option A - if beneficial):**
- [ ] Benchmark: Dueling vs SimpleQNetwork on L1 (10 runs each)
- [ ] If >5% improvement: Integrate into BrainConfig
- [ ] If no improvement: Delete and document decision

**Acceptance Criteria (Option B - if not beneficial):**
- [ ] DuelingQNetwork class deleted
- [ ] Benchmarking results documented in `docs/decisions/`
- [ ] Tests cleaned up

**Effort:** 3-5 days (includes benchmarking)
**Dependencies:** None
**Risk:** Low - Either way (integrate or delete) improves codebase

---

### WP-L4: Keyboard Shortcuts Accessibility Improvements

**Issue:** Grid.vue keyboard handler only excludes INPUT/TEXTAREA, should extend to all form controls.

**Impact:**
- Keyboard shortcuts fire when typing in contenteditable, select, button
- Poor UX for accessible keyboard navigation

**Current State:**
```javascript
// Grid.vue line 221
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    return  // Don't handle shortcuts
  }

  if (e.key === 'h') {
    showHeatMap.value = !showHeatMap.value
  }
})
```

**Implementation Option (Single Path Forward):**
```javascript
// Improved keyboard handler
function isEditableElement(element) {
  const tag = element.tagName
  const editable = element.isContentEditable
  const role = element.getAttribute('role')

  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    tag === 'BUTTON' ||
    editable ||
    role === 'textbox' ||
    role === 'searchbox' ||
    role === 'combobox'
  )
}

window.addEventListener('keydown', (e) => {
  if (isEditableElement(e.target)) {
    return  // Don't handle shortcuts in editable contexts
  }

  // Handle shortcuts
  if (e.key === 'h') {
    showHeatMap.value = !showHeatMap.value
  }

  // Add more shortcuts
  if (e.key === 't') {
    showTrails.value = !showTrails.value
  }

  if (e.key === '?') {
    showKeyboardHelp.value = true
  }
})
```

**Acceptance Criteria:**
- [ ] `isEditableElement()` helper function
- [ ] All form controls excluded from shortcuts
- [ ] Additional shortcuts documented (t = trails, ? = help)
- [ ] Keyboard shortcuts help modal (? key)
- [ ] ARIA live region announces shortcut activations
- [ ] Tests verify shortcuts don't fire in editable contexts

**Effort:** 1-2 days
**Dependencies:** None
**Risk:** None (accessibility improvement)

---

## Summary Tables

### Priority Distribution

| Priority | Count | Total Effort |
|----------|-------|--------------|
| P0: Critical | 0 | 0 days |
| P1: High | 4 | 15-24 days |
| P2: Medium | 7 | 30-46 days |
| P3: Low | 4 | 5-13 days |
| **Total** | **15** | **50-83 days** |

### Effort by Category

| Category | Packages | Effort |
|----------|----------|--------|
| Configuration System | WP-H1, WP-H2, WP-H3 | 10-16 days |
| Network Architecture | WP-H4, WP-M2, WP-M3, WP-L3 | 16-26 days |
| Training Infrastructure | WP-M1, WP-M4, WP-L1 | 8-16 days |
| Code Quality | WP-M5, WP-L2 | 5.5-8.5 days |
| Frontend | WP-M6, WP-M7, WP-L4 | 5-8 days |
| Validation | WP-M7 | 2-3 days |

### Risk Assessment

| Risk Level | Packages | Mitigation |
|------------|----------|------------|
| High | None | - |
| Medium | WP-H2, WP-H4, WP-M1, WP-M3, WP-M4, WP-M5 | Comprehensive testing, staged rollout |
| Low | All others | Standard development practices |

---

## Recommended Implementation Sequence

### Phase 1: Quick Wins (2-3 weeks)
1. **WP-H3:** Exploration thresholds in config (2-3 days)
2. **WP-L2:** Parameter rename (0.5 days)
3. **WP-L4:** Keyboard shortcuts (1-2 days)
4. **WP-M6:** WebSocket error handling (2-3 days)
5. **WP-M7:** POMDP validation at compile time (2-3 days)

**Rationale:** Low risk, high value, builds momentum

### Phase 2: Configuration Completeness (3-4 weeks)
6. **WP-H2:** Curriculum stages YAML (5-8 days)
7. **WP-H1:** Frontend derive from metadata (3-5 days)
8. **WP-M2:** Network architecture registry (3-5 days)

**Rationale:** Completes declarative configuration vision

### Phase 3: Performance & Architecture (4-6 weeks)
9. **WP-H4:** Temporal features investigation (5-8 days)
10. **WP-M1:** Prioritized replay for recurrent (5-8 days)
11. **WP-M3:** BrainConfig hidden dimensions (5-8 days)
12. **WP-M4:** GPU episode containers (3-5 days)

**Rationale:** Requires benchmarking and careful validation

### Phase 4: Code Quality (2-3 weeks)
13. **WP-M5:** UAC refactoring (5-8 days)
14. **WP-L1:** Double DQN documentation (0.5 days)
15. **WP-L3:** DuelingQNetwork decision (3-5 days)

**Rationale:** Internal improvements, no user-facing changes

---

## Appendix: Effort Estimation Guide

**Estimation Methodology:**
- **0.5-1 day:** Simple refactor, rename, documentation
- **2-3 days:** Config schema changes, straightforward implementation
- **3-5 days:** Moderate complexity, requires testing across curriculum
- **5-8 days:** Complex changes, benchmarking, migration required
- **8+ days:** Major refactoring, cross-subsystem coordination

**Assumptions:**
- 1 developer working full-time
- Includes implementation, testing, documentation
- Excludes code review and deployment time
- Assumes familiarity with codebase

**Risk Factors:**
- **Low Risk:** Additive changes, clear requirements, isolated subsystems
- **Medium Risk:** Breaking changes, cross-subsystem, requires migration
- **High Risk:** Core architecture changes, performance-critical paths, unclear requirements

---

## Change Control

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-13 | Initial work package triage | Architecture Analysis |

**Approval Required For:**
- Priority changes (P1 ↔ P2 ↔ P3)
- Effort estimate increases >50%
- Adding new work packages based on discovered issues
- Deferring or canceling packages

**Review Cycle:**
- Monthly review of progress and priorities
- Quarterly architectural review
- Update this document as packages complete
