# Townlet: GPU-Native Sparse Reward System Design

**Date**: 2025-10-29
**Status**: Design Phase
**Target**: Greenfield implementation, will replace Hamlet when proven

---

## 1. Overview & Motivation

### 1.1 Problem Statement

Hamlet's current architecture has fundamental limitations preventing sparse reward implementation:

- **Integration Hell**: Previous attempt at sparse rewards failed due to subsystems "speaking different languages" - no common data contracts
- **CPU-First Design**: Single-agent focus with list-based multi-agent as afterthought
- **Retrofitting Pain**: Adding DTOs halfway through exposed architectural rot
- **No Scale Path**: Going from n=1 to n=100 requires complete rewrite

### 1.2 Why Greenfield ("Townlet")?

Starting fresh as `src/townlet/` (sibling to `src/hamlet/`) gives us:

1. **GPU-native from line 1** - No compromises with existing code
2. **Hamlet as oracle** - Validate correctness by comparing against reference implementation
3. **Clean interfaces** - Design for scale (n=100) from day 1, test incrementally (n=1→10)
4. **Pedagogical value** - Show students "wrong way vs right way"
5. **Easy retirement** - When Townlet proves out, `rm -rf src/hamlet/`

### 1.3 Core Mission

**Prove that agents can learn indefinite survival with pure sparse rewards after proper curriculum preparation.**

This is the "final boss" demonstration: No shaped rewards, no cheating, just terminal feedback and survival time. The agent must learn robust multi-objective optimization through:
- Adversarial curriculum (auto-tuning difficulty)
- Exploration strategies (epsilon-greedy, RND, adaptive intrinsic)
- Population-based training (up to 100 agents on A100)

---

## 2. Goals & Success Criteria

### 2.1 Primary Goals

**Goal 1: Demonstrate Pure Sparse Reward Learning**
- Agent survives indefinitely (1000+ steps) with only terminal rewards
- No dense feedback after curriculum graduation
- Curriculum: shaped → sparse transition via task complexity progression

**Goal 2: Enable Multiple Exploration Approaches**
- Pure sparse (epsilon-greedy only)
- Adaptive intrinsic (RND with auto-annealing)
- Permanent intrinsic (exploration scaffold)
- Configurable via YAML for A/B comparison

**Goal 3: Scale to 100 Agents on A100**
- GPU-native vectorized environment
- Target: >10K FPS (100 agents × 100 Hz)
- Memory: <12GB VRAM
- Test incrementally: 1 → 2 → 5 → 10 → 20 → 50 → 100

### 2.2 Success Criteria

**Must Have**:
- ✅ Indefinite survival with pure sparse rewards (no shaped rewards at endpoint)
- ✅ Policy robustness (transfers to randomized affordance positions, depletion rates)
- ✅ Interpretability (clear Q-value landscapes, sensible action distributions)
- ✅ Scales to n=10 on consumer GPU (2-4 hours per 1000 episodes)
- ✅ Oracle validation (Townlet matches Hamlet's shaped rewards at n=1)

**Nice to Have**:
- 🎯 Sample efficiency (converges by 50K episodes)
- 🎯 Genetic reproduction (population evolves over generations)
- 🎯 Pareto frontier tracking (survival vs exploration trade-offs)

### 2.3 Non-Goals (Phase 1-3)

- ❌ POMDP / partial observability (Level 2 feature, future work)
- ❌ Multi-agent competition (Level 4 feature, future work)
- ❌ Real-time web visualization during training (metrics only)
- ❌ Production deployment (research prototype)

---

## 3. Architecture Overview

### 3.1 Dual-Path Design

**Hot Path** (training loop):
- Batched PyTorch tensors: `[num_agents, ...]`
- GPU-optimized operations (no Python loops)
- Minimal overhead: action selection, reward calculation, state updates
- Runs every step for every agent

**Cold Path** (config/checkpoints/telemetry):
- Pydantic DTOs with validation
- CPU-friendly, human-readable (JSON/YAML)
- Runs once per episode or checkpoint
- Overhead acceptable

**Boundary**: `detach_cpu_summary()` extracts data from hot path for cold path

### 3.2 Core Data Contracts

```python
# Hot Path: Training Loop (Tensors)
class BatchedAgentState:
    """Vectorized agent state [num_agents, ...] for GPU."""
    observations: torch.Tensor      # [batch, obs_dim]
    actions: torch.Tensor           # [batch]
    rewards: torch.Tensor           # [batch]
    dones: torch.Tensor             # [batch] bool
    epsilons: torch.Tensor          # [batch]
    intrinsic_rewards: torch.Tensor # [batch]
    survival_times: torch.Tensor    # [batch]
    curriculum_difficulties: torch.Tensor  # [batch]
    device: torch.device

# Cold Path: Configuration (Pydantic)
class CurriculumDecision(BaseModel):
    """Validated curriculum settings."""
    difficulty_level: float = Field(..., ge=0.0, le=1.0)
    active_meters: List[str] = Field(..., min_length=1, max_length=6)
    depletion_multiplier: float = Field(..., gt=0.0, le=10.0)
    reward_mode: str = Field(..., pattern=r'^(shaped|sparse)$')
    reason: str  # Document WHY this decision

class ExplorationConfig(BaseModel):
    """Exploration strategy configuration."""
    strategy_type: str  # epsilon_greedy, rnd, adaptive_intrinsic
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.995, gt=0.0, le=1.0)
    intrinsic_weight: float = Field(default=0.0, ge=0.0)
    rnd_hidden_dim: int = Field(default=256, gt=0)

class PopulationCheckpoint(BaseModel):
    """Serializable population state."""
    generation: int
    num_agents: int = Field(..., ge=1, le=1000)
    agent_ids: List[str]
    curriculum_states: Dict[str, Dict[str, Any]]
    exploration_states: Dict[str, Dict[str, Any]]
    pareto_frontier: List[str]
```

### 3.3 Component Interfaces

**All interfaces accept/return tensors for GPU compatibility.**

```python
class CurriculumManager(ABC):
    """Manages environment difficulty progression."""

    @abstractmethod
    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: List[str],
    ) -> List[CurriculumDecision]:
        """Get curriculum decisions for batch (once per episode)."""
        pass

    @abstractmethod
    def checkpoint_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        pass

class ExplorationStrategy(ABC):
    """Manages exploration vs exploitation."""

    @abstractmethod
    def select_actions(
        self,
        q_values: torch.Tensor,  # [batch, num_actions]
        agent_states: BatchedAgentState,
    ) -> torch.Tensor:
        """Select actions for batch (GPU, every step)."""
        pass

    @abstractmethod
    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
    ) -> torch.Tensor:
        """Compute intrinsic motivation (GPU)."""
        pass

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> None:
        """Update exploration networks (RND, ICM)."""
        pass

class PopulationManager(ABC):
    """Coordinates multiple agents (vectorized)."""

    @abstractmethod
    def step_population(
        self,
        envs: VectorizedHamletEnv,  # One vectorized env, not list
    ) -> BatchedAgentState:
        """Execute one training step for entire population."""
        pass

    @abstractmethod
    def get_checkpoint(self) -> PopulationCheckpoint:
        pass
```

### 3.4 Project Structure

```
hamlet/
├── src/
│   ├── hamlet/              # Legacy (retire when Townlet proven)
│   │   ├── environment/
│   │   ├── agent/
│   │   └── training/        # Reusable infrastructure
│   │
│   └── townlet/             # GPU-native sparse reward system
│       ├── __init__.py
│       ├── environment/
│       │   ├── __init__.py
│       │   ├── vectorized_env.py      # Core: VectorizedHamletEnv
│       │   └── config.py              # Environment configuration
│       │
│       ├── curriculum/
│       │   ├── __init__.py
│       │   ├── base.py                # CurriculumManager ABC
│       │   ├── static.py              # StaticCurriculum (Phase 1)
│       │   └── adversarial.py         # AdversarialCurriculum (Phase 2)
│       │
│       ├── exploration/
│       │   ├── __init__.py
│       │   ├── base.py                # ExplorationStrategy ABC
│       │   ├── epsilon_greedy.py      # EpsilonGreedy (Phase 1)
│       │   ├── rnd.py                 # RNDExploration (Phase 3)
│       │   └── adaptive_intrinsic.py  # AdaptiveIntrinsic (Phase 3)
│       │
│       ├── population/
│       │   ├── __init__.py
│       │   ├── base.py                # PopulationManager ABC
│       │   └── vectorized.py          # VectorizedPopulation (Phase 1)
│       │
│       └── training/
│           ├── __init__.py
│           ├── state.py               # DTOs (hot/cold path)
│           └── trainer.py             # Orchestrator (reuses Hamlet infra)
│
├── tests/
│   ├── test_hamlet/
│   └── test_townlet/
│       ├── test_vectorized_env.py
│       ├── test_curriculum/
│       ├── test_exploration/
│       ├── test_population/
│       ├── test_oracle_validation.py  # Cross-validate against Hamlet
│       └── test_scaling.py            # n=1,2,5,10,20,50,100
│
├── configs/
│   └── townlet/
│       ├── sparse_pure.yaml           # Pure sparse (epsilon only)
│       ├── sparse_adaptive.yaml       # Adaptive intrinsic (RND)
│       └── sparse_permanent.yaml      # Permanent intrinsic scaffold
│
└── pyproject.toml                     # Both packages declared
```

---

## 4. Component Designs

### 4.1 VectorizedHamletEnv

**Purpose**: GPU-native environment supporting `num_envs` parallel agents.

**State Representation**:
```python
agent_positions: Tensor[num_envs, 2]        # (x, y) coordinates
meters: Tensor[num_envs, 6]                 # energy, hygiene, satiation, money, mood, social
affordance_positions: Tensor[num_aff, 2]    # Shared across all envs
step_counts: Tensor[num_envs]               # Episode lengths
```

**Observation**: `[num_envs, 70]`
- Grid encoding: 64 dims (one-hot agent position in 8×8 grid)
- Meters: 4 dims (energy, hygiene, satiation, money normalized 0-1)
- Proximity: 2 dims (distance to nearest affordance)

**Actions**: `[num_envs]` int tensor (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=INTERACT)

**Key Methods**:
```python
def reset(mask: Optional[Tensor] = None) -> Tensor:
    """Reset environments (selective via mask). Returns obs [num_envs, 70]."""

def step(actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Execute actions. Returns (obs, rewards, dones, infos)."""

def _calculate_shaped_rewards() -> Tensor:
    """Vectorized shaped reward (Hamlet compatibility)."""

def _calculate_sparse_rewards() -> Tensor:
    """Vectorized sparse reward (survival + optional healthy bonus)."""
```

**Vectorization Examples**:
```python
# Movement (vectorized)
deltas = torch.tensor([[0,-1], [0,1], [-1,0], [1,0], [0,0]])  # UP,DOWN,LEFT,RIGHT,INTERACT
self.agent_positions += deltas[actions]
self.agent_positions.clamp_(0, grid_size-1)

# Reward calculation (vectorized thresholding)
normalized = meters / 100.0
rewards += torch.where(normalized > 0.8, 0.4, 0.0).sum(dim=1)
rewards += torch.where((normalized > 0.5) & (normalized <= 0.8), 0.15, 0.0).sum(dim=1)

# Termination check (vectorized)
fatal = meters[:, [0,1,2,4]]  # energy, hygiene, satiation, mood
dones = (fatal <= 0).any(dim=1) | (meters[:, 3] <= -100)  # Fatal depleted OR bankrupt
```

### 4.2 Adversarial Curriculum

**Purpose**: Auto-tune environment difficulty based on agent performance.

**Performance Metrics** (multi-signal):
1. **Survival time**: Avg over last N episodes
2. **Learning progress**: Reward improvement slope
3. **Policy entropy**: Action distribution entropy (prevents premature convergence)

**Decision Logic**:
```python
if avg_survival > threshold_high AND learning_progress > 0 AND entropy < 0.5:
    # Agent has mastered current difficulty
    increase_difficulty()
elif avg_survival < threshold_low OR learning_progress < 0:
    # Agent is struggling
    decrease_difficulty()
else:
    # Maintain current difficulty
    pass
```

**Difficulty Levers**:
- `depletion_multiplier`: 0.1 (10x slower) to 1.0 (normal) to 2.0 (2x faster)
- `active_meters`: Start with ["energy", "hygiene"], add ["satiation"], then ["money", "mood", "social"]
- `reward_mode`: "shaped" → "sparse" transition at final stage

**Curriculum Stages** (example progression):
```
Stage 1: [energy, hygiene] at 0.2x depletion, shaped rewards
Stage 2: [energy, hygiene, satiation] at 0.5x depletion, shaped rewards
Stage 3: [energy, hygiene, satiation, money] at 0.8x depletion, shaped rewards
Stage 4: [all 6 meters] at 1.0x depletion, shaped rewards
Stage 5: [all 6 meters] at 1.0x depletion, SPARSE rewards (graduation!)
```

**State Tracking** (per agent):
```python
class CurriculumState:
    current_stage: int
    survival_history: Deque[int]  # Last 100 episodes
    reward_history: Deque[float]
    entropy_history: Deque[float]
    episodes_at_stage: int
    total_episodes: int
```

### 4.3 Exploration Strategies

#### 4.3.1 EpsilonGreedy (Phase 1)

**Simplest strategy**: ε-greedy with exponential decay.

```python
def select_actions(q_values: Tensor, agent_states: BatchedAgentState) -> Tensor:
    """Vectorized epsilon-greedy sampling."""
    batch_size = q_values.shape[0]

    # Random sampling for exploration
    explore_mask = torch.rand(batch_size, device=device) < agent_states.epsilons
    random_actions = torch.randint(0, num_actions, (batch_size,), device=device)

    # Greedy actions
    greedy_actions = q_values.argmax(dim=1)

    # Select based on mask
    actions = torch.where(explore_mask, random_actions, greedy_actions)

    return actions
```

**No intrinsic rewards**: `compute_intrinsic_rewards()` returns zeros.

#### 4.3.2 RNDExploration (Phase 3)

**Random Network Distillation**: Predict fixed random network to encourage novelty.

**Architecture**:
- Fixed network: `obs → embedding` (frozen random weights)
- Predictor network: `obs → embedding` (trained to match fixed)
- Intrinsic reward: `||fixed(obs) - predictor(obs)||²`

**High prediction error** = novel state = high intrinsic reward.

```python
def compute_intrinsic_rewards(observations: Tensor) -> Tensor:
    """RND intrinsic motivation."""
    with torch.no_grad():
        target_features = self.fixed_network(observations)  # [batch, embed_dim]

    predicted_features = self.predictor_network(observations)

    # MSE as intrinsic reward
    intrinsic = ((target_features - predicted_features) ** 2).mean(dim=1)

    return intrinsic * self.intrinsic_weight

def update(batch: Dict[str, Tensor]) -> None:
    """Train predictor to match fixed network."""
    observations = batch['observations']

    target_features = self.fixed_network(observations).detach()
    predicted_features = self.predictor_network(observations)

    loss = F.mse_loss(predicted_features, target_features)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

#### 4.3.3 AdaptiveIntrinsicExploration (Phase 3)

**Extension of RND**: Auto-anneal intrinsic weight based on competence.

**Annealing Signal**: When agent demonstrates competence (survival time variance decreases), reduce intrinsic weight.

```python
def adapt_intrinsic_weight(self):
    """Reduce intrinsic weight as agent masters environment."""
    survival_variance = torch.var(self.recent_survival_times)

    if survival_variance < self.variance_threshold:
        # Low variance = agent is consistent = competent
        self.intrinsic_weight *= self.decay_rate
        self.intrinsic_weight = max(self.intrinsic_weight, self.min_weight)
```

**Config**:
```yaml
exploration:
  strategy_type: adaptive_intrinsic
  epsilon: 0.1  # Low epsilon (rely on intrinsic)
  intrinsic_weight: 1.0  # Start high
  decay_rate: 0.99
  min_weight: 0.01
  variance_threshold: 100.0
```

### 4.4 Population Manager

**Purpose**: Coordinate multiple agents, track Pareto frontier, handle reproduction (future).

**Phase 1 Implementation** (n=1 to 10):
```python
class VectorizedPopulation(PopulationManager):
    def __init__(self, num_agents: int, config: Config, device: torch.device):
        self.num_agents = num_agents
        self.device = device

        # One vectorized environment for all agents
        self.env = VectorizedHamletEnv(num_agents, config.environment, device)

        # One shared Q-network (or per-agent networks)
        self.q_network = create_network(config.agent.network_type, device)

        # Curriculum manager (tracks per-agent state)
        self.curriculum = create_curriculum(config.curriculum)

        # Exploration strategy
        self.exploration = create_exploration(config.exploration, device)

        # Replay buffer (shared or per-agent)
        self.replay_buffer = ReplayBuffer(config.training.buffer_capacity)

    def step_population(self, envs: VectorizedHamletEnv) -> BatchedAgentState:
        """One training step for all agents."""
        # Get observations [num_agents, obs_dim]
        obs = envs._get_observations()

        # Forward pass (single batch)
        q_values = self.q_network(obs)  # [num_agents, num_actions]

        # Exploration selects actions
        actions = self.exploration.select_actions(q_values, self.current_state)

        # Environment step (vectorized)
        next_obs, rewards, dones, infos = envs.step(actions)

        # Compute intrinsic rewards
        intrinsic = self.exploration.compute_intrinsic_rewards(obs)
        total_rewards = rewards + intrinsic

        # Store transitions
        for i in range(self.num_agents):
            self.replay_buffer.add((
                obs[i], actions[i], total_rewards[i], next_obs[i], dones[i]
            ))

        # Learning update (if enough samples)
        if len(self.replay_buffer) > self.learning_starts:
            self.train_step()

        # Reset done environments
        if dones.any():
            envs.reset(mask=dones)

        # Return state for telemetry
        return BatchedAgentState(
            observations=next_obs,
            actions=actions,
            rewards=total_rewards,
            dones=dones,
            epsilons=self.epsilons,
            intrinsic_rewards=intrinsic,
            survival_times=infos['survival_times'],
            curriculum_difficulties=torch.zeros(self.num_agents),  # Populated by curriculum
            device=self.device,
        )
```

**Pareto Frontier Tracking** (Phase 4):
- Objectives: (survival_time, -sample_efficiency)
- Track non-dominated agents
- Visualize in TensorBoard

---

## 5. Implementation Phases

### Phase 0: Foundation (2-3 days)

**Goal**: Define contracts, zero implementations.

**Deliverables**:
```
src/townlet/training/state.py          # DTOs (Pydantic + BatchedAgentState)
src/townlet/curriculum/base.py         # CurriculumManager ABC
src/townlet/exploration/base.py        # ExplorationStrategy ABC
src/townlet/population/base.py         # PopulationManager ABC
```

**Tests**:
- `test_dto_validation`: Pydantic catches invalid values
- `test_batched_state_shapes`: Tensor dimensions correct
- `test_interface_contracts`: ABCs can't be instantiated
- `test_dto_serialization`: JSON round-trip

**Exit Criteria**:
- ✅ `mypy --strict` passes on all interfaces
- ✅ All DTOs have validation tests (epsilon ∈ [0,1], difficulty ∈ [0,1], etc.)
- ✅ `BatchedAgentState` can be constructed, moved to device, detached to CPU

### Phase 1: GPU Infrastructure + Trivial Implementations (5-7 days)

**Goal**: Vectorized environment + simplest strategy implementations at n=1.

**Deliverables**:
```
src/townlet/environment/vectorized_env.py   # VectorizedHamletEnv (full GPU)
src/townlet/curriculum/static.py            # StaticCurriculum (no adaptation)
src/townlet/exploration/epsilon_greedy.py   # EpsilonGreedy (vectorized)
src/townlet/population/vectorized.py        # VectorizedPopulation (n=1 to 10)
```

**Tests**:
- `test_vectorized_env_step`: Forward pass works, returns correct shapes
- `test_shaped_rewards_vectorized`: Matches Hamlet reference (oracle validation)
- `test_epsilon_greedy_sampling`: Action distribution matches expected epsilon
- `test_train_one_episode`: Full training loop completes at n=1
- `test_checkpoint_save_restore`: Can save/load and resume training

**Exit Criteria**:
- ✅ Single agent (n=1) trains successfully with GPU implementation
- ✅ Works on both CPU (`device='cpu'`) and GPU (`device='cuda'`)
- ✅ Oracle validation: Townlet matches Hamlet shaped rewards within 1e-4
- ✅ Checkpoints save/restore correctly

### Phase 2: Adversarial Curriculum (5-7 days)

**Goal**: Auto-tuning difficulty based on survival + learning + entropy.

**Deliverables**:
```
src/townlet/curriculum/adversarial.py   # AdversarialCurriculum
configs/townlet/curriculum_test.yaml     # Test config with fast progression
```

**Tests**:
- `test_difficulty_increases_on_mastery`: High survival → higher difficulty
- `test_difficulty_decreases_on_struggle`: Low survival → lower difficulty
- `test_entropy_prevents_premature_ramp`: Low entropy blocks difficulty increase
- `test_curriculum_state_checkpoint`: Can resume mid-stage
- `test_shaped_to_sparse_transition`: Final stage switches to sparse rewards

**Integration Test**:
```python
def test_curriculum_progression_end_to_end():
    """Agent should progress through stages and reach sparse mode."""
    config = create_config(curriculum='adversarial')
    population = VectorizedPopulation(num_agents=1, config=config)

    # Train until sparse mode reached
    for episode in range(5000):  # May take 2000-5000 episodes
        population.train_episode()

        if population.curriculum.current_stage == 5:  # Sparse stage
            break

    assert population.curriculum.reward_mode == 'sparse'
    assert population.curriculum.current_stage == 5
```

**Exit Criteria**:
- ✅ Agent progresses through curriculum stages automatically
- ✅ Can reach sparse mode (stage 5) within 5000 episodes
- ✅ Can resume training mid-curriculum from checkpoint

### Phase 3: Intrinsic Exploration (5-7 days)

**Goal**: RND and adaptive intrinsic motivation.

**Deliverables**:
```
src/townlet/exploration/rnd.py                  # RNDExploration
src/townlet/exploration/adaptive_intrinsic.py   # AdaptiveIntrinsicExploration
configs/townlet/sparse_adaptive.yaml            # Adaptive intrinsic config
```

**Tests**:
- `test_rnd_novelty_decreases`: Prediction error decreases for repeated states
- `test_adaptive_annealing_triggers`: Intrinsic weight decays when competent
- `test_intrinsic_reward_added`: Total reward = extrinsic + intrinsic
- `test_curriculum_and_exploration_together`: Both systems work simultaneously

**Integration Test**:
```python
def test_sparse_learning_with_intrinsic():
    """Agent should learn sparse reward task with intrinsic motivation."""
    config = create_config(
        curriculum='adversarial',
        exploration='adaptive_intrinsic',
        reward_mode='sparse'
    )
    population = VectorizedPopulation(num_agents=1, config=config)

    # Train in sparse mode
    for episode in range(10000):
        population.train_episode()

    # Agent should survive longer than random
    avg_survival = population.get_avg_survival(last_n=100)
    assert avg_survival > 100  # Random baseline ~50 steps
```

**Exit Criteria**:
- ✅ RND provides novelty signal (high reward for new states)
- ✅ Adaptive intrinsic weight anneals automatically
- ✅ Agent learns better with intrinsic motivation than pure epsilon-greedy

### Phase 4: Scale Testing (7-10 days)

**Goal**: Validate architecture scales to n=10 without code changes.

**Deliverables**:
```
tests/test_townlet/test_scaling.py   # Parameterized tests at n=1,2,5,10
scripts/profile_scaling.py            # Memory/time profiling
docs/SCALING_REPORT.md                # Document bottlenecks
```

**Scaling Checkpoints**:
```
n=1: Baseline (already working from Phase 1-3)
n=2: Expose parallelism bugs (agents interfere?)
n=5: Stress test coordination overhead
n=10: Production baseline (overnight training feasible?)
```

**Tests at Each Scale**:
- `test_population_trains_successfully`: All agents complete training
- `test_independent_curricula`: Each agent's curriculum progresses independently
- `test_memory_scales_linearly`: Memory = O(n), not O(n²)
- `test_training_time_scales_sublinearly`: Batching provides speedup

**Profiling**:
```bash
# Memory profiling
python -m memory_profiler scripts/profile_scaling.py --num-agents 10

# Time profiling
python -m cProfile -o profile.stats scripts/profile_scaling.py --num-agents 10
python -m pstats profile.stats
```

**Exit Criteria**:
- ✅ All tests pass at n=1, 2, 5, 10
- ✅ n=10 training completes overnight (<8 hours per 1000 episodes)
- ✅ Memory usage < 8GB for n=10 on consumer GPU
- ✅ Identified bottlenecks documented for Phase 5 optimization

### Phase 5: Large-Scale Optimization (5-7 days, future)

**Goal**: Scale to n=20, 50 with CPU/GPU optimizations.

**Optimizations**:
- Shared replay buffer (memory efficiency)
- Mixed precision training (FP16/BF16)
- Gradient checkpointing
- Profiler-guided vectorization

**Exit Criteria**:
- ✅ n=50 completes training in <2 days

### Phase 6: A100 Demo (10-14 days, future)

**Goal**: 100 agents on A100, >10K FPS.

**Optimizations**:
- Multi-GPU training (if needed)
- Kernel fusion
- Memory-efficient attention (if using transformers)

**Exit Criteria**:
- ✅ n=100 achieves >10K FPS on A100
- ✅ Memory < 12GB VRAM
- ✅ 100 agents train to convergence in <1 week

---

## 6. Testing Strategy

### 6.1 Oracle Validation (Critical!)

**Purpose**: Prove Townlet is correct by comparing to Hamlet reference.

```python
# tests/test_townlet/test_oracle_validation.py

@pytest.mark.parametrize("num_steps", [10, 100, 500])
def test_shaped_rewards_match_hamlet(num_steps):
    """Townlet must produce same rewards as Hamlet for same action sequence."""
    from hamlet.environment.hamlet_env import HamletEnv
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    config = EnvironmentConfig(reward_mode="shaped")

    # Hamlet reference
    hamlet = HamletEnv(config)
    hamlet_obs = hamlet.reset()

    # Townlet (n=1)
    townlet = VectorizedHamletEnv(num_envs=1, config=config, device=torch.device('cpu'))
    townlet_obs = townlet.reset()

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    hamlet_rewards = []
    townlet_rewards = []

    for _ in range(num_steps):
        # Same action for both
        action = np.random.randint(0, 5)

        # Hamlet step
        _, hamlet_reward, hamlet_done, _ = hamlet.step(action)
        hamlet_rewards.append(hamlet_reward)

        # Townlet step
        actions = torch.tensor([action])
        _, townlet_reward_tensor, townlet_done, _ = townlet.step(actions)
        townlet_rewards.append(townlet_reward_tensor[0].item())

        if hamlet_done or townlet_done[0]:
            break

    # Compare reward sequences
    np.testing.assert_allclose(hamlet_rewards, townlet_rewards, rtol=1e-4)
```

### 6.2 Scaling Tests (Parameterized)

```python
# tests/test_townlet/test_scaling.py

@pytest.mark.parametrize("num_agents", [1, 2, 5, 10])
class TestScaling:
    def test_population_trains(self, num_agents):
        """All agents should complete training without errors."""
        config = create_config(num_agents=num_agents)
        population = VectorizedPopulation(num_agents, config, device)

        for episode in range(10):  # Quick smoke test
            population.train_episode()

        assert (population.survival_times > 0).all()

    def test_memory_usage(self, num_agents):
        """Memory should scale linearly."""
        mem_before = get_memory_usage()
        population = create_population(num_agents)
        population.train_episode()
        mem_after = get_memory_usage()

        mem_per_agent = (mem_after - mem_before) / num_agents
        assert mem_per_agent < 200  # MB per agent

    def test_training_time(self, num_agents):
        """Time should scale sublinearly (batching helps)."""
        start = time.time()
        population = create_population(num_agents)
        population.train_episode()
        elapsed = time.time() - start

        time_per_agent = elapsed / num_agents
        if num_agents == 1:
            assert time_per_agent < 2.0
        elif num_agents <= 10:
            assert time_per_agent < 3.0  # Some overhead
```

### 6.3 Property-Based Testing (Hypothesis)

```python
# tests/test_townlet/test_properties.py

from hypothesis import given, strategies as st

@given(
    survival_times=st.lists(st.integers(min_value=0, max_value=1000), min_size=10),
    learning_rates=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=10)
)
def test_curriculum_difficulty_always_valid(survival_times, learning_rates):
    """Property: Curriculum should never produce invalid difficulty."""
    curriculum = AdversarialCurriculum()

    for survival, lr in zip(survival_times, learning_rates):
        state = create_mock_state(survival_time=survival)
        decision = curriculum.get_batch_decisions(state, agent_ids=["agent_0"])[0]

        # Properties that must ALWAYS hold
        assert 0.0 <= decision.difficulty_level <= 1.0
        assert decision.depletion_multiplier > 0
        assert decision.reward_mode in {"shaped", "sparse"}
        assert len(decision.reason) > 0

@given(
    q_values=st.lists(st.floats(min_value=-100, max_value=100), min_size=5, max_size=5),
    epsilon=st.floats(min_value=0.0, max_value=1.0)
)
def test_epsilon_greedy_distribution(q_values, epsilon):
    """Property: Epsilon-greedy should respect epsilon probability."""
    exploration = EpsilonGreedyExploration()

    # Run many samples
    q_tensor = torch.tensor([q_values])
    state = create_mock_state(epsilon=epsilon)

    actions = []
    for _ in range(1000):
        action = exploration.select_actions(q_tensor, state)
        actions.append(action[0].item())

    # Greedy action
    greedy = np.argmax(q_values)

    # Frequency of greedy action should be roughly (1 - epsilon + epsilon/5)
    expected_greedy_freq = (1 - epsilon) + (epsilon / 5)
    actual_greedy_freq = actions.count(greedy) / len(actions)

    # Allow some statistical variance
    assert abs(actual_greedy_freq - expected_greedy_freq) < 0.1
```

### 6.4 Integration Tests (Long-Running)

```python
@pytest.mark.slow
def test_sparse_learning_end_to_end():
    """Full curriculum → sparse → indefinite survival."""
    config = create_config(
        curriculum='adversarial',
        exploration='adaptive_intrinsic',
        num_episodes=10000
    )

    population = VectorizedPopulation(num_agents=1, config=config, device='cuda')

    # Train through curriculum
    for episode in range(10000):
        population.train_episode()

        # Check if reached sparse mode
        if population.curriculum.reward_mode == 'sparse':
            sparse_episode = episode
            break

    # Should reach sparse within 5000 episodes
    assert sparse_episode < 5000

    # Continue training in sparse mode
    for episode in range(sparse_episode, 10000):
        population.train_episode()

    # Final performance: indefinite survival
    avg_survival = population.get_avg_survival(last_n=100)
    assert avg_survival > 500  # "Indefinite" = survives many cycles
```

### 6.5 CI/CD Integration

```yaml
# .github/workflows/ci-townlet.yml

name: Townlet CI

on:
  pull_request:
    paths:
      - 'src/townlet/**'
      - 'tests/test_townlet/**'
  push:
    branches: [main, feature/townlet]

jobs:
  lint-and-type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Ruff lint
        run: ruff check src/townlet tests/test_townlet

      - name: Mypy strict (infrastructure only)
        run: |
          mypy src/townlet/training/state.py \
               src/townlet/curriculum/base.py \
               src/townlet/exploration/base.py \
               src/townlet/population/base.py

  unit-tests-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run unit tests (CPU only)
        run: pytest tests/test_townlet -m "not slow and not gpu" --cov-fail-under=80

  unit-tests-gpu:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5

      - name: Run GPU tests
        run: pytest tests/test_townlet -m gpu -v

  oracle-validation:
    runs-on: ubuntu-latest
    steps:
      - name: Validate against Hamlet reference
        run: pytest tests/test_townlet/test_oracle_validation.py -v

  integration-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - name: Run integration tests
        run: pytest tests/test_townlet -m integration -v
```

---

## 7. Tooling & Best Practices

### 7.1 Dependencies & Configuration

**pyproject.toml additions**:

```toml
[project]
name = "hamlet-townlet"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.0.0",
  "numpy>=1.24.0",
  "pydantic>=2.10.0",      # Validation for configs
  "pyyaml>=6.0",
  "pettingzoo>=1.24.0",    # For compatibility (Hamlet uses it)
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "pytest-mock>=3.12.0",
  "pytest-cov>=4.1.0",
  "pytest-xdist>=3.5.0",   # Parallel test execution
  "hypothesis>=6.100.0",    # Property-based testing
  "ruff>=0.14.0",
  "mypy>=1.18.0",
  "pip-audit>=2.9.0",       # Dependency scanning
  "vulture>=2.13",          # Dead code detection
]

[tool.ruff]
line-length = 140
target-version = "py311"

[tool.ruff.lint]
select = [
  "E",   # pycodestyle errors
  "F",   # Pyflakes
  "I",   # Import sorting
  "D",   # pydocstyle
  "S",   # flake8-bandit (security)
  "NPY", # NumPy-specific
  "PT",  # pytest style
]
ignore = [
  "E501",  # Line length (long RL equations)
  "D107",  # __init__ docstrings optional
  "S101",  # Allow asserts in ML code
]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D", "S"]
"demos/**" = ["D"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
check_untyped_defs = true
no_implicit_optional = true
strict_equality = true

[[tool.mypy.overrides]]
module = ["torch.*", "numpy.*", "gymnasium.*"]
ignore_missing_imports = true

# Strict typing on infrastructure
[[tool.mypy.overrides]]
module = [
  "townlet.training.state",
  "townlet.curriculum.base",
  "townlet.exploration.base",
  "townlet.population.base",
]
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
addopts = "-ra --cov=townlet --cov-branch --cov-fail-under=80"
testpaths = ["tests/test_townlet"]
markers = [
  "integration: requires environment simulation (slow)",
  "slow: long-running tests (>1 minute)",
  "gpu: requires CUDA",
]
```

### 7.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml

repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.1
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-yaml
      - id: check-merge-conflict
      - id: detect-private-key
```

### 7.3 Development Workflow

**Day-to-day development**:

```bash
# Setup
git clone <repo>
cd hamlet
pip install -e ".[dev]"
pre-commit install

# Create feature branch
git checkout -b feature/townlet-curriculum

# Run tests (fast)
pytest tests/test_townlet -m "not slow and not gpu" -v

# Run linting
ruff check src/townlet tests/test_townlet
mypy src/townlet/training/state.py  # Strict on interfaces

# Run integration tests (slow)
pytest tests/test_townlet -m integration -v

# Profile performance
python -m cProfile -o profile.stats scripts/train_townlet.py --num-agents 10
python -m pstats profile.stats
```

**CI expectations**:
- Unit tests: <5 minutes
- Integration tests: <30 minutes
- Oracle validation: <10 minutes
- GPU tests: Run on self-hosted GPU runner (not always available)

---

## 8. Migration Plan (Hamlet → Townlet)

### 8.1 Coexistence Strategy

**Phase 1-3**: Both packages coexist
- Hamlet: Production teaching environment (known good)
- Townlet: Experimental sparse reward system
- Shared: `hamlet.training` infrastructure (Trainer, MetricsManager)

**Phase 4**: Townlet proves out
- All new features go to Townlet
- Hamlet enters maintenance mode (bug fixes only)
- Documentation updated to recommend Townlet

**Phase 5**: Retirement
- Hamlet moved to `src/hamlet_legacy/` (archived)
- Townlet becomes primary
- Update all demos, docs, configs to use Townlet

### 8.2 Reusable Hamlet Components

**Keep (reuse in Townlet)**:
- ✅ `hamlet.training.Trainer` (orchestration)
- ✅ `hamlet.training.MetricsManager` (TensorBoard, SQLite, etc.)
- ✅ `hamlet.training.CheckpointManager` (save/load logic)
- ✅ `hamlet.training.ExperimentManager` (MLflow integration)
- ✅ `hamlet.agent.networks` (Q-network architectures)
- ✅ `hamlet.agent.replay_buffer` (experience replay)

**Retire (replaced by Townlet)**:
- ❌ `hamlet.environment.hamlet_env` → `townlet.environment.vectorized_env`
- ❌ `hamlet.agent.drl_agent` → `townlet.population.vectorized`
- ❌ Dense reward logic → Curriculum-managed sparse/shaped switch

### 8.3 Config Compatibility

**Goal**: Same YAML config works for both packages (environment section differs).

```yaml
# configs/townlet/sparse_example.yaml

experiment:
  name: "townlet_sparse_demo"
  description: "Pure sparse rewards with adversarial curriculum"

environment:
  # Townlet-specific
  grid_width: 8
  grid_height: 8
  reward_mode: "sparse"
  sparse_survival_reward: 0.1
  sparse_healthy_meter_bonus: 0.05
  # ... (same as Hamlet EnvironmentConfig)

curriculum:
  type: "adversarial"
  survival_threshold_high: 400
  survival_threshold_low: 100
  learning_progress_window: 50
  entropy_threshold: 0.5

exploration:
  strategy_type: "adaptive_intrinsic"
  epsilon: 0.1
  epsilon_decay: 0.995
  intrinsic_weight: 1.0
  decay_rate: 0.99

agents:
  - agent_id: "agent_0"
    algorithm: "dqn"
    network_type: "spatial_dueling"
    learning_rate: 0.00025
    gamma: 0.99

training:
  num_episodes: 10000
  max_steps_per_episode: 1000
  batch_size: 64
  learning_starts: 1000
  target_update_frequency: 100
  replay_buffer_size: 50000

metrics:
  tensorboard: true
  database: true
  database_path: "townlet_metrics.db"
```

---

## 9. Future Work & Extensions

### 9.1 Phase 7+ (Post-100 Agents)

**Genetic Reproduction**:
- Select parents from Pareto frontier
- Network weight interpolation or config mutation
- Diversity preservation (prevent convergence to single strategy)

**Multi-Zone Environments**:
- Level 3: Home zone, work zone, social zone
- Hierarchical RL (meta-controller selects zone, low-level navigates)

**Partial Observability (POMDP)**:
- Level 2: Agent sees only local 3×3 window
- LSTM/Transformer memory for temporal integration

**Multi-Agent Competition**:
- Level 4: Agents compete for affordances
- Theory of mind (model other agents' intentions)

### 9.2 Research Directions

**Curriculum Learning**:
- Compare adversarial vs. fixed-stage curriculum
- Investigate curriculum "forgetting" (does agent regress when difficulty increases?)

**Exploration Strategies**:
- Prioritized replay buffer (TD-error based)
- Go-Explore (archive + exploration)
- Never Give Up (episodic novelty + RND)

**Algorithm Comparison**:
- DQN vs. PPO vs. SAC on sparse rewards
- Model-based RL (world models for planning)

**Interpretability**:
- Visualize Q-value heatmaps over grid
- Policy distillation to decision trees
- Attention mechanisms for meter importance

---

## 10. Success Metrics & KPIs

### 10.1 Technical Metrics

**Performance** (must meet):
- n=1: <2 seconds per episode (CPU or GPU)
- n=10: <5 seconds per episode (GPU)
- n=100: >10K FPS (A100)

**Memory** (must meet):
- n=1: <500 MB
- n=10: <2 GB
- n=100: <12 GB VRAM

**Learning** (target):
- Sparse mode reached: <5000 episodes (with curriculum)
- Indefinite survival: >500 steps average (last 100 episodes)
- Oracle validation: <1e-4 error vs. Hamlet

### 10.2 Pedagogical Metrics

**Teachable Moments**:
- ✅ Compare dense vs. sparse learning curves (show students the difference)
- ✅ Demonstrate reward hacking (if it emerges)
- ✅ Show curriculum progression (difficulty auto-tunes)
- ✅ Visualize exploration strategies (epsilon vs. intrinsic)

**Demo Quality**:
- ✅ 100-agent A100 demo runs smoothly (<10 minutes to interesting behavior)
- ✅ Web visualization shows live population diversity
- ✅ Can explain every design decision to students

### 10.3 Code Quality Metrics

**Coverage**: >80% on all infrastructure (interfaces, DTOs, environment)
**Type Safety**: 100% mypy strict on interfaces
**Linting**: Zero ruff errors on src/townlet
**Documentation**: Every public function has docstring

---

## 11. Risks & Mitigations

### 11.1 Technical Risks

**Risk**: Sparse rewards too hard, agent never learns
**Mitigation**: Curriculum starts with shaped rewards, gradually transitions. If stuck, extend curriculum stages.

**Risk**: GPU memory overflow at n=100
**Mitigation**: Profile at n=20, 50 to identify bottlenecks. Use gradient checkpointing, mixed precision (FP16).

**Risk**: Oracle validation fails (Townlet != Hamlet)
**Mitigation**: This is critical! Debug immediately if validation fails. Likely sources: floating-point precision, random seed, meter clamping logic.

**Risk**: Vectorized code has subtle bugs (race conditions, broadcasting errors)
**Mitigation**: Extensive testing at n=1, 2, 5 to catch parallelism bugs early. Property-based testing (Hypothesis) for edge cases.

### 11.2 Project Risks

**Risk**: Scope creep (adding genetics, multi-zone, POMDP too early)
**Mitigation**: Strict phase gating. Phase 0-4 only. Defer Phase 5+ to future work.

**Risk**: Integration hell 2.0 (subsystems diverge again)
**Mitigation**: DTOs and interfaces prevent this. All components speak same language (BatchedAgentState). Interface compliance tests catch violations.

**Risk**: Hamlet retirement premature (Townlet not ready)
**Mitigation**: Oracle validation must pass 100%. Townlet must match or exceed Hamlet's pedagogical value before retirement.

---

## 12. Conclusion

Townlet represents a **greenfield opportunity** to build Hamlet's sparse reward system correctly from day 1:

- **GPU-native architecture** eliminates CPU bottlenecks
- **Clean interfaces** prevent integration hell
- **Incremental testing** (1 → 10 → 100) validates scale assumptions
- **Oracle validation** proves correctness against Hamlet reference
- **Pedagogical gold**: Show students the difference between naive and optimized designs

The phased approach (Phases 0-4 for n=1→10, Phases 5-6 for n=50→100) ensures we always have a working system. If Phase 2 curriculum doesn't work, we haven't wasted Phase 3 exploration effort - all components are independently testable.

**Next Steps**:
1. Create feature branch: `git checkout -b feature/townlet-sparse-reward`
2. Set up worktree: `git worktree add ../hamlet-townlet feature/townlet-sparse-reward`
3. Phase 0: Define interfaces (2-3 days)
4. Phase 1: Build VectorizedEnv (5-7 days)
5. Phase 2-3: Curriculum + Exploration (10-14 days)
6. Phase 4: Scale to n=10 (7-10 days)

**Total Timeline**: ~6-8 weeks for Phases 0-4 (single agent mastery at n=1→10). Phases 5-6 (n=50→100) deferred to future work.

---

## Appendix A: Glossary

**Sparse Rewards**: Agent receives feedback only at episode termination (success/failure), not every step. Much harder to learn.

**Shaped Rewards**: Dense feedback every step (gradient rewards for meter states). Easier to learn, but may not generalize.

**Curriculum Learning**: Train on progressively harder tasks. Start easy (slow depletion), gradually increase difficulty.

**Adversarial Curriculum**: Curriculum that auto-tunes difficulty based on agent performance. No manual stage transitions.

**RND (Random Network Distillation)**: Intrinsic motivation by predicting output of fixed random network. High prediction error = novel state.

**Intrinsic Motivation**: Reward signal generated by agent's curiosity (novelty seeking), not environment.

**Pareto Frontier**: Set of non-dominated solutions in multi-objective optimization (e.g., survival vs. sample efficiency).

**Oracle Validation**: Testing new implementation against trusted reference (Hamlet) to prove correctness.

**Hot Path**: Code that runs every step, performance-critical (must be GPU-optimized).

**Cold Path**: Code that runs rarely (per episode, per checkpoint), can be CPU, human-readable.

**DTO (Data Transfer Object)**: Immutable data structure passed between components (Pydantic models).

**Vectorization**: Batch operations on arrays/tensors instead of Python loops. Essential for GPU performance.

---

## Appendix B: References

**Papers**:
- Burda et al. (2018): "Exploration by Random Network Distillation" (RND)
- Andrychowicz et al. (2017): "Hindsight Experience Replay" (sparse reward learning)
- Bengio et al. (2009): "Curriculum Learning"
- Graves et al. (2017): "Automated Curriculum Learning for Neural Networks"

**Codebases**:
- Hamlet: `~/hamlet/` (reference implementation)
- Elspeth: `~/elspeth/` (best practices reference for tooling)

**Tools**:
- PyTorch: Deep learning framework
- Pydantic: Data validation library
- Hypothesis: Property-based testing
- Ruff: Fast Python linter
- Mypy: Static type checker

---

**Document Version**: 1.0
**Last Updated**: 2025-10-29
**Authors**: John + Claude (brainstorming collaboration)
**Status**: Ready for Implementation
