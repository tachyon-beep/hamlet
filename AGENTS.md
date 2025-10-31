# Hamlet Project: AI Agent Memory Document

**Last Updated:** October 31, 2025  
**Purpose:** Comprehensive project documentation for AI assistants and future developers  
**Current Branch:** main  
**Test Coverage:** 16% (250/1521 statements)  
**Authority Order:** ROADMAP.md > actual code > CLAUDE.md (CLAUDE.md is outdated)

---

## Executive Summary

Hamlet is a **pedagogical Deep Reinforcement Learning environment** designed to "trick students into learning graduate-level RL by making them think they're just playing The Sims." Agents learn to survive in a grid world by managing 8 interconnected meters (energy, hygiene, satiation, money, mood, social, health, fitness) through interactions with 15 affordances.

### Current Status: Phase 3 Complete, Phase 3.5 Next

**Phase 3 (✅ COMPLETE):** Intrinsic Exploration - 37/37 tests passing

- RND (Random Network Distillation) for novelty detection
- Adaptive intrinsic motivation with variance-based annealing
- Dual reward system (extrinsic + intrinsic)
- All tests passing, system validated

**Phase 3.5 (🎯 NEXT):** Multi-Day Tech Demo

- Validate system stability over 48+ hours (10K episodes)
- Observe exploration→exploitation transition in production
- Generate teaching materials from real training data
- **Purpose:** Validate foundation before adding POMDP complexity

**Strategic Direction:** 2→3→1 approach

- Multi-Day Demo (validate) → POMDP Extension (add complexity) → Informed Optimization (profile real system)
- **Rationale:** Avoid optimizing twice by waiting until POMDP reveals real bottlenecks

### Key Features

- **Vectorized GPU training** with PyTorch tensors throughout
- **Adversarial curriculum learning** (5 progressive stages from shaped to sparse rewards)
- **RND-based intrinsic motivation** with adaptive annealing
- **Partial observability (POMDP)** support with recurrent neural networks (Level 2 implemented)
- **Temporal mechanics** with time-of-day cycles and multi-tick interactions (Level 2.5 implemented)
- **Live inference server** for real-time visualization during training

**Critical State:** About to undergo major refactoring with only 16% test coverage. RED-GREEN testing approach required.

---

## Project Structure & Entry Points

### ⚠️ Critical: Active vs Legacy Code

**ACTIVE SYSTEM:** `src/townlet/` - All development happens here  
**LEGACY (DO NOT EDIT):** `src/hamlet/` - Obsolete code, will be deleted  
**Exception:** `src/hamlet/demo/runner.py` - Temporary entry point (will move to `townlet/training/`)

### Running the System

```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Training (Multi-Day Demo Setup)
python -m townlet.demo.runner configs/townlet_level_1_5.yaml demo_level1_5.db checkpoints_level1_5 10000

# Live Inference Server (for visualization during training)
python -m townlet.demo.live_inference checkpoints_level1_5 8766 0.2 10000 configs/townlet_level_1_5.yaml
# Args: <checkpoint_dir> <port> <speed> <total_episodes> <config_path>

# Frontend (separate terminal, from main repo not worktree)
cd frontend && npm run dev
# Open http://localhost:5173
```

### Configuration Files (YAML)

**Active Configs:**

- `configs/townlet_level_1_5.yaml` - Full observability baseline (SimpleQNetwork)
- `configs/townlet_level_2_pomdp.yaml` - Partial observability + LSTM (RecurrentSpatialQNetwork)
- `configs/townlet_level_2_5_temporal.yaml` - Temporal mechanics + multi-tick interactions

**Key Config Parameters:**

```yaml
environment:
  grid_size: 8
  partial_observability: false  # true for POMDP
  vision_range: 2  # 5×5 window when partial
  enable_temporal_mechanics: false  # true for time-of-day cycles

population:
  network_type: simple  # or 'recurrent' for LSTM
  learning_rate: 0.00025  # 0.0001 for recurrent networks
  
exploration:
  variance_threshold: 100.0  # Was 10.0 - fixed premature annealing
```

---

## Progressive Complexity Levels

**Level 1** (Obsolete - hamlet legacy): Single agent, full observability  
**Level 1.5** (✅ Current): Full observability, NO proximity shaping, sparse rewards  
**Level 2** (✅ Implemented): Partial observability (POMDP) + LSTM memory  
**Level 2.5** (✅ Implemented): Temporal mechanics + multi-tick interactions  
**Level 3** (🎯 Future): Multi-zone environment with hierarchical RL  
**Level 4** (🎯 Future): Multi-agent competition with theory of mind  
**Level 5** (🎯 Future): Family communication and emergent language

---

## Project Architecture

### Directory Structure

```text
src/townlet/
├── agent/                    # Neural network architectures (0% coverage)
│   ├── networks.py          # SimpleQNetwork, RecurrentSpatialQNetwork
│   └── __init__.py
├── curriculum/              # Progressive difficulty management (0% coverage)
│   ├── adversarial.py       # 5-stage adaptive curriculum with entropy gating
│   ├── static.py            # Fixed difficulty baseline
│   ├── base.py              # Abstract interface
│   └── __init__.py
├── demo/                    # Live inference & visualization (0% coverage)
│   ├── live_inference.py    # WebSocket server for real-time demos
│   ├── runner.py            # Multi-day training orchestration
│   ├── database.py          # SQLite for episode metrics
│   └── __init__.py
├── environment/             # Core RL environment (56% coverage on vectorized_env)
│   ├── vectorized_env.py    # GPU-native vectorized environment ⚠️ COMPLEX
│   ├── affordance_config.py # Multi-tick interaction configs (100% coverage)
│   └── __init__.py
├── exploration/             # Action selection strategies (0% coverage)
│   ├── adaptive_intrinsic.py # RND with variance-based annealing
│   ├── rnd.py               # Random Network Distillation
│   ├── epsilon_greedy.py    # Simple baseline
│   ├── base.py              # Abstract interface
│   └── __init__.py
├── population/              # Agent coordination (0% coverage)
│   ├── vectorized.py        # Training loop orchestration
│   ├── base.py              # Abstract interface
│   └── __init__.py
└── training/                # Supporting infrastructure (0% coverage)
    ├── replay_buffer.py     # Experience replay with dual rewards
    ├── state.py             # DTOs for hot/cold paths
    └── __init__.py

tests/test_townlet/          # Current test suite (19 tests, all passing)
├── test_affordance_config.py       # Affordance structure & math
├── test_multi_interaction.py       # Multi-tick mechanics
├── test_temporal_integration.py    # Time-based systems
├── test_time_based_masking.py      # Operating hours
└── test_vectorized_env_temporal.py # Environment temporal features
```

---

## Core Components Deep Dive

### 1. Environment (`vectorized_env.py`) - 1262 lines - **HIGHEST RISK**

**Complexity Level:** 🔴 EXTREME  
**Coverage:** 56% (241/429 statements)  
**Refactoring Priority:** CRITICAL

#### Responsibilities (Too Many!)

1. **Grid Management** - Agent positions, affordance locations
2. **Meter System** - 8-meter differential equations with cascading effects
3. **Observation Construction** - Full vs partial observability modes
4. **Action Execution** - Movement, interaction, action masking
5. **Temporal Mechanics** - Time-of-day cycles, multi-tick interactions
6. **Reward Calculation** - Milestone bonuses, shaped rewards (complex, mostly disabled)
7. **Terminal Conditions** - Death conditions, cascading meter failures

#### Key Methods

**Reset & Observation (Lines 1-230)**

```python
def reset() -> torch.Tensor
def _get_observations() -> torch.Tensor
def _get_full_observations() -> torch.Tensor  # Level 1: full grid
def _get_partial_observations() -> torch.Tensor  # Level 2 POMDP: 5x5 window
def _get_current_affordance_encoding() -> torch.Tensor  # One-hot affordances
```

**Action System (Lines 230-400)**

```python
def get_action_masks() -> torch.Tensor  # Boundary + affordability checks
def step(actions) -> Tuple[obs, rewards, dones, info]
def _execute_actions(actions) -> dict  # Movement + INTERACT
def _handle_interactions(interact_mask) -> dict  # Multi-tick mechanics
def _handle_interactions_legacy(interact_mask) -> dict  # Single-shot fallback
```

**Meter Dynamics (Lines 700-950) - CASCADING DIFFERENTIAL EQUATIONS**

```python
def _deplete_meters() -> None  # Base passive decay
def _apply_secondary_to_primary_effects() -> None  # Satiation/Fitness/Mood → Health/Energy
def _apply_tertiary_to_secondary_effects() -> None  # Hygiene/Social → Satiation/Fitness/Mood
def _apply_tertiary_to_primary_effects() -> None  # Hygiene/Social → Health/Energy (weak)
def _check_dones() -> None  # Death conditions
```

**Reward System (Lines 950-1150)**

```python
def _calculate_shaped_rewards() -> torch.Tensor  # Milestone bonuses (ACTIVE)
def _calculate_shaped_rewards_COMPLEX_DISABLED() -> torch.Tensor  # Per-step meter rewards (DISABLED - caused negative accumulation)
def _calculate_proximity_rewards() -> torch.Tensor  # Proximity shaping (DISABLED - reward hacking)
```

#### Meter System Architecture

**PRIMARY (Death Conditions):**

- `health` [6]: Are you alive?
- `energy` [0]: Can you move?

**SECONDARY (Strong → Primary):**

- `satiation` [2] → health AND energy (FUNDAMENTAL - affects both!)
- `fitness` [7] → health (unfit → sick → death)
- `mood` [4] → energy (depressed → exhausted → death)

**TERTIARY (Quality of Life):**

- `hygiene` [1] → secondary (strong) + primary (weak)
- `social` [5] → secondary (strong) + primary (weak)

**RESOURCE:**

- `money` [3]: Enables affordances ($0-$100 = 0.0-1.0 normalized)

#### Refactoring Opportunities

**Option A: Vertical Slice by Feature**

```
environment/
├── core/              # Grid, positions, observations
├── meters/            # Meter dynamics, cascading effects
├── actions/           # Movement, interactions, masking
├── rewards/           # Reward calculation strategies
└── temporal/          # Time-based mechanics
```

**Option B: Horizontal Slice by Concern**

```
environment/
├── state.py           # Environment state management
├── dynamics.py        # Meter differential equations
├── interactions.py    # Affordance interaction logic
├── observations.py    # Observation construction
└── rewards.py         # Reward strategies
```

**Recommendation:** Start with extracting `RewardStrategy` and `MeterDynamics` classes - they're most isolated.

---

### 2. Neural Networks (`agent/networks.py`) - 217 lines

**Coverage:** 0%  
**Risk Level:** 🟡 MODERATE

#### Architectures

**SimpleQNetwork (Lines 8-30)**

- MLP: `[obs_dim → 128 → 128 → action_dim]`
- For full observability (Level 1)
- Used when `partial_observability=False`

**RecurrentSpatialQNetwork (Lines 33-217)**

- For partial observability (Level 2 POMDP)
- Components:
  - Vision Encoder: CNN `[5×5 window → 128 features]`
  - Position Encoder: MLP `[x,y → 32 features]`
  - Meter Encoder: MLP `[8 meters → 32 features]`
  - Affordance Encoder: MLP `[15 types → 32 features]`
  - LSTM: `[224 → 256 hidden]`
  - Q-Head: MLP `[256 → 128 → action_dim]`
- Stateful: Maintains hidden state across episode

#### Critical Methods

```python
def forward(obs, hidden=None) -> (q_values, new_hidden)
def reset_hidden_state(batch_size, device) -> None
def set_hidden_state(hidden) -> None
def get_hidden_state() -> Optional[Tuple[h, c]]
```

#### Testing Priorities

1. ✅ Shape validation (input/output dimensions)
2. ✅ Hidden state management (reset, continuity)
3. ✅ Batch processing
4. ⚠️ LSTM memory (does it actually use history?)

---

### 3. Curriculum (`curriculum/adversarial.py`) - 360 lines

**Coverage:** 0%  
**Risk Level:** 🟡 MODERATE  
**Complexity:** State machine with performance tracking

#### 5-Stage Progression

```python
Stage 1: Basic needs (energy, hygiene) @ 20% depletion - SHAPED
Stage 2: + hunger @ 50% depletion - SHAPED
Stage 3: + money @ 80% depletion - SHAPED
Stage 4: All meters @ 100% depletion - SHAPED
Stage 5: All meters @ 100% depletion - SPARSE ← GRADUATION!
```

#### Advancement Logic (Lines 185-210)

Agent advances when:

- Survival rate > 70%
- Learning progress > 0 (reward improvement)
- Entropy < 0.5 (policy converged)
- Min 1000 steps at current stage

Agent retreats when:

- Survival rate < 30%
- Negative learning progress

#### Key Classes

**PerformanceTracker (Lines 60-100)**

```python
episode_rewards: torch.Tensor[num_agents]
episode_steps: torch.Tensor[num_agents]
prev_avg_reward: torch.Tensor[num_agents]
agent_stages: torch.Tensor[num_agents]  # 1-5
steps_at_stage: torch.Tensor[num_agents]

def update_step(rewards, dones) -> None
def get_survival_rate(max_steps) -> torch.Tensor
def get_learning_progress() -> torch.Tensor
```

**AdversarialCurriculum (Lines 103-360)**

```python
def initialize_population(num_agents) -> None
def _should_advance(agent_idx, entropy) -> bool
def _should_retreat(agent_idx) -> bool
def get_batch_decisions_with_qvalues(agent_states, agent_ids, q_values) -> List[CurriculumDecision]
def _calculate_action_entropy(q_values) -> torch.Tensor
def state_dict() -> Dict  # For checkpointing
def load_state_dict(state_dict) -> None
```

#### Testing Priorities

1. ✅ Stage progression logic
2. ✅ Entropy calculation
3. ✅ Survival rate tracking
4. ✅ State persistence (checkpointing)

---

### 4. Exploration (`exploration/` module)

**Coverage:** 0%  
**Risk Level:** 🟢 LOW (well-abstracted)

#### Class Hierarchy

```
ExplorationStrategy (base.py - Abstract)
├── EpsilonGreedyExploration (epsilon_greedy.py)
│   └── Simple ε-greedy with exponential decay
├── RNDExploration (rnd.py)
│   ├── Fixed Network (random, frozen)
│   ├── Predictor Network (trained)
│   └── Intrinsic Reward = Prediction Error
└── AdaptiveIntrinsicExploration (adaptive_intrinsic.py)
    ├── Wraps RNDExploration (composition)
    └── Variance-based annealing of intrinsic weight
```

#### Key Interface (base.py)

```python
@abstractmethod
def select_actions(q_values, agent_states, action_masks) -> torch.Tensor
@abstractmethod
def compute_intrinsic_rewards(observations) -> torch.Tensor
@abstractmethod
def update(batch: Dict[str, torch.Tensor]) -> None
@abstractmethod
def checkpoint_state() -> Dict[str, Any]
@abstractmethod
def load_state(state: Dict[str, Any]) -> None
```

#### RND Details (rnd.py)

```python
RNDNetwork: 3-layer MLP [obs_dim → 256 → 128 → embed_dim]

Fixed Network: Random weights, frozen
Predictor Network: Trained to match fixed network
Intrinsic Reward: MSE(fixed, predictor) per observation
Training: Mini-batch (128) accumulation, then update
```

#### Adaptive Annealing (adaptive_intrinsic.py)

```python
Tracks survival_history: List[float] (window=100)
Anneals intrinsic_weight when:
  - variance < threshold (10.0 → 100.0 to prevent premature annealing)
  - mean_survival > 50 steps (must be succeeding, not just consistent)
Decay: weight *= 0.99 (exponential)
```

#### Testing Priorities

1. ✅ Action selection with masking
2. ✅ RND prediction error calculation
3. ✅ Variance-based annealing triggers
4. ✅ State persistence

---

### 5. Population (`population/vectorized.py`) - 402 lines

**Coverage:** 0%  
**Risk Level:** 🔴 HIGH (orchestrates everything)  
**Role:** Training loop coordinator

#### Responsibilities

1. Q-network management (forward pass, optimization)
2. Action selection (delegates to exploration)
3. Experience collection (replay buffer)
4. Q-network training (DQN updates)
5. RND predictor training
6. Curriculum decision retrieval
7. Episode reset handling
8. Hidden state management (for recurrent networks)

#### Key Methods

**Initialization (Lines 28-95)**

```python
def __init__(env, curriculum, exploration, agent_ids, device, obs_dim, action_dim, ...)
    - Creates Q-network (Simple or Recurrent based on network_type)
    - Creates optimizer (Adam)
    - Creates replay buffer
    - Initializes training counters
```

**Action Selection (Lines 97-196)**

```python
def reset() -> None
def select_greedy_actions(env) -> torch.Tensor  # For inference
def select_epsilon_greedy_actions(env, epsilon) -> torch.Tensor  # With masking
```

**Training Loop (Lines 198-370)**

```python
def step_population(envs) -> BatchedAgentState
    1. Forward pass (Q-values)
    2. Get curriculum decisions
    3. Get action masks
    4. Select actions (exploration strategy)
    5. Step environment
    6. Compute intrinsic rewards
    7. Store in replay buffer
    8. Train RND predictor (if applicable)
    9. Train Q-network (every 4 steps, batch=64)
    10. Update current state
    11. Handle episode resets (annealing, hidden state)
    12. Return BatchedAgentState

def update_curriculum_tracker(rewards, dones) -> None
```

#### Training Details

- Q-network training frequency: Every 4 steps
- Batch size: 64 transitions
- Gradient clipping: max_norm=10.0
- No target network (simplified DQN)
- Recurrent networks: Reset hidden state for batch training

#### Testing Priorities

1. ✅ Q-network shape validation
2. ✅ Action masking integration
3. ✅ Replay buffer flow
4. ✅ DQN update correctness
5. ⚠️ Hidden state management (recurrent)
6. ⚠️ Multi-agent coordination

---

### 6. Affordance Configuration (`environment/affordance_config.py`) - 303 lines

**Coverage:** 100% ✅  
**Risk Level:** 🟢 LOW (well-tested)

#### Structure

```python
AFFORDANCE_CONFIGS: Dict[str, AffordanceConfig] = {
    'Bed': {
        'required_ticks': 5,
        'cost_per_tick': 0.01,  # $1/tick
        'operating_hours': (0, 24),  # 24/7
        'benefits': {
            'linear': {'energy': +0.075},  # Per tick (75% distributed)
            'completion': {'energy': +0.125, 'health': +0.02}  # 25% bonus
        }
    },
    # ... 14 more affordances
}
```

#### Affordance Categories

**24/7 Affordances:**

- `Bed`, `LuxuryBed`: Energy restoration (tier 1 & 2)
- `Shower`: Hygiene restoration
- `HomeMeal`: Satiation + health
- `FastFood`: Quick satiation (fitness/health penalty)
- `Hospital`: Health restoration (tier 2 - expensive)
- `Gym`: Fitness builder

**Business Hours (8am-6pm):**

- `Job`: Office work ($22.50, -15% energy)
- `Labor`: Physical labor ($30, -20% energy, -5% fitness/health)
- `Doctor`: Health restoration (tier 1 - cheaper)
- `Therapist`: Mood restoration (tier 2)
- `Recreation`: Mood + social (8am-10pm)

**Dynamic (Time-Dependent):**

- `CoffeeShop`: Energy + mood + social (8am-6pm)
- `Bar`: Social + mood (6pm-4am - wraps midnight!)
- `Park`: Free fitness + social + mood (6am-10pm)

#### Helper Functions

```python
def is_affordance_open(time_of_day: int, operating_hours: Tuple[int, int]) -> bool
    # Handles midnight wraparound (e.g., Bar: 18-4)

METER_NAME_TO_IDX: Dict[str, int] = {
    'energy': 0, 'hygiene': 1, 'satiation': 2, 'money': 3,
    'mood': 4, 'social': 5, 'health': 6, 'fitness': 7
}
```

---

### 7. Demo Infrastructure (`demo/` module)

**Coverage:** 0%  
**Risk Level:** 🟡 MODERATE (not critical for training)

#### Components

**live_inference.py (573 lines) - WebSocket Server**

```python
class LiveInferenceServer:
    - Loads latest checkpoint during training
    - Runs inference episodes at human-watchable speed (5 steps/sec)
    - Broadcasts state updates to Vue.js frontend via WebSocket
    - Auto-checkpoint mode: checks for new models after each episode
    
Key Methods:
    - startup(): Initialize env, load initial checkpoint
    - _check_and_load_checkpoint() -> bool: Watch for new checkpoints
    - _run_single_episode(): Step-by-step inference with state broadcast
    - _broadcast_state_update(): Send grid, meters, Q-values to frontend
```

**runner.py (326 lines) - Training Orchestration**

```python
class DemoRunner:
    - Multi-day training with checkpointing
    - Loads config from YAML
    - Saves checkpoints every 100 episodes
    - Logs metrics to SQLite database
    - Handles graceful shutdown (SIGTERM, SIGINT)
    
Key Methods:
    - run(): Main training loop
    - save_checkpoint(): Serialize population state
    - load_checkpoint() -> Optional[int]: Resume from latest
```

**database.py (192 lines) - Metrics Storage**

```python
class DemoDatabase:
    - SQLite with WAL mode (concurrent reads)
    - Tables: episodes, affordance_visits, position_heatmap, system_state
    - Thread-safe for single writer + multiple readers
    
Schema:
    episodes: episode_id, timestamp, survival_time, rewards, curriculum_stage, epsilon
    system_state: Key-value store for training status, checkpoint paths
```

---

### 8. Training Support (`training/` module)

**Coverage:** 0%  
**Risk Level:** 🟢 LOW (simple utilities)

#### replay_buffer.py (117 lines)

```python
class ReplayBuffer:
    - Circular buffer (FIFO eviction)
    - Stores: (obs, action, reward_extrinsic, reward_intrinsic, next_obs, done)
    - Sampling: Random mini-batches with combined rewards
    
Key Methods:
    - push(observations, actions, rewards_extrinsic, rewards_intrinsic, next_observations, dones)
    - sample(batch_size, intrinsic_weight) -> Dict[str, torch.Tensor]
        Returns combined_rewards = extrinsic + intrinsic * weight
```

#### state.py (208 lines)

```python
# Cold Path (Pydantic DTOs - validation, serialization)
class CurriculumDecision: difficulty_level, active_meters, depletion_multiplier, reward_mode, reason
class ExplorationConfig: strategy_type, epsilon, epsilon_decay, intrinsic_weight, rnd_hidden_dim
class PopulationCheckpoint: generation, num_agents, agent_ids, curriculum_states, exploration_states

# Hot Path (PyTorch tensors - GPU performance)
class BatchedAgentState:
    __slots__ = ['observations', 'actions', 'rewards', 'dones', 'epsilons', ...]
    # All tensor[num_agents], minimal overhead
```

---

## Current Test Suite

**Location:** `tests/test_townlet/`  
**Total Tests:** 19 (all passing ✅)  
**Coverage:** 16% overall, focused on environment temporal mechanics

### Test Files

1. **test_affordance_config.py** (3 tests)
   - Structure validation
   - Benefit math (linear + completion)
   - Dynamic affordance time windows

2. **test_multi_interaction.py** (4 tests)
   - Progressive benefit accumulation
   - Completion bonuses
   - Early exit preserves progress
   - Money charged per tick

3. **test_temporal_integration.py** (6 tests)
   - Full 24-hour cycle
   - Observation dimensions with temporal features
   - Multi-tick job completion
   - Operating hours masking
   - Early exit from interactions
   - Temporal mechanics disable fallback

4. **test_time_based_masking.py** (3 tests)
   - Job closed outside business hours
   - Bar open after 6pm
   - Bar wraparound midnight

5. **test_vectorized_env_temporal.py** (3 tests)
   - Time-of-day cycles
   - Interaction progress state exists
   - Observation includes time + progress

### Coverage Gaps (84% untested!)

**Critical Untested:**

- 🔴 `vectorized_env.py`: 44% untested (meter dynamics, reward calculation)
- 🔴 All neural networks (0%)
- 🔴 All curriculum logic (0%)
- 🔴 All exploration strategies (0%)
- 🔴 All population coordination (0%)
- 🔴 Replay buffer (0%)

---

## Refactoring Plan: Red-Green Approach

### Phase 1: Safety Net Construction (2-3 weeks)

**Week 1: Core Environment Tests**

- Priority 1: Meter dynamics (cascading effects)
- Priority 2: Action execution (movement, interactions)
- Priority 3: Reward calculation (milestone system)
- Priority 4: Terminal conditions (death cascades)

**Week 2: Training Loop Tests**

- Priority 1: VectorizedPopulation step execution
- Priority 2: Replay buffer operations
- Priority 3: Q-network training (DQN updates)
- Priority 4: Action masking integration

**Week 3: Curriculum & Exploration Tests**

- Priority 1: Stage progression logic
- Priority 2: RND intrinsic rewards
- Priority 3: Adaptive annealing
- Priority 4: State persistence (checkpointing)

**Target:** 70-80% coverage on modules being refactored

### Phase 2: Refactor with Green Tests (2-3 weeks)

**Extraction Candidates (in order of safety):**

1. **RewardStrategy** (from vectorized_env.py)
   - Extract milestone calculation
   - Interface: `def calculate_rewards(step_counts, dones) -> torch.Tensor`
   - Risk: LOW (pure function, no side effects)

2. **MeterDynamics** (from vectorized_env.py)
   - Extract depletion + cascading effects
   - Interface: `def update_meters(meters) -> torch.Tensor`
   - Risk: MODERATE (complex logic, but isolated)

3. **InteractionHandler** (from vectorized_env.py)
   - Extract affordance interaction logic
   - Interface: `def handle_interactions(positions, meters, time_of_day) -> Tuple[meters, success_dict]`
   - Risk: MODERATE (temporal mechanics complexity)

4. **ObservationBuilder** (from vectorized_env.py)
   - Extract full/partial observation construction
   - Interface: `def build_observation(state) -> torch.Tensor`
   - Risk: LOW (data transformation only)

5. **CurriculumStateMachine** (refactor adversarial.py)
   - Separate decision logic from performance tracking
   - Risk: MODERATE (state machine complexity)

### Phase 3: Validate & Iterate (1 week)

- Run full test suite after each extraction
- Performance benchmarks (GPU throughput)
- Integration tests (end-to-end episodes)

---

## Key Design Patterns

### 1. Vectorized Operations

**Everything is batched `[num_agents, ...]` tensors on GPU**

```python
# Good: Vectorized
rewards = torch.where(dones, -100.0, rewards)

# Bad: Python loops
for i in range(num_agents):
    if dones[i]:
        rewards[i] = -100.0
```

### 2. Hot Path vs Cold Path

```python
# Hot Path (GPU, every step): PyTorch tensors, no validation
class BatchedAgentState:
    __slots__ = ['observations', 'actions', ...]  # Memory efficient

# Cold Path (CPU, per episode): Pydantic models, validation
class CurriculumDecision(BaseModel):
    difficulty_level: float = Field(ge=0.0, le=1.0)  # Validated
```

### 3. Composition over Inheritance

```python
# AdaptiveIntrinsicExploration wraps RNDExploration
self.rnd = RNDExploration(...)  # Composition
self.current_intrinsic_weight = 1.0  # Added behavior
```

### 4. Action Masking for Exploration Efficiency

```python
action_masks = env.get_action_masks()  # [batch, 5] bool
masked_q_values = q_values.clone()
masked_q_values[~action_masks] = float('-inf')  # Prevent invalid actions
actions = masked_q_values.argmax(dim=1)
```

---

## Known Issues & Gotchas

### 1. Recurrent Network Hidden State Management

**Problem:** Hidden state must be carefully managed across:

- Episode resets (zero out for terminated agents)
- Batch training (treat transitions independently)
- Multi-agent rollouts (separate state per agent)

**Current Implementation:** Lines 330-352 in `vectorized.py`

```python
# Episode reset: Zero out hidden state for done agents
h, c = self.q_network.get_hidden_state()
h[:, reset_indices, :] = 0.0
c[:, reset_indices, :] = 0.0
self.q_network.set_hidden_state((h, c))
```

### 2. Money Normalization Confusion

**Two Systems Coexist:**

- **Temporal Mechanics:** `cost_per_tick` in config (already normalized)
- **Legacy:** `affordance_costs_dollars` in code (needs conversion)

**Normalization:** `$100 = 1.0`, `$0 = 0.0` (no debt allowed)

### 3. Reward Calculation Evolution

**Three Generations:**

1. **COMPLEX_DISABLED** (Lines 1000-1150): Per-step meter rewards → negative accumulation bug
2. **Proximity Shaping** (Lines 1150-1230): Reward nearness → hacking (standing near affordances)
3. **Milestone Bonuses** (Lines 950-1000): Current system, sparse survival bonuses

**Current Active:** Milestone system only  
**Pedagogical Value:** Proximity hacking demonstrates specification gaming and reward hacking

### 4. Temporal Mechanics Interaction Progress

**State:** `interaction_progress[agent_idx]` tracks ticks completed
**Reset Triggers:**

- Agent moves away from affordance
- Completes full interaction (reaches `required_ticks`)

**Bug Risk:** Progress not reset when affordance becomes unavailable (closed hours)

### 5. Affordance Operating Hours Wraparound

**Midnight Wraparound:**

- Bar: `(18, 4)` = 6pm to 4am (crosses midnight)
- Logic: `time >= open_tick OR time < close_tick`

**Testing Critical:** Edge cases at 0, 23, 24 hour boundaries

### 6. Intrinsic Weight Annealing (FIXED 2025-10-30)

**Previous Bug:** Premature annealing caused by low variance threshold (10.0)

- Agents that "consistently failed" (low variance, low survival) would trigger annealing
- Annealing should only happen when "consistently succeeding"

**Current Fix:**

- Variance threshold: 10.0 → 100.0
- New requirement: mean_survival > 50 steps before annealing
- Prevents "consistently failing" from being confused with "consistent performance"

### 7. Network Architecture Auto-Detection

**Critical:** Observation dimension varies by configuration:

- Full observability: grid_size² + 8 meters = 72 dims (8×8 grid)
- Partial observability: (2*vision_range+1)² + 2 (position) + 8 = 35 dims (5×5 window)
- Temporal mechanics: +2 dims (time_of_day + interaction_progress)

**Do NOT hardcode obs_dim** - always auto-detect from environment:

```python
obs_dim = env.observation_dim  # Let environment calculate based on config
```

---

## Performance Characteristics

### GPU Utilization

- **Good:** Meter updates, action execution, observation construction
- **Bottleneck:** Q-network training (64-batch every 4 steps)
- **CPU Fallback:** Curriculum decisions (per-episode, acceptable)

### Training Speed

- **Target:** ~200 episodes/hour on GPU
- **Actual:** Varies by survival time (longer episodes = slower)
- **Checkpoint Overhead:** Every 100 episodes (~1 second save)

### Memory Usage

- **Replay Buffer:** 10K transitions × obs_dim × 4 bytes
- **Network Weights:** ~500KB (SimpleQNetwork), ~2MB (RecurrentSpatialQNetwork)
- **Batch State:** num_agents × obs_dim × 4 bytes (usually <1MB)

---

## Configuration Management

### Training Config (YAML)

```yaml
environment:
  grid_size: 8
  partial_observability: false
  vision_range: 2
  enable_temporal_mechanics: false

population:
  num_agents: 1
  learning_rate: 0.00025
  gamma: 0.99
  network_type: "simple"  # or "recurrent"

curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5

exploration:
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 100
```

### Device Selection

```python
device_str = config.get('training', {}).get('device', 'cuda')
device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
```

---

## Dependencies & Tooling

### Core Dependencies

```python
pettingzoo>=1.24.0       # Not actually used (legacy import?)
numpy>=1.24.0            # Minimal usage (mostly PyTorch)
torch>=2.0.0             # Everything is PyTorch
fastapi>=0.100.0         # Live inference server
uvicorn[standard]>=0.23.0  # ASGI server
websockets>=11.0         # Frontend communication
pydantic>=2.0.0          # DTOs, validation
mlflow>=2.9.0            # Experiment tracking (not integrated yet)
tensorboard>=2.15.0      # Metrics visualization (not integrated yet)
```

### Dev Dependencies

```python
pytest>=7.4.0
pytest-cov>=4.1.0        # Coverage reporting
pytest-asyncio>=0.21.0   # Async test support
black>=23.7.0            # Code formatting
ruff>=0.0.280            # Linting
mypy>=1.4.0              # Type checking
```

### Testing Commands

```bash
# Run tests with coverage
source .venv/bin/activate
python -m pytest tests/ --cov=src/townlet --cov-report=term-missing --cov-report=json -v

# Current results: 19 passed, 16% coverage
```

---

## Roadmap & Future Development

### Phase 3.5: Multi-Day Tech Demo (🎯 IMMEDIATE)

**Goal:** Validate Phase 3 system over 48+ hours of continuous training

**Success Criteria:**

- Training runs 10K episodes without crashes
- Intrinsic weight anneals from 1.0 → <0.1
- Agent progresses through all 5 curriculum stages
- Final survival time > 200 steps (vs ~115 baseline)
- No memory leaks or performance degradation

**Deliverables:**

- Training logs and metrics
- Screenshots of exploration→exploitation transition
- Teaching materials from real data

**Estimated Duration:** 2-3 days runtime + 1 day setup/monitoring

### Phase 4: POMDP Extension (📋 PLANNED)

**Goal:** Add partial observability and working memory

**Why After Demo:** Validates Phase 3 foundation before adding complexity

**Core Changes:**

- Partial observability: 5×5 vision window (already implemented)
- LSTM memory for sequential reasoning (already implemented)
- Modified reward shaping for hidden affordance discovery
- New testing for hidden state propagation

**Teaching Value:**

- Demonstrates working memory requirements
- Shows partial observability effects on exploration
- Creates "interesting failures" when agents forget locations
- Introduces recurrent architectures

**Note:** Don't optimize during this phase - focus on correctness

### Phase 5: Informed Optimization (📋 PLANNED)

**Goal:** Profile complete system and optimize based on real bottlenecks

**Why Last:** Avoids 1→3→1 trap (optimizing twice)

**Optimization Candidates:**

- Target Network for DQN stability
- Double DQN to reduce overestimation
- Dueling Architecture (value/advantage separation)
- GPU optimization (fix RND CPU transfers)
- LSTM gradient flow tuning
- Sequential replay buffer for recurrent training

**Process:**

1. Profile with cProfile, PyTorch profiler
2. Identify actual bottlenecks (not guesses)
3. Prioritize by impact
4. Validate correctness after each change

**Target:** 2-5x speedup (100 → 200-500 episodes/hour)

### Phase 6: Multi-Agent Competition (📋 FUTURE)

**Goal:** Add competitive multi-agent scenarios

**Concepts:**

- Agents compete for limited resources
- Theory of mind: predict other agents' actions
- Emergent cooperation or competition
- Communication (implicit via observation)

**Teaching Value:** Game theory, social intelligence, emergent behavior

### Phase 7: Emergent Communication (📋 FUTURE)

**Goal:** Family units with emergent language

**Concepts:**

- Family members share information
- Discrete symbol communication channel
- Emergent protocols for coordination
- Language grounding in shared experience

### North Star: Social Hamlet 🌟

**The Ultimate Vision (2-3 years):**

**Environment:**

- 50×50 grid (vs current 8×8)
- Dozens of agents simultaneously
- Continuous affordance usage (blocking mechanics)
- Multiple instances of each affordance type
- Dynamic environment (affordances move/appear/disappear)

**Emergent Behaviors:**

- Strategic competition: "Agent A notices Agent B's routine and adapts"
- Social penalties: proximity costs create natural territoriality
- Economic hierarchy: tiered housing, job competition
- Emergent cooperation patterns despite individual incentives

**Teaching Value:**

- Nash equilibria emerge naturally
- Social intelligence and theory of mind
- Temporal strategy (planning ahead)
- Resource allocation and market dynamics
- Conflict resolution behaviors

**Why Document This Now:** Every phase builds toward this vision. Knowing the endstate helps avoid premature abstraction while keeping doors open (the "for but not with" principle).

---

## Critical Metrics to Track

### Training Metrics

- Survival time (steps before death)
- Curriculum stage distribution
- Epsilon decay curve
- Intrinsic weight annealing
- TD error (Q-learning loss)
- RND prediction error

### Environment Metrics

- Affordance visit frequencies
- Meter value distributions
- Death cause breakdown (energy vs health)
- Money management efficiency

### Performance Metrics

- Episodes per hour (throughput)
- GPU utilization %
- Memory usage (replay buffer growth)
- Checkpoint save/load time

---

## Glossary

**POMDP:** Partially Observable Markov Decision Process - agent sees local window, not full grid  
**RND:** Random Network Distillation - novelty-based intrinsic motivation  
**DQN:** Deep Q-Network - value-based RL algorithm  
**Vectorized:** Batched operations on `[num_agents, ...]` tensors for GPU efficiency  
**Hot Path:** Code executed every step (must be GPU-optimized)  
**Cold Path:** Code executed per episode (CPU validation acceptable)  
**Action Masking:** Prevent invalid actions (boundary violations, unaffordable affordances)  
**Curriculum:** Progressive difficulty increase based on agent performance  
**Sparse Rewards:** Only reward at terminal states (vs shaped rewards every step)  
**Affordance:** Interactable object that modifies agent meters

---

## Contact & Contribution

**Project Owner:** <john@example.com>  
**Repository:** tachyon-beep/hamlet  
**Branch Strategy:** `main` (direct commits), feature branches for major changes  
**PR Requirements:** Tests must pass, coverage should not decrease  

**Before Major Refactoring:**

1. Read this document fully
2. Run coverage analysis: `pytest --cov=src/townlet --cov-report=term-missing`
3. Identify untested code paths
4. Write characterization tests for current behavior
5. Refactor with tests green at each step
6. Re-run coverage to ensure no regression

---

## Important Notes on Documentation

**Authority Order:** ROADMAP.md > actual code > CLAUDE.md

**CLAUDE.md Status:** Outdated in several areas:

- Says "Townlet" is ONLY active system (correct, but details outdated)
- References old entry point locations (some consolidated)
- Some architecture details need updating
- Configs section partially outdated
- **Still useful for:** Command examples, general philosophy, known pitfalls

**This Document (AGENTS.md):** Synthesizes ROADMAP.md (strategic direction), actual code (ground truth), and relevant parts of CLAUDE.md. When in doubt, check ROADMAP.md first, then verify against actual code.

**Pedagogical Mission:** "Trick students into learning graduate-level RL by making them think they're just playing The Sims." This principle guides all design decisions - interesting failures are teaching moments, not bugs to immediately fix.

---

**Remember:** With 16% coverage, we're flying blind. Test first, refactor second. Red-Green-Refactor is not optional.
