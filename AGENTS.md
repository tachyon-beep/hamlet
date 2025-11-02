# Hamlet Project: AI Agent Memory Document

**Last Updated:** November 2, 2025  
**Purpose:** Comprehensive project documentation for AI assistants and future developers  
**Current Branch:** main  
**Test Coverage:** 46% (24 P1.1 tests + prior tests, all passing)  
**Authority Order:** ROADMAP.md > actual code > CLAUDE.md (CLAUDE.md is outdated)  
**Package Manager:** Use `uv` for all pip operations

---

## Executive Summary

Hamlet is a **pedagogical Deep Reinforcement Learning environment** designed to "trick students into learning graduate-level RL by making them think they're just playing The Sims." Agents learn to survive in a grid world by managing 8 interconnected meters (energy, hygiene, satiation, money, mood, social, health, fitness) through interactions with 15 affordances.

### Key Features

- **Config-driven architecture** - All game mechanics defined in YAML files (Nov 2)
- **Vectorized GPU training** with PyTorch tensors throughout
- **Adversarial curriculum learning** (5 progressive stages from shaped to sparse rewards)
- **RND-based intrinsic motivation** with adaptive annealing
- **Partial observability (POMDP)** support with recurrent neural networks (Level 2 implemented)
- **Temporal mechanics** with time-of-day cycles and multi-tick interactions (Level 3 implemented)
- **Unified server** for training + inference in single process

**Recent Milestone:** MAJOR REFACTORING COMPLETE (Nov 2, 2025)

- ✅ **CascadeEngine:** meter_dynamics.py 315→167 lines, pure config-driven
- ✅ **AffordanceEngine:** Removed AFFORDANCE_CONFIGS dict (240 lines)
- ✅ **vectorized_env.py:** All 3 code paths migrated to use AffordanceEngine
- ✅ **Deleted obsolete tests:** 11 tests removed (8 cascade + 3 affordance)
- ✅ **Training validated:** 10 episodes successful with new architecture
- Major refactoring plan documented with 15 actions in `docs/testing/REFACTORING_ACTIONS.md`

---

## Project Structure & Entry Points

### ⚠️ Critical: Active vs Legacy Code

**ACTIVE SYSTEM:** `src/townlet/` - All development happens here  
**LEGACY (DO NOT EDIT):** `src/hamlet/` - Obsolete code, will be deleted  
**Exception:** `src/hamlet/demo/runner.py` - Temporary entry point (will move to `townlet/training/`)

### Running the System

**✅ Current: Unified Server (Training + Inference in one command)**

```bash
# Terminal 1: Start training + inference server
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 10000

# Terminal 2: Start frontend (Vue dev server - optional, for visualization)
cd frontend && npm run dev
# Open http://localhost:5173

# Terminal 3: Start TensorBoard (optional, for metrics visualization)
tensorboard --logdir runs/L1_full_observability/<timestamp>/tensorboard
# Open http://localhost:6006

# Note: The unified server will print the exact commands for you!
```

**What You'll See When Starting:**

```
= ============================================================
✅ Training + Inference servers operational
==============================================================

📊 To view live visualization (optional):
   Terminal 2:
   $ cd frontend && npm run dev -- --host 0.0.0.0
   Then open: http://localhost:5173

📈 To view training metrics (optional):
   Terminal 3:
   $ tensorboard --logdir runs/L1_full_observability/2025-11-02_123456/tensorboard --bind_all
   Then open: http://localhost:6006

💾 Checkpoints: runs/L1_full_observability/2025-11-02_123456/checkpoints
🔌 Inference port: 8766

Press Ctrl+C to stop gracefully
==============================================================
```

**Example Commands:**

```bash
# Short demo (1K episodes, ~2 hours)
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 1000

# Full demo (10K episodes, ~48 hours)
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 10000

# Resume from checkpoint
python run_demo.py --config configs/level_1_full_observability.yaml \
    --checkpoint-dir runs/L1_full_observability/2025-11-02_123456/checkpoints

# Custom inference port
python run_demo.py --config configs/level_1_full_observability.yaml \
    --episodes 5000 --inference-port 8800
```

**Legacy (Still Works, but unified server is preferred):**

```bash
# Three separate terminals (old way)
# Terminal 1: Training only
python -m townlet.demo.runner configs/level_1_full_observability.yaml demo.db checkpoints_dir 10000

# Terminal 2: Inference server
python -m townlet.demo.live_inference checkpoints_dir 8766 0.2 10000 configs/level_1_full_observability.yaml

# Terminal 3: Frontend
cd frontend && npm run dev
```

### Configuration Files (YAML)

**Active Configs:**

- `configs/level_1_full_observability.yaml` - Level 1: Full observability baseline (SimpleQNetwork)
- `configs/level_2_pomdp.yaml` - Level 2: Partial observability + LSTM (RecurrentSpatialQNetwork)
- `configs/level_3_temporal.yaml` - Level 3: Temporal mechanics + multi-tick interactions

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

**See `docs/TRAINING_LEVELS.md` for complete formal specification.**

**Level 1** (✅ Implemented): Full observability baseline - MLP, complete information  
**Level 2** (✅ Implemented): Partial observability (POMDP) - LSTM, 5×5 window, spatial memory  
**Level 3** (✅ Implemented): Temporal mechanics - 24-tick cycles, multi-tick interactions, time planning  
**Level 4** (🎯 Future): Multi-zone environment - Hierarchical RL, zone transitions, long-term planning  
**Level 5** (🎯 Future): Multi-agent competition - Resource contention, theory of mind, emergent strategies  
**Level 6** (🎯 Future): Emergent communication - Discrete symbols, family units, language grounding

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
├── environment/             # Core RL environment (82% coverage on vectorized_env)
│   ├── vectorized_env.py    # GPU-native vectorized environment ⚠️ COMPLEX
│   ├── affordance_config.py # Multi-tick interaction configs (100% coverage)
│   └── __init__.py
├── exploration/             # Action selection strategies (100% coverage on main modules)
│   ├── adaptive_intrinsic.py # RND with variance-based annealing (100% coverage)
│   ├── rnd.py               # Random Network Distillation (82% coverage)
│   ├── epsilon_greedy.py    # Simple baseline (100% coverage)
│   ├── base.py              # Abstract interface (75% coverage)
│   └── __init__.py
├── population/              # Agent coordination (92% coverage on vectorized)
│   ├── vectorized.py        # Training loop orchestration (92% coverage)
│   ├── base.py              # Abstract interface (80% coverage)
│   └── __init__.py
└── training/                # Supporting infrastructure (97% average coverage)
    ├── replay_buffer.py     # Experience replay with dual rewards (100% coverage)
    ├── state.py             # DTOs for hot/cold paths (94% coverage)
    └── __init__.py

tests/test_townlet/          # Current test suite (241 tests, all passing)
├── test_affordance_config.py       # Affordance structure & math (3 tests)
├── test_affordance_effects.py      # Affordance effects validation (23 tests) ⭐ NEW
├── test_multi_interaction.py       # Multi-tick mechanics (4 tests)
├── test_temporal_integration.py    # Time-based systems (6 tests)
├── test_time_based_masking.py      # Operating hours (3 tests)
├── test_vectorized_env_temporal.py # Environment temporal features (3 tests)
├── test_networks.py                # Neural network validation
├── test_epsilon_greedy.py          # Exploration strategy tests
├── test_adaptive_intrinsic.py      # RND with annealing tests
├── test_adversarial_curriculum.py  # Curriculum progression tests
├── test_static_curriculum.py       # Static curriculum tests
├── test_population.py              # Training loop tests
├── test_replay_buffer.py           # Replay buffer tests
└── test_state.py                   # State management tests
```

---

## Core Components Deep Dive

### 1. Environment (`vectorized_env.py`) - 433 lines - **HIGHEST RISK**

**Complexity Level:** 🔴 EXTREME  
**Coverage:** 82% (356/433 statements)  
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

### 1a. CascadeEngine (`environment/cascade_engine.py`) - 305 lines ✨ NEW

**Coverage:** 100% (21/21 tests passing)  
**Risk Level:** 🟢 LOW (fully tested, config-driven)  
**Status:** ✅ READY FOR INTEGRATION (Nov 1, 2025)

#### Purpose

Config-driven cascade system that replaces hardcoded meter dynamics logic. Enables students to experiment with different cascade physics by editing YAML files instead of Python code.

#### Configuration Files

**bars.yaml (109 lines):**

- 8 meter definitions with base depletion rates
- Terminal conditions (health <= 0 OR energy <= 0)
- Note: health base_depletion = 0.0 (handled by fitness modulation)

**cascades.yaml (198 lines):**

- 1 modulation: fitness→health (0.5x-3.0x multiplier)
- 10 threshold cascades with gradient penalties
- Execution order: modulations → primary_to_pivotal → secondary_to_primary → secondary_to_pivotal_weak
- All cascades use 30% thresholds

#### Architecture

```python
class CascadeEngine:
    def __init__(config: EnvironmentConfig, device: torch.device):
        # Pre-build lookup maps and tensors for GPU performance
        self._bar_name_to_idx: Dict[str, int]
        self._base_depletions: torch.Tensor[8]
        self._cascade_data: Dict[str, List[CascadeInfo]]  # By category
        
    def apply_base_depletions(meters) -> meters:
        # Subtract base_depletions tensor, clamp [0,1]
        
    def apply_modulations(meters) -> meters:
        # Fitness → health multiplier (0.5x healthy, 3.0x unfit)
        
    def apply_threshold_cascades(meters, categories) -> meters:
        # Gradient penalty cascades by category
        # penalty = strength * (threshold - source) / threshold
        
    def check_terminal_conditions(meters, dones) -> dones:
        # Death detection (health <= 0 OR energy <= 0)
        
    def apply_full_cascade(meters) -> meters:
        # Complete sequence per execution_order
```

#### Gradient Penalty Math

```python
# When source < threshold (0.3):
deficit = (threshold - source) / threshold  # Normalized [0,1]
penalty = strength * deficit
target -= penalty

# Example: satiation=0.2, threshold=0.3, strength=0.004
deficit = (0.3 - 0.2) / 0.3 = 0.333
penalty = 0.004 * 0.333 = 0.00133
```

#### Integration Status

**✅ Equivalence Verified:** CascadeEngine produces identical results to hardcoded MeterDynamics

- 44/44 tests passing (23 config + 21 engine)
- Tested: healthy agents, low satiation, gradient penalties, modulations, terminal conditions
- **Next Step:** Replace hardcoded logic in MeterDynamics, run full 275-test suite

#### Teaching Value

- Students can experiment with cascade strengths by editing YAML
- Alternative configs available: weak_cascades.yaml, strong_cascades.yaml, sdw_official.yaml
- "Interesting failures" when cascades too weak (death) or too strong (can't recover)
- Demonstrates data-driven system design vs hardcoded logic

---

### 2. Neural Networks (`agent/networks.py`) - 217 lines

**Coverage:** 98%  
**Risk Level:** � LOW (well-tested)

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

#### Testing Status

1. ✅ Shape validation (input/output dimensions) - TESTED
2. ✅ Hidden state management (reset, continuity) - TESTED
3. ✅ Batch processing - TESTED
4. ⚠️ LSTM memory (does it actually use history?) - ACTION #9 will address

---

### 3. Curriculum (`curriculum/adversarial.py`) - 360 lines

**Coverage:** 86%  
**Risk Level:** � LOW (well-tested)  
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

#### Testing Status

1. ✅ Stage progression logic - TESTED
2. ✅ Entropy calculation - TESTED
3. ✅ Survival rate tracking - TESTED
4. ✅ State persistence (checkpointing) - TESTED

---

### 4. Exploration (`exploration/` module)

**Coverage:** 100% on main modules (epsilon_greedy, adaptive_intrinsic)  
**Risk Level:** 🟢 LOW (well-abstracted)#### Class Hierarchy

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

**Coverage:** 92%  
**Risk Level:** � LOW (well-tested)  
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

#### Testing Status

1. ✅ Q-network shape validation - TESTED
2. ✅ Action masking integration - TESTED
3. ✅ Replay buffer flow - TESTED
4. ✅ DQN update correctness - TESTED
5. ⚠️ Hidden state management (recurrent) - Partial coverage
6. ⚠️ Multi-agent coordination - Partial coverage

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
**Total Tests:** 241 (all passing ✅)  
**Coverage:** 64% overall (982/1525 statements)  
**Target:** 70% before major refactoring

### Test Files (Organized by Component)

**Environment Tests (82% coverage on vectorized_env.py):**

1. **test_affordance_config.py** (3 tests)
   - Structure validation
   - Benefit math (linear + completion)
   - Dynamic affordance time windows

2. **test_affordance_effects.py** (23 tests) ⭐ NEW
   - Health restoration: Doctor (+25%), Hospital (+40%)
   - Mood restoration: Therapist (+40%)
   - Park: FREE, fitness +20%, social +15%, mood +15%
   - Bar: Social +50% (BEST), mood +25%, health -5%
   - FastFood: Satiation +45%, fitness -3%, health -2%
   - Job/Labor: Income generation, energy costs, penalties

3. **test_multi_interaction.py** (4 tests)
   - Progressive benefit accumulation
   - Completion bonuses
   - Early exit preserves progress
   - Money charged per tick

4. **test_temporal_integration.py** (6 tests)
   - Full 24-hour cycle
   - Observation dimensions with temporal features
   - Multi-tick job completion
   - Operating hours masking
   - Early exit from interactions
   - Temporal mechanics disable fallback

5. **test_time_based_masking.py** (3 tests)
   - Job closed outside business hours
   - Bar open after 6pm
   - Bar wraparound midnight

6. **test_vectorized_env_temporal.py** (3 tests)
   - Time-of-day cycles
   - Interaction progress state exists
   - Observation includes time + progress

**Neural Network Tests (98% coverage on networks.py):**

7. **test_networks.py** - SimpleQNetwork and RecurrentSpatialQNetwork validation

**Exploration Tests (100% coverage on epsilon_greedy, adaptive_intrinsic):**

8. **test_epsilon_greedy.py** - Epsilon-greedy exploration with decay
9. **test_adaptive_intrinsic.py** - RND with variance-based annealing

**Curriculum Tests (86% coverage on adversarial, 100% on static):**

10. **test_adversarial_curriculum.py** - 5-stage progression, entropy gating
11. **test_static_curriculum.py** - Fixed difficulty baseline

**Population Tests (92% coverage on vectorized.py):**

12. **test_population.py** - Training loop, Q-network updates, replay buffer

**Supporting Module Tests:**

13. **test_replay_buffer.py** (100% coverage) - Dual reward storage
14. **test_state.py** (94% coverage) - DTOs and state management

### Coverage by Module

**🟢 Perfect Coverage (100%):**

- ✅ `static.py` - 100%
- ✅ `epsilon_greedy.py` - 100%
- ✅ `adaptive_intrinsic.py` - 100%
- ✅ `affordance_config.py` - 100%
- ✅ `replay_buffer.py` - 100%

**� Excellent (90%+):**

- ✅ `networks.py` - 98% (1 line missing)
- ✅ `state.py` - 94% (3 lines missing)
- ✅ `population/vectorized.py` - 92% (11 lines missing)

**🟡 Good (80-89%):**

- ✅ `adversarial.py` - 86% (20 lines missing)
- ✅ `vectorized_env.py` - 82% (77 lines missing) - includes 216 lines of DISABLED dead code
- ✅ `rnd.py` - 82% (17 lines missing)
- ✅ `population/base.py` - 80% (2 lines missing)

**🟡 Fair (70-79%):**

- ⚠️ `curriculum/base.py` - 77% (3 lines missing)
- ⚠️ `exploration/base.py` - 75% (5 lines missing)

**🔴 Untested (0%):**

- ❌ `demo/database.py` - 0% (35 lines) - SQLite metrics storage
- ❌ `demo/live_inference.py` - 0% (237 lines) - WebSocket server
- ❌ `demo/runner.py` - 0% (132 lines) - Training orchestration

### Gap to 70% Milestone: +6 percentage points needed

**Fastest Path:**

1. **Execute ACTION #13** (30 min): Remove 216 lines of dead code → vectorized_env.py 82% → ~95%!
2. **Test small gaps** (2-3 hours): Complete base classes (11 lines total)
3. **Result**: Should easily hit 70%+ 🎯

---

## Refactoring Plan: 15 Actions Documented

**Full details:** See `docs/testing/REFACTORING_ACTIONS.md` for comprehensive documentation.

**Total Estimated Time:** 13-20 weeks of focused development  
**Prerequisite:** 70% test coverage milestone (currently at 64%, need +6%)

### High Priority Actions (🔴)

1. **ACTION #1: Configurable Cascade Engine** (2-3 weeks)
   - Replace hardcoded meter cascades with data-driven system
   - Enable students to experiment with different cascade strengths
   - Critical for pedagogical flexibility

2. **ACTION #9: Network Architecture Redesign** (3-4 weeks)
   - "Root and branch reimagining" based on testing discoveries
   - Fix LSTM memory issues, observation handling
   - Discovered through systematic testing October 31, 2025

3. **ACTION #14: Implement Modern CI/CD Pipeline** (3-5 days)
   - Ruff (linter/formatter), Mypy (type checking)
   - Vulture (dead code detection - would have caught 216 DISABLED lines!)
   - Bandit (security), Pre-commit hooks, GitHub Actions

### Medium Priority Actions (🟡)

4. **ACTION #2: Extract RewardStrategy** (3-5 days)
5. **ACTION #3: Extract MeterDynamics** (1-2 weeks)
6. **ACTION #4: Extract ObservationBuilder** (2-3 days)
7. **ACTION #8: Add WAIT Action** (1-2 days) - Elevated priority due to oscillation bugs
8. **ACTION #12: Configuration-Defined Affordances** (1-2 weeks)
   - Move affordance logic to YAML (200+ lines → data)
   - Enable modding and custom affordance creation
9. **ACTION #13: Remove Pedagogical DISABLED Code** (30 minutes)
   - Delete 216 lines of dead code (reward systems that failed)
   - **Impact:** vectorized_env.py 82% → ~95% coverage instantly!
10. **ACTION #15: Unified Training + Inference Server** (1-2 weeks)
    - **Goal:** `python run_demo.py` and you're done!
    - Merge training, inference, AND frontend into single process
    - No more juggling THREE terminals

### Low Priority Actions (🟢)

11. **ACTION #5: Target Network DQN** (1-2 days)
12. **ACTION #6: GPU Optimization RND** (1 day)
13. **ACTION #7: Sequential Replay Buffer** (1 week)
14. **ACTION #10: Deduplicate Epsilon-Greedy** (1-2 hours)
15. **ACTION #11: Remove Legacy Checkpoint Methods** (15 minutes)

### Quick Wins to Hit 70%

Before major refactoring:

1. **ACTION #13** (30 min): Remove dead code → +10-12% coverage on vectorized_env.py
2. **Test small gaps** (2-3 hours): Complete base classes (11 lines total)
3. **Result:** 64% → 70%+ ✅

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

### Package Management

**This project uses `uv` for fast, reliable Python package management.**

```bash
# Install dependencies
uv pip install -e .

# Install dev dependencies
uv pip install pytest pytest-cov pytest-asyncio black ruff mypy

# Add a new package
uv pip install <package-name>
```

### Testing Commands

```bash
# Run tests with coverage (use uv)
uv run pytest tests/ --cov=src/townlet --cov-report=term-missing --cov-report=json -v

# Run specific test file
uv run pytest tests/test_townlet/test_cascade_engine.py -v

# Run with short traceback
uv run pytest tests/ -v --tb=short
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
2. Run coverage analysis: `uv run pytest --cov=src/townlet --cov-report=term-missing`
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

**Remember:** At 64% coverage (241 tests passing), we have a solid foundation. Target 70% before major refactoring. Quick wins available: Remove 216 lines of dead code (ACTION #13) for instant coverage boost!
