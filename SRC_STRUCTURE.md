# src/ Directory Structure

## Overview

The `src/townlet/` directory contains the complete Hamlet RL training system, organized into 7 main subsystems.

---

## Directory Tree

```
src/townlet/
├── __init__.py                          # Package root: "GPU-native sparse reward system"
│
├── agent/                               # Neural Network Architectures
│   ├── __init__.py                      # Agent networks package
│   └── networks.py                      # Neural network architectures (SimpleQNetwork, RecurrentSpatialQNetwork)
│
├── curriculum/                          # Progressive Difficulty Management
│   ├── __init__.py                      # Curriculum package
│   ├── base.py                          # Abstract base class for curriculum managers
│   ├── adversarial.py                   # Auto-tuning curriculum based on performance (5-stage progression)
│   └── static.py                        # Fixed difficulty baseline for testing
│
├── demo/                                # Live Inference & Multi-Day Training
│   ├── __init__.py                      # Demo utilities package
│   ├── database.py                      # SQLite database for episode metrics and state management
│   ├── live_inference.py                # WebSocket server for real-time visualization
│   ├── runner.py                        # Multi-day training orchestrator with checkpointing
│   └── unified_server.py                # Unified orchestrator (training + inference + frontend)
│
├── environment/                         # Core RL Environment
│   ├── __init__.py                      # Environment package
│   ├── affordance_config.py             # Type-safe affordance definitions loaded from YAML
│   ├── affordance_engine.py             # Config-driven affordance interaction processor
│   ├── cascade_config.py                # Type-safe cascade configuration loader (Pydantic)
│   ├── cascade_engine.py                # GPU-accelerated meter cascade system (YAML-driven)
│   ├── meter_dynamics.py                # Meter depletion and coupled cascade effects
│   ├── observation_builder.py           # Observation construction (full/partial/temporal)
│   ├── reward_strategy.py               # Reward calculation logic (baseline-relative)
│   └── vectorized_env.py                # GPU-native vectorized Hamlet environment (CORE)
│
├── exploration/                         # Action Selection Strategies
│   ├── __init__.py                      # Exploration package
│   ├── base.py                          # Abstract base class for exploration strategies
│   ├── action_selection.py              # Shared action selection utilities (avoid duplication)
│   ├── epsilon_greedy.py                # Simple ε-greedy exploration (baseline)
│   ├── rnd.py                           # Random Network Distillation for intrinsic motivation
│   └── adaptive_intrinsic.py            # RND with variance-based annealing
│
├── population/                          # Agent Coordination
│   ├── __init__.py                      # Population package
│   ├── base.py                          # Abstract base class for population managers
│   └── vectorized.py                    # Training loop orchestration (Q-networks, replay, training)
│
└── training/                            # Supporting Infrastructure
    ├── __init__.py                      # Training package
    ├── replay_buffer.py                 # Experience replay with dual rewards (extrinsic + intrinsic)
    ├── sequential_replay_buffer.py      # Episode-based buffer for LSTM training (temporal structure)
    ├── state.py                         # DTOs for hot/cold paths (Pydantic + PyTorch tensors)
    └── tensorboard_logger.py            # TensorBoard integration for training metrics
```

---

## Detailed Module Descriptions

### 1. agent/ - Neural Network Architectures

**networks.py** (217 lines)

- `SimpleQNetwork`: MLP for full observability (Level 1)
  - Architecture: [obs_dim → 128 → 128 → action_dim]
- `RecurrentSpatialQNetwork`: LSTM for partial observability (Level 2)
  - Vision encoder (CNN), position encoder, meter encoder, affordance encoder
  - LSTM with 256 hidden units
  - Stateful hidden state management

---

### 2. curriculum/ - Progressive Difficulty Management

**base.py**

- Abstract interface for curriculum managers
- Defines `CurriculumDecision` contract

**adversarial.py** (360 lines)

- 5-stage progressive difficulty system
- Tracks survival rate, learning progress, policy entropy per agent
- Adaptive advancement/retreat logic with entropy gating
- Stage 1-4: Shaped rewards, Stage 5: SPARSE (graduation!)

**static.py**

- Fixed difficulty baseline
- Used for controlled experiments and validation

---

### 3. demo/ - Live Inference & Multi-Day Training

**database.py** (192 lines)

- SQLite with WAL mode for concurrent reads
- Tables: episodes, affordance_visits, position_heatmap, system_state
- Thread-safe for single writer + multiple readers

**live_inference.py** (573 lines)

- WebSocket server for real-time visualization
- Loads latest checkpoints during training (auto-checkpoint mode)
- Runs inference at human-watchable speed (5 steps/sec default)
- Broadcasts state updates to Vue.js frontend

**runner.py** (471 lines)

- Multi-day training orchestration
- Checkpointing every 100 episodes
- TensorBoard logging integration
- Graceful shutdown handling (SIGTERM, SIGINT)
- Generalization test at episode 5000 (randomizes affordance positions)

**unified_server.py** (NEW)

- Single-process orchestrator
- Coordinates training, inference, and frontend simultaneously
- Eliminates need for multiple terminals

---

### 4. environment/ - Core RL Environment

**affordance_config.py** (303 lines)

- Type-safe affordance definitions loaded from YAML
- 15 affordance types with multi-tick interactions
- Operating hours, costs, benefits (linear + completion bonuses)
- Helper: `is_affordance_open(time_of_day, operating_hours)`

**affordance_engine.py**

- Config-driven affordance interaction processor
- Replaces hardcoded affordance logic
- Handles multi-tick interactions with progressive benefits
- Temporal mechanics support (time-based availability)

**cascade_config.py**

- Pydantic models for YAML validation
- `BarsConfig`, `CascadesConfig` schemas
- Type-safe configuration loading with error reporting

**cascade_engine.py** (305 lines)

- GPU-accelerated cascade system
- Replaces hardcoded meter dynamics with YAML-driven system
- Implements:
  - Base depletions
  - Modulations (fitness → health multiplier)
  - Threshold cascades with gradient penalties
  - Terminal condition checking
- 100% test coverage, production-ready

**meter_dynamics.py**

- Encapsulates meter depletion and cascade effects
- 8-meter system: energy, hygiene, satiation, money, mood, social, health, fitness
- Coupled cascade architecture (meters affect each other)
- Differential equations for passive decay

**observation_builder.py**

- Constructs observations for all observability modes
- Full observability: Complete grid + meters
- Partial observability (POMDP): 5×5 window + position + meters
- Temporal features: time_of_day + interaction_progress

**reward_strategy.py**

- Baseline-relative reward calculation
- Formula: `reward = steps_lived - R`
- Milestone bonuses for survival
- Shaped vs sparse reward modes

**vectorized_env.py** (578 lines) ⚠️ HIGHEST COMPLEXITY

- GPU-native vectorized Hamlet environment
- Batches multiple agents: `[num_agents, ...]` tensors
- Responsibilities:
  - Grid management (8×8 default)
  - Meter dynamics (8 meters with cascades)
  - Action execution (movement + INTERACT)
  - Observation construction (full/partial/temporal)
  - Reward calculation
  - Terminal conditions
  - Multi-tick interactions
- **Known Issue**: Too many responsibilities, refactoring planned

---

### 5. exploration/ - Action Selection Strategies

**base.py**

- Abstract interface for exploration strategies
- Methods: `select_actions()`, `compute_intrinsic_rewards()`, `update()`

**action_selection.py**

- Shared utilities to avoid code duplication
- Common action selection patterns

**epsilon_greedy.py**

- Simple ε-greedy baseline
- Epsilon decay: exponential (default 0.995 per episode)
- No intrinsic motivation

**rnd.py** (220 lines)

- Random Network Distillation for novelty detection
- Fixed network (random, frozen) + Predictor network (trained)
- Intrinsic reward = prediction error (MSE)
- Epsilon-greedy action selection
- Mini-batch training (128 samples)

**adaptive_intrinsic.py**

- Wraps RND with variance-based annealing
- Tracks survival variance over window (100 episodes)
- Anneals intrinsic weight when:
  - Variance < threshold (100.0)
  - Mean survival > 50 steps
- Exponential decay (0.99)

---

### 6. population/ - Agent Coordination

**base.py**

- Abstract interface for population managers
- Future: Pareto frontier tracking, genetic reproduction

**vectorized.py** (556 lines)

- Training loop orchestrator
- Manages:
  - Q-network (forward pass, optimization)
  - Action selection (delegates to exploration strategy)
  - Experience collection (replay buffer)
  - Q-network training (DQN updates every 4 steps, batch=64)
  - RND predictor training
  - Curriculum decision retrieval
  - Episode resets with hidden state management
- Gradient clipping: max_norm=10.0
- No target network (simplified DQN)

---

### 7. training/ - Supporting Infrastructure

**replay_buffer.py** (117 lines)

- Circular buffer for off-policy learning
- Stores: (obs, action, reward_extrinsic, reward_intrinsic, next_obs, done)
- Combined reward sampling: `extrinsic + intrinsic * weight`
- FIFO eviction when full

**sequential_replay_buffer.py**

- Episode-based buffer for LSTM training
- Stores complete episodes (not individual transitions)
- Samples sequences to maintain temporal structure
- Required for recurrent networks (Level 2+)

**state.py** (217 lines)

- DTOs for hot path (training loop) and cold path (config, checkpoints)
- Hot path: `BatchedAgentState` (PyTorch tensors, `__slots__`)
  - observations, actions, rewards, dones, epsilons, intrinsic_rewards, survival_times, info
- Cold path: Pydantic models with validation
  - `CurriculumDecision`, `ExplorationConfig`, `PopulationCheckpoint`

**tensorboard_logger.py**

- TensorBoard integration for training metrics
- Logs:
  - Episode metrics (survival, rewards, stage, epsilon)
  - Training metrics (TD error, loss)
  - Meter dynamics (8 meters over time)
  - Affordance usage (interaction counts)
  - Hyperparameters
- Flush every 10 writes

---

## Entry Points

### Training

```bash
# Unified server (recommended)
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 10000

# Direct runner (legacy)
python -m townlet.demo.runner configs/level_1_full_observability.yaml demo.db checkpoints_dir 10000
```

### Inference

```bash
# Standalone inference server (legacy)
python -m townlet.demo.live_inference checkpoints_dir 8766 0.2 10000 configs/level_1_full_observability.yaml
```

---

## Key Design Patterns

1. **Vectorized Operations**: Everything is batched `[num_agents, ...]` tensors on GPU
2. **Hot Path vs Cold Path**:
   - Hot: PyTorch tensors, no validation, `__slots__`
   - Cold: Pydantic models, validation, serialization
3. **Composition over Inheritance**: e.g., AdaptiveIntrinsicExploration wraps RNDExploration
4. **Config-Driven Architecture**: Cascades and affordances defined in YAML, not Python
5. **Action Masking**: Prevent invalid actions (boundaries, affordability, closed hours)

---

## Complexity & Test Coverage

**High Complexity (Refactoring Targets):**

- `vectorized_env.py`: 578 lines, 82% coverage, **too many responsibilities**
- `vectorized.py` (population): 556 lines, 92% coverage, well-structured
- `live_inference.py`: 573 lines, 0% coverage (not critical for training)

**Production Ready:**

- `cascade_engine.py`: 305 lines, 100% coverage ✅
- `affordance_config.py`: 303 lines, 100% coverage ✅
- `replay_buffer.py`: 117 lines, 100% coverage ✅

**Overall Test Coverage:** 62% (376 tests passing)
**Target:** 70% before major refactoring

---

## Recent Fixes (Nov 2, 2025)

1. **Curriculum Reward Hacking** - Fixed: Curriculum now uses extrinsic rewards only
2. **Affordance Logging Broken** - Fixed: `info` dict now passed through population to runner
3. **Epsilon Decay** - Fixed: `decay_epsilon()` now called each episode

---

## Future Refactoring Priorities

1. **ACTION #1**: Configurable Cascade Engine (✅ COMPLETE)
2. **ACTION #9**: Network Architecture Redesign (🎯 HIGH PRIORITY)
3. **ACTION #14**: Modern CI/CD Pipeline (Ruff, Mypy, Vulture)
4. **Vertical Slice**: Extract RewardStrategy, MeterDynamics, ObservationBuilder from vectorized_env.py

See `docs/testing/REFACTORING_ACTIONS.md` for complete plan.
