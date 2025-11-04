# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Active System

**TOWNLET is the ONLY active training system.**

**DO NOT edit code in `src/hamlet/` directories** - it contains obsolete legacy code that will be deleted. The only exception is `src/hamlet/demo/runner.py` which is the temporary entry point until centralization is complete.

**Active system location**: `src/townlet/`

## Project Overview

HAMLET is a pedagogical Deep Reinforcement Learning (DRL) environment where agents learn to survive by managing multiple competing needs (energy, hygiene, satiation, money, health, fitness, mood, social). The primary mission is to "trick students into learning graduate-level RL by making them think they're just playing The Sims."

**Current Implementation**: **Townlet** - GPU-native vectorized training system with adversarial curriculum and intrinsic exploration.

**Key insight**: The project deliberately produces "interesting failures" (like reward hacking) as teaching moments rather than bugs to fix.

## Development Commands

### Setup

```bash
# Install dependencies
uv sync

# Install with development tools
uv sync --extra dev
```

### Training (Townlet System)

```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Training command pattern:
# python -m townlet.demo.runner --config <config_pack_dir>

# L0: Learn temporal credit assignment (single affordance)
uv run scripts/run_demo.py --config configs/L0_0_minimal

# L0.5: Multiple resource management (4 affordances)
uv run scripts/run_demo.py --config configs/L0_5_dual_resource

# L1: Full observability baseline (all 14 affordances)
uv run scripts/run_demo.py --config configs/L1_full_observability

# L2: Partial observability with LSTM (POMDP)
uv run scripts/run_demo.py --config configs/L2_partial_observability

# L3: Temporal mechanics (time-based affordances)
uv run scripts/run_demo.py --config configs/L3_temporal_mechanics

# Note: Each config pack is a directory containing:
#   bars.yaml, cascades.yaml, affordances.yaml, cues.yaml, training.yaml
```

### Inference Server (Live Visualization)

```bash
# IMPORTANT: Run inference server from worktree (where checkpoints are),
# but run frontend from main repo (where package.json is)

# Terminal 1: Start inference server (from worktree)
cd /home/john/hamlet/.worktrees/temporal-mechanics  # or your worktree path
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# For Level 2 POMDP:
python -m townlet.demo.live_inference checkpoints_level2 8766 0.2 10000 configs/townlet_level_2_pomdp.yaml

# For Level 2.5 Temporal Mechanics:
python -m townlet.demo.live_inference checkpoints_level2_5 8766 0.2 10000 configs/townlet_level_2_5_temporal.yaml

# Args: <checkpoint_dir> <port> <speed> <total_episodes> <config_path>
# - checkpoint_dir: Where checkpoints are saved
# - port: WebSocket port (default 8766)
# - speed: Simulation speed multiplier (0.2 = slower, 2.0 = faster)
# - total_episodes: Number of episodes to run
# - config_path: Training config YAML (REQUIRED for POMDP/temporal to load correct network type)

# Terminal 2: Start frontend (from main repo)
cd /home/john/hamlet/frontend  # Main repo, not worktree!
npm run dev
# Open http://localhost:5173

# Terminal 3 (optional): Monitor training progress (from worktree)
cd /home/john/hamlet/.worktrees/temporal-mechanics
watch -n 5 'sqlite3 demo_level2_5.db "SELECT episode, survival_steps, total_reward, stage FROM training_history ORDER BY episode DESC LIMIT 10"'
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_townlet/test_integration.py

# Run with coverage
uv run pytest --cov=townlet --cov-report=term-missing
```

### Code Quality

```bash
# Format code
uv run black src/townlet tests/

# Lint
uv run ruff check src/townlet

# Type checking
uv run mypy src/townlet
```

## Architecture

### Townlet System (ACTIVE)

```
src/townlet/
├── agent/
│   └── networks.py          # SimpleQNetwork, RecurrentSpatialQNetwork (LSTM)
├── environment/
│   └── vectorized_env.py    # VectorizedHamletEnv (GPU-native batched environments)
├── population/
│   └── vectorized.py        # VectorizedPopulation (batched agent training)
├── curriculum/
│   ├── adversarial.py       # AdversarialCurriculum (adaptive difficulty)
│   └── static.py            # StaticCurriculum (fixed difficulty)
├── exploration/
│   ├── adaptive_intrinsic.py # AdaptiveIntrinsicExploration (RND + annealing)
│   ├── rnd.py               # RNDExploration (novelty rewards)
│   └── epsilon_greedy.py    # EpsilonGreedyExploration
└── training/
    ├── state.py             # BatchedAgentState, PopulationCheckpoint
    ├── replay_buffer.py     # ReplayBuffer (experience replay)
    └── (runner.py coming)   # DemoRunner (main entry point - currently in hamlet/demo/)
```

### Core Components

**1. Environment (`src/townlet/environment/`)**

- `vectorized_env.py`: GPU-native vectorized environments
  - Batches multiple agents for parallel training
  - 8×8 grid with 14 affordances
  - 8 meters: energy, hygiene, satiation, money, mood, social, health, fitness
  - Supports full observability (Level 1.5) and partial observability (Level 2 POMDP)
  - **Partial observability**: 5×5 local vision window, agent must build mental map

**2. Agent Networks (`src/townlet/agent/`)**

- `SimpleQNetwork`: MLP for full observability (~26K params)
- `RecurrentSpatialQNetwork`: CNN + LSTM for partial observability (~600K params)
  - Vision encoder: 5×5 local window → 128 features
  - Position encoder: (x, y) → 32 features
  - Meter encoder: 8 meters → 32 features
  - LSTM: 192 input → 256 hidden (memory for POMDP)
  - Q-head: 256 → 128 → 5 actions

**3. Population (`src/townlet/population/`)**

- `VectorizedPopulation`: Coordinates batched agent training
  - Shared Q-network across agents
  - Experience replay buffer
  - Curriculum-guided training
  - Intrinsic + extrinsic rewards
  - Handles both standard and recurrent networks

**4. Curriculum (`src/townlet/curriculum/`)**

- `AdversarialCurriculum`: Adaptive difficulty scaling
  - 5 stages: 50 → 100 → 200 → 350 → 500 steps
  - Advances on 70% survival rate + minimum entropy
  - Retreats on <30% survival rate
  - Prevents premature advancement (min 1000 episodes per stage)

**5. Exploration (`src/townlet/exploration/`)**

- `AdaptiveIntrinsicExploration`: RND + variance-based annealing
  - Random Network Distillation for novelty rewards
  - Anneals intrinsic weight when agent performs consistently (low variance + high survival)
  - Prevents premature annealing: requires survival >50 steps AND variance <100

**6. Training Entry Point (CURRENT LOCATION)**

- `src/townlet/demo/runner.py`: Main training orchestrator
  - ⚠️ Will be moved to `src/townlet/training/runner.py` during centralization
  - Coordinates: VectorizedHamletEnv, VectorizedPopulation, AdversarialCurriculum, AdaptiveIntrinsicExploration
  - Saves checkpoints, tracks metrics in SQLite database
  - Run with: `python -m townlet.demo.runner` (requires PYTHONPATH=src)

### State Representation

**Fixed Affordance Vocabulary**: All curriculum levels observe the same 14 affordances (for transfer learning and observation stability), even if not all are deployed.

**Full Observability**:

- Grid encoding: grid_size × grid_size one-hot
- Meters: 8 normalized values (energy, health, satiation, money, mood, social, fitness, hygiene)
- Affordance at position: 15 one-hot (14 affordances + "none")
- Temporal extras: 4 values (time_of_day, retirement_age, interaction_progress, interaction_ticks)

**Observation dimensions by level**:

- **L0_0_minimal**: 36 dims (3×3 grid=9 + 8 meters + 15 affordances + 4 extras)
- **L0_5_dual_resource**: 76 dims (7×7 grid=49 + 8 meters + 15 affordances + 4 extras)
- **L1_full_observability**: 91 dims (8×8 grid=64 + 8 meters + 15 affordances + 4 extras)

**Partial Observability (Level 2 POMDP)**:

- Local grid: 5×5 window (25 dims) - agent only sees local region
- Position: normalized (x, y) (2 dims) - where am I on the grid?
- Meters: 8 normalized values (8 dims)
- Affordance at position: 15 one-hot (15 dims)
- Temporal extras: 4 values (4 dims)
- **Total**: 54 dimensions

**Key insight**: Observation dim varies by grid size, but affordance encoding is **constant** (always 14 affordances + "none"). This enables transfer learning - a model trained on L0 can be promoted to L1 without architecture changes.

**Action space**: 5 discrete actions (UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4)

- Note: Action space is currently **hardcoded** but will be moved to YAML (see `docs/TASK-000-UAC-ACTION-SPACE.md`)

### Reward Structure

**Per-Step Survival Rewards** (Current System):

- **Alive agents**: +1.0 reward per step
- **Dead agents**: 0.0 reward (episode ends)
- **NO proximity shaping** (agents must explore and interact to survive)
- **No death penalty**: L0 is unstable by design - agents learn that interacting with bed extends survival
- **Rationale**: Provides dense learning signal for Q-learning, functionally equivalent to accumulating survival time
- Baseline parameter retained for API compatibility but unused in calculations

**Intrinsic Rewards**:

- RND (Random Network Distillation) for novelty
- Adaptive annealing based on performance consistency
- Combined with extrinsic rewards: `total = extrinsic + intrinsic * weight`

### Training Checkpoints

Checkpoints saved periodically contain:

- Q-network state dict
- Optimizer state dict
- Exploration state (RND networks, intrinsic weight, survival history)
- Episode number, timestamp

---

## Configuration System

Training is controlled via YAML configs in `configs/`. Each config pack is a directory containing multiple YAML files that define the universe.

### Active Config Packs (Curriculum Levels)

**L0_0_minimal** - Pedagogical: Learn temporal credit assignment

- Single affordance (Bed) on 3×3 grid
- Teaches spacing behavior (don't spam bed, space out usage)
- Fast learning: epsilon_decay=0.99

**L0_5_dual_resource** - Pedagogical: Multiple resource management

- Four affordances (Bed, Hospital, HomeMeal, Job) on 7×7 grid
- Teaches balancing energy + health cycles
- Moderate learning: epsilon_decay=0.995

**L1_full_observability** - Full observability baseline

- All 14 affordances on 8×8 grid
- Agent sees complete grid (no POMDP)
- Standard MLP Q-network (~26K params)

**L2_partial_observability** - POMDP with LSTM

- All 14 affordances on 8×8 grid
- Agent sees only 5×5 local window (must build mental map)
- Recurrent spatial Q-network with LSTM (~600K params)

**L3_temporal_mechanics** - Time-based affordances + multi-tick interactions

- Operating hours (Job 9am-5pm, Bar 6pm-2am, etc.)
- Multi-tick interactions (75% linear + 25% completion bonus)
- 24-tick day/night cycle

### Config Pack Structure (UNIVERSE_AS_CODE)

Each config pack directory contains:

```
configs/L0_0_minimal/
├── bars.yaml         # Meter definitions (energy, health, money, etc.)
├── cascades.yaml     # Meter relationships (low satiation → drains energy)
├── affordances.yaml  # Interaction definitions (Bed, Hospital, Job, etc.)
├── cues.yaml         # UI metadata for visualization
└── training.yaml     # Hyperparameters and enabled affordances
```

**Key principle**: Everything configurable via YAML (UNIVERSE_AS_CODE). The system loads and validates these files at startup.

**Important**: All curriculum levels use the **same affordance vocabulary** (14 affordances) for observation stability. Only deployment varies via `enabled_affordances` in training.yaml.

### Config Structure (training.yaml)

```yaml
environment:
  grid_size: 8
  partial_observability: false  # true for POMDP
  vision_range: 2  # 5×5 window when partial_observability=true
  enabled_affordances: ["Bed", "Hospital", "HomeMeal", "Job"]  # Which to deploy

population:
  num_agents: 1
  learning_rate: 0.00025  # 0.0001 for recurrent networks
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: simple  # or 'recurrent' for LSTM

curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000

exploration:
  embed_dim: 128
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
  survival_window: 100

training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
```

---

## Progressive Complexity Levels (Curriculum)

The curriculum progresses from simple pedagogical tasks to complex POMDP challenges:

**L0_0_minimal** (✅ Implemented): **Temporal Credit Assignment**

- 3×3 grid, 1 affordance (Bed only)
- Teaches spacing behavior (don't spam, space out usage)
- Fast learning: epsilon_decay=0.99, ~500 episodes
- Pedagogical goal: Learn that immediate actions have delayed consequences

**L0_5_dual_resource** (✅ Implemented): **Multiple Resource Management**

- 7×7 grid, 4 affordances (Bed, Hospital, HomeMeal, Job)
- Teaches balancing energy + health cycles
- Moderate learning: epsilon_decay=0.995
- Introduces economic loop (Job → money → affordances)

**L1_full_observability** (✅ Implemented): **Full Observability Baseline**

- 8×8 grid, all 14 affordances
- Agent sees complete grid (no POMDP)
- Standard MLP Q-network (~26K params)
- NO proximity shaping - must interact to survive
- Baseline for comparing POMDP performance

**L2_partial_observability** (✅ Implemented): **POMDP with LSTM Memory**

- 8×8 grid, all 14 affordances
- Agent sees only 5×5 local window (must build mental map)
- Recurrent spatial Q-network with LSTM (~600K params)
- Teaches spatial memory and exploration under uncertainty

**L3_temporal_mechanics** (✅ Implemented): **Time-Based Dynamics**

- 24-tick day/night cycle with operating hours (Job 9am-5pm, Bar 6pm-2am)
- Multi-tick interactions: 75% linear rewards + 25% completion bonus
- Time-based action masking (closed affordances unavailable)
- Early exit mechanics (agents keep accumulated benefits)
- Per-tick costs prevent "free sampling"
- Agent learns temporal planning and opportunity cost
- **Visualization**: Time-of-day gradient bar, interaction progress ring

**L4_multi_zone** (🎯 Future): Multi-zone environment with hierarchical RL
**L5_multi_agent** (🎯 Future): Multi-agent competition with theory of mind
**L6_communication** (🎯 Future): Family communication and emergent language

---

## Important Implementation Details

### Network Architecture Selection

- **SimpleQNetwork**: Full observability (L0, L0.5, L1)
  - MLP: obs_dim → 256 → 128 → action_dim (5)
  - L0: 36 input dims (~26K params)
  - L0.5: 76 input dims (~60K params)
  - L1: 91 input dims (~70K params)

- **RecurrentSpatialQNetwork**: Partial observability (L2, L3)
  - Vision encoder: 5×5 local window → CNN → 128 features
  - Position encoder: (x, y) → MLP → 32 features
  - Meter encoder: 8 meters → MLP → 32 features
  - LSTM: 192 input → 256 hidden (memory for POMDP)
  - Q-head: 256 → 128 → action_dim (5)
  - Total: ~600K params
  - LSTM hidden state resets at episode start
  - Hidden state persists during episode rollout (memory)
  - Hidden state resets per transition during batch training (simplified approach)

### Gradient Clipping

Q-network uses gradient clipping (`max_norm=10.0`) to prevent exploding gradients.

### Economic Balance

The environment is intentionally balanced for sustainability:

- Full cycle cost varies by affordance choices
- Job payment = $22.5
- Sustainable with proper cycles

### Intrinsic Weight Annealing

**Fixed bug (2025-10-30)**: Premature annealing caused by low variance threshold (10.0)

- **New threshold**: 100.0
- **New requirement**: Mean survival >50 steps before annealing
- Prevents "consistently failing" from triggering "consistent performance" annealing

---

## Testing Strategy

Tests focus on:

- Environment mechanics (vectorized operations, GPU tensors)
- Population training (batched updates, curriculum progression)
- Exploration (RND novelty, annealing logic)
- Integration (full training loop)

**Do NOT test for "correct" strategies** - emergent behaviors are valuable even if unexpected.

---

## Known Behaviors (Not Bugs!)

### Reward Hacking: "Proximity Exploitation" (FIXED)

**Historical issue** (Level 1): Agents would stand near affordances to collect proximity rewards without interacting.

**Fix** (Level 1.5+): Proximity shaping disabled entirely in townlet. Agents must interact to survive.

**Pedagogical value**: Demonstrates specification gaming and reward hacking.

---

## Common Pitfalls

1. **DO NOT edit `src/hamlet/` code** (except `demo/runner.py` temporarily)
2. **Recurrent networks need batch_size reset**: After training batch, reset hidden state to `num_agents`
3. **Partial observability changes obs_dim**: Auto-detect from environment, don't hardcode
4. **Intrinsic weight annealing**: Needs both low variance AND high survival (>50 steps)
5. **Entry point**: Use `python -m townlet.demo.runner` (requires PYTHONPATH=src, will move to townlet.training)

---

## UNIVERSE_AS_CODE: Config-Driven Design

**Core Principle**: "Everything configurable. Schema enforced mercilessly."

HAMLET follows UNIVERSE_AS_CODE philosophy where all game mechanics are defined in YAML configuration files, not hardcoded in Python. This makes the system:

- **Domain-agnostic**: Could model villages, factories, trading bots, or any other universe
- **Experimentable**: Operators can test new mechanics without code changes
- **Pedagogical**: Students learn RL concepts by editing config files

### Active Development Tasks

See `docs/` for detailed tasking statements:

**TASK-000: UAC Action Space** - Move hardcoded action space to actions.yaml

- Current: Action space (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT) is hardcoded
- Goal: Define actions, movement deltas, energy costs in YAML
- Benefits: Support diagonal movement, rest actions, alternative universes

**TASK-001: UAC Contracts (Schema Enforcement)** - DTO-based config validation

- Current: Configs validated at runtime with `.get()` defaults
- Goal: Pydantic DTOs for compile-time validation with **no-defaults principle**
- Benefits: Catch config errors before training starts, operator accountability

**TASK-002: Universe Compilation Pipeline** - Cross-file validation

- Current: Each YAML file validated independently
- Goal: 7-stage compilation with dependency ordering (bars → actions → cascades → affordances → cues → training)
- Benefits: Catch dangling references, missing INTERACT action, spatial impossibilities

### Future: BRAIN_AS_CODE (Longer Term)

**Agent architecture and policy configuration will also be moved to YAML** ("BRAIN_AS_CODE"), compiled at launch:

- Network architecture (layer sizes, activation functions)
- Q-learning hyperparameters (learning rate, gamma, replay buffer size)
- Exploration strategy (epsilon-greedy, RND, UCB, etc.)
- Training schedule (epsilon decay, target update frequency)

This will enable:

- Experimenting with different architectures without code changes
- A/B testing exploration strategies across curriculum levels
- Reproducing exact agent configurations from config files
- Domain-specific architectures (e.g., LSTM for POMDP, MLP for full obs) defined in configs

### No-Defaults Principle

**All behavioral parameters must be explicitly specified in config files.** No implicit defaults allowed.

**Why**: Hidden defaults create non-reproducible configs, operator doesn't know what values are being used, and changing code defaults silently breaks old configs.

**Exemptions**: Only truly optional features (cues.yaml for visualization), metadata (descriptions), and computed values (observation_dim).

**Enforcement**: Pydantic DTOs require all fields. Missing field → clear compilation error with example.

## Future Development Priorities

1. **UNIVERSE_AS_CODE Implementation**:
   - TASK-000: Move action space to YAML
   - TASK-001: Add DTO-based schema validation with no-defaults
   - TASK-002: Implement universe compilation pipeline
2. **Centralize to townlet**: Move `runner.py`, `database.py`, etc. from `hamlet/demo/` to `townlet/`
3. **Delete obsolete hamlet code**: Remove `src/hamlet/environment/`, `src/hamlet/agent/`, `src/hamlet/training/`
4. **Level 3 Multi-zone**: Hierarchical RL with home/work zones
5. **Sequential replay buffer**: Episode sequences for better recurrent training

---

## Development Philosophy

> "Trick students into learning graduate-level RL by making them think they're just playing The Sims."

When in doubt:

- Prioritize pedagogical value over technical purity
- Preserve "interesting failures" as teaching moments
- Document unexpected behaviors rather than immediately fixing them
- Remember: The goal is to teach RL intuitively, not build production-ready agents
- **Work only in `src/townlet/`** - hamlet is obsolete legacy code

---

## Centralization Roadmap

See `PLAN_TOWNLET_CENTRALIZATION.md` for detailed plan to:

1. Move all active components to `src/townlet/`
2. Update entry point to `python -m townlet.training.runner`
3. Delete obsolete `src/hamlet/` code
4. Update all documentation

**Current status**: Entry point still in `hamlet/demo/runner.py`, but all implementation in townlet.
