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
# Level 1.5: Full observability, no proximity shaping
python -m hamlet.demo.runner configs/townlet_level_1_5.yaml demo_level1_5.db checkpoints_level1_5 5000

# Level 2 POMDP: Partial observability (5×5 vision) + LSTM memory
python -m hamlet.demo.runner configs/townlet_level_2_pomdp.yaml demo_level2.db checkpoints_level2 10000

# Arguments: <config> <database> <checkpoint_dir> <max_episodes>
```

### Inference Server (Live Visualization)
```bash
# Terminal 1: Start inference server
python -m hamlet.demo.live_inference checkpoints_level2 8766 0.2 10000
# Args: <checkpoint_dir> <port> <speed> <total_episodes>

# Terminal 2: Start frontend
cd frontend && npm run dev
# Open http://localhost:5173
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

**6. Training Entry Point (TEMPORARY LOCATION)**
- `src/hamlet/demo/runner.py`: Main training orchestrator
  - ⚠️ Will be moved to `src/townlet/training/runner.py` during centralization
  - Coordinates: VectorizedHamletEnv, VectorizedPopulation, AdversarialCurriculum, AdaptiveIntrinsicExploration
  - Saves checkpoints, tracks metrics in SQLite database

### State Representation

**Full Observability (Level 1.5)**:
- Grid encoding: 8×8 one-hot (64 dims)
- Meters: 8 normalized values (8 dims)
- **Total**: 72 dimensions

**Partial Observability (Level 2 POMDP)**:
- Local grid: 5×5 window (25 dims)
- Position: normalized (x, y) (2 dims)
- Meters: 8 normalized values (8 dims)
- **Total**: 35 dimensions

**Action space**: 5 discrete actions (UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4)

### Reward Structure

**Sparse Rewards (NO proximity shaping)**:
- Meter thresholds trigger rewards/penalties
- Interaction rewards based on meter state
- **NO proximity shaping** (agents must explore and interact to survive)
- Economic balance: sustainable income cycles

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

Training is controlled via YAML configs in `configs/`:

### Active Configs (Townlet):
- `townlet_level_1_5.yaml`: Full observability baseline
- `townlet_level_2_pomdp.yaml`: Partial observability + LSTM

### Obsolete Configs (Hamlet - DO NOT USE):
- `example_dqn.yaml`, `example_dueling.yaml`, etc. - For legacy hamlet system

**Config structure**:
```yaml
environment:
  grid_size: 8
  partial_observability: false  # or true for POMDP
  vision_range: 2  # 5×5 window when partial_observability=true

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
```

---

## Progressive Complexity Levels

The architecture design (`docs/ARCHITECTURE_DESIGN.md`) outlines 5 levels:

**Level 1** (✅ Obsolete - hamlet): Single agent, full observability, 8×8 grid
**Level 1.5** (✅ Implemented - townlet): Full observability, NO proximity shaping
**Level 2** (✅ Implemented - townlet): Partial observability (POMDP) with LSTM memory
**Level 3** (🎯 Future): Multi-zone environment with hierarchical RL
**Level 4** (🎯 Future): Multi-agent competition with theory of mind
**Level 5** (🎯 Future): Family communication and emergent language

---

## Important Implementation Details

### Network Architecture Selection

- **SimpleQNetwork**: Full observability (Level 1.5)
- **RecurrentSpatialQNetwork**: Partial observability (Level 2)
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
5. **Entry point**: Use `python -m hamlet.demo.runner` for now (will move to townlet)

---

## Future Development Priorities

1. **Centralize to townlet**: Move `runner.py`, `database.py`, etc. from `hamlet/demo/` to `townlet/`
2. **Delete obsolete hamlet code**: Remove `src/hamlet/environment/`, `src/hamlet/agent/`, `src/hamlet/training/`
3. **Level 3 Multi-zone**: Hierarchical RL with home/work zones
4. **Sequential replay buffer**: Episode sequences for better recurrent training

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
