# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hamlet is a pedagogical Deep Reinforcement Learning (DRL) environment where agents learn to survive by managing multiple competing needs (energy, hygiene, satiation, money). The primary mission is to "trick students into learning graduate-level RL by making them think they're just playing a game."

**Key insight**: The project deliberately produces "interesting failures" (like reward hacking) as teaching moments rather than bugs to fix.

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Install with development tools
uv sync --extra dev
```

### Training
```bash
# Quick training demo (50 episodes)
uv run python demo_training.py

# Full training run (1000 episodes, ~15-20 minutes)
uv run python -c "from demo_training import train_agent; train_agent(num_episodes=1000, batch_size=64, buffer_capacity=50000)"

# Train with config file
uv run python run_experiment.py --config configs/example_dqn.yaml
```

### Visualization
```bash
# Terminal 1: Start backend server (FastAPI + WebSocket on port 8765)
uv run python demo_visualization.py

# Terminal 2: Start frontend (Vue 3 dev server on port 5173)
cd frontend && npm run dev
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_environment.py

# Run with coverage
uv run pytest --cov=hamlet --cov-report=term-missing
```

### Code Quality
```bash
# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/

# Type checking
uv run mypy src/
```

## Architecture

### Core Components

**1. Environment (`src/hamlet/environment/`)**
- `hamlet_env.py`: Main PettingZoo-based environment with 8×8 grid, 4 affordances, 4 meters
- `entities.py`: Agent and Affordance base classes
- `meters.py`: MeterCollection system (energy, hygiene, satiation, money)
- `affordances.py`: Registry of affordances with effects (Bed, Shower, Fridge, Job)
- `grid.py`: Spatial grid management
- `renderer.py`: State-to-JSON serialization for web UI

**2. Agent (`src/hamlet/agent/`)**
- `drl_agent.py`: DQN implementation with epsilon-greedy exploration
- `networks.py`: Four neural architectures:
  - `QNetwork`: Baseline MLP (26K params)
  - `DuelingQNetwork`: Value/advantage separation (500K params)
  - `SpatialQNetwork`: CNN for spatial processing (280K params)
  - `SpatialDuelingQNetwork`: Best of both (520K params)
- `replay_buffer.py`: Experience replay for DQN
- `observation_utils.py`: State preprocessing (70-dim vector)

**3. Web Visualization (`src/hamlet/web/`)**
- `server.py`: FastAPI server on port 8765
- `websocket.py`: WebSocket manager for real-time state streaming
- `simulation_runner.py`: Async simulation orchestrator
- Frontend: Vue 3 + Pinia in `frontend/` directory

**4. Training (`src/hamlet/training/`)**
- `config.py`: Dataclass-based configuration (YAML-loadable)
- `trainer.py`: Main training loop orchestrator
- `checkpoint_manager.py`: Model checkpoint management
- `metrics_manager.py`: TensorBoard, MLflow, SQLite tracking

### State Representation

Observation is a 70-dimensional vector:
- **Grid encoding**: 8×8 = 64 dimensions (one-hot for agent position)
- **Meter values**: 4 dimensions (energy, hygiene, satiation, money normalized 0-1)
- **Proximity features**: 2 dimensions (distance to nearest affordance)

Action space: 5 discrete actions (UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4)

### Reward Structure

**Two-tier hybrid reward shaping**:

**Tier 1 - Gradient-based feedback**:
- Healthy meters (>80%): +0.5
- Okay meters (50-80%): +0.2
- Concerning meters (20-50%): -0.5
- Critical meters (<20%): -2.0
- Need-based interactions: Variable bonus based on meter state

**Tier 2 - Proximity shaping**:
- Small positive reward for being near needed affordances
- Guides exploration toward resources

**Economic balance** (see `affordances.py`):
- Job: +$30 (costs energy/hygiene)
- Bed: -$5, Shower: -$3, Fridge: -$4
- Net surplus per cycle: ~$18 (sustainable with buffer)

### Training Checkpoints

The project uses progressive checkpoints to demonstrate learning stages:
- **Episode ~50 (ε=0.778)**: Random exploration, oscillating behavior
- **Episode ~455 (ε=0.107)**: Partial learning, emerging strategy
- **Episode 1000 (ε=0.050)**: Converged policy, ~+79 reward, 372 step survival

These checkpoints are pedagogically valuable for showing the learning arc.

## Known Behaviors (Not Bugs!)

### Reward Hacking: "Interact Spam"

Trained agents may stand still and repeatedly use INTERACT action (no-op when not near affordances). This is **intentional emergent behavior** valuable for teaching:

**Why it happens**:
- Movement costs: -0.5 energy, -0.3 hygiene, -0.4 satiation per step
- INTERACT when no affordance nearby: 0 cost
- Proximity shaping rewards: Small positive for being near affordances
- Agent optimizes: Standing still > moving around

**Pedagogical value**:
- Demonstrates reward hacking / specification gaming
- Teaches "agents optimize what you measure, not what you mean"
- Perfect introduction to AI alignment problems
- Real-world parallel: Similar to complex exploits in production systems

**Do NOT "fix" without understanding teaching value**. See `docs/scraps/reward_hacking_interact_spam.md` for detailed analysis.

## Configuration System

Training is controlled via YAML configs in `configs/`:
- `example_dqn.yaml`: Baseline MLP architecture
- `example_dueling.yaml`: Dueling DQN architecture
- `example_spatial_dueling.yaml`: CNN + dueling architecture
- `quick_test.yaml`: Fast iteration config for testing

Config structure:
```yaml
experiment:
  name: string
  description: string

environment:
  grid_width: 8
  affordance_positions:
    Bed: [x, y]
    # ...

agents:
  - agent_id: string
    algorithm: dqn
    network_type: qnetwork|dueling|spatial|spatial_dueling
    learning_rate: float
    # ...

training:
  num_episodes: int
  batch_size: int
  # ...
```

## Pedagogical Resources

The `docs/scraps/` directory contains unformalized teaching insights:
- `reward_hacking_interact_spam.md`: Analysis of interact-spam exploit
- `three_stages_of_learning.md`: Using checkpoints to show learning progression
- `trick_students_pedagogy.md`: Complete teaching framework
- `flight_sim_reward_hacking_story.md`: Prior experiment validating patterns
- `session_observations_2025-10-28.md`: Development session log

**These are features, not documentation debt**. Read them before "fixing" unexpected behaviors.

## Progressive Complexity Levels

The architecture design (`docs/ARCHITECTURE_DESIGN.md`) outlines 5 levels:

**Level 1** (✅ Implemented): Single agent, full observability, 8×8 grid
**Level 2** (🔨 Designed): Partial observability (POMDP) with LSTM memory
**Level 3** (🎯 Future): Multi-zone environment with hierarchical RL
**Level 4** (🎯 Future): Multi-agent competition with theory of mind
**Level 5** (🎯 Future): Family communication and emergent language

When implementing new levels, maintain pedagogical value over pure technical correctness.

## Web UI Architecture

**Backend** (Python):
- FastAPI server with CORS enabled (dev mode accepts all origins)
- WebSocket protocol on `/ws` endpoint
- Async simulation runner yields state updates
- Renderer serializes state to JSON

**Frontend** (Vue 3):
- Pinia store for reactive state management
- SVG-based grid rendering (75px cells)
- Real-time meter displays with color coding
- Episode statistics and performance charts
- Controls: play/pause/step/reset/speed

**WebSocket messages**:
- `connected`: Initial handshake with available models
- `episode_start`: New episode begins
- `state_update`: Step-by-step state (enriched with rendered grid)
- `episode_end`: Episode completion summary
- `control`: Client commands (play/pause/step/reset/set_speed/load_model)

## Important Implementation Details

### Network Architecture Selection

The `DRLAgent` class supports 4 architectures via `network_type` parameter:
- Spatial networks require `grid_size` parameter
- Standard networks (qnetwork, dueling) do not
- Architecture is auto-detected in `_create_network()` method

### Checkpoint Format

Saved models (`.pt` files) contain:
```python
{
    'q_network': state_dict,
    'target_network': state_dict,
    'optimizer': state_dict,
    'epsilon': float,
    'state_dim': int,
    'action_dim': int,
}
```

No episode number or metadata in checkpoint (by design, for simplicity).

### Gradient Clipping

DQN uses gradient clipping (`max_grad_norm=10.0`) to prevent exploding gradients. This is critical for stability with shaped rewards.

### Economic Balance

The environment is intentionally balanced for sustainability:
- Full cycle cost (Bed + Shower + Fridge) = $12
- Job payment = $30
- Net per cycle = +$18

This allows agents to survive indefinitely if they learn proper cycles.

## Testing Strategy

Tests focus on:
- Environment mechanics (grid, meters, affordances)
- Agent action selection and learning
- Configuration loading
- Checkpoint save/load

**Do NOT test for "correct" strategies** - emergent behaviors are valuable even if unexpected.

## Common Pitfalls

1. **WebSocket localhost hardcoding**: Use `window.location.hostname` in frontend, not `localhost`
2. **Attribute names**: Affordances use `.name`, not `.affordance_type`
3. **Numpy types in checkpoints**: Checkpoint manager has type checking for dict vs scalar metrics
4. **Timeout in training**: Don't set short timeouts on long training runs (15-20 min for 1000 episodes)
5. **"Fixing" reward hacking**: Read pedagogical docs first - it may be more valuable broken

## Future Development Priorities

1. **Checkpoint saving during training**: Save at episodes [50, 200, 500, 1000] for teaching demos
2. **Model selector in web UI**: Switch between checkpoints to show learning progression
3. **Q-value visualization**: Overlay Q-values on grid for interpretability
4. **Level 2 POMDP**: Partial observability with LSTM (next major feature)

## Development Philosophy

> "Trick students into learning graduate-level RL by making them think they're just playing a game."

When in doubt:
- Prioritize pedagogical value over technical purity
- Preserve "interesting failures" as teaching moments
- Document unexpected behaviors rather than immediately fixing them
- Remember: The goal is to teach RL intuitively, not build production-ready agents
