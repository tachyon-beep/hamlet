# Hamlet DRL Demonstrator - Design Document

**Date:** 2025-10-27
**Status:** Initial Design
**Version:** 0.1.0

## Overview

Hamlet is a Deep Reinforcement Learning (DRL) demonstrator where agents learn to survive in a grid-based world by managing multiple competing needs. Agents must balance work, rest, and self-care while navigating an 8x8 grid world with various service affordances.

## Project Goals

### MVP Goals
- Single agent learning survival behavior in 8x8 grid environment
- Custom DRL implementation (educational focus)
- Multi-meter survival mechanics with interesting tradeoffs
- Web-based visualization suitable for streaming

### Future Goals (Documented, Not Implemented)
- Multiple agents with social interactions
- Relationship meters between agents
- Social affordances (conversation, collaboration)
- Battle royale elimination mode
- Genetic algorithm for agent reproduction (neural network blending)
- Population dynamics and emergent behaviors over generations

### Scaling Vision: Don't Lock These Out!

**Short term (MVP):**
- Single agent, 8x8 grid, 4 basic affordances (bed, fridge, shower, job)
- Simple meter management (energy, hygiene, satiation, money)

**Medium term:**
- Multiple agents (2-10) with social interaction
- Larger grid (16x16 or 32x32)
- More diverse affordances and meters
- Relationships and competition for resources
- Reproduction mechanics

**Long term (Ultimate Stretch Goal):**
- **~1000 agents in a cityscape environment**
- Complex economic system:
  - Diverse job types at different locations (agents travel to work)
  - Job competition (agents can "steal" jobs from each other)
  - Rent/mortgage payments for housing
  - Tiered housing market (expensive houses → stronger benefits, requires better-paying jobs)
  - Economic pressure driving emergent social dynamics
- Large diverse affordance ecosystem
- Spatial complexity (neighborhoods, districts, travel costs)
- Emergent social stratification and economic competition
- "Hey, he stole my job today and I need to pay rent!" dynamics

**Architecture decisions to preserve scalability:**
- Use spatial indexing for large agent counts (grid dict → quadtree/spatial hash)
- Keep agent logic independent (no global state dependencies)
- Design affordances as composable, not hardcoded
- Modular economic system (easy to add currencies, markets, prices)
- Efficient rendering (only render viewport, not entire world)

## Technology Stack

### Core Technologies
- **Python 3.11+**: Modern Python features
- **uv**: Fast, modern package manager for dependency management
- **PettingZoo**: Multi-agent RL environment framework (AEC API)
- **PyTorch**: Neural network implementation
- **NumPy**: Numerical computations

### Web Interface
- **FastAPI**: Async web framework for serving visualization
- **Uvicorn**: ASGI server
- **WebSockets**: Real-time state updates to browser
- **HTML/CSS/JavaScript**: Simple grid rendering

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Fast Python linter
- **mypy**: Static type checking

## Architecture

### Project Structure (src-layout)

```
hamlet/
├── src/
│   └── hamlet/              # Main package
│       ├── environment/     # Grid world and game mechanics
│       ├── agent/           # DRL agent implementation
│       ├── training/        # Training loops and metrics
│       └── web/             # Web visualization
├── tests/                   # pytest test suite
├── docs/                    # Documentation
│   └── plans/              # Design documents
├── pyproject.toml          # Project configuration
├── README.md
├── .gitignore
└── .python-version
```

### Module Organization: Clean Separation

Each major component in its own module with clear boundaries:
- Easier to test in isolation
- Clear ownership of functionality
- Professional structure that scales
- Supports future team collaboration

## Environment Design

### Grid World (`src/hamlet/environment/`)

**Grid** (`grid.py`)
- 8x8 2D grid world
- Cells can contain agents and affordances
- Spatial logic for positioning and movement
- 4-way movement (up, down, left, right) - no diagonals
  - Simpler action space (5 vs 9 actions)
  - Faster learning for demonstration
  - Can add diagonals later if needed

**Entities** (`entities.py`)
- `Agent` class: position, meter collection, state
- Base `Affordance` class for services
- Concrete affordances: Bed, Fridge, Shower, Job
- Future: relationship tracking, social interaction

**Meters** (`meters.py`)
- Base `Meter` class with depletion rate, min/max, thresholds
- Concrete meters:
  - **Energy**: Sleep/tiredness level
  - **Hygiene**: Cleanliness level
  - **Satiation**: Hunger/fullness level
  - **Money**: Currency for purchasing services
- Each meter depletes over time
- `MeterCollection` class for managing agent's meters
- Easily extensible by subclassing

**Affordances** (`affordances.py`)
- Registry pattern for dynamic affordance addition
- Each affordance affects multiple meters (multi-dimensional tradeoffs)

**Core interaction model:**
- **Bed**: Money (-), Energy (++), Hygiene (+)
- **Shower**: Money (-), Hygiene (++), Energy (-)
- **Fridge**: Money (-), Satiation (++), Energy (+)
- **Job**: Money (++), Energy (--), Hygiene (--)
- **Movement**: Energy (-), Hygiene (-), Satiation (-) [small amounts]

This creates interesting optimization problem: agent must work to earn money, but work drains biological needs. Must learn efficient cycles of work → service → work.

**PettingZoo Environment** (`hamlet_env.py`)
- Implements PettingZoo AEC (Agent-Environment-Cycle) API
- Standard methods: reset(), step(), observe(), render()
- Initially single agent, structured for multi-agent expansion
- Episode termination: any critical meter hits zero OR agent can't afford needed service

**Renderer** (`renderer.py`)
- Web-based rendering (avoiding TUI complexity issues)
- Integration with FastAPI for serving
- Simple, clear visualization for streaming

**Observation Space:**
- Grid state (8x8 with channels for agents/affordances)
- Agent's own meter values (normalized 0-1)
- Relative distances/directions to affordances
- Available actions mask

**Action Space:**
- Discrete: {MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT}
- INTERACT triggers affordance at current position

## Agent Design

### DRL Agent (`src/hamlet/agent/`)

**Base Agent** (`base_agent.py`)
- Abstract interface: `select_action()`, `learn()`
- Allows swapping agent types (random, DRL, scripted) for testing

**DRL Agent** (`drl_agent.py`)
- Custom DQN implementation (good starting algorithm for discrete navigation)
- Epsilon-greedy exploration with decay
- Experience storage and learning updates
- Target network for stability

**Networks** (`networks.py`)
- PyTorch neural networks for value estimation
- Input: flattened observation (grid + meters + relative positions)
- Output: Q-values for each discrete action
- Modular architecture for experimentation

**Replay Buffer** (`replay_buffer.py`)
- Stores (state, action, reward, next_state, done) tuples
- Fixed size, oldest replaced (circular buffer)
- Batch sampling for training

**Reward Shaping:**
- Based on meter levels (penalties for low meters)
- Survival time bonus
- Incentivizes balanced behavior across all meters

## Training Design

### Training Loop (`src/hamlet/training/`)

**Trainer** (`trainer.py`)
- Main training loop orchestration
- Episode management (reset, step, collect experiences)
- Triggers agent learning updates
- Save/load checkpoints for long training runs

**Metrics** (`metrics.py`)
- Survival time per episode
- Total reward
- Meter trajectories over time
- Action distribution analysis
- Logging to console and files

**Config** (`config.py`)
- Centralized hyperparameters:
  - Learning rate
  - Epsilon decay schedule
  - Batch size
  - Replay buffer size
  - Target network update frequency
- Easy experimentation with different settings

**Termination Conditions (MVP):**
- Energy, Hygiene, or Satiation hits zero
- Money negative AND agent needs paid service
- Maximum episode length reached (timeout)

## Web Interface Design

### Visualization (`src/hamlet/web/`)

**Server** (`server.py`)
- FastAPI application
- Serves static files (HTML/CSS/JS)
- REST endpoints for control (start/pause/reset)
- WebSocket endpoint for state streaming

**WebSocket** (`websocket.py`)
- Real-time state updates each environment step
- Sends grid state, agent position, meter values
- Efficient JSON serialization

**Frontend** (`static/`)
- `index.html`: Simple layout with grid and controls
- `style.css`: Grid styling, colors for agents/affordances
- `app.js`: WebSocket client, grid rendering, UI controls

**Visualization Features:**
- 8x8 grid rendered as colored div cells
- Agent position clearly marked
- Affordances color-coded by type
- Real-time meter value displays
- Play/pause/reset/speed controls
- Episode statistics (survival time, steps)

**No fancy animations initially** - focus on clarity for streaming/broadcasting.

## Extensibility Design

### Future Multi-Agent Support

Architecture decisions that keep multi-agent expansion straightforward:

1. **PettingZoo AEC API**: Built for multi-agent from the start
2. **Agent Registry**: Environment tracks multiple agents by ID
3. **Grid Design**: Cells support multiple occupants
4. **Meter Independence**: Each agent has own meter collection

### Future Social Features

Prepared but not implemented:

1. **Relationship Meters**: `Agent.relationships: Dict[str, float]` for inter-agent dynamics
2. **Social Affordances**: Conversation, collaboration actions
3. **Interaction Range**: Grid proximity checks for social actions

### Future Genetic/Evolution Features

Architecture supports but doesn't implement:

1. **Neural Network Blending**: PyTorch state_dict manipulation for offspring
2. **Battle Royale Mode**: Elimination mechanics when money < 0
3. **Parent Selection**: High relationship pairs for reproduction
4. **Population Management**: Generational tracking and statistics

## Development Workflow

### Initial Setup
```bash
# Clone repository
cd hamlet

# Initialize uv project
uv init

# Install dependencies
uv sync

# Run tests
uv run pytest

# Start web server (when implemented)
uv run python -m hamlet.web.server
```

### Testing Strategy
- Unit tests for each module (environment, agent, training)
- Integration tests for full training loop
- Environment tests for PettingZoo API compliance
- Agent tests with mock environments

## Success Criteria

### MVP Success
- [ ] Agent learns to survive longer than random baseline
- [ ] Agent develops coherent behavior patterns (work-service cycles)
- [ ] Web visualization clearly shows agent state and decisions
- [ ] Training runs stable without crashes
- [ ] Code is clean, tested, and documented

### Demo Quality
- [ ] Suitable for streaming (clear, watchable, real-time)
- [ ] Interesting emergent behavior (not just random wandering)
- [ ] Fast enough for real-time viewing (not slow-motion)
- [ ] Reliable (doesn't crash during demonstration)

## Open Questions & Future Decisions

1. **Reward function tuning**: May need iteration to get interesting behavior
2. **Grid size scaling**: When to expand from 8x8?
3. **Multi-agent timing**: When to add second agent?
4. **Network architecture**: Size/depth of neural networks
5. **Training time**: How long to train for demo-worthy behavior?
6. **Visualization polish**: What level of visual polish for streaming?

## References

- PettingZoo Documentation: https://pettingzoo.farama.org/
- DQN Paper: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- FastAPI Documentation: https://fastapi.tiangolo.com/
- PyTorch Documentation: https://pytorch.org/docs/

## Appendix: Design Rationale

### Why Custom DRL vs. Stable-Baselines3?
Educational focus - understanding RL fundamentals by implementing from scratch. Makes it easier to customize and experiment with novel ideas (genetic algorithms, social dynamics).

### Why PettingZoo vs. Pure Gymnasium?
Future multi-agent support. Starting with proper multi-agent framework prevents painful refactoring later.

### Why Web UI vs. TUI?
Previous implementation had TUI complexity issues. Web UI is simpler, more flexible, and better for streaming to Twitch/YouTube.

### Why src-layout?
Best practice for Python packages. Prevents accidental imports during development. Better for future distribution/packaging.

### Why 4-way movement (no diagonals)?
Simpler action space speeds up learning. On 8x8 grid, diagonals don't provide significant benefit. Can add later if needed. Focus should be on meter management strategy, not path optimization.

### Why multi-meter affordance impacts?
Creates complex optimization problem. Single-meter interactions would be trivial to solve. Multi-meter tradeoffs create interesting emergent behavior and difficult learning challenge.
