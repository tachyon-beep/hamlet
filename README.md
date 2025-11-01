# Hamlet

**A pedagogical Deep Reinforcement Learning environment designed to "trick students into learning graduate-level RL by making them think they're just playing The Sims."**

## Overview

Hamlet is a GPU-accelerated DRL training environment where agents learn to survive by managing 8 interconnected physiological and economic meters through strategic interaction with 15 affordances. Progressive complexity levels introduce partial observability (POMDP), temporal mechanics, and intrinsic motivation.

### Key Features

- 🎮 **Vectorized GPU Training** - PyTorch tensors throughout for maximum performance
- 🧠 **Progressive Complexity** - Three training levels (L1→L2→L3) with increasing challenge
- 🔄 **Adversarial Curriculum** - Adaptive difficulty from shaped to sparse rewards (5 stages)
- 🌟 **Intrinsic Motivation** - RND-based exploration with variance-based annealing
- 📊 **Live Visualization** - Real-time inference server + Vue.js frontend
- 🧪 **70%+ Test Coverage** - 387 tests passing, production-ready codebase

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (optional but recommended)
- Node.js 18+ (for frontend visualization)

### Installation

```bash
# Clone the repository
git clone https://github.com/tachyon-beep/hamlet
cd hamlet

# Install dependencies using uv
uv sync

# Run tests (387 tests, 70%+ coverage)
uv run pytest tests/ --cov=src/townlet --cov-report=term-missing -v
```

### Run Training + Visualization

**Two-Terminal Workflow (Recommended):**

```bash
# Terminal 1: Training + Inference Server
source .venv/bin/activate
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 10000

# Terminal 2: Frontend (once checkpoints exist)
cd frontend && npm run dev
# Open http://localhost:5173
```

**The unified server:**

- Runs training in background thread
- Saves checkpoints every 100 episodes to `runs/LX_name/timestamp/checkpoints/`
- Inference server watches for new checkpoints and broadcasts state to frontend
- WebSocket server on port 8766

See [UNIFIED_SERVER_USAGE.md](UNIFIED_SERVER_USAGE.md) for complete guide.

## Training Levels (Progressive Complexity)

### Level 1: Full Observability Baseline

**Config:** `configs/level_1_full_observability.yaml`

```bash
python run_demo.py --config configs/level_1_full_observability.yaml --episodes 10000
```

**Features:**

- Agent sees full 8×8 grid (complete information)
- Standard MLP Q-Network (no memory needed)
- Sparse rewards (milestone bonuses only)
- Expected: 1000-2000 episodes to learn, peak survival 250-350 steps

**Why:** Clean baseline for comparing POMDP performance, faster learning curve.

### Level 2: Partial Observability (POMDP)

**Config:** `configs/level_2_pomdp.yaml`

```bash
python run_demo.py --config configs/level_2_pomdp.yaml --episodes 10000
```

**Features:**

- Agent sees only 5×5 local window (partial observability)
- RecurrentSpatialQNetwork with LSTM memory
- Must build mental map through exploration
- Expected: 3000-5000 episodes to learn, peak survival 150-250 steps

**Why:** Introduces working memory, spatial reasoning, and realistic cognitive constraints.

### Level 3: Temporal Mechanics

**Config:** `configs/level_3_temporal.yaml`

```bash
python run_demo.py --config configs/level_3_temporal.yaml --episodes 10000
```

**Features:**

- 24-tick day/night cycle with operating hours
- Multi-tick interactions (jobs take 5 ticks to complete)
- Time-based action masking (Bar: 6pm-4am, Job: 8am-6pm)
- Progressive benefits + completion bonuses
- LSTM learns time-dependent strategies

**Why:** Teaches temporal planning, opportunity cost, and delayed gratification.

See [docs/TRAINING_LEVELS.md](docs/TRAINING_LEVELS.md) for complete formal specification.

## The Environment

### 8 Interconnected Meters

**PRIMARY (Death Conditions):**

- `health` - Are you alive?
- `energy` - Can you move?

**SECONDARY (Strong → Primary):**

- `satiation` - Hunger (affects health AND energy)
- `fitness` - Physical condition (affects health)
- `mood` - Mental state (affects energy)

**TERTIARY (Quality of Life):**

- `hygiene` - Cleanliness (affects secondary + primary)
- `social` - Social needs (affects secondary + primary)

**RESOURCE:**

- `money` - Enables affordances ($0-$100 normalized)

### 15 Affordances

Agents interact with affordances to restore meters and earn money:

**24/7 Available:**

- `Bed` / `LuxuryBed` - Energy restoration (tiered)
- `Shower` - Hygiene restoration
- `HomeMeal` - Satiation + health
- `FastFood` - Quick satiation (fitness/health penalty)
- `Hospital` - Health restoration (tier 2, expensive)
- `Gym` - Fitness builder

**Business Hours (8am-6pm):**

- `Job` - Office work ($22.50, -15% energy)
- `Labor` - Physical labor ($30, -20% energy, -5% fitness/health)
- `Doctor` - Health restoration (tier 1, cheaper than hospital)
- `Therapist` - Mood restoration
- `Recreation` - Mood + social (8am-10pm)

**Dynamic (Time-Dependent):**

- `CoffeeShop` - Energy + mood + social (8am-6pm)
- `Bar` - Social (BEST: +50%) + mood (6pm-4am)
- `Park` - FREE fitness + social + mood (6am-10pm)

### Cascade Physics

Meters cascade downward through 10 threshold-based effects:

```
satiation < 30% → health -0.4%/tick, energy -0.4%/tick
fitness < 30%  → health -0.4%/tick (modulates base health depletion 0.5x-3.0x)
mood < 30%     → energy -0.4%/tick
hygiene < 30%  → satiation -0.4%/tick, fitness -0.4%/tick, mood -0.4%/tick
social < 30%   → satiation -0.4%/tick, fitness -0.4%/tick, mood -0.4%/tick
```

**Teaching Value:** Students experiment with cascade strengths by editing `configs/cascades.yaml`

## Project Structure

```text
hamlet/
├── src/townlet/              # Active codebase (⚠️ src/hamlet/ is legacy)
│   ├── agent/                # Neural networks (Simple, Recurrent)
│   ├── curriculum/           # Adversarial difficulty adjustment
│   ├── demo/                 # Training runner + inference server
│   ├── environment/          # Vectorized grid world + meter dynamics
│   ├── exploration/          # RND + adaptive intrinsic motivation
│   ├── population/           # Training loop coordinator
│   └── training/             # Replay buffer + state management
├── tests/test_townlet/       # 387 tests, 70%+ coverage
├── configs/                  # YAML configurations (L1-L3)
├── frontend/                 # Vue 3 + Vite visualization
├── docs/                     # Documentation
│   ├── TRAINING_LEVELS.md   # Formal level specifications
│   ├── testing/             # Test strategy and refactoring plans
│   └── ...
├── run_demo.py              # Unified server entry point
└── UNIFIED_SERVER_USAGE.md  # Complete usage guide
```

## Visualization

The frontend shows:

- **Grid View** - Agent position, affordances, interaction progress
- **Meter Bars** - All 8 meters with cascade indicators
- **Q-Value Heatmap** - Action preferences by direction
- **Time-of-Day** - Current tick in 24-tick cycle (L3)
- **Affordance Status** - Open/closed, costs, benefits

**Features:**

- Auto-reconnect to inference server
- Speed control (0.1x - 2.0x)
- Episode navigation (watch past episodes)
- Responsive design

## Development

### Run Tests

```bash
# Full test suite with coverage
uv run pytest tests/ --cov=src/townlet --cov-report=term-missing -v

# Specific test file
uv run pytest tests/test_townlet/test_affordance_effects.py -v

# Watch mode (requires pytest-watch)
uv run ptw tests/
```

### View Logs

```bash
# TensorBoard (training metrics)
tensorboard --logdir runs/L1_full_observability/2025-11-02_123456/tensorboard

# SQLite database (episode details)
sqlite3 runs/L1_full_observability/2025-11-02_123456/metrics.db
```

### Code Quality

```bash
# Linting (configured in pyproject.toml)
uv run ruff check src/

# Format code
uv run black src/ tests/

# Type checking
uv run mypy src/
```

## Current Status

**Phase 3 Complete (November 2025):**

- ✅ Vectorized GPU training environment
- ✅ Level 1-3 progressive complexity working
- ✅ Adversarial curriculum (5-stage progression)
- ✅ RND-based intrinsic motivation with adaptive annealing
- ✅ Unified training + inference server
- ✅ Vue 3 frontend with live visualization
- ✅ 70%+ test coverage (387 tests passing)
- ✅ TensorBoard integration
- ✅ SQLite metrics storage

**Phase 3.5: Multi-Day Tech Demo (Next):**

- 🎯 Validate system stability over 48+ hours (10K episodes)
- 🎯 Observe exploration→exploitation transition in production
- 🎯 Generate teaching materials from real training data

## Roadmap

### Phase 4: POMDP Extension

- Validate LSTM memory with systematic testing (ACTION #9)
- Tune recurrent architecture for spatial reasoning
- Add target network for temporal credit assignment

### Phase 5: Informed Optimization

- Profile complete system, optimize real bottlenecks
- Implement Double DQN, Dueling Architecture
- GPU optimization for RND (eliminate CPU transfers)

### Phase 6: Multi-Agent Competition

- Multiple agents compete for resources
- Theory of mind and strategic behavior
- Emergent cooperation vs competition

### Phase 7: Emergent Communication

- Family units with information sharing
- Discrete symbol communication channel
- Language grounding in shared experience

### North Star: Social Hamlet (Vision)

- 50×50 grid with dozens of agents
- Economic hierarchy and job competition
- Emergent social dynamics and territoriality
- Nash equilibria emerging naturally

See [ROADMAP.md](ROADMAP.md) for complete strategic plan.

## Technologies

- **Python 3.12** - Modern Python with typing
- **PyTorch 2.9** - GPU-accelerated neural networks
- **FastAPI + uvicorn** - Async inference server
- **Vue 3 + Vite** - Reactive frontend visualization
- **uv** - Fast Python package manager
- **pytest** - Testing framework (387 tests, 70%+ coverage)
- **TensorBoard** - Training metrics visualization
- **SQLite** - Episode metrics storage

## Documentation

- **[UNIFIED_SERVER_USAGE.md](UNIFIED_SERVER_USAGE.md)** - Complete usage guide
- **[AGENTS.md](AGENTS.md)** - Comprehensive project memory for AI assistants
- **[ROADMAP.md](ROADMAP.md)** - Strategic development plan
- **[docs/TRAINING_LEVELS.md](docs/TRAINING_LEVELS.md)** - Formal level specifications
- **[docs/testing/](docs/testing/)** - Testing strategy and refactoring plans

## Contributing

This is a pedagogical project designed to teach Deep RL concepts through hands-on experimentation. Key principles:

- **"Interesting failures" are features** - Reward hacking and cascade failures create teaching moments
- **Configuration over code** - Students experiment by editing YAML files
- **Progressive complexity** - Start simple (L1), add challenges incrementally
- **Real implementations** - No black boxes, build DRL from scratch

Feel free to experiment, extend, and learn!

## License

MIT License (see LICENSE file)

## Citation

```bibtex
@software{hamlet2025,
  title={Hamlet: A Pedagogical Deep Reinforcement Learning Environment},
  author={Tachyon-Beep},
  year={2025},
  url={https://github.com/tachyon-beep/hamlet}
}
```

## Acknowledgments

Built on foundational RL research:

- **DQN** - Mnih et al. (2015) - [Nature Paper](https://www.nature.com/articles/nature14236)
- **RND** - Burda et al. (2019) - [OpenAI Blog](https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/)
- **Adversarial Curriculum** - Inspired by OpenAI's Dota 2 project
