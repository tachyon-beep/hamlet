# Hamlet

**A pedagogical Deep Reinforcement Learning environment designed to "trick students into learning graduate-level RL by making them think they're just playing The Sims."**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-644%2B%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-70%25-yellowgreen)](tests/)

## Overview

Hamlet is a GPU-accelerated DRL training environment where agents learn to survive by managing 8 interconnected physiological and economic meters through strategic interaction with 14 affordances. Progressive complexity levels introduce partial observability (POMDP), temporal mechanics, and intrinsic motivation.

### Key Features

- 🎮 **Vectorized GPU Training** - PyTorch tensors throughout for maximum performance
- 🧠 **Progressive Complexity** - Three training levels (L1→L2→L3) with increasing challenge
- 🔄 **Adversarial Curriculum** - Adaptive difficulty from shaped to sparse rewards (5 stages)
- 🌟 **Intrinsic Motivation** - RND-based exploration with variance-based annealing
- 📊 **Live Visualization** - Real-time inference server + Vue.js frontend
- 🎬 **Episode Recording & Replay** - Record episodes, replay in real-time, export to YouTube-ready MP4
- 🧪 **70% Test Coverage** - 644+ tests passing (73 for recording system), production-ready codebase

## Quick Start

### Prerequisites

- Python 3.13+
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

# Run tests (644+ tests, 70% coverage)
uv run pytest tests/ --cov=src/townlet --cov-report=term-missing -v
```

## Development Workflow

### Continuous Integration

GitHub Actions keeps the main branch green:

| Workflow | Trigger | What it runs |
| --- | --- | --- |
| `Lint` | push / PR | Ruff (`ruff check`), Black (`--check`), Mypy |
| `Tests` | push / PR | `pytest` (default suite, skips `slow`) |
| `Full Test Suite` | nightly @ 06:00 UTC & manual dispatch | `pytest -m "slow or not slow"` to exercise the entire matrix |

All workflows use `uv` to create the environment and install `.[dev]`, so local parity is as simple as `uv sync`.

### Run Training + Visualization

**Two-Terminal Workflow (Recommended):**

```bash
# Terminal 1: Training + Inference Server
source .venv/bin/activate
python scripts/run_demo.py --config configs/L1_full_observability --episodes 10000

# Terminal 2: Frontend (once checkpoints exist)
cd frontend && npm run dev
# Open http://localhost:5173
```

**The unified server:**

- Runs training in background thread
- Saves checkpoints every 100 episodes to `runs/LX_name/timestamp/checkpoints/`
- Inference server watches for new checkpoints and broadcasts state to frontend
- WebSocket server on port 8766

See [docs/manual/UNIFIED_SERVER_USAGE.md](docs/manual/UNIFIED_SERVER_USAGE.md) for complete guide.

## Training Levels (Progressive Complexity)

### Level 1: Full Observability Baseline

**Config:** `configs/L1_full_observability`

```bash
python scripts/run_demo.py --config configs/L1_full_observability --episodes 10000
```

**Features:**

- Agent sees full 8×8 grid (complete information)
- Standard MLP Q-Network (no memory needed)
- Sparse rewards (milestone bonuses only)
- Expected: 1000-2000 episodes to learn, peak survival 250-350 steps

**Why:** Clean baseline for comparing POMDP performance, faster learning curve.

### Level 2: Partial Observability (POMDP)

**Config:** `configs/L2_partial_observability`

```bash
python scripts/run_demo.py --config configs/L2_partial_observability --episodes 10000
```

**Features:**

- Agent sees only 5×5 local window (partial observability)
- RecurrentSpatialQNetwork with LSTM memory
- Must build mental map through exploration
- Expected: 3000-5000 episodes to learn, peak survival 150-250 steps

**Why:** Introduces working memory, spatial reasoning, and realistic cognitive constraints.

### Level 3: Temporal Mechanics

**Config:** `configs/L3_temporal_mechanics`

```bash
python scripts/run_demo.py --config configs/L3_temporal_mechanics --episodes 10000
```

**Features:**

- 24-tick day/night cycle with operating hours
- Multi-tick interactions (jobs take 5 ticks to complete)
- Time-based action masking (Bar: 6pm-4am, Job: 8am-6pm)
- Progressive benefits + completion bonuses
- LSTM learns time-dependent strategies

**Why:** Teaches temporal planning, opportunity cost, and delayed gratification.

See [docs/architecture/TRAINING_LEVELS.md](docs/architecture/TRAINING_LEVELS.md) for complete formal specification.

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

### 14 Affordances

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

### Observation Space

The observation space is **standardized across all curriculum levels** to enable transfer learning and observation stability.

#### Fixed Affordance Vocabulary

All levels observe the **same 14 affordances** in their state representation, even if not all are deployed in that level:

- `Bed`, `LuxuryBed`, `Shower`, `HomeMeal`, `FastFood`, `Doctor`, `Hospital`, `Therapist`, `Recreation`, `Bar`, `Job`, `Labor`, `Gym`, `Park`

**Key Insight**: A model trained on L0 (minimal) can be promoted to L1 (full) without architecture changes because the affordance encoding dimension is constant.

#### Full Observability (L1)

**Observation components**:

- **Grid encoding**: `grid_size × grid_size` one-hot (e.g., 64 dims for 8×8 grid)
- **Meters**: 8 normalized values [0.0-1.0] (energy, health, satiation, money, mood, social, fitness, hygiene)
- **Affordance at position**: 15 one-hot (14 affordances + "none")
- **Temporal extras**: 4 values (time_of_day, retirement_age, interaction_progress, interaction_ticks)

**Dimensions by level**:

- **L0_minimal**: 36 dims (3×3 grid=9 + 8 meters + 15 affordances + 4 extras)
- **L0_5_dual_resource**: 76 dims (7×7 grid=49 + 8 meters + 15 affordances + 4 extras)
- **L1_full_observability**: 91 dims (8×8 grid=64 + 8 meters + 15 affordances + 4 extras)

**Network**: Standard MLP Q-Network (~26K-70K params depending on grid size)

#### Partial Observability (L2 POMDP)

**Observation components**:

- **Local grid**: 5×5 window (25 dims) - agent sees only nearby region
- **Position**: Normalized (x, y) (2 dims) - "where am I on the grid?"
- **Meters**: 8 normalized values (8 dims)
- **Affordance at position**: 15 one-hot (15 dims)
- **Temporal extras**: 4 values (4 dims)

**Total**: 54 dimensions (fixed regardless of full grid size)

**Network**: RecurrentSpatialQNetwork with LSTM (~600K params) for spatial memory

**Challenge**: Agent must build mental map through exploration under uncertainty.

#### Action Space

**5 discrete actions** (currently hardcoded, will move to YAML per TASK-003):

- `UP` = 0
- `DOWN` = 1
- `LEFT` = 2
- `RIGHT` = 3
- `INTERACT` = 4

**Note**: Action space will become configurable to support diagonal movement, rest actions, and alternative universes.

#### Key Design Principles

1. **Observation stability**: Same affordance vocabulary across all levels
2. **Transfer learning**: Models trained on smaller grids work on larger grids
3. **Temporal awareness**: All levels include time-based features for L3 temporal mechanics
4. **POMDP support**: Partial observability uses fixed 5×5 window regardless of full grid size

## Project Structure

```text
hamlet/
├── src/townlet/              # Active codebase
│   ├── agent/                # Neural networks (Simple, Recurrent)
│   ├── curriculum/           # Adversarial difficulty adjustment
│   ├── demo/                 # Training runner + inference server
│   ├── environment/          # Vectorized grid world + meter dynamics
│   ├── exploration/          # RND + adaptive intrinsic motivation
│   ├── population/           # Training loop coordinator
│   ├── recording/            # Episode recording and replay system
│   └── training/             # Replay buffer + state management
├── tests/test_townlet/       # 644+ tests, 70% coverage
├── configs/                  # YAML configurations (L1-L3)
├── frontend/                 # Vue 3 + Vite visualization
├── scripts/                  # Utility scripts
│   └── run_demo.py           # Unified server entry point
└── docs/                     # Documentation
    ├── architecture/         # System design and roadmap
    └── manual/               # User guides
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

**Phase 3 Complete (2025-11-04):**

- ✅ Vectorized GPU training environment
- ✅ Level 1-3 progressive complexity working
- ✅ Adversarial curriculum (5-stage progression)
- ✅ RND-based intrinsic motivation with adaptive annealing
- ✅ Unified training + inference server
- ✅ Vue 3 frontend with live visualization
- ✅ Episode recording and replay system
- ✅ 70% test coverage (644+ tests passing)
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

See [docs/architecture/ROADMAP.md](docs/architecture/ROADMAP.md) for complete strategic plan.

## Technologies

- **Python 3.13** - Modern Python baseline
- **PyTorch 2.9** - GPU-accelerated neural networks
- **FastAPI + uvicorn** - Async inference server
- **Vue 3 + Vite** - Reactive frontend visualization
- **uv** - Fast Python package manager
- **pytest** - Testing framework (644+ tests, 70% coverage)
- **TensorBoard** - Training metrics visualization
- **SQLite** - Episode metrics storage

## Documentation

- **[docs/manual/UNIFIED_SERVER_USAGE.md](docs/manual/UNIFIED_SERVER_USAGE.md)** - Complete usage guide
- **[docs/architecture/ROADMAP.md](docs/architecture/ROADMAP.md)** - Strategic development plan
- **[docs/architecture/TRAINING_LEVELS.md](docs/architecture/TRAINING_LEVELS.md)** - Formal level specifications
- **[docs/manual/REPLAY_USAGE.md](docs/manual/REPLAY_USAGE.md)** - Real-time episode replay system
- **[docs/manual/VIDEO_EXPORT_USAGE.md](docs/manual/VIDEO_EXPORT_USAGE.md)** - Video export for YouTube
- **[docs/manual/RECORDING_SYSTEM_SUMMARY.md](docs/manual/RECORDING_SYSTEM_SUMMARY.md)** - Complete recording system overview

## Contributing

This is a pedagogical project designed to teach Deep RL concepts through hands-on experimentation. Key principles:

- **"Interesting failures" are features** - Reward hacking and cascade failures create teaching moments
- **Configuration over code** - Students experiment by editing YAML files
- **Progressive complexity** - Start simple (L1), add challenges incrementally
- **Real implementations** - No black boxes, build DRL from scratch

Feel free to experiment, extend, and learn!

## License

MIT License - see [LICENSE](LICENSE) file for details

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
