# Hamlet

A Deep Reinforcement Learning (DRL) demonstrator where agents learn to survive in a grid-based world by managing multiple competing needs.

## Overview

Hamlet is an educational DRL project featuring:
- **Custom DRL implementation** - Learn by building from scratch
- **Multi-meter survival mechanics** - Agents balance energy, hygiene, satiation, and money
- **Complex tradeoffs** - Every action affects multiple meters
- **Web-based visualization** - Watch agents learn in real-time
- **Multi-agent ready** - Built on PettingZoo for future expansion

## Quick Start

### Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hamlet

# Install dependencies using uv
uv sync

# Run tests
uv run pytest

# Inspect failure reasons recorded during training
python analyze_failures.py --summary --db metrics.db

# Start the web interface (when implemented)
uv run python -m hamlet.web.server
```

### Failure Analysis Cheatsheet

```bash
# Show the last five failures for the default agent
python analyze_failures.py --agent agent_0 --limit 5 --db metrics.db

# Focus on bankruptcy episodes only
python analyze_failures.py --summary --reason bankrupt --db metrics.db
```

### Development Installation

```bash
# Install with development dependencies
uv sync --extra dev

# Run linting
uv run ruff check src/

# Format code
uv run black src/ tests/

# Type checking
uv run mypy src/
```

## Project Structure

```
hamlet/
├── src/hamlet/          # Main package
│   ├── environment/     # Grid world and game mechanics
│   ├── agent/           # DRL agent implementation
│   ├── training/        # Training loops and metrics
│   └── web/             # Web visualization
├── tests/               # Test suite
├── docs/                # Documentation
│   └── plans/          # Design documents
└── pyproject.toml      # Project configuration
```

## How It Works

### The Environment

Agents live in an 8x8 grid world with four affordances:
- **Bed** - Restores energy (costs money)
- **Shower** - Restores hygiene (costs money)
- **Fridge** - Restores satiation (costs money)
- **Job** - Earns money (costs energy and hygiene)

### The Challenge

Each affordance affects multiple meters, creating complex optimization problems:
- Working earns money but drains energy and hygiene
- Services restore biological needs but cost money
- Movement gradually depletes all meters
- Agent must learn efficient work-service cycles to survive

### The Goal

Learn a policy that maximizes survival time by balancing competing needs.

## Roadmap

### Current: MVP
- [x] Project structure and design
- [ ] Environment implementation (grid, meters, affordances)
- [ ] DRL agent (DQN with experience replay)
- [ ] Training loop
- [ ] Web visualization

### Future: Multi-Agent
- [ ] Multiple agents with social interaction
- [ ] Relationship meters
- [ ] Competition for resources
- [ ] Reproduction mechanics (neural network blending)

### Vision: Cityscape
- [ ] ~1000 agents in large environment
- [ ] Complex economic system
- [ ] Job competition and property markets
- [ ] Emergent social dynamics

See [design document](docs/plans/2025-10-27-hamlet-drl-design.md) for details.

## Technologies

- **Python 3.11+** - Modern Python features
- **PettingZoo** - Multi-agent RL environment framework
- **PyTorch** - Neural network implementation
- **FastAPI** - Web visualization server
- **uv** - Fast, modern Python package manager

## Contributing

This is an educational project. Feel free to experiment, learn, and extend!

## License

[Add license information]

## Resources

- [Design Document](docs/plans/2025-10-27-hamlet-drl-design.md)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [DQN Paper](https://www.nature.com/articles/nature14236) - Mnih et al. (2015)
