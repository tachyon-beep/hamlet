# Townlet Architecture Documentation

This directory contains comprehensive architectural documentation for the Townlet system, a GPU-native Deep Reinforcement Learning environment.

## Overview

Townlet is a pedagogical RL environment where agents learn to survive by managing multiple competing needs (energy, hygiene, satiation, money, health, fitness, mood, social). The system compiles declarative YAML configurations into optimized GPU-native runtime artifacts.

## Document Index

### 01. System Overview
**File**: `01-system-overview.md`

High-level architecture covering:
- System layers (Configuration, Compilation, Runtime, Persistence)
- Component relationships
- Data flow summary
- Key design patterns

**Start here** for a bird's-eye view of the entire system.

### 02. Compilation Pipeline
**File**: `02-compilation-pipeline.md`

Detailed breakdown of the 7-stage compilation process:
- Stage 1: Parse YAML files into DTOs
- Stage 2: Build symbol tables
- Stage 3: Resolve cross-file references
- Stage 4: Cross-validation (feasibility, balance, cycles)
- Stage 5: Compute metadata and observation specs
- Stage 6: Optimization (pre-compute tensors)
- Stage 7: Emit immutable CompiledUniverse

**Read this** to understand how configs become runtime artifacts.

### 03. Training Loop
**File**: `03-training-loop.md`

Complete training pipeline:
- Main episode loop
- Step-by-step breakdown (action selection, environment step, training)
- Curriculum updates
- Checkpoint management
- Metrics logging

**Read this** to understand the runtime training process.

### 04. Environment Subsystem
**File**: `04-environment-subsystem.md`

Deep dive into VectorizedHamletEnv:
- Substrate hierarchy (Grid2D, Grid3D, GridND, Continuous, Aspatial)
- State tensor structure
- Environment step pipeline
- Meter dynamics and cascades
- Affordance engine
- VFS integration
- Action space composition

**Read this** to understand environment mechanics.

### 05. Agent & Learning Subsystem
**File**: `05-agent-learning-subsystem.md`

Neural network architectures and learning:
- SimpleQNetwork (feedforward MLP)
- RecurrentSpatialQNetwork (CNN + LSTM)
- Replay buffer architectures
- VectorizedPopulation
- Training pipeline (Q-learning, LSTM training)
- Exploration strategies (RND, ε-greedy)
- Curriculum system
- Checkpoint state

**Read this** to understand agent learning mechanics.

### 06. Data Flow & Modules
**File**: `06-data-flow-and-modules.md`

System-wide data flow and interactions:
- Complete pipeline (config → training → persistence)
- Module dependency graph
- Tensor flow through system
- Memory layout (GPU, CPU, Disk)
- Communication patterns
- Config → Runtime transformations
- Error propagation
- Performance characteristics

**Read this** to understand how data moves through the system.

## Quick Reference

### Key Abstractions

| Abstraction | Purpose | Location |
|-------------|---------|----------|
| **CompiledUniverse** | Immutable compiled artifact | `universe/compiled.py` |
| **RuntimeUniverse** | Mutable runtime views | `universe/runtime.py` |
| **Substrate** | Spatial abstraction | `substrate/` |
| **VectorizedHamletEnv** | GPU-native environment | `environment/vectorized_env.py` |
| **VectorizedPopulation** | Training coordinator | `population/vectorized.py` |
| **Q-Network** | Policy neural network | `agent/networks.py` |
| **ReplayBuffer** | Experience storage | `training/replay_buffer.py` |
| **DemoRunner** | Training orchestrator | `demo/runner.py` |

### Key Data Structures

| Structure | Shape | Description |
|-----------|-------|-------------|
| **Positions** | `[num_agents, position_dim]` | Agent positions |
| **Meters** | `[num_agents, meter_count]` | Meter values (energy, health, etc.) |
| **Observations** | `[num_agents, obs_dim]` | Full state observations |
| **Q-Values** | `[num_agents, action_dim]` | Action-value estimates |
| **Actions** | `[num_agents]` | Selected actions |
| **Rewards** | `[num_agents]` | Computed rewards |
| **Dones** | `[num_agents]` | Episode termination flags |

### Tensor Devices

All runtime tensors live on the specified device (typically CUDA):
- **dtype**: `torch.float32` (default for most tensors)
- **device**: `torch.device("cuda")` or `torch.device("cpu")`

### Configuration → Runtime Flow

```
YAML Files
    ↓
UniverseCompiler (7 stages)
    ↓
CompiledUniverse (frozen)
    ↓
RuntimeUniverse (mutable)
    ↓
VectorizedHamletEnv
    ↓
Training Loop
```

### Critical Paths

**Observation Generation**:
```
Environment State → VFS Registry → VFSObservationBuilder → Observations
```

**Action Execution**:
```
Observations → Q-Network → Q-Values → ε-greedy → Actions → Environment
```

**Training Update**:
```
ReplayBuffer → Sample Batch → Q-Network → TD Loss → Backprop → Optimizer
```

## Architecture Principles

### 1. UNIVERSE_AS_CODE
Everything is configurable via YAML. No hardcoded universe parameters.

### 2. No-Defaults Principle (PDR-002)
All behavioral parameters must be explicitly specified in configs. No implicit defaults.

### 3. Immutability
CompiledUniverse is frozen after compilation. Runtime views are mutable.

### 4. GPU-Native
All operations vectorized across `[num_agents, ...]` on GPU tensors.

### 5. Provenance Tracking
Config hash + compiler version + environment tracked in artifacts.

### 6. Separation of Concerns
Clear boundaries: Compilation ≠ Runtime ≠ Persistence.

## Common Patterns

### Vectorized Operations
```python
# All operations batch across agents
positions = substrate.move_agent(
    positions,  # [num_agents, position_dim]
    deltas      # [num_agents, position_dim]
)
# Returns: [num_agents, position_dim]
```

### Device Management
```python
# Tensors are device-aware
meters = torch.zeros(
    (num_agents, meter_count),
    dtype=torch.float32,
    device=device  # "cuda" or "cpu"
)
```

### VFS Access
```python
# VFS provides declarative state management
registry.set("energy", energy_values, writer="engine")
energy = registry.get("energy", reader="agent")
```

### Error Handling
```python
# Compilation errors are collected and reported
errors = CompilationErrorCollector(stage="Stage 3")
errors.add(error_message, code="UAC-RES-001", location="bars.yaml:energy")
errors.check_and_raise("Stage 3")
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Compilation Time | < 5s | First run, < 100ms cache hit |
| Training Throughput | 500-1000 steps/s | GPU, depends on config |
| Checkpoint Save | < 500ms | ~5 MB per checkpoint |
| Checkpoint Load | < 800ms | Includes validation |
| Episode Insert | < 1ms | SQLite database |
| GPU Memory | 2-3 GB | Typical config (4 agents) |

## Common Workflows

### Adding a New Meter
1. Add definition to `bars.yaml`
2. Update `meter_count` in metadata (automatic)
3. Recompile universe
4. Update checkpoint version if needed

### Adding a New Affordance
1. Add definition to `affordances.yaml`
2. Add to `enabled_affordances` in `training.yaml`
3. Recompile universe
4. Place in environment (automatic or via `position`)

### Adding a New Substrate
1. Implement `Substrate` interface in `substrate/`
2. Add to `SubstrateFactory`
3. Add config schema to `substrate/config.py`
4. Update action space if needed

### Debugging Training Issues
1. Check TensorBoard logs for metrics
2. Query SQLite database for episode history
3. Inspect checkpoint state
4. Enable debug logging (`logging.DEBUG`)
5. Check curriculum stage progression
6. Verify epsilon decay schedule

## File Organization

```
src/townlet/
├── agent/               # Neural network architectures
├── compiler/            # CLI entry point
├── config/              # Pydantic DTOs for configs
├── curriculum/          # Curriculum strategies
├── demo/                # Training orchestration
├── environment/         # Environment mechanics
├── exploration/         # Exploration strategies
├── population/          # Agent coordination
├── recording/           # Episode recording (optional)
├── substrate/           # Spatial abstractions
├── training/            # Training infrastructure
├── universe/            # Compilation pipeline
└── vfs/                 # Variable & Feature System
```

## Glossary

- **Affordance**: Interactable object (Bed, Hospital, Job, etc.)
- **Bar/Meter**: Resource variable (energy, health, etc.)
- **Cascade**: Meter-to-meter relationship (low satiation → drains energy)
- **Substrate**: Spatial coordinate system (Grid2D, Aspatial, etc.)
- **VFS**: Variable & Feature System (declarative state management)
- **UAC**: Universe As Code (configuration principle)
- **PDR**: Principle Design Record (architectural decision)
- **POMDP**: Partially Observable Markov Decision Process
- **RND**: Random Network Distillation (exploration method)

## Related Documentation

- **Implementation Plans**: `/home/user/hamlet/docs/plans/` (off-limits per instructions)
- **Config Schemas**: `/home/user/hamlet/docs/config-schemas/` (off-limits per instructions)
- **Task Documents**: `/home/user/hamlet/docs/tasks/` (off-limits per instructions)

Note: The `docs/` directory is off-limits per analysis instructions. These diagrams are derived purely from white-box code analysis.

## Diagram Format

All diagrams use Mermaid syntax for rendering:
- **Flowcharts**: Process flows and pipelines
- **Graphs**: Module relationships and hierarchies
- **Sequence Diagrams**: Component interactions over time
- **Class Diagrams**: Object-oriented structures

## Contributing to Diagrams

When updating diagrams:
1. Keep them **conceptual**, not implementation-detailed
2. Use **consistent styling** (colors indicate layer/purpose)
3. **Annotate** tensor shapes and data types
4. Include **performance notes** where relevant
5. Cross-reference related diagrams

## Color Coding

Diagrams use consistent color coding:
- **Blue** (`#e1f5fe`): Data structures, inputs
- **Yellow** (`#fff9c4`): Compilation artifacts
- **Purple** (`#f3e5f5`): Runtime components
- **Green** (`#c8e6c9`): Successful operations, outputs
- **Orange** (`#ffccbc`): Training operations, critical paths
- **Lavender** (`#d1c4e9`): Persistence layer

## Version Info

- **Compiler Version**: 0.1.0
- **Schema Version**: 1.0
- **Documentation Date**: 2025-11-10
- **Code Branch**: `claude/004a-compiler-implementation-011CUyzijQruciX7BGnqGPbk`
