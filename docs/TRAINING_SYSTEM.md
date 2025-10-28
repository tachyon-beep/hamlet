# Hamlet Training System Documentation

## Overview

The Hamlet training system provides a production-ready infrastructure for training Deep Reinforcement Learning (DRL) agents. It features:

- **Multi-Agent Ready**: Scalable architecture supporting 1-100 agents
- **Comprehensive Metrics**: TensorBoard, SQLite database, episode replays, live broadcasting
- **Experiment Tracking**: MLflow integration for comparing runs
- **Smart Checkpointing**: Automatic checkpoint management with best-N selection
- **Pluggable Algorithms**: Easy to add new RL algorithms beyond DQN
- **YAML Configuration**: Demo-friendly config files

## Quick Start

### 1. Run an Example Experiment

```bash
# From the hamlet root directory
python run_experiment.py configs/example_dqn.yaml
```

### 2. View Results

```bash
# TensorBoard (real-time metrics)
tensorboard --logdir runs

# MLflow (experiment comparison)
mlflow ui --backend-store-uri mlruns

# Open browser to:
# - TensorBoard: http://localhost:6006
# - MLflow: http://localhost:5000
```

### 3. Query Metrics Database

```python
import sqlite3

conn = sqlite3.connect('metrics.db')
cursor = conn.execute("""
    SELECT episode, metric_name, value
    FROM metrics
    WHERE agent_id = 'agent_0'
    ORDER BY episode
""")

for row in cursor:
    print(row)
```

## Architecture

### Component Overview

The training system consists of 5 main managers orchestrated by the Trainer:

```
Trainer (orchestrator)
├── AgentManager      # Multi-agent management, buffer switching
├── MetricsManager    # TensorBoard, DB, replays, broadcasting
├── ExperimentManager # MLflow experiment tracking
├── CheckpointManager # Model checkpointing with best-N selection
└── Environment       # HamletEnv instance
```

### AgentManager

**Purpose**: Manages multiple agents with intelligent buffer mode switching.

**Features**:
- Automatic buffer mode switching:
  - `<10 agents`: Per-agent buffers for learning isolation
  - `≥10 agents`: Shared buffer for memory efficiency
- Pluggable algorithm interface (DQN, PPO, A2C, etc.)
- Unified API for single or multi-agent training

**Example Usage**:
```python
from hamlet.training.agent_manager import AgentManager
from hamlet.training.config import AgentConfig

manager = AgentManager(buffer_size=10000, buffer_threshold=10)

# Add agents
for i in range(5):
    config = AgentConfig(agent_id=f"agent_{i}", algorithm="dqn")
    manager.add_agent(config)

# Store experience
manager.store_experience(
    "agent_0", state, action, reward, next_state, done
)

# Sample batch
batch = manager.sample_batch(batch_size=64, agent_id="agent_0")
```

### MetricsManager

**Purpose**: Comprehensive metrics tracking with multiple outputs.

**Metrics Outputs**:
1. **TensorBoard**: Real-time visualization during training
2. **SQLite Database**: Structured storage for queries
3. **Episode Replays**: Full trajectory storage (JSON)
4. **Live Broadcast**: WebSocket streaming (for viz)

**Example Usage**:
```python
from hamlet.training.metrics_manager import MetricsManager
from hamlet.training.config import MetricsConfig

config = MetricsConfig(
    tensorboard=True,
    database=True,
    replay_storage=True,
    replay_sample_rate=0.1  # Save 10% of episodes
)

manager = MetricsManager(config, experiment_name="my_experiment")

# Log episode metrics
manager.log_episode(
    episode=10,
    agent_id="agent_0",
    metrics={"total_reward": 100.5, "episode_length": 250}
)

# Query metrics
results = manager.query_metrics(
    agent_id="agent_0",
    metric_name="total_reward",
    min_episode=0,
    max_episode=100
)
```

### ExperimentManager

**Purpose**: MLflow integration for experiment tracking and comparison.

**Features**:
- Automatic experiment creation
- Parameter logging
- Metric tracking with steps
- Model artifact storage
- Run comparison in MLflow UI

**Example Usage**:
```python
from hamlet.training.experiment_manager import ExperimentManager
from hamlet.training.config import ExperimentConfig

config = ExperimentConfig(
    name="hamlet_dqn",
    description="DQN experiments",
    tracking_uri="mlruns"
)

manager = ExperimentManager(config)
manager.start_run(run_name="run_001")

# Log parameters
manager.log_params({
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon_decay": 0.995
})

# Log metrics
manager.log_metric("reward", 100.5, step=10)

manager.end_run()
```

### CheckpointManager

**Purpose**: Intelligent checkpoint management with best-N selection.

**Features**:
- Automatic checkpoint versioning
- Keep best N by metric (reward, loss, etc.)
- Or keep most recent N
- Multi-agent checkpoint support
- Metadata storage

**Example Usage**:
```python
from hamlet.training.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="checkpoints/",
    max_checkpoints=5,
    keep_best=True,
    metric_name="total_reward",
    metric_mode="max"  # or "min" for loss
)

# Save checkpoint
checkpoint_path = manager.save_checkpoint(
    episode=100,
    agents={"agent_0": agent},
    metrics={"total_reward": 150.5}
)

# Load best checkpoint
manager.load_best_checkpoint(agents)

# Get checkpoint info
info = manager.get_checkpoint_info()
print(f"Best episode: {info['best_episode']}")
print(f"Best reward: {info['best_metric_value']}")
```

## Configuration

### YAML Config Structure

```yaml
experiment:
  name: "my_experiment"
  description: "Description of experiment"
  tracking_uri: "mlruns"

environment:
  grid_width: 8
  grid_height: 8
  initial_energy: 100.0
  # ... other environment settings

agents:
  - agent_id: "agent_0"
    algorithm: "dqn"
    learning_rate: 0.001
    gamma: 0.99
    # ... other agent settings

training:
  num_episodes: 1000
  batch_size: 64
  replay_buffer_size: 10000
  # ... other training settings

metrics:
  tensorboard: true
  database: true
  replay_storage: true
  # ... other metrics settings
```

### Creating Custom Configs

1. Copy `configs/example_dqn.yaml` as a template
2. Modify experiment name and description
3. Adjust hyperparameters (learning_rate, gamma, etc.)
4. Configure metrics outputs
5. Run with `python run_experiment.py your_config.yaml`

## Adding New Algorithms

The training system supports pluggable algorithms via the `BaseAlgorithm` interface:

```python
from hamlet.agent.base_algorithm import BaseAlgorithm

class MyNewAlgorithm(BaseAlgorithm):
    def select_action(self, observation, explore=True):
        # Implement action selection
        pass

    def learn(self, batch):
        # Implement learning update
        pass

    def save(self, filepath):
        # Implement checkpoint saving
        pass

    def load(self, filepath):
        # Implement checkpoint loading
        pass
```

Then register in `AgentManager`:
```python
# In agent_manager.py, add to add_agent():
elif config.algorithm.lower() == "my_algorithm":
    agent = MyNewAlgorithm(...)
```

## Testing

### Unit Tests

```bash
# Test individual managers
uv run pytest tests/test_training/test_agent_manager.py -v
uv run pytest tests/test_training/test_metrics_manager.py -v
uv run pytest tests/test_training/test_experiment_checkpoint.py -v
```

### Integration Tests

```bash
# Test complete training pipeline
uv run pytest tests/test_training/test_trainer_integration.py -v
```

### Running All Tests

```bash
# All training tests
uv run pytest tests/test_training/ -v

# With coverage
uv run pytest tests/test_training/ --cov=hamlet.training
```

## Performance Tips

### Buffer Mode Switching

The AgentManager automatically switches buffer modes:
- **Per-agent mode** (`<10 agents`): Better learning isolation, more memory
- **Shared mode** (`≥10 agents`): Memory efficient, faster sampling

Adjust threshold in `AgentManager(buffer_threshold=N)`.

### Checkpoint Frequency

Balance checkpoint frequency vs disk I/O:
- **High frequency** (every 10 episodes): More granular recovery, more disk writes
- **Low frequency** (every 100 episodes): Less disk I/O, coarser recovery

Configure with `save_frequency` in training config.

### Metrics Sampling

Reduce metrics overhead:
- **Replay sampling**: Set `replay_sample_rate: 0.1` to save only 10% of episodes
- **Logging frequency**: Set `log_frequency: 10` to log every 10th episode
- **Database**: Disable if not needed for analysis

## Troubleshooting

### Out of Memory

**Symptoms**: Training crashes with OOM error

**Solutions**:
1. Reduce `replay_buffer_size`
2. Reduce `batch_size`
3. Increase buffer mode `buffer_threshold` for shared buffers
4. Disable replay storage if not needed

### Slow Training

**Symptoms**: Training takes longer than expected

**Solutions**:
1. Check `learning_starts` - high values delay learning
2. Reduce `log_frequency` - logging has overhead
3. Disable TensorBoard/replays if not needed
4. Use GPU: Set `device: cuda` in agent config

### Checkpoints Not Saving

**Symptoms**: No checkpoint files created

**Solutions**:
1. Check `save_frequency` - may be too high
2. Verify `checkpoint_dir` is writable
3. Check disk space

## Advanced Usage

### Multi-Agent Training (Future)

The architecture supports multi-agent training:

```yaml
agents:
  - agent_id: "agent_0"
    algorithm: "dqn"
    learning_rate: 0.001

  - agent_id: "agent_1"
    algorithm: "dqn"
    learning_rate: 0.002

  # ... up to 100 agents
```

Buffer mode will automatically switch to shared when ≥10 agents.

### Custom Metrics Queries

Use SQL for advanced metrics analysis:

```python
import sqlite3

conn = sqlite3.connect('metrics.db')

# Average reward over time
cursor = conn.execute("""
    SELECT episode, AVG(value) as avg_reward
    FROM metrics
    WHERE metric_name = 'total_reward'
    GROUP BY episode
    ORDER BY episode
""")

# Agent comparison
cursor = conn.execute("""
    SELECT agent_id, AVG(value) as avg_reward
    FROM metrics
    WHERE metric_name = 'total_reward'
    GROUP BY agent_id
""")
```

### Continuing Training

Load a checkpoint and continue training:

```python
trainer = Trainer.from_yaml("config.yaml")

# Load checkpoint
checkpoint_path = "checkpoints/checkpoint_ep500"
metadata = trainer.checkpoint_manager.load_checkpoint(
    checkpoint_path,
    trainer.agent_manager.agents
)

# Continue training from episode 500
trainer.train()  # Will start from current state
```

## API Reference

See docstrings in source code:
- `src/hamlet/training/trainer.py` - Main Trainer class
- `src/hamlet/training/agent_manager.py` - AgentManager
- `src/hamlet/training/metrics_manager.py` - MetricsManager
- `src/hamlet/training/experiment_manager.py` - ExperimentManager
- `src/hamlet/training/checkpoint_manager.py` - CheckpointManager
- `src/hamlet/training/config.py` - Configuration classes

## Contributing

When adding new features to the training system:

1. **Add tests** in `tests/test_training/`
2. **Update config** in `hamlet/training/config.py`
3. **Document** in this README
4. **Add example** config in `configs/`

## License

See LICENSE file in project root.
