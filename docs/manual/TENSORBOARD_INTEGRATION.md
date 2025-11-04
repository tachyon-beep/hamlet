# TensorBoard Integration for Hamlet

**Status:** ✅ READY TO USE  
**Version:** Initial implementation (November 1, 2025)  
**Module:** `townlet.training.tensorboard_logger`

## Overview

TensorBoard integration provides real-time visualization of training metrics alongside the existing SQLite database. The two systems complement each other:

- **TensorBoard**: Real-time visualization, interactive exploration, gradient tracking
- **SQLite Database**: Structured queries, long-term storage, reproducibility

## Quick Start

### 1. Basic Integration

Add TensorBoard logging to your training loop in ~5 lines:

```python
from townlet.training.tensorboard_logger import TensorBoardLogger

# Initialize logger
tb_logger = TensorBoardLogger(
    log_dir="runs/my_experiment",
    flush_every=10,  # Write to disk every 10 episodes
)

# In your training loop
for episode in range(max_episodes):
    # ... run episode ...
    
    # Log episode metrics
    tb_logger.log_episode(
        episode=episode,
        survival_time=survival_time,
        total_reward=total_reward,
        extrinsic_reward=extrinsic_reward,
        intrinsic_reward=intrinsic_reward,
        curriculum_stage=curriculum.tracker.agent_stages[0].item(),
        epsilon=exploration.rnd.epsilon,
        intrinsic_weight=exploration.get_intrinsic_weight(),
    )

# Close when done
tb_logger.close()
```

### 2. View in Browser

```bash
# Start TensorBoard server
tensorboard --logdir runs

# Open browser to http://localhost:6006
```

### 3. Compare Multiple Runs

```bash
# Run multiple experiments
python demo_runner.py configs/level_1.yaml --log-dir runs/level_1
python demo_runner.py configs/level_2.yaml --log-dir runs/level_2

# TensorBoard automatically compares them
tensorboard --logdir runs
```

## Integration Points

### Current System: `demo/runner.py`

The `DemoRunner` class currently logs to SQLite. Add TensorBoard logging alongside:

```python
class DemoRunner:
    def __init__(self, config_path, db_path, checkpoint_dir, max_episodes):
        # Existing initialization...
        self.db = DemoDatabase(self.db_path)
        
        # NEW: Add TensorBoard logger
        self.tb_logger = TensorBoardLogger(
            log_dir=checkpoint_dir / "tensorboard",
            flush_every=10,
            log_histograms=True,
        )
    
    def run(self):
        # Training loop...
        for episode in range(self.max_episodes):
            # ... run episode ...
            
            # Existing: Log to SQLite
            self.db.insert_episode(
                episode_id=self.current_episode,
                timestamp=time.time(),
                survival_time=survival_time,
                total_reward=episode_reward,
                # ... other fields ...
            )
            
            # NEW: Log to TensorBoard
            self.tb_logger.log_episode(
                episode=self.current_episode,
                survival_time=survival_time,
                total_reward=episode_reward,
                extrinsic_reward=extrinsic_reward,
                intrinsic_reward=intrinsic_reward,
                curriculum_stage=self.curriculum.tracker.agent_stages[0].item(),
                epsilon=self.exploration.rnd.epsilon,
                intrinsic_weight=self.exploration.get_intrinsic_weight(),
            )
            
            # Log training step metrics (optional, more detailed)
            if step % 100 == 0:  # Every 100 training steps
                self.tb_logger.log_training_step(
                    step=global_step,
                    td_error=td_error,  # From population.step_population()
                    q_values=q_values,  # From Q-network forward pass
                    loss=loss,  # From training loss
                )
```

### Integration with `VectorizedPopulation`

For training-level metrics (Q-values, TD error, gradients):

```python
class VectorizedPopulation:
    def __init__(self, ..., tensorboard_logger=None):
        self.tb_logger = tensorboard_logger
        self.training_steps = 0
    
    def step_population(self, envs):
        # ... existing training logic ...
        
        # Log Q-value statistics
        if self.tb_logger and self.training_steps % 100 == 0:
            with torch.no_grad():
                q_values = self.q_network(self.current_obs)
                self.tb_logger.log_training_step(
                    step=self.training_steps,
                    q_values=q_values,
                )
        
        self.training_steps += 1
        return batch_state
    
    def train_q_network(self, batch):
        # ... DQN training ...
        
        # Log loss and TD error
        if self.tb_logger:
            self.tb_logger.log_training_step(
                step=self.training_steps,
                td_error=td_error.mean().item(),
                loss=loss.item(),
            )
```

## Available Metrics

### Episode-Level Metrics

Logged once per episode:

```python
tb_logger.log_episode(
    episode=100,
    survival_time=250,          # Steps survived
    total_reward=42.5,          # Combined reward
    extrinsic_reward=10.0,      # Environment reward
    intrinsic_reward=32.5,      # RND novelty reward
    curriculum_stage=3,         # Current difficulty (1-5)
    epsilon=0.1,                # Exploration rate
    intrinsic_weight=0.5,       # Intrinsic motivation weight
)
```

**TensorBoard Panels:**

- `Episode/Survival_Time` - Learning curve
- `Episode/Total_Reward` - Cumulative reward
- `Episode/Intrinsic_Ratio` - Novelty vs environment reward
- `Curriculum/Stage` - Difficulty progression
- `Exploration/Epsilon` - Exploration decay
- `Exploration/Intrinsic_Weight` - Motivation annealing

### Training-Step Metrics

Logged during training (higher frequency):

```python
tb_logger.log_training_step(
    step=10000,
    td_error=0.5,                           # Temporal difference error
    q_values=torch.tensor([1.0, 2.0]),     # Q-value distribution
    loss=0.3,                               # Training loss
    rnd_prediction_error=0.1,               # Novelty detection error
)
```

**TensorBoard Panels:**

- `Training/TD_Error` - Value function learning
- `Training/Loss` - DQN loss curve
- `Training/Q_Mean` - Average Q-value (check for divergence)
- `Training/Q_Std` - Q-value variance (check for collapse)
- `Training/RND_Error` - Novelty detection quality

### Meter Dynamics

Logged per-step within episodes:

```python
tb_logger.log_meters(
    episode=100,
    step=50,
    meters={
        'health': 0.8,
        'energy': 0.6,
        'satiation': 0.7,
        'mood': 0.5,
        'social': 0.4,
        'hygiene': 0.9,
        'fitness': 0.8,
        'money': 0.3,
    }
)
```

**TensorBoard Panels:**

- `Meters/Health`, `Meters/Energy`, etc. - Meter trajectories over time
- Use for debugging cascade effects and death conditions

### Network Statistics

Logged every N episodes:

```python
tb_logger.log_network_stats(
    episode=100,
    q_network=population.q_network,
    target_network=population.target_network,
    optimizer=population.optimizer,
)
```

**TensorBoard Panels:**

- `Weights/*` - Weight distributions (histograms)
- `Gradients/Total_Norm` - Gradient explosion/vanishing detection
- `Training/Learning_Rate` - LR schedule visualization

## Advanced Features

### 1. Hyperparameter Comparison

Compare different hyperparameters:

```python
# At end of training
tb_logger.log_hyperparameters(
    hparams={
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'epsilon_decay': 0.995,
        'intrinsic_weight': 1.0,
        'curriculum': 'adversarial',
    },
    metrics={
        'final_survival': 250,
        'final_reward': 100.5,
        'episodes_to_stage_5': 3000,
    }
)
```

TensorBoard will show a parallel coordinates plot comparing all runs.

### 2. Affordance Usage Tracking

Visualize which affordances agents use:

```python
tb_logger.log_affordance_usage(
    episode=100,
    affordance_counts={
        'Bed': 15,
        'Shower': 8,
        'HomeMeal': 12,
        'Job': 5,
        'Park': 3,
    }
)
```

### 3. Multi-Agent Scenarios

Track individual agents separately:

```python
for agent_id in agent_ids:
    tb_logger.log_episode(
        episode=episode,
        survival_time=survival_times[agent_id],
        total_reward=rewards[agent_id],
        agent_id=agent_id,  # Creates separate panels
    )
```

TensorBoard will create separate panels for each agent (e.g., `agent_0/Episode/Survival_Time`).

### 4. Custom Metrics

Log experimental metrics:

```python
tb_logger.log_custom_metric(
    tag="Debug/StateEntropy",
    value=entropy,
    step=episode,
)
```

## Performance Considerations

### Overhead

- **Episode logging**: ~1ms per call (negligible)
- **Training step logging**: ~2ms per call
- **Network stats**: ~50ms per call (log every 100 episodes)

### Best Practices

1. **Flush frequency**: Set `flush_every=10` for good balance
2. **Histogram logging**: Disable for production (`log_histograms=False`)
3. **Gradient logging**: Only enable for debugging (`log_gradients=False`)
4. **Training step logging**: Sample every 100 steps, not every step

```python
# Efficient configuration
tb_logger = TensorBoardLogger(
    log_dir="runs/production",
    flush_every=10,           # Flush every 10 episodes
    log_histograms=False,     # Disable for speed
    log_gradients=False,      # Disable for speed
)

# Log training metrics sparingly
if step % 100 == 0:
    tb_logger.log_training_step(...)
```

## Migration Path

### Phase 1: Minimal Integration (5 minutes)

Add episode-level logging only:

```python
# In DemoRunner.__init__
self.tb_logger = TensorBoardLogger(log_dir=checkpoint_dir / "tensorboard")

# In training loop
self.tb_logger.log_episode(episode, survival_time, total_reward, ...)
```

**Benefit:** Instant visualization of learning curves

### Phase 2: Training Metrics (15 minutes)

Add training-step logging:

```python
# In VectorizedPopulation.__init__
self.tb_logger = tensorboard_logger

# In step_population
if self.training_steps % 100 == 0:
    self.tb_logger.log_training_step(...)
```

**Benefit:** Monitor Q-value divergence, TD errors

### Phase 3: Full Integration (30 minutes)

Add meter tracking, affordance usage, network stats:

```python
# Per-step meter logging
tb_logger.log_meters(episode, step, meters)

# Per-episode affordance tracking
tb_logger.log_affordance_usage(episode, affordance_counts)

# Every 100 episodes
if episode % 100 == 0:
    tb_logger.log_network_stats(episode, q_network, optimizer)
```

**Benefit:** Complete visibility into agent behavior

## Comparison with Existing Systems

| Feature | SQLite (Current) | TensorBoard (New) | MLflow (Planned) |
|---------|-----------------|-------------------|------------------|
| Real-time viz | ❌ | ✅ | ❌ |
| Queries | ✅ | ❌ | ❌ |
| Histograms | ❌ | ✅ | ✅ |
| Comparisons | Manual | ✅ Automatic | ✅ Automatic |
| Storage | Permanent | Temporary | Permanent |
| Use case | Production data | Development | Experiment tracking |

**Recommendation:** Use all three!

- **SQLite**: Permanent production data, queries
- **TensorBoard**: Real-time development, debugging
- **MLflow**: Experiment comparison, hyperparameter sweeps

## Example: Full Integration

```python
# demo/runner.py
class DemoRunner:
    def __init__(self, config_path, db_path, checkpoint_dir, max_episodes):
        # Existing
        self.db = DemoDatabase(db_path)
        
        # NEW: TensorBoard
        self.tb_logger = TensorBoardLogger(
            log_dir=checkpoint_dir / "tensorboard",
            flush_every=10,
        )
    
    def run(self):
        # Create components with TensorBoard logger
        self.population = VectorizedPopulation(
            ...,
            tensorboard_logger=self.tb_logger,  # Pass logger
        )
        
        for episode in range(self.max_episodes):
            # Run episode
            survival_time, rewards, affordance_counts = self._run_episode()
            
            # Log to both systems
            self.db.insert_episode(...)  # SQLite
            self.tb_logger.log_episode(  # TensorBoard
                episode=episode,
                survival_time=survival_time,
                total_reward=rewards['total'],
                extrinsic_reward=rewards['extrinsic'],
                intrinsic_reward=rewards['intrinsic'],
                curriculum_stage=self.curriculum.current_stage,
                epsilon=self.exploration.epsilon,
                intrinsic_weight=self.exploration.intrinsic_weight,
            )
            
            # Affordance tracking
            self.tb_logger.log_affordance_usage(episode, affordance_counts)
            
            # Network stats every 100 episodes
            if episode % 100 == 0:
                self.tb_logger.log_network_stats(
                    episode,
                    self.population.q_network,
                    optimizer=self.population.optimizer,
                )
        
        # Hyperparameter summary
        self.tb_logger.log_hyperparameters(
            hparams=self._get_hparams(),
            metrics=self._get_final_metrics(),
        )
        
        self.tb_logger.close()
```

## Next Steps

1. **Immediate (5 min):** Add `log_episode()` to `demo/runner.py`
2. **Short-term (30 min):** Add `log_training_step()` to `VectorizedPopulation`
3. **Medium-term (2 hours):** Add meter tracking and affordance usage
4. **Long-term (1 day):** Full integration with all metrics

## Testing

```bash
# Run a short training session
PYTHONPATH=src python -m townlet.demo.runner \
    configs/level_1_1_integration_test.yaml \
    test_demo.db \
    test_checkpoints \
    100

# Start TensorBoard
tensorboard --logdir test_checkpoints/tensorboard

# Open http://localhost:6006
```

You should see:

- Episode/Survival_Time curve
- Episode/Total_Reward curve
- Curriculum/Stage progression
- Exploration/Epsilon decay

## Troubleshooting

**Issue:** "No dashboards are active for the current data set"  
**Solution:** Make sure you're logging at least one scalar metric with `log_episode()`

**Issue:** "Slow performance"  
**Solution:** Increase `flush_every`, disable `log_histograms` and `log_gradients`

**Issue:** "TensorBoard shows old data"  
**Solution:** Delete the log directory or create a new one with a timestamp:

```python
log_dir=f"runs/exp_{int(time.time())}"
```

**Issue:** "Multi-agent panels cluttered"  
**Solution:** Use TensorBoard regex filtering: `agent_0/.*` to show only one agent
