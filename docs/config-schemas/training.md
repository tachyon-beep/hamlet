# training.yaml Configuration

**Purpose**: Configure hyperparameters for DQN training, including learning rates, replay buffer settings, exploration strategy, and algorithm variants.

**Location**: `<config_pack>/training.yaml`

**Pattern**: All training parameters must be explicitly specified (no-defaults principle). This ensures reproducibility and prevents silent behavioral changes when code defaults evolve.

---

## Core Training Parameters

### `device` (string, REQUIRED)

**Type**: `str`
**Required**: Yes (no default)
**Example**: `device: cuda`

Hardware device for training and inference.

**Options**:
- `"cuda"`: Use GPU acceleration (recommended for training)
- `"cpu"`: Use CPU only (slower, but works without CUDA)

**Validation**: Must be one of the above values. PyTorch will raise error if CUDA requested but unavailable.

---

### `max_episodes` (integer, REQUIRED)

**Type**: `int`
**Required**: Yes (no default)
**Example**: `max_episodes: 5000`

Total number of episodes to run during training.

**Validation**: Must be positive integer (> 0)

**Pedagogical Note**: Students should experiment with different episode counts to observe convergence behavior.

---

## Q-Learning Hyperparameters

### `train_frequency` (integer, REQUIRED)

**Type**: `int`
**Required**: Yes (no default)
**Example**: `train_frequency: 4`

Number of environment steps between each training update (batch sampling from replay buffer).

**Typical Values**:
- `4`: Standard DQN (Mnih et al. 2015)
- `1`: Update every step (more stable, slower)
- `10`: Less frequent updates (faster, more off-policy)

**Validation**: Must be positive integer (> 0)

---

### `target_update_frequency` (integer, REQUIRED)

**Type**: `int`
**Required**: Yes (no default)
**Example**: `target_update_frequency: 100`

Number of **training updates** (not environment steps) between target network synchronizations.

**Typical Values**:
- `100`: Recommended for small networks (L0, L0.5)
- `500`: Acceptable for larger networks, but may cause instability
- `1000`: Too infrequent (Q-targets become stale)

**Validation**: Must be positive integer (> 0)

**Formula**: Episodes between target updates ≈ `target_update_frequency * train_frequency / avg_episode_length`

**Example**: With `train_frequency=4`, `target_update_frequency=100`, and average episode length 50 steps:
- Target updates every 100 batches
- 50 steps/episode ÷ 4 steps/batch = 12.5 batches/episode
- 100 batches ÷ 12.5 = ~8 episodes between target updates

---

### `batch_size` (integer, REQUIRED)

**Type**: `int`
**Required**: Yes (no default)
**Example**: `batch_size: 64`

Number of transitions sampled from replay buffer for each training update.

**Typical Values**:
- `32`: Small batch (faster, more variance)
- `64`: Standard DQN (good balance)
- `128`: Large batch (more stable, slower)

**Validation**: Must be positive integer (> 0)

**Note**: For recurrent networks, this is the number of *sequences* sampled (each of length `sequence_length`).

---

### `max_grad_norm` (float, REQUIRED)

**Type**: `float`
**Required**: Yes (no default)
**Example**: `max_grad_norm: 10.0`

Gradient clipping threshold to prevent exploding gradients.

**Typical Values**:
- `10.0`: Standard for feedforward networks
- `5.0`: Tighter clipping for recurrent networks
- `1.0`: Very conservative (may slow learning)

**Validation**: Must be positive float (> 0.0)

**Technical Detail**: Uses `torch.nn.utils.clip_grad_norm_()` on all network parameters.

---

### `use_double_dqn` (boolean, REQUIRED)

**Type**: `bool`
**Required**: Yes (no default)
**Example**: `use_double_dqn: false`

Selects the Q-learning algorithm variant:
- `true`: Use Double DQN (van Hasselt et al. 2016) - reduces Q-value overestimation
- `false`: Use vanilla DQN (Mnih et al. 2015) - original algorithm

**Algorithm Comparison**:

**Vanilla DQN** (use_double_dqn: false):
- Target: `Q_target = r + γ * max_a Q_target(s', a)`
- Uses target network for both action selection and evaluation
- Tends to overestimate Q-values due to max operator bias
- Good for baseline comparisons

**Double DQN** (use_double_dqn: true):
- Target: `Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))`
- Uses online network for action selection, target network for evaluation
- Reduces overestimation by decoupling selection from evaluation
- Generally converges faster and more stably

**When to Use**:
- Start with `false` to establish vanilla DQN baseline
- Switch to `true` if training shows high variance or value overestimation
- Use `true` for production training after baseline comparison

**Performance Impact**:
- Feedforward networks: Negligible (<1% overhead)
- Recurrent networks: ~50% overhead (3 forward passes vs 2 for LSTM)

**Implementation Notes**:
- Supported for both `SimpleQNetwork` (feedforward) and `RecurrentSpatialQNetwork` (LSTM)
- Checkpoint persistence: `use_double_dqn` flag saved in checkpoint metadata
- Curriculum compatibility: All levels support both algorithms

**References**:
- Mnih et al. 2015: "Human-level control through deep reinforcement learning" (vanilla DQN)
- van Hasselt et al. 2016: "Deep Reinforcement Learning with Double Q-learning" (Double DQN)

---

## Exploration Strategy

### `epsilon_start` (float, REQUIRED)

**Type**: `float`
**Required**: Yes (no default)
**Example**: `epsilon_start: 1.0`

Initial epsilon value for ε-greedy exploration (probability of random action).

**Typical Values**:
- `1.0`: Full random exploration at start (recommended)
- `0.5`: Start with some exploitation

**Validation**: Must be in range [0.0, 1.0], and >= `epsilon_min`

---

### `epsilon_decay` (float, REQUIRED)

**Type**: `float`
**Required**: Yes (no default)
**Example**: `epsilon_decay: 0.995`

Multiplicative decay factor applied to epsilon after each episode: `ε_new = ε_old * epsilon_decay`

**Typical Values**:
- `0.975`: Aggressive decay (reaches ε=0.1 at episode 90)
- `0.995`: Moderate decay (reaches ε=0.1 at episode 450)
- `0.9995`: Very slow decay (reaches ε=0.1 at episode 4600)

**Validation**: Must be in range (0.0, 1.0]

**Formula**: Episodes to reach target epsilon:
```
n = log(ε_target / ε_start) / log(epsilon_decay)
```

**Example**: To reach ε=0.1 from ε=1.0:
- `decay=0.975`: 90 episodes
- `decay=0.995`: 450 episodes

**Warning**: Slow decay (>0.995) may cause excessive exploration, preventing convergence.

---

### `epsilon_min` (float, REQUIRED)

**Type**: `float`
**Required**: Yes (no default)
**Example**: `epsilon_min: 0.01`

Minimum epsilon value (floor for epsilon decay).

**Typical Values**:
- `0.01`: Highly exploitative (1% random actions)
- `0.05`: Balanced (5% random actions)
- `0.1`: More exploratory (10% random actions)

**Validation**: Must be in range [0.0, 1.0], and <= `epsilon_start`

---

## Recurrent Network Parameters

### `sequence_length` (integer, REQUIRED)

**Type**: `int`
**Required**: Yes (no default)
**Example**: `sequence_length: 8`

Number of consecutive timesteps in each sequence when training recurrent networks (LSTM).

**Typical Values**:
- `8`: Standard for POMDP tasks
- `16`: Longer memory horizon
- `4`: Shorter sequences (faster training)

**Validation**: Must be positive integer (> 0)

**Note**: Only used when `network_type="recurrent"`. Ignored for feedforward networks.

---

## Action Space Configuration

### `enabled_actions` (list of strings, OPTIONAL)

**Type**: `list[str]`
**Required**: No (defaults to all actions enabled)
**Example**:
```yaml
training:
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
```

List of action names to enable for this config. Actions not listed are masked (not selectable).

**Purpose**: Progressive curriculum - enable subset of actions per level while maintaining same `action_dim` for checkpoint transfer.

**Validation**:
- All names must exist in global vocabulary (substrate actions + `configs/global_actions.yaml`)
- No duplicates allowed
- Empty list disables all actions (for testing only)

**See**: `docs/config-schemas/enabled_actions.md` for detailed examples

---

## Example Configuration

```yaml
training:
  # Hardware
  device: cuda
  max_episodes: 5000

  # Q-learning hyperparameters
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  use_double_dqn: true  # Enable Double DQN

  # Exploration strategy (ε-greedy)
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01

  # Recurrent networks (LSTM)
  sequence_length: 8

  # Action space (optional)
  enabled_actions:
    - "UP"
    - "DOWN"
    - "LEFT"
    - "RIGHT"
    - "INTERACT"
    - "WAIT"
    - "REST"
```

---

## Curriculum Progression

**L0_0_minimal** (Temporal credit assignment):
- `max_episodes: 500`
- `epsilon_decay: 0.995`
- `use_double_dqn: false` (baseline)

**L0_5_dual_resource** (Multiple resources):
- `max_episodes: 1000`
- `epsilon_decay: 0.975`
- `use_double_dqn: true` (improved algorithm)

**L1_full_observability** (Full observability):
- `max_episodes: 2000`
- `epsilon_decay: 0.995`
- `use_double_dqn: false` (for comparison with L2)

**L2_partial_observability** (POMDP + LSTM):
- `max_episodes: 5000`
- `sequence_length: 8`
- `epsilon_decay: 0.995`
- `use_double_dqn: true` (helps with sparse rewards)

**L3_temporal_mechanics** (Time-based dynamics):
- `max_episodes: 10000`
- `sequence_length: 16` (longer horizon for temporal planning)
- `epsilon_decay: 0.995`
- `use_double_dqn: true`

---

## Validation

**Config Loading**: `townlet.config.training.load_training_config(config_dir: Path) -> TrainingConfig`

**Unit Tests**: `tests/test_townlet/unit/config/test_training_config_dto.py`

**Integration Tests**: `tests/test_townlet/integration/test_double_dqn_training.py`

---

## Best Practices

1. **Start with vanilla DQN**: Establish baseline with `use_double_dqn: false`
2. **Compare algorithms**: Run A/B tests by toggling `use_double_dqn`
3. **Monitor target updates**: Verify target network updates frequently enough (check TensorBoard)
4. **Tune epsilon decay**: Match exploration schedule to task difficulty
5. **Document choices**: Add YAML comments explaining hyperparameter decisions
6. **No magic defaults**: Always specify all required fields explicitly

---

## Related Documentation

- **Action Space**: `docs/config-schemas/enabled_actions.md`
- **Substrate Configuration**: `docs/config-schemas/substrate.md`
- **Training System**: `docs/manual/TRAINING_SYSTEM.md`
- **TensorBoard Integration**: `docs/manual/TENSORBOARD_INTEGRATION.md`
