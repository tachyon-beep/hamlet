# TASK-005: BRAIN_AS_CODE - Agent Architecture Configuration

## Problem: Hardcoded Network Architecture Prevents Experimentation

### Current State

Agent network architecture and learning hyperparameters are **hardcoded in Python**, preventing experimentation without modifying source code.

**Current Implementation** (`src/townlet/agent/networks.py` + `src/townlet/population/vectorized.py`):

```python
# Hardcoded network architecture
class SimpleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # ❌ Layer sizes hardcoded
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, action_dim)

    def forward(self, x):
        # ❌ Activation function hardcoded
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_head(x)

class RecurrentSpatialQNetwork(nn.Module):
    def __init__(self, ...):
        # ❌ CNN architecture hardcoded
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # ❌ LSTM size hardcoded
        self.lstm = nn.LSTM(192, 256, batch_first=True)
        # ❌ Q-head architecture hardcoded
        self.q_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

# Hardcoded training hyperparameters
class VectorizedPopulation:
    def __init__(self, ...):
        # ❌ Optimizer hardcoded to Adam
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # ❌ Loss function hardcoded to MSE
        self.loss_fn = nn.MSELoss()
```

### Why This Violates BRAIN_AS_CODE

1. **No Architecture Experimentation**: Cannot test different network depths without code changes:
   - Try 3-layer vs 5-layer MLPs
   - Test different hidden sizes (128, 256, 512)
   - Experiment with activation functions (ReLU, GELU, Swish)
   - Compare CNN architectures for vision encoder

2. **Hardcoded Optimizer Choice**: Cannot experiment with:
   - Different optimizers (Adam, AdamW, RMSprop, SGD)
   - Learning rate schedules (step decay, cosine annealing, warmup)
   - Optimizer hyperparameters (betas, weight decay, momentum)

3. **LSTM Architecture Locked**: Cannot adjust:
   - Hidden state size (256 → 128 or 512?)
   - Number of LSTM layers (1 vs 2)
   - Dropout between layers
   - Bidirectional vs unidirectional

4. **Cannot Reproduce Experiments**: "What network architecture did experiment #42 use?" → Must read git history

5. **Coupling Between Universe and Brain**: Network architecture should be separate from universe config (bars, actions, affordances)

## Solution: BRAIN_AS_CODE Configuration System

### Proposed Architecture

Define agent architecture and learning configuration in YAML, separate from universe config.

**Create `brain.yaml` in config packs:**

```yaml
version: "1.0"
description: "Agent brain configuration for L0_minimal"

# Network architecture selection
architecture:
  type: "feedforward"  # or "recurrent" for LSTM

  # Feedforward architecture (for full observability)
  feedforward:
    hidden_layers: [256, 128]  # List of hidden layer sizes
    activation: "relu"  # relu, gelu, swish, tanh
    output_activation: null  # null = linear output for Q-values
    dropout: 0.0  # Dropout rate (0.0 = no dropout)
    layer_norm: false  # Apply layer normalization

  # Recurrent architecture (for partial observability)
  recurrent:
    # Vision encoder (CNN for local window)
    vision_encoder:
      type: "cnn"
      channels: [32, 64]  # Channel progression
      kernel_sizes: [3, 3]  # Kernel size per layer
      strides: [1, 1]
      padding: [1, 1]
      activation: "relu"
      pool: null  # null, "max", "avg"

    # Position encoder (MLP for x,y coords)
    position_encoder:
      hidden_sizes: [32]
      activation: "relu"

    # Meter encoder (MLP for 8 meters)
    meter_encoder:
      hidden_sizes: [32]
      activation: "relu"

    # LSTM memory
    lstm:
      hidden_size: 256
      num_layers: 1
      dropout: 0.0  # Between LSTM layers if num_layers > 1
      bidirectional: false

    # Q-value head
    q_head:
      hidden_layers: [128]
      activation: "relu"
      output_activation: null

# Learning configuration
learning:
  # Optimizer
  optimizer:
    type: "adam"  # adam, adamw, rmsprop, sgd
    learning_rate: 0.00025
    # Optimizer-specific parameters
    adam:
      betas: [0.9, 0.999]
      eps: 1.0e-8
      weight_decay: 0.0  # L2 regularization
    adamw:
      betas: [0.9, 0.999]
      eps: 1.0e-8
      weight_decay: 0.01
    sgd:
      momentum: 0.9
      nesterov: true
      weight_decay: 0.0

  # Loss function
  loss:
    type: "mse"  # mse, huber, smooth_l1
    huber_delta: 1.0  # For Huber loss

  # Training schedule
  schedule:
    type: "constant"  # constant, step_decay, cosine, exponential
    step_decay:
      step_size: 1000  # Episodes between LR drops
      gamma: 0.1  # LR multiplier
    cosine:
      T_max: 5000  # Cosine period (max episodes)
      eta_min: 0.00001  # Minimum LR
    exponential:
      gamma: 0.9999  # LR decay per episode

  # Gradient clipping
  gradient_clipping:
    enabled: true
    max_norm: 10.0
    norm_type: 2.0  # L2 norm

  # Regularization
  regularization:
    l1_lambda: 0.0  # L1 penalty weight
    l2_lambda: 0.0  # L2 penalty weight (separate from optimizer weight_decay)
    entropy_bonus: 0.0  # Policy entropy bonus (for policy gradient methods)

# Q-learning configuration
q_learning:
  gamma: 0.99  # Discount factor
  target_update_frequency: 100  # Steps between target network updates
  double_dqn: false  # Use Double DQN
  dueling_dqn: false  # Use Dueling DQN architecture

# Experience replay configuration
replay:
  capacity: 10000
  batch_size: 64
  sequence_length: 8  # For recurrent networks (LSTM)
  prioritized: false  # Use prioritized experience replay
  priority_alpha: 0.6  # Prioritization exponent
  priority_beta: 0.4  # Importance sampling exponent
  priority_beta_annealing: true  # Anneal beta to 1.0

# Initialization
initialization:
  weights: "xavier_uniform"  # xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal
  bias: "zeros"  # zeros, ones, constant
  lstm_forget_bias: 1.0  # LSTM forget gate bias initialization
```

### Architecture Examples

**Example 1: L0 Minimal (Simple MLP)**

```yaml
# configs/L0_minimal/brain.yaml
version: "1.0"
description: "Simple feedforward Q-network for L0 temporal credit assignment"

architecture:
  type: "feedforward"
  feedforward:
    hidden_layers: [128, 64]  # Smaller network for simple task
    activation: "relu"
    dropout: 0.0
    layer_norm: false

learning:
  optimizer:
    type: "adam"
    learning_rate: 0.001  # Higher LR for faster learning
    adam:
      betas: [0.9, 0.999]
      weight_decay: 0.0
  loss:
    type: "mse"
  schedule:
    type: "constant"
  gradient_clipping:
    enabled: true
    max_norm: 10.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  double_dqn: false
  dueling_dqn: false

replay:
  capacity: 5000  # Smaller buffer for simple task
  batch_size: 32
  prioritized: false
```

**Example 2: L1 Full Observability (Larger MLP)**

```yaml
# configs/L1_full_observability/brain.yaml
version: "1.0"
description: "Standard MLP Q-network for full observability baseline"

architecture:
  type: "feedforward"
  feedforward:
    hidden_layers: [256, 128]  # Standard DQN architecture
    activation: "relu"
    dropout: 0.0
    layer_norm: false

learning:
  optimizer:
    type: "adam"
    learning_rate: 0.00025  # Atari DQN standard
    adam:
      betas: [0.9, 0.999]
      weight_decay: 0.0
  loss:
    type: "mse"
  schedule:
    type: "constant"
  gradient_clipping:
    enabled: true
    max_norm: 10.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  double_dqn: true  # Use Double DQN for better value estimates
  dueling_dqn: false

replay:
  capacity: 10000
  batch_size: 64
  prioritized: false
```

**Example 3: L2 POMDP (LSTM with CNN Vision)**

```yaml
# configs/L2_partial_observability/brain.yaml
version: "1.0"
description: "Recurrent spatial Q-network with LSTM for POMDP"

architecture:
  type: "recurrent"
  recurrent:
    vision_encoder:
      type: "cnn"
      channels: [32, 64]
      kernel_sizes: [3, 3]
      strides: [1, 1]
      padding: [1, 1]
      activation: "relu"
      pool: null

    position_encoder:
      hidden_sizes: [32]
      activation: "relu"

    meter_encoder:
      hidden_sizes: [32]
      activation: "relu"

    lstm:
      hidden_size: 256
      num_layers: 1
      dropout: 0.0
      bidirectional: false

    q_head:
      hidden_layers: [128]
      activation: "relu"

learning:
  optimizer:
    type: "adam"
    learning_rate: 0.0001  # Lower LR for recurrent networks
    adam:
      betas: [0.9, 0.999]
      weight_decay: 0.0
  loss:
    type: "huber"  # Huber loss more stable for recurrent
    huber_delta: 1.0
  schedule:
    type: "constant"
  gradient_clipping:
    enabled: true
    max_norm: 10.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  double_dqn: true
  dueling_dqn: false

replay:
  capacity: 10000
  batch_size: 16  # Smaller batch for recurrent (memory intensive)
  sequence_length: 8  # LSTM sequence length
  prioritized: false
```

**Example 4: Experimental Deep Network**

```yaml
# configs/experiments/deep_network/brain.yaml
version: "1.0"
description: "Experimental deep network with layer norm and dropout"

architecture:
  type: "feedforward"
  feedforward:
    hidden_layers: [512, 256, 256, 128]  # Deeper network
    activation: "gelu"  # GELU activation
    dropout: 0.1  # Light dropout
    layer_norm: true  # Layer normalization for stability

learning:
  optimizer:
    type: "adamw"  # AdamW with weight decay
    learning_rate: 0.0003
    adamw:
      betas: [0.9, 0.999]
      weight_decay: 0.01  # L2 regularization
  loss:
    type: "huber"
    huber_delta: 1.0
  schedule:
    type: "cosine"  # Cosine annealing
    cosine:
      T_max: 5000
      eta_min: 0.00001
  gradient_clipping:
    enabled: true
    max_norm: 5.0  # Tighter clipping for deep network

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  double_dqn: true
  dueling_dqn: true  # Dueling architecture

replay:
  capacity: 50000  # Larger buffer
  batch_size: 128
  prioritized: true  # Prioritized replay
  priority_alpha: 0.6
  priority_beta: 0.4
  priority_beta_annealing: true

initialization:
  weights: "kaiming_uniform"  # Kaiming init for deeper networks
  bias: "zeros"
```

## Implementation Plan

### Phase 1: Create Brain Config Schema

**File**: `src/townlet/agent/brain_config.py`

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional

class FeedforwardConfig(BaseModel):
    """Feedforward MLP architecture configuration."""
    hidden_layers: list[int] = Field(min_length=1)  # REQUIRED: No default!
    activation: Literal["relu", "gelu", "swish", "tanh", "elu"]  # REQUIRED
    output_activation: Literal["linear", "tanh"] | None = None
    dropout: float = Field(ge=0.0, lt=1.0)  # REQUIRED
    layer_norm: bool  # REQUIRED

class CNNConfig(BaseModel):
    """CNN architecture configuration."""
    channels: list[int] = Field(min_length=1)  # REQUIRED
    kernel_sizes: list[int] = Field(min_length=1)  # REQUIRED
    strides: list[int] = Field(min_length=1)  # REQUIRED
    padding: list[int] = Field(min_length=1)  # REQUIRED
    activation: Literal["relu", "gelu", "swish"]  # REQUIRED
    pool: Literal["max", "avg"] | None = None

    @model_validator(mode="after")
    def validate_layer_consistency(self) -> "CNNConfig":
        """Ensure all layer lists have same length."""
        lengths = [len(self.channels), len(self.kernel_sizes),
                   len(self.strides), len(self.padding)]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All CNN layer lists must have same length. "
                f"Got: channels={len(self.channels)}, kernel_sizes={len(self.kernel_sizes)}, "
                f"strides={len(self.strides)}, padding={len(self.padding)}"
            )
        return self

class MLPEncoderConfig(BaseModel):
    """MLP encoder configuration."""
    hidden_sizes: list[int] = Field(min_length=1)  # REQUIRED
    activation: Literal["relu", "gelu", "swish"]  # REQUIRED

class LSTMConfig(BaseModel):
    """LSTM configuration."""
    hidden_size: int = Field(gt=0)  # REQUIRED
    num_layers: int = Field(ge=1)  # REQUIRED
    dropout: float = Field(ge=0.0, lt=1.0)  # REQUIRED
    bidirectional: bool  # REQUIRED

class RecurrentConfig(BaseModel):
    """Recurrent architecture with CNN vision, MLP encoders, and LSTM."""
    vision_encoder: CNNConfig  # REQUIRED
    position_encoder: MLPEncoderConfig  # REQUIRED
    meter_encoder: MLPEncoderConfig  # REQUIRED
    lstm: LSTMConfig  # REQUIRED
    q_head: MLPEncoderConfig  # REQUIRED (reuse MLP config for Q-head)

class ArchitectureConfig(BaseModel):
    """Neural network architecture configuration."""
    type: Literal["feedforward", "recurrent"]  # REQUIRED
    feedforward: FeedforwardConfig | None = None
    recurrent: RecurrentConfig | None = None

    @model_validator(mode="after")
    def validate_architecture_type(self) -> "ArchitectureConfig":
        """Ensure architecture matches type."""
        if self.type == "feedforward" and self.feedforward is None:
            raise ValueError("type='feedforward' but feedforward config missing")
        if self.type == "recurrent" and self.recurrent is None:
            raise ValueError("type='recurrent' but recurrent config missing")
        return self

class OptimizerConfig(BaseModel):
    """Optimizer configuration."""
    type: Literal["adam", "adamw", "rmsprop", "sgd"]  # REQUIRED
    learning_rate: float = Field(gt=0.0)  # REQUIRED
    # Optimizer-specific configs (validated based on type)
    adam: dict | None = None
    adamw: dict | None = None
    sgd: dict | None = None
    rmsprop: dict | None = None

class LossConfig(BaseModel):
    """Loss function configuration."""
    type: Literal["mse", "huber", "smooth_l1"]  # REQUIRED
    huber_delta: float = Field(default=1.0, gt=0.0)  # Optional with default

class ScheduleConfig(BaseModel):
    """Learning rate schedule configuration."""
    type: Literal["constant", "step_decay", "cosine", "exponential"]  # REQUIRED
    step_decay: dict | None = None
    cosine: dict | None = None
    exponential: dict | None = None

class GradientClippingConfig(BaseModel):
    """Gradient clipping configuration."""
    enabled: bool  # REQUIRED
    max_norm: float = Field(gt=0.0)  # REQUIRED if enabled
    norm_type: float = Field(default=2.0, gt=0.0)  # Optional with default

class LearningConfig(BaseModel):
    """Learning configuration (optimizer, loss, schedule)."""
    optimizer: OptimizerConfig  # REQUIRED
    loss: LossConfig  # REQUIRED
    schedule: ScheduleConfig  # REQUIRED
    gradient_clipping: GradientClippingConfig  # REQUIRED
    regularization: dict | None = None  # Optional regularization

class QLearningConfig(BaseModel):
    """Q-learning algorithm configuration."""
    gamma: float = Field(ge=0.0, le=1.0)  # REQUIRED
    target_update_frequency: int = Field(gt=0)  # REQUIRED
    double_dqn: bool  # REQUIRED
    dueling_dqn: bool  # REQUIRED

class ReplayConfig(BaseModel):
    """Experience replay configuration."""
    capacity: int = Field(gt=0)  # REQUIRED
    batch_size: int = Field(gt=0)  # REQUIRED
    sequence_length: int = Field(default=8, gt=0)  # Optional with default
    prioritized: bool  # REQUIRED
    priority_alpha: float = Field(default=0.6, ge=0.0, le=1.0)  # Optional
    priority_beta: float = Field(default=0.4, ge=0.0, le=1.0)  # Optional
    priority_beta_annealing: bool = Field(default=True)  # Optional

class InitializationConfig(BaseModel):
    """Weight initialization configuration."""
    weights: Literal["xavier_uniform", "xavier_normal", "kaiming_uniform",
                     "kaiming_normal", "orthogonal"]  # REQUIRED
    bias: Literal["zeros", "ones", "constant"]  # REQUIRED
    lstm_forget_bias: float = Field(default=1.0)  # Optional with default

class BrainConfig(BaseModel):
    """
    Complete brain configuration (agent architecture + learning).

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.
    """
    version: str  # REQUIRED
    description: str  # REQUIRED (metadata)
    architecture: ArchitectureConfig  # REQUIRED
    learning: LearningConfig  # REQUIRED
    q_learning: QLearningConfig  # REQUIRED
    replay: ReplayConfig  # REQUIRED
    initialization: InitializationConfig  # REQUIRED

def load_brain_config(config_path: Path) -> BrainConfig:
    """Load and validate brain configuration."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return BrainConfig(**data)
```

### Phase 2: Network Factory

**File**: `src/townlet/agent/network_factory.py`

```python
class NetworkFactory:
    """
    Builds neural networks from brain configuration.

    Replaces hardcoded network classes with config-driven construction.
    """

    @staticmethod
    def build_network(brain_config: BrainConfig, obs_dim: int, action_dim: int) -> nn.Module:
        """Build Q-network from brain config."""
        arch_type = brain_config.architecture.type

        if arch_type == "feedforward":
            return NetworkFactory._build_feedforward(
                brain_config.architecture.feedforward,
                obs_dim,
                action_dim,
                brain_config.initialization
            )
        elif arch_type == "recurrent":
            return NetworkFactory._build_recurrent(
                brain_config.architecture.recurrent,
                obs_dim,
                action_dim,
                brain_config.initialization
            )
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

    @staticmethod
    def _build_feedforward(
        config: FeedforwardConfig,
        obs_dim: int,
        action_dim: int,
        init_config: InitializationConfig
    ) -> nn.Module:
        """Build feedforward MLP from config."""
        layers = []
        in_features = obs_dim

        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))

            if config.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            # Activation function from config
            layers.append(NetworkFactory._get_activation(config.activation))

            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))

            in_features = hidden_size

        # Q-value output head
        layers.append(nn.Linear(in_features, action_dim))

        network = nn.Sequential(*layers)

        # Initialize weights from config
        NetworkFactory._initialize_network(network, init_config)

        return network

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function from config."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # Swish = SiLU in PyTorch
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations[activation]

    @staticmethod
    def _initialize_network(network: nn.Module, init_config: InitializationConfig):
        """Initialize network weights from config."""
        for module in network.modules():
            if isinstance(module, nn.Linear):
                # Weight initialization
                if init_config.weights == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init_config.weights == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif init_config.weights == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif init_config.weights == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif init_config.weights == "orthogonal":
                    nn.init.orthogonal_(module.weight)

                # Bias initialization
                if module.bias is not None:
                    if init_config.bias == "zeros":
                        nn.init.zeros_(module.bias)
                    elif init_config.bias == "ones":
                        nn.init.ones_(module.bias)

            elif isinstance(module, nn.LSTM):
                # LSTM forget gate bias initialization
                for name, param in module.named_parameters():
                    if 'bias' in name:
                        # Set forget gate bias to init_config.lstm_forget_bias
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(init_config.lstm_forget_bias)
```

### Phase 3: Brain Compiler

**File**: `src/townlet/agent/brain_compiler.py`

```python
class BrainCompiler:
    """
    Compiles brain configuration into executable components.

    Validates brain config and creates optimizer, loss function, scheduler.
    """

    def __init__(self, brain_config: BrainConfig, network: nn.Module):
        self.brain_config = brain_config
        self.network = network

    def compile(self) -> dict:
        """
        Compile brain into executable training components.

        Returns:
            dict with keys: optimizer, loss_fn, scheduler, replay_buffer
        """
        optimizer = self._build_optimizer()
        loss_fn = self._build_loss_function()
        scheduler = self._build_scheduler(optimizer)
        replay_buffer = self._build_replay_buffer()

        return {
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "scheduler": scheduler,
            "replay_buffer": replay_buffer,
            "metadata": self._compute_metadata()
        }

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        opt_config = self.brain_config.learning.optimizer
        lr = opt_config.learning_rate

        if opt_config.type == "adam":
            params = opt_config.adam or {}
            return optim.Adam(
                self.network.parameters(),
                lr=lr,
                betas=params.get("betas", [0.9, 0.999]),
                eps=params.get("eps", 1e-8),
                weight_decay=params.get("weight_decay", 0.0)
            )
        elif opt_config.type == "adamw":
            params = opt_config.adamw or {}
            return optim.AdamW(
                self.network.parameters(),
                lr=lr,
                betas=params.get("betas", [0.9, 0.999]),
                eps=params.get("eps", 1e-8),
                weight_decay=params.get("weight_decay", 0.01)
            )
        elif opt_config.type == "sgd":
            params = opt_config.sgd or {}
            return optim.SGD(
                self.network.parameters(),
                lr=lr,
                momentum=params.get("momentum", 0.9),
                nesterov=params.get("nesterov", False),
                weight_decay=params.get("weight_decay", 0.0)
            )
        elif opt_config.type == "rmsprop":
            params = opt_config.rmsprop or {}
            return optim.RMSprop(
                self.network.parameters(),
                lr=lr,
                alpha=params.get("alpha", 0.99),
                eps=params.get("eps", 1e-8),
                weight_decay=params.get("weight_decay", 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config.type}")

    def _build_loss_function(self) -> nn.Module:
        """Build loss function from config."""
        loss_config = self.brain_config.learning.loss

        if loss_config.type == "mse":
            return nn.MSELoss()
        elif loss_config.type == "huber":
            return nn.HuberLoss(delta=loss_config.huber_delta)
        elif loss_config.type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_config.type}")

    def _build_scheduler(self, optimizer: optim.Optimizer):
        """Build learning rate scheduler from config."""
        sched_config = self.brain_config.learning.schedule

        if sched_config.type == "constant":
            return None  # No scheduler
        elif sched_config.type == "step_decay":
            params = sched_config.step_decay
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=params["step_size"],
                gamma=params["gamma"]
            )
        elif sched_config.type == "cosine":
            params = sched_config.cosine
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params["T_max"],
                eta_min=params["eta_min"]
            )
        elif sched_config.type == "exponential":
            params = sched_config.exponential
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=params["gamma"]
            )
        else:
            raise ValueError(f"Unknown schedule type: {sched_config.type}")

    def _build_replay_buffer(self):
        """Build experience replay buffer from config."""
        replay_config = self.brain_config.replay

        if replay_config.prioritized:
            return PrioritizedReplayBuffer(
                capacity=replay_config.capacity,
                alpha=replay_config.priority_alpha,
                beta=replay_config.priority_beta,
                beta_annealing=replay_config.priority_beta_annealing
            )
        else:
            return ReplayBuffer(capacity=replay_config.capacity)

    def _compute_metadata(self) -> dict:
        """Compute brain metadata."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "architecture_type": self.brain_config.architecture.type,
            "optimizer_type": self.brain_config.learning.optimizer.type,
            "loss_type": self.brain_config.learning.loss.type,
        }
```

### Phase 4: Update Runner to Use Brain Config

**File**: `src/townlet/demo/runner.py` (or future `src/townlet/training/runner.py`)

```python
# BEFORE: Hardcoded network creation
if network_type == "simple":
    q_network = SimpleQNetwork(obs_dim, action_dim).to(device)
else:
    q_network = RecurrentSpatialQNetwork(...).to(device)

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# AFTER: Brain config-driven
brain_config_path = config_pack_path / "brain.yaml"
brain_config = load_brain_config(brain_config_path)

# Build network from config
q_network = NetworkFactory.build_network(
    brain_config,
    obs_dim=obs_dim,
    action_dim=action_dim
).to(device)

# Compile brain (optimizer, loss, scheduler, replay buffer)
brain_compiler = BrainCompiler(brain_config, q_network)
compiled_brain = brain_compiler.compile()

optimizer = compiled_brain["optimizer"]
loss_fn = compiled_brain["loss_fn"]
scheduler = compiled_brain["scheduler"]
replay_buffer = compiled_brain["replay_buffer"]

logger.info(f"🧠 Brain compiled: {compiled_brain['metadata']}")
logger.info(f"   Total params: {compiled_brain['metadata']['total_params']:,}")
logger.info(f"   Architecture: {compiled_brain['metadata']['architecture_type']}")
logger.info(f"   Optimizer: {compiled_brain['metadata']['optimizer_type']}")
```

## Integration with Universe Compiler

Brain compilation happens **after** universe compilation:

```python
# Stage 1: Compile Universe (TASK-003)
universe_compiler = UniverseCompiler("configs/L0_minimal")
universe = universe_compiler.compile()

# Stage 2: Compile Brain (TASK-005)
brain_config_path = Path("configs/L0_minimal/brain.yaml")
brain_config = load_brain_config(brain_config_path)

# Observation dim comes from compiled universe
obs_dim = universe["metadata"].observation_dim
action_dim = universe["metadata"].action_dim

# Build network
q_network = NetworkFactory.build_network(brain_config, obs_dim, action_dim)

# Compile brain
brain_compiler = BrainCompiler(brain_config, q_network)
compiled_brain = brain_compiler.compile()

# Ready to train!
```

## Benefits

1. **Architecture Experimentation**: Test different networks without code changes
2. **Optimizer Comparison**: A/B test Adam vs AdamW vs RMSprop
3. **Reproducibility**: Brain config = complete specification of agent architecture
4. **Curriculum Flexibility**: Different brains for different curriculum levels
5. **Pedagogical Value**: Students learn network architecture by editing YAML
6. **Domain-Specific Architectures**: CNN for vision, LSTM for POMDP, MLP for full obs
7. **Hyperparameter Search**: Easy to generate configs for grid/random search
8. **Version Control**: Track brain architecture changes in git

## Success Criteria

- [ ] `brain.yaml` schema defined with Pydantic DTOs
- [ ] All config packs have `brain.yaml` (L0, L0.5, L1, L2, L3)
- [ ] NetworkFactory builds networks from config (feedforward + recurrent)
- [ ] BrainCompiler creates optimizer, loss, scheduler from config
- [ ] Runner uses brain config instead of hardcoded networks
- [ ] Can switch between architectures by editing YAML (no code changes)
- [ ] Brain compilation errors caught at load time with clear messages
- [ ] No-defaults enforcement: All behavioral parameters required
- [ ] Config templates created showing all architecture options
- [ ] Documentation updated with brain config examples

## Estimated Effort

- **Phase 1** (brain config schema): 4-6 hours
- **Phase 2** (network factory): 6-8 hours
- **Phase 3** (brain compiler): 4-6 hours
- **Phase 4** (runner integration): 3-4 hours
- **Phase 5** (create brain.yaml for all config packs): 2-3 hours
- **Phase 6** (testing + documentation): 3-4 hours
- **Total**: 22-31 hours

## Dependencies

- **TASK-002A**: Spatial substrates (provides substrate metadata for obs_dim computation)
- **TASK-001**: DTO schemas (provides Pydantic patterns)
- **TASK-003**: Universe compiler (provides obs_dim, action_dim)

**Recommended order**: TASK-002A → TASK-001 → TASK-003 → **TASK-005**

### Critical Dependency: Substrate Metadata from TASK-002A

**From Research** (`docs/research/RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md`):

The brain compilation process needs substrate metadata to compute `obs_dim` correctly:

**Required substrate properties**:
1. `position_encoding_dim` - How many dimensions for position encoding?
   - 2D one-hot (8×8): 64 dims
   - 2D coordinates: 2 dims
   - 3D coordinates (8×8×3): 3 dims
   - Aspatial: 0 dims

2. `position_encoding` strategy - Which encoding method?
   - `"onehot"` - One-hot grid encoding (small 2D grids only)
   - `"coords"` - Normalized coordinate encoding (3D, large grids)
   - `"fourier"` - Sinusoidal position encoding (continuous spaces)
   - `"none"` - No position (aspatial substrates)

**obs_dim computation**:
```python
# Brain compiler must query substrate config
obs_dim = (
    substrate.position_encoding_dim +  # Variable by substrate type
    num_meters +                       # Variable by bars.yaml
    num_affordance_types + 1 +         # Fixed vocabulary + "none"
    4                                  # Temporal extras (fixed)
)
```

**Example obs_dim variations**:
| Substrate | Position Enc | Meters | Affordances | Temporal | **Total** |
|-----------|--------------|--------|-------------|----------|-----------|
| 2D (8×8, onehot) | 64 | 8 | 15 | 4 | **91** |
| 2D (8×8, coords) | 2 | 8 | 15 | 4 | **29** |
| 3D (8×8×3, coords) | 3 | 8 | 15 | 4 | **30** |
| Aspatial | 0 | 8 | 15 | 4 | **27** |

**Implication for brain.yaml**:

Brain config CANNOT hardcode `obs_dim` - it must be computed from universe config at compile time.

**Option 1: Auto-compute (recommended)**:
```yaml
# brain.yaml
network:
  architecture: "simple_q"
  encoding_aware: true  # Network handles coordinate OR one-hot position encoding
  # obs_dim: COMPUTED from substrate + meters + affordances
  # action_dim: COMPUTED from actions.yaml
  hidden_dims: [256, 128]
  activation: "relu"
```

**Option 2: Validate (strict mode)**:
```yaml
# brain.yaml
network:
  architecture: "simple_q"
  expected_obs_dim: 91  # Validated against computed obs_dim
  action_dim: 5
  hidden_dims: [256, 128]
```

**Compilation error if mismatch**:
```
❌ BRAIN COMPILATION FAILED
Expected obs_dim=91 but computed obs_dim=29 from universe config.

Universe configuration:
  - Substrate: 2D (8×8) with coordinate encoding (2 dims)
  - Meters: 8
  - Affordances: 15
  - Temporal: 4
  - Computed obs_dim: 2 + 8 + 15 + 4 = 29

Your brain.yaml specifies expected_obs_dim=91, which assumes one-hot encoding (64 dims).

Fix: Either remove expected_obs_dim (auto-compute) or update to:
  expected_obs_dim: 29
```

**Benefits**:
1. ✅ **Position encoding agnostic**: Network doesn't care if position is one-hot or coordinates
2. ✅ **Substrate agnostic**: Same brain.yaml works for 2D, 3D, hex, aspatial
3. ✅ **Meter count agnostic**: Same brain.yaml works for 4-meter or 12-meter universes
4. ✅ **Clear compilation errors**: Mismatch caught at load time, not runtime

## Design Principles

**Separation of Concerns**:
- **Universe config** (bars, actions, affordances): Defines the world
- **Brain config** (architecture, optimizer, learning): Defines the agent
- **Training config** (epsilon, curriculum, exploration): Defines the learning process

**No-Defaults Enforcement**:
- ALL architectural choices must be explicit (hidden layer sizes, activation functions, etc.)
- NO magic numbers hidden in code
- Config file is complete specification of agent brain

**Conceptual Agnosticism**:
- Brain compiler doesn't assume "reasonable" architectures
- Allows experimental configurations (1000-layer network, weird activations)
- Validates structure (correct types), not semantics (sensible choices)

**Compilation Errors Over Runtime Errors**:
- Invalid brain config → clear error at load time
- Missing required fields → helpful error message with example
- Type mismatches caught by Pydantic

**Mental Model**: "The brain is compiled, not interpreted."

---

## Optional Future Extension: RND Architecture Configuration

**Added**: 2025-11-04 (from research findings - Gap 3)
**Status**: DEFERRED (Low Priority)
**Related Research**: `docs/research/RESEARCH-ENVIRONMENT-CONSTANTS-EXPOSURE.md` (Gap 3)
**Effort**: +1-2 hours if implemented

### Problem

Current RND (Random Network Distillation) exploration architecture is hardcoded:

```python
# src/townlet/exploration/rnd.py (current)
class RNDExploration:
    def __init__(self, obs_dim: int, embed_dim: int = 128):
        # ❌ Architecture hardcoded
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, 256),  # Hidden size hardcoded
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        self.predictor_network = nn.Sequential(
            nn.Linear(obs_dim, 256),  # Same hardcoded size
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
```

**Limitation**: Cannot experiment with different RND architectures without code changes.

### Solution (if requested)

Add optional `rnd` section to `brain.yaml`:

```yaml
# brain.yaml (optional RND architecture)
exploration:
  type: "rnd"  # or "epsilon_greedy", "ucb", etc.

  rnd:
    hidden_dims: [256, 256]  # Configurable hidden layers
    activation: "relu"        # or "gelu", "silu", "tanh"
    embed_dim: 128           # Latent embedding dimension
    learning_rate: 0.0001    # RND predictor learning rate
```

**Benefits**:
- Researchers can experiment with different RND architectures
- Test impact of RND capacity on exploration
- A/B test RND vs UCB vs epsilon-greedy via config change

**Priority Justification: Why DEFERRED?**

RND architecture is an **implementation detail**, not a **learning concept**:
- Students learn "intrinsic motivation" concept, not "how many hidden layers RND should have"
- Hardcoded RND works fine for pedagogical purposes
- Low pedagogical value compared to other UAC features

**Recommendation**: Implement only if researchers specifically request RND architecture experimentation. Otherwise, keep hardcoded RND as "good enough" default.

**Implementation Notes** (if needed):
1. Add `RNDConfig` Pydantic DTO to `brain_config.py`
2. Update `RNDExploration.__init__()` to accept config
3. Validate `embed_dim` matches network architecture expectations
4. Add example to `brain.yaml` templates

**Estimated Effort**: 1-2 hours (low complexity, but low priority)
