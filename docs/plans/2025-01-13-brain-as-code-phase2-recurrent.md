# Brain As Code - Phase 2: Recurrent Networks & Learning Rate Schedulers

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add recurrent network architecture configuration (CNN + LSTM) and learning rate schedulers to brain.yaml, enabling POMDP experiments.

**Architecture:** Extend brain_config.py with RecurrentConfig for RecurrentSpatialQNetwork. Add ScheduleConfig for learning rate schedules. Update NetworkFactory to build recurrent networks. Create brain.yaml for L1, L2, L3.

**Tech Stack:** Pydantic (validation), PyTorch (LSTM, LR schedulers), YAML (config)

**Prerequisites:** Phase 1 complete (feedforward networks working)

**Scope:**
- ✅ RecurrentConfig (CNN vision encoder, position/meter/affordance encoders, LSTM, Q-head)
- ✅ Learning rate schedulers (StepLR, CosineAnnealingLR, ExponentialLR)
- ✅ NetworkFactory.build_recurrent()
- ✅ Update VectorizedPopulation to handle recurrent networks
- ✅ Create brain.yaml for L1, L2, L3
- ❌ Dueling DQN (Phase 3)
- ❌ Prioritized Experience Replay (Phase 3)

---

## Task 1: Add RecurrentConfig DTOs

**Files:**
- Modify: `src/townlet/agent/brain_config.py`
- Test: `tests/test_townlet/unit/agent/test_brain_config.py`

**Step 1: Write failing test for RecurrentConfig**

Add to `tests/test_townlet/unit/agent/test_brain_config.py`:

```python
from townlet.agent.brain_config import (
    RecurrentConfig,
    CNNEncoderConfig,
    MLPEncoderConfig,
    LSTMConfig,
)


def test_cnn_encoder_config_valid():
    """CNNEncoderConfig accepts valid CNN parameters."""
    config = CNNEncoderConfig(
        channels=[16, 32],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        padding=[1, 1],
        activation="relu",
    )
    assert config.channels == [16, 32]
    assert config.kernel_sizes == [3, 3]


def test_cnn_encoder_config_rejects_mismatched_lengths():
    """CNNEncoderConfig requires all lists to have same length."""
    with pytest.raises(ValidationError) as exc_info:
        CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3],  # Wrong length!
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        )
    assert "same length" in str(exc_info.value).lower()


def test_mlp_encoder_config_valid():
    """MLPEncoderConfig accepts valid MLP parameters."""
    config = MLPEncoderConfig(
        hidden_sizes=[32],
        activation="relu",
    )
    assert config.hidden_sizes == [32]


def test_lstm_config_valid():
    """LSTMConfig accepts valid LSTM parameters."""
    config = LSTMConfig(
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    )
    assert config.hidden_size == 256
    assert config.num_layers == 1


def test_lstm_config_rejects_zero_hidden_size():
    """LSTMConfig rejects hidden_size=0."""
    with pytest.raises(ValidationError) as exc_info:
        LSTMConfig(
            hidden_size=0,
            num_layers=1,
            dropout=0.0,
        )
    assert "hidden_size" in str(exc_info.value)


def test_recurrent_config_valid():
    """RecurrentConfig accepts complete recurrent architecture."""
    config = RecurrentConfig(
        vision_encoder=CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        ),
        position_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        meter_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        affordance_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        lstm=LSTMConfig(
            hidden_size=256,
            num_layers=1,
            dropout=0.0,
        ),
        q_head=MLPEncoderConfig(
            hidden_sizes=[128],
            activation="relu",
        ),
    )
    assert config.lstm.hidden_size == 256
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_cnn_encoder_config_valid -v
```

Expected: `ImportError: cannot import name 'CNNEncoderConfig'`

**Step 3: Implement CNNEncoderConfig, MLPEncoderConfig, LSTMConfig, RecurrentConfig**

Add to `src/townlet/agent/brain_config.py`:

```python
class CNNEncoderConfig(BaseModel):
    """CNN encoder configuration for vision processing.

    Example:
        >>> vision = CNNEncoderConfig(
        ...     channels=[16, 32],
        ...     kernel_sizes=[3, 3],
        ...     strides=[1, 1],
        ...     padding=[1, 1],
        ...     activation="relu",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    channels: list[int] = Field(
        min_length=1,
        description="Channel progression for CNN layers (e.g., [16, 32])"
    )
    kernel_sizes: list[int] = Field(
        min_length=1,
        description="Kernel size for each CNN layer"
    )
    strides: list[int] = Field(
        min_length=1,
        description="Stride for each CNN layer"
    )
    padding: list[int] = Field(
        min_length=1,
        description="Padding for each CNN layer"
    )
    activation: Literal["relu", "gelu", "swish"] = Field(
        description="Activation function for CNN"
    )

    @model_validator(mode="after")
    def validate_layer_consistency(self) -> "CNNEncoderConfig":
        """Ensure all layer lists have same length."""
        lengths = {
            "channels": len(self.channels),
            "kernel_sizes": len(self.kernel_sizes),
            "strides": len(self.strides),
            "padding": len(self.padding),
        }
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"All CNN layer lists must have same length. Got: {lengths}"
            )
        return self


class MLPEncoderConfig(BaseModel):
    """MLP encoder configuration.

    Used for position, meter, affordance encoders, and Q-head.

    Example:
        >>> position = MLPEncoderConfig(
        ...     hidden_sizes=[32],
        ...     activation="relu",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    hidden_sizes: list[int] = Field(
        min_length=1,
        description="Hidden layer sizes (e.g., [32] for single layer)"
    )
    activation: Literal["relu", "gelu", "swish"] = Field(
        description="Activation function"
    )


class LSTMConfig(BaseModel):
    """LSTM configuration for recurrent networks.

    Example:
        >>> lstm = LSTMConfig(
        ...     hidden_size=256,
        ...     num_layers=1,
        ...     dropout=0.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(
        gt=0,
        description="LSTM hidden state dimension"
    )
    num_layers: int = Field(
        ge=1,
        le=4,
        description="Number of stacked LSTM layers (1-4)"
    )
    dropout: float = Field(
        ge=0.0,
        lt=1.0,
        description="Dropout between LSTM layers (0.0 = no dropout)"
    )


class RecurrentConfig(BaseModel):
    """Recurrent architecture configuration for POMDP.

    Architecture: CNN vision → Position MLP → Meter MLP → Affordance MLP → LSTM → Q-head

    Example:
        >>> config = RecurrentConfig(
        ...     vision_encoder=CNNEncoderConfig(...),
        ...     position_encoder=MLPEncoderConfig(...),
        ...     meter_encoder=MLPEncoderConfig(...),
        ...     affordance_encoder=MLPEncoderConfig(...),
        ...     lstm=LSTMConfig(...),
        ...     q_head=MLPEncoderConfig(...),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    vision_encoder: CNNEncoderConfig = Field(
        description="CNN encoder for local vision window"
    )
    position_encoder: MLPEncoderConfig = Field(
        description="MLP encoder for position (x, y, z)"
    )
    meter_encoder: MLPEncoderConfig = Field(
        description="MLP encoder for meter values"
    )
    affordance_encoder: MLPEncoderConfig = Field(
        description="MLP encoder for affordance types"
    )
    lstm: LSTMConfig = Field(
        description="LSTM for temporal memory"
    )
    q_head: MLPEncoderConfig = Field(
        description="MLP Q-value head"
    )
```

**Step 4: Update ArchitectureConfig to support recurrent type**

Modify `ArchitectureConfig` in `src/townlet/agent/brain_config.py`:

```python
class ArchitectureConfig(BaseModel):
    """Neural network architecture configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["feedforward", "recurrent"] = Field(  # Add "recurrent"
        description="Architecture type"
    )

    # Architecture-specific configs
    feedforward: FeedforwardConfig | None = Field(
        default=None,
        description="Feedforward MLP config (required when type=feedforward)"
    )
    recurrent: RecurrentConfig | None = Field(  # NEW
        default=None,
        description="Recurrent LSTM config (required when type=recurrent)"
    )

    @model_validator(mode="after")
    def validate_architecture_match(self) -> "ArchitectureConfig":
        """Ensure architecture config matches type."""
        if self.type == "feedforward" and self.feedforward is None:
            raise ValueError("type='feedforward' requires feedforward config")
        if self.type == "recurrent" and self.recurrent is None:
            raise ValueError("type='recurrent' requires recurrent config")
        return self
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py -k "recurrent or cnn or mlp or lstm" -v
```

Expected: All new tests PASS

**Step 6: Commit**

```bash
git add src/townlet/agent/brain_config.py tests/test_townlet/unit/agent/test_brain_config.py
git commit -m "feat(brain): add RecurrentConfig DTOs for LSTM networks

- Add CNNEncoderConfig (vision encoder)
- Add MLPEncoderConfig (position/meter/affordance/q_head)
- Add LSTMConfig (recurrent memory)
- Add RecurrentConfig (complete architecture)
- Update ArchitectureConfig to support type=recurrent
- Validate CNN layer list lengths match
- Add comprehensive unit tests

Part of TASK-005 Phase 2 (1/10)"
```

---

## Task 2: Add Learning Rate Schedule Configuration

**Files:**
- Modify: `src/townlet/agent/brain_config.py`
- Test: `tests/test_townlet/unit/agent/test_brain_config.py`

**Step 1: Write failing test for ScheduleConfig**

Add to `tests/test_townlet/unit/agent/test_brain_config.py`:

```python
from townlet.agent.brain_config import ScheduleConfig


def test_schedule_config_constant():
    """ScheduleConfig accepts constant (no schedule)."""
    config = ScheduleConfig(type="constant")
    assert config.type == "constant"


def test_schedule_config_step_decay():
    """ScheduleConfig accepts StepLR parameters."""
    config = ScheduleConfig(
        type="step_decay",
        step_size=1000,
        gamma=0.1,
    )
    assert config.type == "step_decay"
    assert config.step_size == 1000
    assert config.gamma == 0.1


def test_schedule_config_cosine():
    """ScheduleConfig accepts CosineAnnealingLR parameters."""
    config = ScheduleConfig(
        type="cosine",
        t_max=5000,
        eta_min=0.00001,
    )
    assert config.type == "cosine"
    assert config.t_max == 5000


def test_schedule_config_exponential():
    """ScheduleConfig accepts ExponentialLR parameters."""
    config = ScheduleConfig(
        type="exponential",
        gamma=0.9999,
    )
    assert config.type == "exponential"
    assert config.gamma == 0.9999


def test_schedule_config_requires_params_for_step_decay():
    """ScheduleConfig validates step_decay requires step_size and gamma."""
    # Valid step_decay needs both params (will validate in model_validator)
    config = ScheduleConfig(
        type="step_decay",
        step_size=100,
        gamma=0.1,
    )
    assert config.step_size is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_schedule_config_constant -v
```

Expected: `ImportError: cannot import name 'ScheduleConfig'`

**Step 3: Implement ScheduleConfig**

Add to `src/townlet/agent/brain_config.py`:

```python
class ScheduleConfig(BaseModel):
    """Learning rate schedule configuration.

    Example:
        >>> constant = ScheduleConfig(type="constant")
        >>> step = ScheduleConfig(type="step_decay", step_size=1000, gamma=0.1)
        >>> cosine = ScheduleConfig(type="cosine", t_max=5000, eta_min=0.00001)
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["constant", "step_decay", "cosine", "exponential"] = Field(
        description="Learning rate schedule type"
    )

    # StepLR parameters
    step_size: int | None = Field(
        default=None,
        gt=0,
        description="Step size for StepLR (required for type=step_decay)"
    )
    gamma: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Multiplicative factor for StepLR or ExponentialLR"
    )

    # CosineAnnealingLR parameters
    t_max: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of iterations for cosine schedule"
    )
    eta_min: float | None = Field(
        default=None,
        ge=0.0,
        description="Minimum learning rate for cosine schedule"
    )

    @model_validator(mode="after")
    def validate_schedule_params(self) -> "ScheduleConfig":
        """Ensure required parameters present for each schedule type."""
        if self.type == "step_decay":
            if self.step_size is None or self.gamma is None:
                raise ValueError("type='step_decay' requires step_size and gamma")
        elif self.type == "cosine":
            if self.t_max is None or self.eta_min is None:
                raise ValueError("type='cosine' requires t_max and eta_min")
        elif self.type == "exponential":
            if self.gamma is None:
                raise ValueError("type='exponential' requires gamma")
        return self
```

**Step 4: Add schedule field to OptimizerConfig**

Modify `OptimizerConfig` in `src/townlet/agent/brain_config.py`:

```python
class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["adam", "adamw", "sgd", "rmsprop"]
    learning_rate: float = Field(gt=0.0)

    # ... existing optimizer params ...

    # Learning rate schedule (NEW)
    schedule: ScheduleConfig = Field(
        description="Learning rate schedule"
    )
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py -k "schedule" -v
```

Expected: All schedule tests PASS

**Step 6: Commit**

```bash
git add src/townlet/agent/brain_config.py tests/test_townlet/unit/agent/test_brain_config.py
git commit -m "feat(brain): add ScheduleConfig for learning rate schedules

- Add ScheduleConfig (constant, step_decay, cosine, exponential)
- Validate schedule-specific parameters
- Add schedule field to OptimizerConfig
- Support StepLR, CosineAnnealingLR, ExponentialLR
- Add unit tests for all schedule types

Part of TASK-005 Phase 2 (2/10)"
```

---

## Task 3: Update OptimizerFactory to Support Schedulers

**Files:**
- Modify: `src/townlet/agent/optimizer_factory.py`
- Test: `tests/test_townlet/unit/agent/test_optimizer_factory.py`

**Step 1: Write failing test for scheduler creation**

Add to `tests/test_townlet/unit/agent/test_optimizer_factory.py`:

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from townlet.agent.brain_config import ScheduleConfig


def test_build_with_constant_schedule():
    """OptimizerFactory.build returns (optimizer, None) for constant schedule."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="constant"),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, optim.Adam)
    assert scheduler is None


def test_build_with_step_decay_schedule():
    """OptimizerFactory.build returns (optimizer, StepLR) for step_decay."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="step_decay", step_size=100, gamma=0.1),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, optim.Adam)
    assert isinstance(scheduler, StepLR)


def test_build_with_cosine_schedule():
    """OptimizerFactory.build returns (optimizer, CosineAnnealingLR) for cosine."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="cosine", t_max=1000, eta_min=0.00001),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(scheduler, CosineAnnealingLR)


def test_build_with_exponential_schedule():
    """OptimizerFactory.build returns (optimizer, ExponentialLR) for exponential."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
        schedule=ScheduleConfig(type="exponential", gamma=0.9999),
    )

    optimizer, scheduler = OptimizerFactory.build(config, network.parameters())

    assert isinstance(scheduler, ExponentialLR)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_optimizer_factory.py::test_build_with_constant_schedule -v
```

Expected: Tests fail (OptimizerFactory.build returns single value, not tuple)

**Step 3: Update OptimizerFactory.build to return (optimizer, scheduler) tuple**

Modify `src/townlet/agent/optimizer_factory.py`:

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from townlet.agent.brain_config import OptimizerConfig


class OptimizerFactory:
    """Factory for building PyTorch optimizers and LR schedulers from configuration."""

    @staticmethod
    def build(config: OptimizerConfig, parameters) -> tuple[optim.Optimizer, object | None]:
        """Build optimizer and optional scheduler from configuration.

        Args:
            config: Optimizer configuration (includes schedule)
            parameters: Network parameters (from network.parameters())

        Returns:
            Tuple of (optimizer, scheduler or None)

        Example:
            >>> config = OptimizerConfig(
            ...     type="adam",
            ...     learning_rate=0.001,
            ...     adam_beta1=0.9,
            ...     adam_beta2=0.999,
            ...     adam_eps=1e-8,
            ...     weight_decay=0.0,
            ...     schedule=ScheduleConfig(type="step_decay", step_size=100, gamma=0.1),
            ... )
            >>> optimizer, scheduler = OptimizerFactory.build(config, network.parameters())
        """
        # Build optimizer (unchanged logic)
        if config.type == "adam":
            optimizer = optim.Adam(
                parameters,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
        elif config.type == "adamw":
            optimizer = optim.AdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
        elif config.type == "sgd":
            optimizer = optim.SGD(
                parameters,
                lr=config.learning_rate,
                momentum=config.sgd_momentum,
                nesterov=config.sgd_nesterov,
                weight_decay=config.weight_decay,
            )
        elif config.type == "rmsprop":
            optimizer = optim.RMSprop(
                parameters,
                lr=config.learning_rate,
                alpha=config.rmsprop_alpha,
                eps=config.rmsprop_eps,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.type}")

        # Build scheduler (NEW)
        scheduler = OptimizerFactory._build_scheduler(config.schedule, optimizer)

        return optimizer, scheduler

    @staticmethod
    def _build_scheduler(schedule_config, optimizer):
        """Build learning rate scheduler from configuration.

        Args:
            schedule_config: ScheduleConfig
            optimizer: PyTorch optimizer

        Returns:
            Scheduler or None (for constant)
        """
        if schedule_config.type == "constant":
            return None
        elif schedule_config.type == "step_decay":
            return StepLR(
                optimizer,
                step_size=schedule_config.step_size,
                gamma=schedule_config.gamma,
            )
        elif schedule_config.type == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=schedule_config.t_max,
                eta_min=schedule_config.eta_min,
            )
        elif schedule_config.type == "exponential":
            return ExponentialLR(
                optimizer,
                gamma=schedule_config.gamma,
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_config.type}")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_optimizer_factory.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/optimizer_factory.py tests/test_townlet/unit/agent/test_optimizer_factory.py
git commit -m "feat(brain): add scheduler support to OptimizerFactory

- Update OptimizerFactory.build to return (optimizer, scheduler) tuple
- Add _build_scheduler() method
- Support StepLR, CosineAnnealingLR, ExponentialLR
- Return None for constant schedule
- Update all tests to expect tuple return

Part of TASK-005 Phase 2 (3/10)"
```

---

## Task 4: Add NetworkFactory.build_recurrent

**Files:**
- Modify: `src/townlet/agent/network_factory.py`
- Test: `tests/test_townlet/unit/agent/test_network_factory.py`

**Step 1: Write failing test for build_recurrent**

Add to `tests/test_townlet/unit/agent/test_network_factory.py`:

```python
from townlet.agent.brain_config import (
    RecurrentConfig,
    CNNEncoderConfig,
    MLPEncoderConfig,
    LSTMConfig,
)


def test_build_recurrent_basic():
    """NetworkFactory builds RecurrentSpatialQNetwork from RecurrentConfig."""
    config = RecurrentConfig(
        vision_encoder=CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        ),
        position_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        meter_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        affordance_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        lstm=LSTMConfig(
            hidden_size=256,
            num_layers=1,
            dropout=0.0,
        ),
        q_head=MLPEncoderConfig(
            hidden_sizes=[128],
            activation="relu",
        ),
    )

    network = NetworkFactory.build_recurrent(
        config=config,
        action_dim=8,
        window_size=5,
        position_dim=2,
        num_meters=8,
        num_affordance_types=14,
    )

    # Test forward pass with dummy observation
    batch_size = 4
    obs_dim = (5 * 5) + 2 + 8 + 15  # grid + position + meters + affordances
    obs = torch.randn(batch_size, obs_dim)

    q_values, hidden = network(obs)

    assert q_values.shape == (batch_size, 8)
    assert isinstance(hidden, tuple)
    assert hidden[0].shape == (1, batch_size, 256)  # h
    assert hidden[1].shape == (1, batch_size, 256)  # c


def test_build_recurrent_parameter_count():
    """NetworkFactory creates recurrent network with expected parameter count."""
    config = RecurrentConfig(
        vision_encoder=CNNEncoderConfig(
            channels=[16, 32],
            kernel_sizes=[3, 3],
            strides=[1, 1],
            padding=[1, 1],
            activation="relu",
        ),
        position_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        meter_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        affordance_encoder=MLPEncoderConfig(
            hidden_sizes=[32],
            activation="relu",
        ),
        lstm=LSTMConfig(
            hidden_size=256,
            num_layers=1,
            dropout=0.0,
        ),
        q_head=MLPEncoderConfig(
            hidden_sizes=[128],
            activation="relu",
        ),
    )

    network = NetworkFactory.build_recurrent(
        config=config,
        action_dim=8,
        window_size=5,
        position_dim=2,
        num_meters=8,
        num_affordance_types=14,
    )

    total_params = sum(p.numel() for p in network.parameters())
    # Recurrent networks are larger (~500K-700K params)
    assert 400_000 < total_params < 1_000_000
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_network_factory.py::test_build_recurrent_basic -v
```

Expected: `AttributeError: 'NetworkFactory' has no attribute 'build_recurrent'`

**Step 3: Implement NetworkFactory.build_recurrent**

Add to `src/townlet/agent/network_factory.py`:

```python
from townlet.agent.brain_config import RecurrentConfig
from townlet.agent.networks import RecurrentSpatialQNetwork


class NetworkFactory:
    """Factory for building Q-networks from declarative configuration."""

    # ... existing build_feedforward() ...

    @staticmethod
    def build_recurrent(
        config: RecurrentConfig,
        action_dim: int,
        window_size: int,
        position_dim: int,
        num_meters: int,
        num_affordance_types: int,
    ) -> RecurrentSpatialQNetwork:
        """Build recurrent LSTM Q-network from configuration.

        Args:
            config: Recurrent architecture configuration
            action_dim: Number of actions
            window_size: Vision window size (5 for 5×5)
            position_dim: Position dimensionality (2 for Grid2D, 3 for Grid3D, 0 for Aspatial)
            num_meters: Number of meter values
            num_affordance_types: Number of affordance types

        Returns:
            RecurrentSpatialQNetwork

        Note:
            This builds a RecurrentSpatialQNetwork but with configurable dimensions
            instead of hardcoded values. The network structure matches the original
            but dimensions come from config.

        Example:
            >>> config = RecurrentConfig(...)
            >>> network = NetworkFactory.build_recurrent(
            ...     config=config,
            ...     action_dim=8,
            ...     window_size=5,
            ...     position_dim=2,
            ...     num_meters=8,
            ...     num_affordance_types=14,
            ... )
        """
        # Extract dimensions from config
        lstm_hidden_size = config.lstm.hidden_size

        # Create RecurrentSpatialQNetwork with config-driven dimensions
        # We still use the existing RecurrentSpatialQNetwork class but pass
        # hidden_dim from config instead of hardcoding 256
        network = RecurrentSpatialQNetwork(
            action_dim=action_dim,
            window_size=window_size,
            position_dim=position_dim,
            num_meters=num_meters,
            num_affordance_types=num_affordance_types,
            enable_temporal_features=False,  # Will be determined by env
            hidden_dim=lstm_hidden_size,  # From config instead of hardcoded!
        )

        return network
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_network_factory.py::test_build_recurrent_basic -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/network_factory.py tests/test_townlet/unit/agent/test_network_factory.py
git commit -m "feat(brain): add NetworkFactory.build_recurrent for LSTM networks

- Add build_recurrent() method
- Build RecurrentSpatialQNetwork from RecurrentConfig
- Pass lstm.hidden_size from config instead of hardcoded 256
- Add unit tests for recurrent network building
- Validate output shapes and parameter counts

Part of TASK-005 Phase 2 (4/10)"
```

---

## Task 5: Update VectorizedPopulation to Support Recurrent Networks

**Files:**
- Modify: `src/townlet/population/vectorized.py` (lines 119-169)

**Step 1: Update network building to support recurrent type**

Modify `src/townlet/population/vectorized.py` network instantiation:

```python
# Build Q-network from brain config (replace existing lines 119-141)
if self.brain_config.architecture.type == "feedforward":
    self.q_network = NetworkFactory.build_feedforward(
        config=self.brain_config.architecture.feedforward,
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(device)
elif self.brain_config.architecture.type == "recurrent":
    self.q_network = NetworkFactory.build_recurrent(
        config=self.brain_config.architecture.recurrent,
        action_dim=action_dim,
        window_size=vision_window_size,
        position_dim=env.substrate.position_dim,
        num_meters=env.meter_count,
        num_affordance_types=env.num_affordance_types,
    ).to(device)
    self.is_recurrent = True  # Enable recurrent mode
else:
    raise ValueError(
        f"Unsupported architecture type: {self.brain_config.architecture.type}. "
        f"Supported: feedforward, recurrent"
    )

# Build target network (same architecture)
if self.brain_config.architecture.type == "feedforward":
    self.target_network = NetworkFactory.build_feedforward(
        config=self.brain_config.architecture.feedforward,
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(device)
elif self.brain_config.architecture.type == "recurrent":
    self.target_network = NetworkFactory.build_recurrent(
        config=self.brain_config.architecture.recurrent,
        action_dim=action_dim,
        window_size=vision_window_size,
        position_dim=env.substrate.position_dim,
        num_meters=env.meter_count,
        num_affordance_types=env.num_affordance_types,
    ).to(device)
else:
    raise ValueError(f"Unsupported architecture type: {self.brain_config.architecture.type}")

# Initialize target network
self.target_network.load_state_dict(self.q_network.state_dict())
self.target_network.eval()

# Store is_recurrent flag from architecture type
self.is_recurrent = (self.brain_config.architecture.type == "recurrent")
```

**Step 2: Update optimizer creation to handle scheduler**

Modify optimizer creation (around line 173):

```python
# Build optimizer and scheduler from brain config
self.optimizer, self.scheduler = OptimizerFactory.build(
    config=self.brain_config.optimizer,
    parameters=self.q_network.parameters(),
)
```

**Step 3: Add scheduler.step() calls in training loop**

Add after optimizer.step() in both feedforward and recurrent training loops:

```python
# After optimizer.step() in feedforward training (around line 708)
self.optimizer.step()

# Step scheduler if present
if self.scheduler is not None:
    self.scheduler.step()

# After optimizer.step() in recurrent training (around line 660)
self.optimizer.step()

# Step scheduler if present
if self.scheduler is not None:
    self.scheduler.step()
```

**Step 4: Commit**

```bash
git add src/townlet/population/vectorized.py
git commit -m "feat(brain): support recurrent networks in VectorizedPopulation

- Add recurrent branch to network building (use build_recurrent)
- Update optimizer creation to unpack (optimizer, scheduler) tuple
- Add scheduler.step() calls after optimizer.step() in training loops
- Set is_recurrent flag based on architecture.type
- Support both feedforward and recurrent from brain config

Part of TASK-005 Phase 2 (5/10)"
```

---

## Task 6: Create brain.yaml for L1_full_observability

**Files:**
- Create: `configs/L1_full_observability/brain.yaml`

**Step 1: Create brain.yaml for L1**

Create `configs/L1_full_observability/brain.yaml`:

```yaml
# Brain Configuration for L1_full_observability
#
# Standard feedforward Q-network for full observability baseline.
#
# Network: 29→256→128→8 (3-layer MLP)
# Optimizer: Adam with Atari DQN standard learning rate
# Loss: MSE (baseline)
# Q-learning: Double DQN (reduces Q-value overestimation)
# Schedule: Constant learning rate
#
# This is the reference architecture for full observability.
# Checkpoint transfer works across all Grid2D configs (constant obs_dim=29).

version: "1.0"

description: "Standard MLP Q-network for full observability baseline"

architecture:
  type: feedforward
  feedforward:
    hidden_layers: [256, 128]
    activation: relu
    dropout: 0.0
    layer_norm: true

optimizer:
  type: adam
  learning_rate: 0.00025  # Atari DQN standard
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0
  schedule:
    type: constant

loss:
  type: mse
  huber_delta: 1.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: true  # Use Double DQN for better value estimates
```

**Step 2: Validate config loads**

```bash
python -c "
from pathlib import Path
from townlet.agent.brain_config import load_brain_config, compute_brain_hash

config = load_brain_config(Path('configs/L1_full_observability'))
print(f'L1 architecture: {config.architecture.type}')
print(f'L1 hidden layers: {config.architecture.feedforward.hidden_layers}')
print(f'L1 uses Double DQN: {config.q_learning.use_double_dqn}')
print(f'L1 brain_hash: {compute_brain_hash(config)[:16]}...')
"
```

Expected output:
```
L1 architecture: feedforward
L1 hidden layers: [256, 128]
L1 uses Double DQN: True
L1 brain_hash: <16 hex chars>...
```

**Step 3: Commit**

```bash
git add configs/L1_full_observability/brain.yaml
git commit -m "feat(brain): add brain.yaml for L1_full_observability

- Standard architecture: [256, 128] hidden layers
- Double DQN enabled for reduced overestimation
- Adam optimizer with lr=0.00025 (Atari DQN standard)
- Constant learning rate schedule
- Reference baseline for full observability

Part of TASK-005 Phase 2 (6/10)"
```

---

## Task 7: Create brain.yaml for L2_partial_observability (LSTM)

**Files:**
- Create: `configs/L2_partial_observability/brain.yaml`

**Step 1: Create brain.yaml for L2 with recurrent architecture**

Create `configs/L2_partial_observability/brain.yaml`:

```yaml
# Brain Configuration for L2_partial_observability
#
# Recurrent spatial Q-network with LSTM for POMDP.
#
# Architecture: CNN (5×5 vision) + MLP encoders + LSTM(256) + Q-head
# Optimizer: Adam with lower learning rate (LSTM stability)
# Loss: Huber (more stable for recurrent networks)
# Q-learning: Double DQN
# Schedule: Constant learning rate
#
# POMDP requires memory (LSTM) to integrate observations over time.
# Vision window: 5×5 local grid (partial observability)

version: "1.0"

description: "Recurrent spatial Q-network with LSTM for POMDP"

architecture:
  type: recurrent
  recurrent:
    vision_encoder:
      channels: [16, 32]
      kernel_sizes: [3, 3]
      strides: [1, 1]
      padding: [1, 1]
      activation: relu

    position_encoder:
      hidden_sizes: [32]
      activation: relu

    meter_encoder:
      hidden_sizes: [32]
      activation: relu

    affordance_encoder:
      hidden_sizes: [32]
      activation: relu

    lstm:
      hidden_size: 256
      num_layers: 1
      dropout: 0.0

    q_head:
      hidden_sizes: [128]
      activation: relu

optimizer:
  type: adam
  learning_rate: 0.0001  # Lower LR for LSTM stability
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0
  schedule:
    type: constant

loss:
  type: huber  # Huber more stable for recurrent
  huber_delta: 1.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: true
```

**Step 2: Validate config loads**

```bash
python -c "
from pathlib import Path
from townlet.agent.brain_config import load_brain_config, compute_brain_hash

config = load_brain_config(Path('configs/L2_partial_observability'))
print(f'L2 architecture: {config.architecture.type}')
print(f'L2 LSTM hidden: {config.architecture.recurrent.lstm.hidden_size}')
print(f'L2 vision channels: {config.architecture.recurrent.vision_encoder.channels}')
print(f'L2 loss type: {config.loss.type}')
print(f'L2 brain_hash: {compute_brain_hash(config)[:16]}...')
"
```

Expected output:
```
L2 architecture: recurrent
L2 LSTM hidden: 256
L2 vision channels: [16, 32]
L2 loss type: huber
L2 brain_hash: <16 hex chars>...
```

**Step 3: Commit**

```bash
git add configs/L2_partial_observability/brain.yaml
git commit -m "feat(brain): add brain.yaml for L2_partial_observability (LSTM)

- Recurrent architecture with LSTM(256) for POMDP
- CNN vision encoder: [16, 32] channels for 5×5 window
- MLP encoders: [32] for position/meter/affordance
- Q-head: [128] hidden layer
- Huber loss for recurrent stability
- Lower learning rate (0.0001) for LSTM training

Part of TASK-005 Phase 2 (7/10)"
```

---

## Task 8: Create brain.yaml for L3_temporal_mechanics (LSTM)

**Files:**
- Create: `configs/L3_temporal_mechanics/brain.yaml`

**Step 1: Create brain.yaml for L3 with recurrent architecture**

Create `configs/L3_temporal_mechanics/brain.yaml`:

```yaml
# Brain Configuration for L3_temporal_mechanics
#
# Recurrent spatial Q-network for temporal dynamics (day/night cycles).
#
# Architecture: Similar to L2 but may benefit from deeper LSTM
# Optimizer: Adam with lower learning rate
# Loss: Huber (stable for recurrent)
# Q-learning: Double DQN
# Schedule: Exponential decay (gradual LR reduction over long training)
#
# L3 adds temporal mechanics (24-tick day/night cycle, operating hours).
# LSTM memory helps agent learn time-dependent patterns.

version: "1.0"

description: "Recurrent Q-network for temporal mechanics with LSTM"

architecture:
  type: recurrent
  recurrent:
    vision_encoder:
      channels: [16, 32]
      kernel_sizes: [3, 3]
      strides: [1, 1]
      padding: [1, 1]
      activation: relu

    position_encoder:
      hidden_sizes: [32]
      activation: relu

    meter_encoder:
      hidden_sizes: [32]
      activation: relu

    affordance_encoder:
      hidden_sizes: [32]
      activation: relu

    lstm:
      hidden_size: 256
      num_layers: 1
      dropout: 0.0

    q_head:
      hidden_sizes: [128]
      activation: relu

optimizer:
  type: adam
  learning_rate: 0.0001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0
  schedule:
    type: exponential  # Gradual LR decay over long training
    gamma: 0.9999      # Very slow decay (reaches 0.5× LR after ~6900 episodes)

loss:
  type: huber
  huber_delta: 1.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: true
```

**Step 2: Validate config loads**

```bash
python -c "
from pathlib import Path
from townlet.agent.brain_config import load_brain_config

config = load_brain_config(Path('configs/L3_temporal_mechanics'))
print(f'L3 architecture: {config.architecture.type}')
print(f'L3 schedule: {config.optimizer.schedule.type}')
print(f'L3 schedule gamma: {config.optimizer.schedule.gamma}')
"
```

Expected output:
```
L3 architecture: recurrent
L3 schedule: exponential
L3 schedule gamma: 0.9999
```

**Step 3: Commit**

```bash
git add configs/L3_temporal_mechanics/brain.yaml
git commit -m "feat(brain): add brain.yaml for L3_temporal_mechanics

- Recurrent architecture with LSTM for temporal patterns
- Exponential LR decay schedule (gamma=0.9999)
- Same network architecture as L2
- Huber loss for stability
- Supports day/night cycle learning

Part of TASK-005 Phase 2 (8/10)"
```

---

## Task 9: Update Existing Tests to Use brain_config

**Files:**
- Modify: `tests/test_townlet/unit/population/test_vectorized_population.py` (if exists)
- Run: All existing tests

**Step 1: Fix broken unit tests from Phase 1**

Many tests will fail because VectorizedPopulation now requires brain_config parameter.

Add helper function to generate test brain configs:

```python
# Add to test file helpers
def create_test_brain_config(architecture_type="feedforward"):
    """Create minimal valid BrainConfig for testing."""
    from townlet.agent.brain_config import (
        BrainConfig,
        ArchitectureConfig,
        FeedforwardConfig,
        OptimizerConfig,
        LossConfig,
        QLearningConfig,
        ScheduleConfig,
    )

    if architecture_type == "feedforward":
        return BrainConfig(
            version="1.0",
            description="Test config",
            architecture=ArchitectureConfig(
                type="feedforward",
                feedforward=FeedforwardConfig(
                    hidden_layers=[128],
                    activation="relu",
                    dropout=0.0,
                    layer_norm=True,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
                schedule=ScheduleConfig(type="constant"),
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
        )
```

Then update any VectorizedPopulation instantiation to include brain_config:

```python
population = VectorizedPopulation(
    env=mock_env,
    curriculum=mock_curriculum,
    exploration=mock_exploration,
    agent_ids=["agent_0"],
    device=torch.device("cpu"),
    brain_config=create_test_brain_config("feedforward"),  # ADD THIS
    obs_dim=29,
    action_dim=8,
    # ... rest of params
)
```

**Step 2: Run full test suite**

```bash
pytest tests/test_townlet/unit/agent/ -v
pytest tests/test_townlet/unit/population/ -v
pytest tests/test_townlet/integration/ -v
```

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/
git commit -m "test(brain): update tests to use brain_config parameter

- Add create_test_brain_config() helper for unit tests
- Update VectorizedPopulation instantiations with brain_config
- Fix broken tests from Phase 1 changes
- Validate all unit and integration tests pass

Part of TASK-005 Phase 2 (9/10)"
```

---

## Task 10: End-to-End Validation with L2 Training

**Files:**
- Test: Run L2_partial_observability training

**Step 1: Run short L2 training (50 episodes)**

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run L2 POMDP training with LSTM
python -m townlet.demo.run_demo --config configs/L2_partial_observability --max-episodes 50
```

Expected: Training runs successfully, LSTM network built from brain.yaml

**Step 2: Verify checkpoint contains correct brain_hash**

```bash
python -c "
import torch
from pathlib import Path
from townlet.agent.brain_config import load_brain_config, compute_brain_hash

# Load expected hash
expected_hash = compute_brain_hash(load_brain_config(Path('configs/L2_partial_observability')))
print(f'Expected brain_hash: {expected_hash[:16]}...')

# Load checkpoint hash
checkpoints = list(Path('checkpoints').glob('L2_partial_observability*/checkpoint_*.pt'))
if checkpoints:
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    checkpoint = torch.load(latest)
    actual_hash = checkpoint['brain_hash']
    print(f'Checkpoint brain_hash: {actual_hash[:16]}...')
    print(f'Match: {expected_hash == actual_hash}')
"
```

Expected output:
```
Expected brain_hash: <16 hex chars>...
Checkpoint brain_hash: <16 hex chars>...
Match: True
```

**Step 3: Test scheduler stepping**

```bash
python -c "
import torch
from pathlib import Path

# Check that optimizer state includes scheduler (if schedule != constant)
checkpoints = list(Path('checkpoints').glob('L2_partial_observability*/checkpoint_*.pt'))
if checkpoints:
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    checkpoint = torch.load(latest)
    print('Checkpoint keys:', list(checkpoint.keys()))
    print('Has optimizer_state_dict:', 'optimizer_state_dict' in checkpoint)
"
```

**Step 4: Run L3 training to test exponential scheduler**

```bash
# Run L3 with exponential LR decay
python -m townlet.demo.run_demo --config configs/L3_temporal_mechanics --max-episodes 50
```

Expected: Training runs with gradual LR decay

**Step 5: Commit**

```bash
git commit --allow-empty -m "test(brain): validate Phase 2 end-to-end with L2/L3 training

- Validate L2 POMDP training with LSTM from brain.yaml
- Validate L3 training with exponential LR scheduler
- Verify checkpoint brain_hash matches config
- Confirm recurrent networks build and train correctly
- Confirm schedulers step properly during training

Phase 2 Complete! ✅

Part of TASK-005 Phase 2 (10/10 - PHASE 2 COMPLETE!)"
```

---

## Phase 2 Complete! Summary

**Configuration System:**
- ✅ `RecurrentConfig`: CNN + MLP encoders + LSTM + Q-head
- ✅ `ScheduleConfig`: StepLR, CosineAnnealingLR, ExponentialLR
- ✅ Updated OptimizerFactory to return (optimizer, scheduler) tuple

**Network Building:**
- ✅ `NetworkFactory.build_recurrent()`: Build LSTM networks from config
- ✅ VectorizedPopulation supports both feedforward and recurrent

**Configuration Files:**
- ✅ `configs/L1_full_observability/brain.yaml` (feedforward, Double DQN)
- ✅ `configs/L2_partial_observability/brain.yaml` (LSTM, Huber loss)
- ✅ `configs/L3_temporal_mechanics/brain.yaml` (LSTM, exponential schedule)

**Testing:**
- ✅ Unit tests for all new DTOs and factories
- ✅ Updated existing tests to use brain_config
- ✅ End-to-end validation with L2/L3 training

**Scheduler Integration:**
- ✅ Scheduler.step() called after optimizer.step()
- ✅ Constant, step_decay, cosine, exponential schedules supported

**Next Phase:** Dueling DQN + Prioritized Experience Replay (Phase 3)

---
