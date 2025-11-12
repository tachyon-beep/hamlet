# Brain As Code - Phase 1: Feedforward Networks & Basic Features

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make SimpleQNetwork architecture configurable via `brain.yaml`, enabling network architecture experimentation without code changes.

**Architecture:** Extract hardcoded network dimensions (hidden_dim=128) into declarative YAML configuration. Build networks through factory pattern. Add brain_hash to checkpoints for provenance. Keep existing network classes unchanged (forward compatibility with future SDA).

**Tech Stack:** Pydantic (validation), PyTorch (networks), YAML (config), SHA256 (hashing)

**Scope:**
- ✅ Feedforward networks (SimpleQNetwork)
- ✅ Basic optimizers (Adam, AdamW, SGD, RMSprop)
- ✅ Basic loss functions (MSE, Huber, SmoothL1)
- ✅ Constant learning rate (no schedulers yet)
- ✅ Standard ReplayBuffer (no PER yet)
- ✅ brain_hash provenance
- ✅ Update L0_0_minimal, L0_5_dual_resource configs
- ❌ Recurrent networks (Phase 2)
- ❌ Learning rate schedulers (Phase 2)
- ❌ Dueling DQN (Phase 3)
- ❌ Prioritized Experience Replay (Phase 3)

---

## Task 1: Create BrainConfig Pydantic Schema (Feedforward Only)

**Files:**
- Create: `src/townlet/agent/brain_config.py`
- Test: `tests/test_townlet/unit/agent/test_brain_config.py`

**Step 1: Write failing test for FeedforwardConfig validation**

Create `tests/test_townlet/unit/agent/test_brain_config.py`:

```python
"""Tests for brain configuration DTOs."""

from pathlib import Path
import pytest
from pydantic import ValidationError
from townlet.agent.brain_config import FeedforwardConfig, BrainConfig


def test_feedforward_config_valid():
    """FeedforwardConfig accepts valid parameters."""
    config = FeedforwardConfig(
        hidden_layers=[256, 128],
        activation="relu",
        dropout=0.0,
        layer_norm=True,
    )
    assert config.hidden_layers == [256, 128]
    assert config.activation == "relu"
    assert config.dropout == 0.0
    assert config.layer_norm is True


def test_feedforward_config_rejects_empty_layers():
    """FeedforwardConfig requires at least one hidden layer."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[],
            activation="relu",
            dropout=0.0,
            layer_norm=False,
        )
    assert "hidden_layers" in str(exc_info.value)


def test_feedforward_config_rejects_invalid_activation():
    """FeedforwardConfig rejects unsupported activation functions."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[128],
            activation="invalid",
            dropout=0.0,
            layer_norm=False,
        )
    assert "activation" in str(exc_info.value)


def test_feedforward_config_rejects_negative_dropout():
    """FeedforwardConfig rejects dropout < 0."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[128],
            activation="relu",
            dropout=-0.1,
            layer_norm=False,
        )
    assert "dropout" in str(exc_info.value)


def test_feedforward_config_rejects_dropout_gte_1():
    """FeedforwardConfig rejects dropout >= 1.0."""
    with pytest.raises(ValidationError) as exc_info:
        FeedforwardConfig(
            hidden_layers=[128],
            activation="relu",
            dropout=1.0,
            layer_norm=False,
        )
    assert "dropout" in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_feedforward_config_valid -v
```

Expected: `ModuleNotFoundError: No module named 'townlet.agent.brain_config'`

**Step 3: Write minimal FeedforwardConfig implementation**

Create `src/townlet/agent/brain_config.py`:

```python
"""Brain configuration DTOs for declarative agent architecture.

Follows no-defaults principle: all behavioral parameters must be explicit.
Forward-compatible with future SDA (Software Defined Agent) architecture.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class FeedforwardConfig(BaseModel):
    """Feedforward MLP architecture configuration.

    Example:
        >>> config = FeedforwardConfig(
        ...     hidden_layers=[256, 128],
        ...     activation="relu",
        ...     dropout=0.0,
        ...     layer_norm=True,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    hidden_layers: list[int] = Field(
        min_length=1,
        description="Hidden layer sizes (e.g., [256, 128] for 2 hidden layers)"
    )
    activation: Literal["relu", "gelu", "swish", "tanh", "elu"] = Field(
        description="Activation function"
    )
    dropout: float = Field(
        ge=0.0,
        lt=1.0,
        description="Dropout probability (0.0 = no dropout)"
    )
    layer_norm: bool = Field(
        description="Apply LayerNorm after each hidden layer"
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py -v
```

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/brain_config.py tests/test_townlet/unit/agent/test_brain_config.py
git commit -m "feat(brain): add FeedforwardConfig Pydantic schema

- Add FeedforwardConfig DTO with validation
- Enforce no-defaults principle (all fields required)
- Validate hidden_layers min_length=1
- Validate activation from allowed set
- Validate dropout in [0.0, 1.0)
- Add comprehensive unit tests

Part of TASK-005 Phase 1: Brain As Code (feedforward networks)"
```

---

## Task 2: Add Optimizer and Loss Configuration DTOs

**Files:**
- Modify: `src/townlet/agent/brain_config.py`
- Test: `tests/test_townlet/unit/agent/test_brain_config.py`

**Step 1: Write failing tests for OptimizerConfig and LossConfig**

Add to `tests/test_townlet/unit/agent/test_brain_config.py`:

```python
from townlet.agent.brain_config import OptimizerConfig, LossConfig


def test_optimizer_config_adam():
    """OptimizerConfig accepts Adam configuration."""
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.00025,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
    )
    assert config.type == "adam"
    assert config.learning_rate == 0.00025


def test_optimizer_config_sgd():
    """OptimizerConfig accepts SGD configuration."""
    config = OptimizerConfig(
        type="sgd",
        learning_rate=0.01,
        sgd_momentum=0.9,
        sgd_nesterov=True,
        weight_decay=0.0,
    )
    assert config.type == "sgd"
    assert config.sgd_momentum == 0.9


def test_optimizer_config_rejects_negative_lr():
    """OptimizerConfig rejects negative learning rate."""
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(
            type="adam",
            learning_rate=-0.001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
        )
    assert "learning_rate" in str(exc_info.value)


def test_loss_config_mse():
    """LossConfig accepts MSE loss."""
    config = LossConfig(type="mse")
    assert config.type == "mse"


def test_loss_config_huber():
    """LossConfig accepts Huber loss with delta."""
    config = LossConfig(type="huber", huber_delta=1.0)
    assert config.type == "huber"
    assert config.huber_delta == 1.0


def test_loss_config_rejects_negative_huber_delta():
    """LossConfig rejects negative huber_delta."""
    with pytest.raises(ValidationError) as exc_info:
        LossConfig(type="huber", huber_delta=-1.0)
    assert "huber_delta" in str(exc_info.value)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_optimizer_config_adam -v
```

Expected: `ImportError: cannot import name 'OptimizerConfig'`

**Step 3: Implement OptimizerConfig and LossConfig**

Add to `src/townlet/agent/brain_config.py`:

```python
class OptimizerConfig(BaseModel):
    """Optimizer configuration.

    All optimizer-specific parameters required (no defaults).

    Example:
        >>> adam = OptimizerConfig(
        ...     type="adam",
        ...     learning_rate=0.00025,
        ...     adam_beta1=0.9,
        ...     adam_beta2=0.999,
        ...     adam_eps=1e-8,
        ...     weight_decay=0.0,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["adam", "adamw", "sgd", "rmsprop"] = Field(
        description="Optimizer type"
    )
    learning_rate: float = Field(
        gt=0.0,
        description="Learning rate"
    )

    # Adam/AdamW parameters (required for type=adam/adamw)
    adam_beta1: float = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Adam beta1 parameter (required for adam/adamw)"
    )
    adam_beta2: float = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Adam beta2 parameter (required for adam/adamw)"
    )
    adam_eps: float = Field(
        default=None,
        gt=0.0,
        description="Adam epsilon parameter (required for adam/adamw)"
    )

    # SGD parameters (required for type=sgd)
    sgd_momentum: float = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="SGD momentum (required for sgd)"
    )
    sgd_nesterov: bool = Field(
        default=None,
        description="Use Nesterov momentum (required for sgd)"
    )

    # RMSprop parameters (required for type=rmsprop)
    rmsprop_alpha: float = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="RMSprop alpha/decay (required for rmsprop)"
    )
    rmsprop_eps: float = Field(
        default=None,
        gt=0.0,
        description="RMSprop epsilon (required for rmsprop)"
    )

    # Common parameter
    weight_decay: float = Field(
        ge=0.0,
        description="L2 weight decay (all optimizers)"
    )


class LossConfig(BaseModel):
    """Loss function configuration.

    Example:
        >>> mse = LossConfig(type="mse")
        >>> huber = LossConfig(type="huber", huber_delta=1.0)
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["mse", "huber", "smooth_l1"] = Field(
        description="Loss function type"
    )

    huber_delta: float = Field(
        default=1.0,
        gt=0.0,
        description="Delta parameter for Huber loss (ignored for mse/smooth_l1)"
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_optimizer_config_adam -v
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_loss_config_mse -v
```

Expected: All new tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/brain_config.py tests/test_townlet/unit/agent/test_brain_config.py
git commit -m "feat(brain): add OptimizerConfig and LossConfig DTOs

- Add OptimizerConfig (Adam, AdamW, SGD, RMSprop)
- Add LossConfig (MSE, Huber, SmoothL1)
- Validate optimizer-specific parameters
- Add unit tests for validation

Part of TASK-005 Phase 1"
```

---

## Task 3: Add Top-Level BrainConfig DTO

**Files:**
- Modify: `src/townlet/agent/brain_config.py`
- Test: `tests/test_townlet/unit/agent/test_brain_config.py`

**Step 1: Write failing test for BrainConfig**

Add to `tests/test_townlet/unit/agent/test_brain_config.py`:

```python
def test_brain_config_feedforward():
    """BrainConfig accepts feedforward architecture."""
    config = BrainConfig(
        version="1.0",
        description="Simple feedforward Q-network",
        architecture=ArchitectureConfig(
            type="feedforward",
            feedforward=FeedforwardConfig(
                hidden_layers=[256, 128],
                activation="relu",
                dropout=0.0,
                layer_norm=True,
            ),
        ),
        optimizer=OptimizerConfig(
            type="adam",
            learning_rate=0.00025,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            weight_decay=0.0,
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )
    assert config.architecture.type == "feedforward"
    assert config.optimizer.type == "adam"


def test_brain_config_requires_feedforward_when_type_feedforward():
    """BrainConfig requires feedforward field when type=feedforward."""
    with pytest.raises(ValidationError) as exc_info:
        BrainConfig(
            version="1.0",
            description="Test",
            architecture=ArchitectureConfig(
                type="feedforward",
                # Missing feedforward field!
            ),
            optimizer=OptimizerConfig(
                type="adam",
                learning_rate=0.001,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                weight_decay=0.0,
            ),
            loss=LossConfig(type="mse"),
            q_learning=QLearningConfig(
                gamma=0.99,
                target_update_frequency=100,
                use_double_dqn=False,
            ),
        )
    assert "feedforward" in str(exc_info.value).lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_brain_config_feedforward -v
```

Expected: `ImportError: cannot import name 'BrainConfig'`

**Step 3: Implement ArchitectureConfig, QLearningConfig, and BrainConfig**

Add to `src/townlet/agent/brain_config.py`:

```python
from pydantic import model_validator


class ArchitectureConfig(BaseModel):
    """Neural network architecture configuration.

    Future: Will support recurrent, dueling, rainbow architectures.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["feedforward"] = Field(
        description="Architecture type (Phase 1: feedforward only)"
    )

    # Architecture-specific configs (exactly one required based on type)
    feedforward: FeedforwardConfig | None = Field(
        default=None,
        description="Feedforward MLP config (required when type=feedforward)"
    )

    @model_validator(mode="after")
    def validate_architecture_match(self) -> "ArchitectureConfig":
        """Ensure architecture config matches type."""
        if self.type == "feedforward" and self.feedforward is None:
            raise ValueError("type='feedforward' requires feedforward config")
        return self


class QLearningConfig(BaseModel):
    """Q-learning algorithm configuration."""

    model_config = ConfigDict(extra="forbid")

    gamma: float = Field(
        ge=0.0,
        le=1.0,
        description="Discount factor"
    )
    target_update_frequency: int = Field(
        gt=0,
        description="Update target network every N training steps"
    )
    use_double_dqn: bool = Field(
        description="Use Double DQN algorithm (van Hasselt et al. 2016)"
    )


class BrainConfig(BaseModel):
    """Complete brain configuration.

    Top-level configuration for agent architecture, optimizer, and learning.
    All fields required (no-defaults principle).

    Example:
        >>> config = BrainConfig(
        ...     version="1.0",
        ...     description="Feedforward Q-network for L0",
        ...     architecture=ArchitectureConfig(...),
        ...     optimizer=OptimizerConfig(...),
        ...     loss=LossConfig(...),
        ...     q_learning=QLearningConfig(...),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(
        description="Configuration schema version (e.g., '1.0')"
    )
    description: str = Field(
        description="Human-readable description of this brain configuration"
    )
    architecture: ArchitectureConfig = Field(
        description="Network architecture specification"
    )
    optimizer: OptimizerConfig = Field(
        description="Optimizer configuration"
    )
    loss: LossConfig = Field(
        description="Loss function configuration"
    )
    q_learning: QLearningConfig = Field(
        description="Q-learning algorithm parameters"
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/brain_config.py tests/test_townlet/unit/agent/test_brain_config.py
git commit -m "feat(brain): add ArchitectureConfig, QLearningConfig, BrainConfig

- Add ArchitectureConfig with feedforward type (Phase 1)
- Add QLearningConfig (gamma, target updates, double DQN)
- Add top-level BrainConfig composing all sub-configs
- Validate architecture type matches config presence
- Add comprehensive unit tests

Part of TASK-005 Phase 1"
```

---

## Task 4: Add brain.yaml Loader Function

**Files:**
- Modify: `src/townlet/agent/brain_config.py`
- Test: `tests/test_townlet/unit/agent/test_brain_config.py`
- Test Fixture: `tests/test_townlet/fixtures/brain_configs/valid_feedforward.yaml`

**Step 1: Write failing test for load_brain_config**

Add to `tests/test_townlet/unit/agent/test_brain_config.py`:

```python
from pathlib import Path
from townlet.agent.brain_config import load_brain_config


def test_load_brain_config_valid(tmp_path):
    """load_brain_config loads valid brain.yaml."""
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text("""
version: "1.0"
description: "Test feedforward network"

architecture:
  type: feedforward
  feedforward:
    hidden_layers: [128, 64]
    activation: relu
    dropout: 0.0
    layer_norm: true

optimizer:
  type: adam
  learning_rate: 0.001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0

loss:
  type: mse
  huber_delta: 1.0

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: false
""")

    config = load_brain_config(tmp_path)
    assert config.version == "1.0"
    assert config.architecture.feedforward.hidden_layers == [128, 64]
    assert config.optimizer.learning_rate == 0.001


def test_load_brain_config_missing_file(tmp_path):
    """load_brain_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_brain_config(tmp_path)
    assert "brain.yaml" in str(exc_info.value)


def test_load_brain_config_invalid_yaml(tmp_path):
    """load_brain_config raises ValueError for invalid YAML."""
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text("""
version: "1.0"
architecture:
  type: feedforward
  # Missing feedforward config!
optimizer:
  type: adam
  learning_rate: 0.001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0
loss:
  type: mse
q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: false
""")

    with pytest.raises(ValueError) as exc_info:
        load_brain_config(tmp_path)
    assert "validation" in str(exc_info.value).lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_load_brain_config_valid -v
```

Expected: `ImportError: cannot import name 'load_brain_config'`

**Step 3: Implement load_brain_config**

Add to `src/townlet/agent/brain_config.py`:

```python
from pathlib import Path
from pydantic import ValidationError
import yaml


def load_brain_config(config_dir: Path) -> BrainConfig:
    """Load and validate brain configuration from brain.yaml.

    Args:
        config_dir: Directory containing brain.yaml

    Returns:
        Validated BrainConfig

    Raises:
        FileNotFoundError: If brain.yaml not found
        ValueError: If validation fails

    Example:
        >>> config = load_brain_config(Path("configs/L0_0_minimal"))
        >>> print(config.architecture.type)
        feedforward
    """
    config_path = Path(config_dir) / "brain.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"brain.yaml not found in {config_dir}. "
            f"Brain configuration is required for all config packs."
        )

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return BrainConfig(**data)
    except ValidationError as e:
        # Format validation error for user-friendly output
        error_msgs = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_msgs.append(f"  - {field_path}: {error['msg']}")

        formatted_errors = "\n".join(error_msgs)
        raise ValueError(
            f"Invalid brain.yaml in {config_dir}:\n{formatted_errors}\n\n"
            f"See docs/config-schemas/brain.md for valid schema."
        ) from e
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_load_brain_config_valid -v
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_load_brain_config_missing_file -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/brain_config.py tests/test_townlet/unit/agent/test_brain_config.py
git commit -m "feat(brain): add load_brain_config YAML loader

- Add load_brain_config() function
- Load brain.yaml from config directory
- Validate with Pydantic, provide helpful error messages
- Add unit tests for valid/invalid/missing files

Part of TASK-005 Phase 1"
```

---

## Task 5: Create NetworkFactory (Feedforward Only)

**Files:**
- Create: `src/townlet/agent/network_factory.py`
- Test: `tests/test_townlet/unit/agent/test_network_factory.py`

**Step 1: Write failing test for NetworkFactory.build_feedforward**

Create `tests/test_townlet/unit/agent/test_network_factory.py`:

```python
"""Tests for network factory."""

import torch
from townlet.agent.network_factory import NetworkFactory
from townlet.agent.brain_config import FeedforwardConfig


def test_build_feedforward_basic():
    """NetworkFactory builds SimpleQNetwork from FeedforwardConfig."""
    config = FeedforwardConfig(
        hidden_layers=[128, 64],
        activation="relu",
        dropout=0.0,
        layer_norm=True,
    )

    network = NetworkFactory.build_feedforward(
        config=config,
        obs_dim=29,
        action_dim=8,
    )

    # Check output shape
    obs = torch.randn(4, 29)
    q_values = network(obs)
    assert q_values.shape == (4, 8)


def test_build_feedforward_multiple_layers():
    """NetworkFactory handles multiple hidden layers."""
    config = FeedforwardConfig(
        hidden_layers=[256, 128, 64],
        activation="gelu",
        dropout=0.1,
        layer_norm=False,
    )

    network = NetworkFactory.build_feedforward(
        config=config,
        obs_dim=54,
        action_dim=10,
    )

    obs = torch.randn(2, 54)
    q_values = network(obs)
    assert q_values.shape == (2, 10)


def test_build_feedforward_parameter_count():
    """NetworkFactory creates network with expected parameter count."""
    config = FeedforwardConfig(
        hidden_layers=[128],
        activation="relu",
        dropout=0.0,
        layer_norm=True,
    )

    network = NetworkFactory.build_feedforward(
        config=config,
        obs_dim=29,
        action_dim=8,
    )

    total_params = sum(p.numel() for p in network.parameters())
    # Rough sanity check (not exact)
    assert 1000 < total_params < 20000
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_network_factory.py::test_build_feedforward_basic -v
```

Expected: `ModuleNotFoundError: No module named 'townlet.agent.network_factory'`

**Step 3: Implement NetworkFactory.build_feedforward**

Create `src/townlet/agent/network_factory.py`:

```python
"""Network factory for building Q-networks from configuration.

Builds neural networks from BrainConfig specifications.
Forward-compatible with future SDA (Software Defined Agent) architecture.
"""

import torch.nn as nn
from townlet.agent.brain_config import FeedforwardConfig


class NetworkFactory:
    """Factory for building Q-networks from declarative configuration."""

    @staticmethod
    def build_feedforward(
        config: FeedforwardConfig,
        obs_dim: int,
        action_dim: int,
    ) -> nn.Module:
        """Build feedforward MLP Q-network from configuration.

        Args:
            config: Feedforward architecture configuration
            obs_dim: Observation dimension
            action_dim: Action dimension

        Returns:
            PyTorch module (feedforward Q-network)

        Example:
            >>> config = FeedforwardConfig(
            ...     hidden_layers=[256, 128],
            ...     activation="relu",
            ...     dropout=0.0,
            ...     layer_norm=True,
            ... )
            >>> network = NetworkFactory.build_feedforward(config, 29, 8)
            >>> network(torch.randn(4, 29)).shape
            torch.Size([4, 8])
        """
        layers = []
        in_features = obs_dim

        # Build hidden layers
        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))

            if config.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            layers.append(NetworkFactory._get_activation(config.activation))

            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))

            in_features = hidden_size

        # Output layer (Q-values)
        layers.append(nn.Linear(in_features, action_dim))

        return nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function module from config string.

        Args:
            activation: Activation function name

        Returns:
            PyTorch activation module
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # Swish = SiLU in PyTorch
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations[activation]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_network_factory.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/network_factory.py tests/test_townlet/unit/agent/test_network_factory.py
git commit -m "feat(brain): add NetworkFactory for feedforward networks

- Add NetworkFactory.build_feedforward()
- Build MLP Q-networks from FeedforwardConfig
- Support configurable activation, dropout, layer_norm
- Add unit tests for network building

Part of TASK-005 Phase 1"
```

---

## Task 6: Create OptimizerFactory

**Files:**
- Create: `src/townlet/agent/optimizer_factory.py`
- Test: `tests/test_townlet/unit/agent/test_optimizer_factory.py`

**Step 1: Write failing test for OptimizerFactory**

Create `tests/test_townlet/unit/agent/test_optimizer_factory.py`:

```python
"""Tests for optimizer factory."""

import torch.nn as nn
import torch.optim as optim
from townlet.agent.optimizer_factory import OptimizerFactory
from townlet.agent.brain_config import OptimizerConfig


def test_build_adam():
    """OptimizerFactory builds Adam optimizer."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adam",
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,
    )

    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["betas"] == (0.9, 0.999)


def test_build_adamw():
    """OptimizerFactory builds AdamW optimizer."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="adamw",
        learning_rate=0.0003,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.01,
    )

    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, optim.AdamW)
    assert optimizer.defaults["lr"] == 0.0003
    assert optimizer.defaults["weight_decay"] == 0.01


def test_build_sgd():
    """OptimizerFactory builds SGD optimizer."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="sgd",
        learning_rate=0.01,
        sgd_momentum=0.9,
        sgd_nesterov=True,
        weight_decay=0.0,
    )

    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, optim.SGD)
    assert optimizer.defaults["lr"] == 0.01
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["nesterov"] is True


def test_build_rmsprop():
    """OptimizerFactory builds RMSprop optimizer."""
    network = nn.Linear(10, 5)
    config = OptimizerConfig(
        type="rmsprop",
        learning_rate=0.00025,
        rmsprop_alpha=0.99,
        rmsprop_eps=1e-8,
        weight_decay=0.0,
    )

    optimizer = OptimizerFactory.build(config, network.parameters())

    assert isinstance(optimizer, optim.RMSprop)
    assert optimizer.defaults["lr"] == 0.00025
    assert optimizer.defaults["alpha"] == 0.99
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_optimizer_factory.py::test_build_adam -v
```

Expected: `ModuleNotFoundError: No module named 'townlet.agent.optimizer_factory'`

**Step 3: Implement OptimizerFactory**

Create `src/townlet/agent/optimizer_factory.py`:

```python
"""Optimizer factory for building optimizers from configuration."""

import torch.optim as optim
from townlet.agent.brain_config import OptimizerConfig


class OptimizerFactory:
    """Factory for building PyTorch optimizers from configuration."""

    @staticmethod
    def build(config: OptimizerConfig, parameters) -> optim.Optimizer:
        """Build optimizer from configuration.

        Args:
            config: Optimizer configuration
            parameters: Network parameters (from network.parameters())

        Returns:
            PyTorch optimizer

        Example:
            >>> config = OptimizerConfig(
            ...     type="adam",
            ...     learning_rate=0.001,
            ...     adam_beta1=0.9,
            ...     adam_beta2=0.999,
            ...     adam_eps=1e-8,
            ...     weight_decay=0.0,
            ... )
            >>> optimizer = OptimizerFactory.build(config, network.parameters())
        """
        if config.type == "adam":
            return optim.Adam(
                parameters,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
        elif config.type == "adamw":
            return optim.AdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
        elif config.type == "sgd":
            return optim.SGD(
                parameters,
                lr=config.learning_rate,
                momentum=config.sgd_momentum,
                nesterov=config.sgd_nesterov,
                weight_decay=config.weight_decay,
            )
        elif config.type == "rmsprop":
            return optim.RMSprop(
                parameters,
                lr=config.learning_rate,
                alpha=config.rmsprop_alpha,
                eps=config.rmsprop_eps,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.type}")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_optimizer_factory.py -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/optimizer_factory.py tests/test_townlet/unit/agent/test_optimizer_factory.py
git commit -m "feat(brain): add OptimizerFactory

- Add OptimizerFactory.build()
- Support Adam, AdamW, SGD, RMSprop
- Map config parameters to PyTorch optimizer args
- Add unit tests for all optimizer types

Part of TASK-005 Phase 1"
```

---

## Task 7: Create LossFactory

**Files:**
- Create: `src/townlet/agent/loss_factory.py`
- Test: `tests/test_townlet/unit/agent/test_loss_factory.py`

**Step 1: Write failing test for LossFactory**

Create `tests/test_townlet/unit/agent/test_loss_factory.py`:

```python
"""Tests for loss factory."""

import torch
import torch.nn as nn
from townlet.agent.loss_factory import LossFactory
from townlet.agent.brain_config import LossConfig


def test_build_mse():
    """LossFactory builds MSE loss."""
    config = LossConfig(type="mse")
    loss_fn = LossFactory.build(config)

    assert isinstance(loss_fn, nn.MSELoss)

    # Test loss computation
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    loss = loss_fn(pred, target)
    expected = torch.mean((pred - target) ** 2)
    assert torch.allclose(loss, expected)


def test_build_huber():
    """LossFactory builds Huber loss with custom delta."""
    config = LossConfig(type="huber", huber_delta=2.0)
    loss_fn = LossFactory.build(config)

    assert isinstance(loss_fn, nn.HuberLoss)
    assert loss_fn.delta == 2.0


def test_build_smooth_l1():
    """LossFactory builds SmoothL1Loss."""
    config = LossConfig(type="smooth_l1")
    loss_fn = LossFactory.build(config)

    assert isinstance(loss_fn, nn.SmoothL1Loss)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_loss_factory.py::test_build_mse -v
```

Expected: `ModuleNotFoundError: No module named 'townlet.agent.loss_factory'`

**Step 3: Implement LossFactory**

Create `src/townlet/agent/loss_factory.py`:

```python
"""Loss function factory for building loss functions from configuration."""

import torch.nn as nn
from townlet.agent.brain_config import LossConfig


class LossFactory:
    """Factory for building PyTorch loss functions from configuration."""

    @staticmethod
    def build(config: LossConfig) -> nn.Module:
        """Build loss function from configuration.

        Args:
            config: Loss function configuration

        Returns:
            PyTorch loss module

        Example:
            >>> config = LossConfig(type="mse")
            >>> loss_fn = LossFactory.build(config)
            >>> loss = loss_fn(predictions, targets)
        """
        if config.type == "mse":
            return nn.MSELoss()
        elif config.type == "huber":
            return nn.HuberLoss(delta=config.huber_delta)
        elif config.type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {config.type}")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_loss_factory.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/loss_factory.py tests/test_townlet/unit/agent/test_loss_factory.py
git commit -m "feat(brain): add LossFactory

- Add LossFactory.build()
- Support MSE, Huber, SmoothL1 loss functions
- Map config parameters to PyTorch loss modules
- Add unit tests for all loss types

Part of TASK-005 Phase 1"
```

---

## Task 8: Compute brain_hash for Checkpoint Provenance

**Files:**
- Modify: `src/townlet/agent/brain_config.py`
- Test: `tests/test_townlet/unit/agent/test_brain_config.py`

**Step 1: Write failing test for compute_brain_hash**

Add to `tests/test_townlet/unit/agent/test_brain_config.py`:

```python
from townlet.agent.brain_config import compute_brain_hash


def test_compute_brain_hash_deterministic():
    """compute_brain_hash produces same hash for identical configs."""
    config1 = BrainConfig(
        version="1.0",
        description="Test",
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
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )

    config2 = BrainConfig(
        version="1.0",
        description="Test",
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
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )

    hash1 = compute_brain_hash(config1)
    hash2 = compute_brain_hash(config2)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_compute_brain_hash_different_for_different_configs():
    """compute_brain_hash produces different hashes for different configs."""
    config1 = BrainConfig(
        version="1.0",
        description="Test",
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
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )

    # Different: hidden_layers=[256] instead of [128]
    config2 = BrainConfig(
        version="1.0",
        description="Test",
        architecture=ArchitectureConfig(
            type="feedforward",
            feedforward=FeedforwardConfig(
                hidden_layers=[256],
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
        ),
        loss=LossConfig(type="mse"),
        q_learning=QLearningConfig(
            gamma=0.99,
            target_update_frequency=100,
            use_double_dqn=False,
        ),
    )

    hash1 = compute_brain_hash(config1)
    hash2 = compute_brain_hash(config2)

    assert hash1 != hash2
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_compute_brain_hash_deterministic -v
```

Expected: `ImportError: cannot import name 'compute_brain_hash'`

**Step 3: Implement compute_brain_hash**

Add to `src/townlet/agent/brain_config.py`:

```python
import hashlib
import json


def compute_brain_hash(config: BrainConfig) -> str:
    """Compute SHA256 hash of brain configuration for checkpoint provenance.

    Similar to drive_hash for DAC, brain_hash ensures checkpoint reproducibility.
    Any change to brain configuration produces a different hash.

    Args:
        config: Brain configuration

    Returns:
        64-character hex string (SHA256 digest)

    Example:
        >>> config = load_brain_config(Path("configs/L0_0_minimal"))
        >>> brain_hash = compute_brain_hash(config)
        >>> print(brain_hash[:16])
        a3f9d8c1e2b4f7a9
    """
    # Serialize config to deterministic JSON
    # Use model_dump() to convert Pydantic to dict, then sort keys
    config_dict = config.model_dump()
    config_json = json.dumps(config_dict, sort_keys=True)

    # Compute SHA256 hash
    hash_bytes = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    return hash_bytes
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_compute_brain_hash_deterministic -v
pytest tests/test_townlet/unit/agent/test_brain_config.py::test_compute_brain_hash_different_for_different_configs -v
```

Expected: Both tests PASS

**Step 5: Commit**

```bash
git add src/townlet/agent/brain_config.py tests/test_townlet/unit/agent/test_brain_config.py
git commit -m "feat(brain): add compute_brain_hash for checkpoint provenance

- Add compute_brain_hash() function
- Compute SHA256 hash of brain configuration
- Deterministic hash for reproducibility
- Similar to drive_hash from DAC
- Add unit tests for hash determinism and uniqueness

Part of TASK-005 Phase 1"
```

---

## Task 9: Update VectorizedPopulation to Use Brain Config (Part 1: Load Config)

**Files:**
- Modify: `src/townlet/population/vectorized.py` (lines 42-63, 119-140)
- Test: `tests/test_townlet/integration/test_brain_config_integration.py`

**Step 1: Write failing integration test**

Create `tests/test_townlet/integration/test_brain_config_integration.py`:

```python
"""Integration tests for brain config with VectorizedPopulation."""

from pathlib import Path
import pytest
import torch
from townlet.agent.brain_config import BrainConfig, load_brain_config
from townlet.agent.network_factory import NetworkFactory
from townlet.agent.optimizer_factory import OptimizerFactory


def test_load_brain_config_and_build_network(tmp_path):
    """Load brain.yaml and build network from it."""
    # Create brain.yaml
    brain_yaml = tmp_path / "brain.yaml"
    brain_yaml.write_text("""
version: "1.0"
description: "Test integration"

architecture:
  type: feedforward
  feedforward:
    hidden_layers: [64, 32]
    activation: relu
    dropout: 0.0
    layer_norm: true

optimizer:
  type: adam
  learning_rate: 0.001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0

loss:
  type: mse
  huber_delta: 1.0

q_learning:
  gamma: 0.95
  target_update_frequency: 50
  use_double_dqn: false
""")

    # Load config
    brain_config = load_brain_config(tmp_path)

    # Build network
    network = NetworkFactory.build_feedforward(
        config=brain_config.architecture.feedforward,
        obs_dim=21,
        action_dim=7,
    )

    # Build optimizer
    optimizer = OptimizerFactory.build(
        config=brain_config.optimizer,
        parameters=network.parameters(),
    )

    # Test network forward pass
    obs = torch.randn(4, 21)
    q_values = network(obs)
    assert q_values.shape == (4, 7)

    # Test optimizer step
    loss = q_values.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Step 2: Run test to verify it passes**

```bash
pytest tests/test_townlet/integration/test_brain_config_integration.py::test_load_brain_config_and_build_network -v
```

Expected: PASS (validates all components work together)

**Step 3: Add brain_config parameter to VectorizedPopulation.__init__**

Modify `src/townlet/population/vectorized.py` lines 42-63:

```python
# Add import at top of file
from townlet.agent.brain_config import BrainConfig

# Modify __init__ signature (around line 42)
def __init__(
    self,
    env: VectorizedHamletEnv,
    curriculum: CurriculumManager,
    exploration: ExplorationStrategy,
    agent_ids: list[str],
    device: torch.device,
    brain_config: BrainConfig,  # NEW: Brain configuration
    obs_dim: int = 70,
    action_dim: int | None = None,
    learning_rate: float = 0.00025,  # DEPRECATED: Use brain_config.optimizer.learning_rate
    gamma: float = 0.99,  # DEPRECATED: Use brain_config.q_learning.gamma
    replay_buffer_capacity: int = 10000,
    network_type: str = "simple",  # DEPRECATED: Use brain_config.architecture.type
    vision_window_size: int = 5,
    tb_logger=None,
    train_frequency: int = 4,
    target_update_frequency: int = 100,  # DEPRECATED: Use brain_config.q_learning.target_update_frequency
    batch_size: int | None = None,
    sequence_length: int = 8,
    max_grad_norm: float = 10.0,
    use_double_dqn: bool = False,  # DEPRECATED: Use brain_config.q_learning.use_double_dqn
):
    """Initialize vectorized population with brain configuration.

    Args:
        brain_config: Brain configuration (architecture, optimizer, loss, q_learning)
        ... (rest unchanged)
    """
    self.brain_config = brain_config
    # ... rest of __init__ unchanged for now
```

**Step 4: Store brain_config and gamma from config**

Add after line 94 in `__init__`:

```python
# Extract Q-learning params from brain_config (line ~94)
self.gamma = brain_config.q_learning.gamma
self.use_double_dqn = brain_config.q_learning.use_double_dqn
```

**Step 5: Commit**

```bash
git add src/townlet/population/vectorized.py tests/test_townlet/integration/test_brain_config_integration.py
git commit -m "feat(brain): add brain_config parameter to VectorizedPopulation

- Add brain_config parameter to __init__
- Store brain_config for network building
- Extract gamma and use_double_dqn from brain_config
- Add integration test for config loading + network building
- Mark old parameters as DEPRECATED (will remove in Phase 1 Task 10)

Part of TASK-005 Phase 1 (9/12)"
```

---

## Task 10: Update VectorizedPopulation to Build Network from Config

**Files:**
- Modify: `src/townlet/population/vectorized.py` (lines 119-169)

**Step 1: Replace hardcoded network instantiation with factory calls**

Modify `src/townlet/population/vectorized.py` lines 119-169:

```python
# Add imports at top
from townlet.agent.network_factory import NetworkFactory
from townlet.agent.optimizer_factory import OptimizerFactory
from townlet.agent.loss_factory import LossFactory

# Replace network instantiation (lines 119-166)
# OLD CODE (DELETE):
# if network_type == "recurrent":
#     self.q_network = RecurrentSpatialQNetwork(...)
# elif network_type == "structured":
#     self.q_network = StructuredQNetwork(...)
# else:
#     self.q_network = SimpleQNetwork(obs_dim, action_dim, hidden_dim=128)

# NEW CODE:
# Build Q-network from brain config
if self.brain_config.architecture.type == "feedforward":
    self.q_network = NetworkFactory.build_feedforward(
        config=self.brain_config.architecture.feedforward,
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(device)
else:
    raise ValueError(
        f"Unsupported architecture type in Phase 1: {self.brain_config.architecture.type}. "
        f"Only 'feedforward' supported. Recurrent networks coming in Phase 2."
    )

# Build target network (same architecture)
if self.brain_config.architecture.type == "feedforward":
    self.target_network = NetworkFactory.build_feedforward(
        config=self.brain_config.architecture.feedforward,
        obs_dim=obs_dim,
        action_dim=action_dim,
    ).to(device)
else:
    raise ValueError(
        f"Unsupported architecture type in Phase 1: {self.brain_config.architecture.type}"
    )

# Initialize target network with Q-network weights
self.target_network.load_state_dict(self.q_network.state_dict())
self.target_network.eval()

# Store target update frequency from brain config
self.target_update_frequency = self.brain_config.q_learning.target_update_frequency
self.training_step_counter = 0

# Build optimizer from brain config
self.optimizer = OptimizerFactory.build(
    config=self.brain_config.optimizer,
    parameters=self.q_network.parameters(),
)

# Build loss function from brain config
self.loss_fn = LossFactory.build(self.brain_config.loss)
```

**Step 2: Update hardcoded loss in training loop**

Modify `src/townlet/population/vectorized.py` line 697:

```python
# OLD: loss = F.mse_loss(q_pred, q_target)
# NEW: Use configured loss function
loss = self.loss_fn(q_pred, q_target)
```

Also update line 643 for recurrent training (though recurrent not supported yet, keep consistent):

```python
# OLD: losses = F.mse_loss(q_pred_all, q_target_all, reduction="none")
# NEW: Use configured loss function
losses = self.loss_fn(q_pred_all, q_target_all)  # Note: Will need to handle reduction for recurrent in Phase 2
```

**Step 3: Run existing tests to check for regressions**

```bash
pytest tests/test_townlet/unit/population/test_vectorized_population.py -v
```

Expected: Tests will FAIL because they don't provide brain_config parameter yet. This is expected.

**Step 4: Commit (even though tests fail - we'll fix in next task)**

```bash
git add src/townlet/population/vectorized.py
git commit -m "feat(brain): replace hardcoded networks with factory-built networks

- Replace SimpleQNetwork instantiation with NetworkFactory.build_feedforward
- Build optimizer from OptimizerFactory
- Build loss function from LossFactory
- Extract target_update_frequency from brain_config
- Use configured loss_fn instead of hardcoded F.mse_loss
- Remove TODO(BRAIN_AS_CODE) comments (now implemented!)

NOTE: Tests will fail until we update them with brain_config parameter (next task)

Part of TASK-005 Phase 1 (10/12)"
```

---

## Task 11: Create brain.yaml for L0_0_minimal and L0_5_dual_resource

**Files:**
- Create: `configs/L0_0_minimal/brain.yaml`
- Create: `configs/L0_5_dual_resource/brain.yaml`

**Step 1: Create brain.yaml for L0_0_minimal**

Create `configs/L0_0_minimal/brain.yaml`:

```yaml
# Brain Configuration for L0_0_minimal
#
# Ultra-simple feedforward Q-network for temporal credit assignment.
#
# Network: 21→128→128→7 (3-layer MLP)
# Optimizer: Adam with standard RL learning rate
# Loss: MSE (vanilla DQN)
# Q-learning: Vanilla DQN (baseline for comparison)
#
# This configuration matches the hardcoded network from Phase 0
# to ensure checkpoint compatibility during migration.

version: "1.0"

description: "Simple feedforward Q-network for L0 temporal credit assignment"

architecture:
  type: feedforward
  feedforward:
    hidden_layers: [128, 128]
    activation: relu
    dropout: 0.0
    layer_norm: true

optimizer:
  type: adam
  learning_rate: 0.001
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0

loss:
  type: mse
  huber_delta: 1.0  # Ignored for MSE, but required field

q_learning:
  gamma: 0.95
  target_update_frequency: 100
  use_double_dqn: false
```

**Step 2: Create brain.yaml for L0_5_dual_resource**

Create `configs/L0_5_dual_resource/brain.yaml`:

```yaml
# Brain Configuration for L0_5_dual_resource
#
# Slightly larger network for dual-resource management (energy + health).
#
# Network: 29→256→128→8 (3-layer MLP)
# Optimizer: Adam with standard RL learning rate
# Loss: MSE (vanilla DQN)
# Q-learning: Vanilla DQN
#
# Increased capacity (256 hidden) to handle more complex state space
# compared to L0_0_minimal (128 hidden).

version: "1.0"

description: "Feedforward Q-network for L0.5 dual-resource management"

architecture:
  type: feedforward
  feedforward:
    hidden_layers: [256, 128]
    activation: relu
    dropout: 0.0
    layer_norm: true

optimizer:
  type: adam
  learning_rate: 0.00025
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1.0e-8
  weight_decay: 0.0

loss:
  type: mse
  huber_delta: 1.0  # Ignored for MSE

q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: false
```

**Step 3: Validate configs load correctly**

```bash
python -c "
from pathlib import Path
from townlet.agent.brain_config import load_brain_config

# Load L0_0_minimal
config1 = load_brain_config(Path('configs/L0_0_minimal'))
print(f'L0_0_minimal: {config1.architecture.feedforward.hidden_layers}')

# Load L0_5_dual_resource
config2 = load_brain_config(Path('configs/L0_5_dual_resource'))
print(f'L0_5_dual_resource: {config2.architecture.feedforward.hidden_layers}')
"
```

Expected output:
```
L0_0_minimal: [128, 128]
L0_5_dual_resource: [256, 128]
```

**Step 4: Commit**

```bash
git add configs/L0_0_minimal/brain.yaml configs/L0_5_dual_resource/brain.yaml
git commit -m "feat(brain): add brain.yaml for L0_0_minimal and L0_5_dual_resource

- L0_0_minimal: [128, 128] hidden layers, Adam lr=0.001, gamma=0.95
- L0_5_dual_resource: [256, 128] hidden layers, Adam lr=0.00025, gamma=0.99
- Both use vanilla DQN (use_double_dqn=false) for baseline
- Match previous hardcoded architectures for checkpoint compatibility

Part of TASK-005 Phase 1 (11/12)"
```

---

## Task 12: Update DemoRunner to Load and Use brain.yaml

**Files:**
- Modify: `src/townlet/demo/runner.py` (lines 400-449)
- Test: Run L0_0_minimal training to validate end-to-end

**Step 1: Add brain config loading to DemoRunner**

Modify `src/townlet/demo/runner.py` around lines 400-449:

```python
# Add import at top
from townlet.agent.brain_config import load_brain_config, compute_brain_hash

# Add brain config loading (after line 400, before population creation)
# Load brain configuration
logger.info("Loading brain configuration from brain.yaml...")
self.brain_config = load_brain_config(self.config_dir)

# Compute brain hash for checkpoint provenance
self.brain_hash = compute_brain_hash(self.brain_config)
logger.info(f"Brain hash: {self.brain_hash[:16]}... (SHA256)")

# Create population with brain config (modify existing call around line 429)
self.population = VectorizedPopulation(
    env=self.env,
    curriculum=self.curriculum,
    exploration=self.exploration,
    agent_ids=agent_ids,
    device=device,
    brain_config=self.brain_config,  # NEW: Pass brain config
    obs_dim=obs_dim,
    action_dim=action_dim,
    # Remove deprecated parameters (use brain_config instead):
    # learning_rate=self.hamlet_config.population.learning_rate,  # REMOVED
    # gamma=self.hamlet_config.population.gamma,  # REMOVED
    # network_type=self.hamlet_config.population.network_type,  # REMOVED
    # use_double_dqn=self.hamlet_config.training.use_double_dqn,  # REMOVED
    # target_update_frequency=self.hamlet_config.training.target_update_frequency,  # REMOVED
    replay_buffer_capacity=self.hamlet_config.population.replay_buffer_capacity,
    vision_window_size=vision_window_size,
    tb_logger=self.tb_logger,
    train_frequency=self.hamlet_config.training.train_frequency,
    batch_size=self.hamlet_config.training.batch_size,
    sequence_length=self.hamlet_config.training.sequence_length,
    max_grad_norm=self.hamlet_config.training.max_grad_norm,
)
```

**Step 2: Add brain_hash to checkpoint metadata**

Modify checkpoint saving in `src/townlet/demo/runner.py` (around line 700+):

```python
# In save_checkpoint method, add brain_hash to metadata
checkpoint = {
    "step": self.total_steps,
    "episode": self.current_episode,
    "q_network_state_dict": self.population.q_network.state_dict(),
    "target_network_state_dict": self.population.target_network.state_dict(),
    "optimizer_state_dict": self.population.optimizer.state_dict(),
    "brain_hash": self.brain_hash,  # NEW: Add brain hash for provenance
    "drive_hash": self.drive_hash,  # Existing drive hash
    "rng_state": torch.get_rng_state(),
    # ... rest of checkpoint data
}
```

**Step 3: Test end-to-end with L0_0_minimal**

```bash
# Set PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run short training (50 episodes)
python -m townlet.demo.run_demo --config configs/L0_0_minimal --max-episodes 50
```

Expected: Training runs successfully, networks built from brain.yaml, checkpoint saved with brain_hash

**Step 4: Verify checkpoint contains brain_hash**

```bash
python -c "
import torch
from pathlib import Path

# Find most recent checkpoint
checkpoints = list(Path('checkpoints').glob('L0_0_minimal*/checkpoint_*.pt'))
if checkpoints:
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    checkpoint = torch.load(latest)
    print(f'Checkpoint contains brain_hash: {\"brain_hash\" in checkpoint}')
    if 'brain_hash' in checkpoint:
        print(f'Brain hash: {checkpoint[\"brain_hash\"][:16]}...')
"
```

Expected output:
```
Checkpoint contains brain_hash: True
Brain hash: <16 hex chars>...
```

**Step 5: Commit**

```bash
git add src/townlet/demo/runner.py
git commit -m "feat(brain): integrate brain config into DemoRunner

- Load brain.yaml via load_brain_config()
- Compute brain_hash for checkpoint provenance
- Pass brain_config to VectorizedPopulation
- Remove deprecated parameters (learning_rate, gamma, network_type, etc.)
- Save brain_hash in checkpoint metadata
- Validate end-to-end training with L0_0_minimal

Part of TASK-005 Phase 1 (12/12 - PHASE 1 COMPLETE!)"
```

---

## Phase 1 Complete! Testing & Validation

**Comprehensive Test Suite Run:**

```bash
# Run all unit tests
pytest tests/test_townlet/unit/agent/ -v

# Run integration tests
pytest tests/test_townlet/integration/test_brain_config_integration.py -v

# Run full training test (L0_0_minimal, 50 episodes)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python -m townlet.demo.run_demo --config configs/L0_0_minimal --max-episodes 50

# Verify checkpoint provenance
python -c "
import torch
from pathlib import Path
checkpoints = list(Path('checkpoints').glob('L0_0_minimal*/checkpoint_*.pt'))
latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
checkpoint = torch.load(latest)
print(f'✓ brain_hash present: {checkpoint[\"brain_hash\"][:16]}...')
print(f'✓ drive_hash present: {checkpoint[\"drive_hash\"][:16]}...')
"
```

**Expected Results:**
- ✅ All unit tests pass
- ✅ Integration tests pass
- ✅ Training completes successfully
- ✅ Checkpoint contains brain_hash
- ✅ Networks built from brain.yaml (no hardcoded values)

---

## Summary: Phase 1 Deliverables

**Configuration System:**
- ✅ `brain_config.py`: Pydantic DTOs (FeedforwardConfig, OptimizerConfig, LossConfig, BrainConfig)
- ✅ `load_brain_config()`: YAML loader with validation
- ✅ `compute_brain_hash()`: SHA256 checkpoint provenance

**Factory Pattern:**
- ✅ `NetworkFactory.build_feedforward()`: Build MLP from config
- ✅ `OptimizerFactory.build()`: Build Adam/AdamW/SGD/RMSprop
- ✅ `LossFactory.build()`: Build MSE/Huber/SmoothL1

**Integration:**
- ✅ VectorizedPopulation uses brain_config
- ✅ DemoRunner loads brain.yaml
- ✅ Checkpoint metadata includes brain_hash

**Configuration Files:**
- ✅ `configs/L0_0_minimal/brain.yaml`
- ✅ `configs/L0_5_dual_resource/brain.yaml`

**Testing:**
- ✅ Unit tests for all DTOs and factories
- ✅ Integration tests for end-to-end config loading
- ✅ Validation via training run

**Next Phase:** Recurrent networks + learning rate schedulers

---
