# TASK-003: TDD Implementation Plan
# UAC Contracts - Core DTOs

**Status**: READY FOR IMPLEMENTATION
**Created**: 2025-11-07
**Estimated Total Effort**: 9-13 hours (including plan creation)
**Approach**: Test-Driven Development (Red-Green-Refactor)

**Key Changes from Original Task**:
- SubstrateConfig marked as "✅ Already done (TASK-002A)"
- ActionConfig marked as "✅ Already done (TASK-002B)"
- Reduced scope from 10 DTOs to 8 DTOs
- Added incremental delivery strategy
- Added backward compatibility migration path

---

## Pre-Implementation Setup (30 minutes)

### Backup Strategy

```bash
# Backup existing configs (in case validation breaks them)
cp -r configs configs.backup-$(date +%Y%m%d)

# Create branch
git checkout -b task-003-uac-core-dtos

# Verify current configs load correctly
uv run python -m townlet.demo.runner --config configs/L0_0_minimal --max-episodes 1
uv run python -m townlet.demo.runner --config configs/L1_full_observability --max-episodes 1
```

### Directory Structure

```bash
mkdir -p src/townlet/config
mkdir -p tests/test_townlet/unit/config
mkdir -p tests/test_townlet/integration/config
mkdir -p configs/templates
```

---

## Cycle 0: Foundation & Validation Tooling (1-2 hours)

**Goal**: Create base infrastructure and validation patterns before DTOs

### RED Phase (30 minutes)

**Test**: `tests/test_townlet/unit/config/test_base_config.py`

```python
"""Base configuration testing patterns."""
import pytest
from pydantic import BaseModel, ValidationError, Field

def test_pydantic_imports_work():
    """Verify Pydantic is available and working."""
    class SimpleConfig(BaseModel):
        value: int = Field(gt=0)

    config = SimpleConfig(value=42)
    assert config.value == 42

    with pytest.raises(ValidationError):
        SimpleConfig(value=-1)  # Violates gt=0 constraint

def test_validation_error_messages():
    """Verify validation errors are clear."""
    class RequiredFieldConfig(BaseModel):
        required_field: str

    with pytest.raises(ValidationError) as exc_info:
        RequiredFieldConfig()

    error_msg = str(exc_info.value)
    assert "required_field" in error_msg
    assert "required" in error_msg.lower()
```

**Run**: `uv run pytest tests/test_townlet/unit/config/test_base_config.py`
**Expected**: ✅ PASS (Pydantic already installed)

### GREEN Phase (30 minutes)

**Deliverable**: `src/townlet/config/__init__.py`

```python
"""Configuration DTOs for UNIVERSE_AS_CODE validation.

Philosophy: All behavioral parameters must be explicitly specified.
No implicit defaults. Operator accountability.
"""
from pathlib import Path

# Version tracking for schema evolution
CONFIG_SCHEMA_VERSION = "1.0.0"

__all__ = [
    "CONFIG_SCHEMA_VERSION",
]
```

**Deliverable**: `src/townlet/config/base.py`

```python
"""Base configuration utilities."""
from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError

def load_yaml_section(config_dir: Path, filename: str, section: str) -> dict[str, Any]:
    """Load a section from a YAML file.

    Args:
        config_dir: Config pack directory
        filename: YAML filename (e.g., "training.yaml")
        section: Top-level section name (e.g., "training")

    Returns:
        Dict of configuration data

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If section doesn't exist
    """
    config_file = config_dir / filename
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Expected: {config_dir}/{filename}"
        )

    with open(config_file) as f:
        data = yaml.safe_load(f)

    if section not in data:
        raise KeyError(
            f"Section '{section}' not found in {filename}\n"
            f"Available sections: {list(data.keys())}"
        )

    return data[section]


def format_validation_error(error: ValidationError, context: str) -> str:
    """Format Pydantic ValidationError with helpful context.

    Args:
        error: Pydantic validation error
        context: Context string (e.g., "training.yaml")

    Returns:
        Formatted error message with fix suggestions
    """
    lines = [
        f"❌ {context} VALIDATION FAILED",
        "",
        str(error),
        "",
        "All parameters must be explicitly specified.",
        "See configs/templates/ for reference templates.",
    ]
    return "\n".join(lines)
```

**Test**: `tests/test_townlet/unit/config/test_base.py`

```python
"""Tests for base configuration utilities."""
import pytest
from pathlib import Path
from pydantic import BaseModel, ValidationError
from townlet.config.base import load_yaml_section, format_validation_error

def test_load_yaml_section_success(tmp_path):
    """Test loading valid YAML section."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
training:
  epsilon_start: 1.0
  epsilon_decay: 0.995
""")

    data = load_yaml_section(tmp_path, "test.yaml", "training")
    assert data["epsilon_start"] == 1.0
    assert data["epsilon_decay"] == 0.995

def test_load_yaml_section_missing_file(tmp_path):
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_yaml_section(tmp_path, "missing.yaml", "training")

    assert "missing.yaml" in str(exc_info.value)

def test_load_yaml_section_missing_section(tmp_path):
    """Test error when section doesn't exist."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("other_section: {}")

    with pytest.raises(KeyError) as exc_info:
        load_yaml_section(tmp_path, "test.yaml", "training")

    assert "training" in str(exc_info.value)
    assert "other_section" in str(exc_info.value)

def test_format_validation_error():
    """Test validation error formatting."""
    class TestConfig(BaseModel):
        required_field: int

    try:
        TestConfig()
    except ValidationError as e:
        formatted = format_validation_error(e, "test.yaml")
        assert "❌ test.yaml VALIDATION FAILED" in formatted
        assert "required_field" in formatted
        assert "templates" in formatted
```

**Run**: `uv run pytest tests/test_townlet/unit/config/`
**Expected**: ✅ ALL PASS

### REFACTOR Phase (30 minutes)

- Add type hints throughout
- Add docstrings for all functions
- Create `configs/templates/README.md` explaining template system

**COMMIT**: `feat(config): Add base configuration infrastructure and utilities`

---

## Cycle 1: TrainingConfig DTO (2-3 hours)

**Goal**: Create and validate training hyperparameters DTO

### RED Phase (45 minutes)

**Test**: `tests/test_townlet/unit/config/test_training_config.py`

```python
"""Tests for TrainingConfig DTO."""
import pytest
import math
from pydantic import ValidationError
from townlet.config.training import TrainingConfig

class TestTrainingConfigValidation:
    """Test TrainingConfig schema validation."""

    def test_all_fields_required(self):
        """All fields must be explicitly specified (no defaults)."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig()

        error = str(exc_info.value)
        # Check that key fields are mentioned as missing
        assert any(field in error for field in [
            "device", "max_episodes", "epsilon_start"
        ])

    def test_valid_config(self):
        """Valid config loads successfully."""
        config = TrainingConfig(
            device="cuda",
            max_episodes=5000,
            train_frequency=4,
            target_update_frequency=100,
            batch_size=64,
            max_grad_norm=10.0,
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
        )
        assert config.device == "cuda"
        assert config.epsilon_decay == 0.995

    def test_device_must_be_valid(self):
        """Device must be cpu, cuda, or mps."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                device="invalid",  # Not in Literal
                max_episodes=5000,
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
            )

    def test_max_episodes_must_be_positive(self):
        """max_episodes must be > 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                device="cuda",
                max_episodes=0,  # Must be gt=0
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
            )

    def test_epsilon_start_in_range(self):
        """epsilon_start must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                device="cuda",
                max_episodes=5000,
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                epsilon_start=1.5,  # Out of range
                epsilon_decay=0.995,
                epsilon_min=0.01,
            )

    def test_epsilon_decay_in_range(self):
        """epsilon_decay must be in (0.0, 1.0) - exclusive."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                device="cuda",
                max_episodes=5000,
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                epsilon_start=1.0,
                epsilon_decay=1.0,  # Must be lt=1.0
                epsilon_min=0.01,
            )

    def test_epsilon_order_validation(self):
        """epsilon_start must be >= epsilon_min."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                device="cuda",
                max_episodes=5000,
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                epsilon_start=0.01,  # Less than min
                epsilon_decay=0.995,
                epsilon_min=0.1,    # Greater than start
            )

        assert "epsilon_start" in str(exc_info.value)
        assert "epsilon_min" in str(exc_info.value)


class TestTrainingConfigWarnings:
    """Test semantic warnings (not errors)."""

    def test_slow_epsilon_decay_warning(self, caplog):
        """Warn if epsilon_decay is very slow."""
        config = TrainingConfig(
            device="cuda",
            max_episodes=5000,
            train_frequency=4,
            target_update_frequency=100,
            batch_size=64,
            max_grad_norm=10.0,
            epsilon_start=1.0,
            epsilon_decay=0.9995,  # Very slow
            epsilon_min=0.01,
        )

        # Should create config successfully (warning, not error)
        assert config.epsilon_decay == 0.9995

        # But should log warning
        # Note: Test will check if warning was logged


class TestTrainingConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        """Load TrainingConfig from YAML file."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
""")

        from townlet.config.training import load_training_config
        config = load_training_config(tmp_path)

        assert config.device == "cuda"
        assert config.max_episodes == 5000
        assert config.epsilon_decay == 0.995

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
training:
  device: cuda
  max_episodes: 5000
  # Missing epsilon params!
""")

        from townlet.config.training import load_training_config
        with pytest.raises(ValueError) as exc_info:
            load_training_config(tmp_path)

        error = str(exc_info.value)
        assert "training.yaml" in error
        assert "validation failed" in error.lower()
```

**Run**: `uv run pytest tests/test_townlet/unit/config/test_training_config.py`
**Expected**: ❌ FAIL (TrainingConfig doesn't exist yet)

### GREEN Phase (60 minutes)

**Deliverable**: `src/townlet/config/training.py`

```python
"""Training configuration DTO with no-defaults enforcement."""
from pathlib import Path
from typing import Literal
import math
import logging
from pydantic import BaseModel, Field, model_validator

from townlet.config.base import load_yaml_section, format_validation_error

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration.

    ALL FIELDS REQUIRED (no defaults) - enforces operator accountability.
    Operator must explicitly specify all parameters that affect training.

    Philosophy: If it affects the universe, it's in the config. No exceptions.
    """

    # Compute device (REQUIRED)
    device: Literal["cpu", "cuda", "mps"]

    # Training duration (REQUIRED)
    max_episodes: int = Field(gt=0, description="Total episodes to train")

    # Q-learning hyperparameters (ALL REQUIRED)
    train_frequency: int = Field(gt=0, description="Train Q-network every N steps")
    target_update_frequency: int = Field(
        gt=0, description="Update target network every N training steps"
    )
    batch_size: int = Field(gt=0, description="Experience replay batch size")
    max_grad_norm: float = Field(gt=0, description="Gradient clipping threshold")

    # Epsilon-greedy exploration (ALL REQUIRED)
    epsilon_start: float = Field(
        ge=0.0, le=1.0, description="Initial exploration rate (1.0 = 100% random)"
    )
    epsilon_decay: float = Field(
        gt=0.0, lt=1.0, description="Decay per episode (< 1.0)"
    )
    epsilon_min: float = Field(
        ge=0.0, le=1.0, description="Minimum exploration rate (floor)"
    )

    @model_validator(mode="after")
    def validate_epsilon_order(self) -> "TrainingConfig":
        """Ensure epsilon_start >= epsilon_min."""
        if self.epsilon_start < self.epsilon_min:
            raise ValueError(
                f"epsilon_start ({self.epsilon_start}) must be >= "
                f"epsilon_min ({self.epsilon_min})"
            )
        return self

    @model_validator(mode="after")
    def validate_epsilon_decay_speed(self) -> "TrainingConfig":
        """Warn (not error) if epsilon decay seems unreasonable.

        NOTE: This is a HINT, not enforcement. Operator may intentionally
        set slow decay for their experiment. We validate structure, not semantics.
        """
        # Calculate episodes to reach ε=0.1
        episodes_to_01 = math.log(0.1) / math.log(self.epsilon_decay)

        if self.epsilon_decay > 0.999:
            logger.warning(
                f"epsilon_decay={self.epsilon_decay} is very slow. "
                f"Will take {episodes_to_01:.0f} episodes to reach ε=0.1. "
                f"Typical values: 0.99 (L0 fast), 0.995 (L0.5/L1 moderate), "
                f"0.998 (L2 POMDP slow)."
            )
        elif self.epsilon_decay < 0.95:
            logger.warning(
                f"epsilon_decay={self.epsilon_decay} is very fast. "
                f"Will reach ε=0.1 in {episodes_to_01:.0f} episodes. "
                f"Agent may not explore enough."
            )

        return self


def load_training_config(config_dir: Path) -> TrainingConfig:
    """Load and validate training configuration.

    Args:
        config_dir: Directory containing training.yaml

    Returns:
        Validated TrainingConfig

    Raises:
        FileNotFoundError: If training.yaml not found
        ValueError: If validation fails (with helpful error message)
    """
    try:
        data = load_yaml_section(config_dir, "training.yaml", "training")
        return TrainingConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "training.yaml")) from e
```

**Run**: `uv run pytest tests/test_townlet/unit/config/test_training_config.py`
**Expected**: ✅ ALL PASS

### REFACTOR Phase (30 minutes)

- Extract epsilon validation to helper function
- Add more descriptive error messages
- Create `configs/templates/training.yaml.template` with annotated fields

**COMMIT**: `feat(config): Add TrainingConfig DTO with no-defaults validation`

---

## Cycle 2: EnvironmentConfig + PopulationConfig (2-3 hours)

**Goal**: Create environment and population DTOs

### RED Phase (45 minutes)

**Test**: `tests/test_townlet/unit/config/test_environment_config.py`

```python
"""Tests for EnvironmentConfig DTO."""
import pytest
from pydantic import ValidationError
from townlet.config.environment import EnvironmentConfig

class TestEnvironmentConfigValidation:
    def test_all_fields_required(self):
        """All fields must be explicitly specified."""
        with pytest.raises(ValidationError):
            EnvironmentConfig()

    def test_valid_config(self):
        """Valid config loads successfully."""
        config = EnvironmentConfig(
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enabled_affordances=["Bed", "Hospital", "Job"],
        )
        assert config.grid_size == 8
        assert len(config.enabled_affordances) == 3

    def test_grid_size_must_be_positive(self):
        """Grid size must be > 0."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(
                grid_size=0,
                partial_observability=False,
                vision_range=2,
                enabled_affordances=["Bed"],
            )

    def test_grid_size_reasonable_upper_bound(self):
        """Grid size should have reasonable upper bound."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(
                grid_size=1000,  # Too large
                partial_observability=False,
                vision_range=2,
                enabled_affordances=["Bed"],
            )

    def test_affordances_fit_in_grid(self):
        """Ensure enough grid cells for affordances + agent."""
        with pytest.raises(ValidationError) as exc_info:
            EnvironmentConfig(
                grid_size=3,  # 3×3 = 9 cells
                partial_observability=False,
                vision_range=2,
                enabled_affordances=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                # 10 affordances + 1 agent = 11 cells required > 9 available
            )

        error = str(exc_info.value)
        assert "too small" in error.lower() or "not enough" in error.lower()


class TestEnvironmentConfigWarnings:
    def test_vision_range_warning_when_full_observability(self, caplog):
        """Warn if vision_range specified but not used."""
        config = EnvironmentConfig(
            grid_size=8,
            partial_observability=False,  # Full obs
            vision_range=2,  # Won't be used
            enabled_affordances=["Bed"],
        )
        assert config.vision_range == 2  # Allowed but unused
```

**Test**: `tests/test_townlet/unit/config/test_population_config.py`

```python
"""Tests for PopulationConfig DTO."""
import pytest
from pydantic import ValidationError
from townlet.config.population import PopulationConfig

class TestPopulationConfigValidation:
    def test_all_fields_required(self):
        """All fields must be explicitly specified."""
        with pytest.raises(ValidationError):
            PopulationConfig()

    def test_valid_config(self):
        """Valid config loads successfully."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=10000,
            network_type="simple",
        )
        assert config.num_agents == 1
        assert config.network_type == "simple"

    def test_network_type_must_be_valid(self):
        """Network type must be 'simple' or 'recurrent'."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.00025,
                gamma=0.99,
                replay_buffer_capacity=10000,
                network_type="invalid",  # Not in Literal
            )

    def test_gamma_in_range(self):
        """Gamma must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.00025,
                gamma=1.5,  # Out of range
                replay_buffer_capacity=10000,
                network_type="simple",
            )
```

**Run**: `uv run pytest tests/test_townlet/unit/config/test_environment_config.py test_population_config.py`
**Expected**: ❌ FAIL (DTOs don't exist yet)

### GREEN Phase (75 minutes)

**Deliverable**: `src/townlet/config/environment.py`

```python
"""Environment configuration DTO."""
from pathlib import Path
import logging
from pydantic import BaseModel, Field, field_validator

from townlet.config.base import load_yaml_section, format_validation_error

logger = logging.getLogger(__name__)


class EnvironmentConfig(BaseModel):
    """Environment configuration.

    ALL FIELDS REQUIRED - no implicit defaults.
    """

    # Grid parameters (REQUIRED)
    grid_size: int = Field(gt=0, le=64, description="Grid dimensions (N×N)")

    # Observability (REQUIRED)
    partial_observability: bool = Field(description="POMDP (True) or full obs (False)")
    vision_range: int = Field(gt=0, description="Local window radius (for POMDP)")

    # Enabled affordances (REQUIRED)
    enabled_affordances: list[str] = Field(
        description="List of affordance IDs to deploy"
    )

    @field_validator("vision_range")
    @classmethod
    def validate_vision_range_usage(cls, v: int, info) -> int:
        """Warn if vision_range specified but partial_observability=False."""
        partial_obs = info.data.get("partial_observability", False)
        grid_size = info.data.get("grid_size", 0)

        if not partial_obs and v != grid_size:
            logger.warning(
                f"vision_range={v} specified but partial_observability=False. "
                f"Agent will see full grid regardless. "
                f"(This is allowed, but may be unintentional)"
            )

        return v

    @field_validator("enabled_affordances")
    @classmethod
    def validate_affordances_fit(cls, v: list[str], info) -> list[str]:
        """Ensure enough grid cells for affordances + agent."""
        grid_size = info.data.get("grid_size", 0)
        total_cells = grid_size * grid_size
        required_cells = len(v) + 1  # affordances + agent

        if required_cells > total_cells:
            raise ValueError(
                f"Grid too small: {grid_size}×{grid_size} = {total_cells} cells, "
                f"but need {required_cells} cells ({len(v)} affordances + 1 agent). "
                f"Increase grid_size or reduce enabled_affordances."
            )

        return v


def load_environment_config(config_dir: Path) -> EnvironmentConfig:
    """Load and validate environment configuration."""
    try:
        data = load_yaml_section(config_dir, "training.yaml", "environment")
        return EnvironmentConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "environment section")) from e
```

**Deliverable**: `src/townlet/config/population.py`

```python
"""Population configuration DTO."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

from townlet.config.base import load_yaml_section, format_validation_error


class PopulationConfig(BaseModel):
    """Population/agent configuration.

    ALL FIELDS REQUIRED - no implicit defaults.
    """

    # Agent count (REQUIRED)
    num_agents: int = Field(gt=0, description="Number of agents to train")

    # Q-learning parameters (ALL REQUIRED)
    learning_rate: float = Field(gt=0.0, description="Adam optimizer learning rate")
    gamma: float = Field(ge=0.0, le=1.0, description="Q-learning discount factor")
    replay_buffer_capacity: int = Field(
        gt=0, description="Experience replay buffer size"
    )

    # Network architecture (REQUIRED)
    network_type: Literal["simple", "recurrent"] = Field(
        description="MLP (simple) or LSTM (recurrent) Q-network"
    )


def load_population_config(config_dir: Path) -> PopulationConfig:
    """Load and validate population configuration."""
    try:
        data = load_yaml_section(config_dir, "training.yaml", "population")
        return PopulationConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "population section")) from e
```

**Run**: `uv run pytest tests/test_townlet/unit/config/`
**Expected**: ✅ ALL PASS

### REFACTOR Phase (30 minutes)

- Extract grid capacity check to utility function
- Add config templates for environment and population

**COMMIT**: `feat(config): Add EnvironmentConfig and PopulationConfig DTOs`

---

## Cycle 3: CurriculumConfig (1-2 hours)

**Goal**: Create curriculum progression DTO

### RED + GREEN + REFACTOR

**Test**: `tests/test_townlet/unit/config/test_curriculum_config.py`

```python
"""Tests for CurriculumConfig DTO."""
import pytest
from pydantic import ValidationError
from townlet.config.curriculum import CurriculumConfig

class TestCurriculumConfigValidation:
    def test_all_fields_required(self):
        """All fields must be explicitly specified."""
        with pytest.raises(ValidationError):
            CurriculumConfig()

    def test_valid_config(self):
        """Valid config loads successfully."""
        config = CurriculumConfig(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=1000,
        )
        assert config.max_steps_per_episode == 500

    def test_threshold_order_validation(self):
        """Retreat threshold must be < advance threshold."""
        with pytest.raises(ValidationError) as exc_info:
            CurriculumConfig(
                max_steps_per_episode=500,
                survival_advance_threshold=0.3,  # Lower than retreat!
                survival_retreat_threshold=0.7,  # Higher than advance!
                entropy_gate=0.5,
                min_steps_at_stage=1000,
            )

        error = str(exc_info.value)
        assert "retreat" in error.lower()
        assert "advance" in error.lower()
```

**Deliverable**: `src/townlet/config/curriculum.py`

```python
"""Curriculum configuration DTO."""
from pathlib import Path
from pydantic import BaseModel, Field, model_validator

from townlet.config.base import load_yaml_section, format_validation_error


class CurriculumConfig(BaseModel):
    """Adversarial curriculum configuration.

    ALL FIELDS REQUIRED - no implicit defaults.
    """

    # Stage parameters (ALL REQUIRED)
    max_steps_per_episode: int = Field(gt=0, description="Episode length limit")
    survival_advance_threshold: float = Field(
        ge=0.0, le=1.0, description="Advance to next stage if survival > threshold"
    )
    survival_retreat_threshold: float = Field(
        ge=0.0, le=1.0, description="Retreat to prev stage if survival < threshold"
    )
    entropy_gate: float = Field(
        ge=0.0, le=1.0, description="Minimum policy entropy to advance"
    )
    min_steps_at_stage: int = Field(
        gt=0, description="Minimum training steps before stage change"
    )

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "CurriculumConfig":
        """Ensure retreat < advance thresholds."""
        if self.survival_retreat_threshold >= self.survival_advance_threshold:
            raise ValueError(
                f"survival_retreat_threshold ({self.survival_retreat_threshold}) "
                f"must be < survival_advance_threshold ({self.survival_advance_threshold}). "
                f"Otherwise curriculum can't have stable intermediate stages."
            )
        return self


def load_curriculum_config(config_dir: Path) -> CurriculumConfig:
    """Load and validate curriculum configuration."""
    try:
        data = load_yaml_section(config_dir, "training.yaml", "curriculum")
        return CurriculumConfig(**data)
    except ValidationError as e:
        raise ValueError(format_validation_error(e, "curriculum section")) from e
```

**COMMIT**: `feat(config): Add CurriculumConfig DTO with threshold validation`

---

## Cycle 4: Basic DTOs (BarConfig, CascadeConfig, AffordanceConfig) (2-3 hours)

**Goal**: Create basic versions of existing DTOs (without advanced features)

**Note**: These DTOs already exist in some form. This cycle creates **no-defaults** versions.

### Deliverables

**`src/townlet/config/bar.py`** - Basic meter definitions
**`src/townlet/config/cascade.py`** - Basic cascade definitions
**`src/townlet/config/affordance.py`** - Basic affordance definitions (no capabilities)

**Scope**:
- Basic fields only
- No cross-file validation (defer to TASK-004A)
- No capabilities/effect pipelines (defer to TASK-004B)
- Focus on structural validation only

**COMMIT**: `feat(config): Add BarConfig, CascadeConfig, and AffordanceConfig basic DTOs`

---

## Cycle 5: HamletConfig Master DTO (1-2 hours)

**Goal**: Create master config that composes all sub-configs

### RED Phase

**Test**: `tests/test_townlet/integration/config/test_hamlet_config.py`

```python
"""Integration tests for HamletConfig master DTO."""
import pytest
from pathlib import Path
from pydantic import ValidationError
from townlet.config.hamlet import HamletConfig, load_hamlet_config

class TestHamletConfigComposition:
    def test_load_valid_config_directory(self, tmp_path):
        """Load complete config from directory."""
        # Create complete config pack
        (tmp_path / "training.yaml").write_text("""
training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01

environment:
  grid_size: 8
  partial_observability: false
  vision_range: 2
  enabled_affordances: ["Bed", "Hospital"]

population:
  num_agents: 1
  learning_rate: 0.00025
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: simple

curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000
""")

        config = load_hamlet_config(tmp_path)
        assert config.training.device == "cuda"
        assert config.environment.grid_size == 8
        assert config.population.num_agents == 1
        assert config.curriculum.max_steps_per_episode == 500

    def test_cross_config_validation(self, tmp_path):
        """Validate constraints across configs."""
        # Create config with batch_size > replay_buffer_capacity
        (tmp_path / "training.yaml").write_text("""
training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 20000  # Larger than replay buffer!
  max_grad_norm: 10.0
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01

population:
  num_agents: 1
  learning_rate: 0.00025
  gamma: 0.99
  replay_buffer_capacity: 10000  # Smaller than batch size!
  network_type: simple

environment:
  grid_size: 8
  partial_observability: false
  vision_range: 2
  enabled_affordances: ["Bed"]

curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000
""")

        with pytest.raises(ValueError) as exc_info:
            load_hamlet_config(tmp_path)

        error = str(exc_info.value)
        assert "batch_size" in error.lower()
        assert "replay_buffer" in error.lower()


class TestHamletConfigL0L1Loading:
    """Test loading existing L0/L1 configs (once updated)."""

    @pytest.mark.integration
    def test_load_l0_0_minimal(self):
        """Load L0_0_minimal config."""
        config_dir = Path("configs/L0_0_minimal")
        config = load_hamlet_config(config_dir)

        # Basic smoke test
        assert config.environment.grid_size == 3
        assert "Bed" in config.environment.enabled_affordances

    @pytest.mark.integration
    def test_load_l1_full_observability(self):
        """Load L1_full_observability config."""
        config_dir = Path("configs/L1_full_observability")
        config = load_hamlet_config(config_dir)

        assert config.environment.grid_size == 8
        assert len(config.environment.enabled_affordances) >= 10
```

**Run**: `uv run pytest tests/test_townlet/integration/config/test_hamlet_config.py`
**Expected**: ❌ FAIL (HamletConfig doesn't exist yet)

### GREEN Phase

**Deliverable**: `src/townlet/config/hamlet.py`

```python
"""HamletConfig master DTO composing all sub-configs."""
from pathlib import Path
import logging
from pydantic import BaseModel, model_validator

from townlet.config.training import TrainingConfig, load_training_config
from townlet.config.environment import EnvironmentConfig, load_environment_config
from townlet.config.population import PopulationConfig, load_population_config
from townlet.config.curriculum import CurriculumConfig, load_curriculum_config

logger = logging.getLogger(__name__)


class HamletConfig(BaseModel):
    """Master configuration composing all sub-configs.

    Philosophy: Universe fully specified by config files. No hidden state.
    """

    # Sub-configurations (ALL REQUIRED)
    training: TrainingConfig
    environment: EnvironmentConfig
    curriculum: CurriculumConfig
    population: PopulationConfig

    @model_validator(mode="after")
    def validate_cross_config_constraints(self) -> "HamletConfig":
        """Validate constraints across configs."""

        # Network type should match observability
        if (
            self.environment.partial_observability
            and self.population.network_type != "recurrent"
        ):
            logger.warning(
                f"partial_observability=True but network_type={self.population.network_type}. "
                f"Consider network_type='recurrent' for POMDP (memory for hidden state)."
            )

        # Batch size should be <= replay buffer capacity
        if self.training.batch_size > self.population.replay_buffer_capacity:
            raise ValueError(
                f"batch_size ({self.training.batch_size}) cannot exceed "
                f"replay_buffer_capacity ({self.population.replay_buffer_capacity}). "
                f"Can't sample {self.training.batch_size} experiences from buffer "
                f"of size {self.population.replay_buffer_capacity}."
            )

        return self


def load_hamlet_config(config_dir: Path) -> HamletConfig:
    """Load and compile universe configuration from directory.

    Args:
        config_dir: Directory containing training.yaml, etc.

    Returns:
        Validated HamletConfig

    Raises:
        FileNotFoundError: If required config files missing
        ValueError: If universe compilation fails

    Example:
        >>> config = load_hamlet_config(Path("configs/L0_0_minimal"))
        ✅ Universe compiled successfully
        >>> config.training.epsilon_decay
        0.99
    """
    try:
        config = HamletConfig(
            training=load_training_config(config_dir),
            environment=load_environment_config(config_dir),
            population=load_population_config(config_dir),
            curriculum=load_curriculum_config(config_dir),
        )
        logger.info("✅ Universe compiled successfully")
        return config
    except Exception as e:
        logger.error("❌ UNIVERSE COMPILATION FAILED")
        logger.error(str(e))
        raise
```

**Run**: `uv run pytest tests/test_townlet/integration/config/test_hamlet_config.py`
**Expected**: ✅ PASS

**COMMIT**: `feat(config): Add HamletConfig master DTO with cross-config validation`

---

## Cycle 6: Config Pack Updates (2-3 hours)

**Goal**: Update all L0-L3 configs to pass DTO validation

### Process

For each config pack (L0_0_minimal, L0_5_dual_resource, L1, L2, L3):

1. **Test current config**: `python -m townlet.config.validate configs/L0_0_minimal`
2. **Fix missing fields**: Add required parameters
3. **Verify fixes**: Run validator again
4. **Test training**: Run 1 episode to ensure nothing broken

**Deliverables**:
- Updated `configs/L0_0_minimal/training.yaml`
- Updated `configs/L0_5_dual_resource/training.yaml`
- Updated `configs/L1_full_observability/training.yaml`
- Updated `configs/L2_partial_observability/training.yaml`
- Updated `configs/L3_temporal_mechanics/training.yaml`

**COMMIT**: `fix(config): Update all config packs to pass DTO validation`

---

## Cycle 7: runner.py Integration (2-3 hours)

**Goal**: Replace dict access with DTO access in runner.py

### Strategy: Gradual Migration

**Phase 1**: Add DTO loading with fallback

```python
# Load config through DTOs
try:
    hamlet_config = load_hamlet_config(config_dir)
    use_dtos = True
except Exception as e:
    logger.warning(f"Failed to load DTOs, using legacy: {e}")
    use_dtos = False

# Use DTOs if available, fall back to legacy
if use_dtos:
    epsilon_start = hamlet_config.training.epsilon_start
else:
    epsilon_start = training_cfg.get("epsilon_start", 1.0)
```

**Phase 2**: Remove fallbacks (after all configs pass)

```python
# Load config through DTOs (must succeed)
hamlet_config = load_hamlet_config(config_dir)
epsilon_start = hamlet_config.training.epsilon_start
```

**COMMIT**: `refactor(runner): Use DTOs for config loading (BREAKING)`

---

## Cycle 8: CI Integration & Documentation (1-2 hours)

**Goal**: Add config validation to CI and update docs

### Deliverables

1. **CI validation script**: `scripts/validate_configs.py`
2. **GitHub workflow**: `.github/workflows/validate-configs.yml`
3. **Documentation**: Update CLAUDE.md with DTO usage
4. **Config templates**: Complete all `.template` files

**COMMIT**: `chore(ci): Add config validation to CI pipeline`

---

## Success Criteria

- [ ] All 8 DTOs created (Training, Environment, Population, Curriculum, Bar, Cascade, Affordance, Hamlet)
- [ ] No-defaults principle enforced (all fields required)
- [ ] 40+ unit tests passing (schema validation, validators, edge cases)
- [ ] 10+ integration tests passing (L0-L3 config loading)
- [ ] All L0-L3 configs load through DTOs
- [ ] runner.py uses DTOs exclusively (no dict access)
- [ ] CI validates all configs automatically
- [ ] Config templates created for all DTOs
- [ ] Documentation updated (CLAUDE.md, config-schemas/)
- [ ] Zero backward compatibility code remains

---

## Estimated Effort

| Cycle | Description | Estimate |
|-------|-------------|----------|
| 0 | Foundation & tooling | 1-2h |
| 1 | TrainingConfig DTO | 2-3h |
| 2 | Environment + Population DTOs | 2-3h |
| 3 | CurriculumConfig DTO | 1-2h |
| 4 | Basic DTOs (Bar, Cascade, Affordance) | 2-3h |
| 5 | HamletConfig master | 1-2h |
| 6 | Config pack updates | 2-3h |
| 7 | runner.py integration | 2-3h |
| 8 | CI & documentation | 1-2h |
| **Total** | | **14-23h** |

**Includes**: Plan creation (1h), testing time, refactoring, documentation

**Note**: Upper bound accounts for discovering issues in existing configs, debugging integration, and comprehensive testing.

---

## Risk Mitigation

**Risk: Breaking existing training runs**
- Mitigation: Incremental delivery (DTOs → config fixes → runner integration)

**Risk: Scope creep into TASK-004A**
- Mitigation: Explicitly defer cross-file validation, strict boundaries

**Risk: Config fixes uncover deeper issues**
- Mitigation: Fix incrementally, commit often, test after each fix

**Risk: runner.py refactor more complex than expected**
- Mitigation: Gradual migration with fallback, separate commits per section

---

## Post-Implementation

After implementation complete:

1. **Create PR** with comprehensive description
2. **Request review** focusing on:
   - No-defaults enforcement
   - Validator logic correctness
   - Error message clarity
   - Cross-config constraint validation
3. **Merge to main**
4. **Mark TASK-003 complete** in task tracker
5. **Update TASK-004A status** (unblocked)

---

**Status**: READY FOR IMPLEMENTATION
**Confidence**: HIGH (based on TASK-002A/B/C TDD success)
**Next Action**: Begin Cycle 0 (Foundation)
