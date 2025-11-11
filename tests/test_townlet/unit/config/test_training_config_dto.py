"""Tests for TrainingConfig DTO (Cycle 1)."""

import pytest
from pydantic import ValidationError

from townlet.config.training import TrainingConfig, load_training_config


class TestTrainingConfigValidation:
    """Test TrainingConfig schema validation (no-defaults principle)."""

    def test_all_fields_required(self):
        """All fields must be explicitly specified (no defaults)."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig()

        error = str(exc_info.value)
        # Check that key fields are mentioned as missing
        required_fields = ["device", "max_episodes", "epsilon_start", "epsilon_decay"]
        assert any(field in error for field in required_fields)

    def test_valid_config_minimal(self):
        """Valid config with all required fields loads successfully."""
        config = TrainingConfig(
            device="cuda",
            max_episodes=5000,
            train_frequency=4,
            target_update_frequency=100,
            batch_size=64,
            max_grad_norm=10.0,
            use_double_dqn=False,
            reward_strategy="multiplicative",
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            sequence_length=8,
        )
        assert config.device == "cuda"
        assert config.max_episodes == 5000
        assert config.epsilon_decay == 0.995
        assert config.sequence_length == 8

    def test_device_must_be_valid(self):
        """Device must be cpu, cuda, or mps."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                device="invalid_device",  # Not in Literal
                max_episodes=5000,
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                use_double_dqn=False,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                sequence_length=8,
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
                use_double_dqn=False,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                sequence_length=8,
            )

    def test_train_frequency_must_be_positive(self):
        """train_frequency must be > 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                device="cuda",
                max_episodes=5000,
                train_frequency=0,  # Must be gt=0
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                use_double_dqn=False,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                sequence_length=8,
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
                use_double_dqn=False,
                epsilon_start=1.5,  # Out of range
                epsilon_decay=0.995,
                epsilon_min=0.01,
                sequence_length=8,
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
                use_double_dqn=False,
                epsilon_start=1.0,
                epsilon_decay=1.0,  # Must be lt=1.0
                epsilon_min=0.01,
                sequence_length=8,
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
                use_double_dqn=False,
                reward_strategy="multiplicative",
                epsilon_start=0.01,  # Less than min
                epsilon_decay=0.995,
                epsilon_min=0.1,  # Greater than start
                sequence_length=8,
            )

        error = str(exc_info.value)
        assert "epsilon_start" in error
        assert "epsilon_min" in error


class TestDoubleDQNConfiguration:
    """Test Double DQN configuration field."""

    def test_use_double_dqn_field_required(self):
        """use_double_dqn must be explicitly specified (no defaults)."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                device="cuda",
                max_episodes=5000,
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                sequence_length=8,
                # Missing: use_double_dqn
            )

        error = str(exc_info.value)
        assert "use_double_dqn" in error.lower()

    def test_use_double_dqn_accepts_true(self):
        """use_double_dqn=True enables Double DQN."""
        config = TrainingConfig(
            device="cuda",
            max_episodes=5000,
            train_frequency=4,
            target_update_frequency=100,
            batch_size=64,
            max_grad_norm=10.0,
            reward_strategy="multiplicative",
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            sequence_length=8,
            use_double_dqn=True,
        )
        assert config.use_double_dqn is True

    def test_use_double_dqn_accepts_false(self):
        """use_double_dqn=False uses vanilla DQN."""
        config = TrainingConfig(
            device="cuda",
            max_episodes=5000,
            train_frequency=4,
            target_update_frequency=100,
            batch_size=64,
            max_grad_norm=10.0,
            reward_strategy="multiplicative",
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            sequence_length=8,
            use_double_dqn=False,
        )
        assert config.use_double_dqn is False

    def test_use_double_dqn_rejects_non_bool(self):
        """use_double_dqn must be bool, not invalid string."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                device="cuda",
                max_episodes=5000,
                train_frequency=4,
                target_update_frequency=100,
                batch_size=64,
                max_grad_norm=10.0,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                sequence_length=8,
                use_double_dqn="invalid",  # Invalid string that can't be coerced
            )

    def test_enabled_actions_must_be_unique(self):
        base_kwargs = dict(
            device="cuda",
            max_episodes=100,
            train_frequency=4,
            target_update_frequency=32,
            batch_size=16,
            max_grad_norm=10.0,
            use_double_dqn=False,
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.1,
            sequence_length=8,
        )

        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**base_kwargs, enabled_actions=["UP", "UP"])

        assert "duplicate" in str(exc_info.value)

    def test_enabled_actions_rejects_empty_strings(self):
        base_kwargs = dict(
            device="cuda",
            max_episodes=100,
            train_frequency=4,
            target_update_frequency=32,
            batch_size=16,
            max_grad_norm=10.0,
            use_double_dqn=False,
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.1,
            sequence_length=8,
        )

        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**base_kwargs, enabled_actions=["  "])

        assert "non-empty" in str(exc_info.value)


class TestTrainingConfigWarnings:
    """Test semantic warnings (not errors - permissive semantics)."""

    def test_slow_epsilon_decay_warning(self, caplog):
        """Warn if epsilon_decay is very slow (but allow it)."""
        import logging

        caplog.set_level(logging.WARNING)

        config = TrainingConfig(
            device="cuda",
            max_episodes=5000,
            train_frequency=4,
            target_update_frequency=100,
            batch_size=64,
            max_grad_norm=10.0,
            use_double_dqn=False,
            reward_strategy="multiplicative",
            epsilon_start=1.0,
            epsilon_decay=0.9995,  # Very slow
            epsilon_min=0.01,
            sequence_length=8,
        )

        # Should create config successfully (warning, not error)
        assert config.epsilon_decay == 0.9995

        # Should log warning
        assert any("epsilon_decay" in record.message for record in caplog.records)
        assert any("slow" in record.message.lower() for record in caplog.records)

    def test_fast_epsilon_decay_warning(self, caplog):
        """Warn if epsilon_decay is very fast (but allow it)."""
        import logging

        caplog.set_level(logging.WARNING)

        config = TrainingConfig(
            device="cuda",
            max_episodes=5000,
            train_frequency=4,
            target_update_frequency=100,
            batch_size=64,
            max_grad_norm=10.0,
            use_double_dqn=False,
            reward_strategy="multiplicative",
            epsilon_start=1.0,
            epsilon_decay=0.9,  # Very fast
            epsilon_min=0.01,
            sequence_length=8,
        )

        # Should create config successfully
        assert config.epsilon_decay == 0.9

        # Should log warning
        assert any("epsilon_decay" in record.message for record in caplog.records)
        assert any("fast" in record.message.lower() for record in caplog.records)


class TestTrainingConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        """Load TrainingConfig from YAML file."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
training:
  device: cuda
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  use_double_dqn: false
  reward_strategy: multiplicative
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
  sequence_length: 8
  enabled_actions:
    - "UP"
    - "WAIT"
"""
        )

        config = load_training_config(tmp_path)

        assert config.device == "cuda"
        assert config.max_episodes == 5000
        assert config.epsilon_decay == 0.995
        assert config.batch_size == 64
        assert config.enabled_actions == ["UP", "WAIT"]

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
training:
  device: cuda
  max_episodes: 5000
  # Missing epsilon params!
"""
        )

        with pytest.raises(ValueError) as exc_info:
            load_training_config(tmp_path)

        error = str(exc_info.value)
        assert "training.yaml" in error.lower() or "validation failed" in error.lower()

    def test_load_invalid_device_error(self, tmp_path):
        """Invalid device value raises clear error."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
training:
  device: invalid
  max_episodes: 5000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  use_double_dqn: false
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
  sequence_length: 8
"""
        )

        with pytest.raises(ValueError) as exc_info:
            load_training_config(tmp_path)

        error = str(exc_info.value)
        assert "device" in error.lower()

    def test_load_cpu_device(self, tmp_path):
        """Load config with CPU device."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
training:
  device: cpu
  max_episodes: 1000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 32
  max_grad_norm: 10.0
  use_double_dqn: false
  reward_strategy: multiplicative
  epsilon_start: 1.0
  epsilon_decay: 0.99
  epsilon_min: 0.01
  sequence_length: 8
"""
        )

        config = load_training_config(tmp_path)
        assert config.device == "cpu"
        assert config.max_episodes == 1000

    def test_load_mps_device(self, tmp_path):
        """Load config with MPS (Apple Silicon) device."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
training:
  device: mps
  max_episodes: 2000
  train_frequency: 4
  target_update_frequency: 100
  batch_size: 64
  max_grad_norm: 10.0
  use_double_dqn: false
  reward_strategy: multiplicative
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
  sequence_length: 8
"""
        )

        config = load_training_config(tmp_path)
        assert config.device == "mps"
