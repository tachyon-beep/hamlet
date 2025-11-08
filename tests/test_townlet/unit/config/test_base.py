"""Tests for base configuration utilities."""


import pytest
from pydantic import BaseModel, ValidationError

from townlet.config.base import format_validation_error, load_yaml_section


class TestLoadYamlSection:
    """Test YAML section loading utility."""

    def test_load_valid_section(self, tmp_path):
        """Load valid YAML section successfully."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
training:
  epsilon_start: 1.0
  epsilon_decay: 0.995
  max_episodes: 5000

environment:
  grid_size: 8
"""
        )

        data = load_yaml_section(tmp_path, "test.yaml", "training")
        assert data["epsilon_start"] == 1.0
        assert data["epsilon_decay"] == 0.995
        assert data["max_episodes"] == 5000

    def test_load_different_section(self, tmp_path):
        """Load different section from same file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
training:
  epsilon_start: 1.0

environment:
  grid_size: 8
  partial_observability: false
"""
        )

        data = load_yaml_section(tmp_path, "test.yaml", "environment")
        assert data["grid_size"] == 8
        assert data["partial_observability"] is False

    def test_missing_file_error(self, tmp_path):
        """Raise clear error when file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_yaml_section(tmp_path, "missing.yaml", "training")

        error_msg = str(exc_info.value)
        assert "missing.yaml" in error_msg
        assert "not found" in error_msg.lower()
        assert str(tmp_path) in error_msg

    def test_missing_section_error(self, tmp_path):
        """Raise clear error when section doesn't exist."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
environment:
  grid_size: 8

population:
  num_agents: 1
"""
        )

        with pytest.raises(KeyError) as exc_info:
            load_yaml_section(tmp_path, "test.yaml", "training")

        error_msg = str(exc_info.value)
        assert "training" in error_msg
        assert "environment" in error_msg  # Shows available sections
        assert "population" in error_msg

    def test_empty_file_error(self, tmp_path):
        """Raise error when file is empty."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError) as exc_info:
            load_yaml_section(tmp_path, "empty.yaml", "training")

        assert "empty" in str(exc_info.value).lower()

    def test_nested_section_data(self, tmp_path):
        """Load section with nested data structures."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
training:
  device: cuda
  hyperparameters:
    learning_rate: 0.001
    gamma: 0.99
  enabled_affordances:
    - Bed
    - Hospital
"""
        )

        data = load_yaml_section(tmp_path, "test.yaml", "training")
        assert data["device"] == "cuda"
        assert data["hyperparameters"]["learning_rate"] == 0.001
        assert "Bed" in data["enabled_affordances"]


class TestFormatValidationError:
    """Test validation error formatting utility."""

    def test_format_includes_context(self):
        """Formatted error includes context string."""

        class TestConfig(BaseModel):
            required_field: int

        try:
            TestConfig()  # Missing required field
        except ValidationError as e:
            formatted = format_validation_error(e, "test.yaml")
            # Check context appears with exact casing preserved
            assert "test.yaml" in formatted
            assert "VALIDATION FAILED" in formatted

    def test_format_includes_error_details(self):
        """Formatted error includes Pydantic error details."""

        class TestConfig(BaseModel):
            required_field: int

        try:
            TestConfig()
        except ValidationError as e:
            formatted = format_validation_error(e, "test.yaml")
            assert "required_field" in formatted
            assert "required" in formatted.lower()

    def test_format_includes_helpful_guidance(self):
        """Formatted error includes guidance on how to fix."""

        class TestConfig(BaseModel):
            required_field: int

        try:
            TestConfig()
        except ValidationError as e:
            formatted = format_validation_error(e, "test.yaml")
            assert "templates" in formatted.lower()
            assert "specified" in formatted.lower()
            assert "no-defaults" in formatted.lower()

    def test_format_with_type_error(self):
        """Format error for type mismatches."""

        class TestConfig(BaseModel):
            number_field: int

        try:
            TestConfig(number_field="not_a_number")
        except ValidationError as e:
            formatted = format_validation_error(e, "training.yaml")
            # Check context appears with exact casing preserved
            assert "training.yaml" in formatted
            assert "number_field" in formatted

    def test_format_with_range_error(self):
        """Format error for range constraint violations."""
        from pydantic import Field

        class TestConfig(BaseModel):
            probability: float = Field(ge=0.0, le=1.0)

        try:
            TestConfig(probability=1.5)
        except ValidationError as e:
            formatted = format_validation_error(e, "config section")
            # Check context appears with exact casing preserved
            assert "config section" in formatted
            assert "probability" in formatted
