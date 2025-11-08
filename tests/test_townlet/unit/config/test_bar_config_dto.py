"""Tests for BarConfig DTO."""

import pytest
from pydantic import ValidationError


class TestBarConfigValidation:
    """Test BarConfig schema validation."""

    def test_all_required_fields_present(self):
        """All core fields must be specified."""
        from townlet.config.bar import BarConfig

        # Valid config with all required fields
        config = BarConfig(
            name="energy",
            index=0,
            tier="pivotal",
            range=[0.0, 1.0],
            initial=1.0,
            base_depletion=0.005,
        )
        assert config.name == "energy"
        assert config.initial == 1.0

    def test_missing_required_field(self):
        """Missing required field raises error."""
        from townlet.config.bar import BarConfig

        with pytest.raises(ValidationError):
            BarConfig(
                name="energy",
                # Missing index, tier, range, initial, base_depletion
            )

    def test_name_must_be_nonempty(self):
        """Name cannot be empty string."""
        from townlet.config.bar import BarConfig

        with pytest.raises(ValidationError):
            BarConfig(
                name="",  # Empty
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.005,
            )

    def test_range_must_have_two_elements(self):
        """Range must be [min, max] with exactly 2 elements."""
        from townlet.config.bar import BarConfig

        # Too few elements
        with pytest.raises(ValidationError):
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0],  # Only 1 element
                initial=1.0,
                base_depletion=0.005,
            )

        # Too many elements
        with pytest.raises(ValidationError):
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 0.5, 1.0],  # 3 elements
                initial=1.0,
                base_depletion=0.005,
            )

    def test_initial_must_be_in_range(self):
        """Initial value must be within [min, max]."""
        from townlet.config.bar import BarConfig

        # Initial below min
        with pytest.raises(ValidationError) as exc_info:
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=-0.5,  # Below min
                base_depletion=0.005,
            )
        assert "initial" in str(exc_info.value).lower()

        # Initial above max
        with pytest.raises(ValidationError) as exc_info:
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.5,  # Above max
                base_depletion=0.005,
            )
        assert "initial" in str(exc_info.value).lower()

    def test_range_min_must_be_less_than_max(self):
        """Range min must be < max."""
        from townlet.config.bar import BarConfig

        with pytest.raises(ValidationError) as exc_info:
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[1.0, 0.0],  # min > max (reversed)
                initial=0.5,
                base_depletion=0.005,
            )
        assert "range" in str(exc_info.value).lower()

    def test_optional_description_field(self):
        """Description field is optional."""
        from townlet.config.bar import BarConfig

        # Without description
        config1 = BarConfig(
            name="energy",
            index=0,
            tier="pivotal",
            range=[0.0, 1.0],
            initial=1.0,
            base_depletion=0.005,
        )
        assert config1.description is None

        # With description
        config2 = BarConfig(
            name="energy",
            index=0,
            tier="pivotal",
            range=[0.0, 1.0],
            initial=1.0,
            base_depletion=0.005,
            description="Energy meter",
        )
        assert config2.description == "Energy meter"


class TestBarConfigLoading:
    """Test loading BarConfig from YAML."""

    def test_load_from_yaml(self, tmp_path):
        """Load bar configs from YAML file."""
        from townlet.config.bar import load_bars_config

        config_file = tmp_path / "bars.yaml"
        config_file.write_text(
            """
version: "1.0"

bars:
  - name: "energy"
    index: 0
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.005
    description: "Energy level"

  - name: "health"
    index: 1
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.0
    description: "Health level"
"""
        )

        bars = load_bars_config(tmp_path)
        assert len(bars) == 2
        assert bars[0].name == "energy"
        assert bars[1].name == "health"
        assert bars[0].initial == 1.0

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        from townlet.config.bar import load_bars_config

        config_file = tmp_path / "bars.yaml"
        config_file.write_text(
            """
version: "1.0"

bars:
  - name: "energy"
    # Missing index, tier, range, initial, base_depletion!
"""
        )

        with pytest.raises(ValueError) as exc_info:
            load_bars_config(tmp_path)

        error = str(exc_info.value)
        assert "bars.yaml" in error.lower() or "validation" in error.lower()
