"""Tests for CascadeConfig DTO."""
import pytest
from pydantic import ValidationError


class TestCascadeConfigValidation:
    """Test CascadeConfig schema validation."""

    def test_all_required_fields_present(self):
        """All core fields must be specified."""
        from townlet.config.cascade import CascadeConfig

        # Valid config with all required fields
        config = CascadeConfig(
            name="satiation_to_health",
            description="Starvation makes you sick",
            source="satiation",
            target="health",
            threshold=0.3,
            strength=0.004,
        )
        assert config.name == "satiation_to_health"
        assert config.threshold == 0.3

    def test_missing_required_field(self):
        """Missing required field raises error."""
        from townlet.config.cascade import CascadeConfig

        with pytest.raises(ValidationError):
            CascadeConfig(
                name="test",
                # Missing description, source, target, threshold, strength
            )

    def test_name_must_be_nonempty(self):
        """Name cannot be empty string."""
        from townlet.config.cascade import CascadeConfig

        with pytest.raises(ValidationError):
            CascadeConfig(
                name="",  # Empty
                description="Test",
                source="satiation",
                target="health",
                threshold=0.3,
                strength=0.004,
            )

    def test_threshold_in_range(self):
        """Threshold must be in [0.0, 1.0]."""
        from townlet.config.cascade import CascadeConfig

        # Threshold below 0
        with pytest.raises(ValidationError):
            CascadeConfig(
                name="test",
                description="Test",
                source="satiation",
                target="health",
                threshold=-0.1,  # Invalid
                strength=0.004,
            )

        # Threshold above 1
        with pytest.raises(ValidationError):
            CascadeConfig(
                name="test",
                description="Test",
                source="satiation",
                target="health",
                threshold=1.5,  # Invalid
                strength=0.004,
            )

    def test_source_and_target_different(self):
        """Source and target meters must be different."""
        from townlet.config.cascade import CascadeConfig

        with pytest.raises(ValidationError) as exc_info:
            CascadeConfig(
                name="self_cascade",
                description="Test",
                source="energy",
                target="energy",  # Same as source!
                threshold=0.3,
                strength=0.004,
            )
        assert "source" in str(exc_info.value).lower() and "target" in str(exc_info.value).lower()

    def test_optional_fields(self):
        """Category and other fields are optional."""
        from townlet.config.cascade import CascadeConfig

        # Without category
        config1 = CascadeConfig(
            name="test",
            description="Test",
            source="satiation",
            target="health",
            threshold=0.3,
            strength=0.004,
        )
        assert config1.category is None

        # With category
        config2 = CascadeConfig(
            name="test",
            description="Test",
            source="satiation",
            target="health",
            threshold=0.3,
            strength=0.004,
            category="primary_to_pivotal",
        )
        assert config2.category == "primary_to_pivotal"


class TestCascadeConfigLoading:
    """Test loading CascadeConfig from YAML."""

    def test_load_from_yaml(self, tmp_path):
        """Load cascade configs from YAML file."""
        from townlet.config.cascade import load_cascades_config

        config_file = tmp_path / "cascades.yaml"
        config_file.write_text("""
version: "1.0"

cascades:
  - name: "satiation_to_health"
    description: "Starvation makes you sick"
    category: "primary_to_pivotal"
    source: "satiation"
    target: "health"
    threshold: 0.3
    strength: 0.004

  - name: "mood_to_energy"
    description: "Depression leads to exhaustion"
    category: "primary_to_pivotal"
    source: "mood"
    target: "energy"
    threshold: 0.3
    strength: 0.005
""")

        cascades = load_cascades_config(tmp_path)
        assert len(cascades) == 2
        assert cascades[0].name == "satiation_to_health"
        assert cascades[1].source == "mood"
        assert cascades[1].target == "energy"

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        from townlet.config.cascade import load_cascades_config

        config_file = tmp_path / "cascades.yaml"
        config_file.write_text("""
version: "1.0"

cascades:
  - name: "test"
    # Missing description, source, target, threshold, strength!
""")

        with pytest.raises(ValueError) as exc_info:
            load_cascades_config(tmp_path)

        error = str(exc_info.value)
        assert "cascades.yaml" in error.lower() or "validation" in error.lower()
