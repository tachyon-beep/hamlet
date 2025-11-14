"""Tests for AffordanceConfig DTO."""

import pytest
from pydantic import ValidationError


class TestAffordanceConfigValidation:
    """Test AffordanceConfig schema validation."""

    def test_all_required_fields_present(self):
        from townlet.config.affordance import AffordanceConfig
        from townlet.config.effect_pipeline import AffordanceEffect, EffectPipeline

        config = AffordanceConfig(
            id="0",
            name="Bed",
            category="energy_restoration",
            operating_hours=[0, 24],
            costs=[{"meter": "money", "amount": 0.05}],
            effect_pipeline=EffectPipeline(on_completion=[AffordanceEffect(meter="energy", amount=0.50)]),
        )
        assert config.id == "0"
        assert config.effect_pipeline is not None

    def test_missing_required_field(self):
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(id="0")

    def test_empty_effect_pipeline_is_valid(self):
        """Empty EffectPipeline is valid - effects are optional."""
        from townlet.config.affordance import AffordanceConfig
        from townlet.config.effect_pipeline import EffectPipeline

        # This should not raise - empty effects are valid
        config = AffordanceConfig(id="0", name="Bed", operating_hours=[0, 24], effect_pipeline=EffectPipeline())
        assert config.effect_pipeline is not None
        assert not config.effect_pipeline.has_effects()

    def test_effect_pipeline_stages(self):
        from townlet.config.affordance import AffordanceConfig
        from townlet.config.effect_pipeline import AffordanceEffect, EffectPipeline

        config = AffordanceConfig(
            id="staged",
            name="StagedEffects",
            operating_hours=[0, 24],
            effect_pipeline=EffectPipeline(
                on_completion=[AffordanceEffect(meter="energy", amount=0.2)],
                per_tick=[AffordanceEffect(meter="money", amount=0.1)],
            ),
        )
        assert config.effect_pipeline is not None
        assert len(config.effect_pipeline.on_completion) == 1
        assert config.effect_pipeline.per_tick[0].meter == "money"

    def test_operating_hours_validation(self):
        from townlet.config.affordance import AffordanceConfig
        from townlet.config.effect_pipeline import AffordanceEffect, EffectPipeline

        config = AffordanceConfig(
            id="1",
            name="Shop",
            operating_hours=[9, 17],
            effect_pipeline=EffectPipeline(on_completion=[AffordanceEffect(meter="money", amount=1.0)]),
        )
        assert config.operating_hours == [9, 17]

        with pytest.raises(ValidationError):
            AffordanceConfig(
                id="2",
                name="NightClub",
                operating_hours=[-1, 30],
                effect_pipeline=EffectPipeline(on_completion=[AffordanceEffect(meter="mood", amount=0.1)]),
            )

    def test_operating_hours_required(self):
        """BUG-30: operating_hours must be required field (no-defaults principle)."""
        from townlet.config.affordance import AffordanceConfig

        # Missing operating_hours should fail validation
        with pytest.raises(ValidationError, match="operating_hours"):
            AffordanceConfig(id="missing_hours", name="MissingHours")

        # Explicit None should also fail
        with pytest.raises(ValidationError, match="operating_hours"):
            AffordanceConfig(id="none_hours", name="NoneHours", operating_hours=None)

        # For 24/7 availability, operators must explicitly specify [0, 24]
        config_247 = AffordanceConfig(id="always_open", name="AlwaysOpen", operating_hours=[0, 24])
        assert config_247.operating_hours == [0, 24]

    def test_capabilities_parsed(self):
        from townlet.config.affordance import AffordanceConfig
        from townlet.config.effect_pipeline import AffordanceEffect, EffectPipeline

        config = AffordanceConfig(
            id="job",
            name="Job",
            operating_hours=[9, 17],
            duration_ticks=5,
            capabilities=[
                {"type": "multi_tick"},
                {"type": "meter_gated", "meter": "energy", "min": 0.3},
            ],
            effect_pipeline=EffectPipeline(on_completion=[AffordanceEffect(meter="money", amount=1.0)]),
        )
        assert len(config.capabilities) == 2
        assert config.capabilities[0].type == "multi_tick"

    def test_availability_constraints_require_bounds(self):
        from townlet.config.affordance import AffordanceConfig
        from townlet.config.effect_pipeline import AffordanceEffect, EffectPipeline

        with pytest.raises(ValidationError):
            AffordanceConfig(
                id="gym",
                name="Gym",
                operating_hours=[6, 22],
                availability=[{"meter": "energy"}],
                effect_pipeline=EffectPipeline(on_completion=[AffordanceEffect(meter="fitness", amount=0.1)]),
            )

    def test_id_and_name_cannot_be_empty(self):
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(id="", name="Bed", operating_hours=[0, 24])

        with pytest.raises(ValidationError):
            AffordanceConfig(id="123", name="", operating_hours=[0, 24])

    def test_negative_effects_and_costs_supported(self):
        from townlet.config.affordance import AffordanceConfig
        from townlet.config.effect_pipeline import AffordanceEffect, EffectPipeline

        config = AffordanceConfig(
            id="fastfood",
            name="FastFood",
            operating_hours=[0, 24],
            costs=[{"meter": "money", "amount": 0.1}],
            effect_pipeline=EffectPipeline(
                on_completion=[
                    AffordanceEffect(meter="satiation", amount=0.45),
                    AffordanceEffect(meter="health", amount=-0.02),
                ]
            ),
        )
        assert config.effect_pipeline.on_completion[1].amount == -0.02


class TestAffordanceConfigLoading:
    """Test loading AffordanceConfig from YAML."""

    def test_load_from_yaml(self, tmp_path):
        """Load affordance configs from YAML file."""
        from townlet.config.affordance import load_affordances_config

        config_file = tmp_path / "affordances.yaml"
        config_file.write_text(
            """
version: "1.0"

affordances:
  - id: "0"
    name: "Bed"
    category: "energy_restoration"
    operating_hours: [0, 24]
    costs:
      - { meter: "money", amount: 0.05 }
    effect_pipeline:
      on_completion:
        - { meter: "energy", amount: 0.50 }
        - { meter: "health", amount: 0.02 }

  - id: "10"
    name: "Job"
    category: "income"
    operating_hours: [9, 17]
    costs: []
    effect_pipeline:
      on_completion:
        - { meter: "money", amount: 0.225 }
        - { meter: "energy", amount: -0.15 }
"""
        )

        affordances = load_affordances_config(tmp_path)
        assert len(affordances) == 2
        assert affordances[0].name == "Bed"
        assert affordances[1].id == "10"
        assert len(affordances[0].effect_pipeline.on_completion) == 2

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        from townlet.config.affordance import load_affordances_config

        config_file = tmp_path / "affordances.yaml"
        config_file.write_text(
            """
version: "1.0"

affordances:
  - id: "0"
    # Missing name, costs, effects!
"""
        )

        with pytest.raises(ValueError) as exc_info:
            load_affordances_config(tmp_path)

        error = str(exc_info.value)
        assert "affordances.yaml" in error.lower() or "validation" in error.lower()
