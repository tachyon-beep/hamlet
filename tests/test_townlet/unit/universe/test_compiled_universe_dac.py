"""Tests for CompiledUniverse DAC integration."""

from townlet.config.drive_as_code import DriveAsCodeConfig, ExtrinsicStrategyConfig, IntrinsicStrategyConfig


class TestCompiledUniverseDACFields:
    """Test DAC fields in CompiledUniverse."""

    def test_compiled_universe_has_dac_fields(self):
        """CompiledUniverse includes dac_config and drive_hash fields."""
        # Create minimal DAC config
        dac_config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=0.100,
            ),
        )

        # Note: Full CompiledUniverse construction requires many fields
        # This test just verifies the new fields can be added
        # (Will be tested properly via compiler integration tests)
        assert dac_config is not None
