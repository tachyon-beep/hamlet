"""
Tests for MeterDynamics integration with CascadeEngine.

Verifies that config-driven CascadeEngine produces identical results
to the legacy hardcoded implementation (zero behavioral change).
"""

import pytest
import torch

from townlet.environment.meter_dynamics import MeterDynamics


@pytest.fixture
def device():
    """Get compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestMeterDynamicsEquivalence:
    """Test that CascadeEngine mode produces identical results to legacy mode."""

    def test_deplete_meters_equivalence(self, device):
        """Test that deplete_meters produces identical results in both modes."""
        # Create instances in both modes
        md_legacy = MeterDynamics(3, device, use_cascade_engine=False)
        md_config = MeterDynamics(3, device, use_cascade_engine=True)

        # Test meters: healthy, low satiation, low fitness
        meters = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Healthy
                [0.8, 0.9, 0.2, 0.5, 0.8, 0.7, 1.0, 0.9],  # Low satiation
                [0.8, 0.9, 0.8, 0.5, 0.8, 0.7, 1.0, 0.1],  # Low fitness
            ],
            device=device,
        )

        # Apply deplete_meters
        meters_legacy = md_legacy.deplete_meters(meters.clone())
        meters_config = md_config.deplete_meters(meters.clone())

        # Should be identical within floating point tolerance
        assert torch.allclose(meters_legacy, meters_config, atol=1e-5), (
            f"deplete_meters mismatch:\nLegacy: {meters_legacy}\nConfig: {meters_config}"
        )

    def test_secondary_to_primary_effects_equivalence(self, device):
        """Test secondary→primary cascade equivalence."""
        md_legacy = MeterDynamics(2, device, use_cascade_engine=False)
        md_config = MeterDynamics(2, device, use_cascade_engine=True)

        # Low satiation (triggers cascade)
        meters = torch.tensor(
            [
                [1.0, 1.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0],  # Low satiation
                [0.8, 0.9, 0.1, 0.5, 0.2, 0.7, 1.0, 0.9],  # Low satiation + mood
            ],
            device=device,
        )

        meters_legacy = md_legacy.apply_secondary_to_primary_effects(meters.clone())
        meters_config = md_config.apply_secondary_to_primary_effects(meters.clone())

        assert torch.allclose(meters_legacy, meters_config, atol=1e-5), (
            f"secondary_to_primary mismatch:\nLegacy: {meters_legacy}\nConfig: {meters_config}"
        )

    def test_tertiary_to_secondary_effects_equivalence(self, device):
        """Test tertiary→secondary cascade equivalence."""
        md_legacy = MeterDynamics(2, device, use_cascade_engine=False)
        md_config = MeterDynamics(2, device, use_cascade_engine=True)

        # Low hygiene (triggers cascade)
        meters = torch.tensor(
            [
                [1.0, 0.2, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Low hygiene
                [0.8, 0.1, 0.8, 0.5, 0.8, 0.2, 1.0, 0.9],  # Low hygiene + social
            ],
            device=device,
        )

        meters_legacy = md_legacy.apply_tertiary_to_secondary_effects(meters.clone())
        meters_config = md_config.apply_tertiary_to_secondary_effects(meters.clone())

        assert torch.allclose(meters_legacy, meters_config, atol=1e-5), (
            f"tertiary_to_secondary mismatch:\nLegacy: {meters_legacy}\nConfig: {meters_config}"
        )

    def test_tertiary_to_primary_effects_equivalence(self, device):
        """Test tertiary→primary weak cascade equivalence."""
        md_legacy = MeterDynamics(2, device, use_cascade_engine=False)
        md_config = MeterDynamics(2, device, use_cascade_engine=True)

        # Low hygiene + social (triggers weak cascade)
        meters = torch.tensor(
            [
                [1.0, 0.2, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Low hygiene
                [0.8, 0.1, 0.8, 0.5, 0.8, 0.2, 1.0, 0.9],  # Low hygiene + social
            ],
            device=device,
        )

        meters_legacy = md_legacy.apply_tertiary_to_primary_effects(meters.clone())
        meters_config = md_config.apply_tertiary_to_primary_effects(meters.clone())

        assert torch.allclose(meters_legacy, meters_config, atol=1e-5), (
            f"tertiary_to_primary mismatch:\nLegacy: {meters_legacy}\nConfig: {meters_config}"
        )

    def test_full_cascade_sequence_equivalence(self, device):
        """Test complete cascade sequence equivalence."""
        md_legacy = MeterDynamics(4, device, use_cascade_engine=False)
        md_config = MeterDynamics(4, device, use_cascade_engine=True)

        # Diverse agent states
        meters = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # Healthy
                [0.8, 0.2, 0.2, 0.5, 0.8, 0.7, 1.0, 0.9],  # Low hygiene + satiation
                [0.8, 0.9, 0.8, 0.5, 0.2, 0.2, 1.0, 0.1],  # Low mood + social + fitness
                [0.5, 0.1, 0.1, 0.3, 0.1, 0.1, 0.8, 0.1],  # Everything low
            ],
            device=device,
        )

        # Apply full cascade sequence (as in vectorized_env)
        meters_legacy = meters.clone()
        meters_legacy = md_legacy.deplete_meters(meters_legacy)
        meters_legacy = md_legacy.apply_secondary_to_primary_effects(meters_legacy)
        meters_legacy = md_legacy.apply_tertiary_to_secondary_effects(meters_legacy)
        meters_legacy = md_legacy.apply_tertiary_to_primary_effects(meters_legacy)

        meters_config = meters.clone()
        meters_config = md_config.deplete_meters(meters_config)
        meters_config = md_config.apply_secondary_to_primary_effects(meters_config)
        meters_config = md_config.apply_tertiary_to_secondary_effects(meters_config)
        meters_config = md_config.apply_tertiary_to_primary_effects(meters_config)

        # Full sequence should produce identical results
        assert torch.allclose(meters_legacy, meters_config, atol=1e-5), (
            f"Full cascade sequence mismatch:\n"
            f"Legacy: {meters_legacy}\n"
            f"Config: {meters_config}\n"
            f"Diff: {(meters_legacy - meters_config).abs()}"
        )


class TestCascadeEngineMode:
    """Test CascadeEngine-specific functionality."""

    def test_cascade_engine_initialized_when_requested(self, device):
        """Test that CascadeEngine is created when use_cascade_engine=True."""
        md = MeterDynamics(1, device, use_cascade_engine=True)
        assert md.cascade_engine is not None
        assert md.use_cascade_engine is True

    def test_cascade_engine_not_initialized_in_legacy_mode(self, device):
        """Test that CascadeEngine is not created in legacy mode."""
        md = MeterDynamics(1, device, use_cascade_engine=False)
        assert md.cascade_engine is None
        assert md.use_cascade_engine is False

    def test_cascade_engine_uses_default_config(self, device):
        """Test that default config is loaded when no path specified."""
        md = MeterDynamics(1, device, use_cascade_engine=True)
        # Should load successfully without errors
        assert md.cascade_engine is not None

        # Verify base depletions match expected values
        expected_depletions = torch.tensor(
            [0.005, 0.003, 0.004, 0.0, 0.001, 0.006, 0.0, 0.002], device=device
        )
        assert torch.allclose(md.cascade_engine._base_depletions, expected_depletions)
