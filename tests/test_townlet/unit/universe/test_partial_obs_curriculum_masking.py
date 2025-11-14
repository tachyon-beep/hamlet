"""Test that partial observability uses curriculum masking instead of changing obs_dim.

BUG-43: Partial observability should mask fields via curriculum_active, not swap them out.
This enables transfer learning from full-obs to POMDP with the same Q-network architecture.
"""

from pathlib import Path

import pytest

from townlet.universe.compiler import UniverseCompiler


class TestPartialObservabilityCurriculumMasking:
    """Test that partial_observability uses masking instead of field swapping."""

    @pytest.fixture
    def l1_compiled(self):
        """Compile L1_full_observability config."""
        compiler = UniverseCompiler()
        config_dir = Path("configs/L1_full_observability")
        return compiler.compile(config_dir)

    @pytest.fixture
    def l2_compiled(self):
        """Compile L2_partial_observability config."""
        compiler = UniverseCompiler()
        config_dir = Path("configs/L2_partial_observability")
        return compiler.compile(config_dir)

    def test_observation_dim_is_constant_across_full_and_partial_obs(self, l1_compiled, l2_compiled):
        """L1 (full obs) and L2 (partial obs) should have SAME observation_dim."""
        l1_obs_dim = l1_compiled.metadata.observation_dim
        l2_obs_dim = l2_compiled.metadata.observation_dim

        assert l1_obs_dim == l2_obs_dim, (
            f"Observation dim should be constant for transfer learning, but got "
            f"L1={l1_obs_dim} vs L2={l2_obs_dim}. "
            f"Partial observability should use curriculum masking, not field swapping."
        )

    def test_observation_activity_total_dims_constant(self, l1_compiled, l2_compiled):
        """ObservationActivity.total_dims should be same across levels."""
        l1_total = l1_compiled.observation_activity.total_dims
        l2_total = l2_compiled.observation_activity.total_dims

        assert l1_total == l2_total, f"ObservationActivity.total_dims should be constant, but got " f"L1={l1_total} vs L2={l2_total}"

    def test_active_dim_count_differs_due_to_masking(self, l1_compiled, l2_compiled):
        """active_dim_count should differ because different fields are masked."""
        l1_active = l1_compiled.observation_activity.active_dim_count
        l2_active = l2_compiled.observation_activity.active_dim_count

        # L1 has grid_encoding (64 dims) active, local_window inactive
        # L2 has local_window (25 dims) active, grid_encoding inactive
        # So L1 should have 39 more active dims than L2
        expected_diff = 64 - 25  # 39 dims

        actual_diff = l1_active - l2_active

        assert actual_diff == expected_diff, (
            f"Expected L1 to have {expected_diff} more active dims than L2 "
            f"(grid_encoding 64 vs local_window 25), but got diff={actual_diff} "
            f"(L1={l1_active}, L2={l2_active})"
        )

    def test_both_levels_have_both_spatial_fields_in_spec(self, l1_compiled, l2_compiled):
        """Both L1 and L2 should have BOTH grid_encoding and local_window fields (one masked)."""
        # Get field names from observation spec
        l1_field_names = {field.name for field in l1_compiled.observation_spec.fields}
        l2_field_names = {field.name for field in l2_compiled.observation_spec.fields}

        # Both should have the full field set (superset contract)
        assert l1_field_names == l2_field_names, (
            f"L1 and L2 should have identical field sets (superset contract), but:\n"
            f"L1 only: {l1_field_names - l2_field_names}\n"
            f"L2 only: {l2_field_names - l1_field_names}"
        )

        # Both should contain grid_encoding and local_window
        assert "obs_grid_encoding" in l1_field_names, "L1 missing obs_grid_encoding"
        assert "obs_local_window" in l1_field_names, "L1 missing obs_local_window"
        assert "obs_grid_encoding" in l2_field_names, "L2 missing obs_grid_encoding"
        assert "obs_local_window" in l2_field_names, "L2 missing obs_local_window"

    def test_active_mask_differs_for_spatial_fields(self, l1_compiled, l2_compiled):
        """active_mask should show grid_encoding active in L1, local_window active in L2."""
        # Find indices of grid_encoding and local_window fields
        l1_fields = {field.name: field for field in l1_compiled.observation_spec.fields}
        l2_fields = {field.name: field for field in l2_compiled.observation_spec.fields}

        grid_field_l1 = l1_fields["obs_grid_encoding"]
        local_field_l1 = l1_fields["obs_local_window"]
        grid_field_l2 = l2_fields["obs_grid_encoding"]
        local_field_l2 = l2_fields["obs_local_window"]

        # L1 (full obs): grid_encoding active, local_window inactive
        l1_grid_active = all(
            l1_compiled.observation_activity.active_mask[i] for i in range(grid_field_l1.start_index, grid_field_l1.end_index)
        )
        l1_local_active = all(
            l1_compiled.observation_activity.active_mask[i] for i in range(local_field_l1.start_index, local_field_l1.end_index)
        )

        assert l1_grid_active, "L1 should have grid_encoding active (full observability)"
        assert not l1_local_active, "L1 should have local_window inactive"

        # L2 (partial obs): local_window active, grid_encoding inactive
        l2_grid_active = any(
            l2_compiled.observation_activity.active_mask[i] for i in range(grid_field_l2.start_index, grid_field_l2.end_index)
        )
        l2_local_active = all(
            l2_compiled.observation_activity.active_mask[i] for i in range(local_field_l2.start_index, local_field_l2.end_index)
        )

        assert not l2_grid_active, "L2 should have grid_encoding inactive (POMDP)"
        assert l2_local_active, "L2 should have local_window active (POMDP)"
