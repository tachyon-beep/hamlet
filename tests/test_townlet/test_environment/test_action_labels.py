"""Tests for N-dimensional action label support.

This test suite verifies that action labels work correctly for N-dimensional
substrates (N≥4), including GridND and ContinuousND substrates.
"""

import pytest

from townlet.environment.action_labels import get_labels


class TestActionLabels4D:
    """Test action labels for 4D substrates."""

    def test_action_labels_4d_math_preset(self):
        """4D substrate should generate D0-D3 labels with math preset."""
        labels = get_labels(preset="math", substrate_position_dim=4)

        # Movement actions (negative directions)
        assert labels.get_label(0) == "D0_NEG"
        assert labels.get_label(1) == "D1_NEG"
        assert labels.get_label(2) == "D2_NEG"
        assert labels.get_label(3) == "D3_NEG"

        # Movement actions (positive directions)
        assert labels.get_label(4) == "D0_POS"
        assert labels.get_label(5) == "D1_POS"
        assert labels.get_label(6) == "D2_POS"
        assert labels.get_label(7) == "D3_POS"

        # Meta actions
        assert labels.get_label(8) == "INTERACT"
        assert labels.get_label(9) == "WAIT"

        # Action count
        assert labels.get_action_count() == 10  # 2*4 + 2

    def test_action_labels_4d_gaming_preset(self):
        """4D substrate should use dimension index notation even with gaming preset."""
        labels = get_labels(preset="gaming", substrate_position_dim=4)

        # All presets use D{i}_NEG/D{i}_POS for N≥4
        assert labels.get_label(0) == "D0_NEG"
        assert labels.get_label(3) == "D3_NEG"
        assert labels.get_label(4) == "D0_POS"
        assert labels.get_label(7) == "D3_POS"
        assert labels.get_label(8) == "INTERACT"
        assert labels.get_label(9) == "WAIT"

        assert labels.get_action_count() == 10  # 2*4 + 2

    def test_action_labels_4d_custom(self):
        """4D substrate should accept custom labels."""
        custom = {
            0: "AXIS0_NEG",
            1: "AXIS1_NEG",
            2: "AXIS2_NEG",
            3: "AXIS3_NEG",
            4: "AXIS0_POS",
            5: "AXIS1_POS",
            6: "AXIS2_POS",
            7: "AXIS3_POS",
            8: "INTERACT",
            9: "WAIT",
        }
        labels = get_labels(custom_labels=custom, substrate_position_dim=4)

        assert labels.get_label(0) == "AXIS0_NEG"
        assert labels.get_label(7) == "AXIS3_POS"
        assert labels.get_label(8) == "INTERACT"
        assert labels.get_label(9) == "WAIT"
        assert labels.get_action_count() == 10

    def test_action_labels_4d_partial_custom(self):
        """4D substrate with partial custom labels should use fallbacks."""
        # Only provide labels for first 2 dimensions
        custom = {
            0: "LEFT",
            1: "RIGHT",  # D0 positive/negative (wait, this is wrong - let me fix)
            2: "DOWN",
            3: "UP",  # D1 positive/negative
            # D2 and D3 missing - should fallback to D2_NEG, D2_POS, D3_NEG, D3_POS
        }
        labels = get_labels(custom_labels=custom, substrate_position_dim=4)

        # Custom labels used
        assert labels.get_label(0) == "LEFT"
        assert labels.get_label(1) == "RIGHT"
        assert labels.get_label(2) == "DOWN"
        assert labels.get_label(3) == "UP"

        # Fallback labels generated for missing dimensions
        assert labels.get_label(4) == "D0_POS"
        assert labels.get_label(5) == "D1_POS"
        assert labels.get_label(6) == "D2_POS"
        assert labels.get_label(7) == "D3_POS"
        assert labels.get_label(8) == "INTERACT"
        assert labels.get_label(9) == "WAIT"


class TestActionLabels7D:
    """Test action labels for 7D substrates."""

    def test_action_labels_7d_math_preset(self):
        """7D substrate should generate D0-D6 labels."""
        labels = get_labels(preset="math", substrate_position_dim=7)

        # Check negative directions
        assert labels.get_label(0) == "D0_NEG"
        assert labels.get_label(6) == "D6_NEG"

        # Check positive directions
        assert labels.get_label(7) == "D0_POS"
        assert labels.get_label(13) == "D6_POS"

        # Meta actions
        assert labels.get_label(14) == "INTERACT"
        assert labels.get_label(15) == "WAIT"

        assert labels.get_action_count() == 16  # 2*7 + 2

    def test_action_labels_7d_gaming_preset(self):
        """7D substrate should use dimension index notation even with gaming preset."""
        labels = get_labels(preset="gaming", substrate_position_dim=7)

        # All presets use D{i}_NEG/D{i}_POS for N≥4
        assert labels.get_label(0) == "D0_NEG"
        assert labels.get_label(6) == "D6_NEG"
        assert labels.get_label(7) == "D0_POS"
        assert labels.get_label(13) == "D6_POS"
        assert labels.get_label(14) == "INTERACT"
        assert labels.get_label(15) == "WAIT"

        assert labels.get_action_count() == 16  # 2*7 + 2


class TestActionLabelsEdgeCases:
    """Test action labels for edge cases."""

    def test_action_labels_10d_edge_case(self):
        """10D substrate should handle double-digit dimension indices."""
        labels = get_labels(preset="math", substrate_position_dim=10)

        # Check double-digit dimension labels
        assert labels.get_label(9) == "D9_NEG"
        assert labels.get_label(19) == "D9_POS"
        assert labels.get_label(20) == "INTERACT"
        assert labels.get_label(21) == "WAIT"

        assert labels.get_action_count() == 22  # 2*10 + 2

    def test_action_labels_5d_custom(self):
        """5D substrate should accept complete custom labels."""
        custom = {
            0: "DIM0_NEG",
            1: "DIM1_NEG",
            2: "DIM2_NEG",
            3: "DIM3_NEG",
            4: "DIM4_NEG",
            5: "DIM0_POS",
            6: "DIM1_POS",
            7: "DIM2_POS",
            8: "DIM3_POS",
            9: "DIM4_POS",
            10: "INTERACT",
            11: "WAIT",
        }
        labels = get_labels(custom_labels=custom, substrate_position_dim=5)

        assert labels.get_label(0) == "DIM0_NEG"
        assert labels.get_label(9) == "DIM4_POS"
        assert labels.get_label(10) == "INTERACT"
        assert labels.get_label(11) == "WAIT"
        assert labels.get_action_count() == 12  # 2*5 + 2

    @pytest.mark.parametrize("preset_name", ["gaming", "6dof", "cardinal", "math"])
    def test_all_presets_support_nd(self, preset_name):
        """All presets should work with N≥4 substrates."""
        labels = get_labels(preset=preset_name, substrate_position_dim=5)

        # All use dimension index notation
        assert labels.get_label(0) == "D0_NEG"
        assert labels.get_label(5) == "D0_POS"
        assert labels.get_label(10) == "INTERACT"
        assert labels.get_label(11) == "WAIT"
        assert labels.get_action_count() == 12  # 2*5 + 2


class TestActionLabelsBackwardCompatibility:
    """Test that 0-3D substrates remain unchanged."""

    def test_2d_substrate_still_works(self):
        """2D substrate should use existing hardcoded mapping."""
        labels = get_labels(preset="gaming", substrate_position_dim=2)

        # Should still use traditional labels, not D0_NEG/D0_POS
        assert labels.get_label(0) == "UP"  # MOVE_Y_POSITIVE
        assert labels.get_label(1) == "DOWN"  # MOVE_Y_NEGATIVE
        assert labels.get_label(2) == "LEFT"  # MOVE_X_NEGATIVE
        assert labels.get_label(3) == "RIGHT"  # MOVE_X_POSITIVE
        assert labels.get_label(4) == "INTERACT"
        assert labels.get_label(5) == "WAIT"
        assert labels.get_action_count() == 6

    def test_3d_substrate_still_works(self):
        """3D substrate should use existing hardcoded mapping."""
        labels = get_labels(preset="gaming", substrate_position_dim=3)

        # Should still use traditional labels
        assert labels.get_label(0) == "UP"
        assert labels.get_label(6) == "FORWARD"  # MOVE_Z_POSITIVE
        assert labels.get_label(7) == "BACKWARD"  # MOVE_Z_NEGATIVE
        assert labels.get_action_count() == 8
