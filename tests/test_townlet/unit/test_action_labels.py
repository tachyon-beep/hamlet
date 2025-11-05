"""Unit tests for configurable action label system."""

import pytest

from townlet.environment.action_labels import (
    PRESET_LABELS,
    ActionLabels,
    CanonicalAction,
    get_labels,
)
from townlet.substrate.config import ActionLabelConfig


class TestCanonicalAction:
    """Test canonical action enum."""

    def test_canonical_action_indices(self):
        """Canonical actions have correct integer indices."""
        assert CanonicalAction.MOVE_X_NEGATIVE == 0
        assert CanonicalAction.MOVE_X_POSITIVE == 1
        assert CanonicalAction.MOVE_Y_NEGATIVE == 2
        assert CanonicalAction.MOVE_Y_POSITIVE == 3
        assert CanonicalAction.INTERACT == 4
        assert CanonicalAction.WAIT == 5
        assert CanonicalAction.MOVE_Z_POSITIVE == 6
        assert CanonicalAction.MOVE_Z_NEGATIVE == 7


class TestPresetLabels:
    """Test preset label configurations."""

    def test_all_presets_exist(self):
        """All 4 preset label sets exist."""
        assert "gaming" in PRESET_LABELS
        assert "6dof" in PRESET_LABELS
        assert "cardinal" in PRESET_LABELS
        assert "math" in PRESET_LABELS

    def test_gaming_preset(self):
        """Gaming preset has correct labels."""
        labels = PRESET_LABELS["gaming"]
        assert labels.domain == "gaming"
        assert labels.labels[CanonicalAction.MOVE_X_NEGATIVE] == "LEFT"
        assert labels.labels[CanonicalAction.MOVE_X_POSITIVE] == "RIGHT"
        assert labels.labels[CanonicalAction.MOVE_Y_NEGATIVE] == "DOWN"
        assert labels.labels[CanonicalAction.MOVE_Y_POSITIVE] == "UP"
        assert labels.labels[CanonicalAction.INTERACT] == "INTERACT"
        assert labels.labels[CanonicalAction.WAIT] == "WAIT"
        assert labels.labels[CanonicalAction.MOVE_Z_POSITIVE] == "FORWARD"
        assert labels.labels[CanonicalAction.MOVE_Z_NEGATIVE] == "BACKWARD"

    def test_6dof_preset(self):
        """6-DoF preset has robotics terminology."""
        labels = PRESET_LABELS["6dof"]
        assert labels.domain == "robotics"
        assert labels.labels[CanonicalAction.MOVE_X_NEGATIVE] == "SWAY_LEFT"
        assert labels.labels[CanonicalAction.MOVE_X_POSITIVE] == "SWAY_RIGHT"
        assert labels.labels[CanonicalAction.MOVE_Y_NEGATIVE] == "HEAVE_DOWN"
        assert labels.labels[CanonicalAction.MOVE_Y_POSITIVE] == "HEAVE_UP"
        assert labels.labels[CanonicalAction.MOVE_Z_POSITIVE] == "SURGE_FORWARD"
        assert labels.labels[CanonicalAction.MOVE_Z_NEGATIVE] == "SURGE_BACKWARD"

    def test_cardinal_preset(self):
        """Cardinal preset has compass directions."""
        labels = PRESET_LABELS["cardinal"]
        assert labels.domain == "navigation"
        assert labels.labels[CanonicalAction.MOVE_X_NEGATIVE] == "WEST"
        assert labels.labels[CanonicalAction.MOVE_X_POSITIVE] == "EAST"
        assert labels.labels[CanonicalAction.MOVE_Y_NEGATIVE] == "SOUTH"
        assert labels.labels[CanonicalAction.MOVE_Y_POSITIVE] == "NORTH"
        assert labels.labels[CanonicalAction.MOVE_Z_POSITIVE] == "ASCEND"
        assert labels.labels[CanonicalAction.MOVE_Z_NEGATIVE] == "DESCEND"

    def test_math_preset(self):
        """Math preset has explicit axis notation."""
        labels = PRESET_LABELS["math"]
        assert labels.domain == "mathematics"
        assert labels.labels[CanonicalAction.MOVE_X_NEGATIVE] == "X_NEG"
        assert labels.labels[CanonicalAction.MOVE_X_POSITIVE] == "X_POS"
        assert labels.labels[CanonicalAction.MOVE_Y_NEGATIVE] == "Y_NEG"
        assert labels.labels[CanonicalAction.MOVE_Y_POSITIVE] == "Y_POS"
        assert labels.labels[CanonicalAction.MOVE_Z_POSITIVE] == "Z_POS"
        assert labels.labels[CanonicalAction.MOVE_Z_NEGATIVE] == "Z_NEG"


class TestActionLabelsDataclass:
    """Test ActionLabels dataclass."""

    def test_action_labels_immutable(self):
        """ActionLabels is immutable (frozen)."""
        labels = ActionLabels(labels={0: "TEST"}, description="Test labels", domain="test")

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            labels.labels = {1: "MODIFIED"}

    def test_get_label(self):
        """get_label returns correct label for index."""
        labels = ActionLabels(labels={0: "LEFT", 1: "RIGHT"}, description="Test", domain="test")

        assert labels.get_label(0) == "LEFT"
        assert labels.get_label(1) == "RIGHT"

    def test_get_label_invalid_index(self):
        """get_label raises KeyError for invalid index."""
        labels = ActionLabels(labels={0: "LEFT"}, description="Test", domain="test")

        with pytest.raises(KeyError):
            labels.get_label(1)

    def test_get_all_labels(self):
        """get_all_labels returns copy of labels dict."""
        labels = ActionLabels(labels={0: "LEFT", 1: "RIGHT"}, description="Test", domain="test")

        all_labels = labels.get_all_labels()
        assert all_labels == {0: "LEFT", 1: "RIGHT"}

        # Verify it's a copy (modifying returned dict doesn't affect original)
        all_labels[2] = "MODIFIED"
        assert 2 not in labels.labels

    def test_get_action_count(self):
        """get_action_count returns number of actions."""
        labels = ActionLabels(labels={0: "A", 1: "B", 2: "C"}, description="Test", domain="test")

        assert labels.get_action_count() == 3


class TestGetLabelsFunction:
    """Test get_labels() function."""

    def test_get_labels_with_preset_2d(self):
        """get_labels with gaming preset for 2D substrate."""
        labels = get_labels(preset="gaming", substrate_position_dim=2)

        assert labels.domain == "gaming"
        assert labels.get_action_count() == 6  # 2D has 6 actions
        assert labels.get_label(0) == "UP"
        assert labels.get_label(1) == "DOWN"
        assert labels.get_label(2) == "LEFT"
        assert labels.get_label(3) == "RIGHT"
        assert labels.get_label(4) == "INTERACT"
        assert labels.get_label(5) == "WAIT"

    def test_get_labels_with_preset_3d(self):
        """get_labels with gaming preset for 3D substrate."""
        labels = get_labels(preset="gaming", substrate_position_dim=3)

        assert labels.get_action_count() == 8  # 3D has 8 actions
        assert labels.get_label(6) == "FORWARD"
        assert labels.get_label(7) == "BACKWARD"

    def test_get_labels_with_preset_1d(self):
        """get_labels with gaming preset for 1D substrate."""
        labels = get_labels(preset="gaming", substrate_position_dim=1)

        assert labels.get_action_count() == 4  # 1D has 4 actions
        assert labels.get_label(0) == "LEFT"
        assert labels.get_label(1) == "RIGHT"
        assert labels.get_label(2) == "INTERACT"
        assert labels.get_label(3) == "WAIT"

    def test_get_labels_with_preset_aspatial(self):
        """get_labels with gaming preset for aspatial substrate."""
        labels = get_labels(preset="gaming", substrate_position_dim=0)

        assert labels.get_action_count() == 2  # Aspatial has 2 actions
        assert labels.get_label(0) == "INTERACT"
        assert labels.get_label(1) == "WAIT"

    def test_get_labels_with_custom_labels(self):
        """get_labels with custom submarine labels."""
        custom = {
            0: "PORT",
            1: "STARBOARD",
            2: "AFT",
            3: "FORE",
            4: "INTERACT",
            5: "WAIT",
            6: "SURFACE",
            7: "DIVE",
        }

        labels = get_labels(custom_labels=custom, substrate_position_dim=3)

        assert labels.domain == "custom"
        assert labels.get_label(6) == "SURFACE"
        assert labels.get_label(7) == "DIVE"

    def test_get_labels_unknown_preset(self):
        """get_labels raises ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_labels(preset="unknown", substrate_position_dim=2)

    def test_get_labels_no_preset_or_custom(self):
        """get_labels raises ValueError if neither preset nor custom provided."""
        with pytest.raises(ValueError, match="Must provide either preset or custom_labels"):
            get_labels(substrate_position_dim=2)


class TestDimensionalityFiltering:
    """Test label filtering for different substrate dimensionalities."""

    def test_filtering_aspatial(self):
        """Aspatial (0D) has only INTERACT and WAIT."""
        labels = get_labels(preset="gaming", substrate_position_dim=0)

        assert labels.get_action_count() == 2
        assert 0 in labels.get_all_labels()  # INTERACT
        assert 1 in labels.get_all_labels()  # WAIT

    def test_filtering_1d(self):
        """1D has X movement + INTERACT + WAIT."""
        labels = get_labels(preset="gaming", substrate_position_dim=1)

        assert labels.get_action_count() == 4
        assert labels.get_label(0) == "LEFT"  # MOVE_X_NEGATIVE remapped to 0
        assert labels.get_label(1) == "RIGHT"  # MOVE_X_POSITIVE remapped to 1
        assert labels.get_label(2) == "INTERACT"  # Remapped to 2
        assert labels.get_label(3) == "WAIT"  # Remapped to 3

    def test_filtering_2d(self):
        """2D has XY movement + INTERACT + WAIT."""
        labels = get_labels(preset="gaming", substrate_position_dim=2)

        assert labels.get_action_count() == 6
        # Check action order matches 2D substrate
        assert labels.get_label(0) == "UP"  # MOVE_Y_POSITIVE
        assert labels.get_label(1) == "DOWN"  # MOVE_Y_NEGATIVE
        assert labels.get_label(2) == "LEFT"  # MOVE_X_NEGATIVE
        assert labels.get_label(3) == "RIGHT"  # MOVE_X_POSITIVE
        assert labels.get_label(4) == "INTERACT"
        assert labels.get_label(5) == "WAIT"

    def test_filtering_3d(self):
        """3D has XYZ movement + INTERACT + WAIT."""
        labels = get_labels(preset="gaming", substrate_position_dim=3)

        assert labels.get_action_count() == 8
        # Check Z-axis labels present
        assert labels.get_label(6) == "FORWARD"  # MOVE_Z_POSITIVE
        assert labels.get_label(7) == "BACKWARD"  # MOVE_Z_NEGATIVE


class TestActionLabelConfig:
    """Test ActionLabelConfig Pydantic schema."""

    def test_config_with_preset(self):
        """ActionLabelConfig validates preset correctly."""
        config = ActionLabelConfig(preset="gaming")

        assert config.preset == "gaming"
        assert config.custom is None

    def test_config_with_custom(self):
        """ActionLabelConfig validates custom labels correctly."""
        custom = {0: "PORT", 1: "STARBOARD", 4: "INTERACT", 5: "WAIT"}
        config = ActionLabelConfig(custom=custom)

        assert config.preset is None
        assert config.custom == custom

    def test_config_requires_preset_or_custom(self):
        """ActionLabelConfig requires either preset or custom."""
        with pytest.raises(ValueError, match="Must specify either 'preset' or 'custom'"):
            ActionLabelConfig()

    def test_config_rejects_both_preset_and_custom(self):
        """ActionLabelConfig rejects both preset and custom."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            ActionLabelConfig(preset="gaming", custom={0: "TEST"})

    def test_config_invalid_preset(self):
        """ActionLabelConfig rejects invalid preset names."""
        with pytest.raises(ValueError, match="Invalid preset"):
            ActionLabelConfig(preset="invalid")

    def test_config_invalid_custom_key_type(self):
        """ActionLabelConfig rejects non-integer keys."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            ActionLabelConfig(custom={"a": "TEST"})  # type: ignore

    def test_config_invalid_custom_key_range(self):
        """ActionLabelConfig rejects keys outside 0-7 range."""
        with pytest.raises(ValueError, match="must be integers 0-7"):
            ActionLabelConfig(custom={8: "TEST"})

    def test_config_invalid_custom_value_type(self):
        """ActionLabelConfig rejects non-string values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be a valid string"):
            ActionLabelConfig(custom={0: 123})  # type: ignore

    def test_config_invalid_custom_empty_string(self):
        """ActionLabelConfig rejects empty string values."""
        with pytest.raises(ValueError, match="must be non-empty strings"):
            ActionLabelConfig(custom={0: ""})
