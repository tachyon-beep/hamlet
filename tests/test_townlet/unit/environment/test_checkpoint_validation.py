"""Unit tests for checkpoint validation and error handling.

This module tests edge cases and error paths in checkpoint save/load that are
not covered by integration tests:
- Legacy checkpoint detection (missing position_dim)
- Position dimension mismatch validation
- Custom action label loading from config

Coverage targets:
- src/townlet/environment/vectorized_env.py:878-910 (checkpoint validation)
- src/townlet/environment/vectorized_env.py:106-119 (action label loading)
"""

import shutil
from pathlib import Path

import pytest
import yaml

from tests.test_townlet.utils.builders import make_vectorized_env_from_pack


class TestCheckpointValidation:
    """Test checkpoint validation and error handling."""

    def test_legacy_checkpoint_rejected_missing_position_dim(self, cpu_device, test_config_pack_path):
        """Should reject legacy checkpoints missing position_dim field.

        This tests the breaking change in Phase 4 where position_dim became
        required in checkpoint format. Legacy checkpoints should be rejected
        with a clear error message.

        Coverage target: lines 878-891 (legacy checkpoint detection)
        """
        env = make_vectorized_env_from_pack(
            test_config_pack_path,
            num_agents=1,
            device=cpu_device,
        )

        # Create legacy checkpoint data (missing position_dim field)
        legacy_checkpoint = {
            "positions": {"Bed": [3, 4], "Hospital": [1, 2]},
            "ordering": ["Bed", "Hospital"],
            # NOTE: position_dim field is missing (legacy format)
        }

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            env.set_affordance_positions(legacy_checkpoint)

        # Verify error message provides clear guidance
        error_msg = str(exc_info.value)
        assert "position_dim" in error_msg.lower(), "Error should mention missing field"
        assert "no longer supported" in error_msg.lower(), "Error should indicate format is obsolete"
        assert "delete" in error_msg.lower() or "retrain" in error_msg.lower(), "Error should provide remediation steps"

    def test_checkpoint_position_dim_mismatch_rejected(self, cpu_device, test_config_pack_path):
        """Should reject checkpoints with incompatible position dimensions.

        If checkpoint was saved from a 2D substrate but loaded into a 3D substrate,
        the position dimensions won't match and load should fail.

        Coverage target: lines 893-899 (dimension mismatch validation)
        """
        # Create 2D environment
        env_2d = make_vectorized_env_from_pack(
            test_config_pack_path,
            num_agents=1,
            device=cpu_device,
        )

        # Create checkpoint with wrong position_dim (pretend it came from 3D substrate)
        checkpoint_3d = {
            "positions": {"Bed": [3, 4, 2], "Hospital": [1, 2, 5]},  # 3D positions
            "ordering": ["Bed", "Hospital"],
            "position_dim": 3,  # Wrong dimension!
        }

        # Should raise ValueError about dimension mismatch
        with pytest.raises(ValueError) as exc_info:
            env_2d.set_affordance_positions(checkpoint_3d)

        error_msg = str(exc_info.value)
        assert "position_dim mismatch" in error_msg.lower(), "Error should mention dimension mismatch"
        assert "3" in error_msg, "Error should mention checkpoint dimension (3D)"
        assert "2" in error_msg, "Error should mention substrate dimension (2D)"

    def test_checkpoint_loads_with_correct_position_dim(self, cpu_device, test_config_pack_path):
        """Should successfully load checkpoint with matching position_dim.

        This is the happy path - verifies that validation passes when dimensions match.

        Coverage target: lines 901-910 (successful checkpoint loading)
        """
        env = make_vectorized_env_from_pack(
            test_config_pack_path,
            num_agents=1,
            device=cpu_device,
        )

        # Create valid checkpoint with correct position_dim
        valid_checkpoint = {
            "positions": {"Bed": [3, 4], "Hospital": [1, 2]},
            "ordering": ["Bed", "Hospital"],
            "position_dim": 2,  # Matches Grid2D substrate
        }

        # Should load successfully (no exception)
        env.set_affordance_positions(valid_checkpoint)

        # Verify positions were loaded
        assert "Bed" in env.affordances
        assert "Hospital" in env.affordances
        assert env.affordances["Bed"].tolist() == [3, 4]
        assert env.affordances["Hospital"].tolist() == [1, 2]


class TestActionLabelLoading:
    """Test custom action label loading from config."""

    def test_custom_action_labels_from_config(self, cpu_device, tmp_path):
        """Should load custom action labels from action_labels.yaml.

        This tests the optional action label loading feature where users can
        provide custom terminology for actions (e.g., submarine controls).

        Coverage target: lines 106-119 (custom action label loading)
        """
        config_dir = tmp_path / "custom_labels_config"
        shutil.copytree(Path("configs/test"), config_dir)

        action_labels_config = {
            "custom": {
                0: "PORT",  # Custom name for LEFT
                1: "STARBOARD",  # Custom name for RIGHT
                2: "AFT",  # Custom name for DOWN
                3: "FORE",  # Custom name for UP
                4: "INTERACT",
                5: "WAIT",
            }
        }
        with open(config_dir / "action_labels.yaml", "w") as f:
            yaml.safe_dump(action_labels_config, f, sort_keys=False)

        env = make_vectorized_env_from_pack(
            config_dir,
            num_agents=1,
            device=cpu_device,
        )

        # Verify custom labels were loaded (action_labels is an ActionLabels object with labels dict)
        assert env.action_labels is not None
        labels_dict = env.action_labels.labels
        label_values = list(labels_dict.values())
        assert "PORT" in label_values, "Custom label 'PORT' should be loaded"
        assert "STARBOARD" in label_values, "Custom label 'STARBOARD' should be loaded"
        assert "AFT" in label_values, "Custom label 'AFT' should be loaded"
        assert "FORE" in label_values, "Custom label 'FORE' should be loaded"

    def test_default_action_labels_when_no_config(self, cpu_device, test_config_pack_path):
        """Should use default 'gaming' preset when no action_labels.yaml exists.

        Coverage target: lines 120-122 (default label fallback)
        """
        # test_config_pack_path doesn't have action_labels.yaml, so should use default
        env = make_vectorized_env_from_pack(
            test_config_pack_path,
            num_agents=1,
            device=cpu_device,
        )

        # Verify default gaming labels are used (action_labels is an ActionLabels object)
        assert env.action_labels is not None
        labels_dict = env.action_labels.labels
        label_values = list(labels_dict.values())
        # Gaming preset uses: UP, DOWN, LEFT, RIGHT for 2D
        assert "UP" in label_values or "DOWN" in label_values, "Default gaming labels should be present"


class TestAffordancePositionSerialization:
    """Test affordance position serialization edge cases."""

    def test_aspatial_affordance_positions_empty_list(self, cpu_device):
        """Aspatial substrates should serialize affordance positions as empty lists.

        When substrate has position_dim=0 (aspatial), affordance positions should
        be empty lists in the checkpoint.

        Coverage target: lines 854-857 (aspatial position handling)
        """
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        aspatial_config_path = repo_root / "configs" / "aspatial_test"

        # Skip test if aspatial config doesn't exist
        if not aspatial_config_path.exists():
            pytest.skip("Aspatial test config not found")

        # Use aspatial config directly - no parameter injection needed
        # Config packs are atomic artifacts (no individual file overrides)
        env = make_vectorized_env_from_pack(
            aspatial_config_path,
            num_agents=1,
            device=cpu_device,
        )

        env.reset()

        # Get affordance positions
        checkpoint_data = env.get_affordance_positions()

        # Verify position_dim is 0
        assert checkpoint_data["position_dim"] == 0, "Aspatial substrate should have position_dim=0"

        # Verify all affordance positions are empty lists
        for name, pos in checkpoint_data["positions"].items():
            assert pos == [], f"Aspatial affordance {name} should have empty position list, got {pos}"
