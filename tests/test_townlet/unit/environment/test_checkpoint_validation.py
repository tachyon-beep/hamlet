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

from pathlib import Path

import pytest
import yaml

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestCheckpointValidation:
    """Test checkpoint validation and error handling."""

    def test_legacy_checkpoint_rejected_missing_position_dim(self, cpu_device, test_config_pack_path):
        """Should reject legacy checkpoints missing position_dim field.

        This tests the breaking change in Phase 4 where position_dim became
        required in checkpoint format. Legacy checkpoints should be rejected
        with a clear error message.

        Coverage target: lines 878-891 (legacy checkpoint detection)
        """
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
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

        # Verify error message mentions legacy checkpoint
        error_msg = str(exc_info.value)
        assert "legacy checkpoint" in error_msg.lower(), "Error should mention legacy checkpoint"
        assert "position_dim" in error_msg.lower(), "Error should mention missing field"
        assert "Phase 4" in error_msg, "Error should reference breaking change"
        assert "Delete old checkpoint" in error_msg, "Error should provide remediation steps"

    def test_checkpoint_position_dim_mismatch_rejected(self, cpu_device, test_config_pack_path):
        """Should reject checkpoints with incompatible position dimensions.

        If checkpoint was saved from a 2D substrate but loaded into a 3D substrate,
        the position dimensions won't match and load should fail.

        Coverage target: lines 893-899 (dimension mismatch validation)
        """
        # Create 2D environment
        env_2d = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
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
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
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
        # Create minimal config pack with custom action labels
        config_dir = tmp_path / "custom_labels_config"
        config_dir.mkdir()

        # Create substrate.yaml (Grid2D)
        substrate_config = {
            "version": "1.0",
            "description": "2D grid for action label test",
            "type": "grid",
            "grid": {
                "topology": "square",
                "width": 5,
                "height": 5,
                "boundary": "clamp",
                "distance_metric": "manhattan",
                "observation_encoding": "relative",
            },
        }
        with open(config_dir / "substrate.yaml", "w") as f:
            yaml.safe_dump(substrate_config, f)

        # Create action_labels.yaml with custom labels
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
            yaml.safe_dump(action_labels_config, f)

        # Create minimal bars.yaml
        bars_config = {
            "version": "2.0",
            "description": "Minimal test config",
            "bars": [
                {
                    "name": "energy",
                    "index": 0,
                    "tier": "pivotal",
                    "range": [0.0, 1.0],
                    "initial": 1.0,
                    "base_depletion": 0.005,
                    "description": "Energy meter",
                }
            ],
            "terminal_conditions": [{"meter": "energy", "operator": "<=", "value": 0.0, "description": "Death"}],
        }
        with open(config_dir / "bars.yaml", "w") as f:
            yaml.safe_dump(bars_config, f)

        # Create minimal cascades.yaml
        cascades_config = {
            "version": "2.0",
            "description": "Minimal test cascades",
            "math_type": "gradient_penalty",
            "modulations": [],
            "cascades": [],
            "execution_order": [],
        }
        with open(config_dir / "cascades.yaml", "w") as f:
            yaml.safe_dump(cascades_config, f)

        # Create minimal affordances.yaml
        affordances_config = {
            "version": "2.0",
            "description": "Minimal test affordances",
            "status": "TEST",
            "affordances": [
                {
                    "id": "0",
                    "name": "Bed",
                    "category": "energy",
                    "interaction_type": "instant",
                    "costs": [],
                    "effects": [{"meter": "energy", "amount": 0.5}],
                    "operating_hours": [0, 24],
                }
            ],
        }
        with open(config_dir / "affordances.yaml", "w") as f:
            yaml.safe_dump(affordances_config, f)

        # Create minimal variables_reference.yaml (1 meter: energy)
        vfs_config = {
            "version": "1.0",
            "variables": [
                {"id": "grid_encoding", "scope": "agent", "type": "vecNf", "dims": 25, "lifetime": "tick", "readable_by": ["agent", "engine"], "writable_by": ["engine"], "default": [0.0] * 25},
                {"id": "position", "scope": "agent", "type": "vecNf", "dims": 2, "lifetime": "episode", "readable_by": ["agent", "engine"], "writable_by": ["engine"], "default": [0.0, 0.0]},
                {"id": "energy", "scope": "agent", "type": "scalar", "lifetime": "episode", "readable_by": ["agent", "engine"], "writable_by": ["engine"], "default": 1.0},
                {"id": "affordance_at_position", "scope": "agent", "type": "vecNf", "dims": 15, "lifetime": "tick", "readable_by": ["agent", "engine"], "writable_by": ["engine"], "default": [0.0] * 14 + [1.0]},
                {"id": "time_sin", "scope": "global", "type": "scalar", "lifetime": "tick", "readable_by": ["agent"], "writable_by": ["engine"], "default": 0.0},
                {"id": "time_cos", "scope": "global", "type": "scalar", "lifetime": "tick", "readable_by": ["agent"], "writable_by": ["engine"], "default": 1.0},
                {"id": "interaction_progress", "scope": "agent", "type": "scalar", "lifetime": "tick", "readable_by": ["agent"], "writable_by": ["engine"], "default": 0.0},
                {"id": "lifetime_progress", "scope": "agent", "type": "scalar", "lifetime": "episode", "readable_by": ["agent"], "writable_by": ["engine"], "default": 0.0},
            ],
            "exposed_observations": [
                {"id": "obs_grid_encoding", "source_variable": "grid_encoding", "exposed_to": ["agent"], "shape": [25], "normalization": None},
                {"id": "obs_position", "source_variable": "position", "exposed_to": ["agent"], "shape": [2], "normalization": None},
                {"id": "obs_energy", "source_variable": "energy", "exposed_to": ["agent"], "shape": [], "normalization": None},
                {"id": "obs_affordance_at_position", "source_variable": "affordance_at_position", "exposed_to": ["agent"], "shape": [15], "normalization": None},
                {"id": "obs_time_sin", "source_variable": "time_sin", "exposed_to": ["agent"], "shape": [], "normalization": None},
                {"id": "obs_time_cos", "source_variable": "time_cos", "exposed_to": ["agent"], "shape": [], "normalization": None},
                {"id": "obs_interaction_progress", "source_variable": "interaction_progress", "exposed_to": ["agent"], "shape": [], "normalization": None},
                {"id": "obs_lifetime_progress", "source_variable": "lifetime_progress", "exposed_to": ["agent"], "shape": [], "normalization": None},
            ],
        }
        with open(config_dir / "variables_reference.yaml", "w") as f:
            yaml.safe_dump(vfs_config, f, sort_keys=False)

        # Create environment with custom action labels config
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=cpu_device,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=config_dir,
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
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=test_config_pack_path,
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
        # Create aspatial environment

        repo_root = Path(__file__).parent.parent.parent.parent.parent
        aspatial_config_path = repo_root / "configs" / "aspatial_test"

        # Skip test if aspatial config doesn't exist
        if not aspatial_config_path.exists():
            pytest.skip("Aspatial test config not found")

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=1,  # Ignored for aspatial
            device=cpu_device,
            partial_observability=False,
            vision_range=1,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            config_pack_path=aspatial_config_path,
        )

        env.reset()

        # Get affordance positions
        checkpoint_data = env.get_affordance_positions()

        # Verify position_dim is 0
        assert checkpoint_data["position_dim"] == 0, "Aspatial substrate should have position_dim=0"

        # Verify all affordance positions are empty lists
        for name, pos in checkpoint_data["positions"].items():
            assert pos == [], f"Aspatial affordance {name} should have empty position list, got {pos}"
