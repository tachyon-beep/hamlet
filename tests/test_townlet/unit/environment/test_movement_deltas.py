"""Tests for movement delta construction from ActionConfig in VectorizedHamletEnv."""

from pathlib import Path

import torch


class TestMovementDeltasFromActionConfig:
    """Test that movement deltas come from ActionConfig, not hardcoded arrays."""

    def test_movement_deltas_from_action_config_grid2d(self, env_factory, cpu_device):
        """Movement deltas should come from ActionConfig, not hardcoded arrays."""
        env = env_factory(
            config_dir=Path("configs/L1_full_observability"),
            num_agents=1,
            device_override=cpu_device,
        )

        env.reset()

        # Get UP action from substrate defaults
        substrate_actions = env.substrate.get_default_actions()
        up_action = next(a for a in substrate_actions if a.name == "UP")

        # Verify delta tensor contains correct delta at action's index
        expected_delta = torch.tensor(up_action.delta, device=env.device, dtype=env.substrate.position_dtype)
        actual_delta = env._movement_deltas[up_action.id]

        assert torch.equal(actual_delta, expected_delta), f"Delta mismatch for UP action: expected {expected_delta}, got {actual_delta}"

    def test_all_movement_deltas_match_action_config(self, env_factory, cpu_device):
        """All movement actions should have correct deltas from ActionConfig."""
        env = env_factory(
            config_dir=Path("configs/L1_full_observability"),
            num_agents=1,
            device_override=cpu_device,
        )

        env.reset()

        # Verify all movement actions
        substrate_actions = env.substrate.get_default_actions()
        for action in substrate_actions:
            if action.delta is not None:
                expected_delta = torch.tensor(
                    action.delta,
                    device=env.device,
                    dtype=env.substrate.position_dtype,
                )
                actual_delta = env._movement_deltas[action.id]

                assert torch.equal(
                    actual_delta, expected_delta
                ), f"Delta mismatch for {action.name} (id={action.id}): expected {expected_delta}, got {actual_delta}"

    def test_non_movement_actions_have_zero_deltas(self, env_factory, cpu_device):
        """INTERACT and WAIT actions should have zero deltas."""
        env = env_factory(
            config_dir=Path("configs/L1_full_observability"),
            num_agents=1,
            device_override=cpu_device,
        )

        env.reset()

        # Get non-movement actions
        substrate_actions = env.substrate.get_default_actions()
        interact_action = next(a for a in substrate_actions if a.name == "INTERACT")
        wait_action = next(a for a in substrate_actions if a.name == "WAIT")

        # Zero delta expected
        zero_delta = torch.zeros(
            env.substrate.position_dim,
            device=env.device,
            dtype=env.substrate.position_dtype,
        )

        assert torch.equal(
            env._movement_deltas[interact_action.id], zero_delta
        ), f"INTERACT should have zero delta, got {env._movement_deltas[interact_action.id]}"
        assert torch.equal(
            env._movement_deltas[wait_action.id], zero_delta
        ), f"WAIT should have zero delta, got {env._movement_deltas[wait_action.id]}"
