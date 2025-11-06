"""Tests for movement delta construction from ActionConfig in VectorizedHamletEnv."""

from pathlib import Path

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestMovementDeltasFromActionConfig:
    """Test that movement deltas come from ActionConfig, not hardcoded arrays."""

    def test_movement_deltas_from_action_config_grid2d(self):
        """Movement deltas should come from ActionConfig, not hardcoded arrays."""
        env = VectorizedHamletEnv(
            config_pack_path=Path("configs/L1_full_observability"),
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.5,
            wait_energy_cost=0.1,
            interact_energy_cost=0.3,
            agent_lifespan=1000,
            device=torch.device("cpu"),
        )

        env.reset()

        # Get UP action from substrate defaults
        substrate_actions = env.substrate.get_default_actions()
        up_action = next(a for a in substrate_actions if a.name == "UP")

        # Verify delta tensor contains correct delta at action's index
        expected_delta = torch.tensor(up_action.delta, device=env.device, dtype=env.substrate.position_dtype)
        actual_delta = env._movement_deltas[up_action.id]

        assert torch.equal(actual_delta, expected_delta), f"Delta mismatch for UP action: expected {expected_delta}, " f"got {actual_delta}"

    def test_all_movement_deltas_match_action_config(self):
        """All movement actions should have correct deltas from ActionConfig."""
        env = VectorizedHamletEnv(
            config_pack_path=Path("configs/L1_full_observability"),
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.5,
            wait_energy_cost=0.1,
            interact_energy_cost=0.3,
            agent_lifespan=1000,
            device=torch.device("cpu"),
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

                assert torch.equal(actual_delta, expected_delta), (
                    f"Delta mismatch for {action.name} (id={action.id}): " f"expected {expected_delta}, got {actual_delta}"
                )

    def test_non_movement_actions_have_zero_deltas(self):
        """INTERACT and WAIT actions should have zero deltas."""
        env = VectorizedHamletEnv(
            config_pack_path=Path("configs/L1_full_observability"),
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.5,
            wait_energy_cost=0.1,
            interact_energy_cost=0.3,
            agent_lifespan=1000,
            device=torch.device("cpu"),
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
