"""Integration tests for action cost application from bars.yaml.

Tests verify that VectorizedHamletEnv correctly applies:
- base_depletion: Passive decay every tick (existence cost)
- base_move_depletion: Additional cost for movement actions
- base_interaction_cost: Additional cost for INTERACT action

Architecture: Three fundamental action types
1. Existence → base_depletion only
2. Movement → base_depletion + base_move_depletion
3. Interaction → base_depletion + base_interaction_cost
4. WAIT → base_depletion only (optional action)
"""

import torch


class TestMovementCosts:
    """Test that movement actions apply base_move_depletion from bars.yaml."""

    def test_movement_applies_base_move_depletion(self, cpu_env_factory):
        """Movement actions should apply base_move_depletion from bars.yaml."""
        env = cpu_env_factory()
        obs = env.reset()

        # Get initial energy level
        initial_energy = env.meters[0, 0].item()

        # Execute UP action (movement) - UP is typically index 0
        # In Grid2D: [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT, ...]
        up_action = 0
        actions = torch.tensor([up_action], device=env.device)

        # One step includes: base_depletion + movement cost
        obs, rewards, dones, info = env.step(actions)

        # Expected depletion:
        # base_depletion (from bars.yaml): 0.005 (for energy)
        # base_move_depletion (from bars.yaml): 0.005 (for energy)
        # Total: 0.01 (1% per movement tick)

        expected_energy = initial_energy - 0.01
        actual_energy = env.meters[0, 0].item()

        assert abs(actual_energy - expected_energy) < 1e-4, (
            f"Movement should apply base_depletion + base_move_depletion. " f"Expected {expected_energy:.4f}, got {actual_energy:.4f}"
        )

    def test_movement_costs_per_meter_from_bars(self, cpu_env_factory):
        """Only energy should have movement cost; other meters should not."""
        env = cpu_env_factory()
        env.reset()

        # Get initial meter state
        initial_meters = env.meters.clone()

        # Execute UP action
        up_action = 0
        actions = torch.tensor([up_action], device=env.device)
        env.step(actions)

        # Calculate meter changes
        meter_deltas = initial_meters - env.meters

        # Expected: Only energy (index 0) should have base_move_depletion applied
        # Energy: base_depletion (0.005) + base_move_depletion (0.005) = 0.01
        # Others: only base_depletion (various rates)

        # Energy should have depleted more than just base_depletion
        energy_delta = meter_deltas[0, 0].item()
        assert energy_delta > 0.005, f"Energy should deplete by base_depletion + base_move_depletion. " f"Got delta: {energy_delta:.4f}"


class TestInteractionCosts:
    """Test that INTERACT action applies base_interaction_cost from bars.yaml."""

    def test_interact_applies_base_interaction_cost(self, cpu_env_factory):
        """INTERACT action should apply base_interaction_cost from bars.yaml."""
        env = cpu_env_factory()
        obs = env.reset()

        # Get initial energy level
        initial_energy = env.meters[0, 0].item()

        # Execute INTERACT action - typically index 4 in Grid2D
        # In Grid2D: [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT, ...]
        interact_action = 4
        actions = torch.tensor([interact_action], device=env.device)

        # One step includes: base_depletion + interaction cost
        obs, rewards, dones, info = env.step(actions)

        # Expected depletion:
        # base_depletion (from bars.yaml): 0.005 (for energy)
        # base_interaction_cost (from bars.yaml): 0.005 (for energy)
        # Total: 0.01 (1% per interaction tick)

        expected_energy = initial_energy - 0.01
        actual_energy = env.meters[0, 0].item()

        assert abs(actual_energy - expected_energy) < 1e-4, (
            f"INTERACT should apply base_depletion + base_interaction_cost. " f"Expected {expected_energy:.4f}, got {actual_energy:.4f}"
        )


class TestWaitActionIsolation:
    """Test that WAIT action only pays base_depletion (no extra costs)."""

    def test_wait_only_pays_base_depletion(self, cpu_env_factory):
        """WAIT action should NOT apply any movement or interaction costs."""
        env = cpu_env_factory()
        obs = env.reset()

        # Get initial energy level
        initial_energy = env.meters[0, 0].item()

        # Execute WAIT action - typically index 5 in Grid2D
        # In Grid2D: [UP, DOWN, LEFT, RIGHT, INTERACT, WAIT, ...]
        wait_action = 5
        actions = torch.tensor([wait_action], device=env.device)

        # One step includes: only base_depletion (no action costs)
        obs, rewards, dones, info = env.step(actions)

        # Expected depletion:
        # base_depletion (from bars.yaml): 0.005 (for energy)
        # NO base_move_depletion (WAIT doesn't move)
        # NO base_interaction_cost (WAIT doesn't interact)
        # Total: 0.005 (0.5% per WAIT tick)

        expected_energy = initial_energy - 0.005
        actual_energy = env.meters[0, 0].item()

        assert abs(actual_energy - expected_energy) < 1e-4, (
            f"WAIT should only apply base_depletion. " f"Expected {expected_energy:.4f}, got {actual_energy:.4f}"
        )

    def test_wait_vs_movement_cost_difference(self, cpu_env_factory):
        """WAIT should cost less than movement (demonstrates meaningful action choice)."""
        env = cpu_env_factory()

        # Test WAIT
        env.reset()
        initial_energy_wait = env.meters[0, 0].item()
        wait_action = 5
        env.step(torch.tensor([wait_action], device=env.device))
        wait_energy_cost = initial_energy_wait - env.meters[0, 0].item()

        # Test UP (movement)
        env.reset()
        initial_energy_move = env.meters[0, 0].item()
        up_action = 0
        env.step(torch.tensor([up_action], device=env.device))
        move_energy_cost = initial_energy_move - env.meters[0, 0].item()

        # Movement should cost more than WAIT (by base_move_depletion amount)
        assert move_energy_cost > wait_energy_cost, f"Movement ({move_energy_cost:.4f}) should cost more than WAIT ({wait_energy_cost:.4f})"

        # Specifically, movement should cost base_move_depletion (0.005) more
        expected_difference = 0.005
        actual_difference = move_energy_cost - wait_energy_cost

        assert abs(actual_difference - expected_difference) < 1e-4, (
            f"Movement should cost 0.005 more than WAIT. " f"Expected difference: {expected_difference:.4f}, got: {actual_difference:.4f}"
        )
