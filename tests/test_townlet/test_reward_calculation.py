"""
Tests for reward calculation in VectorizedHamletEnv.

Focus on milestone survival rewards (active system).
"""

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestMilestoneRewards:
    """Test milestone-based survival reward system."""

    def test_no_milestone_no_bonus(self):
        """No bonus if not at milestone."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Step to 5 (not a milestone)
        env.step_counts[0] = 5
        env.dones[0] = False

        rewards = env._calculate_shaped_rewards()

        # No milestone, no bonus
        assert abs(rewards[0] - 0.0) < 1e-6

    def test_decade_milestone_bonus(self):
        """Every 10 steps gives +0.5 bonus."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Agent 0 at step 10, agent 1 at step 20
        env.step_counts[0] = 10
        env.step_counts[1] = 20
        env.dones[0] = False
        env.dones[1] = False

        rewards = env._calculate_shaped_rewards()

        # Both get decade milestone bonus
        assert abs(rewards[0] - 0.5) < 1e-6
        assert abs(rewards[1] - 0.5) < 1e-6

    def test_century_milestone_bonus(self):
        """Every 100 steps gives +5.0 bonus."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Step 100 (both decade AND century milestone)
        env.step_counts[0] = 100
        env.dones[0] = False

        rewards = env._calculate_shaped_rewards()

        # Gets both bonuses: 0.5 (decade) + 5.0 (century) = 5.5
        assert abs(rewards[0] - 5.5) < 1e-6

    def test_century_milestone_at_200(self):
        """Century milestone works at 200, 300, etc."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Step 200
        env.step_counts[0] = 200
        env.dones[0] = False

        rewards = env._calculate_shaped_rewards()

        # Gets both bonuses
        assert abs(rewards[0] - 5.5) < 1e-6

    def test_death_penalty_overrides_milestones(self):
        """Death gives -100.0 penalty regardless of milestones."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Step 100 (would be +5.5 milestone) BUT agent died
        env.step_counts[0] = 100
        env.dones[0] = True

        rewards = env._calculate_shaped_rewards()

        # Death penalty overrides everything
        assert abs(rewards[0] - (-100.0)) < 1e-6

    def test_only_alive_agents_get_milestones(self):
        """Dead agents don't get milestone bonuses."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # All at step 10, but agent 1 is dead
        env.step_counts = torch.tensor([10, 10, 10])
        env.dones = torch.tensor([False, True, False])

        rewards = env._calculate_shaped_rewards()

        # Alive agents get bonus
        assert abs(rewards[0] - 0.5) < 1e-6
        assert abs(rewards[2] - 0.5) < 1e-6

        # Dead agent gets death penalty
        assert abs(rewards[1] - (-100.0)) < 1e-6

    def test_milestone_30_is_decade_not_century(self):
        """Step 30 is a decade milestone, not century."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        env.step_counts[0] = 30
        env.dones[0] = False

        rewards = env._calculate_shaped_rewards()

        # Only decade bonus
        assert abs(rewards[0] - 0.5) < 1e-6

    def test_milestone_50_is_decade_not_century(self):
        """Step 50 is a decade milestone, not century."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        env.step_counts[0] = 50
        env.dones[0] = False

        rewards = env._calculate_shaped_rewards()

        # Only decade bonus
        assert abs(rewards[0] - 0.5) < 1e-6


class TestDeathConditions:
    """Test terminal condition checking."""

    def test_health_zero_causes_death(self):
        """Agent dies when health reaches 0."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set health to 0
        env.meters[0, 6] = 0.0
        env.meters[0, 0] = 0.5  # Energy is fine

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0]

    def test_energy_zero_causes_death(self):
        """Agent dies when energy reaches 0."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set energy to 0
        env.meters[0, 0] = 0.0
        env.meters[0, 6] = 0.5  # Health is fine

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0]

    def test_both_zero_causes_death(self):
        """Agent dies when both health and energy reach 0."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set both to 0
        env.meters[0, 0] = 0.0
        env.meters[0, 6] = 0.0

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0]

    def test_low_but_positive_doesnt_cause_death(self):
        """Agent survives with very low but positive health and energy."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set to very low but positive
        env.meters[0, 0] = 0.01  # Energy barely positive
        env.meters[0, 6] = 0.01  # Health barely positive
        env.dones[0] = False

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        # Should NOT be dead
        assert not env.dones[0]

    def test_multiple_agents_independent_death(self):
        """Each agent's death is independent."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Agent 0: health=0, energy=1 (DEAD)
        # Agent 1: health=1, energy=0 (DEAD)
        # Agent 2: health=0.5, energy=0.5 (ALIVE)
        env.meters[0, 6] = 0.0
        env.meters[0, 0] = 1.0
        env.meters[1, 6] = 1.0
        env.meters[1, 0] = 0.0
        env.meters[2, 6] = 0.5
        env.meters[2, 0] = 0.5

        env.dones = env.meter_dynamics.check_terminal_conditions(env.meters, env.dones)

        assert env.dones[0]  # Dead from health
        assert env.dones[1]  # Dead from energy
        assert not env.dones[2]  # Alive
