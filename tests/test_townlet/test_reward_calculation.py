"""
Tests for reward calculation in VectorizedHamletEnv.

Interoception-aware per-step survival reward system:
- Alive: health × energy (both normalized to [0,1])
- Dead: 0.0

This models human interoception - awareness of internal body states.
"""

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.runtime_registry import AgentRuntimeRegistry


def _attach_registry_with_baseline(env: VectorizedHamletEnv, baseline: float | list[float]) -> AgentRuntimeRegistry:
    registry = AgentRuntimeRegistry(
        agent_ids=[f"agent-{idx}" for idx in range(env.num_agents)],
        device=env.device,
    )
    env.attach_runtime_registry(registry)

    if isinstance(baseline, (float, int)):
        values = torch.full((env.num_agents,), float(baseline), dtype=torch.float32, device=env.device)
    else:
        values = torch.tensor(baseline, dtype=torch.float32, device=env.device)

    registry.set_baselines(values)
    return registry


class TestPerStepSurvivalRewards:
    """Test per-step survival reward system."""

    def test_reward_while_alive(self):
        """Alive agent gets health × energy reward per step."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Agent alive at step 50 with full health/energy
        env.step_counts[0] = 50
        env.dones[0] = False
        env.meters[0, 0] = 100.0  # energy
        env.meters[0, 6] = 100.0  # health

        rewards = env._calculate_shaped_rewards()

        # Alive: health × energy = 1.0 × 1.0 = 1.0
        assert abs(rewards[0] - 1.0) < 1e-6

    def test_reward_on_death_short_survival(self):
        """Dead agent gets 0.0 reward."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Agent dies at step 50
        env.step_counts[0] = 50
        env.dones[0] = True
        env.meters[0, 0] = 0.0  # depleted energy
        env.meters[0, 6] = 0.0  # depleted health

        rewards = env._calculate_shaped_rewards()

        # Dead agent: 0.0
        assert abs(rewards[0] - 0.0) < 1e-6

    def test_reward_on_death_long_survival(self):
        """Dead agent gets 0.0 reward."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Agent dies at step 150
        env.step_counts[0] = 150
        env.dones[0] = True
        env.meters[0, 0] = 0.0  # depleted energy
        env.meters[0, 6] = 0.0  # depleted health

        rewards = env._calculate_shaped_rewards()

        # Dead agent: 0.0
        assert abs(rewards[0] - 0.0) < 1e-6

    def test_reward_at_baseline_is_zero(self):
        """Dead agent gets zero reward regardless of baseline."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Agent dies at exactly baseline
        env.step_counts[0] = 100
        env.dones[0] = True
        env.meters[0, 0] = 0.0  # depleted energy
        env.meters[0, 6] = 0.0  # depleted health

        rewards = env._calculate_shaped_rewards()

        # Dead: 0.0
        assert abs(rewards[0] - 0.0) < 1e-6

    def test_multiple_agents_different_survival_times(self):
        """All dead agents get 0.0 reward."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # All agents die at different steps
        env.step_counts = torch.tensor([50, 100, 150])
        env.dones = torch.tensor([True, True, True])
        env.meters[:, 0] = 0.0  # all depleted energy
        env.meters[:, 6] = 0.0  # all depleted health

        rewards = env._calculate_shaped_rewards()

        # All dead agents get 0.0
        assert abs(rewards[0] - 0.0) < 1e-6
        assert abs(rewards[1] - 0.0) < 1e-6
        assert abs(rewards[2] - 0.0) < 1e-6

    def test_only_dead_agents_get_rewards(self):
        """Alive agents get health × energy, dead agents get 0.0."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # All at step 150, but only agent 1 is dead
        env.step_counts = torch.tensor([150, 150, 150])
        env.dones = torch.tensor([False, True, False])

        # Alive agents: full energy and health
        env.meters[0, 0] = 100.0  # energy
        env.meters[0, 6] = 100.0  # health
        env.meters[2, 0] = 100.0
        env.meters[2, 6] = 100.0

        # Dead agent: depleted
        env.meters[1, 0] = 0.0
        env.meters[1, 6] = 0.0

        rewards = env._calculate_shaped_rewards()

        # Alive agents get health × energy = 1.0 × 1.0 = 1.0
        assert abs(rewards[0] - 1.0) < 1e-6
        assert abs(rewards[2] - 1.0) < 1e-6

        # Dead agent gets 0.0
        assert abs(rewards[1] - 0.0) < 1e-6

    def test_baseline_affects_reward_calculation(self):
        """Baseline parameter is retained for API compatibility but unused."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=5,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Both agents survive 100 steps and die
        env.step_counts = torch.tensor([100, 100])
        env.dones = torch.tensor([True, True])
        env.meters[:, 0] = 0.0  # depleted energy
        env.meters[:, 6] = 0.0  # depleted health

        registry = _attach_registry_with_baseline(env, 50.0)
        rewards_low_baseline = env._calculate_shaped_rewards()

        registry.set_baselines(torch.full((env.num_agents,), 150.0, dtype=torch.float32, device=env.device))
        rewards_high_baseline = env._calculate_shaped_rewards()

        # Both dead: 0.0 regardless of baseline
        assert abs(rewards_low_baseline[0] - 0.0) < 1e-6
        assert abs(rewards_high_baseline[0] - 0.0) < 1e-6


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
