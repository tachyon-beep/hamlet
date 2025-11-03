"""
Tests for interoception-aware per-step survival reward calculation system.

Reward Formula:
- Alive: health × energy (both normalized to [0,1])
- Dead: 0.0

This models human interoception - awareness of internal body states.
"""

import torch

from townlet.environment.reward_strategy import RewardStrategy
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.runtime_registry import AgentRuntimeRegistry


def _attach_registry_with_baseline(env: VectorizedHamletEnv, baseline: float | list[float]) -> AgentRuntimeRegistry:
    """Helper to attach runtime registry to environment with preset baseline."""
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


class TestRewardCalculationViaEnv:
    """Test reward calculation through environment interface."""

    def test_milestone_every_10_steps(self):
        """Alive agents get health × energy reward."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Both alive at steps 10 and 20, full health/energy
        env.step_counts = torch.tensor([10, 20])
        env.dones = torch.tensor([False, False])
        env.meters[0, 0] = 100.0  # energy
        env.meters[0, 6] = 100.0  # health
        env.meters[1, 0] = 100.0
        env.meters[1, 6] = 100.0

        rewards = env._calculate_shaped_rewards()

        # Interoception: health × energy = 1.0 × 1.0 = 1.0
        assert torch.isclose(rewards[0], torch.tensor(1.0))
        assert torch.isclose(rewards[1], torch.tensor(1.0))

    def test_milestone_every_100_steps(self):
        """Alive agents get health × energy reward, regardless of step count."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Both alive at steps 100 and 200, full health/energy
        env.step_counts = torch.tensor([100, 200])
        env.dones = torch.tensor([False, False])
        env.meters[0, 0] = 100.0  # energy
        env.meters[0, 6] = 100.0  # health
        env.meters[1, 0] = 100.0
        env.meters[1, 6] = 100.0

        rewards = env._calculate_shaped_rewards()

        # Interoception: health × energy = 1.0 × 1.0 = 1.0
        assert torch.isclose(rewards[0], torch.tensor(1.0))
        assert torch.isclose(rewards[1], torch.tensor(1.0))

    def test_death_penalty(self):
        """Mixed alive/dead agents get correct rewards."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # One alive, one dead at step 50
        env.step_counts = torch.tensor([50, 50])
        env.dones = torch.tensor([False, True])
        env.meters[0, 0] = 100.0  # alive agent: full energy
        env.meters[0, 6] = 100.0  # alive agent: full health
        env.meters[1, 0] = 0.0    # dead agent: no energy
        env.meters[1, 6] = 0.0    # dead agent: no health

        rewards = env._calculate_shaped_rewards()

        # Alive agent: health × energy = 1.0 × 1.0 = 1.0
        assert torch.isclose(rewards[0], torch.tensor(1.0))
        # Dead agent: 0.0
        assert torch.isclose(rewards[1], torch.tensor(0.0))

    def test_dead_agents_no_milestone_bonus(self):
        """Dead agents receive 0.0 reward regardless of meters."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Dead agent at step 150
        env.step_counts = torch.tensor([150])
        env.dones = torch.tensor([True])
        env.meters[0, 0] = 50.0  # Energy doesn't matter when dead
        env.meters[0, 6] = 50.0  # Health doesn't matter when dead

        rewards = env._calculate_shaped_rewards()

        # Dead agent: 0.0
        assert torch.isclose(rewards[0], torch.tensor(0.0))

    def test_zero_steps_both_milestones(self):
        """Agents dying at step 0 get 0.0."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 100.0)

        # Both dead at step 0 (immediate death)
        env.step_counts = torch.tensor([0, 0])
        env.dones = torch.tensor([True, True])
        env.meters[0, 0] = 0.0  # No energy
        env.meters[0, 6] = 0.0  # No health
        env.meters[1, 0] = 0.0
        env.meters[1, 6] = 0.0

        rewards = env._calculate_shaped_rewards()

        # Dead agents: 0.0
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(0.0))

    def test_combined_milestones(self):
        """Dead agent gets 0.0 regardless of survival time and meters."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        _attach_registry_with_baseline(env, 50.0)

        # Agent dies at step 100
        env.step_counts = torch.tensor([100])
        env.dones = torch.tensor([True])
        env.meters[0, 0] = 25.0  # Meters don't matter when dead
        env.meters[0, 6] = 75.0

        rewards = env._calculate_shaped_rewards()

        # Dead agent: 0.0
        assert torch.isclose(rewards[0], torch.tensor(0.0))


class TestRewardStrategyDirect:
    """Test RewardStrategy class directly."""

    def test_decade_milestone_direct(self):
        """Alive agents get health × energy reward."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([10, 20, 30], device=device)
        dones = torch.tensor([False, False, False], device=device)
        baseline = torch.full((3,), 100.0, dtype=torch.float32, device=device)

        # Full health and energy for all agents
        meters = torch.zeros(3, 8, device=device)
        meters[:, 0] = 100.0  # energy
        meters[:, 6] = 100.0  # health

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)
        # health × energy = 1.0 × 1.0 = 1.0
        assert torch.allclose(rewards, torch.ones_like(rewards))

    def test_century_milestone_direct(self):
        """Dead agents get 0.0 reward."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=2)

        step_counts = torch.tensor([100, 200], device=device)
        dones = torch.tensor([True, True], device=device)
        baseline = torch.full((2,), 100.0, dtype=torch.float32, device=device)

        meters = torch.zeros(2, 8, device=device)
        meters[:, 0] = 0.0  # energy depleted
        meters[:, 6] = 0.0  # health depleted

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(0.0))

    def test_death_penalty_direct(self):
        """Dead agents get 0.0 reward."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=2)

        step_counts = torch.tensor([50, 80], device=device)
        dones = torch.tensor([True, True], device=device)
        baseline = torch.full((2,), 100.0, dtype=torch.float32, device=device)

        meters = torch.zeros(2, 8, device=device)
        meters[:, 0] = 10.0  # low energy (doesn't matter when dead)
        meters[:, 6] = 20.0  # low health (doesn't matter when dead)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(0.0))

    def test_mixed_alive_dead_direct(self):
        """Mixed alive/dead: alive get health × energy, dead get 0.0."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([50, 150, 100], device=device)
        dones = torch.tensor([False, True, True], device=device)
        baseline = torch.full((3,), 100.0, dtype=torch.float32, device=device)

        meters = torch.zeros(3, 8, device=device)
        meters[0, 0] = 100.0  # alive: full energy
        meters[0, 6] = 100.0  # alive: full health
        meters[1, 0] = 0.0    # dead: no energy
        meters[1, 6] = 0.0    # dead: no health
        meters[2, 0] = 0.0    # dead: no energy
        meters[2, 6] = 0.0    # dead: no health

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)
        assert torch.isclose(rewards[0], torch.tensor(1.0))  # alive: 1.0 × 1.0 = 1.0
        assert torch.isclose(rewards[1], torch.tensor(0.0))  # dead: 0.0
        assert torch.isclose(rewards[2], torch.tensor(0.0))  # dead: 0.0

    def test_step_zero_edge_case_direct(self):
        """Immediate death (step 0) gets 0.0 reward."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=2)

        step_counts = torch.tensor([0, 0], device=device)
        dones = torch.tensor([True, True], device=device)
        baseline = torch.full((2,), 100.0, dtype=torch.float32, device=device)

        meters = torch.zeros(2, 8, device=device)
        meters[:, 0] = 0.0  # no energy
        meters[:, 6] = 0.0  # no health

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)
        assert torch.allclose(rewards, torch.tensor([0.0, 0.0]))

    def test_non_milestone_steps_direct(self):
        """Alive agents get health × energy through RewardStrategy."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=4)

        step_counts = torch.tensor([5, 15, 25, 99], device=device)
        dones = torch.tensor([False, False, False, False], device=device)
        baseline = torch.full((4,), 100.0, dtype=torch.float32, device=device)

        meters = torch.zeros(4, 8, device=device)
        meters[:, 0] = 100.0  # full energy
        meters[:, 6] = 100.0  # full health

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)
        # health × energy = 1.0 × 1.0 = 1.0
        assert torch.allclose(rewards, torch.ones_like(rewards))

    def test_vectorized_baseline_support(self):
        """Baseline parameter retained for API compatibility but unused."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([100, 100, 100], device=device)
        dones = torch.tensor([True, True, True], device=device)
        baseline = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float32, device=device)

        meters = torch.zeros(3, 8, device=device)
        meters[:, 0] = 50.0  # meters don't matter when dead
        meters[:, 6] = 50.0

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)
        # All dead agents get 0.0 regardless of baseline or meters
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(0.0))
        assert torch.isclose(rewards[2], torch.tensor(0.0))
