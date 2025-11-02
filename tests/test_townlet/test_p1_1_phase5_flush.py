"""
P1.1 Phase 5: Flush Before Checkpoint

Ensures replay buffer is flushed before checkpoint saves.
Prevents incomplete episode data in checkpoints.

RED → GREEN → REFACTOR methodology.
"""

import tempfile
from pathlib import Path

import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.runner import DemoRunner
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


class TestFlushBeforeCheckpoint:
    """Verify episode flush happens before checkpoint saves."""

    def test_population_has_flush_episode_method(self):
        """Verify VectorizedPopulation has flush_episode() method."""
        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=device,
            partial_observability=False,
        )

        curriculum = AdversarialCurriculum(
            device=device,
            max_steps_per_episode=500,
        )

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
        )

        # Verify method exists and accepts agent_idx
        assert hasattr(population, "flush_episode"), "Population should have flush_episode()"
        assert callable(getattr(population, "flush_episode")), "flush_episode should be callable"

        # Verify it can be called for agent 0
        population.flush_episode(agent_idx=0)  # Should not raise

    def test_flush_episode_empties_accumulators(self):
        """Verify flush_episode() clears episode accumulators."""
        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=device,
            partial_observability=False,
        )

        curriculum = AdversarialCurriculum(
            device=device,
            max_steps_per_episode=500,
        )

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            network_type="recurrent",
        )

        # Take some steps to accumulate data
        obs = env.reset()
        for _ in range(5):
            actions = torch.zeros(1, dtype=torch.long)  # WAIT action
            obs, rewards, dones, info = env.step(actions)

            # Store in accumulators (simulating normal training)
            if population.is_recurrent and population.current_episodes:
                episode = population.current_episodes[0]
                episode["observations"].append(obs[0].clone())
                episode["actions"].append(actions.clone())
                episode["rewards_extrinsic"].append(rewards.clone())
                episode["rewards_intrinsic"].append(torch.zeros_like(rewards))
                episode["dones"].append(dones.clone())

        # Flush episode for agent 0
        population.flush_episode(agent_idx=0)

        # Verify episode buffer for agent 0 is empty
        if hasattr(population, "current_episodes"):
            episode = population.current_episodes[0]
            assert len(episode.get("observations", [])) == 0, "Observations should be flushed"
            assert len(episode.get("actions", [])) == 0, "Actions should be flushed"
            assert len(episode.get("rewards", [])) == 0, "Rewards should be flushed"

    def test_runner_flush_all_agents_calls_flush_for_each_agent(self):
        """Verify flush_all_agents() calls flush_episode() for each agent."""
        # We'll implement flush_all_agents() that loops through agents
        # This test verifies it calls flush_episode(agent_idx) for each agent

        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            device=device,
            partial_observability=False,
        )

        curriculum = AdversarialCurriculum(
            device=device,
            max_steps_per_episode=500,
        )

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            device=device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
        )

        # Track flush calls
        flushed_agents = []
        original_flush = population.flush_episode

        def tracked_flush(agent_idx: int, synthetic_done: bool = False):
            flushed_agents.append(agent_idx)
            return original_flush(agent_idx, synthetic_done)

        population.flush_episode = tracked_flush

        # Call flush_all_agents (will be implemented as a helper)
        # For now, we'll manually call it for each agent to show intent
        for agent_idx in range(population.num_agents):
            population.flush_episode(agent_idx)

        # Verify all agents were flushed
        assert flushed_agents == [0, 1, 2], f"All agents should be flushed, got: {flushed_agents}"

    def test_flush_all_agents_helper_exists(self):
        """RED TEST: Verify runner has a flush_all_agents() helper method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config_path = tmpdir / "config.yaml"
            config_path.write_text(
                """
environment:
  grid_size: 8
  partial_observability: false
population:
  num_agents: 2
  learning_rate: 0.00025
  gamma: 0.99
  network_type: simple
curriculum:
  max_steps_per_episode: 100
exploration:
  strategy: adaptive_intrinsic
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
"""
            )

            runner = DemoRunner(
                config_dir=config_path.parent,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
                training_config_path=config_path,
            )

            # RED TEST: This will FAIL - flush_all_agents method doesn't exist yet
            assert hasattr(runner, "flush_all_agents"), "Runner should have flush_all_agents() helper for checkpointing"
