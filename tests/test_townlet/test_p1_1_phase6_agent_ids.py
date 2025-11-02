"""
P1.1 Phase 6: Multi-Agent Preparation

Adds agent_ids field to checkpoints for future multi-agent support.
Ensures curriculum state can be properly indexed by agent ID.

RED → GREEN → REFACTOR methodology.
"""

import tempfile
import torch
from pathlib import Path

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.demo.runner import DemoRunner
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


class TestAgentIdsCheckpointing:
    """Verify agent_ids are saved in checkpoints for multi-agent coordination."""

    def test_checkpoint_includes_agent_ids(self):
        """RED TEST: Verify checkpoint includes agent_ids field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            # Create config with 2 agents
            config_path = tmpdir / "config.yaml"
            config_path.write_text("""
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
""")

            from townlet.curriculum.adversarial import AdversarialCurriculum
            from townlet.environment.vectorized_env import VectorizedHamletEnv
            from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
            from townlet.population.vectorized import VectorizedPopulation

            runner = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            )

            # Initialize components manually
            device = torch.device("cpu")
            runner.env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=device, partial_observability=False)
            runner.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
            runner.curriculum.initialize_population(2)
            runner.exploration = AdaptiveIntrinsicExploration(obs_dim=runner.env.observation_dim, device=device)
            runner.population = VectorizedPopulation(
                env=runner.env,
                curriculum=runner.curriculum,
                exploration=runner.exploration,
                agent_ids=["agent_0", "agent_1"],
                device=device,
                obs_dim=runner.env.observation_dim,
                action_dim=runner.env.action_dim,
            )

            # Save checkpoint
            runner.save_checkpoint()

            # Load checkpoint
            checkpoint_path = checkpoint_dir / "checkpoint_ep00000.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Verify agent_ids field exists
            assert "agent_ids" in checkpoint, "Checkpoint should include agent_ids field"

    def test_agent_ids_matches_population(self):
        """Verify agent_ids in checkpoint matches population agent_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config_path = tmpdir / "config.yaml"
            config_path.write_text("""
environment:
  grid_size: 8
  partial_observability: false

population:
  num_agents: 3
  learning_rate: 0.00025
  gamma: 0.99
  network_type: simple

curriculum:
  max_steps_per_episode: 100

exploration:
  strategy: adaptive_intrinsic
  initial_intrinsic_weight: 1.0
  variance_threshold: 100.0
""")

            from townlet.curriculum.adversarial import AdversarialCurriculum
            from townlet.environment.vectorized_env import VectorizedHamletEnv
            from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
            from townlet.population.vectorized import VectorizedPopulation

            runner = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            )

            # Initialize components manually
            device = torch.device("cpu")
            runner.env = VectorizedHamletEnv(num_agents=3, grid_size=8, device=device, partial_observability=False)
            runner.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
            runner.curriculum.initialize_population(3)
            runner.exploration = AdaptiveIntrinsicExploration(obs_dim=runner.env.observation_dim, device=device)
            runner.population = VectorizedPopulation(
                env=runner.env,
                curriculum=runner.curriculum,
                exploration=runner.exploration,
                agent_ids=["agent_0", "agent_1", "agent_2"],
                device=device,
                obs_dim=runner.env.observation_dim,
                action_dim=runner.env.action_dim,
            )

            # Save checkpoint
            runner.save_checkpoint()

            # Load checkpoint
            checkpoint_path = checkpoint_dir / "checkpoint_ep00000.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Verify agent_ids matches population
            agent_ids = checkpoint.get("agent_ids")
            assert agent_ids is not None, "agent_ids should be in checkpoint"
            assert len(agent_ids) == 3, "Should have 3 agent IDs"
            assert agent_ids == ["agent_0", "agent_1", "agent_2"], (
                f"Agent IDs should be ['agent_0', 'agent_1', 'agent_2'], got: {agent_ids}"
            )

    def test_curriculum_state_aligned_with_agent_ids(self):
        """Verify curriculum state can be indexed by agent_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config_path = tmpdir / "config.yaml"
            config_path.write_text("""
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
""")

            from townlet.curriculum.adversarial import AdversarialCurriculum
            from townlet.environment.vectorized_env import VectorizedHamletEnv
            from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
            from townlet.population.vectorized import VectorizedPopulation

            runner = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            )

            # Initialize components manually
            device = torch.device("cpu")
            runner.env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=device, partial_observability=False)
            runner.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
            runner.curriculum.initialize_population(2)
            runner.exploration = AdaptiveIntrinsicExploration(obs_dim=runner.env.observation_dim, device=device)
            runner.population = VectorizedPopulation(
                env=runner.env,
                curriculum=runner.curriculum,
                exploration=runner.exploration,
                agent_ids=["agent_0", "agent_1"],
                device=device,
                obs_dim=runner.env.observation_dim,
                action_dim=runner.env.action_dim,
            )

            # Modify curriculum stages for verification
            if runner.curriculum and hasattr(runner.curriculum, "tracker"):
                runner.curriculum.tracker.agent_stages[0] = 2
                runner.curriculum.tracker.agent_stages[1] = 3

            # Save checkpoint
            runner.save_checkpoint()

            # Load checkpoint
            checkpoint_path = checkpoint_dir / "checkpoint_ep00000.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Verify curriculum state matches agent count
            curriculum_state = checkpoint.get("curriculum_state", {})
            agent_stages = curriculum_state.get("agent_stages")
            agent_ids = checkpoint.get("agent_ids")

            assert agent_ids is not None, "agent_ids should be in checkpoint"
            assert agent_stages is not None, "curriculum should have agent_stages"

            # Verify tensor size matches number of agents
            assert len(agent_stages) == len(agent_ids), (
                f"Curriculum stages ({len(agent_stages)}) should match agent count ({len(agent_ids)})"
            )

    def test_runner_restores_agent_ids(self):
        """Verify runner can restore agent_ids from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config_path = tmpdir / "config.yaml"
            config_path.write_text("""
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
""")

            from townlet.curriculum.adversarial import AdversarialCurriculum
            from townlet.environment.vectorized_env import VectorizedHamletEnv
            from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
            from townlet.population.vectorized import VectorizedPopulation

            # First runner - save checkpoint
            runner1 = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test1.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            )

            device = torch.device("cpu")
            runner1.env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=device, partial_observability=False)
            runner1.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
            runner1.curriculum.initialize_population(2)
            runner1.exploration = AdaptiveIntrinsicExploration(obs_dim=runner1.env.observation_dim, device=device)
            runner1.population = VectorizedPopulation(
                env=runner1.env,
                curriculum=runner1.curriculum,
                exploration=runner1.exploration,
                agent_ids=["agent_0", "agent_1"],
                device=device,
                obs_dim=runner1.env.observation_dim,
                action_dim=runner1.env.action_dim,
            )

            original_agent_ids = runner1.population.agent_ids
            runner1.save_checkpoint()

            # Second runner - load checkpoint
            runner2 = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test2.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            )

            runner2.env = VectorizedHamletEnv(num_agents=2, grid_size=8, device=device, partial_observability=False)
            runner2.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
            runner2.curriculum.initialize_population(2)
            runner2.exploration = AdaptiveIntrinsicExploration(obs_dim=runner2.env.observation_dim, device=device)
            runner2.population = VectorizedPopulation(
                env=runner2.env,
                curriculum=runner2.curriculum,
                exploration=runner2.exploration,
                agent_ids=["agent_0", "agent_1"],
                device=device,
                obs_dim=runner2.env.observation_dim,
                action_dim=runner2.env.action_dim,
            )

            # Load checkpoint
            runner2.load_checkpoint()

            # Verify agent_ids were restored
            restored_agent_ids = runner2.population.agent_ids
            assert restored_agent_ids == original_agent_ids, f"Agent IDs should be restored correctly, got: {restored_agent_ids}"
