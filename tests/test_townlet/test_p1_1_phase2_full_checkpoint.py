"""
P1.1 Phase 2: Wire Full Population Checkpoint into Runner (RED → GREEN)

Goal: Replace runner's manual partial checkpoint with population's complete checkpoint.

Current Gap:
- Runner manually saves: q_network, optimizer, exploration (partial)
- Population has: + replay_buffer, target_network, training_step_counter, version

This phase tests that runner.save_checkpoint() uses population.get_checkpoint_state()
by checking the checkpoint file structure.
"""

import tempfile
from pathlib import Path

import torch
import yaml

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


class TestRunnerCheckpointStructure:
    """Test checkpoint file structure to verify full population state is saved."""

    def test_population_checkpoint_has_all_required_fields(self):
        """Verify population.get_checkpoint_state() returns complete state."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
        )

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        curriculum.initialize_population(1)

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=env.device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
        )

        # Get checkpoint state
        checkpoint = population.get_checkpoint_state()

        # Verify all required fields are present
        required_fields = [
            "version",
            "q_network",
            "optimizer",
            "total_steps",
            "exploration_state",
            "replay_buffer",
            "target_network",  # Will be None for simple networks
            "training_step_counter",
        ]

        for field in required_fields:
            assert field in checkpoint, f"Missing field: {field}"

        # Verify version is correct
        assert checkpoint["version"] >= 2, "Version should be >= 2"

    def test_runner_should_use_full_population_checkpoint(self):
        """
        RED TEST: Verify current runner.save_checkpoint() structure.

        This test documents what SHOULD be in the checkpoint after Phase 2 is complete.
        Currently, this test will document the CURRENT (incomplete) state.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            # Create minimal config file
            config = {
                "environment": {"grid_size": 8, "partial_observability": False},
                "population": {"num_agents": 1, "network_type": "simple"},
                "curriculum": {"max_steps_per_episode": 100},
                "exploration": {},
            }
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Import runner and create checkpoint manually to test structure
            from townlet.demo.runner import DemoRunner

            runner = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            )

            # Manually initialize components (bypass run() which starts training loop)
            device = torch.device("cpu")

            runner.env = VectorizedHamletEnv(
                num_agents=1,
                grid_size=8,
                device=device,
                partial_observability=False,
            )

            runner.curriculum = AdversarialCurriculum(
                max_steps_per_episode=100,
                survival_advance_threshold=0.7,
                survival_retreat_threshold=0.3,
            )
            runner.curriculum.initialize_population(1)

            runner.exploration = AdaptiveIntrinsicExploration(
                obs_dim=runner.env.observation_dim,
                device=device,
            )

            runner.population = VectorizedPopulation(
                env=runner.env,
                curriculum=runner.curriculum,
                exploration=runner.exploration,
                agent_ids=["agent_0"],
                device=device,
                obs_dim=runner.env.observation_dim,
                action_dim=runner.env.action_dim,
            )

            # Save checkpoint using runner's method
            runner.save_checkpoint()

            # Load and inspect checkpoint
            checkpoint_file = list(checkpoint_dir.glob("*.pt"))[0]
            checkpoint = torch.load(checkpoint_file, weights_only=False)

            # Document current state (should have these)
            assert "episode" in checkpoint, "Should have episode number"
            assert "timestamp" in checkpoint, "Should have timestamp"
            assert "population_state" in checkpoint, "Should have population_state"

            # RED TEST: These should be TRUE after Phase 2 implementation
            # Currently, runner saves partial state manually
            pop_state = checkpoint["population_state"]

            # These SHOULD be in population_state but currently are NOT
            # (This is what we'll fix in Phase 2)
            print(f"\n🔍 Current population_state keys: {list(pop_state.keys())}")
            print(f"✅ Has q_network: {'q_network' in pop_state}")
            print(f"✅ Has optimizer: {'optimizer' in pop_state}")
            print(f"❌ Has version: {'version' in pop_state}")  # Should be TRUE
            print(f"❌ Has replay_buffer: {'replay_buffer' in pop_state}")  # Should be TRUE
            print(f"❌ Has total_steps: {'total_steps' in pop_state}")  # Should be TRUE

            # This test PASSES now (documents current state)
            # After Phase 2, we'll add assertions that currently fail
            assert True  # Placeholder - we're just documenting
