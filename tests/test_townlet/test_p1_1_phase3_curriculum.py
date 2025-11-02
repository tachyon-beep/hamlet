"""
P1.1 Phase 3: Add Curriculum State to Checkpoints (RED → GREEN)

Goal: Save and restore curriculum progression (agent stages, performance trackers).

Current Gap:
- Runner doesn't save curriculum state
- On restart, agents reset to stage 1
- All progression history lost

This phase adds curriculum.state_dict() to checkpoints.
"""

import tempfile
from pathlib import Path

import torch
import yaml

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


class TestCurriculumStateCheckpointing:
    """Test that curriculum progression is saved and restored."""

    def test_curriculum_has_state_dict_method(self):
        """Verify curriculum has state_dict() method."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        curriculum.initialize_population(1)

        # Verify method exists
        assert hasattr(curriculum, "state_dict"), "Curriculum should have state_dict()"

        state = curriculum.state_dict()
        assert isinstance(state, dict), "state_dict should return dict"

        # Verify expected keys (tracker fields are returned directly)
        assert "agent_stages" in state, "Should have agent_stages"
        assert "episode_rewards" in state, "Should have episode_rewards"
        assert "steps_at_stage" in state, "Should have steps_at_stage"

    def test_curriculum_has_load_state_dict_method(self):
        """Verify curriculum has load_state_dict() method."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        curriculum.initialize_population(1)

        # Verify method exists
        assert hasattr(curriculum, "load_state_dict"), "Curriculum should have load_state_dict()"

    def test_curriculum_state_preserves_agent_stages(self):
        """Curriculum stages should be preserved across save/load."""
        curriculum1 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        curriculum1.initialize_population(1)

        # Directly set curriculum stage to test persistence
        curriculum1.tracker.agent_stages[0] = 3
        curriculum1.tracker.steps_at_stage[0] = 5000

        original_stage = curriculum1.tracker.agent_stages[0].item()
        assert original_stage == 3, "Should be at stage 3"

        # Save state
        state = curriculum1.state_dict()

        # Create new curriculum and load
        curriculum2 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        curriculum2.initialize_population(1)

        # Verify fresh curriculum starts at stage 1
        fresh_stage = curriculum2.tracker.agent_stages[0].item()
        assert fresh_stage == 1, "Fresh curriculum should start at stage 1"

        # Load saved state
        curriculum2.load_state_dict(state)

        # Verify stage was restored
        restored_stage = curriculum2.tracker.agent_stages[0].item()
        assert restored_stage == original_stage, f"Stage should be restored: {original_stage} vs {restored_stage}"

    def test_runner_checkpoint_includes_curriculum_state(self):
        """RED TEST: Runner checkpoint should include curriculum state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            # Create config
            config = {
                "environment": {"grid_size": 8, "partial_observability": False},
                "population": {"num_agents": 1, "network_type": "simple"},
                "curriculum": {"max_steps_per_episode": 100},
                "exploration": {},
            }
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            from townlet.demo.runner import DemoRunner

            runner = DemoRunner(
                config_dir=config_path.parent,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
                training_config_path=config_path,
            )

            # Manually initialize components
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

            # Directly set curriculum stage to test persistence
            runner.curriculum.tracker.agent_stages[0] = 3
            runner.curriculum.tracker.steps_at_stage[0] = 5000

            original_stage = runner.curriculum.tracker.agent_stages[0].item()
            assert original_stage == 3, "Should be at stage 3"

            # Save checkpoint
            runner.save_checkpoint()

            # Load checkpoint
            checkpoint_file = list(checkpoint_dir.glob("*.pt"))[0]
            checkpoint = torch.load(checkpoint_file, weights_only=False)

            # RED TEST: This will FAIL - curriculum_state not yet added
            assert "curriculum_state" in checkpoint, "Checkpoint should include curriculum_state"

    def test_runner_restores_curriculum_state(self):
        """RED TEST: Runner should restore curriculum progression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()

            config = {
                "environment": {"grid_size": 8, "partial_observability": False},
                "population": {"num_agents": 1, "network_type": "simple"},
                "curriculum": {"max_steps_per_episode": 100},
                "exploration": {},
            }
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            from townlet.demo.runner import DemoRunner

            # First runner - advance and save
            runner1 = DemoRunner(
                config_dir=config_path.parent,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
                training_config_path=config_path,
            )

            device = torch.device("cpu")
            runner1.env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=device, partial_observability=False)
            runner1.curriculum = AdversarialCurriculum(
                max_steps_per_episode=100,
                survival_advance_threshold=0.7,
                survival_retreat_threshold=0.3,
            )
            runner1.curriculum.initialize_population(1)
            runner1.exploration = AdaptiveIntrinsicExploration(obs_dim=runner1.env.observation_dim, device=device)
            runner1.population = VectorizedPopulation(
                env=runner1.env,
                curriculum=runner1.curriculum,
                exploration=runner1.exploration,
                agent_ids=["agent_0"],
                device=device,
                obs_dim=runner1.env.observation_dim,
                action_dim=runner1.env.action_dim,
            )

            # Directly set curriculum stage
            runner1.curriculum.tracker.agent_stages[0] = 4
            runner1.curriculum.tracker.steps_at_stage[0] = 6000

            original_stage = runner1.curriculum.tracker.agent_stages[0].item()
            runner1.save_checkpoint()

            # Second runner - load and verify
            runner2 = DemoRunner(
                config_dir=config_path.parent,
                db_path=tmpdir / "test2.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
                training_config_path=config_path,
            )

            runner2.env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=device, partial_observability=False)
            runner2.curriculum = AdversarialCurriculum(
                max_steps_per_episode=100,
                survival_advance_threshold=0.7,
                survival_retreat_threshold=0.3,
            )
            runner2.curriculum.initialize_population(1)
            runner2.exploration = AdaptiveIntrinsicExploration(obs_dim=runner2.env.observation_dim, device=device)
            runner2.population = VectorizedPopulation(
                env=runner2.env,
                curriculum=runner2.curriculum,
                exploration=runner2.exploration,
                agent_ids=["agent_0"],
                device=device,
                obs_dim=runner2.env.observation_dim,
                action_dim=runner2.env.action_dim,
            )

            # Verify starts at stage 1
            assert runner2.curriculum.tracker.agent_stages[0].item() == 1

            # Load checkpoint
            runner2.load_checkpoint()

            # RED TEST: This will FAIL - curriculum not restored yet
            restored_stage = runner2.curriculum.tracker.agent_stages[0].item()
            assert restored_stage == original_stage, f"Stage should match: {original_stage} vs {restored_stage}"
