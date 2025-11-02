"""
P1.1 Phase 4: Add Affordance Layout to Checkpoints (RED → GREEN)

Goal: Save and restore affordance positions on the grid.

Current Gap:
- Runner doesn't save affordance positions
- On restart, affordances spawn at random locations
- Agent must re-learn environment layout

This phase adds affordance layout to checkpoints.
"""

import tempfile
from pathlib import Path

import torch
import yaml

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestAffordanceLayoutCheckpointing:
    """Test that affordance positions are saved and restored."""

    def test_environment_has_get_affordance_positions_method(self):
        """Verify environment has get_affordance_positions() method."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
        )

        # Verify method exists
        assert hasattr(env, "get_affordance_positions"), "Environment should have get_affordance_positions()"

        positions = env.get_affordance_positions()
        assert isinstance(positions, dict), "Should return dict"
        assert len(positions) == 14, "Should have 14 affordances"
        # Verify each position has x,y coordinates
        for name, pos in positions.items():
            assert isinstance(pos, list), f"{name} position should be list"
            assert len(pos) == 2, f"{name} position should have x,y"

    def test_environment_has_set_affordance_positions_method(self):
        """Verify environment has set_affordance_positions() method."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
        )

        # Verify method exists
        assert hasattr(env, "set_affordance_positions"), "Environment should have set_affordance_positions()"

    def test_affordance_positions_preserved_across_set_get(self):
        """Affordance positions should be preserved across set/get."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
        )

        # Get original positions
        original_positions = env.get_affordance_positions()

        # Create modified positions - move Bed from wherever it is to (5, 5)
        modified_positions = {k: v[:] for k, v in original_positions.items()}  # Deep copy
        bed_original = modified_positions["Bed"][:]
        modified_positions["Bed"] = [5, 5]

        # Set new positions
        env.set_affordance_positions(modified_positions)

        # Get positions again
        restored_positions = env.get_affordance_positions()

        # Verify Bed moved to [5, 5]
        assert restored_positions["Bed"] == [5, 5], "Bed position should be updated to [5, 5]"

        # Verify it differs from original
        assert restored_positions["Bed"] != bed_original, "Bed position should differ from original"

    def test_runner_checkpoint_includes_affordance_layout(self):
        """RED TEST: Runner checkpoint should include affordance_layout."""
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

            from townlet.curriculum.adversarial import AdversarialCurriculum
            from townlet.demo.runner import DemoRunner
            from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
            from townlet.population.vectorized import VectorizedPopulation

            runner = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
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

            # Get original positions
            original_positions = runner.env.get_affordance_positions()
            assert original_positions is not None

            # Save checkpoint
            runner.save_checkpoint()

            # Load checkpoint
            checkpoint_file = list(checkpoint_dir.glob("*.pt"))[0]
            checkpoint = torch.load(checkpoint_file, weights_only=False)

            # RED TEST: This will FAIL - affordance_layout not yet added
            assert "affordance_layout" in checkpoint, "Checkpoint should include affordance_layout"

    def test_runner_restores_affordance_layout(self):
        """RED TEST: Runner should restore affordance positions."""
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

            from townlet.curriculum.adversarial import AdversarialCurriculum
            from townlet.demo.runner import DemoRunner
            from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
            from townlet.population.vectorized import VectorizedPopulation

            # First runner - save positions
            runner1 = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
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

            # Get original positions and modify them - move Bed to (7, 7)
            original_positions = runner1.env.get_affordance_positions()
            modified_positions = {k: v[:] for k, v in original_positions.items()}
            modified_positions["Bed"] = [7, 7]
            runner1.env.set_affordance_positions(modified_positions)

            runner1.save_checkpoint()

            # Second runner - load and verify
            runner2 = DemoRunner(
                config_path=config_path,
                db_path=tmpdir / "test2.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
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

            # Load checkpoint
            runner2.load_checkpoint()

            # Verify Bed position was restored to [7, 7]
            restored_positions = runner2.env.get_affordance_positions()
            assert restored_positions["Bed"] == [7, 7], "Bed position should match saved position [7, 7]"
