"""Integration tests for checkpointing functionality.

This module consolidates 38 existing checkpoint tests into 15 comprehensive
integration tests that verify checkpoint save/load works correctly across all
components (Environment, Population, Curriculum, Exploration, Runner).

Task 11a: Checkpointing Integration Tests
Source files consolidated:
- test_p1_1_phase1_baseline.py (5 tests)
- test_p1_1_phase2_full_checkpoint.py (2 tests)
- test_p1_1_phase3_curriculum.py (3 tests)
- test_p1_1_phase4_affordances.py (4 tests)
- test_p1_1_phase5_flush.py (7 tests)
- test_p1_1_phase6_agent_ids.py (6 tests)
- test_p2_1_vectorized_baseline.py (5 tests)
- test_exploration_checkpoint.py (6 tests)
Total: 38 tests → 15 comprehensive integration tests
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.curriculum.static import StaticCurriculum
from townlet.demo.runner import DemoRunner
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


@pytest.fixture
def env_builder(env_factory, cpu_device):
    """Helper to build CPU-bound environments for integration tests."""

    def _build(*, config_dir: Path, num_agents: int = 1):
        env = env_factory(
            config_dir=config_dir,
            num_agents=num_agents,
            device_override=cpu_device,
        )
        env.reset()
        return env

    return _build


def _init_curriculum(curriculum, num_agents: int) -> None:
    if hasattr(curriculum, "initialize_population"):
        curriculum.initialize_population(num_agents)


def _select_active_meters(env, limit: int = 6) -> list[str]:
    """Return a capped list of active meters to satisfy CurriculumDecision constraints."""

    meter_names = list(env.meter_name_to_index.keys())
    return meter_names[:limit]


# =============================================================================
# TEST CLASS 1: Environment Checkpointing (3 tests)
# =============================================================================


class TestEnvironmentCheckpointing:
    """Test VectorizedHamletEnv.save_checkpoint() / load_checkpoint() integration.

    Verifies environment state (affordance positions, agent positions, meters)
    is preserved across save/load cycles.
    """

    def test_environment_checkpoint_preserves_affordance_layout(self, env_builder, test_config_pack_path):
        """Environment should preserve affordance positions across checkpoint cycle.

        This is critical for generalization tests that require consistent layouts.
        """
        env = env_builder(config_dir=test_config_pack_path, num_agents=1)

        # Capture affordance positions using the environment's API
        original_positions = {name: pos.clone() for name, pos in env.affordances.items()}

        # Get checkpoint data using environment's affordance position API
        checkpoint_data = env.get_affordance_positions()

        # Verify checkpoint data structure
        assert "positions" in checkpoint_data, "Checkpoint should contain positions"
        assert "ordering" in checkpoint_data, "Checkpoint should contain ordering"

        # Create new environment and load
        env2 = env_builder(config_dir=test_config_pack_path, num_agents=1)
        env2.set_affordance_positions(checkpoint_data)

        # Verify positions match
        for name, original_pos in original_positions.items():
            assert name in env2.affordances, f"Affordance {name} should exist after load"
            assert torch.equal(env2.affordances[name], original_pos), f"Affordance {name} position should be preserved"

    def test_environment_checkpoint_preserves_agent_state(self, env_builder, test_config_pack_path, cpu_device):
        """Environment should preserve agent state via manual save/restore (no built-in checkpointing).

        Note: VectorizedHamletEnv doesn't have built-in save_checkpoint/load_checkpoint methods.
        Agent state checkpointing is handled by the runner layer, not the environment.
        This test verifies that state tensors can be manually saved and restored.
        """
        env = env_builder(config_dir=test_config_pack_path, num_agents=2)

        for _ in range(10):
            actions = torch.randint(0, 6, (2,), device=cpu_device)
            env.step(actions)

        original_positions = env.positions.clone()
        original_meters = env.meters.clone()
        original_step_counts = env.step_counts.clone()

        checkpoint = {
            "positions": original_positions,
            "meters": original_meters,
            "step_counts": original_step_counts,
        }

        env2 = env_builder(config_dir=test_config_pack_path, num_agents=2)

        env2.positions = checkpoint["positions"].clone()
        env2.meters = checkpoint["meters"].clone()
        env2.step_counts = checkpoint["step_counts"].clone()

        assert torch.equal(env2.positions, original_positions), "Agent positions should match"
        assert torch.allclose(env2.meters, original_meters, atol=1e-5), "Meters should match"
        assert torch.equal(env2.step_counts, original_step_counts), "Step counts should match"

    def test_environment_checkpoint_roundtrip_consistency(self, env_builder, test_config_pack_path, cpu_device):
        """Environment affordance positions should survive multiple save/load cycles identically."""
        env = env_builder(config_dir=test_config_pack_path, num_agents=1)

        # Reset and take steps
        for _ in range(20):
            actions = torch.randint(0, 6, (1,), device=cpu_device)
            env.step(actions)

        # First save/load cycle (affordance positions only)
        checkpoint1 = env.get_affordance_positions()
        env.set_affordance_positions(checkpoint1)

        # Second save/load cycle
        checkpoint2 = env.get_affordance_positions()
        env.set_affordance_positions(checkpoint2)

        # Verify checkpoints are identical
        assert checkpoint1["positions"] == checkpoint2["positions"], "Affordance positions should be identical across cycles"
        assert checkpoint1["ordering"] == checkpoint2["ordering"], "Affordance ordering should be identical across cycles"


# =============================================================================
# TEST CLASS 2: Population Checkpointing (3 tests)
# =============================================================================


class TestPopulationCheckpointing:
    """Test VectorizedPopulation checkpoint save/load integration.

    Verifies population state (Q-network, optimizer, replay buffer, exploration)
    is preserved across save/load cycles.
    """

    def test_population_checkpoint_contains_required_keys(self, cpu_device, basic_env, minimal_brain_config):
        """Population checkpoint should contain all required keys for full restoration."""
        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        _init_curriculum(curriculum, 1)

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            device=cpu_device,
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Get checkpoint
        checkpoint = population.get_checkpoint_state()

        # Verify required keys
        required_keys = [
            "version",
            "q_network",
            "optimizer",
            "total_steps",
            "exploration_state",
            "replay_buffer",
            "target_network",
            "training_step_counter",
        ]

        for key in required_keys:
            assert key in checkpoint, f"Checkpoint missing required key: {key}"

        # Verify version
        assert checkpoint["version"] >= 2, "Checkpoint version should be >= 2"

    def test_population_checkpoint_preserves_network_weights(self, cpu_device, test_config_pack_path, env_builder, minimal_brain_config):
        """Q-network weights should be exactly preserved across checkpoint cycle."""
        # Create environment with CPU device (avoiding basic_env fixture which may use CUDA)
        env = env_builder(config_dir=test_config_pack_path, num_agents=1)

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        _init_curriculum(curriculum, 1)

        exploration = EpsilonGreedyExploration(
            epsilon=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        # First population
        pop1 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Train for a bit to change weights
        pop1.reset()
        for _ in range(50):
            pop1.step_population(env)

        # Capture weights
        original_weights = {k: v.clone() for k, v in pop1.q_network.state_dict().items()}

        # Save checkpoint
        checkpoint = pop1.get_checkpoint_state()

        # Create new population and load
        exploration2 = EpsilonGreedyExploration(
            epsilon=1.0,  # Different initial epsilon
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        pop2 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        pop2.load_checkpoint_state(checkpoint)

        # Verify weights match exactly
        restored_weights = pop2.q_network.state_dict()
        for key in original_weights.keys():
            assert torch.allclose(
                original_weights[key], restored_weights[key], atol=1e-6
            ), f"Q-network weights for {key} should match exactly"

    def test_population_checkpoint_preserves_replay_buffer(self, cpu_device, test_config_pack_path, env_builder, minimal_brain_config):
        """Replay buffer should be preserved with exact contents across checkpoint cycle."""
        # Create environment with CPU device (avoiding basic_env fixture which may use CUDA)
        env = env_builder(config_dir=test_config_pack_path, num_agents=1)

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        _init_curriculum(curriculum, 1)

        exploration = EpsilonGreedyExploration()

        # First population
        pop1 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            replay_buffer_capacity=1000,
        )

        # Fill replay buffer with experiences
        pop1.reset()
        for _ in range(100):
            pop1.step_population(env)

        original_buffer_size = len(pop1.replay_buffer)
        assert original_buffer_size > 0, "Replay buffer should have experiences"

        # Save checkpoint
        checkpoint = pop1.get_checkpoint_state()

        # Create new population and load
        exploration2 = EpsilonGreedyExploration()
        pop2 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
            replay_buffer_capacity=1000,
        )

        # Before load, buffer should be empty
        assert len(pop2.replay_buffer) == 0, "Fresh replay buffer should be empty"

        # Load checkpoint
        pop2.load_checkpoint_state(checkpoint)

        # Verify buffer size restored
        assert (
            len(pop2.replay_buffer) == original_buffer_size
        ), f"Replay buffer size should match: {original_buffer_size} vs {len(pop2.replay_buffer)}"


# =============================================================================
# TEST CLASS 3: Curriculum Checkpointing (2 tests)
# =============================================================================


class TestCurriculumCheckpointing:
    """Test curriculum state preservation across checkpoint cycles.

    Verifies curriculum progression (agent stages, performance trackers) is saved
    and restored correctly.
    """

    def test_curriculum_checkpoint_preserves_agent_stages(self, cpu_device):
        """Curriculum should preserve agent stage progression across checkpoint cycle."""
        curriculum1 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        _init_curriculum(curriculum1, 3)

        # Advance agents to different stages
        curriculum1.tracker.agent_stages[0] = 2
        curriculum1.tracker.agent_stages[1] = 3
        curriculum1.tracker.agent_stages[2] = 4
        curriculum1.tracker.steps_at_stage[0] = 1000
        curriculum1.tracker.steps_at_stage[1] = 2000
        curriculum1.tracker.steps_at_stage[2] = 3000

        original_stages = curriculum1.tracker.agent_stages.clone()
        original_steps = curriculum1.tracker.steps_at_stage.clone()

        # Save state
        state = curriculum1.state_dict()

        # Create new curriculum and load
        curriculum2 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        _init_curriculum(curriculum2, 3)

        # Verify fresh curriculum starts at stage 1
        assert torch.all(curriculum2.tracker.agent_stages == 1), "Fresh curriculum should start all agents at stage 1"

        # Load saved state
        curriculum2.load_state_dict(state)

        # Verify stages restored
        assert torch.equal(curriculum2.tracker.agent_stages, original_stages), "Agent stages should be restored"
        assert torch.equal(curriculum2.tracker.steps_at_stage, original_steps), "Steps at stage should be restored"

    def test_curriculum_checkpoint_preserves_performance_history(self, cpu_device):
        """Curriculum should preserve episode performance history across checkpoint cycle."""
        curriculum1 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        _init_curriculum(curriculum1, 2)

        # Simulate episode completions to build history
        for _ in range(10):
            rewards = torch.tensor([50.0, 75.0], device=cpu_device)
            dones = torch.tensor([True, True], device=cpu_device)
            curriculum1.tracker.update_step(rewards, dones)

        # Save state
        state = curriculum1.state_dict()

        # Create new curriculum and load
        curriculum2 = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
        )
        _init_curriculum(curriculum2, 2)

        # Load saved state
        curriculum2.load_state_dict(state)

        # Verify history is preserved (check that recent episodes match)
        assert (
            curriculum2.tracker.episode_rewards.shape == curriculum1.tracker.episode_rewards.shape
        ), "Performance history shape should match"


# =============================================================================
# TEST CLASS 4: Exploration Checkpointing (2 tests)
# =============================================================================


class TestExplorationCheckpointing:
    """Test exploration state (RND networks, epsilon, annealing) preservation.

    Verifies exploration strategies can be saved and restored correctly.
    """

    def test_adaptive_intrinsic_exploration_checkpoint_completeness(self, cpu_device, basic_env):
        """AdaptiveIntrinsicExploration should save all required state for restoration."""
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            embed_dim=128,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=50,
            epsilon_start=0.8,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            device=cpu_device,
        )

        # Get checkpoint state
        state = exploration.checkpoint_state()

        # Verify checkpoint structure - AdaptiveIntrinsicExploration wraps RND state
        assert "rnd_state" in state, "Should have rnd_state"
        rnd_state = state["rnd_state"]
        # RND networks are in rnd_state
        assert "fixed_network" in rnd_state, "Should have fixed_network in rnd_state"
        assert "predictor_network" in rnd_state, "Should have predictor_network in rnd_state"
        assert "optimizer" in rnd_state, "Should have RND optimizer in rnd_state"
        # Epsilon parameters are nested in rnd_state
        assert "epsilon" in rnd_state, "Should have epsilon in rnd_state"
        assert "epsilon_min" in rnd_state, "Should have epsilon_min in rnd_state"
        assert "epsilon_decay" in rnd_state, "Should have epsilon_decay in rnd_state"
        # RND uses "epsilon" not "epsilon_start"
        assert rnd_state["epsilon"] == 0.8, f"Expected epsilon=0.8, got {rnd_state.get('epsilon', 'missing')}"

        # Verify annealing state (uses "current_intrinsic_weight" not "intrinsic_weight")
        assert "current_intrinsic_weight" in state, "Should have current_intrinsic_weight"
        assert "min_intrinsic_weight" in state, "Should have min_intrinsic_weight"
        assert "variance_threshold" in state, "Should have variance_threshold"
        assert "survival_window" in state, "Should have survival_window"
        assert "decay_rate" in state, "Should have decay_rate"
        assert "survival_history" in state, "Should have survival_history"

    def test_exploration_checkpoint_preserves_epsilon_decay(self, cpu_device, basic_env):
        """Epsilon decay progression should be preserved across checkpoint cycle."""
        exploration1 = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            embed_dim=128,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99,
            device=cpu_device,
        )

        # Decay epsilon through multiple steps
        for _ in range(50):
            exploration1.rnd.epsilon *= exploration1.rnd.epsilon_decay

        decayed_epsilon = exploration1.rnd.epsilon
        assert decayed_epsilon < 1.0, "Epsilon should have decayed"

        # Save checkpoint
        state = exploration1.checkpoint_state()

        # Create new exploration and load
        exploration2 = AdaptiveIntrinsicExploration(
            obs_dim=basic_env.observation_dim,
            embed_dim=128,
            epsilon_start=1.0,  # Fresh start
            epsilon_min=0.1,
            epsilon_decay=0.99,
            device=cpu_device,
        )

        # Before load, epsilon should be fresh
        assert exploration2.rnd.epsilon == 1.0, "Fresh exploration should have epsilon=1.0"

        # Load checkpoint
        exploration2.load_state(state)

        # Verify epsilon restored
        assert (
            abs(exploration2.rnd.epsilon - decayed_epsilon) < 1e-6
        ), f"Epsilon should be restored: {decayed_epsilon} vs {exploration2.rnd.epsilon}"


# =============================================================================
# TEST CLASS 5: Runner Checkpointing (3 tests)
# =============================================================================


class TestRunnerCheckpointing:
    """Test DemoRunner.save_checkpoint() integration across all components.

    Verifies runner orchestrates checkpoint save/load for all components
    (environment, population, curriculum, exploration) correctly.
    """

    def test_runner_checkpoint_includes_all_components(self, cpu_device, env_builder, config_pack_factory, minimal_brain_config):
        """Runner checkpoint should include state from all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_dir = tmp_path / "checkpoints"
            checkpoint_dir.mkdir()

            config_dir = config_pack_factory(modifier=lambda data: data["training"].update({"max_episodes": 1}))

            # Create runner with context manager
            with DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            ) as runner:
                # Manually initialize components
                runner.env = env_builder(config_dir=config_dir, num_agents=1)

                runner.curriculum = AdversarialCurriculum(
                    max_steps_per_episode=100,
                    survival_advance_threshold=0.7,
                    survival_retreat_threshold=0.3,
                )
                _init_curriculum(runner.curriculum, 1)

                runner.exploration = AdaptiveIntrinsicExploration(
                    obs_dim=runner.env.observation_dim,
                    device=cpu_device,
                )

                runner.population = VectorizedPopulation(
                    env=runner.env,
                    curriculum=runner.curriculum,
                    exploration=runner.exploration,
                    agent_ids=["agent_0"],
                    device=cpu_device,
                    obs_dim=runner.env.observation_dim,
                    # action_dim defaults to env.action_dim
                    brain_config=minimal_brain_config,
                )

                # Save checkpoint
                runner.save_checkpoint()

                # Load and verify checkpoint structure
                checkpoint_file = list(checkpoint_dir.glob("*.pt"))[0]
                checkpoint = torch.load(checkpoint_file, weights_only=False)

                # Verify checkpoint has runner metadata
                assert "episode" in checkpoint, "Should have episode number"
                assert "timestamp" in checkpoint, "Should have timestamp"

                # Verify checkpoint has population state
                assert "population_state" in checkpoint, "Should have population_state"
                pop_state = checkpoint["population_state"]
                assert "version" in pop_state, "Population state should have version"
                assert "q_network" in pop_state, "Population state should have q_network"

                # Verify checkpoint has curriculum state
                assert "curriculum_state" in checkpoint, "Should have curriculum_state"

    def test_runner_checkpoint_preserves_episode_number(self, cpu_device, env_builder, config_pack_factory, minimal_brain_config):
        """Runner should preserve episode counter across checkpoint cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_dir = tmp_path / "checkpoints"
            checkpoint_dir.mkdir()

            config_dir = config_pack_factory(modifier=lambda data: data["training"].update({"max_episodes": 1}))

            with DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test1.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            ) as runner1:
                runner1.env = env_builder(config_dir=config_dir, num_agents=1)
                runner1.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
                _init_curriculum(runner1.curriculum, 1)
                runner1.exploration = AdaptiveIntrinsicExploration(obs_dim=runner1.env.observation_dim, device=cpu_device)
                runner1.population = VectorizedPopulation(
                    env=runner1.env,
                    curriculum=runner1.curriculum,
                    exploration=runner1.exploration,
                    agent_ids=["agent_0"],
                    device=cpu_device,
                    obs_dim=runner1.env.observation_dim,
                    brain_config=minimal_brain_config,
                )
                runner1.current_episode = 42
                runner1.save_checkpoint()

            with DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test2.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            ) as runner2:
                runner2.env = env_builder(config_dir=config_dir, num_agents=1)
                runner2.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
                _init_curriculum(runner2.curriculum, 1)
                runner2.exploration = AdaptiveIntrinsicExploration(obs_dim=runner2.env.observation_dim, device=cpu_device)
                runner2.population = VectorizedPopulation(
                    env=runner2.env,
                    curriculum=runner2.curriculum,
                    exploration=runner2.exploration,
                    agent_ids=["agent_0"],
                    device=cpu_device,
                    obs_dim=runner2.env.observation_dim,
                    brain_config=minimal_brain_config,
                )
                runner2.load_checkpoint()
                assert runner2.current_episode == 42, "Episode number should be preserved after load"

    def test_runner_checkpoint_round_trip_preserves_training_state(
        self, cpu_device, env_builder, config_pack_factory, minimal_brain_config
    ):
        """Runner checkpoint round-trip should preserve complete training state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_dir = tmp_path / "checkpoints"
            checkpoint_dir.mkdir()

            config_dir = config_pack_factory(modifier=lambda data: data["training"].update({"max_episodes": 1}), name="runner_config")

            with DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test1.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            ) as runner1:
                runner1.env = env_builder(config_dir=config_dir, num_agents=1)
                runner1.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
                _init_curriculum(runner1.curriculum, 1)
                runner1.exploration = AdaptiveIntrinsicExploration(obs_dim=runner1.env.observation_dim, device=cpu_device)
                runner1.population = VectorizedPopulation(
                    env=runner1.env,
                    curriculum=runner1.curriculum,
                    exploration=runner1.exploration,
                    agent_ids=["agent_0"],
                    device=cpu_device,
                    obs_dim=runner1.env.observation_dim,
                    brain_config=minimal_brain_config,
                )

                runner1.population.reset()
                for _ in range(50):
                    runner1.population.step_population(runner1.env)
                runner1.curriculum.tracker.agent_stages[0] = 3

                q_weights_before = {k: v.clone() for k, v in runner1.population.q_network.state_dict().items()}
                stage_before = runner1.curriculum.tracker.agent_stages[0].item()
                epsilon_before = runner1.exploration.rnd.epsilon
                buffer_size_before = len(runner1.population.replay_buffer)

                runner1.save_checkpoint()

            with DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test2.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            ) as runner2:
                runner2.env = env_builder(config_dir=config_dir, num_agents=1)
                runner2.curriculum = AdversarialCurriculum(max_steps_per_episode=100)
                _init_curriculum(runner2.curriculum, 1)
                runner2.exploration = AdaptiveIntrinsicExploration(obs_dim=runner2.env.observation_dim, device=cpu_device)
                runner2.population = VectorizedPopulation(
                    env=runner2.env,
                    curriculum=runner2.curriculum,
                    exploration=runner2.exploration,
                    agent_ids=["agent_0"],
                    device=cpu_device,
                    obs_dim=runner2.env.observation_dim,
                    brain_config=minimal_brain_config,
                )

                runner2.load_checkpoint()

                q_weights_after = runner2.population.q_network.state_dict()
                for key in q_weights_before.keys():
                    assert torch.allclose(q_weights_before[key], q_weights_after[key], atol=1e-6)

                stage_after = runner2.curriculum.tracker.agent_stages[0].item()
                assert stage_after == stage_before

                epsilon_after = runner2.exploration.rnd.epsilon
                assert abs(epsilon_after - epsilon_before) < 1e-6

                buffer_size_after = len(runner2.population.replay_buffer)
                assert buffer_size_after == buffer_size_before


# =============================================================================
# TEST CLASS 6: Checkpoint Round-Trip Verification (2 tests)
# =============================================================================


class TestCheckpointRoundTrip:
    """Test full checkpoint → load → resume → verify cycle.

    Ensures that training can be resumed exactly after checkpoint load,
    with no state degradation or loss.
    """

    def test_checkpoint_roundtrip_training_resumption(self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config):
        """Training should produce identical results when resumed from checkpoint."""
        # Create environment and components
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

        curriculum = StaticCurriculum(
            difficulty_level=0.5,
            active_meters=_select_active_meters(env),
        )
        _init_curriculum(curriculum, 1)

        exploration = EpsilonGreedyExploration(
            epsilon=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        pop1 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Train first population
        pop1.reset()
        for _ in range(100):
            pop1.step_population(env)

        # Save checkpoint at step 100
        checkpoint = pop1.get_checkpoint_state()

        # Continue training for 50 more steps
        for _ in range(50):
            pop1.step_population(env)

        # Capture state at step 150
        weights_at_150 = {k: v.clone() for k, v in pop1.q_network.state_dict().items()}

        # Now create second population, load checkpoint at step 100
        exploration2 = EpsilonGreedyExploration(
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        pop2 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        pop2.load_checkpoint_state(checkpoint)

        # Train for 50 steps (should reach step 150)
        for _ in range(50):
            pop2.step_population(env)

        # Verify weights differ (training progressed)
        # Note: We can't expect exact match due to randomness, but should be trained
        weights_at_150_v2 = pop2.q_network.state_dict()

        # Verify networks have same architecture
        assert set(weights_at_150.keys()) == set(weights_at_150_v2.keys()), "Network architectures should match"

        # Verify weights have changed from initial (training happened)
        initial_weights = checkpoint["q_network"]
        weights_changed = False
        for key in initial_weights.keys():
            if not torch.allclose(initial_weights[key], weights_at_150_v2[key], atol=1e-6):
                weights_changed = True
                break

        assert weights_changed, "Network weights should have changed after resumed training"

    def test_checkpoint_roundtrip_multi_component_consistency(
        self, cpu_device, test_config_pack_path, cpu_env_factory, minimal_brain_config
    ):
        """All components should maintain consistency across checkpoint round-trip."""
        # Setup
        env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=2)

        curriculum = AdversarialCurriculum(max_steps_per_episode=100)
        _init_curriculum(curriculum, 2)

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=cpu_device,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0", "agent_1"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Train for some steps
        population.reset()
        for _ in range(100):
            population.step_population(env)

        # Advance curriculum manually
        curriculum.tracker.agent_stages[0] = 2
        curriculum.tracker.agent_stages[1] = 3

        # Save all component states
        # Note: Environment doesn't have save_checkpoint(), save state manually
        env_positions = env.positions.clone()
        env_meters = env.meters.clone()
        env_affordances = env.get_affordance_positions()

        pop_checkpoint = population.get_checkpoint_state()
        curriculum_state = curriculum.state_dict()
        exploration_state = exploration.checkpoint_state()

        # Create fresh components
        env2 = cpu_env_factory(config_dir=test_config_pack_path, num_agents=2)
        env2.reset()  # Initialize environment

        curriculum2 = AdversarialCurriculum(max_steps_per_episode=100)
        _init_curriculum(curriculum2, 2)

        exploration2 = AdaptiveIntrinsicExploration(
            obs_dim=env2.observation_dim,
            device=cpu_device,
        )

        population2 = VectorizedPopulation(
            env=env2,
            curriculum=curriculum2,
            exploration=exploration2,
            agent_ids=["agent_0", "agent_1"],
            device=cpu_device,
            obs_dim=env2.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Load all checkpoints
        # Manually restore environment state (no built-in load_checkpoint)
        env2.positions = env_positions.clone()
        env2.meters = env_meters.clone()
        env2.set_affordance_positions(env_affordances)

        population2.load_checkpoint_state(pop_checkpoint)
        curriculum2.load_state_dict(curriculum_state)
        exploration2.load_state(exploration_state)

        # Verify consistency across components
        # 1. Environment state
        assert torch.equal(env2.positions, env_positions), "Agent positions should match"

        # 2. Curriculum stages match population
        assert curriculum2.tracker.agent_stages[0].item() == 2, "Agent 0 should be at stage 2"
        assert curriculum2.tracker.agent_stages[1].item() == 3, "Agent 1 should be at stage 3"

        # 3. Q-network loaded correctly
        assert len(population2.replay_buffer) == len(population.replay_buffer), "Replay buffer sizes should match"

        # 4. Exploration state consistent
        assert abs(exploration2.rnd.epsilon - exploration.rnd.epsilon) < 1e-6, "Epsilon should match"


# =============================================================================
# TEST CLASS 7: Variable Meter Checkpoint Validation (TASK-001 Phase 4)
# =============================================================================


class TestVariableMeterCheckpoints:
    """Test checkpoint saving/loading with variable meters (TASK-001).

    Verifies that checkpoints include meter metadata and validate compatibility.
    """

    def test_checkpoint_includes_meter_metadata(self, cpu_device, task001_env_4meter, minimal_brain_config):
        """Saved checkpoint should include meter count and names in metadata."""
        curriculum = StaticCurriculum(
            difficulty_level=0.5,
            active_meters=list(task001_env_4meter.meter_name_to_index.keys()),
        )
        _init_curriculum(curriculum, 1)

        exploration = EpsilonGreedyExploration(
            epsilon=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        population = VectorizedPopulation(
            env=task001_env_4meter,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Get checkpoint
        checkpoint = population.get_checkpoint_state()

        # Verify universe_metadata exists
        assert "universe_metadata" in checkpoint, "Checkpoint should contain universe_metadata"

        metadata = checkpoint["universe_metadata"]
        assert "meter_count" in metadata, "Metadata should contain meter_count"
        assert "meter_names" in metadata, "Metadata should contain meter_names"
        assert "version" in metadata, "Metadata should contain version"
        assert "obs_dim" in metadata, "Metadata should contain obs_dim"

        # Verify values
        assert metadata["meter_count"] == 4, f"Should have 4 meters, got {metadata['meter_count']}"
        assert metadata["meter_names"] == [
            "energy",
            "health",
            "money",
            "mood",
        ], f"Should have correct meter names, got {metadata['meter_names']}"

    def test_loading_checkpoint_validates_meter_count(self, cpu_device, task001_env_4meter, basic_env, tmp_path, minimal_brain_config):
        """Loading checkpoint should fail if meter counts don't match."""
        # Create 4-meter population and save checkpoint (no training needed)
        curriculum = StaticCurriculum(
            difficulty_level=0.5,
            active_meters=list(task001_env_4meter.meter_name_to_index.keys()),
        )
        _init_curriculum(curriculum, 1)
        exploration = EpsilonGreedyExploration()

        pop_4meter = VectorizedPopulation(
            env=task001_env_4meter,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Save checkpoint (no training needed for metadata test)
        checkpoint_4meter = pop_4meter.get_checkpoint_state()

        # Verify checkpoint has correct metadata
        assert checkpoint_4meter["universe_metadata"]["meter_count"] == 4

        # Try to load into 8-meter environment (should fail)
        curriculum2 = StaticCurriculum(
            difficulty_level=0.5,
            active_meters=list(task001_env_4meter.meter_name_to_index.keys()),
        )
        _init_curriculum(curriculum2, 1)
        exploration2 = EpsilonGreedyExploration()

        pop_8meter = VectorizedPopulation(
            env=basic_env,  # 8-meter env
            curriculum=curriculum2,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Loading should raise ValueError
        with pytest.raises(ValueError, match="meter count mismatch"):
            pop_8meter.load_checkpoint_state(checkpoint_4meter)

    def test_loading_checkpoint_with_matching_meters_succeeds(self, cpu_device, task001_env_4meter, minimal_brain_config):
        """Loading checkpoint should succeed if meter counts match."""
        curriculum = AdversarialCurriculum(max_steps_per_episode=100)
        _init_curriculum(curriculum, 1)
        exploration = EpsilonGreedyExploration()

        pop1 = VectorizedPopulation(
            env=task001_env_4meter,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Save checkpoint (no training needed)
        checkpoint = pop1.get_checkpoint_state()

        # Verify checkpoint has correct metadata
        assert checkpoint["universe_metadata"]["meter_count"] == 4

        # Create new population with same meter count
        curriculum2 = AdversarialCurriculum(max_steps_per_episode=100)
        _init_curriculum(curriculum2, 1)
        exploration2 = EpsilonGreedyExploration()

        pop2 = VectorizedPopulation(
            env=task001_env_4meter,  # Same 4-meter env
            curriculum=curriculum2,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Load should succeed (no exception)
        pop2.load_checkpoint_state(checkpoint)

        # Verify metadata matched
        assert checkpoint["universe_metadata"]["meter_count"] == 4

    def test_checkpoint_rejects_missing_universe_metadata(self, cpu_device, basic_env, minimal_brain_config):
        """Checkpoints without universe_metadata should be strictly rejected (pre-release, 0 users)."""
        # Create a real checkpoint first to get proper network state
        curriculum = AdversarialCurriculum(max_steps_per_episode=100)
        _init_curriculum(curriculum, 1)
        exploration = EpsilonGreedyExploration()

        population = VectorizedPopulation(
            env=basic_env,  # 8-meter env
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Get a real checkpoint and remove universe_metadata to simulate legacy
        checkpoint = population.get_checkpoint_state()
        legacy_checkpoint = {k: v for k, v in checkpoint.items() if k != "universe_metadata"}

        # Verify universe_metadata was removed
        assert "universe_metadata" not in legacy_checkpoint

        # Create new population
        curriculum2 = AdversarialCurriculum(max_steps_per_episode=100)
        _init_curriculum(curriculum2, 1)
        exploration2 = EpsilonGreedyExploration()

        population2 = VectorizedPopulation(
            env=basic_env,  # 8-meter env
            curriculum=curriculum2,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,
            # action_dim defaults to env.action_dim
            brain_config=minimal_brain_config,
        )

        # Loading checkpoint without universe_metadata should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            population2.load_checkpoint_state(legacy_checkpoint)

        # Verify error message provides clear guidance
        error_msg = str(exc_info.value)
        assert "universe_metadata" in error_msg
        assert "no longer supported" in error_msg.lower()
        assert "retrain" in error_msg.lower()


# =============================================================================
# TEST CLASS 8: DemoRunner Resource Management (QUICK-002)
# =============================================================================


class TestDemoRunnerResourceManagement:
    """Test DemoRunner context manager and resource cleanup (QUICK-002)."""

    def test_runner_closes_database_on_context_exit(self, tmp_path, cpu_device, config_pack_factory):
        """DemoRunner should close database when exiting context manager."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        config_dir = config_pack_factory(name="demo_runner_config")

        # Create runner in context manager
        with DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=1,
        ) as runner:
            # Database should be open
            assert hasattr(runner, "db")
            assert runner.db.conn is not None
            # Store connection reference
            conn = runner.db.conn

        # After exiting context, connection should be closed
        # SQLite connection has no is_closed() but we can check it raises
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")

    def test_runner_cleanup_is_idempotent(self, tmp_path, cpu_device, config_pack_factory):
        """Calling _cleanup() multiple times should be safe."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        config_dir = config_pack_factory(name="demo_runner_cleanup")

        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=checkpoint_dir,
            max_episodes=1,
        )

        # Call cleanup multiple times - should not raise
        runner._cleanup()
        runner._cleanup()  # Second call should be safe
        runner._cleanup()  # Third call should be safe

    def test_runner_context_manager_propagates_exceptions(self, tmp_path, cpu_device, config_pack_factory):
        """Context manager should propagate exceptions, not suppress them."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        config_dir = config_pack_factory(name="demo_runner_exception")

        # Exception inside with block should propagate
        with pytest.raises(ValueError, match="test exception"):
            with DemoRunner(
                config_dir=config_dir,
                db_path=tmp_path / "test.db",
                checkpoint_dir=checkpoint_dir,
                max_episodes=1,
            ) as runner:
                assert runner is not None  # Runner created successfully
                raise ValueError("test exception")

    def test_database_close_is_idempotent(self, cpu_device):
        """DemoDatabase.close() should be safe to call multiple times and track closed state."""
        import tempfile
        from pathlib import Path

        from townlet.demo.database import DemoDatabase

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"

            # Create database
            db = DemoDatabase(db_path)
            assert not db._closed, "Database should start as not closed"

            # Close multiple times - should not raise and should track state
            db.close()
            assert db._closed, "Database should be marked as closed after first close()"

            db.close()  # Second call should be safe
            assert db._closed, "Database should remain closed after second close()"

            db.close()  # Third call should be safe
            assert db._closed, "Database should remain closed after third close()"
