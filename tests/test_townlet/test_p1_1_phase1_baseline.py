"""
P1.1 Phase 1: Baseline Tests (What Already Works)

This phase establishes GREEN baseline before adding new features.
Tests verify current checkpoint functionality.
"""

import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


class TestPopulationCheckpointBaseline:
    """Test population-level checkpointing (already implemented)."""

    def test_get_checkpoint_state_returns_dict(self):
        """Population should return a dict with expected keys."""
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

        checkpoint = population.get_checkpoint_state()

        # Verify it's a dict
        assert isinstance(checkpoint, dict), "Checkpoint should be a dict"

        # Verify expected keys exist
        assert "version" in checkpoint, "Should have version"
        assert "q_network" in checkpoint, "Should have q_network"
        assert "optimizer" in checkpoint, "Should have optimizer"
        assert "total_steps" in checkpoint, "Should have total_steps"
        assert "exploration_state" in checkpoint, "Should have exploration_state"
        assert "replay_buffer" in checkpoint, "Should have replay_buffer"

    def test_checkpoint_version_is_2_or_higher(self):
        """Version should be >= 2 for full-fidelity checkpoints."""
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

        checkpoint = population.get_checkpoint_state()
        assert checkpoint["version"] >= 2, f"Version should be >= 2, got {checkpoint['version']}"

    def test_load_checkpoint_state_restores_qnetwork(self):
        """Q-network state_dict should be restored after load."""
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

        # First population
        pop1 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
        )

        # Save checkpoint
        checkpoint = pop1.get_checkpoint_state()

        # Verify q_network state dict is present and non-empty
        assert "q_network" in checkpoint, "Should have q_network in checkpoint"
        assert len(checkpoint["q_network"]) > 0, "Q-network state dict should not be empty"

        # Second population (fresh weights)
        exploration2 = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=env.device,
        )
        pop2 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
        )

        # Load checkpoint - should not raise
        pop2.load_checkpoint_state(checkpoint)

        # Verify state dicts have same keys
        original_keys = set(checkpoint["q_network"].keys())
        restored_keys = set(pop2.q_network.state_dict().keys())
        assert original_keys == restored_keys, "State dict keys should match after restore"

    def test_replay_buffer_serialization_roundtrip(self):
        """Replay buffer should survive save/load cycle."""
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

        # First population
        pop1 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
        )

        # Train to add experiences
        pop1.reset()  # Initialize current_obs
        for _ in range(50):  # Enough to populate buffer
            pop1.step_population(env)

        original_buffer_size = len(pop1.replay_buffer)
        assert original_buffer_size > 0, "Buffer should have experiences"

        # Save checkpoint
        checkpoint = pop1.get_checkpoint_state()

        # Second population (empty buffer)
        exploration2 = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=env.device,
        )
        pop2 = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=env.device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
        )

        assert len(pop2.replay_buffer) == 0, "Fresh buffer should be empty"

        # Load checkpoint
        pop2.load_checkpoint_state(checkpoint)

        # Verify buffer size restored
        restored_buffer_size = len(pop2.replay_buffer)
        assert restored_buffer_size == original_buffer_size, f"Buffer size should match: {original_buffer_size} vs {restored_buffer_size}"
