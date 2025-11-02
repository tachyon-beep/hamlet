"""
TDD coverage for runtime telemetry wiring (Task TLM-001).

Ensures VectorizedPopulation writes live curriculum, survival, and exploration
metrics into the runtime registry and exposes JSON-safe snapshots.
"""

import json

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


def _make_population(num_agents: int) -> VectorizedPopulation:
    """Helper to construct a small deterministic population for tests."""
    device = torch.device("cpu")
    env = VectorizedHamletEnv(
        num_agents=num_agents,
        grid_size=5,
        device=device,
        partial_observability=False,
    )

    curriculum = AdversarialCurriculum(
        max_steps_per_episode=200,
        device=device,
    )
    curriculum.initialize_population(num_agents)

    exploration = AdaptiveIntrinsicExploration(
        obs_dim=env.observation_dim,
        device=device,
    )

    population = VectorizedPopulation(
        env=env,
        curriculum=curriculum,
        exploration=exploration,
        agent_ids=[f"agent_{i}" for i in range(num_agents)],
        device=device,
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
    )
    population.reset()
    return population


def test_curriculum_metrics_flow_into_runtime_registry():
    """
    Red phase: curriculum stage/baseline should be written per agent.

    We manually place agents at different stages before a population step.
    Telemetry should reflect those stages and non-zero baselines afterwards.
    """
    population = _make_population(num_agents=2)

    # Place agents at distinct curriculum stages before the step.
    population.curriculum.tracker.agent_stages[0] = 4
    population.curriculum.tracker.agent_stages[1] = 2

    # Run a single population step (hot path that should update telemetry).
    population.step_population(population.env)

    snapshot_agent0 = population.runtime_registry.get_snapshot_for_agent(0).to_dict()
    snapshot_agent1 = population.runtime_registry.get_snapshot_for_agent(1).to_dict()

    assert snapshot_agent0["curriculum_stage"] == 4
    assert snapshot_agent1["curriculum_stage"] == 2
    assert snapshot_agent0["baseline_survival_steps"] > 0.0
    assert snapshot_agent1["baseline_survival_steps"] > 0.0


def test_survival_and_exploration_metrics_reflected_in_snapshot():
    """
    Red phase: survival time, epsilon, and intrinsic weight must surface in telemetry.
    """
    population = _make_population(num_agents=1)

    # Advance curriculum to a non-default stage to ensure stage is propagated.
    population.curriculum.tracker.agent_stages[0] = 3
    population.step_population(population.env)

    # Simulate a completed episode with 123 survival steps.
    population._finalize_episode(agent_idx=0, survival_time=123)

    # Decay epsilon (episode boundary) and sync exploration metrics.
    population.exploration.decay_epsilon()
    population.sync_exploration_metrics()

    snapshot = population.build_telemetry_snapshot(episode_index=5)
    json.dumps(snapshot)  # Must be serialisable

    assert snapshot["schema_version"] == "1.0.0"
    assert snapshot["episode_index"] == 5
    agent_payload = snapshot["agents"][0]

    assert agent_payload["survival_time"] == 123
    assert agent_payload["curriculum_stage"] == 3
    assert agent_payload["baseline_survival_steps"] > 0.0
    assert agent_payload["epsilon"] == pytest.approx(population.exploration.rnd.epsilon)
    assert agent_payload["intrinsic_weight"] == pytest.approx(population.exploration.get_intrinsic_weight())
