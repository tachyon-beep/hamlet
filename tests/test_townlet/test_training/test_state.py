"""Tests for Townlet state DTOs (cold path)."""

import pytest
from pydantic import ValidationError


def test_curriculum_decision_valid():
    """CurriculumDecision should accept valid parameters."""
    from townlet.training.state import CurriculumDecision

    decision = CurriculumDecision(
        difficulty_level=0.5,
        active_meters=["energy", "hygiene"],
        depletion_multiplier=1.0,
        reward_mode="sparse",
        reason="Test curriculum decision"
    )

    assert decision.difficulty_level == 0.5
    assert decision.active_meters == ["energy", "hygiene"]
    assert decision.depletion_multiplier == 1.0
    assert decision.reward_mode == "sparse"
    assert decision.reason == "Test curriculum decision"


def test_curriculum_decision_difficulty_out_of_range():
    """CurriculumDecision should reject difficulty outside [0, 1]."""
    from townlet.training.state import CurriculumDecision

    with pytest.raises(ValidationError) as exc_info:
        CurriculumDecision(
            difficulty_level=1.5,  # Invalid: > 1.0
            active_meters=["energy"],
            depletion_multiplier=1.0,
            reward_mode="sparse",
            reason="Test"
        )

    assert "difficulty_level" in str(exc_info.value)


def test_curriculum_decision_invalid_reward_mode():
    """CurriculumDecision should reject invalid reward_mode."""
    from townlet.training.state import CurriculumDecision

    with pytest.raises(ValidationError) as exc_info:
        CurriculumDecision(
            difficulty_level=0.5,
            active_meters=["energy"],
            depletion_multiplier=1.0,
            reward_mode="invalid_mode",  # Invalid
            reason="Test"
        )

    assert "reward_mode" in str(exc_info.value)


def test_curriculum_decision_immutable():
    """CurriculumDecision should be immutable (frozen)."""
    from townlet.training.state import CurriculumDecision

    decision = CurriculumDecision(
        difficulty_level=0.5,
        active_meters=["energy"],
        depletion_multiplier=1.0,
        reward_mode="sparse",
        reason="Test"
    )

    with pytest.raises(ValidationError):
        decision.difficulty_level = 0.8  # Should fail (frozen)


def test_exploration_config_valid():
    """ExplorationConfig should accept valid parameters."""
    from townlet.training.state import ExplorationConfig

    config = ExplorationConfig(
        strategy_type="epsilon_greedy",
        epsilon=0.5,
        epsilon_decay=0.995,
        intrinsic_weight=0.0
    )

    assert config.strategy_type == "epsilon_greedy"
    assert config.epsilon == 0.5
    assert config.epsilon_decay == 0.995
    assert config.intrinsic_weight == 0.0


def test_exploration_config_invalid_strategy():
    """ExplorationConfig should reject invalid strategy_type."""
    from townlet.training.state import ExplorationConfig

    with pytest.raises(ValidationError):
        ExplorationConfig(strategy_type="invalid_strategy")


def test_exploration_config_epsilon_out_of_range():
    """ExplorationConfig should reject epsilon outside [0, 1]."""
    from townlet.training.state import ExplorationConfig

    with pytest.raises(ValidationError) as exc_info:
        ExplorationConfig(epsilon=1.5)

    assert "epsilon" in str(exc_info.value)


def test_exploration_config_defaults():
    """ExplorationConfig should have sensible defaults."""
    from townlet.training.state import ExplorationConfig

    config = ExplorationConfig(strategy_type="epsilon_greedy")

    assert config.epsilon == 1.0  # Default: full exploration initially
    assert config.epsilon_decay == 0.995
    assert config.intrinsic_weight == 0.0


def test_population_checkpoint_valid():
    """PopulationCheckpoint should accept valid parameters."""
    from townlet.training.state import PopulationCheckpoint

    checkpoint = PopulationCheckpoint(
        generation=5,
        num_agents=10,
        agent_ids=["agent_0", "agent_1"],
        curriculum_states={"agent_0": {"stage": 2}},
        exploration_states={"agent_0": {"epsilon": 0.5}},
        pareto_frontier=["agent_0"],
        metrics_summary={"avg_survival": 100.0}
    )

    assert checkpoint.generation == 5
    assert checkpoint.num_agents == 10
    assert len(checkpoint.agent_ids) == 2


def test_population_checkpoint_num_agents_limit():
    """PopulationCheckpoint should enforce agent count limits."""
    from townlet.training.state import PopulationCheckpoint

    # Valid: 1000 agents (max)
    checkpoint = PopulationCheckpoint(
        generation=0,
        num_agents=1000,
        agent_ids=[f"agent_{i}" for i in range(1000)],
        curriculum_states={},
        exploration_states={},
        pareto_frontier=[],
        metrics_summary={}
    )
    assert checkpoint.num_agents == 1000

    # Invalid: 1001 agents (exceeds max)
    with pytest.raises(ValidationError):
        PopulationCheckpoint(
            generation=0,
            num_agents=1001,
            agent_ids=[],
            curriculum_states={},
            exploration_states={},
            pareto_frontier=[],
            metrics_summary={}
        )


def test_population_checkpoint_serialization():
    """PopulationCheckpoint should serialize to JSON."""
    from townlet.training.state import PopulationCheckpoint

    checkpoint = PopulationCheckpoint(
        generation=1,
        num_agents=2,
        agent_ids=["agent_0", "agent_1"],
        curriculum_states={},
        exploration_states={},
        pareto_frontier=[],
        metrics_summary={}
    )

    # Serialize to JSON
    json_str = checkpoint.model_dump_json()
    assert "generation" in json_str
    assert "num_agents" in json_str

    # Deserialize from JSON
    from townlet.training.state import PopulationCheckpoint
    restored = PopulationCheckpoint.model_validate_json(json_str)
    assert restored.generation == checkpoint.generation
    assert restored.num_agents == checkpoint.num_agents
