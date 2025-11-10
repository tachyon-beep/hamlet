"""
Unit tests for TASK-004B affordance capabilities (skill_scaling, probabilistic).

Tests the runtime implementation of capability composition system.
"""

import torch
import pytest
from unittest.mock import Mock

from townlet.environment.affordance_engine import AffordanceEngine


class MockAffordanceConfig:
    """Mock affordance config with capabilities support."""
    
    def __init__(self, name, capabilities=None, effect_pipeline=None):
        self.id = name
        self.name = name
        self.category = "test"
        self.interaction_type = "instant"
        self.costs = []
        self.costs_per_tick = []
        self.effects = []
        self.effects_per_tick = []
        self.completion_bonus = []
        self.operating_hours = [0, 24]
        self.required_ticks = None
        self.capabilities = capabilities or []
        self.effect_pipeline = effect_pipeline


class MockCapability:
    """Mock capability object."""
    
    def __init__(self, type, **kwargs):
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockEffectPipeline:
    """Mock effect pipeline object."""
    
    def __init__(self, on_start=None, per_tick=None, on_completion=None, on_failure=None):
        self.on_start = on_start or []
        self.per_tick = per_tick or []
        self.on_completion = on_completion or []
        self.on_failure = on_failure or []


class MockEffect:
    """Mock effect object."""
    
    def __init__(self, meter, amount):
        self.meter = meter
        self.amount = amount


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def meter_map():
    return {
        "energy": 0,
        "hygiene": 1,
        "fitness": 2,
        "money": 3,
        "health": 4,
    }


@pytest.fixture
def mock_engine(device, meter_map):
    """Create mock AffordanceEngine with test affordances."""
    mock_collection = Mock()
    mock_collection.affordances = []
    
    engine = AffordanceEngine(
        affordance_config=mock_collection,
        num_agents=10,
        device=device,
        meter_name_to_idx=meter_map,
    )
    
    return engine


def test_get_capability_returns_none_when_no_capabilities():
    """Test that _get_capability returns None for affordances without capabilities."""
    affordance = MockAffordanceConfig("test")
    del affordance.capabilities  # Remove capabilities attribute
    
    engine_mock = Mock()
    from townlet.environment.affordance_engine import AffordanceEngine
    result = AffordanceEngine._get_capability(engine_mock, affordance, "skill_scaling")
    
    assert result is None


def test_get_capability_returns_matching_capability():
    """Test that _get_capability returns the correct capability when present."""
    skill_cap = MockCapability("skill_scaling", skill="fitness", base_multiplier=1.0, max_multiplier=2.0)
    affordance = MockAffordanceConfig("test", capabilities=[skill_cap])
    
    engine_mock = Mock()
    from townlet.environment.affordance_engine import AffordanceEngine
    result = AffordanceEngine._get_capability(engine_mock, affordance, "skill_scaling")
    
    assert result is skill_cap
    assert result.skill == "fitness"


def test_compute_skill_multiplier_returns_ones_when_no_capability(mock_engine, device):
    """Test that skill multiplier is 1.0 when no skill_scaling capability."""
    affordance = MockAffordanceConfig("test")
    meters = torch.rand(10, 5, device=device)
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    multiplier = mock_engine._compute_skill_multiplier(affordance, meters, agent_mask)
    
    assert multiplier.shape == (10,)
    assert torch.allclose(multiplier, torch.ones(10, device=device))


def test_compute_skill_multiplier_with_skill_scaling(mock_engine, device):
    """Test that skill multiplier interpolates correctly based on skill meter."""
    skill_cap = MockCapability("skill_scaling", skill="fitness", base_multiplier=1.0, max_multiplier=2.0)
    affordance = MockAffordanceConfig("test", capabilities=[skill_cap])
    
    # Create meters with varying fitness levels
    meters = torch.zeros(10, 5, device=device)
    meters[:, 2] = torch.linspace(0.0, 1.0, 10)  # fitness = column 2
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    multiplier = mock_engine._compute_skill_multiplier(affordance, meters, agent_mask)
    
    # Check multiplier interpolation: base + (max - base) * skill
    # At fitness=0.0: multiplier = 1.0 + (2.0 - 1.0) * 0.0 = 1.0
    # At fitness=0.5: multiplier = 1.0 + (2.0 - 1.0) * 0.5 = 1.5
    # At fitness=1.0: multiplier = 1.0 + (2.0 - 1.0) * 1.0 = 2.0
    assert torch.allclose(multiplier[0], torch.tensor(1.0))
    assert torch.allclose(multiplier[4], torch.tensor(1.5), atol=0.05)
    assert torch.allclose(multiplier[9], torch.tensor(2.0), atol=0.05)


def test_apply_instant_interaction_legacy_format(mock_engine, device):
    """Test that legacy format (effects list) still works."""
    affordance = MockAffordanceConfig("test")
    affordance.effects = [MockEffect("energy", 0.5)]
    
    mock_engine.affordance_map = {"test": affordance}
    
    meters = torch.zeros(10, 5, device=device)
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    updated_meters = mock_engine.apply_instant_interaction(meters, "test", agent_mask)
    
    # Energy (index 0) should have increased by 0.5
    assert torch.allclose(updated_meters[:, 0], torch.tensor(0.5))


def test_apply_instant_interaction_with_skill_scaling(mock_engine, device):
    """Test that skill_scaling modifies effect amounts."""
    skill_cap = MockCapability("skill_scaling", skill="fitness", base_multiplier=1.0, max_multiplier=2.0)
    
    effect_pipeline = MockEffectPipeline(
        on_completion=[MockEffect("energy", 0.5)]
    )
    
    affordance = MockAffordanceConfig("test", capabilities=[skill_cap], effect_pipeline=effect_pipeline)
    mock_engine.affordance_map = {"test": affordance}
    
    # Create meters with varying fitness levels
    meters = torch.zeros(10, 5, device=device)
    meters[:, 2] = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # fitness
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    updated_meters = mock_engine.apply_instant_interaction(meters, "test", agent_mask)
    
    # Agent 0 (fitness=0.0): energy += 0.5 * 1.0 = 0.5
    # Agent 1 (fitness=0.5): energy += 0.5 * 1.5 = 0.75
    # Agent 2 (fitness=1.0): energy += 0.5 * 2.0 = 1.0 (clamped)
    assert torch.allclose(updated_meters[0, 0], torch.tensor(0.5))
    assert torch.allclose(updated_meters[1, 0], torch.tensor(0.75))
    assert torch.allclose(updated_meters[2, 0], torch.tensor(1.0))


def test_apply_instant_interaction_with_probabilistic_success(mock_engine, device):
    """Test probabilistic interaction with successful outcome."""
    prob_cap = MockCapability("probabilistic", success_probability=1.0)  # 100% success
    
    effect_pipeline = MockEffectPipeline(
        on_completion=[MockEffect("energy", 0.5)],
        on_failure=[MockEffect("energy", -0.2)]
    )
    
    affordance = MockAffordanceConfig("test", capabilities=[prob_cap], effect_pipeline=effect_pipeline)
    mock_engine.affordance_map = {"test": affordance}
    
    meters = torch.zeros(10, 5, device=device)
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    updated_meters = mock_engine.apply_instant_interaction(meters, "test", agent_mask)
    
    # All agents should get success effect (energy += 0.5)
    assert torch.allclose(updated_meters[:, 0], torch.tensor(0.5))


def test_apply_instant_interaction_with_probabilistic_failure(mock_engine, device):
    """Test probabilistic interaction with failure outcome."""
    prob_cap = MockCapability("probabilistic", success_probability=0.0)  # 0% success (100% failure)
    
    effect_pipeline = MockEffectPipeline(
        on_completion=[MockEffect("energy", 0.5)],
        on_failure=[MockEffect("energy", 0.1)]  # Smaller failure reward
    )
    
    affordance = MockAffordanceConfig("test", capabilities=[prob_cap], effect_pipeline=effect_pipeline)
    mock_engine.affordance_map = {"test": affordance}
    
    meters = torch.zeros(10, 5, device=device)
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    updated_meters = mock_engine.apply_instant_interaction(meters, "test", agent_mask)
    
    # All agents should get failure effect (energy += 0.1)
    assert torch.allclose(updated_meters[:, 0], torch.tensor(0.1))


def test_apply_instant_interaction_with_combined_capabilities(mock_engine, device):
    """Test that skill_scaling and probabilistic work together."""
    skill_cap = MockCapability("skill_scaling", skill="fitness", base_multiplier=1.0, max_multiplier=2.0)
    prob_cap = MockCapability("probabilistic", success_probability=1.0)  # 100% success for testing
    
    effect_pipeline = MockEffectPipeline(
        on_completion=[MockEffect("energy", 0.5)],
        on_failure=[MockEffect("energy", -0.2)]
    )
    
    affordance = MockAffordanceConfig("test", capabilities=[skill_cap, prob_cap], effect_pipeline=effect_pipeline)
    mock_engine.affordance_map = {"test": affordance}
    
    # Create meters with varying fitness levels
    meters = torch.zeros(10, 5, device=device)
    meters[:, 2] = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # fitness
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    torch.manual_seed(42)
    updated_meters = mock_engine.apply_instant_interaction(meters, "test", agent_mask)
    
    # Skills scale success effects:
    # Agent 0 (fitness=0.0): energy += 0.5 * 1.0 = 0.5
    # Agent 1 (fitness=0.5): energy += 0.5 * 1.5 = 0.75
    # Agent 2 (fitness=1.0): energy += 0.5 * 2.0 = 1.0
    assert torch.allclose(updated_meters[0, 0], torch.tensor(0.5))
    assert torch.allclose(updated_meters[1, 0], torch.tensor(0.75))
    assert torch.allclose(updated_meters[2, 0], torch.tensor(1.0))


def test_apply_multi_tick_interaction_with_skill_scaling(mock_engine, device):
    """Test that skill_scaling works in multi-tick interactions."""
    skill_cap = MockCapability("skill_scaling", skill="fitness", base_multiplier=1.0, max_multiplier=2.0)
    
    effect_pipeline = MockEffectPipeline(
        per_tick=[MockEffect("energy", 0.1)],
        on_completion=[MockEffect("energy", 0.2)]
    )
    
    affordance = MockAffordanceConfig("test", capabilities=[skill_cap], effect_pipeline=effect_pipeline)
    affordance.interaction_type = "multi_tick"
    affordance.required_ticks = 3
    mock_engine.affordance_map = {"test": affordance}
    
    # Create meters with varying fitness levels
    meters = torch.zeros(10, 5, device=device)
    meters[:, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # fitness
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    # Tick 0: per_tick effect only
    updated_meters = mock_engine.apply_multi_tick_interaction(meters, "test", 0, agent_mask)
    # Agent 0 (fitness=0.0): energy += 0.1 * 1.0 = 0.1
    # Agent 1 (fitness=1.0): energy += 0.1 * 2.0 = 0.2
    assert torch.allclose(updated_meters[0, 0], torch.tensor(0.1))
    assert torch.allclose(updated_meters[1, 0], torch.tensor(0.2))
    
    # Final tick: per_tick + on_completion
    updated_meters = mock_engine.apply_multi_tick_interaction(updated_meters, "test", 2, agent_mask)
    # Agent 0: += (0.1 + 0.2) * 1.0 = 0.3, total = 0.4
    # Agent 1: += (0.1 + 0.2) * 2.0 = 0.6, total = 0.8
    assert torch.allclose(updated_meters[0, 0], torch.tensor(0.4))
    assert torch.allclose(updated_meters[1, 0], torch.tensor(0.8))


def test_apply_multi_tick_interaction_with_probabilistic_completion(mock_engine, device):
    """Test that probabilistic works on final tick of multi-tick interactions."""
    prob_cap = MockCapability("probabilistic", success_probability=1.0)  # 100% success
    
    effect_pipeline = MockEffectPipeline(
        per_tick=[MockEffect("energy", 0.1)],
        on_completion=[MockEffect("money", 0.5)],
        on_failure=[MockEffect("money", -0.1)]
    )
    
    affordance = MockAffordanceConfig("test", capabilities=[prob_cap], effect_pipeline=effect_pipeline)
    affordance.interaction_type = "multi_tick"
    affordance.required_ticks = 3
    mock_engine.affordance_map = {"test": affordance}
    
    meters = torch.zeros(10, 5, device=device)
    agent_mask = torch.ones(10, dtype=torch.bool, device=device)
    
    # Non-final tick: only per_tick effect
    updated_meters = mock_engine.apply_multi_tick_interaction(meters, "test", 0, agent_mask)
    assert torch.allclose(updated_meters[:, 0], torch.tensor(0.1))  # energy
    assert torch.allclose(updated_meters[:, 3], torch.tensor(0.0))  # money unchanged
    
    # Final tick: per_tick + probabilistic on_completion
    torch.manual_seed(42)
    updated_meters = mock_engine.apply_multi_tick_interaction(updated_meters, "test", 2, agent_mask)
    assert torch.allclose(updated_meters[:, 0], torch.tensor(0.2))  # energy += 0.1
    assert torch.allclose(updated_meters[:, 3], torch.tensor(0.5))  # money += 0.5 (success)
