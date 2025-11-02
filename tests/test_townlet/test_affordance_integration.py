"""
Integration tests for AffordanceEngine → VectorizedHamletEnv.

TDD Approach: Red → Green → Refactor
- Red: Tests fail because vectorized_env still uses hardcoded logic
- Green: Replace hardcoded logic with engine calls
- Refactor: Clean up and optimize

Test Coverage:
1. Single affordance interaction (Bed)
2. Multiple agents, different affordances simultaneously
3. Affordability checks (money constraints)
4. Operating hours enforcement
5. Meter effects match hardcoded logic exactly
"""

from pathlib import Path

import pytest
import torch
import yaml

from townlet.environment.affordance_config import AffordanceConfigCollection
from townlet.environment.affordance_engine import AffordanceEngine
from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def affordance_config():
    """Load production config (affordances.yaml)."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "affordances.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return AffordanceConfigCollection.model_validate(config_dict)


@pytest.fixture
def simple_env_config():
    """Minimal environment config for testing."""
    return {
        "grid_size": 8,
        "partial_observability": False,
        "enable_temporal_mechanics": False,
        "num_agents": 1,
    }


class TestAffordanceEngineIntegration:
    """Test that VectorizedHamletEnv uses AffordanceEngine correctly."""

    def test_environment_has_affordance_engine(self, simple_env_config, affordance_config):
        """RED: Environment should have an affordance engine instance."""
        env = VectorizedHamletEnv(
            num_agents=simple_env_config["num_agents"],
            grid_size=simple_env_config["grid_size"],
            partial_observability=simple_env_config["partial_observability"],
            enable_temporal_mechanics=simple_env_config["enable_temporal_mechanics"],
        )

        # This will FAIL until we add engine to __init__
        assert hasattr(env, "affordance_engine"), "Environment should have affordance_engine"
        assert isinstance(env.affordance_engine, AffordanceEngine)

    def test_bed_interaction_uses_engine(self, simple_env_config, affordance_config):
        """RED: Bed interaction should use engine, not hardcoded logic."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()  # Initialize meters

        # Set up agent at Bed position
        bed_pos = env.affordances["Bed"]
        env.positions[0] = bed_pos

        # Set initial meters (low energy, sufficient money)
        env.meters[0, 0] = 0.3  # energy=30%
        env.meters[0, 3] = 0.10  # money=$10 (enough for Bed=$5)

        initial_energy = env.meters[0, 0].item()
        initial_money = env.meters[0, 3].item()

        # Trigger INTERACT action (action 4)
        interact_action = torch.tensor([4], dtype=torch.long)
        env.step(interact_action)

        # Verify Bed effects applied
        # Bed: +50% energy, +2% health, -$5
        # Plus step dynamics: -0.5% energy (passive depletion)
        expected_energy = min(1.0, initial_energy + 0.50 - 0.005)  # +Bed -depletion
        expected_money = initial_money - 0.05  # $5

        assert abs(env.meters[0, 0].item() - expected_energy) < 1e-3, f"Energy should be ~{expected_energy}, got {env.meters[0, 0].item()}"
        assert abs(env.meters[0, 3].item() - expected_money) < 1e-3, f"Money should be ~{expected_money}, got {env.meters[0, 3].item()}"

        # This test will PASS even with hardcoded logic (validates baseline)
        # But we'll refactor to use engine for maintainability

    def test_multiple_affordances_simultaneously(self, simple_env_config, affordance_config):
        """RED: Multiple agents at different affordances should all work."""
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()  # Initialize meters

        # Agent 0: Bed
        # Agent 1: Shower
        # Agent 2: HomeMeal
        env.positions[0] = env.affordances["Bed"]
        env.positions[1] = env.affordances["Shower"]
        env.positions[2] = env.affordances["HomeMeal"]

        # Set initial meters (sufficient money for all)
        env.meters[:, 3] = 0.20  # $20 each
        env.meters[0, 0] = 0.2  # Agent 0: low energy
        env.meters[1, 1] = 0.3  # Agent 1: low hygiene
        env.meters[2, 2] = 0.4  # Agent 2: low satiation

        # All INTERACT
        interact_actions = torch.tensor([4, 4, 4], dtype=torch.long)
        env.step(interact_actions)

        # Verify each agent got correct effects (accounting for passive depletion)
        # Bed: +50% energy, +2% health, -$5, then -0.5% energy (depletion)
        # 0.2 + 0.5 - 0.005 = 0.695
        assert env.meters[0, 0] >= 0.69, "Agent 0 should have restored energy"
        assert env.meters[0, 3] < 0.20, "Agent 0 should have spent money"

        # Shower: +40% hygiene, -$3, then -0.3% hygiene (depletion)
        # 0.3 + 0.4 - 0.003 = 0.697
        assert env.meters[1, 1] >= 0.69, "Agent 1 should have restored hygiene"

        # HomeMeal: +45% satiation, +3% health, -$3, then -0.4% satiation (depletion)
        # 0.4 + 0.45 - 0.004 = 0.846
        assert env.meters[2, 2] >= 0.84, "Agent 2 should have restored satiation"

    def test_affordability_check(self, simple_env_config, affordance_config):
        """RED: Agents without enough money should not interact."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()  # Initialize meters

        # Agent at Bed but NO money
        env.positions[0] = env.affordances["Bed"]
        env.meters[0, 0] = 0.2  # Low energy (needs Bed)
        env.meters[0, 3] = 0.02  # Only $2 (needs $5)

        initial_energy = env.meters[0, 0].item()
        initial_money = env.meters[0, 3].item()

        # Try to INTERACT (should fail affordability check)
        interact_action = torch.tensor([4], dtype=torch.long)
        env.step(interact_action)

        # Verify NO AFFORDANCE effects applied (but passive depletion still happens)
        # Energy: 0.2 - 0.005 (passive) = 0.195 (Bed effect NOT applied)
        expected_energy_after_depletion = initial_energy - 0.005
        assert (
            abs(env.meters[0, 0].item() - expected_energy_after_depletion) < 1e-3
        ), f"Energy should be {expected_energy_after_depletion} (passive depletion only), got {env.meters[0, 0].item()}"
        assert abs(env.meters[0, 3].item() - initial_money) < 1e-3, "Money should not change (can't afford)"

    def test_park_is_free(self, simple_env_config, affordance_config):
        """RED: Park should work even with $0."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()  # Initialize meters

        # Agent at Park with NO money
        env.positions[0] = env.affordances["Park"]
        env.meters[0, 7] = 0.3  # Low fitness
        env.meters[0, 3] = 0.00  # $0 (Park is FREE)

        initial_fitness = env.meters[0, 7].item()
        initial_money = env.meters[0, 3].item()

        # INTERACT at Park
        interact_action = torch.tensor([4], dtype=torch.long)
        env.step(interact_action)

        # Verify effects applied (Park: +20% fitness, +15% social, +15% mood, -15% energy)
        assert env.meters[0, 7].item() > initial_fitness, "Fitness should increase"
        assert abs(env.meters[0, 3].item() - initial_money) < 1e-6, "Money should not change (Park is free)"

    def test_engine_method_exists(self, simple_env_config, affordance_config):
        """RED: AffordanceEngine should have apply_interaction method."""
        engine = AffordanceEngine(affordance_config, num_agents=1, device=torch.device("cpu"))

        # This will FAIL until we add apply_interaction method
        assert hasattr(engine, "apply_interaction"), "Engine should have apply_interaction method"

        # Method signature check
        import inspect

        sig = inspect.signature(engine.apply_interaction)
        params = list(sig.parameters.keys())

        # Expected: (meters, affordance_name, agent_mask)
        assert "meters" in params, "Should accept meters tensor"
        assert "affordance_name" in params, "Should accept affordance name"
        assert "agent_mask" in params, "Should accept agent mask"


class TestAffordanceEngineMethod:
    """Test the new apply_interaction method we'll add to AffordanceEngine."""

    def test_apply_interaction_bed(self, affordance_config):
        """RED: Engine should apply Bed effects correctly."""
        engine = AffordanceEngine(affordance_config, num_agents=1, device=torch.device("cpu"))

        # Create meters tensor [1 agent, 8 meters]
        meters = torch.zeros(1, 8, dtype=torch.float32)
        meters[0, 0] = 0.3  # energy=30%
        meters[0, 3] = 0.10  # money=$10

        # Apply Bed interaction
        agent_mask = torch.tensor([True])
        result_meters = engine.apply_interaction(meters=meters, affordance_name="Bed", agent_mask=agent_mask)

        # Verify Bed effects
        # Bed: +50% energy, +2% health, -$5
        expected_energy = min(1.0, 0.3 + 0.50)
        expected_money = 0.10 - 0.05

        assert abs(result_meters[0, 0].item() - expected_energy) < 1e-6
        assert abs(result_meters[0, 3].item() - expected_money) < 1e-6

    def test_apply_interaction_multiple_agents(self, affordance_config):
        """RED: Engine should handle multiple agents with masks."""
        engine = AffordanceEngine(affordance_config, num_agents=1, device=torch.device("cpu"))

        # Create meters tensor [3 agents, 8 meters]
        meters = torch.zeros(3, 8, dtype=torch.float32)
        meters[:, 0] = 0.3  # All low energy
        meters[:, 3] = 0.10  # All have $10

        # Only agent 0 and 2 interact
        agent_mask = torch.tensor([True, False, True])

        result_meters = engine.apply_interaction(meters=meters, affordance_name="Bed", agent_mask=agent_mask)

        # Agent 0: effects applied
        assert result_meters[0, 0].item() > 0.3

        # Agent 1: NO effects (mask=False)
        assert abs(result_meters[1, 0].item() - 0.3) < 1e-6

        # Agent 2: effects applied
        assert result_meters[2, 0].item() > 0.3


class TestFullIntegration:
    """End-to-end integration tests."""

    def test_environment_uses_engine_for_all_affordances(self, affordance_config):
        """RED: Environment should delegate ALL affordances to engine."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()  # Initialize meters

        # Test all 14 affordances
        affordance_names = [
            "Bed",
            "LuxuryBed",
            "Shower",
            "HomeMeal",
            "FastFood",
            "Doctor",
            "Hospital",
            "Therapist",
            "Recreation",
            "Bar",
            "Job",
            "Labor",
            "Gym",
            "Park",
        ]

        for aff_name in affordance_names:
            # Reset meters
            env.meters[0] = torch.tensor([0.5] * 8, dtype=torch.float32)
            env.meters[0, 3] = 0.20  # $20 (enough for any affordance)

            # Position at affordance
            env.positions[0] = env.affordances[aff_name]

            # Store initial state
            initial_meters = env.meters[0].clone()

            # INTERACT
            interact_action = torch.tensor([4], dtype=torch.long)
            env.step(interact_action)

            # Verify SOME effect occurred (meters changed)
            meter_diff = (env.meters[0] - initial_meters).abs().sum()
            assert meter_diff > 1e-6, f"{aff_name} should modify meters (got diff={meter_diff})"

    def test_hardcoded_logic_removed(self):
        """GREEN: _handle_interactions_legacy now uses AffordanceEngine."""
        import inspect

        from townlet.environment import vectorized_env

        source = inspect.getsource(vectorized_env.VectorizedHamletEnv._handle_interactions_legacy)

        # After integration, hardcoded elif blocks should be GONE
        # Integration complete: replaced ~160 lines with engine call
        assert "elif affordance_name ==" not in source, "Hardcoded elif blocks should be removed (integration complete)"

        # Verify engine is being used instead
        assert "affordance_engine.apply_interaction" in source, "Should delegate to AffordanceEngine.apply_interaction()"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
