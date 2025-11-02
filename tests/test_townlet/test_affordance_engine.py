"""
Test suite for AffordanceEngine - config-driven affordance interaction system.

Following TDD: These tests are written BEFORE implementation.
Tests will fail initially, then pass as we build affordance_engine.py.

Test Coverage:
1. Instant interactions (Shower, HomeMeal)
2. Multi-tick interactions (Bed, Job)
3. Operating hours masking
4. Cost application (deducted before effects)
5. Completion bonuses
6. Integration with vectorized environment
"""

import pytest
import torch
from pathlib import Path


# These imports will fail initially - that's expected in TDD!
try:
    from townlet.environment.affordance_engine import AffordanceEngine
    from townlet.environment.affordance_config import load_affordance_config

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="affordance_engine.py not yet implemented")


@pytest.fixture
def device():
    """Test on CPU for reproducibility."""
    return torch.device("cpu")


@pytest.fixture
def affordance_config():
    """Load affordance configuration (corrected version matching hardcoded logic)."""
    config_path = Path("configs/test/affordances.yaml")
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    return load_affordance_config(config_path)


@pytest.fixture
def engine(affordance_config, device):
    """Create AffordanceEngine instance."""
    return AffordanceEngine(affordance_config=affordance_config, num_agents=4, device=device)


class TestAffordanceEngineBasics:
    """Test basic AffordanceEngine functionality."""

    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.num_agents == 4
        assert len(engine.affordances) == 14  # Corrected config has 14 affordances (no CoffeeShop)

    def test_affordance_lookup_by_id(self, engine):
        """Test looking up affordances by ID (numeric IDs in corrected config)."""
        bed = engine.get_affordance("0")  # Bed has ID "0" in corrected config
        assert bed is not None
        assert bed.name == "Bed"
        assert bed.id == "0"

        shower = engine.get_affordance("2")  # Shower has ID "2"
        assert shower is not None
        assert shower.name == "Shower"
        assert shower.interaction_type == "dual"  # All affordances are now dual-mode

    def test_affordance_name_to_index_mapping(self, engine):
        """
        Test that affordance names map to indices in CONFIG ORDER.

        This is now dynamic! The order comes from affordances_corrected.yaml,
        not from hardcoded Python lists. This test verifies the engine
        correctly builds the mapping from the config file.
        """
        action_map = engine.get_affordance_action_map()

        # Verify all affordances have unique indices
        assert len(action_map) == engine.get_num_affordances()
        assert len(set(action_map.values())) == len(action_map)

        # Verify indices start at 0 and are contiguous
        expected_indices = set(range(len(action_map)))
        actual_indices = set(action_map.values())
        assert actual_indices == expected_indices

        # Verify known affordances exist (from corrected config)
        assert "Bed" in action_map
        assert "Shower" in action_map
        assert "Bar" in action_map


class TestInstantInteractions:
    """Test instant affordance interactions."""

    def test_shower_instant_effect(self, engine, device):
        """Test Shower (instant hygiene restoration)."""
        num_agents = 4

        # Initial meters (hygiene low, money available)
        meters = torch.tensor(
            [
                [0.5, 0.2, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],  # Agent 0: low hygiene
                [0.5, 0.8, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],  # Agent 1: high hygiene
                [0.5, 0.1, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],  # Agent 2: very low hygiene
                [0.5, 0.5, 0.5, 0.01, 0.5, 0.5, 0.5, 0.5],  # Agent 3: no money
            ],
            device=device,
        )

        # Apply Shower effect
        updated_meters = engine.apply_instant_interaction(
            meters=meters,
            affordance_name="Shower",
            agent_mask=torch.tensor([True, True, True, True], device=device),
        )

        # Check effects (Shower: +0.50 hygiene, -$0.01 money)
        # Agents 0-2 should have hygiene increase
        assert updated_meters[0, 1] > meters[0, 1]  # Hygiene increased
        assert updated_meters[0, 3] < meters[0, 3]  # Money decreased

        # Agent 3 has no money, so effect might be blocked
        # (depends on implementation - document behavior)

    def test_home_meal_multi_effect(self, engine, device):
        """Test HomeMeal (instant satiation + health restoration)."""
        num_agents = 2

        # Initial meters (low satiation, money available)
        meters = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.50, 0.5, 0.5, 0.6, 0.5],  # Low satiation, moderate health
                [0.5, 0.5, 0.8, 0.50, 0.5, 0.5, 0.9, 0.5],  # High satiation, high health
            ],
            device=device,
        )

        updated_meters = engine.apply_instant_interaction(
            meters=meters,
            affordance_name="HomeMeal",
            agent_mask=torch.tensor([True, True], device=device),
        )

        # Check effects (HomeMeal: +0.60 satiation, +0.03 health, -$0.15 money)
        assert updated_meters[0, 2] > meters[0, 2]  # Satiation increased
        assert updated_meters[0, 6] > meters[0, 6]  # Health increased
        assert updated_meters[0, 3] < meters[0, 3]  # Money decreased

    def test_fastfood_penalties(self, engine, device):
        """Test FastFood (satiation + fitness/health penalties)."""
        num_agents = 1

        # Initial meters
        meters = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.50, 0.5, 0.5, 0.7, 0.8],  # Low satiation, decent fitness/health
            ],
            device=device,
        )

        updated_meters = engine.apply_instant_interaction(
            meters=meters,
            affordance_name="FastFood",
            agent_mask=torch.tensor([True], device=device),
        )

        # Check effects (FastFood: +0.45 satiation, -0.03 fitness, -0.02 health, -$0.10)
        assert updated_meters[0, 2] > meters[0, 2]  # Satiation increased
        assert updated_meters[0, 7] < meters[0, 7]  # Fitness decreased (penalty)
        assert updated_meters[0, 6] < meters[0, 6]  # Health decreased (penalty)
        assert updated_meters[0, 3] < meters[0, 3]  # Money decreased


class TestMultiTickInteractions:
    """Test multi-tick affordance interactions."""

    def test_bed_multi_tick_progression(self, engine, device):
        """Test Bed multi-tick interaction (5 ticks)."""
        num_agents = 1

        # Initial meters (low energy, money available)
        meters = torch.tensor(
            [
                [0.2, 0.5, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],  # Low energy
            ],
            device=device,
        )

        # Simulate 5 ticks of Bed interaction
        for tick in range(5):
            meters = engine.apply_multi_tick_interaction(
                meters=meters,
                affordance_name="Bed",
                current_tick=tick,
                agent_mask=torch.tensor([True], device=device),
            )

        # After 5 ticks: should have full energy restoration
        # Linear: 5 * 0.075 = 0.375, Completion bonus: 0.125, Total: 0.50
        expected_energy = 0.2 + 0.50  # Initial + restoration
        assert abs(meters[0, 0] - expected_energy) < 0.01

        # Money should be decreased (5 ticks * $0.01 = $0.05)
        expected_money = 0.50 - 0.05
        assert abs(meters[0, 3] - expected_money) < 0.01

    def test_bed_early_exit_no_bonus(self, engine, device):
        """Test Bed early exit (no completion bonus)."""
        num_agents = 1

        meters = torch.tensor(
            [
                [0.2, 0.5, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],
            ],
            device=device,
        )

        # Only 3 ticks (not full 5)
        for tick in range(3):
            meters = engine.apply_multi_tick_interaction(
                meters=meters,
                affordance_name="Bed",
                current_tick=tick,
                agent_mask=torch.tensor([True], device=device),
            )

        # Should have linear benefits (3 * 0.075 = 0.225) but NO completion bonus
        expected_energy = 0.2 + (3 * 0.075)  # No +0.125 bonus
        assert abs(meters[0, 0] - expected_energy) < 0.01

    def test_job_income_generation(self, engine, device):
        """Test Job INSTANT interaction (income generation) - matches hardcoded logic."""
        num_agents = 1

        # Initial meters (moderate energy, some money)
        meters = torch.tensor(
            [
                [0.6, 0.5, 0.5, 0.20, 0.5, 0.5, 0.5, 0.5],  # Moderate energy, low money
            ],
            device=device,
        )

        initial_energy = meters[0, 0].item()
        initial_money = meters[0, 3].item()

        # Apply Job as instant interaction (matching hardcoded behavior)
        meters = engine.apply_instant_interaction(
            meters=meters,
            affordance_name="Job",
            agent_mask=torch.tensor([True], device=device),
            check_affordability=False,
        )

        # From hardcoded logic: Money +$22.50 (0.225), Energy -15% (0.15)
        expected_money = min(1.0, initial_money + 0.225)  # Clamped to [0,1]
        expected_energy = max(0.0, initial_energy - 0.15)

        assert abs(meters[0, 3] - expected_money) < 0.01, f"Money: expected {expected_money}, got {meters[0, 3]}"
        assert abs(meters[0, 0] - expected_energy) < 0.01, f"Energy: expected {expected_energy}, got {meters[0, 0]}"


class TestOperatingHours:
    """Test time-of-day operating hours constraints."""

    def test_job_business_hours_only(self, engine, device):
        """Test Job is only available during business hours (8am-6pm)."""
        num_agents = 1
        meters = torch.tensor([[0.5] * 8], device=device)

        # Test various times of day
        # Job operating_hours: [8, 18] (8am-6pm)

        # 6am - should be CLOSED
        is_open_6am = engine.is_affordance_open("Job", time_of_day=6)
        assert not is_open_6am

        # 8am - should be OPEN
        is_open_8am = engine.is_affordance_open("Job", time_of_day=8)
        assert is_open_8am

        # 12pm - should be OPEN
        is_open_12pm = engine.is_affordance_open("Job", time_of_day=12)
        assert is_open_12pm

        # 6pm (18:00) - should be CLOSED (exclusive end)
        is_open_6pm = engine.is_affordance_open("Job", time_of_day=18)
        assert not is_open_6pm

        # 10pm - should be CLOSED
        is_open_10pm = engine.is_affordance_open("Job", time_of_day=22)
        assert not is_open_10pm

    def test_bar_midnight_wraparound(self, engine, device):
        """Test Bar operating hours wrap around midnight (6pm-4am)."""
        # Bar operating_hours: [18, 28] (6pm to 4am, wraps midnight)

        # 4pm - CLOSED
        assert not engine.is_affordance_open("Bar", time_of_day=16)

        # 6pm - OPEN
        assert engine.is_affordance_open("Bar", time_of_day=18)

        # 11pm - OPEN
        assert engine.is_affordance_open("Bar", time_of_day=23)

        # Midnight (0) - OPEN (wrapped)
        assert engine.is_affordance_open("Bar", time_of_day=0)

        # 2am - OPEN (wrapped)
        assert engine.is_affordance_open("Bar", time_of_day=2)

        # 4am - CLOSED (exclusive end: 28 % 24 = 4)
        assert not engine.is_affordance_open("Bar", time_of_day=4)

        # 5am - CLOSED
        assert not engine.is_affordance_open("Bar", time_of_day=5)

    def test_bed_always_open(self, engine, device):
        """Test Bed is available 24/7."""
        # Bed operating_hours: [0, 24]

        for hour in range(24):
            assert engine.is_affordance_open("Bed", time_of_day=hour), f"Bed should be open at {hour}:00"


class TestCostApplication:
    """Test that costs are properly applied before effects."""

    def test_insufficient_money_blocks_interaction(self, engine, device):
        """Test that insufficient money prevents interaction."""
        num_agents = 1

        # Agent with no money
        meters = torch.tensor(
            [
                [0.5, 0.2, 0.5, 0.00, 0.5, 0.5, 0.5, 0.5],  # No money
            ],
            device=device,
        )

        # Try to use Shower (costs $0.01)
        updated_meters = engine.apply_instant_interaction(
            meters=meters,
            affordance_name="Shower",
            agent_mask=torch.tensor([True], device=device),
            check_affordability=True,
        )

        # Hygiene should NOT increase (interaction blocked)
        assert updated_meters[0, 1] == meters[0, 1]

    def test_sufficient_money_allows_interaction(self, engine, device):
        """Test that sufficient money allows interaction."""
        num_agents = 1

        # Agent with money
        meters = torch.tensor(
            [
                [0.5, 0.2, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],  # Has money
            ],
            device=device,
        )

        # Use Shower (costs $0.01)
        updated_meters = engine.apply_instant_interaction(
            meters=meters,
            affordance_name="Shower",
            agent_mask=torch.tensor([True], device=device),
            check_affordability=True,
        )

        # Hygiene should increase (interaction allowed)
        assert updated_meters[0, 1] > meters[0, 1]
        assert updated_meters[0, 3] < meters[0, 3]  # Money decreased


class TestAffordanceEngineIntegration:
    """Test integration points with vectorized environment."""

    def test_get_action_masks_with_time(self, engine, device):
        """Test action masking based on time of day."""
        num_agents = 2

        # Create state at 10am (business hours)
        meters = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.50, 0.5, 0.5, 0.5, 0.5],
            ],
            device=device,
        )

        time_of_day = 10  # 10am

        # Get action masks
        action_masks = engine.get_action_masks(meters=meters, time_of_day=time_of_day, check_affordability=True, check_hours=True)

        # At 10am:
        # - Job should be AVAILABLE (8am-6pm)
        # - Bar should be UNAVAILABLE (6pm-4am)

        job_idx = engine.affordance_name_to_idx["Job"]
        bar_idx = engine.affordance_name_to_idx["Bar"]

        # Account for movement actions (UP, DOWN, LEFT, RIGHT) before affordances
        num_movement_actions = 4

        assert action_masks[0, num_movement_actions + job_idx]  # Job available
        assert not action_masks[0, num_movement_actions + bar_idx]  # Bar closed

    def test_batch_processing(self, engine, device):
        """Test that engine handles batched operations correctly."""
        num_agents = 8  # Larger batch

        meters = torch.rand(num_agents, 8, device=device)
        meters[:, 3] = 0.50  # Ensure all have money

        # Apply Shower to all agents
        agent_mask = torch.ones(num_agents, dtype=torch.bool, device=device)

        updated_meters = engine.apply_instant_interaction(meters=meters, affordance_name="Shower", agent_mask=agent_mask)

        # All agents should have hygiene increased
        assert torch.all(updated_meters[:, 1] > meters[:, 1])

        # All agents should have money decreased
        assert torch.all(updated_meters[:, 3] < meters[:, 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
