"""
Tests for Agent and Affordance classes.
"""

import pytest
from hamlet.environment.entities import Agent, Affordance, Bed, Shower, Fridge, Job
from hamlet.environment.affordances import create_affordance


def test_agent_initialization():
    """Test that agent initializes with meters."""
    agent = Agent("agent_0", 3, 4)
    assert agent.agent_id == "agent_0"
    assert agent.x == 3
    assert agent.y == 4
    assert agent.meters is not None
    assert "energy" in agent.meters.meters
    assert "money" in agent.meters.meters


def test_agent_meters_start_full():
    """Test that agent starts with full meters (except money)."""
    agent = Agent("agent_0", 0, 0)
    assert agent.meters.get("energy").value == 100.0
    assert agent.meters.get("hygiene").value == 100.0
    assert agent.meters.get("satiation").value == 100.0
    assert agent.meters.get("money").value == 50.0  # Money starts at 50


def test_affordance_with_effects():
    """Test affordance with meter effects."""
    bed = create_affordance("Bed", 1, 1)
    assert bed.name == "Bed"
    assert bed.x == 1
    assert bed.y == 1
    assert "money" in bed.meter_effects
    assert "energy" in bed.meter_effects


def test_bed_interaction():
    """Test interacting with bed."""
    agent = Agent("agent_0", 1, 1)
    agent.meters.get("energy").value = 50.0  # Tired

    bed = create_affordance("Bed", 1, 1)
    changes = bed.interact(agent)

    assert changes  # Should return dict of changes
    assert agent.meters.get("energy").value > 50.0  # Energy restored
    assert agent.meters.get("money").value < 50.0   # Money spent


def test_shower_interaction():
    """Test interacting with shower."""
    agent = Agent("agent_0", 2, 2)
    agent.meters.get("hygiene").value = 30.0  # Dirty

    shower = create_affordance("Shower", 2, 2)
    changes = shower.interact(agent)

    assert changes
    assert agent.meters.get("hygiene").value > 30.0  # Hygiene restored
    assert agent.meters.get("money").value < 50.0    # Money spent


def test_fridge_interaction():
    """Test interacting with fridge."""
    agent = Agent("agent_0", 3, 3)
    agent.meters.get("satiation").value = 20.0  # Hungry

    fridge = create_affordance("Fridge", 3, 3)
    changes = fridge.interact(agent)

    assert changes
    assert agent.meters.get("satiation").value > 20.0  # Hunger satisfied
    assert agent.meters.get("money").value < 50.0      # Money spent


def test_job_interaction():
    """Test interacting with job."""
    agent = Agent("agent_0", 4, 4)
    initial_money = agent.meters.get("money").value

    job = create_affordance("Job", 4, 4)
    changes = job.interact(agent)

    assert changes
    assert agent.meters.get("money").value > initial_money  # Earned money
    assert agent.meters.get("energy").value < 100.0         # Energy spent
    assert agent.meters.get("hygiene").value < 100.0        # Hygiene decreased


def test_cant_afford_service():
    """Test that agent can't use service if money is too low."""
    agent = Agent("agent_0", 1, 1)
    agent.meters.get("money").value = 2.0  # Only 2 money

    bed = create_affordance("Bed", 1, 1)  # Costs 10 money
    changes = bed.interact(agent)

    assert not changes  # Should return empty dict
    assert agent.meters.get("money").value == 2.0  # No change


def test_job_works_with_no_money():
    """Test that job works even when broke."""
    agent = Agent("agent_0", 4, 4)
    agent.meters.get("money").value = 0.0

    job = create_affordance("Job", 4, 4)
    changes = job.interact(agent)

    assert changes  # Job should work
    assert agent.meters.get("money").value > 0.0  # Earned money


def test_affordance_meter_effects_defined():
    """Test that all affordances have meter effects defined."""
    bed = create_affordance("Bed", 0, 0)
    shower = create_affordance("Shower", 0, 0)
    fridge = create_affordance("Fridge", 0, 0)
    job = create_affordance("Job", 0, 0)

    assert bed.meter_effects
    assert shower.meter_effects
    assert fridge.meter_effects
    assert job.meter_effects


def test_multiple_interactions():
    """Test multiple interactions in sequence."""
    agent = Agent("agent_0", 0, 0)

    # Work to earn money
    job = create_affordance("Job", 0, 0)
    job.interact(agent)
    money_after_work = agent.meters.get("money").value
    assert money_after_work > 50.0

    # Spend money on bed
    bed = create_affordance("Bed", 0, 0)
    bed.interact(agent)
    assert agent.meters.get("money").value < money_after_work


def test_agent_can_go_broke():
    """Test that agent can spend all money and go negative."""
    agent = Agent("agent_0", 1, 1)
    bed = create_affordance("Bed", 1, 1)

    # Use bed multiple times to spend money
    for _ in range(10):
        changes = bed.interact(agent)
        if not changes:  # Can't afford anymore
            break

    # Should eventually run out of money
    assert agent.meters.get("money").value < 50.0
