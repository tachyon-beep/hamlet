# tests/test_townlet/test_affordance_config.py
import pytest
from townlet.environment.affordance_config import AffordanceConfig, AFFORDANCE_CONFIGS


def test_affordance_config_structure():
    """Verify affordance config has required fields."""
    bed_config = AFFORDANCE_CONFIGS['Bed']

    assert bed_config['required_ticks'] == 5
    assert bed_config['cost_per_tick'] == 0.01
    assert bed_config['operating_hours'] == (0, 24)
    assert 'linear' in bed_config['benefits']
    assert 'completion' in bed_config['benefits']


def test_affordance_config_benefit_math():
    """Verify 75/25 split is correct."""
    job_config = AFFORDANCE_CONFIGS['Job']

    # Job total: $22.5, 4 ticks
    # Linear: 75% = $16.875 / 4 = $4.21875 per tick
    # Completion: 25% = $5.625

    linear_money = job_config['benefits']['linear']['money']
    completion_money = job_config['benefits']['completion']['money']

    assert abs(linear_money - 0.140625) < 0.0001  # ($22.5 * 0.75) / 4
    assert abs(completion_money - 0.05625) < 0.0001  # $22.5 * 0.25


def test_dynamic_affordance_coffeeshop_bar():
    """Verify CoffeeShop and Bar share position but different hours."""
    coffee = AFFORDANCE_CONFIGS['CoffeeShop']
    bar = AFFORDANCE_CONFIGS['Bar']

    # CoffeeShop: daytime (8-18)
    assert coffee['operating_hours'] == (8, 18)

    # Bar: evening/night (18-4, wraps midnight)
    assert bar['operating_hours'] == (18, 4)
