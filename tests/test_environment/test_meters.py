"""
Tests for Meter classes.
"""

import pytest
from hamlet.environment.meters import Meter, Energy, Hygiene, Satiation, Money, MeterCollection


def test_meter_initialization():
    """Test that meters initialize with correct values."""
    energy = Energy()
    assert energy.name == "energy"
    assert energy.value == 100.0
    assert energy.min_value == 0.0
    assert energy.max_value == 100.0
    assert energy.depletion_rate == 0.5


def test_meter_update_positive():
    """Test updating meter with positive value."""
    meter = Energy()
    meter.value = 50.0
    meter.update(20.0)
    assert meter.value == 70.0


def test_meter_update_negative():
    """Test updating meter with negative value."""
    meter = Energy()
    meter.update(-30.0)
    assert meter.value == 70.0


def test_meter_update_clamps_to_max():
    """Test that update clamps to maximum value."""
    meter = Energy()
    meter.update(50.0)  # 100 + 50 = 150, should clamp to 100
    assert meter.value == 100.0


def test_meter_update_clamps_to_min():
    """Test that update clamps to minimum value."""
    meter = Energy()
    meter.update(-150.0)  # 100 - 150 = -50, should clamp to 0
    assert meter.value == 0.0


def test_meter_deplete():
    """Test natural depletion."""
    meter = Energy()
    initial = meter.value
    meter.deplete()
    assert meter.value == initial - meter.depletion_rate


def test_meter_normalize():
    """Test normalization to [0, 1] range."""
    meter = Energy()
    meter.value = 50.0
    assert meter.normalize() == 0.5

    meter.value = 100.0
    assert meter.normalize() == 1.0

    meter.value = 0.0
    assert meter.normalize() == 0.0


def test_meter_is_critical():
    """Test critical threshold detection."""
    meter = Energy()
    meter.value = 50.0
    assert not meter.is_critical()

    meter.value = 19.0  # Below 20% of 100
    assert meter.is_critical()

    meter.value = 0.0
    assert meter.is_critical()


def test_money_meter_negative_allowed():
    """Test that money meter allows negative values."""
    money = Money()
    money.update(-200.0)
    assert money.value == -100.0  # 50 - 200 = -150, clamped to min -100
    assert money.value == money.min_value


def test_meter_collection():
    """Test that meter collection manages multiple meters."""
    collection = MeterCollection()
    assert "energy" in collection.meters
    assert "hygiene" in collection.meters
    assert "satiation" in collection.meters
    assert "money" in collection.meters


def test_meter_collection_get():
    """Test getting individual meters."""
    collection = MeterCollection()
    energy = collection.get("energy")
    assert energy.name == "energy"
    assert isinstance(energy, Energy)


def test_meter_collection_update_all():
    """Test updating multiple meters at once."""
    collection = MeterCollection()
    deltas = {
        "energy": -20.0,
        "hygiene": -10.0,
        "money": 25.0
    }
    collection.update_all(deltas)

    assert collection.get("energy").value == 80.0  # 100 - 20
    assert collection.get("hygiene").value == 90.0  # 100 - 10
    assert collection.get("money").value == 75.0   # 50 + 25


def test_meter_collection_deplete_all():
    """Test depleting all meters."""
    collection = MeterCollection()
    initial_energy = collection.get("energy").value
    initial_hygiene = collection.get("hygiene").value

    collection.deplete_all()

    assert collection.get("energy").value == initial_energy - 0.5
    assert collection.get("hygiene").value == initial_hygiene - 0.3
    assert collection.get("satiation").value == 100.0 - 0.4
    assert collection.get("money").value == 50.0  # Money doesn't deplete


def test_meter_collection_get_normalized_values():
    """Test getting all normalized meter values."""
    collection = MeterCollection()
    collection.get("energy").value = 50.0
    collection.get("money").value = 0.0

    normalized = collection.get_normalized_values()

    assert normalized["energy"] == 0.5
    assert normalized["hygiene"] == 1.0
    assert normalized["money"] == 0.5  # (0 - (-100)) / (100 - (-100)) = 100/200 = 0.5


def test_meter_collection_is_any_critical():
    """Test checking if any meter is critical."""
    collection = MeterCollection()
    assert not collection.is_any_critical()

    collection.get("energy").value = 10.0  # Critical!
    assert collection.is_any_critical()


def test_multiple_depletion_cycles():
    """Test meters depleting over multiple cycles."""
    meter = Energy()
    for _ in range(10):
        meter.deplete()

    assert meter.value == 100.0 - (0.5 * 10)
    assert meter.value == 95.0
