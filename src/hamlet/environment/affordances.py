"""
Affordance registry and configuration.

Centralizes affordance definitions and makes it easy to add new affordances
without modifying core environment code.
"""

from typing import Dict, Type
from .entities import Affordance, Bed, Shower, HomeMeal, FastFood, Job, Recreation, Bar


# Affordance registry - maps affordance types to their meter effects
# UPDATED: Rebalanced economics for sustainability (Tier 1 economic fix)
AFFORDANCE_EFFECTS: Dict[str, Dict[str, float]] = {
    "Bed": {
        "money": -5.0,   # Reduced from -10.0
        "energy": 50.0,
        "hygiene": 5.0,
    },
    "Shower": {
        "money": -3.0,   # Reduced from -5.0
        "hygiene": 40.0,
        "energy": -3.0,
    },
    "HomeMeal": {
        "money": -3.0,     # Cheap (home cooking)
        "satiation": 45.0,
        "energy": 35.0,    # Good energy boost (nutritious)
    },
    "FastFood": {
        "money": -10.0,    # Expensive (convenience premium)
        "satiation": 45.0,
        "energy": 15.0,    # Mediocre energy (less healthy)
    },
    "Job": {
        # NOTE: Job has dynamic payment ($15-30 based on energy/hygiene)
        # These are placeholder values; actual effects calculated in Job.interact()
        "money": 30.0,
        "energy": -15.0,
        "hygiene": -10.0,
        "stress": 25.0,
    },
    "Recreation": {
        "money": -8.0,   # Costs money (entertainment/leisure)
        "stress": -40.0, # Significantly reduces stress
        "energy": 10.0,  # Small energy boost (relaxing)
    },
    "Bar": {
        "money": -15.0,    # Expensive night out
        "energy": -20.0,   # Tiring (late night)
        "hygiene": -15.0,  # Get dirty/sweaty
        "social": 50.0,    # ONLY source of social (mandatory!)
        "satiation": 30.0, # Eat while there
        "stress": -25.0,   # Social reduces stress
    },
}

# Economic balance (updated with dual food sources and job penalties):
# Home zone cycle: Bed ($5) + Shower ($3) + HomeMeal ($3) = $11
# Work zone: FastFood ($10)
# Social: Recreation ($8) + Bar ($15) = $23
# Full cycle (all affordances): $11 + $10 + $23 = $44
# Job income: $30 (healthy) or $15 (tired/dirty)
# DEFICIT: -$14/cycle (healthy) or -$29/cycle (unhealthy)
# Strategy required:
#   - Must work 2x per cycle minimum (if healthy)
#   - Must maintain energy/hygiene for full job pay
#   - Choose HomeMeal (cheap) vs FastFood (convenient near work)


# Affordance class registry
AFFORDANCE_CLASSES: Dict[str, Type[Affordance]] = {
    "Bed": Bed,
    "Shower": Shower,
    "HomeMeal": HomeMeal,
    "FastFood": FastFood,
    "Job": Job,
    "Recreation": Recreation,
    "Bar": Bar,
}


def create_affordance(affordance_type: str, x: int, y: int) -> Affordance:
    """
    Factory function to create affordances with proper configuration.

    Args:
        affordance_type: Type name (e.g., "Bed", "Shower")
        x: X position
        y: Y position

    Returns:
        Configured affordance instance
    """
    if affordance_type not in AFFORDANCE_CLASSES:
        raise ValueError(f"Unknown affordance type: {affordance_type}")

    affordance = AFFORDANCE_CLASSES[affordance_type](x, y)
    affordance.meter_effects = AFFORDANCE_EFFECTS[affordance_type].copy()
    return affordance


def register_affordance(name: str, affordance_class: Type[Affordance], effects: Dict[str, float]):
    """
    Register a new affordance type.

    Args:
        name: Affordance type name
        affordance_class: Class implementing the affordance
        effects: Dictionary of meter effects
    """
    AFFORDANCE_CLASSES[name] = affordance_class
    AFFORDANCE_EFFECTS[name] = effects.copy()
