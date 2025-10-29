"""
Meter system for Hamlet.

Defines the base Meter class and concrete meter implementations.
Meters represent agent needs that deplete over time and can be replenished
through affordance interactions.
"""

from typing import Dict, Optional


class Meter:
    """
    Base class for agent meters.

    Meters represent quantifiable agent needs (energy, hygiene, etc.)
    that deplete over time and affect rewards/survival.
    """

    def __init__(
        self,
        name: str,
        initial_value: float = 100.0,
        min_value: float = 0.0,
        max_value: float = 100.0,
        depletion_rate: float = 1.0,
    ):
        """
        Initialize a meter.

        Args:
            name: Meter name
            initial_value: Starting value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            depletion_rate: Amount depleted per timestep
        """
        self.name = name
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.depletion_rate = depletion_rate

    def update(self, delta: float):
        """
        Update meter value by delta.

        Args:
            delta: Amount to change (positive or negative)
        """
        self.value += delta
        self.value = max(self.min_value, min(self.max_value, self.value))

    def deplete(self):
        """Apply natural depletion for one timestep."""
        self.update(-self.depletion_rate)

    def normalize(self) -> float:
        """Return normalized value in range [0, 1]."""
        if self.max_value == self.min_value:
            return 0.0
        return (self.value - self.min_value) / (self.max_value - self.min_value)

    def is_critical(self) -> bool:
        """Check if meter is at critically low level."""
        threshold = self.min_value + 0.2 * (self.max_value - self.min_value)
        return self.value < threshold


class Energy(Meter):
    """Energy meter: depletes with activity, restored by bed."""

    def __init__(self):
        super().__init__(name="energy", depletion_rate=0.5)


class Hygiene(Meter):
    """Hygiene meter: depletes with activity, restored by shower."""

    def __init__(self):
        super().__init__(name="hygiene", depletion_rate=0.3)


class Satiation(Meter):
    """Satiation meter: hunger/fullness, depletes over time, restored by fridge."""

    def __init__(self):
        super().__init__(name="satiation", depletion_rate=0.4)


class Money(Meter):
    """Money meter: earned from job, spent on services."""

    def __init__(self):
        super().__init__(name="money", initial_value=50.0, min_value=-100.0, depletion_rate=0.0)


class Mood(Meter):
    """Mood meter: declines over time, boosted by recreation/socialising."""

    def __init__(self):
        # Starts high (good mood) and gently declines without upkeep
        super().__init__(name="mood", initial_value=100.0, min_value=0.0, depletion_rate=0.1)


class Social(Meter):
    """Social meter: depletes over time, ONLY restored by Bar (mandatory sink)."""

    def __init__(self):
        # Starts at 50 (mid-level), depletes faster than bio meters
        super().__init__(name="social", initial_value=50.0, depletion_rate=0.6)


class MeterCollection:
    """
    Manages all meters for an agent.

    Provides unified interface for updating, querying, and managing
    multiple meters simultaneously.
    """

    def __init__(
        self,
        initial_values: Optional[Dict[str, float]] = None,
        depletion_rates: Optional[Dict[str, float]] = None,
        min_values: Optional[Dict[str, float]] = None,
        max_values: Optional[Dict[str, float]] = None,
    ):
        """Initialize meters with optional configuration overrides."""

        self.meters = {
            "energy": Energy(),
            "hygiene": Hygiene(),
            "satiation": Satiation(),
            "money": Money(),
            "mood": Mood(),
            "social": Social(),
        }

        if initial_values:
            for name, value in initial_values.items():
                if name in self.meters:
                    self.meters[name].value = value

        if depletion_rates:
            for name, rate in depletion_rates.items():
                if name in self.meters:
                    self.meters[name].depletion_rate = rate

        if min_values:
            for name, minimum in min_values.items():
                if name in self.meters:
                    self.meters[name].min_value = minimum

        if max_values:
            for name, maximum in max_values.items():
                if name in self.meters:
                    self.meters[name].max_value = maximum

    def get(self, name: str) -> Meter:
        """Get meter by name."""
        return self.meters[name]

    def update_all(self, deltas: dict):
        """
        Update multiple meters at once.

        Args:
            deltas: Dict mapping meter names to change amounts
        """
        for name, delta in deltas.items():
            if name in self.meters:
                self.meters[name].update(delta)

    def deplete_all(self):
        """Apply natural depletion to all meters."""
        for meter in self.meters.values():
            meter.deplete()

    def get_normalized_values(self) -> dict:
        """Get all meter values normalized to [0, 1]."""
        return {name: meter.normalize() for name, meter in self.meters.items()}

    def is_any_critical(self) -> bool:
        """Check if any meter is critically low."""
        return any(meter.is_critical() for meter in self.meters.values())
