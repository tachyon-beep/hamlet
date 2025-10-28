"""
Entity definitions for Hamlet.

Defines agents and affordances (services) that exist in the grid world.
"""

from .meters import MeterCollection


class Agent:
    """
    An agent in the Hamlet world.

    Has a position, meter collection, and can take actions.
    Future: relationship tracking for multi-agent social dynamics.
    """

    def __init__(self, agent_id: str, x: int, y: int):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier for this agent
            x: Initial x position
            y: Initial y position
        """
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.meters = MeterCollection()


class Affordance:
    """
    Base class for affordances (services) in the world.

    Affordances are locations agents can interact with to affect their meters.
    Examples: Bed, Fridge, Shower, Job
    """

    def __init__(self, x: int, y: int, name: str):
        """
        Initialize an affordance.

        Args:
            x: X position in grid
            y: Y position in grid
            name: Human-readable name
        """
        self.x = x
        self.y = y
        self.name = name
        self.meter_effects = {}  # Dict[str, float] - meter name -> delta

    def interact(self, agent: Agent) -> dict:
        """
        Agent interacts with this affordance.

        Args:
            agent: The agent interacting

        Returns:
            Dictionary of meter changes applied
        """
        # Check if agent can afford (money check for paid services)
        if "money" in self.meter_effects and self.meter_effects["money"] < 0:
            required_money = abs(self.meter_effects["money"])
            if agent.meters.get("money").value < required_money:
                # Can't afford this service
                return {}

        # Apply meter effects
        agent.meters.update_all(self.meter_effects)

        return self.meter_effects.copy()


class Bed(Affordance):
    """Bed affordance: Money (-), Energy (++), Hygiene (+)"""

    def __init__(self, x: int, y: int):
        super().__init__(x, y, "Bed")


class Shower(Affordance):
    """Shower affordance: Money (-), Hygiene (++), Energy (-)"""

    def __init__(self, x: int, y: int):
        super().__init__(x, y, "Shower")


class HomeMeal(Affordance):
    """HomeMeal: Cheap, healthy home cooking. Money (-), Satiation (++), Energy (++)"""

    def __init__(self, x: int, y: int):
        super().__init__(x, y, "HomeMeal")


class FastFood(Affordance):
    """FastFood: Expensive, convenient near work. Money (---), Satiation (++), Energy (+)"""

    def __init__(self, x: int, y: int):
        super().__init__(x, y, "FastFood")


class Job(Affordance):
    """Job affordance: Money (++ varies by energy/hygiene), Energy (--), Hygiene (--), Stress (++), Social (+)"""

    def __init__(self, x: int, y: int):
        super().__init__(x, y, "Job")

    def interact(self, agent: Agent) -> dict:
        """
        Dynamic job payment based on agent condition.

        Low energy or hygiene reduces productivity → lower pay
        """
        energy_normalized = agent.meters.get("energy").normalize()
        hygiene_normalized = agent.meters.get("hygiene").normalize()

        # Base payment: $30
        # Penalty: -50% if energy < 40% OR hygiene < 40%
        base_payment = 30.0

        if energy_normalized < 0.4 or hygiene_normalized < 0.4:
            payment = base_payment * 0.5  # $15 (tired or dirty = poor performance)
        else:
            payment = base_payment  # $30 (healthy = full productivity)

        # Apply dynamic effects
        effects = {
            "money": payment,
            "energy": -15.0,
            "hygiene": -10.0,
            "stress": 25.0,
            "social": 0.05,  # Small social bump from coworker interactions
        }

        agent.meters.update_all(effects)
        return effects


class Recreation(Affordance):
    """Recreation affordance: Money (-), Stress (---), Energy (+)"""

    def __init__(self, x: int, y: int):
        super().__init__(x, y, "Recreation")


class Bar(Affordance):
    """Bar affordance: Energy (--), Hygiene (--), Money (---), Social (+++), Satiation (+), Stress (--)"""

    def __init__(self, x: int, y: int):
        super().__init__(x, y, "Bar")
