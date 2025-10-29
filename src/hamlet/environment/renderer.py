"""
Rendering utilities for Hamlet environment.

Provides methods to render the grid world state for visualization.
Integrates with web interface for real-time streaming.
"""


class Renderer:
    """
    Renders the Hamlet environment state.

    Converts grid world state into formats suitable for web visualization.
    Supports JSON serialization for WebSocket transmission.
    """

    def __init__(self, grid, agents, affordances):
        """
        Initialize renderer.

        Args:
            grid: Grid instance
            agents: List of Agent instances
            affordances: List of Affordance instances
        """
        self.grid = grid
        self.agents = agents
        self.affordances = affordances

    def render_to_dict(self) -> dict:
        """
        Render current state as dictionary.

        Returns:
            Dictionary containing grid state, agent positions, meters, etc.
            Suitable for JSON serialization and WebSocket transmission.
        """
        # Serialize agents
        agents_data = []
        agents_meters = {}

        for agent in self.agents:
            agents_data.append({
                "id": agent.agent_id,
                "x": agent.x,
                "y": agent.y,
                "color": "#3b82f6",  # Default blue, can be customized
            })

            # Get meter values
            meters = agent.meters.get_normalized_values()
            agents_meters[agent.agent_id] = {
                "meters": {
                    "energy": meters["energy"],
                    "hygiene": meters["hygiene"],
                    "satiation": meters["satiation"],
                    "money": agent.meters.get("money").value,  # Money as absolute value
                    "mood": agent.meters.get("mood").value,     # Mood level (0-100)
                    "social": meters["social"],  # Social connections (normalized)
                },
            }

        # Serialize affordances
        affordances_data = []
        for affordance in self.affordances:
            affordances_data.append({
                "type": affordance.name,
                "x": affordance.x,
                "y": affordance.y,
                "qualifier": getattr(affordance, "qualifier", None),
            })

        return {
            "grid": {
                "width": self.grid.width,
                "height": self.grid.height,
                "agents": agents_data,
                "affordances": affordances_data,
            },
            "agents": agents_meters,
        }

    def render_text(self) -> str:
        """
        Render current state as ASCII text.

        Returns:
            String representation of grid for console debugging.
        """
        pass
