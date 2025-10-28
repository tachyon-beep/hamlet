"""
Main PettingZoo environment for Hamlet.

Implements the PettingZoo AEC (Agent-Environment-Cycle) API for the grid world
simulation where agents learn to survive by managing multiple meters.
"""

import numpy as np
from typing import Dict, Tuple, Any
from .grid import Grid
from .entities import Agent
from .affordances import create_affordance
from ..training.config import EnvironmentConfig


class HamletEnv:
    """
    PettingZoo AEC environment for Hamlet survival simulation.

    Single agent initially, designed for multi-agent expansion.
    Agents navigate an 8x8 grid and interact with affordances to manage
    their energy, hygiene, satiation, and money meters.
    """

    # Action mappings
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_INTERACT = 4

    def __init__(self, config: EnvironmentConfig = None):
        """Initialize the Hamlet environment."""
        self.config = config or EnvironmentConfig()

        self.grid = Grid(
            width=self.config.grid_width,
            height=self.config.grid_height
        )

        self.agents = {}
        self.affordances = []
        self.current_step = 0
        self.num_actions = 5

        # Movement costs per step
        self.movement_cost = {
            "energy": -0.5,
            "hygiene": -0.3,
            "satiation": -0.4,
        }

        # Reward shaping config
        self.use_shaped_rewards = True  # Enable improved reward function
        self.use_proximity_shaping = True  # Enable proximity guidance

    def reset(self) -> Dict:
        """Reset the environment to initial state."""
        self.current_step = 0

        # Clear grid
        self.grid.cells = {}
        self.agents = {}
        self.affordances = []

        # Create single agent at center
        center_x = self.grid.width // 2
        center_y = self.grid.height // 2
        agent = Agent("agent_0", center_x, center_y)
        self.agents["agent_0"] = agent
        self.grid.add_entity(agent, agent.x, agent.y)

        # Place affordances
        for affordance_type, (x, y) in self.config.affordance_positions.items():
            affordance = create_affordance(affordance_type, x, y)
            self.affordances.append(affordance)
            self.grid.add_entity(affordance, x, y)

        return self.observe("agent_0")

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one timestep in the environment.

        Args:
            action: Action index (0-4)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        agent = self.agents["agent_0"]
        reward = 0.0

        # Store previous meter values for need-based rewards
        prev_meters = agent.meters.get_normalized_values()

        # Execute action
        interaction_affordance = None
        if action == self.ACTION_UP:
            self.grid.move_entity(agent, 0, -1)
            agent.meters.update_all(self.movement_cost)
        elif action == self.ACTION_DOWN:
            self.grid.move_entity(agent, 0, 1)
            agent.meters.update_all(self.movement_cost)
        elif action == self.ACTION_LEFT:
            self.grid.move_entity(agent, -1, 0)
            agent.meters.update_all(self.movement_cost)
        elif action == self.ACTION_RIGHT:
            self.grid.move_entity(agent, 1, 0)
            agent.meters.update_all(self.movement_cost)
        elif action == self.ACTION_INTERACT:
            # Try to interact with affordance at current position
            cell_contents = self.grid.get_cell_contents(agent.x, agent.y)
            interaction_found = False
            for entity in cell_contents:
                if hasattr(entity, "interact"):
                    changes = entity.interact(agent)
                    if changes:
                        interaction_affordance = entity
                        interaction_found = True
                    break

            # Penalty for failed interaction (wasted effort)
            if not interaction_found:
                agent.meters.update_all({"energy": -1.0})  # Small cost for failed attempt

        # Apply time-based depletion
        agent.meters.deplete_all()

        # Calculate reward with shaped rewards
        if self.use_shaped_rewards:
            step_reward = self._calculate_shaped_reward(agent, prev_meters, interaction_affordance)
        else:
            step_reward = self._calculate_reward(agent)  # Legacy reward function

        reward += step_reward

        # Check termination
        done = self._check_done(agent)

        # Death penalty
        if done:
            reward -= 100.0

        self.current_step += 1

        obs = self.observe("agent_0")
        info = {"step": self.current_step}

        return obs, reward, done, info

    def observe(self, agent_id: str) -> Dict:
        """
        Return observation for the specified agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Observation dictionary
        """
        agent = self.agents[agent_id]

        # Create grid representation
        grid_array = np.zeros((self.grid.height, self.grid.width), dtype=np.float32)

        # Mark affordances
        for affordance in self.affordances:
            if affordance.name == "Bed":
                grid_array[affordance.y, affordance.x] = 1.0
            elif affordance.name == "Shower":
                grid_array[affordance.y, affordance.x] = 2.0
            elif affordance.name == "HomeMeal":
                grid_array[affordance.y, affordance.x] = 3.0
            elif affordance.name == "FastFood":
                grid_array[affordance.y, affordance.x] = 4.0
            elif affordance.name == "Job":
                grid_array[affordance.y, affordance.x] = 5.0
            elif affordance.name == "Recreation":
                grid_array[affordance.y, affordance.x] = 6.0
            elif affordance.name == "Bar":
                grid_array[affordance.y, affordance.x] = 7.0

        # Mark agent
        grid_array[agent.y, agent.x] = 8.0

        obs = {
            "position": np.array([agent.x, agent.y], dtype=np.float32),
            "meters": agent.meters.get_normalized_values(),
            "grid": grid_array,
        }

        return obs

    def render(self) -> Dict:
        """
        Render the current environment state.

        Returns:
            State dictionary for visualization
        """
        agent = self.agents["agent_0"]

        state = {
            "grid": {
                "width": self.grid.width,
                "height": self.grid.height,
            },
            "agents": [
                {
                    "id": agent.agent_id,
                    "x": agent.x,
                    "y": agent.y,
                    "meters": {
                        name: meter.value
                        for name, meter in agent.meters.meters.items()
                    },
                }
            ],
            "affordances": [
                {"name": aff.name, "x": aff.x, "y": aff.y}
                for aff in self.affordances
            ],
            "step": self.current_step,
        }

        return state

    def _calculate_shaped_reward(
        self,
        agent: Agent,
        prev_meters: Dict[str, float],
        interaction_affordance = None
    ) -> float:
        """
        Calculate shaped reward using hybrid approach (Tier 1 + Tier 2).

        Combines:
        - Gradient meter rewards (continuous feedback)
        - Need-based interaction rewards (reward good decisions)
        - Proximity shaping (guide toward needed resources)

        Args:
            agent: Agent to calculate reward for
            prev_meters: Meter values before action (for need calculation)
            interaction_affordance: Affordance interacted with (if any)

        Returns:
            Shaped reward value
        """
        reward = 0.0

        # Tier 1: Gradient-based meter health rewards (biological + social)
        for meter_name in ["energy", "hygiene", "satiation", "social"]:
            meter = agent.meters.get(meter_name)
            normalized = meter.normalize()

            if normalized > 0.8:
                reward += 0.5  # Healthy state
            elif normalized > 0.5:
                reward += 0.2  # Okay state
            elif normalized > 0.2:
                reward -= 0.5  # Concerning state
            else:
                reward -= 2.0  # Critical state

        # Tier 1: Money gradient rewards (strategic buffer maintenance)
        # Different thresholds than biological meters - money is a buffer resource
        money_meter = agent.meters.get("money")
        money_normalized = money_meter.normalize()

        if money_normalized > 0.6:
            reward += 0.5  # Comfortable buffer (2+ cycles)
        elif money_normalized > 0.4:
            reward += 0.2  # Adequate buffer (1-2 cycles)
        elif money_normalized > 0.2:
            reward -= 0.5  # Low buffer - work soon!
        else:
            reward -= 2.0  # Critical - work now!

        # Tier 1: Stress gradient rewards (INVERTED - low stress is good)
        stress_meter = agent.meters.get("stress")
        stress_normalized = stress_meter.normalize()

        if stress_normalized < 0.2:
            reward += 0.5  # Low stress (healthy)
        elif stress_normalized < 0.5:
            reward += 0.2  # Moderate stress (manageable)
        elif stress_normalized < 0.8:
            reward -= 0.5  # High stress (concerning)
        else:
            reward -= 2.0  # Critical stress (take a break!)

        # Tier 1: Need-based interaction rewards
        if interaction_affordance is not None:
            interaction_reward = self._calculate_need_based_interaction_reward(
                agent, interaction_affordance, prev_meters
            )
            reward += interaction_reward

        # Tier 2: Proximity shaping (guide toward needed resources)
        if self.use_proximity_shaping:
            proximity_reward = self._calculate_proximity_reward(agent)
            reward += proximity_reward

        return reward

    def _calculate_need_based_interaction_reward(
        self,
        agent: Agent,
        affordance,
        prev_meters: Dict[str, float]
    ) -> float:
        """
        Calculate interaction reward based on meter need.

        Rewards using an affordance more when the corresponding meter is low.

        Args:
            agent: Agent that interacted
            affordance: Affordance that was used
            prev_meters: Meter values before interaction

        Returns:
            Need-based interaction reward
        """
        reward = 0.0

        # Check each meter affected by the affordance
        for meter_name, delta in affordance.meter_effects.items():
            if delta > 0 and meter_name in prev_meters:  # Positive effect on a tracked meter
                need = 1.0 - prev_meters[meter_name]  # Higher when meter was low

                # Money gets amplified need to encourage buffer maintenance
                # Agent should work proactively (at 40-50%), not desperately (at 10-20%)
                if meter_name == "money":
                    need = min(need * 1.5, 1.0)  # 1.5x multiplier, capped at 1.0

                # Reward = need × effect_strength × scaling
                # Effect strength normalized to 0-1 (divide by typical max effect ~50)
                effect_strength = delta / 50.0
                reward += need * effect_strength * 2.0

        return reward

    def _calculate_proximity_reward(self, agent: Agent) -> float:
        """
        Calculate proximity reward to guide agent toward needed resources.

        When a meter is low (<50%), reward moving closer to the affordance
        that would help that meter.

        Args:
            agent: Agent to calculate reward for

        Returns:
            Proximity shaping reward
        """
        # Find most critical meter
        critical_meter = None
        urgency = 0.0

        # Check depletion-based meters (LOW is bad)
        for meter_name in ["energy", "hygiene", "satiation", "money", "social"]:
            meter = agent.meters.get(meter_name)
            normalized = meter.normalize()
            meter_urgency = 1.0 - normalized  # Higher when meter is lower
            if meter_urgency > urgency and normalized < 0.5:  # Only if below 50%
                urgency = meter_urgency
                critical_meter = meter_name

        # Check stress meter (HIGH is bad - inverted logic)
        stress_meter = agent.meters.get("stress")
        stress_normalized = stress_meter.normalize()
        stress_urgency = stress_normalized  # Higher when stress is higher
        if stress_urgency > urgency and stress_normalized > 0.5:  # Only if above 50%
            urgency = stress_urgency
            critical_meter = "stress"

        # Only provide proximity guidance if a critical meter was found
        if critical_meter is None or urgency == 0.0:
            return 0.0

        # Map meter to affordance
        meter_to_affordance = {
            "energy": "Bed",
            "hygiene": "Shower",
            "satiation": "HomeMeal",  # Guide to cheap, healthy option (agent learns FastFood trade-off)
            "money": "Job",           # Guide toward work when money is low
            "stress": "Recreation",   # Guide toward recreation when stress is high
            "social": "Bar",          # Guide toward bar when lonely (ONLY source!)
        }

        if critical_meter not in meter_to_affordance:
            return 0.0

        # Find target affordance
        target_name = meter_to_affordance[critical_meter]
        target = None
        for affordance in self.affordances:
            if affordance.name == target_name:
                target = affordance
                break

        if target is None:
            return 0.0

        # Calculate Manhattan distance
        dist = abs(agent.x - target.x) + abs(agent.y - target.y)
        max_dist = self.grid.width + self.grid.height

        # Calculate proximity (urgency already calculated above)
        proximity = 1.0 - (dist / max_dist)  # Higher when closer

        # Reward = urgency × proximity × scaling
        # Scaled to be smaller than main rewards (max ~0.5)
        reward = urgency * proximity * 0.5

        return reward

    def _calculate_reward(self, agent: Agent) -> float:
        """
        Calculate reward for current state (LEGACY VERSION).

        This is the original simple reward function, kept for comparison.
        Use use_shaped_rewards=False to enable.

        Args:
            agent: Agent to calculate reward for

        Returns:
            Reward value
        """
        reward = 1.0  # Survival bonus

        # Penalty for critical meters
        for meter_name in ["energy", "hygiene", "satiation"]:
            meter = agent.meters.get(meter_name)
            if meter.is_critical():
                reward -= 2.0  # Heavy penalty for critical state
            elif meter.normalize() < 0.5:
                reward -= 0.5  # Moderate penalty for low meters

        return reward

    def _check_done(self, agent: Agent) -> bool:
        """
        Check if episode should terminate.

        Args:
            agent: Agent to check

        Returns:
            True if episode should end
        """
        # Check if any biological meter is at zero
        for meter_name in ["energy", "hygiene", "satiation"]:
            meter = agent.meters.get(meter_name)
            if meter.value <= 0.0:
                return True

        return False
