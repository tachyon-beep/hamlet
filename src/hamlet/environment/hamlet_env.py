"""
Main PettingZoo environment for Hamlet.

Implements the PettingZoo AEC (Agent-Environment-Cycle) API for the grid world
simulation where agents learn to survive by managing multiple meters.
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
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

        # Meter configuration derived from EnvironmentConfig
        self.initial_meter_values = {
            "energy": self.config.initial_energy,
            "hygiene": self.config.initial_hygiene,
            "satiation": self.config.initial_satiation,
            "money": self.config.initial_money,
            "mood": self.config.initial_mood,
            "social": self.config.initial_social,
        }

        self.meter_depletion_rates = {
            "energy": self.config.energy_depletion,
            "hygiene": self.config.hygiene_depletion,
            "satiation": self.config.satiation_depletion,
            "money": self.config.money_depletion,
            "mood": self.config.mood_depletion,
            "social": self.config.social_depletion,
        }

        self.meter_min_values = {
            "money": self.config.money_min,
        }

        self.meter_max_values = {}

        self.last_failure_reason: Optional[str] = None

        # Movement costs per step
        self.movement_cost = {
            "energy": -0.5,
            "hygiene": -0.3,
            "satiation": -0.4,
        }

        # Reward shaping config
        self.reward_mode = self.config.reward_mode
        self.use_shaped_rewards = self.reward_mode == "shaped"
        # Proximity shaping DISABLED - Level 2 progression (agent must interact to survive)
        # Proximity was causing reward hacking (standing near affordances without interacting)
        self.use_proximity_shaping = False

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
        agent = Agent(
            "agent_0",
            center_x,
            center_y,
            initial_meter_values=self.initial_meter_values,
            meter_depletion_rates=self.meter_depletion_rates,
            meter_min_values=self.meter_min_values,
            meter_max_values=self.meter_max_values,
        )
        self.agents["agent_0"] = agent
        self.grid.add_entity(agent, agent.x, agent.y)

        self.last_failure_reason = None

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

        # Apply social-driven mood penalty when lonely
        self._apply_social_mood_penalty(agent)

        # Determine termination before assigning rewards so sparse mode can respond
        done, failure_reason = self._check_done(agent)
        self.last_failure_reason = failure_reason

        if self.reward_mode == "shaped":
            reward += self._calculate_shaped_reward(
                agent,
                prev_meters,
                interaction_affordance
            )
            if done:
                reward -= 100.0  # Preserve strong penalty for failure in dense mode
        elif self.reward_mode == "sparse":
            reward += self._calculate_sparse_reward(agent, done, failure_reason)
        else:
            # Fallback to legacy reward if reward_mode is misconfigured at runtime
            reward += self._calculate_reward(agent)
            if done:
                reward -= 100.0

        self.current_step += 1

        obs = self.observe("agent_0")
        info = {
            "step": self.current_step,
            "failure_reason": failure_reason,
        }

        return obs, reward, done, info

    def observe(self, agent_id: str) -> Dict:
        """
        Return observation for the specified agent.

        Supports both full observability and partial observability (5×5 window).

        Args:
            agent_id: Agent identifier

        Returns:
            Observation dictionary
        """
        agent = self.agents[agent_id]

        # Check if partial observability is enabled
        if self.config.partial_observability:
            return self._observe_partial(agent)
        else:
            return self._observe_full(agent)

    def _observe_full(self, agent: Agent) -> Dict:
        """Full observability: agent sees entire grid."""
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
            elif affordance.name == "Gym":
                grid_array[affordance.y, affordance.x] = 8.0

        # Mark agent
        grid_array[agent.y, agent.x] = 9.0

        obs = {
            "position": np.array([agent.x, agent.y], dtype=np.float32),
            "meters": agent.meters.get_normalized_values(),
            "grid": grid_array,
        }

        return obs

    def _observe_partial(self, agent: Agent) -> Dict:
        """
        Partial observability: agent sees only local 5×5 window (Level 2 POMDP).

        The agent maintains its position in the full grid, but only observes
        a local window centered on itself. Areas outside the window are unknown.

        Returns:
            Observation dict with local window and absolute position
        """
        vision_range = self.config.vision_range  # Default 2 (for 5×5 window)
        window_size = 2 * vision_range + 1  # 5×5 window

        # Create local observation window (padded with zeros outside grid bounds)
        local_grid = np.zeros((window_size, window_size), dtype=np.float32)

        # Extract local window centered on agent
        for dy in range(-vision_range, vision_range + 1):
            for dx in range(-vision_range, vision_range + 1):
                world_x = agent.x + dx
                world_y = agent.y + dy

                # Check if within grid bounds
                if 0 <= world_x < self.grid.width and 0 <= world_y < self.grid.height:
                    # Local coordinates in observation window
                    local_x = dx + vision_range
                    local_y = dy + vision_range

                    # Check for affordances at this position
                    for affordance in self.affordances:
                        if affordance.x == world_x and affordance.y == world_y:
                            if affordance.name == "Bed":
                                local_grid[local_y, local_x] = 1.0
                            elif affordance.name == "Shower":
                                local_grid[local_y, local_x] = 2.0
                            elif affordance.name == "HomeMeal":
                                local_grid[local_y, local_x] = 3.0
                            elif affordance.name == "FastFood":
                                local_grid[local_y, local_x] = 4.0
                            elif affordance.name == "Job":
                                local_grid[local_y, local_x] = 5.0
                            elif affordance.name == "Recreation":
                                local_grid[local_y, local_x] = 6.0
                            elif affordance.name == "Bar":
                                local_grid[local_y, local_x] = 7.0
                            elif affordance.name == "Gym":
                                local_grid[local_y, local_x] = 8.0
                            break

        # Mark agent at center of local window
        center = vision_range
        local_grid[center, center] = 9.0

        obs = {
            "position": np.array([agent.x, agent.y], dtype=np.float32),  # Absolute position (for learning)
            "meters": agent.meters.get_normalized_values(),
            "grid": local_grid,  # 5×5 local observation window
        }

        return obs

    def _apply_social_mood_penalty(self, agent: Agent):
        """Apply additional mood drain when the agent is socially isolated."""
        if self.config.mood_social_penalty <= 0:
            return

        social_meter = agent.meters.get("social")
        mood_meter = agent.meters.get("mood")

        social_level = social_meter.normalize()

        if social_level >= self.config.mood_social_threshold:
            return

        threshold = max(self.config.mood_social_threshold, 1e-6)
        deficit_ratio = (threshold - social_level) / threshold
        mood_penalty = self.config.mood_social_penalty * deficit_ratio

        if mood_penalty > 0:
            mood_meter.update(-mood_penalty)

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

        essential_meters = ["energy", "hygiene", "satiation"]
        support_meters = ["mood"]

        for meter_name in essential_meters:
            meter = agent.meters.get(meter_name)
            normalized = meter.normalize()

            if normalized > 0.8:
                reward += 0.4
            elif normalized > 0.5:
                reward += 0.15
            elif normalized > 0.3:
                reward -= 0.6
            else:
                reward -= 2.5

        for meter_name in support_meters:
            meter = agent.meters.get(meter_name)
            normalized = meter.normalize()

            if normalized > 0.8:
                reward += 0.2
            elif normalized > 0.5:
                reward += 0.1
            elif normalized > 0.2:
                reward -= 0.3
            else:
                reward -= 1.0

        social_meter = agent.meters.get("social")
        social_normalized = social_meter.normalize()
        if social_normalized > 0.8:
            reward += 0.15
        elif social_normalized > 0.5:
            reward += 0.05
        elif social_normalized > 0.2:
            reward -= 0.3
        else:
            reward -= 1.2

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

        # Tier 1: Need-based interaction rewards
        if interaction_affordance is not None:
            interaction_reward = self._calculate_need_based_interaction_reward(
                agent, interaction_affordance, prev_meters
            )
            reward += interaction_reward

        # Tier 2: Proximity shaping (DISABLED for Level 2 - agent must explore/remember)
        # Previously: guided agent toward needed resources, but caused reward hacking
        # Now: agent must learn through interaction, not proximity
        # if self.use_proximity_shaping:
        #     proximity_reward = self._calculate_proximity_reward(agent)
        #     reward += proximity_reward

        # Urgency penalty DISABLED - also a form of proximity shaping
        # Agent should learn urgency through meter gradient rewards, not distance penalties
        # satiation_meter = agent.meters.get("satiation")
        # satiation_norm = satiation_meter.normalize()
        # if satiation_norm < 0.4:
        #     urgency = min(1.0, (0.4 - satiation_norm) / 0.4)
        #     dist = self._distance_to_affordances(agent, ["HomeMeal", "FastFood"])
        #     max_dist = self.grid.width + self.grid.height
        #     reward -= urgency * (dist / max_dist) * 2.0

        return reward

    def _calculate_sparse_reward(
        self,
        agent: Agent,
        done: bool,
        failure_reason: Optional[str]
    ) -> float:
        """
        Calculate sparse reward signal.

        Sparse mode focuses on survival time and terminal outcomes.
        Optional healthy-meter bonus encourages maintaining a buffer
        without providing dense directional feedback.
        """
        reward = 0.0

        if not done:
            reward += self.config.sparse_survival_reward

            if self.config.sparse_healthy_meter_bonus > 0:
                healthy_threshold = self.config.sparse_healthy_meter_threshold
                key_meters = ["energy", "hygiene", "satiation", "mood", "social"]
                if all(
                    agent.meters.get(meter_name).normalize() >= healthy_threshold
                    for meter_name in key_meters
                ):
                    reward += self.config.sparse_healthy_meter_bonus
        else:
            if failure_reason is None:
                reward += self.config.sparse_terminal_reward_success
            else:
                reward += self.config.sparse_terminal_reward_failure

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
        for meter_name in ["energy", "hygiene", "satiation", "money", "social", "mood"]:
            meter = agent.meters.get(meter_name)
            normalized = meter.normalize()
            meter_urgency = 1.0 - normalized  # Higher when meter is lower
            threshold = 0.5 if meter_name != "money" else 0.4
            if meter_urgency > urgency and normalized < threshold:
                urgency = meter_urgency
                critical_meter = meter_name

        # Only provide proximity guidance if a critical meter was found
        if critical_meter is None or urgency == 0.0:
            return 0.0

        # Map meter to affordance
        meter_to_affordance = {
            "energy": "Bed",
            "hygiene": "Shower",
            "satiation": "HomeMeal",  # Guide to cheap, healthy option (agent learns FastFood trade-off)
            "money": "Job",           # Guide toward work when money is low
            "mood": "Gym",            # Guide toward gym when mood is low
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

    def _distance_to_affordances(self, agent: Agent, names) -> float:
        max_dist = self.grid.width + self.grid.height
        min_dist = max_dist
        for affordance in self.affordances:
            if affordance.name in names:
                dist = abs(agent.x - affordance.x) + abs(agent.y - affordance.y)
                if dist < min_dist:
                    min_dist = dist
        return float(min_dist)

    def _check_done(self, agent: Agent) -> Tuple[bool, Optional[str]]:
        """
        Check if episode should terminate.

        Args:
            agent: Agent to check

        Returns:
            Tuple of (done flag, failure reason string)
        """
        fatal_meters = ["energy", "hygiene", "satiation", "mood"]

        for meter_name in fatal_meters:
            meter = agent.meters.get(meter_name)
            if meter.value <= meter.min_value:
                return True, f"{meter_name}_depleted"

        money_meter = agent.meters.get("money")
        if money_meter.value <= money_meter.min_value:
            return True, "bankrupt"

        return False, None
