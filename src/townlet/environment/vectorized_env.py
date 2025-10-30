"""
Vectorized Hamlet environment for GPU-native training.

Batches multiple independent Hamlet environments into a single vectorized
environment with tensor operations [num_agents, ...].
"""

import torch
import numpy as np
from typing import Tuple, Optional


class VectorizedHamletEnv:
    """
    GPU-native vectorized Hamlet environment.

    Batches multiple independent environments for parallel execution.
    All state is stored as PyTorch tensors on specified device.
    """

    def __init__(
        self,
        num_agents: int,
        grid_size: int = 8,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize vectorized environment.

        Args:
            num_agents: Number of parallel agents
            grid_size: Grid dimension (grid_size × grid_size)
            device: PyTorch device (cpu or cuda)
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device

        # Observation: grid one-hot (64) + 7 meters (normalized)
        self.observation_dim = grid_size * grid_size + 7
        self.action_dim = 5  # UP, DOWN, LEFT, RIGHT, INTERACT

        # Affordance positions (from Hamlet default layout)
        self.affordances = {
            'Bed': torch.tensor([1, 1], device=device),
            'Shower': torch.tensor([2, 2], device=device),
            'HomeMeal': torch.tensor([1, 3], device=device),
            'FastFood': torch.tensor([5, 6], device=device),
            'Job': torch.tensor([6, 6], device=device),
            'Gym': torch.tensor([7, 3], device=device),
            'Bar': torch.tensor([7, 0], device=device),
            'Recreation': torch.tensor([0, 7], device=device),
            'Park': torch.tensor([0, 4], device=device),      # Free generalist (left side)
            'Doctor': torch.tensor([5, 1], device=device),    # Health specialist (between zones)
        }

        # State tensors (initialized in reset)
        self.positions: Optional[torch.Tensor] = None  # [num_agents, 2]
        self.meters: Optional[torch.Tensor] = None  # [num_agents, 7]
        self.dones: Optional[torch.Tensor] = None  # [num_agents]
        self.step_counts: Optional[torch.Tensor] = None  # [num_agents]

    def reset(self) -> torch.Tensor:
        """
        Reset all environments.

        Returns:
            observations: [num_agents, observation_dim]
        """
        # Random starting positions
        self.positions = torch.randint(
            0, self.grid_size, (self.num_agents, 2), device=self.device
        )

        # Initial meter values (normalized to [0, 1])
        # [energy, hygiene, satiation, money, mood, social, health]
        # NOTE: money=0.75 corresponds to Hamlet's money=50 in range [-100, 100]
        self.meters = torch.tensor([
            [1.0, 1.0, 1.0, 0.75, 1.0, 0.5, 1.0]  # Default initial values
        ], device=self.device).repeat(self.num_agents, 1)

        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.step_counts = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)

        return self._get_observations()

    def _get_observations(self) -> torch.Tensor:
        """
        Construct observation vector.

        Returns:
            observations: [num_agents, observation_dim]
        """
        # Grid encoding: one-hot position
        # positions[:, 0] = x (column), positions[:, 1] = y (row)
        grid_encoding = torch.zeros(
            self.num_agents, self.grid_size * self.grid_size, device=self.device
        )
        flat_indices = self.positions[:, 1] * self.grid_size + self.positions[:, 0]
        grid_encoding.scatter_(1, flat_indices.unsqueeze(1), 1.0)

        # Concatenate grid + meters
        observations = torch.cat([grid_encoding, self.meters], dim=1)

        return observations

    def step(
        self,
        actions: torch.Tensor,  # [num_agents]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one step for all agents.

        Args:
            actions: [num_agents] tensor of actions (0-4)

        Returns:
            observations: [num_agents, observation_dim]
            rewards: [num_agents]
            dones: [num_agents] bool
            info: dict with metadata
        """
        # 1. Execute actions
        self._execute_actions(actions)

        # 2. Deplete meters
        self._deplete_meters()

        # 3. Apply social-mood penalty
        self._apply_social_mood_penalty()

        # 4. Check terminal conditions
        self._check_dones()

        # 5. Calculate rewards (shaped rewards for now)
        rewards = self._calculate_shaped_rewards()

        # 5. Increment step counts
        self.step_counts += 1

        observations = self._get_observations()

        info = {
            'step_counts': self.step_counts.clone(),
            'positions': self.positions.clone(),
        }

        return observations, rewards, self.dones, info

    def _execute_actions(self, actions: torch.Tensor) -> None:
        """
        Execute movement and interaction actions.

        Args:
            actions: [num_agents] tensor
                0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=INTERACT
        """
        # Movement deltas (x, y) coordinates
        # x = horizontal (column), y = vertical (row)
        deltas = torch.tensor([
            [0, -1],  # UP - decreases y, x unchanged
            [0, 1],   # DOWN - increases y, x unchanged
            [-1, 0],  # LEFT - decreases x, y unchanged
            [1, 0],   # RIGHT - increases x, y unchanged
            [0, 0],   # INTERACT (no movement)
        ], device=self.device)

        # Apply movement
        movement_deltas = deltas[actions]  # [num_agents, 2]
        new_positions = self.positions + movement_deltas

        # Clamp to grid boundaries
        new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)

        self.positions = new_positions

        # Apply movement costs (matching Hamlet exactly)
        # Movement costs: energy -0.5%, hygiene -0.3%, satiation -0.4%
        movement_mask = (actions < 4)  # Actions 0-3 are movement
        if movement_mask.any():
            movement_costs = torch.tensor([
                0.005,  # energy: -0.5%
                0.003,  # hygiene: -0.3%
                0.004,  # satiation: -0.4%
                0.0,    # money: no cost
                0.0,    # mood: no cost
                0.0,    # social: no cost
                0.0,    # health: no cost
            ], device=self.device)
            self.meters[movement_mask] -= movement_costs.unsqueeze(0)
            self.meters = torch.clamp(self.meters, 0.0, 1.0)

        # Handle INTERACT actions
        interact_mask = (actions == 4)
        if interact_mask.any():
            self._handle_interactions(interact_mask)

    def _handle_interactions(self, interact_mask: torch.Tensor) -> None:
        """
        Handle INTERACT action at affordances.

        Args:
            interact_mask: [num_agents] bool mask
        """
        # Check each affordance
        for affordance_name, affordance_pos in self.affordances.items():
            # Distance to affordance
            distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
            at_affordance = (distances == 0) & interact_mask

            if not at_affordance.any():
                continue

            # Apply affordance effects (matching Hamlet exactly)
            # NOTE: Money is in range [-100, 100], so $X = X/200 in normalized [0, 1]
            if affordance_name == 'Bed':
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] + 0.5, 0.0, 1.0
                )  # Energy +50%
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.02, 0.0, 1.0
                )  # Health +2%
                self.meters[at_affordance, 3] -= 0.025  # Money -$5 = -5/200
            elif affordance_name == 'Shower':
                self.meters[at_affordance, 1] = torch.clamp(
                    self.meters[at_affordance, 1] + 0.4, 0.0, 1.0
                )  # Hygiene +40%
                self.meters[at_affordance, 3] -= 0.015  # Money -$3 = -3/200
            elif affordance_name == 'HomeMeal':
                self.meters[at_affordance, 2] = torch.clamp(
                    self.meters[at_affordance, 2] + 0.45, 0.0, 1.0
                )  # Satiation +45%
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.03, 0.0, 1.0
                )  # Health +3%
                self.meters[at_affordance, 3] -= 0.015  # Money -$3 = -3/200
            elif affordance_name == 'Job':
                self.meters[at_affordance, 3] += 0.1125  # Money +$22.5 = 22.5/200
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.15, 0.0, 1.0
                )  # Energy -15%
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] - 0.03, 0.0, 1.0
                )  # Health -3% (work stress)
            elif affordance_name == 'FastFood':
                self.meters[at_affordance, 2] = torch.clamp(
                    self.meters[at_affordance, 2] + 0.45, 0.0, 1.0
                )  # Satiation +45%
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] + 0.15, 0.0, 1.0
                )  # Energy +15%
                self.meters[at_affordance, 5] = torch.clamp(
                    self.meters[at_affordance, 5] + 0.01, 0.0, 1.0
                )  # Social +1%
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] - 0.02, 0.0, 1.0
                )  # Health -2% (junk food)
                self.meters[at_affordance, 3] -= 0.05  # Money -$10 = -10/200
            elif affordance_name == 'Bar':
                # Best for social + mood, but health penalty
                self.meters[at_affordance, 5] = torch.clamp(
                    self.meters[at_affordance, 5] + 0.5, 0.0, 1.0
                )  # Social +50% (BEST)
                self.meters[at_affordance, 4] = torch.clamp(
                    self.meters[at_affordance, 4] + 0.25, 0.0, 1.0
                )  # Mood +25%
                self.meters[at_affordance, 2] = torch.clamp(
                    self.meters[at_affordance, 2] + 0.3, 0.0, 1.0
                )  # Satiation +30%
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.2, 0.0, 1.0
                )  # Energy -20%
                self.meters[at_affordance, 1] = torch.clamp(
                    self.meters[at_affordance, 1] - 0.15, 0.0, 1.0
                )  # Hygiene -15%
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] - 0.05, 0.0, 1.0
                )  # Health -5% (late nights, drinking)
                self.meters[at_affordance, 3] -= 0.075  # Money -$15 = -15/200
            elif affordance_name == 'Recreation':
                self.meters[at_affordance, 4] = torch.clamp(
                    self.meters[at_affordance, 4] + 0.25, 0.0, 1.0
                )  # Mood +25%
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] + 0.12, 0.0, 1.0
                )  # Energy +12%
                self.meters[at_affordance, 3] -= 0.03  # Money -$6 = -6/200
            elif affordance_name == 'Gym':
                # Best for health + mood combo
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.15, 0.0, 1.0
                )  # Health +15%
                self.meters[at_affordance, 4] = torch.clamp(
                    self.meters[at_affordance, 4] + 0.45, 0.0, 1.0
                )  # Mood +45% (BEST)
                self.meters[at_affordance, 5] = torch.clamp(
                    self.meters[at_affordance, 5] + 0.02, 0.0, 1.0
                )  # Social +2%
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.08, 0.0, 1.0
                )  # Energy -8%
                self.meters[at_affordance, 3] -= 0.03  # Money -$6 = -6/200
            elif affordance_name == 'Park':
                # Free generalist - small amounts of health, social, mood
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.10, 0.0, 1.0
                )  # Health +10%
                self.meters[at_affordance, 5] = torch.clamp(
                    self.meters[at_affordance, 5] + 0.15, 0.0, 1.0
                )  # Social +15%
                self.meters[at_affordance, 4] = torch.clamp(
                    self.meters[at_affordance, 4] + 0.15, 0.0, 1.0
                )  # Mood +15%
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.15, 0.0, 1.0
                )  # Energy -15% (time cost)
                # Money: $0 (FREE!)
            elif affordance_name == 'Doctor':
                # Best pure health source
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.35, 0.0, 1.0
                )  # Health +35% (BEST)
                self.meters[at_affordance, 3] -= 0.06  # Money -$12 = -12/200

    def _deplete_meters(self) -> None:
        """Deplete meters each step."""
        # Depletion rates (per step, from Hamlet)
        depletions = torch.tensor([
            0.005,  # energy: 0.5% per step
            0.003,  # hygiene: 0.3%
            0.004,  # satiation: 0.4%
            0.0,    # money: no passive depletion
            0.001,  # mood: 0.1%
            0.006,  # social: 0.6%
            0.001,  # health: 0.1% (slow burn - 5x slower than energy)
        ], device=self.device)

        self.meters = torch.clamp(
            self.meters - depletions, 0.0, 1.0
        )

    def _apply_social_mood_penalty(self) -> None:
        """
        Apply additional mood drain when socially isolated.

        Matches Hamlet's social-mood coupling where low social causes
        additional mood penalties.
        """
        # Constants from EnvironmentConfig
        mood_social_penalty = 5.0  # max additional mood drop per step
        mood_social_threshold = 0.3  # below this social level, apply penalty

        social_values = self.meters[:, 5]  # social meter
        mood_values = self.meters[:, 4]    # mood meter

        # Calculate penalty only for agents below threshold
        below_threshold = social_values < mood_social_threshold

        if below_threshold.any():
            # deficit_ratio = (threshold - social) / threshold
            # Capped to prevent division issues
            threshold = max(mood_social_threshold, 1e-6)
            deficit_ratio = (threshold - social_values[below_threshold]) / threshold
            mood_penalty = mood_social_penalty * deficit_ratio / 100.0  # Normalize to [0, 1]

            # Apply penalty to mood
            self.meters[below_threshold, 4] = torch.clamp(
                mood_values[below_threshold] - mood_penalty,
                0.0,
                1.0
            )

    def _check_dones(self) -> None:
        """Check terminal conditions."""
        # Terminal if any critical meter (energy, hygiene, satiation, health) hits 0
        critical_energy_hygiene_satiation = self.meters[:, :3]  # energy, hygiene, satiation
        critical_health = self.meters[:, 6:7]  # health (slow burn)
        critical_meters = torch.cat([critical_energy_hygiene_satiation, critical_health], dim=1)
        self.dones = (critical_meters <= 0.0).any(dim=1)

    def _calculate_shaped_rewards(self) -> torch.Tensor:
        """
        Calculate shaped rewards (Hamlet-style two-tier).

        Returns:
            rewards: [num_agents]
        """
        rewards = torch.zeros(self.num_agents, device=self.device)

        # Tier 1: Essential meter-based feedback (energy, hygiene, satiation)
        for i, meter_name in enumerate(['energy', 'hygiene', 'satiation']):
            meter_values = self.meters[:, i]

            # Healthy (>0.8): +0.4
            rewards += torch.where(meter_values > 0.8, 0.4, 0.0)

            # Okay (0.5-0.8): +0.15
            rewards += torch.where(
                (meter_values > 0.5) & (meter_values <= 0.8), 0.15, 0.0
            )

            # Concerning (0.3-0.5): -0.6
            rewards += torch.where(
                (meter_values > 0.3) & (meter_values <= 0.5), -0.6, 0.0
            )

            # Critical (<=0.3): -2.5
            rewards += torch.where(meter_values <= 0.3, -2.5, 0.0)

        # Money gradient rewards (strategic buffer maintenance)
        money_values = self.meters[:, 3]

        # Comfortable buffer (>0.6): +0.5
        rewards += torch.where(money_values > 0.6, 0.5, 0.0)

        # Adequate buffer (0.4-0.6): +0.2
        rewards += torch.where(
            (money_values > 0.4) & (money_values <= 0.6), 0.2, 0.0
        )

        # Low buffer (0.2-0.4): -0.5
        rewards += torch.where(
            (money_values > 0.2) & (money_values <= 0.4), -0.5, 0.0
        )

        # Critical (<=0.2): -2.0
        rewards += torch.where(money_values <= 0.2, -2.0, 0.0)

        # Mood meter rewards (support meter)
        mood_values = self.meters[:, 4]

        # High mood (>0.8): +0.2
        rewards += torch.where(mood_values > 0.8, 0.2, 0.0)

        # Good mood (0.5-0.8): +0.1
        rewards += torch.where(
            (mood_values > 0.5) & (mood_values <= 0.8), 0.1, 0.0
        )

        # Low mood (0.2-0.5): -0.3
        rewards += torch.where(
            (mood_values > 0.2) & (mood_values <= 0.5), -0.3, 0.0
        )

        # Critical mood (<=0.2): -1.0
        rewards += torch.where(mood_values <= 0.2, -1.0, 0.0)

        # Social meter rewards (support meter)
        social_values = self.meters[:, 5]

        # High social (>0.8): +0.15
        rewards += torch.where(social_values > 0.8, 0.15, 0.0)

        # Good social (0.5-0.8): +0.05
        rewards += torch.where(
            (social_values > 0.5) & (social_values <= 0.8), 0.05, 0.0
        )

        # Low social (0.2-0.5): -0.3
        rewards += torch.where(
            (social_values > 0.2) & (social_values <= 0.5), -0.3, 0.0
        )

        # Critical social (<=0.2): -1.2
        rewards += torch.where(social_values <= 0.2, -1.2, 0.0)

        # Health meter rewards (slow burn - long-term thinking)
        health_values = self.meters[:, 6]

        # Healthy (>0.8): +0.3
        rewards += torch.where(health_values > 0.8, 0.3, 0.0)

        # Good health (0.6-0.8): +0.1
        rewards += torch.where(
            (health_values > 0.6) & (health_values <= 0.8), 0.1, 0.0
        )

        # Low health (0.4-0.6): -0.2
        rewards += torch.where(
            (health_values > 0.4) & (health_values <= 0.6), -0.2, 0.0
        )

        # Poor health (0.2-0.4): -0.5
        rewards += torch.where(
            (health_values > 0.2) & (health_values <= 0.4), -0.5, 0.0
        )

        # Critical health (<=0.2): -1.0 (will die soon!)
        rewards += torch.where(health_values <= 0.2, -1.0, 0.0)

        # Tier 2: Proximity shaping (simplified version)
        # Add small reward for being near needed affordances
        proximity_rewards = self._calculate_proximity_rewards()
        rewards += proximity_rewards

        # Urgency penalty: when satiation is low and far from food, subtract
        satiation_values = self.meters[:, 2]
        low_satiation = satiation_values < 0.4
        if low_satiation.any():
            # Calculate distance to food affordances for agents with low satiation
            for agent_idx in range(self.num_agents):
                if not low_satiation[agent_idx]:
                    continue

                # Find nearest food affordance (HomeMeal or FastFood)
                food_affordances = ['HomeMeal', 'FastFood']
                min_dist = float('inf')
                for food_name in food_affordances:
                    if food_name in self.affordances:
                        food_pos = self.affordances[food_name]
                        dist = torch.abs(self.positions[agent_idx] - food_pos).sum().item()
                        min_dist = min(min_dist, dist)

                if min_dist < float('inf'):
                    urgency = min(1.0, (0.4 - satiation_values[agent_idx].item()) / 0.4)
                    max_dist = self.grid_size * 2
                    rewards[agent_idx] -= urgency * (min_dist / max_dist) * 2.0

        # Terminal penalty
        rewards = torch.where(self.dones, -100.0, rewards)

        return rewards

    def _calculate_proximity_rewards(self) -> torch.Tensor:
        """
        Calculate proximity shaping rewards (simplified).

        Returns small positive reward for being near needed affordances.

        Returns:
            proximity_rewards: [num_agents]
        """
        rewards = torch.zeros(self.num_agents, device=self.device)

        # Meter index to affordance name mapping
        meter_to_affordance = {
            0: 'Bed',       # energy
            1: 'Shower',    # hygiene
            2: 'HomeMeal',  # satiation
            3: 'Job',       # money
            4: 'Gym',       # mood
            5: 'Bar',       # social
            6: 'Doctor',    # health
        }

        # For each agent, find most critical meter and reward proximity
        for agent_idx in range(self.num_agents):
            # Find most critical meter (highest urgency = lowest value)
            meter_vals = self.meters[agent_idx]
            urgency = 1.0 - meter_vals  # Higher when meter is lower

            # Only consider meters below threshold
            # [energy, hygiene, satiation, money, mood, social, health]
            threshold = torch.tensor([0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5], device=self.device)
            below_threshold = meter_vals < threshold

            if not below_threshold.any():
                continue

            # Find most urgent meter below threshold
            urgency_masked = urgency * below_threshold.float()
            most_urgent_idx = urgency_masked.argmax().item()

            if urgency_masked[most_urgent_idx] == 0:
                continue

            # Get target affordance
            affordance_name = meter_to_affordance.get(most_urgent_idx)
            if affordance_name is None or affordance_name not in self.affordances:
                continue

            # Calculate Manhattan distance to target
            target_pos = self.affordances[affordance_name]
            agent_pos = self.positions[agent_idx]
            distance = torch.abs(agent_pos - target_pos).sum().float()

            # Max distance on grid
            max_dist = self.grid_size * 2

            # Proximity (higher when closer)
            proximity = 1.0 - (distance / max_dist)

            # Reward = urgency × proximity × scaling (max ~0.5)
            urgency_val = urgency_masked[most_urgent_idx]
            rewards[agent_idx] = urgency_val * proximity * 0.5

        return rewards

    def get_affordance_positions(self) -> dict[str, tuple[int, int]]:
        """Get current affordance positions.

        Returns:
            Dictionary mapping affordance names to (x, y) positions
        """
        positions = {}
        for name, pos_tensor in self.affordances.items():
            # Convert tensor position to tuple
            pos = pos_tensor.cpu().tolist()
            positions[name] = (int(pos[0]), int(pos[1]))
        return positions

    def randomize_affordance_positions(self):
        """Randomize affordance positions for generalization testing.

        Ensures no two affordances occupy the same position.
        """
        import random

        # Generate list of all grid positions
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        # Shuffle and assign to affordances
        random.shuffle(all_positions)

        # Assign new positions to affordances
        for i, affordance_name in enumerate(self.affordances.keys()):
            new_pos = all_positions[i]
            self.affordances[affordance_name] = torch.tensor(new_pos, dtype=torch.long, device=self.device)
