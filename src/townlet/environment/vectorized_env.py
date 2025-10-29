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

        # Observation: grid one-hot (64) + 6 meters (normalized)
        self.observation_dim = grid_size * grid_size + 6
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
        }

        # State tensors (initialized in reset)
        self.positions: Optional[torch.Tensor] = None  # [num_agents, 2]
        self.meters: Optional[torch.Tensor] = None  # [num_agents, 6]
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
        # [energy, hygiene, satiation, money, mood, social]
        self.meters = torch.tensor([
            [1.0, 1.0, 1.0, 0.5, 1.0, 0.5]  # Default initial values
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
        grid_encoding = torch.zeros(
            self.num_agents, self.grid_size * self.grid_size, device=self.device
        )
        flat_indices = self.positions[:, 0] * self.grid_size + self.positions[:, 1]
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

        # 3. Check terminal conditions
        self._check_dones()

        # 4. Calculate rewards (shaped rewards for now)
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
        # Movement deltas
        deltas = torch.tensor([
            [-1, 0],  # UP
            [1, 0],   # DOWN
            [0, -1],  # LEFT
            [0, 1],   # RIGHT
            [0, 0],   # INTERACT (no movement)
        ], device=self.device)

        # Apply movement
        movement_deltas = deltas[actions]  # [num_agents, 2]
        new_positions = self.positions + movement_deltas

        # Clamp to grid boundaries
        new_positions = torch.clamp(new_positions, 0, self.grid_size - 1)

        self.positions = new_positions

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

            # Apply affordance effects (simplified from Hamlet)
            if affordance_name == 'Bed':
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] + 0.8, 0.0, 1.0
                )  # Energy +80%
            elif affordance_name == 'Shower':
                self.meters[at_affordance, 1] = torch.clamp(
                    self.meters[at_affordance, 1] + 0.6, 0.0, 1.0
                )  # Hygiene +60%
            elif affordance_name == 'HomeMeal':
                self.meters[at_affordance, 2] = torch.clamp(
                    self.meters[at_affordance, 2] + 0.5, 0.0, 1.0
                )  # Satiation +50%
                self.meters[at_affordance, 3] -= 0.04  # Money -$4
            elif affordance_name == 'Job':
                self.meters[at_affordance, 3] += 0.3  # Money +$30
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.2, 0.0, 1.0
                )  # Energy -20%

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
        ], device=self.device)

        self.meters = torch.clamp(
            self.meters - depletions, 0.0, 1.0
        )

    def _check_dones(self) -> None:
        """Check terminal conditions."""
        # Terminal if any critical meter (energy, hygiene, satiation) hits 0
        critical_meters = self.meters[:, :3]  # energy, hygiene, satiation
        self.dones = (critical_meters <= 0.0).any(dim=1)

    def _calculate_shaped_rewards(self) -> torch.Tensor:
        """
        Calculate shaped rewards (Hamlet-style two-tier).

        Returns:
            rewards: [num_agents]
        """
        rewards = torch.zeros(self.num_agents, device=self.device)

        # Tier 1: Meter-based feedback
        for i, meter_name in enumerate(['energy', 'hygiene', 'satiation']):
            meter_values = self.meters[:, i]

            # Healthy (>0.8): +0.5
            rewards += torch.where(meter_values > 0.8, 0.5, 0.0)

            # Okay (0.5-0.8): +0.2
            rewards += torch.where(
                (meter_values > 0.5) & (meter_values <= 0.8), 0.2, 0.0
            )

            # Concerning (0.2-0.5): -0.5
            rewards += torch.where(
                (meter_values > 0.2) & (meter_values <= 0.5), -0.5, 0.0
            )

            # Critical (<0.2): -2.0
            rewards += torch.where(meter_values <= 0.2, -2.0, 0.0)

        # Terminal penalty
        rewards = torch.where(self.dones, -100.0, rewards)

        return rewards
