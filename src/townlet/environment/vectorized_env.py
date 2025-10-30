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
        partial_observability: bool = False,
        vision_range: int = 2,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_agents: Number of parallel agents
            grid_size: Grid dimension (grid_size × grid_size)
            device: PyTorch device (cpu or cuda)
            partial_observability: If True, agent sees only local window (POMDP)
            vision_range: Radius of vision window (2 = 5×5 window)
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device
        self.partial_observability = partial_observability
        self.vision_range = vision_range

        # Observation dimensions depend on observability mode
        if partial_observability:
            # Level 2 POMDP: local window + position + meters
            window_size = 2 * vision_range + 1  # 5×5 for vision_range=2
            self.observation_dim = window_size * window_size + 2 + 8  # Grid + position + 8 meters
        else:
            # Level 1: full grid one-hot + meters
            self.observation_dim = grid_size * grid_size + 8  # Grid one-hot + 8 meters

        self.action_dim = 5  # UP, DOWN, LEFT, RIGHT, INTERACT

        # Affordance positions (from Hamlet default layout)
        self.affordances = {
            # Basic survival (tiered)
            'Bed': torch.tensor([1, 1], device=device),           # Energy restoration tier 1
            'LuxuryBed': torch.tensor([2, 1], device=device),     # Energy restoration tier 2
            'Shower': torch.tensor([2, 2], device=device),
            'HomeMeal': torch.tensor([1, 3], device=device),
            'FastFood': torch.tensor([5, 6], device=device),
            # Income sources
            'Job': torch.tensor([6, 6], device=device),          # Office work
            'Labor': torch.tensor([7, 6], device=device),        # Physical labor
            # Fitness/Social builders (secondary meters)
            'Gym': torch.tensor([7, 3], device=device),
            'Bar': torch.tensor([7, 0], device=device),
            'Park': torch.tensor([0, 4], device=device),
            # Mood restoration (primary meter - tier 1 & 2)
            'Recreation': torch.tensor([0, 7], device=device),
            'Therapist': torch.tensor([1, 7], device=device),
            # Health restoration (primary meter - tier 1 & 2)
            'Doctor': torch.tensor([5, 1], device=device),
            'Hospital': torch.tensor([6, 1], device=device),
        }

        # State tensors (initialized in reset)
        self.positions: Optional[torch.Tensor] = None  # [num_agents, 2]
        self.meters: Optional[torch.Tensor] = None  # [num_agents, 8]
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
        # [energy, hygiene, satiation, money, mood, social, health, fitness]
        # NOTE: money=0.75 corresponds to $75 in range [0, 100] (no debt allowed)
        self.meters = torch.tensor([
            [1.0, 1.0, 1.0, 0.75, 1.0, 0.5, 1.0, 0.5]  # Default initial values (fitness starts moderate)
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
        if self.partial_observability:
            return self._get_partial_observations()
        else:
            return self._get_full_observations()

    def _get_full_observations(self) -> torch.Tensor:
        """Get full grid observations (Level 1)."""
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

    def _get_partial_observations(self) -> torch.Tensor:
        """
        Get partial observations (Level 2 POMDP).

        Agent sees only local 5×5 window centered on its position.
        Observation includes:
        - Local grid: affordances visible in window (5×5 = 25 dims)
        - Position: agent's (x, y) coordinates normalized (2 dims)
        - Meters: 8 meter values (8 dims)
        Total: 35 dims

        Returns:
            observations: [num_agents, 35]
        """
        window_size = 2 * self.vision_range + 1  # 5 for vision_range=2
        local_grids = []

        for agent_idx in range(self.num_agents):
            agent_pos = self.positions[agent_idx]  # [x, y]
            local_grid = torch.zeros(window_size * window_size, device=self.device)

            # Extract local window centered on agent
            for dy in range(-self.vision_range, self.vision_range + 1):
                for dx in range(-self.vision_range, self.vision_range + 1):
                    world_x = agent_pos[0] + dx
                    world_y = agent_pos[1] + dy

                    # Check if position is within grid bounds
                    if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                        # Check if there's an affordance at this position
                        has_affordance = False
                        for affordance_pos in self.affordances.values():
                            if (affordance_pos[0] == world_x and
                                affordance_pos[1] == world_y):
                                has_affordance = True
                                break

                        # Encode in local grid (1 = affordance, 0 = empty/out-of-bounds)
                        if has_affordance:
                            local_y = dy + self.vision_range
                            local_x = dx + self.vision_range
                            local_idx = local_y * window_size + local_x
                            local_grid[local_idx] = 1.0

            local_grids.append(local_grid)

        # Stack all local grids
        local_grids_batch = torch.stack(local_grids)  # [num_agents, 25]

        # Normalize positions to [0, 1]
        normalized_positions = self.positions.float() / (self.grid_size - 1)

        # Concatenate: local_grid + position + meters
        observations = torch.cat([
            local_grids_batch,
            normalized_positions,
            self.meters
        ], dim=1)

        return observations

    def get_action_masks(self) -> torch.Tensor:
        """
        Get action masks for all agents (invalid actions = False).

        Action masking prevents agents from selecting movements that would
        take them off the grid. This saves exploration budget and speeds learning.

        Returns:
            action_masks: [num_agents, 5] bool tensor
                True = valid action, False = invalid
                Actions: [UP, DOWN, LEFT, RIGHT, INTERACT]
        """
        action_masks = torch.ones(
            self.num_agents, 5, dtype=torch.bool, device=self.device
        )

        # Check boundary constraints
        # positions[:, 0] = x (column), positions[:, 1] = y (row)
        at_top = (self.positions[:, 1] == 0)                     # y == 0
        at_bottom = (self.positions[:, 1] == self.grid_size - 1) # y == max
        at_left = (self.positions[:, 0] == 0)                    # x == 0
        at_right = (self.positions[:, 0] == self.grid_size - 1)  # x == max

        # Mask invalid movements
        action_masks[at_top, 0] = False    # Can't go UP at top edge
        action_masks[at_bottom, 1] = False # Can't go DOWN at bottom edge
        action_masks[at_left, 2] = False   # Can't go LEFT at left edge
        action_masks[at_right, 3] = False  # Can't go RIGHT at right edge
        # INTERACT (action 4) is always valid

        return action_masks

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

        # 2. Deplete meters (base passive decay)
        self._deplete_meters()

        # 3. Cascading effects (coupled differential equations!)
        self._apply_secondary_to_primary_effects()  # Satiation/Fitness/Mood → Health/Energy
        self._apply_tertiary_to_secondary_effects() # Hygiene/Social → Satiation/Fitness/Mood
        self._apply_tertiary_to_primary_effects()   # Hygiene/Social → Health/Energy (weak)

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
                0.0,    # fitness: no cost
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
        # Affordance costs in dollars (will convert to normalized form)
        # Money normalization: normalized = (dollars / 200) + 0.5
        # So $0 = 0.5, $5 = 0.525, $100 = 1.0
        # Free affordances: Job, Labor, Park
        affordance_costs_dollars = {
            'Bed': 5,
            'LuxuryBed': 11,
            'Shower': 3,
            'HomeMeal': 3,
            'FastFood': 10,
            'Recreation': 6,
            'Gym': 8,
            'Bar': 15,
            'Therapist': 15,
            'Doctor': 8,
            'Hospital': 15,
            'Job': 0,
            'Labor': 0,
            'Park': 0,
        }

        # Check each affordance
        for affordance_name, affordance_pos in self.affordances.items():
            # Distance to affordance
            distances = torch.abs(self.positions - affordance_pos).sum(dim=1)
            at_affordance = (distances == 0) & interact_mask

            if not at_affordance.any():
                continue

            # Check affordability (money >= cost)
            cost_dollars = affordance_costs_dollars.get(affordance_name, 0)
            if cost_dollars > 0:
                # Convert cost to normalized form: normalized_money = (dollars / 200) + 0.5
                # To check if agent can afford: (normalized - 0.5) * 200 >= cost_dollars
                # Which is: normalized >= (cost_dollars / 200) + 0.5
                cost_normalized = (cost_dollars / 200.0) + 0.5
                can_afford = self.meters[:, 3] >= cost_normalized
                at_affordance = at_affordance & can_afford

                if not at_affordance.any():
                    # No one at this affordance can afford it, skip
                    continue

            # Apply affordance effects (matching Hamlet exactly)
            # NOTE: Money is in range [0, 100] (no debt), so $X = X/100 in normalized [0, 1]
            if affordance_name == 'Bed':
                # Energy restoration tier 1 (affordable)
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] + 0.50, 0.0, 1.0
                )  # Energy +50%
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.02, 0.0, 1.0
                )  # Health +2%
                self.meters[at_affordance, 3] -= 0.025  # Money -$5
            elif affordance_name == 'LuxuryBed':
                # Energy restoration tier 2 (premium rest)
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] + 0.75, 0.0, 1.0
                )  # Energy +75% (50% more than Bed)
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.05, 0.0, 1.0
                )  # Health +5%
                self.meters[at_affordance, 3] -= 0.055  # Money -$11 (2.2x cost of Bed)
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
                # Office work - sustainable income
                self.meters[at_affordance, 3] += 0.1125  # Money +$22.5
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.15, 0.0, 1.0
                )  # Energy -15%
                self.meters[at_affordance, 5] = torch.clamp(
                    self.meters[at_affordance, 5] + 0.02, 0.0, 1.0
                )  # Social +2% (coworker interaction)
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] - 0.03, 0.0, 1.0
                )  # Health -3% (work stress)
            elif affordance_name == 'Labor':
                # Physical labor - higher pay, higher costs
                self.meters[at_affordance, 3] += 0.150  # Money +$30 (33% more than Job)
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.20, 0.0, 1.0
                )  # Energy -20% (exhausting)
                self.meters[at_affordance, 7] = torch.clamp(
                    self.meters[at_affordance, 7] - 0.05, 0.0, 1.0
                )  # Fitness -5% (physical wear and tear)
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] - 0.05, 0.0, 1.0
                )  # Health -5% (injury risk)
                self.meters[at_affordance, 5] = torch.clamp(
                    self.meters[at_affordance, 5] + 0.01, 0.0, 1.0
                )  # Social +1% (minimal - hard physical work)
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
                self.meters[at_affordance, 7] = torch.clamp(
                    self.meters[at_affordance, 7] - 0.03, 0.0, 1.0
                )  # Fitness -3% (unhealthy food)
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
                # Fitness builder (secondary meter - prevents health decline)
                self.meters[at_affordance, 7] = torch.clamp(
                    self.meters[at_affordance, 7] + 0.30, 0.0, 1.0
                )  # Fitness +30%
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.08, 0.0, 1.0
                )  # Energy -8% (workout is tiring)
                self.meters[at_affordance, 3] -= 0.04  # Money -$8 = -8/200
            elif affordance_name == 'Park':
                # Free generalist - fitness, social, mood (no direct health)
                self.meters[at_affordance, 7] = torch.clamp(
                    self.meters[at_affordance, 7] + 0.20, 0.0, 1.0
                )  # Fitness +20% (secondary - prevents health decline)
                self.meters[at_affordance, 5] = torch.clamp(
                    self.meters[at_affordance, 5] + 0.15, 0.0, 1.0
                )  # Social +15% (secondary - prevents mood decline)
                self.meters[at_affordance, 4] = torch.clamp(
                    self.meters[at_affordance, 4] + 0.15, 0.0, 1.0
                )  # Mood +15% (primary)
                self.meters[at_affordance, 0] = torch.clamp(
                    self.meters[at_affordance, 0] - 0.15, 0.0, 1.0
                )  # Energy -15% (time/effort cost)
                # Money: $0 (FREE!)
            elif affordance_name == 'Doctor':
                # Health restoration tier 1 (affordable clinic)
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.25, 0.0, 1.0
                )  # Health +25%
                self.meters[at_affordance, 3] -= 0.04  # Money -$8
            elif affordance_name == 'Hospital':
                # Health restoration tier 2 (expensive emergency care)
                self.meters[at_affordance, 6] = torch.clamp(
                    self.meters[at_affordance, 6] + 0.40, 0.0, 1.0
                )  # Health +40% (intensive care)
                self.meters[at_affordance, 3] -= 0.075  # Money -$15
            elif affordance_name == 'Therapist':
                # Mood restoration tier 2 (professional mental health)
                self.meters[at_affordance, 4] = torch.clamp(
                    self.meters[at_affordance, 4] + 0.40, 0.0, 1.0
                )  # Mood +40% (intensive therapy)
                self.meters[at_affordance, 3] -= 0.075  # Money -$15 = -12/200

    def _deplete_meters(self) -> None:
        """Deplete meters each step."""
        # Base depletion rates (per step)
        depletions = torch.tensor([
            0.005,  # energy: 0.5% per step
            0.003,  # hygiene: 0.3%
            0.004,  # satiation: 0.4%
            0.0,    # money: no passive depletion
            0.001,  # mood: 0.1%
            0.006,  # social: 0.6%
            0.0,    # health: modulated by fitness (see below)
            0.002,  # fitness: 0.2% (slower than energy, faster than health)
        ], device=self.device)

        # Apply base depletions
        self.meters = torch.clamp(
            self.meters - depletions, 0.0, 1.0
        )

        # Fitness-modulated health depletion
        # Low fitness (<30%): 0.003/step (3x baseline - get sick easily)
        # Medium fitness (30-70%): 0.001/step (baseline)
        # High fitness (>70%): 0.0005/step (0.5x baseline - stay healthier)
        fitness = self.meters[:, 7]
        health_depletion = torch.where(
            fitness < 0.3,
            0.003,  # Low fitness: 3x depletion
            torch.where(
                fitness > 0.7,
                0.0005,  # High fitness: 0.5x depletion
                0.001  # Medium fitness: baseline
            )
        )
        self.meters[:, 6] = torch.clamp(
            self.meters[:, 6] - health_depletion, 0.0, 1.0
        )

    def _apply_secondary_to_primary_effects(self) -> None:
        """
        SECONDARY → PRIMARY (Aggressive effects).

        **Satiation is FUNDAMENTAL** (affects BOTH primaries):
        - Low Satiation → Health decline ↑↑↑ (starving → sick → death)
        - Low Satiation → Energy decline ↑↑↑ (hungry → exhausted → death)

        **Specialized secondaries** (each affects one primary):
        - Low Fitness → Health decline ↑↑↑ (unfit → sick → death)
        - Low Mood → Energy decline ↑↑↑ (depressed → exhausted → death)

        This creates asymmetry: FOOD FIRST, then everything else.
        """
        threshold = 0.3  # below this, aggressive penalties apply

        # SATIATION → BOTH PRIMARIES (fundamental need!)
        satiation = self.meters[:, 2]
        low_satiation = satiation < threshold
        if low_satiation.any():
            deficit = (threshold - satiation[low_satiation]) / threshold

            # Health damage (starving → sick)
            health_penalty = 0.004 * deficit  # 0.4% at threshold, up to ~0.8% at 0
            self.meters[low_satiation, 6] = torch.clamp(
                self.meters[low_satiation, 6] - health_penalty, 0.0, 1.0
            )

            # Energy damage (hungry → exhausted)
            energy_penalty = 0.005 * deficit  # 0.5% at threshold, up to ~1.0% at 0
            self.meters[low_satiation, 0] = torch.clamp(
                self.meters[low_satiation, 0] - energy_penalty, 0.0, 1.0
            )

        # FITNESS → HEALTH (specialized)
        # (Already implemented in _deplete_meters via fitness-modulated health depletion)
        # Low fitness creates 3x health depletion multiplier

        # MOOD → ENERGY (specialized)
        mood = self.meters[:, 4]
        low_mood = mood < threshold
        if low_mood.any():
            deficit = (threshold - mood[low_mood]) / threshold
            energy_penalty = 0.005 * deficit  # 0.5% at threshold, up to ~1.0% at 0
            self.meters[low_mood, 0] = torch.clamp(
                self.meters[low_mood, 0] - energy_penalty, 0.0, 1.0
            )

    def _apply_tertiary_to_secondary_effects(self) -> None:
        """
        TERTIARY → SECONDARY (Aggressive effects).

        - Low Hygiene → Satiation/Fitness/Mood decline ↑↑
        - Low Social → Mood decline ↑↑
        """
        threshold = 0.3

        # Low hygiene → secondary meters
        hygiene = self.meters[:, 1]
        low_hygiene = hygiene < threshold
        if low_hygiene.any():
            deficit = (threshold - hygiene[low_hygiene]) / threshold

            # Satiation penalty (being dirty → loss of appetite)
            satiation_penalty = 0.002 * deficit
            self.meters[low_hygiene, 2] = torch.clamp(
                self.meters[low_hygiene, 2] - satiation_penalty, 0.0, 1.0
            )

            # Fitness penalty (being dirty → harder to exercise)
            fitness_penalty = 0.002 * deficit
            self.meters[low_hygiene, 7] = torch.clamp(
                self.meters[low_hygiene, 7] - fitness_penalty, 0.0, 1.0
            )

            # Mood penalty (being dirty → feel bad)
            mood_penalty = 0.003 * deficit
            self.meters[low_hygiene, 4] = torch.clamp(
                self.meters[low_hygiene, 4] - mood_penalty, 0.0, 1.0
            )

        # Low social → mood
        social = self.meters[:, 5]
        low_social = social < threshold
        if low_social.any():
            deficit = (threshold - social[low_social]) / threshold
            mood_penalty = 0.004 * deficit  # Stronger than hygiene
            self.meters[low_social, 4] = torch.clamp(
                self.meters[low_social, 4] - mood_penalty, 0.0, 1.0
            )

    def _apply_tertiary_to_primary_effects(self) -> None:
        """
        TERTIARY → PRIMARY (Weak direct effects).

        - Low Hygiene → Health/Energy decline ↑ (weak)
        - Low Social → Energy decline ↑ (weak)
        """
        threshold = 0.3

        # Low hygiene → health (weak)
        hygiene = self.meters[:, 1]
        low_hygiene = hygiene < threshold
        if low_hygiene.any():
            deficit = (threshold - hygiene[low_hygiene]) / threshold

            health_penalty = 0.0005 * deficit  # Weak effect
            self.meters[low_hygiene, 6] = torch.clamp(
                self.meters[low_hygiene, 6] - health_penalty, 0.0, 1.0
            )

            energy_penalty = 0.0005 * deficit  # Weak effect
            self.meters[low_hygiene, 0] = torch.clamp(
                self.meters[low_hygiene, 0] - energy_penalty, 0.0, 1.0
            )

        # Low social → energy (weak)
        social = self.meters[:, 5]
        low_social = social < threshold
        if low_social.any():
            deficit = (threshold - social[low_social]) / threshold
            energy_penalty = 0.0008 * deficit  # Weak effect
            self.meters[low_social, 0] = torch.clamp(
                self.meters[low_social, 0] - energy_penalty, 0.0, 1.0
            )

    def _check_dones(self) -> None:
        """Check terminal conditions.

        Coupled cascade architecture:

        **PRIMARY (Death Conditions):**
        - Health: Are you alive?
        - Energy: Can you move?

        **SECONDARY (Aggressive → Primary):**
        - Satiation ──strong──> Health AND Energy (FUNDAMENTAL - affects both!)
        - Fitness ──strong──> Health (unfit → sick → death)
        - Mood ──strong──> Energy (depressed → exhausted → death)

        **TERTIARY (Quality of Life):**
        - Hygiene ──strong──> Secondary + weak──> Primary
        - Social ──strong──> Secondary + weak──> Primary

        **RESOURCE:**
        - Money (Enables affordances)

        **Key Insight:** Satiation is THE foundational need - hungry makes you
        BOTH sick AND exhausted. Food must be prioritized above all else.
        """
        # Death if either PRIMARY meter hits 0
        health_values = self.meters[:, 6]  # health
        energy_values = self.meters[:, 0]  # energy

        self.dones = ((health_values <= 0.0) | (energy_values <= 0.0))

    def _calculate_shaped_rewards(self) -> torch.Tensor:
        """
        SIMPLE SURVIVAL REWARD: Directly reward staying alive.

        Problem with old complex rewards: Longer survival → more accumulated penalties → negative rewards.
        Solution: +1.0 per step survived, -100 for dying.

        Returns:
            rewards: [num_agents]
        """
        # Base reward: +1.0 for surviving this step
        rewards = torch.ones(self.num_agents, device=self.device)

        # Death penalty: -100.0 for dying
        rewards = torch.where(self.dones, -100.0, rewards)

        return rewards

    def _calculate_shaped_rewards_COMPLEX_DISABLED(self) -> torch.Tensor:
        """
        DISABLED: Complex meter-based rewards caused accumulating penalties.

        Problem: Longer survival → more penalties → negative rewards
        Example: 200 steps with low meters = -2000 reward (backwards!)

        This is kept for reference but not used.
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

        # Fitness meter rewards (support meter - prevents health decline)
        fitness_values = self.meters[:, 7]

        # High fitness (>0.7): +0.15 (staying healthy)
        rewards += torch.where(fitness_values > 0.7, 0.15, 0.0)

        # Moderate fitness (0.4-0.7): +0.05
        rewards += torch.where(
            (fitness_values > 0.4) & (fitness_values <= 0.7), 0.05, 0.0
        )

        # Low fitness (0.2-0.4): -0.2 (getting sick more easily)
        rewards += torch.where(
            (fitness_values > 0.2) & (fitness_values <= 0.4), -0.2, 0.0
        )

        # Critical fitness (<=0.2): -0.8 (health declining rapidly)
        rewards += torch.where(fitness_values <= 0.2, -0.8, 0.0)

        # Tier 2: Proximity shaping (DISABLED for Level 2 - agent must interact to survive)
        # Previously caused reward hacking (standing near affordances without interacting)
        # proximity_rewards = self._calculate_proximity_rewards()
        # rewards += proximity_rewards

        # Urgency penalty: DISABLED (also a form of proximity shaping)
        # satiation_values = self.meters[:, 2]
        # low_satiation = satiation_values < 0.4
        # if low_satiation.any():
        #     # Calculate distance to food affordances for agents with low satiation
        #     for agent_idx in range(self.num_agents):
        #         if not low_satiation[agent_idx]:
        #             continue
        #
        #         # Find nearest food affordance (HomeMeal or FastFood)
        #         food_affordances = ['HomeMeal', 'FastFood']
        #         min_dist = float('inf')
        #         for food_name in food_affordances:
        #             if food_name in self.affordances:
        #                 food_pos = self.affordances[food_name]
        #                 dist = torch.abs(self.positions[agent_idx] - food_pos).sum().item()
        #                 min_dist = min(min_dist, dist)
        #
        #         if min_dist < float('inf'):
        #             urgency = min(1.0, (0.4 - satiation_values[agent_idx].item()) / 0.4)
        #             max_dist = self.grid_size * 2
        #             rewards[agent_idx] -= urgency * (min_dist / max_dist) * 2.0

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

        # Meter index to affordance name mapping (proximity guidance)
        meter_to_affordance = {
            0: 'Bed',         # energy
            1: 'Shower',      # hygiene
            2: 'HomeMeal',    # satiation
            3: 'Job',         # money
            4: 'Recreation',  # mood (tier 1 - cheaper)
            5: 'Bar',         # social
            6: 'Doctor',      # health (tier 1 - cheaper)
            7: 'Gym',         # fitness
        }

        # For each agent, find most critical meter and reward proximity
        for agent_idx in range(self.num_agents):
            # Find most critical meter (highest urgency = lowest value)
            meter_vals = self.meters[agent_idx]
            urgency = 1.0 - meter_vals  # Higher when meter is lower

            # Only consider meters below threshold
            # [energy, hygiene, satiation, money, mood, social, health, fitness]
            threshold = torch.tensor([0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5], device=self.device)
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
