"""
Meter Dynamics Module

Encapsulates meter depletion and cascade effects for the Hamlet environment.
Implements the coupled cascade architecture where meters affect each other.

All cascade logic is config-driven via CascadeEngine and YAML files.
"""

from pathlib import Path
from typing import Optional

import torch

from townlet.environment.cascade_config import (
    load_default_config,
    load_environment_config,
)
from townlet.environment.cascade_engine import CascadeEngine


class MeterDynamics:
    """
    Manages meter depletion and cascade effects using config-driven CascadeEngine.

    Architecture:
    - PRIMARY (Death Conditions): Health, Energy
    - SECONDARY (Aggressive → Primary): Satiation, Fitness, Mood
    - TERTIARY (Quality of Life): Hygiene, Social
    - RESOURCE: Money

    Key Insight: Satiation is THE foundational need - affects BOTH primaries.

    All cascade physics are defined in configs/bars.yaml and configs/cascades.yaml.
    Students can experiment with different cascade strengths by editing YAML files.
    """

    def __init__(
        self,
        num_agents: int,
        device: torch.device,
        cascade_config_dir: Optional[Path] = None,
    ):
        """
        Initialize meter dynamics with config-driven cascade system.

        Args:
            num_agents: Number of agents in the environment
            device: torch device for tensor operations
            cascade_config_dir: Directory containing bars.yaml and cascades.yaml
                              (defaults to project configs/ directory)
        """
        self.num_agents = num_agents
        self.device = device

        # Load cascade configuration
        if cascade_config_dir is None:
            # Load from project configs/ directory
            env_config = load_default_config()
        else:
            env_config = load_environment_config(cascade_config_dir)

        # Initialize config-driven CascadeEngine
        self.cascade_engine = CascadeEngine(env_config, device)

    def deplete_meters(self, meters: torch.Tensor) -> torch.Tensor:
        """
        Apply base depletion rates and modulations to all meters.

        Args:
            meters: [num_agents, 8] current meter values

        Returns:
            meters: [num_agents, 8] meters after depletion and modulations
        """
        meters = self.cascade_engine.apply_base_depletions(meters)
        meters = self.cascade_engine.apply_modulations(meters)
        return meters

    def apply_secondary_to_primary_effects(self, meters: torch.Tensor) -> torch.Tensor:
        """
        SECONDARY → PRIMARY (Aggressive effects).

        **Satiation is FUNDAMENTAL** (affects BOTH primaries):
        - Low Satiation → Health decline ↑↑↑ (starving → sick → death)
        - Low Satiation → Energy decline ↑↑↑ (hungry → exhausted → death)

        **Specialized secondaries** (each affects one primary):
        - Low Fitness → Health decline ↑↑↑ (unfit → sick → death)
        - Low Mood → Energy decline ↑↑↑ (depressed → exhausted → death)

        This creates asymmetry: FOOD FIRST, then everything else.

        Args:
            meters: [num_agents, 8] current meter values

        Returns:
            meters: [num_agents, 8] meters after cascade effects
        """
        return self.cascade_engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

    def apply_tertiary_to_secondary_effects(self, meters: torch.Tensor) -> torch.Tensor:
        """
        TERTIARY → SECONDARY (Aggressive effects).

        - Low Hygiene → Satiation/Fitness/Mood decline ↑↑
        - Low Social → Mood decline ↑↑

        Args:
            meters: [num_agents, 8] current meter values

        Returns:
            meters: [num_agents, 8] meters after cascade effects
        """
        return self.cascade_engine.apply_threshold_cascades(meters, ["secondary_to_primary"])

    def apply_tertiary_to_primary_effects(self, meters: torch.Tensor) -> torch.Tensor:
        """
        TERTIARY → PRIMARY (Weak direct effects).

        - Low Hygiene → Health/Energy decline ↑ (weak)
        - Low Social → Energy decline ↑ (weak)

        Args:
            meters: [num_agents, 8] current meter values

        Returns:
            meters: [num_agents, 8] meters after cascade effects
        """
        return self.cascade_engine.apply_threshold_cascades(meters, ["secondary_to_pivotal_weak"])

    def check_terminal_conditions(self, meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Check terminal conditions.

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

        Args:
            meters: [num_agents, 8] current meter values
            dones: [num_agents] current done flags (unused, kept for compatibility)

        Returns:
            dones: [num_agents] updated done flags
        """
        return self.cascade_engine.check_terminal_conditions(meters, dones)
