"""
Meter Dynamics Module

Encapsulates meter depletion and cascade effects for the Hamlet environment.
Implements the coupled cascade architecture where meters affect each other.

Now supports both:
- Config-driven cascades (via CascadeEngine + YAML)
- Legacy hardcoded cascades (for backward compatibility)
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
    Manages meter depletion and cascade effects.

    Architecture:
    - PRIMARY (Death Conditions): Health, Energy
    - SECONDARY (Aggressive → Primary): Satiation, Fitness, Mood
    - TERTIARY (Quality of Life): Hygiene, Social
    - RESOURCE: Money

    Key Insight: Satiation is THE foundational need - affects BOTH primaries.
    """

    def __init__(
        self,
        num_agents: int,
        device: torch.device,
        use_cascade_engine: bool = False,
        cascade_config_dir: Optional[Path] = None,
    ):
        """
        Initialize meter dynamics.

        Args:
            num_agents: Number of agents in the environment
            device: torch device for tensor operations
            use_cascade_engine: If True, use config-driven CascadeEngine
            cascade_config_dir: Directory containing bars.yaml and cascades.yaml
                              (defaults to project configs/ directory)
        """
        self.num_agents = num_agents
        self.device = device
        self.use_cascade_engine = use_cascade_engine

        # Initialize CascadeEngine if requested
        if use_cascade_engine:
            if cascade_config_dir is None:
                # Load from project configs/ directory
                env_config = load_default_config()
            else:
                env_config = load_environment_config(cascade_config_dir)

            self.cascade_engine = CascadeEngine(env_config, device)
        else:
            self.cascade_engine = None

    def deplete_meters(self, meters: torch.Tensor) -> torch.Tensor:
        """
        Apply base depletion rates to all meters.

        Args:
            meters: [num_agents, 8] current meter values

        Returns:
            meters: [num_agents, 8] meters after depletion
        """
        if self.use_cascade_engine:
            # Use config-driven CascadeEngine
            meters = self.cascade_engine.apply_base_depletions(meters)
            meters = self.cascade_engine.apply_modulations(meters)
            return meters

        # Legacy hardcoded implementation
        # Base depletion rates (per step)
        depletions = torch.tensor(
            [
                0.005,  # energy: 0.5% per step
                0.003,  # hygiene: 0.3%
                0.004,  # satiation: 0.4%
                0.0,  # money: no passive depletion
                0.001,  # mood: 0.1%
                0.006,  # social: 0.6%
                0.0,  # health: modulated by fitness (see below)
                0.002,  # fitness: 0.2% (slower than energy, faster than health)
            ],
            device=self.device,
        )

        # Apply base depletions
        meters = torch.clamp(meters - depletions, 0.0, 1.0)

        # Fitness-modulated health depletion (GRADIENT approach for consistency)
        # All cascade effects use gradient calculation based on current level
        # fitness=100%: multiplier=0.5x (0.0005/step - very healthy)
        # fitness=50%: multiplier=1.75x (0.00175/step - moderate decline)
        # fitness=0%: multiplier=3.0x (0.003/step - get sick easily)
        baseline_health_depletion = 0.001
        fitness = meters[:, 7]

        # Calculate multiplier: ranges from 0.5 (100% fitness) to 3.0 (0% fitness)
        fitness_penalty_strength = 1.0 - fitness  # 0.0 at 100%, 1.0 at 0%
        multiplier = 0.5 + (2.5 * fitness_penalty_strength)  # Linear gradient

        health_depletion = baseline_health_depletion * multiplier
        meters[:, 6] = torch.clamp(meters[:, 6] - health_depletion, 0.0, 1.0)

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
        if self.use_cascade_engine:
            # Use config-driven CascadeEngine
            return self.cascade_engine.apply_threshold_cascades(meters, ["primary_to_pivotal"])

        # Legacy hardcoded implementation
        threshold = 0.3  # below this, aggressive penalties apply

        # SATIATION → BOTH PRIMARIES (fundamental need!)
        satiation = meters[:, 2]
        low_satiation = satiation < threshold
        if low_satiation.any():
            deficit = (threshold - satiation[low_satiation]) / threshold

            # Health damage (starving → sick)
            health_penalty = 0.004 * deficit  # 0.4% at threshold, up to ~0.8% at 0
            meters[low_satiation, 6] = torch.clamp(
                meters[low_satiation, 6] - health_penalty, 0.0, 1.0
            )

            # Energy damage (hungry → exhausted)
            energy_penalty = 0.005 * deficit  # 0.5% at threshold, up to ~1.0% at 0
            meters[low_satiation, 0] = torch.clamp(
                meters[low_satiation, 0] - energy_penalty, 0.0, 1.0
            )

        # FITNESS → HEALTH (specialized)
        # (Already implemented in deplete_meters via fitness-modulated health depletion)
        # Low fitness creates 3x health depletion multiplier

        # MOOD → ENERGY (specialized)
        mood = meters[:, 4]
        low_mood = mood < threshold
        if low_mood.any():
            deficit = (threshold - mood[low_mood]) / threshold
            energy_penalty = 0.005 * deficit  # 0.5% at threshold, up to ~1.0% at 0
            meters[low_mood, 0] = torch.clamp(meters[low_mood, 0] - energy_penalty, 0.0, 1.0)

        return meters

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
        if self.use_cascade_engine:
            # Use config-driven CascadeEngine
            return self.cascade_engine.apply_threshold_cascades(meters, ["secondary_to_primary"])

        # Legacy hardcoded implementation
        threshold = 0.3

        # Low hygiene → secondary meters
        hygiene = meters[:, 1]
        low_hygiene = hygiene < threshold
        if low_hygiene.any():
            deficit = (threshold - hygiene[low_hygiene]) / threshold

            # Satiation penalty (being dirty → loss of appetite)
            satiation_penalty = 0.002 * deficit
            meters[low_hygiene, 2] = torch.clamp(
                meters[low_hygiene, 2] - satiation_penalty, 0.0, 1.0
            )

            # Fitness penalty (being dirty → harder to exercise)
            fitness_penalty = 0.002 * deficit
            meters[low_hygiene, 7] = torch.clamp(meters[low_hygiene, 7] - fitness_penalty, 0.0, 1.0)

            # Mood penalty (being dirty → feel bad)
            mood_penalty = 0.003 * deficit
            meters[low_hygiene, 4] = torch.clamp(meters[low_hygiene, 4] - mood_penalty, 0.0, 1.0)

        # Low social → mood
        social = meters[:, 5]
        low_social = social < threshold
        if low_social.any():
            deficit = (threshold - social[low_social]) / threshold
            mood_penalty = 0.004 * deficit  # Stronger than hygiene
            meters[low_social, 4] = torch.clamp(meters[low_social, 4] - mood_penalty, 0.0, 1.0)

        return meters

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
        if self.use_cascade_engine:
            # Use config-driven CascadeEngine
            return self.cascade_engine.apply_threshold_cascades(
                meters, ["secondary_to_pivotal_weak"]
            )

        # Legacy hardcoded implementation
        threshold = 0.3

        # Low hygiene → health (weak)
        hygiene = meters[:, 1]
        low_hygiene = hygiene < threshold
        if low_hygiene.any():
            deficit = (threshold - hygiene[low_hygiene]) / threshold

            health_penalty = 0.0005 * deficit  # Weak effect
            meters[low_hygiene, 6] = torch.clamp(meters[low_hygiene, 6] - health_penalty, 0.0, 1.0)

            energy_penalty = 0.0005 * deficit  # Weak effect
            meters[low_hygiene, 0] = torch.clamp(meters[low_hygiene, 0] - energy_penalty, 0.0, 1.0)

        # Low social → energy (weak)
        social = meters[:, 5]
        low_social = social < threshold
        if low_social.any():
            deficit = (threshold - social[low_social]) / threshold
            energy_penalty = 0.0008 * deficit  # Weak effect
            meters[low_social, 0] = torch.clamp(meters[low_social, 0] - energy_penalty, 0.0, 1.0)

        return meters

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
            dones: [num_agents] current done flags

        Returns:
            dones: [num_agents] updated done flags
        """
        # Death if either PRIMARY meter hits 0
        health_values = meters[:, 6]  # health
        energy_values = meters[:, 0]  # energy

        dones = (health_values <= 0.0) | (energy_values <= 0.0)

        return dones
