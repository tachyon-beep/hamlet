"""
Cascade Engine Module

GPU-accelerated cascade engine that applies meter dynamics based on YAML configuration.
Replaces hardcoded cascade logic with config-driven system.

Key Features:
- Reads validated configuration from bars.yaml and cascades.yaml
- Applies base depletions, modulations, and threshold cascades
- GPU-optimized with PyTorch tensors
- Zero behavioral change from hardcoded implementation
"""

from __future__ import annotations

import torch

from townlet.environment.cascade_config import EnvironmentConfig

CascadeEntry = dict[str, float | int | str]
CascadeData = dict[str, list[CascadeEntry]]


class CascadeEngine:
    """
    Config-driven cascade engine for meter dynamics.

    Applies meter cascades based on YAML configuration instead of hardcoded logic.
    All operations are GPU-accelerated using PyTorch tensors.

    Architecture:
    1. Base depletions (from bars.yaml)
    2. Modulations (fitness → health multiplier)
    3. Threshold cascades (gradient penalties by category)
    4. Terminal conditions (death checks)
    """

    def __init__(self, config: EnvironmentConfig, device: torch.device):
        """
        Initialize cascade engine with configuration.

        Args:
            config: Validated environment configuration
            device: PyTorch device for tensor operations
        """
        self.config = config
        self.device = device

        # Pre-build lookup maps for performance
        self._bar_name_to_idx = {bar.name: bar.index for bar in config.bars.bars}
        self._bar_idx_to_name = {bar.index: bar.name for bar in config.bars.bars}

        # Pre-compute base depletion tensor [meter_count]
        self._base_depletions = self._build_base_depletion_tensor()

        # Pre-build cascade tensors for efficient batch application
        self._cascade_data = self._build_cascade_data()

        # Pre-build modulation data
        self._modulation_data = self._build_modulation_data()

        # Pre-build terminal condition data
        self._terminal_data = self._build_terminal_data()

    def _build_base_depletion_tensor(self) -> torch.Tensor:
        """
        Build tensor of base depletion rates [meter_count].

        Returns:
            Tensor of shape [meter_count] with depletion rates by index
        """
        meter_count = self.config.bars.meter_count
        depletions = torch.zeros(meter_count, device=self.device)

        for bar in self.config.bars.bars:
            depletions[bar.index] = bar.base_depletion

        return depletions

    def _build_modulation_data(self) -> list[dict]:
        """
        Build modulation data for efficient application.

        Returns:
            List of dicts with pre-computed modulation parameters
        """
        modulation_data = []

        for mod in self.config.cascades.modulations:
            modulation_data.append(
                {
                    "name": mod.name,
                    "source_idx": self._bar_name_to_idx[mod.source],
                    "target_idx": self._bar_name_to_idx[mod.target],
                    "base_multiplier": mod.base_multiplier,
                    "range": mod.range,
                    "baseline_depletion": mod.baseline_depletion,
                }
            )

        return modulation_data

    def _build_cascade_data(self) -> dict[str, list[dict]]:
        """
        Build cascade data organized by category for efficient batch application.

        Returns:
            Dict mapping category -> list of cascade dicts
        """
        cascade_data: CascadeData = {}

        for cascade in self.config.cascades.cascades:
            category = cascade.category

            if category not in cascade_data:
                cascade_data[category] = []

            cascade_data[category].append(
                {
                    "name": cascade.name,
                    "source_idx": cascade.source_index,
                    "target_idx": cascade.target_index,
                    "threshold": cascade.threshold,
                    "strength": cascade.strength,
                }
            )

        return cascade_data

    def _build_terminal_data(self) -> list[dict]:
        """
        Build terminal condition data.

        Returns:
            List of dicts with meter indices and thresholds
        """
        terminal_data = []

        for tc in self.config.bars.terminal_conditions:
            meter_idx = self._bar_name_to_idx[tc.meter]
            terminal_data.append({"meter_idx": meter_idx, "operator": tc.operator, "value": tc.value})

        return terminal_data

    def apply_base_depletions(self, meters: torch.Tensor, depletion_multiplier: float = 1.0) -> torch.Tensor:
        """
        Apply base depletion rates to all meters with curriculum difficulty scaling.

        Args:
            meters: [num_agents, meter_count] current meter values
            depletion_multiplier: Curriculum difficulty multiplier (0.2 = 20% difficulty)

        Returns:
            meters: [num_agents, meter_count] meters after base depletions
        """
        # Apply curriculum difficulty scaling to base depletions
        scaled_depletions = self._base_depletions * depletion_multiplier
        # Subtract scaled depletions and clamp to [0, 1]
        meters = torch.clamp(meters - scaled_depletions, 0.0, 1.0)
        return meters

    def apply_modulations(self, meters: torch.Tensor) -> torch.Tensor:
        """
        Apply modulation effects (e.g., fitness modulates health depletion).

        Args:
            meters: [num_agents, meter_count] current meter values

        Returns:
            meters: [num_agents, meter_count] meters after modulations
        """
        for mod in self._modulation_data:
            # Get source and target meter values
            source_values = meters[:, mod["source_idx"]]
            target_values = meters[:, mod["target_idx"]]

            # Calculate modulation multiplier
            # multiplier = base + (range * (1.0 - source))
            # Example: fitness=1.0 → 0.5x, fitness=0.0 → 3.0x
            penalty_strength = 1.0 - source_values
            multiplier = mod["base_multiplier"] + (mod["range"] * penalty_strength)

            # Apply modulated depletion
            depletion = mod["baseline_depletion"] * multiplier
            target_values = torch.clamp(target_values - depletion, 0.0, 1.0)

            # Update meters
            meters[:, mod["target_idx"]] = target_values

        return meters

    def apply_threshold_cascades(self, meters: torch.Tensor, categories: list[str]) -> torch.Tensor:
        """
        Apply threshold-based cascades for specified categories.

        Gradient penalty approach:
        - When source < threshold, calculate deficit = (threshold - source) / threshold
        - Apply penalty = strength * deficit to target

        Args:
            meters: [num_agents, meter_count] current meter values
            categories: List of cascade categories to apply (e.g., ['primary_to_pivotal'])

        Returns:
            meters: [num_agents, meter_count] meters after cascades
        """
        for category in categories:
            if category not in self._cascade_data:
                continue

            for cascade in self._cascade_data[category]:
                # Get source and target values
                source_values = meters[:, cascade["source_idx"]]
                target_values = meters[:, cascade["target_idx"]]

                # Identify agents below threshold
                threshold = cascade["threshold"]
                low_mask = source_values < threshold

                if low_mask.any():
                    # Calculate normalized deficit [0, 1]
                    deficit = (threshold - source_values[low_mask]) / threshold

                    # Calculate penalty = strength * deficit
                    penalty = cascade["strength"] * deficit

                    # Apply penalty to target
                    target_values[low_mask] = torch.clamp(target_values[low_mask] - penalty, 0.0, 1.0)

                    # Update meters
                    meters[:, cascade["target_idx"]] = target_values

        return meters

    def check_terminal_conditions(self, meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Check terminal conditions (death).

        Args:
            meters: [num_agents, meter_count] current meter values
            dones: [num_agents] current done flags

        Returns:
            dones: [num_agents] updated done flags
        """
        # Start with False for all agents
        terminal_mask = torch.zeros_like(dones, dtype=torch.bool)

        for tc in self._terminal_data:
            meter_values = meters[:, tc["meter_idx"]]
            operator = tc["operator"]
            threshold = tc["value"]

            # Apply operator
            if operator == "<=":
                condition = meter_values <= threshold
            elif operator == ">=":
                condition = meter_values >= threshold
            elif operator == "<":
                condition = meter_values < threshold
            elif operator == ">":
                condition = meter_values > threshold
            elif operator == "==":
                condition = torch.isclose(meter_values, torch.tensor(threshold, device=self.device))
            else:
                raise ValueError(f"Unknown operator: {operator}")

            # OR with existing terminal mask (death if ANY condition met)
            terminal_mask = terminal_mask | condition

        return terminal_mask

    def apply_full_cascade(self, meters: torch.Tensor) -> torch.Tensor:
        """
        Apply complete cascade sequence in execution order.

        This is the main entry point that applies:
        1. Base depletions
        2. Modulations (fitness → health)
        3. Primary → Pivotal cascades
        4. Secondary → Primary cascades
        5. Secondary → Pivotal (weak) cascades

        Args:
            meters: [num_agents, meter_count] current meter values

        Returns:
            meters: [num_agents, meter_count] meters after all cascades
        """
        # Get execution order from config
        execution_order = self.config.cascades.execution_order

        for stage in execution_order:
            if stage == "modulations":
                meters = self.apply_modulations(meters)
            else:
                # stage is a cascade category
                meters = self.apply_threshold_cascades(meters, [stage])

        return meters

    def get_bar_name(self, index: int) -> str:
        """Get bar name from index."""
        return self._bar_idx_to_name[index]

    def get_bar_index(self, name: str) -> int:
        """Get bar index from name."""
        return self._bar_name_to_idx[name]

    def get_base_depletion(self, name: str) -> float:
        """Return base depletion rate for the specified meter."""
        idx = self.get_bar_index(name)
        return float(self._base_depletions[idx].item())

    def get_initial_meter_values(self) -> torch.Tensor:
        """
        Get initial meter values from bars.yaml configuration.

        Returns:
            Tensor of shape [meter_count] with initial values for all meters by index
        """
        meter_count = self.config.bars.meter_count
        initial_values = torch.zeros(meter_count, device=self.device)

        for bar in self.config.bars.bars:
            initial_values[bar.index] = bar.initial

        return initial_values
