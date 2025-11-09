"""Tensor-driven meter dynamics used by the runtime environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


class MeterDynamics:
    """Apply depletion, modulations, cascades, and terminal checks from tensors."""

    def __init__(
        self,
        *,
        base_depletions: torch.Tensor,
        cascade_data: Mapping[str, Sequence[Mapping[str, Any]]],
        modulation_data: Sequence[Mapping[str, Any]],
        terminal_conditions: Sequence[Mapping[str, Any]],
        meter_name_to_index: Mapping[str, int],
        device: torch.device,
    ) -> None:
        """Initialize meter dynamics with compiler-provided tensors."""

        self.device = device
        self.base_depletions = base_depletions.to(device=device, dtype=torch.float32).clone()
        self.meter_name_to_index = dict(meter_name_to_index)

        self._cascade_tables = {
            category: [
                {
                    "source_idx": int(entry["source_idx"]),
                    "target_idx": int(entry["target_idx"]),
                    "threshold": float(entry["threshold"]),
                    "strength": float(entry["strength"]),
                }
                for entry in entries
            ]
            for category, entries in cascade_data.items()
        }

        self._modulations = [
            {
                "source_idx": int(entry["source_idx"]),
                "target_idx": int(entry["target_idx"]),
                "base_multiplier": float(entry["base_multiplier"]),
                "range": float(entry["range"]),
                "baseline_depletion": float(entry["baseline_depletion"]),
            }
            for entry in modulation_data
        ]

        self._terminal_conditions = [
            {
                "meter_idx": int(entry["meter_idx"]),
                "operator": entry["operator"],
                "value": float(entry["value"]),
            }
            for entry in terminal_conditions
        ]

    def deplete_meters(self, meters: torch.Tensor, depletion_multiplier: float = 1.0) -> torch.Tensor:
        """Apply base depletion and modulations using precomputed tensors."""

        scaled_depletions = self.base_depletions * depletion_multiplier
        meters = torch.clamp(meters - scaled_depletions, 0.0, 1.0)

        for modulation in self._modulations:
            source_values = meters[:, modulation["source_idx"]]
            target_idx = modulation["target_idx"]
            penalty_strength = 1.0 - source_values
            multiplier = modulation["base_multiplier"] + (modulation["range"] * penalty_strength)
            depletion = modulation["baseline_depletion"] * multiplier
            meters[:, target_idx] = torch.clamp(meters[:, target_idx] - depletion, 0.0, 1.0)

        return meters

    def apply_secondary_to_primary_effects(self, meters: torch.Tensor) -> torch.Tensor:
        """Apply primary_to_pivotal cascades (secondary → primary)."""

        return self._apply_cascades(meters, ["primary_to_pivotal"])

    def apply_tertiary_to_secondary_effects(self, meters: torch.Tensor) -> torch.Tensor:
        """Apply secondary_to_primary cascades (tertiary → secondary)."""

        return self._apply_cascades(meters, ["secondary_to_primary"])

    def apply_tertiary_to_primary_effects(self, meters: torch.Tensor) -> torch.Tensor:
        """Apply secondary_to_pivotal_weak cascades (tertiary → primary)."""

        return self._apply_cascades(meters, ["secondary_to_pivotal_weak"])

    def check_terminal_conditions(self, meters: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Evaluate terminal conditions using compiler-provided thresholds."""

        terminal_mask = torch.zeros_like(dones, dtype=torch.bool)

        for condition in self._terminal_conditions:
            meter_values = meters[:, condition["meter_idx"]]
            threshold = condition["value"]
            operator = condition["operator"]

            if operator == "<=":
                current = meter_values <= threshold
            elif operator == ">=":
                current = meter_values >= threshold
            elif operator == "<":
                current = meter_values < threshold
            elif operator == ">":
                current = meter_values > threshold
            elif operator == "==":
                current = torch.isclose(meter_values, torch.tensor(threshold, device=meter_values.device))
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown terminal condition operator: {operator}")

            terminal_mask |= current

        return terminal_mask

    def get_base_depletion(self, meter_name: str) -> float:
        """Expose base depletion for configuration tests."""

        idx = self.meter_name_to_index.get(meter_name)
        if idx is None:
            raise KeyError(f"Meter '{meter_name}' not found in lookup.")
        return float(self.base_depletions[idx].item())

    def _apply_cascades(self, meters: torch.Tensor, categories: Sequence[str]) -> torch.Tensor:
        for category in categories:
            cascades = self._cascade_tables.get(category)
            if not cascades:
                continue

            for cascade in cascades:
                source_values = meters[:, cascade["source_idx"]]
                low_mask = source_values < cascade["threshold"]
                if not low_mask.any():
                    continue

                deficit = (cascade["threshold"] - source_values[low_mask]) / cascade["threshold"]
                penalty = cascade["strength"] * deficit
                target_idx = cascade["target_idx"]
                meters[low_mask, target_idx] = torch.clamp(
                    meters[low_mask, target_idx] - penalty,
                    0.0,
                    1.0,
                )

        return meters
