"""Targeted unit tests for MeterDynamics."""

from __future__ import annotations

import torch

from townlet.environment.meter_dynamics import MeterDynamics


def make_meter_dynamics(device: torch.device = torch.device("cpu")) -> MeterDynamics:
    base = torch.tensor([0.1, 0.0], dtype=torch.float32)
    cascade_data = {
        "primary_to_pivotal": [
            {
                "source_idx": 0,  # energy
                "target_idx": 1,  # health
                "threshold": 0.4,
                "strength": 0.2,
            }
        ]
    }
    modulation_data = [
        {
            "source_idx": 0,
            "target_idx": 1,
            "base_multiplier": 1.0,
            "range": 1.0,
            "baseline_depletion": 0.05,
        }
    ]
    terminal_conditions = [
        {"meter_idx": 1, "operator": "<=", "value": 0.1},
    ]

    meter_lookup = {"energy": 0, "health": 1}
    return MeterDynamics(
        base_depletions=base,
        cascade_data=cascade_data,
        modulation_data=modulation_data,
        terminal_conditions=terminal_conditions,
        meter_name_to_index=meter_lookup,
        device=device,
    )


class TestMeterDynamics:
    def test_deplete_meters_applies_base_and_modulations(self):
        md = make_meter_dynamics()
        meters = torch.tensor([[0.5, 1.0]], dtype=torch.float32)

        updated = md.deplete_meters(meters.clone())

        # Energy depleted by base amount, clamped to [0, 1]
        assert torch.allclose(updated[:, 0], torch.tensor([0.4]))

        # Modulation sees energy after depletion (0.4 → deficit 0.6 → multiplier 1.6)
        expected_health = 1.0 - (0.05 * 1.6)
        assert torch.allclose(updated[:, 1], torch.tensor([expected_health]))

    def test_cascade_penalises_target_below_threshold(self):
        md = make_meter_dynamics()
        meters = torch.tensor([[0.2, 0.5]], dtype=torch.float32)

        cascaded = md.apply_secondary_to_primary_effects(meters.clone())
        # threshold 0.4 → deficit (0.4-0.2)/0.4 = 0.5, penalty strength 0.5 * 0.2 = 0.1
        assert torch.allclose(cascaded[:, 1], torch.tensor([0.4]))

    def test_terminal_condition_detects_death(self):
        md = make_meter_dynamics()
        meters = torch.tensor([[0.3, 0.05], [0.3, 0.5]], dtype=torch.float32)
        dones = torch.zeros(2, dtype=torch.bool)

        mask = md.check_terminal_conditions(meters, dones)
        assert torch.equal(mask, torch.tensor([True, False]))
