"""
Tests for the AgentRuntimeRegistry, ensuring tensor-backed state with
JSON-safe snapshots for telemetry.
"""

import json

import pytest
import torch

from townlet.population.runtime_registry import AgentRuntimeRegistry


class TestAgentRuntimeRegistry:
    """Red phase: specify behaviour for the runtime registry."""

    def test_initialises_zero_baselines_on_device(self):
        device = torch.device("cpu")
        registry = AgentRuntimeRegistry(agent_ids=["agent-0", "agent-1"], device=device)

        baselines = registry.get_baseline_tensor()
        assert baselines.shape == (2,)
        assert baselines.device.type == device.type
        assert torch.allclose(baselines, torch.zeros(2, device=device))

    def test_set_baseline_accepts_python_scalars(self):
        registry = AgentRuntimeRegistry(agent_ids=["agent-0", "agent-1"], device=torch.device("cpu"))

        registry.set_baseline(agent_idx=0, value=120.5)
        registry.set_baseline(agent_idx=1, value=80)

        baselines = registry.get_baseline_tensor()
        assert pytest.approx(baselines[0].item(), rel=1e-6) == 120.5
        assert pytest.approx(baselines[1].item(), rel=1e-6) == 80.0

    def test_set_baselines_accepts_tensor_batches(self):
        registry = AgentRuntimeRegistry(agent_ids=["agent-0", "agent-1"], device=torch.device("cpu"))

        registry.set_baselines(torch.tensor([200.0, 150.0]))

        baselines = registry.get_baseline_tensor()
        assert torch.allclose(baselines, torch.tensor([200.0, 150.0]))

    def test_snapshot_returns_json_serialisable_payload(self):
        registry = AgentRuntimeRegistry(agent_ids=["agent-0", "agent-1"], device=torch.device("cpu"))

        registry.set_baseline(0, 100.0)
        registry.set_baseline(1, torch.tensor(75.0))
        registry.record_survival_time(0, 42)
        registry.record_survival_time(1, 17)
        registry.set_curriculum_stage(1, stage=3)
        registry.set_epsilon(0, 0.05)
        registry.set_intrinsic_weight(1, 0.15)

        snapshot = registry.get_snapshot_for_agent(1)
        payload = snapshot.to_dict()

        # Ensure plain Python types for JSON serialisation
        json.dumps(payload)
        assert payload["agent_id"] == "agent-1"
        assert payload["baseline_survival_steps"] == pytest.approx(75.0)
        assert payload["curriculum_stage"] == 3
        assert payload["survival_time"] == 17
