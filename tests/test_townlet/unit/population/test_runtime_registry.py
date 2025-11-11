"""Unit tests for AgentRuntimeRegistry telemetry helpers."""

from __future__ import annotations

import time

import pytest
import torch

from townlet.population.runtime_registry import AgentRuntimeRegistry


class TestAgentRuntimeRegistryInitialization:
    def test_requires_agent_ids(self):
        """Initialization should fail when no agent ids are provided."""

        with pytest.raises(ValueError, match="agent_ids must contain at least one entry"):
            AgentRuntimeRegistry(agent_ids=[], device=torch.device("cpu"))

    def test_initial_tensors_live_on_device(self):
        """All tensors should be created on the requested device."""

        device = torch.device("cpu")
        registry = AgentRuntimeRegistry(agent_ids=["a0", "a1"], device=device)

        assert registry.get_survival_time_tensor().device == device
        assert registry.get_curriculum_stage_tensor().shape == (2,)


class TestAgentRuntimeRegistryMutations:
    def test_record_survival_time_accepts_tensor_input(self):
        registry = AgentRuntimeRegistry(agent_ids=["a0"], device=torch.device("cpu"))

        registry.record_survival_time(0, torch.tensor(7, dtype=torch.long))
        assert registry.get_survival_time(0) == 7

    def test_setters_accept_python_scalars(self):
        registry = AgentRuntimeRegistry(agent_ids=["a0"], device=torch.device("cpu"))

        registry.set_curriculum_stage(0, 3)
        registry.set_epsilon(0, 0.25)
        registry.set_intrinsic_weight(0, 0.75)

        assert registry.get_curriculum_stage(0) == 3
        assert registry.get_epsilon(0) == pytest.approx(0.25)
        assert registry.get_intrinsic_weight(0) == pytest.approx(0.75)

    def test_setters_accept_list_values(self):
        registry = AgentRuntimeRegistry(agent_ids=["a0"], device=torch.device("cpu"))

        registry.record_survival_time(0, [42])
        assert registry.get_survival_time(0) == 42

    def test_invalid_value_type_raises(self):
        registry = AgentRuntimeRegistry(agent_ids=["a0"], device=torch.device("cpu"))

        with pytest.raises(TypeError):
            registry._ensure_tensor({"bad": "value"}, dtype=torch.float32)  # type: ignore[arg-type]


class TestAgentRuntimeRegistrySnapshots:
    def test_snapshot_returns_json_safe_payload(self, monkeypatch):
        registry = AgentRuntimeRegistry(agent_ids=["agent_7"], device=torch.device("cpu"))
        registry.record_survival_time(0, 12)
        registry.set_curriculum_stage(0, 4)
        registry.set_epsilon(0, 0.1)
        registry.set_intrinsic_weight(0, 0.33)

        monkeypatch.setattr(time, "time", lambda: 1234.567)

        snapshot = registry.get_snapshot_for_agent(0).to_dict()

        assert snapshot == {
            "agent_id": "agent_7",
            "survival_time": 12,
            "curriculum_stage": 4,
            "epsilon": pytest.approx(0.1),
            "intrinsic_weight": pytest.approx(0.33),
            "timestamp_unix": pytest.approx(1234.567),
        }
