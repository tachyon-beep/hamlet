# Agent Runtime Registry & Telemetry Separation Plan

**Date:** 2025-11-02
**Owner:** Codex (with handoff to Hamlet core team)
**Goal:** Eliminate tensor→JSON regressions by formalising per-agent runtime state, snapshotting, and reward baseline ownership while setting the groundwork for multi-agent ops telemetry.

---

## 0. Scope & Success Criteria

- Reward baseline data flows through a dedicated `AgentRuntimeRegistry`, not `RewardStrategy`.
- Live inference and any telemetry clients receive only scalar/JSON-safe payloads.
- New design is covered by automated tests exercising registry behaviour, reward integration, and snapshot serialisation.
- Legacy scalar behaviour remains intact during migration (no UI downtime).

Constraints: maintain compatibility with current config files and training entry points; no network schema breaking changes without version gating.

---

## 1. Dependencies & Touch Points

- `src/townlet/population/vectorized.py`
- `src/townlet/environment/reward_strategy.py`
- `src/townlet/demo/live_inference.py`
- Potential new module `src/townlet/population/runtime_registry.py` (name TBD)
- Tests under `tests/test_townlet/`

Assumption: existing population tests simulate single-agent scenarios; we will extend them to multi-agent cases to verify registry tensor handling.

---

## 2. Implementation Phases (TDD: Red → Green → Refactor)

### Phase 1 — Introduce Agent Runtime Registry (Data Model Formalisation)

- **Red**
  - Add `tests/test_townlet/test_runtime_registry.py` covering:
    - Registration: initial baselines default to zero tensor on correct device.
    - `.set_baseline(agent_idx, value)` accepts float or tensor and stores tensor[].
    - `.get_snapshot_for_agent()` returns Python scalars (use `json.dumps` to assert serialisability).
    - Multi-agent scenario verifying independent baselines.
- **Green**
  - Implement `AgentRuntimeRegistry` in new module under `src/townlet/population/`.
  - Instantiate registry in `VectorizedPopulation.__init__`, expose via `self.runtime_registry`.
  - Wire existing curriculum callbacks (where baseline currently calculated) to update the registry while keeping legacy `RewardStrategy` field in place for now (temporary sync in `VectorizedPopulation`).
- **Refactor**
  - Replace ad-hoc baseline tensors in population with registry calls.
  - Add minimal inline docs and type hints.

### Phase 2 — Reward Strategy Consumes Registry (Ownership Correction)

- **Red**
  - Extend or create tests in `tests/test_townlet/test_reward_strategy.py` (new file if needed) ensuring:
    - `calculate_rewards(step_counts, dones, baseline_steps_tensor)` expects baseline tensor argument.
    - Passing tensors with gradient tracking disabled yields correct shaped rewards identical to legacy scalar baseline.
  - Adjust existing environment/population tests to use new signature (fail until implemented).
- **Green**
  - Modify `RewardStrategy` to remove mutable `baseline_survival_steps` field.
  - Update environment/population call sites to pass `runtime_registry.get_baseline_tensor()`.
- **Refactor**
  - Delete legacy scalar storage in `RewardStrategy`.
  - Ensure docstrings reflect new stateless contract.

### Phase 3 — Telemetry Snapshot Boundary (Serialization Contract)

- **Red**
  - Add tests in `tests/test_townlet/test_live_inference.py` (new, using lightweight fake broadcaster) verifying `_build_agent_payloads()` (new helper) returns primitives only and raises if tensors leak.
  - Add regression test for scalar baseline broadcast (simulate multi-agent snapshot).
- **Green**
  - Refactor `live_inference.py` to request `AgentRuntimeRegistry` snapshots, wrap them in versioned DTO (`AgentTelemetrySnapshotV1`), and broadcast via `.to_dict()`.
  - Include `"schema_version": 1` envelope.
- **Refactor**
  - Remove direct access to `reward_strategy.baseline_survival_steps`.
  - Centralise telemetry payload construction (single helper function).

### Phase 4 — Logging & Versioned DTO Infrastructure (Stageable Enhancement)

- **Red**
  - Add tests ensuring DTO carries `version` field and serialises deterministically.
  - Extend database integration test (if available) or add unit test using mock DB to confirm snapshot JSON persists.
- **Green**
  - Introduce `AgentTelemetrySnapshotV1` dataclass with `.to_dict()` conversion.
  - Update SQLite logging path to persist snapshots per episode (optional flag initially).
- **Refactor**
  - Document schema in `docs/testing/telemetry_schema.md` (or existing docs section).
  - Mark DTO creation as single choke point for future schema evolution.

---

## 3. Migration & Rollout

1. **Feature Flag (Optional but recommended):** Keep current scalar baseline in place behind `use_runtime_registry` flag during early integration. Flip default once tests pass.
2. **Frontend Coordination:** Notify frontend owners about new payload envelope; provide sample payload and version contract ahead of merge.
3. **Backfill Tests:** Ensure existing population/environment tests run under both CPU and CUDA devices (use `torch.device` parametrisation in new tests).
4. **Documentation:** Update `docs/townlet/` or relevant READMEs summarising telemetry changes after Phase 3.

---

## 4. Risks & Mitigations

- **Risk:** Test gaps for live inference (currently 0% coverage).
  **Mitigation:** Introduce helper-level tests decoupled from websockets; rely on dependency injection for broadcaster.

- **Risk:** Multi-agent tensor handling may surface device mismatch bugs.
  **Mitigation:** Parametrise registry tests across CPU/GPU (skip GPU if unavailable).

- **Risk:** Downstream consumers (frontend, logging) may assume scalar payloads.
  **Mitigation:** Versioned schema, communication, and sample payloads before deployment.

---

## 5. Validation Checklist

- [ ] All new tests pass in CPU CI lane.
- [ ] Optional CUDA lane (if available) validates registry tensor device logic.
- [ ] Live inference no longer accesses tensors directly; smoke-test UI.
- [ ] Reward shaping regression test confirms identical outputs pre/post refactor.
- [ ] Documentation updated.

---

## 6. Follow-On Work (Post-Plan)

- Implement per-agent audit logging pipeline once telemetry DTO stabilises.
- Extend registry to track affordance visit counters & action entropy.
- Evaluate integration with `docs/testing/REFACTORING_ACTIONS.md` timeline (ACTION #14 synergy for CI checks).
