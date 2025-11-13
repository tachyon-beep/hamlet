# Brain As Code: Strategic Roadmap to Full Vision

> **For Claude:** This is a STRATEGIC ROADMAP, not a single implementation plan. Each phase should be executed as a separate project with its own detailed implementation plan.

**Current State:** Phase 3 Complete (Dueling DQN + Prioritized Experience Replay)

**Goal:** Reach the full BAC vision from `docs/architecture/BRAIN_AS_CODE.md` - Software Defined Agents with 3-layer configuration, graph execution, panic/ethics, and glass-box telemetry.

**Progress:** ~5-10% complete. We have declarative Q-network configuration (Layer 2 subset). Need to build Layers 1+3, modular brain architecture, graph execution engine, and governance infrastructure.

**Timeline:** 9 phases over 12-18 months

**Strategy:** Incremental evolution from "enhanced Q-learning" → "modular SDA". Each phase delivers standalone value without breaking existing systems.

---

## Architecture Context

The full BAC vision (v2.5) requires three fundamental shifts:

1. **Configuration Layers** - From single `brain.yaml` → 3-layer system (cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml)
2. **Modular Brain** - From monolithic Q-network → separate perception/world-model/social/policy modules
3. **Graph Execution** - From hardcoded `step()` → DAG executor with explicit panic/ethics veto points

Current Phase 3 delivers:
- ✅ Declarative Q-network architecture (feedforward, recurrent, dueling)
- ✅ Declarative optimizer/loss/replay configuration
- ✅ Checkpoint provenance via brain_hash
- ✅ Test-driven development with 100% coverage

Missing from full vision:
- ❌ Layer 1 (cognitive_topology.yaml) - behavioral intent, personality, panic, compliance
- ❌ Layer 3 (execution_graph.yaml) - cognition DAG, veto points
- ❌ Modular brain (perception/world/social as separate modules)
- ❌ Graph execution engine (graph_executor.py, GraphAgent)
- ❌ Safety modules (EthicsFilter, panic_controller)
- ❌ Glass-box telemetry (per-tick structured logs with veto/panic tracking)
- ❌ Run bundles (5-YAML structure with snapshot discipline)
- ❌ Declarative goals (termination conditions DSL)

---

## Phase Overview

### **Phase 4: Safety Wrappers** (2 weeks)
Layer 1 Lite + panic_controller + EthicsFilter as wrappers around existing Q-network.
- **Value**: Governance-grade compliance (forbid_actions) and crisis handling (panic thresholds)
- **Risk**: Low - additive wrappers, no Q-network changes

### **Phase 5: Run Bundles & Provenance** (2 weeks)
5-YAML structure, snapshot copying, cognitive_hash expansion, resume from checkpoint-only.
- **Value**: Reproducibility, auditability, experimental forking
- **Risk**: Low - infrastructure refactor, preserves training logic

### **Phase 6: Glass-Box Telemetry** (2 weeks)
Per-tick structured logs, Run Context Panel UI, veto/panic attribution.
- **Value**: Interpretability, teaching material, safety audits
- **Risk**: Low - additive telemetry, no training changes

### **Phase 7: Modular Perception** (3 weeks)
Separate perception_encoder module, belief distribution interface, factored observation processing.
- **Value**: Reusable perception, pretraining support, cleaner architecture
- **Risk**: Medium - refactors observation flow, requires testing

### **Phase 8: World Model + Social Model** (4 weeks)
Predictive modules (next_state, next_reward, next_done, next_value, other_agent_goals).
- **Value**: Planning-capable agents, theory of mind, richer behaviors
- **Risk**: Medium - new training objectives, requires datasets

### **Phase 9: Hierarchical Policy** (4 weeks)
Meta-controller (goal selection) + controller (action selection), multi-timescale reasoning.
- **Value**: Strategic agents, explainable goals, curriculum progression
- **Risk**: Medium-High - replaces monolithic Q-network policy head

### **Phase 10: Graph Execution Engine** (6 weeks)
graph_executor.py, execution_graph.yaml, GraphAgent, factory.py, full SDA architecture.
- **Value**: Complete BAC vision, swappable modules, declarative cognition
- **Risk**: High - major architectural rewrite, migration path required

### **Phase 11: Declarative Goals** (2 weeks)
Goal termination conditions DSL, declarative completion criteria, curriculum integration.
- **Value**: Readable goals, diffable behavior, curriculum science
- **Risk**: Low - additive feature on Phase 9 hierarchical policy

### **Phase 12: Advanced Compliance** (2 weeks)
Situational bans, contextual norms, compliance DSL, richer ethics.
- **Value**: Nuanced safety rules, social reasoning constraints
- **Risk**: Low - extends Phase 4 EthicsFilter

---

## Detailed Phase Roadmaps

---

## Phase 4: Safety Wrappers (2 weeks)

**Status**: Next Phase (ready to start)

**Goal**: Add governance-grade compliance and crisis handling to existing Q-learning architecture without changing Q-network internals.

**Deliverables**:
1. `cognitive_topology.yaml` schema (Layer 1 Lite - just panic_thresholds and compliance)
2. `panic_controller.py` - threshold-based action override (energy < 15%, health < 25%)
3. `ethics_filter.py` - compliance enforcement (forbid_actions, penalize_actions)
4. Integration as wrappers in VectorizedPopulation.step() (action selection → panic → ethics → env)
5. Basic telemetry (log veto_reason, panic_reason to TensorBoard)

**Architecture**:
- Wrappers around existing Q-network action selection
- No changes to Q-network architecture or training loop
- Additive configuration file (cognitive_topology.yaml)

**Integration Point**:
```python
# VectorizedPopulation.step() - after Q-network action selection
candidate_action = q_network.select_action(obs)
panic_action, panic_reason = panic_controller.maybe_override(candidate_action, bars, thresholds)
final_action, veto_reason = ethics_filter.enforce(panic_action, forbid_actions)
env.step(final_action)
```

**Success Criteria**:
- [ ] Agent respects forbid_actions (e.g., never steals even when desperate)
- [ ] Panic overrides trigger when bars below thresholds (e.g., emergency hospital visit)
- [ ] TensorBoard shows veto_count, panic_count, veto_reason, panic_reason
- [ ] cognitive_topology.yaml controls behavior (edit thresholds → observe panic changes)

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase4-safety-wrappers.md`

**Estimated Effort**: 2 weeks (5 days implementation, 3 days testing, 2 days docs)

---

## Phase 5: Run Bundles & Provenance (2 weeks)

**Status**: Blocked by Phase 4

**Goal**: Expand provenance system from `brain.yaml` → 5-YAML run bundles with snapshot discipline, enabling reproducible experiments and governance audits.

**Deliverables**:
1. Run bundle structure: `configs/<run_name>/` with 5 YAMLs:
   - `config.yaml` (simulation runtime: ticks, population, seed, curriculum)
   - `universe_as_code.yaml` (world config: meters, affordances, map) [already exists as substrate.yaml/bars.yaml/affordances.yaml]
   - `cognitive_topology.yaml` (Layer 1: panic, compliance) [from Phase 4]
   - `agent_architecture.yaml` (Layer 2: Q-network, optimizer, loss) [rename brain.yaml]
   - `execution_graph.yaml` (Layer 3: think loop) [stub for now]
2. Launcher creates `runs/<run_name>__<timestamp>/config_snapshot/` (deep copy, not symlink)
3. Cognitive hash expansion: hash all 5 YAMLs + compiled architectures
4. Checkpoint expansion: include RNG state + full config_snapshot
5. Resume semantics: load from checkpoint-only, ignore mutable configs/

**Architecture**:
- No training loop changes
- Refactor checkpoint save/load to include config_snapshot + RNG
- Refactor cognitive_hash to cover all 5 configs

**Integration Point**:
```python
# DemoRunner.launch()
run_bundle = load_bundle(configs/<run_name>/)  # 5 YAMLs
snapshot_dir = f"runs/{run_name}__{timestamp}/config_snapshot/"
copy_bundle(run_bundle, snapshot_dir)  # freeze provenance
cognitive_hash = compute_hash(snapshot_dir)  # hash all 5 YAMLs
```

**Success Criteria**:
- [ ] Launch from `configs/<run_name>/` creates `runs/<run_name>__<timestamp>/config_snapshot/` with 5 YAMLs
- [ ] Checkpoint includes weights + optimizers + RNG + config_snapshot + cognitive_hash.txt
- [ ] Resume loads from checkpoint-only (ignores configs/)
- [ ] Mutating snapshot before resume changes cognitive_hash (fork detection)
- [ ] Can reproduce run exactly from checkpoint alone

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase5-run-bundles.md`

**Estimated Effort**: 2 weeks (5 days implementation, 3 days migration, 2 days testing/docs)

---

## Phase 6: Glass-Box Telemetry (2 weeks)

**Status**: Blocked by Phases 4-5

**Goal**: Replace TensorBoard scalars with per-tick structured telemetry, enabling forensic analysis, teaching material, and safety audits.

**Deliverables**:
1. Per-tick telemetry writer: `runs/<run_id>/telemetry/tick_NNNNNN.json` (or batched CSV)
2. Telemetry schema: run_id, tick, cognitive_hash, candidate_action, panic_adjusted_action, final_action, veto_reason, panic_reason, bars, current_goal (stub)
3. Run Context Panel (UI): live display of run_id, short_hash, tick, panic_state, ethics_veto_last_tick, planning_depth (stub)
4. Telemetry replay tool: load telemetry from past runs for analysis

**Architecture**:
- Log structured data per tick (or batched every N ticks)
- UI websocket stream includes telemetry fields
- No training loop changes

**Integration Point**:
```python
# VectorizedPopulation.step() - after action selection
telemetry_record = {
    "run_id": self.run_id,
    "tick": self.total_steps,
    "cognitive_hash": self.cognitive_hash,
    "candidate_action": candidate_action,
    "panic_adjusted_action": panic_action,
    "final_action": final_action,
    "veto_reason": veto_reason,
    "panic_reason": panic_reason,
    "bars": {bar: value for bar, value in bars.items()},
}
write_telemetry(telemetry_record, self.telemetry_dir)
```

**Success Criteria**:
- [ ] Every tick logged to `runs/<run_id>/telemetry/` with structured schema
- [ ] UI shows Run Context Panel with run_id, hash, tick, panic_state, veto_last_tick
- [ ] Can replay past run's telemetry to reconstruct agent's decision chain
- [ ] Telemetry includes attribution (which action led to veto/panic)

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase6-telemetry.md`

**Estimated Effort**: 2 weeks (4 days telemetry, 4 days UI, 2 days testing/docs)

---

## Phase 7: Modular Perception (3 weeks)

**Status**: Blocked by Phases 4-6

**Goal**: Separate perception from Q-network, creating reusable perception_encoder module with belief distribution interface.

**Deliverables**:
1. `perception_encoder.py` - separate module for observation processing (CNN + MLP + GRU)
2. Belief distribution interface: standardized output (belief_dim, e.g., 128)
3. Refactor Q-network to accept belief distribution as input (not raw obs)
4. agent_architecture.yaml schema for perception_encoder (spatial_frontend, vector_frontend, core, heads)
5. Perception pretraining support (reconstruction + next_step objectives)

**Architecture**:
- Perception becomes first-class module, separate from Q-network
- Q-network consumes belief distribution (processed obs) instead of raw obs
- Perception can be pretrained independently on observation logs

**Integration Point**:
```python
# VectorizedPopulation.step()
belief_distribution = perception_encoder(raw_obs)  # [batch, belief_dim=128]
q_values = q_network(belief_distribution)  # Q-network now takes belief, not raw obs
```

**Success Criteria**:
- [ ] perception_encoder builds from agent_architecture.yaml
- [ ] Q-network accepts belief_dim input (not raw obs_dim)
- [ ] Can swap perception_encoder without changing Q-network
- [ ] Perception pretraining on observation logs works

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase7-modular-perception.md`

**Estimated Effort**: 3 weeks (6 days implementation, 4 days testing, 3 days migration/docs)

**Migration Risk**: Medium - changes observation flow, requires updating all Q-networks

---

## Phase 8: World Model + Social Model (4 weeks)

**Status**: Blocked by Phase 7

**Goal**: Add predictive modules for planning (world_model) and theory-of-mind reasoning (social_model).

**Deliverables**:
1. `world_model.py` - predicts next_state (belief), next_reward, next_done, next_value
2. `social_model.py` - predicts other_agent_goals, other_agent_next_action
3. agent_architecture.yaml schemas for both modules (core networks, heads, optimizers)
4. World model pretraining on ground truth logs (dynamics + value objectives)
5. Social model pretraining on multi-agent logs (CTDE intent prediction)
6. Integration as services (world_model_service, social_model_service) for hierarchical policy

**Architecture**:
- World model: MLP/GRU that predicts (s', r', done', V') from (s, a)
- Social model: GRU that predicts (goal_dist, action_dist) for other agents
- Both modules train alongside Q-network, provide context for action selection

**Integration Point**:
```python
# VectorizedPopulation.step() - during action selection
belief = perception_encoder(obs)
world_prediction = world_model.predict(belief, candidate_action)  # s', r', done'
social_prediction = social_model.predict(belief, other_obs)  # other goals/actions
# Use predictions to inform policy (Phase 9)
```

**Success Criteria**:
- [ ] World model predicts next state with reasonable accuracy (measured on held-out data)
- [ ] Social model predicts other agent actions above chance
- [ ] Both modules build from agent_architecture.yaml
- [ ] Pretraining objectives converge

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase8-predictive-models.md`

**Estimated Effort**: 4 weeks (8 days implementation, 4 days pretraining, 4 days testing/docs)

**Migration Risk**: Medium - new training objectives, requires logged data

---

## Phase 9: Hierarchical Policy (4 weeks)

**Status**: Blocked by Phase 8

**Goal**: Replace monolithic Q-network with hierarchical policy (meta-controller selects goals, controller selects actions to achieve goal).

**Deliverables**:
1. `hierarchical_policy.py` - meta_controller (goal selection) + controller (action selection)
2. Goal vocabulary: SURVIVAL, THRIVING, SOCIAL, EXPLORATION, ECONOMIC
3. Meta-controller uses world/social predictions to select goal (every N ticks)
4. Controller uses goal + belief to select action (every tick)
5. agent_architecture.yaml schema for hierarchical_policy (meta/controller networks, goal_vector_dim)
6. Telemetry expansion: log current_goal, goal_reason

**Architecture**:
- Meta-controller: MLP/GRU that outputs goal_vector (dim=16) every N ticks
- Controller: MLP/GRU that outputs action conditioned on (belief, goal_vector)
- Replaces monolithic Q-network for action selection

**Integration Point**:
```python
# VectorizedPopulation.step()
belief = perception_encoder(obs)
if tick % meta_period == 0:
    current_goal = meta_controller(belief, world_service, social_service)  # every 50 ticks
action = controller(belief, current_goal)  # every tick
```

**Success Criteria**:
- [ ] Meta-controller selects goals (SURVIVAL when bars low, THRIVING when bars high)
- [ ] Controller achieves selected goal (e.g., SURVIVAL → hospital visit)
- [ ] Telemetry shows goal transitions (THRIVING → SURVIVAL when energy drops)
- [ ] Interpretable: "agent chose X because current_goal=SURVIVAL"

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase9-hierarchical-policy.md`

**Estimated Effort**: 4 weeks (8 days implementation, 4 days training, 4 days testing/docs)

**Migration Risk**: Medium-High - replaces Q-network action selection, requires retraining

---

## Phase 10: Graph Execution Engine (6 weeks)

**Status**: Blocked by Phase 9

**Goal**: Full SDA architecture - replace hardcoded step() with DAG executor, complete 3-layer config system.

**Deliverables**:
1. `agent/factory.py` - build GraphAgent from 5-YAML bundle
2. `agent/graph_agent.py` - generic nn.Module owning module registry + recurrent state
3. `agent/graph_executor.py` - execute cognition DAG from execution_graph.yaml
4. execution_graph.yaml schema (inputs, services, steps, outputs)
5. Refactor all modules (perception, world, social, policy, panic, ethics) to fit graph interface
6. Cognitive hash expansion: include compiled execution graph
7. Migration path: VectorizedPopulation → GraphPopulation

**Architecture**:
- GraphAgent replaces monolithic Q-network
- graph_executor compiles execution_graph.yaml into ordered step list
- Each step: node (module), inputs (data dependencies), outputs (to scratchpad)
- Execution order: perception → policy → panic_controller → EthicsFilter → final_action

**Integration Point**:
```python
# GraphAgent.think()
data_cache = {}
for step in self.executor.steps:
    node = self.modules[step.node]
    inputs = [data_cache[dep] for dep in step.inputs]
    outputs = node(*inputs)
    data_cache[step.name] = outputs
return data_cache["final_action"], data_cache["new_recurrent_state"]
```

**Success Criteria**:
- [ ] GraphAgent builds from 5-YAML bundle (factory.py)
- [ ] execution_graph.yaml controls cognition order (edit YAML → change think loop)
- [ ] Can swap modules without changing GraphAgent (e.g., replace world_model)
- [ ] Cognitive hash changes when execution graph changes
- [ ] Migration complete: all curriculum levels run on GraphAgent

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase10-graph-execution.md`

**Estimated Effort**: 6 weeks (12 days implementation, 6 days migration, 6 days testing/docs)

**Migration Risk**: High - major architectural rewrite, requires staged rollout

---

## Phase 11: Declarative Goals (2 weeks)

**Status**: Blocked by Phase 10

**Goal**: Add goal termination conditions DSL for readable, diffable, reproducible goal completion logic.

**Deliverables**:
1. Goal termination DSL: `all`/`any` trees over bar comparisons (energy >= 0.8, money >= 1.0)
2. cognitive_topology.yaml schema expansion: goal_definitions with termination conditions
3. Lightweight interpreter: evaluate termination trees at runtime
4. Integration with hierarchical policy: goal completes when termination satisfied

**Architecture**:
- Goals defined declaratively in cognitive_topology.yaml
- Interpreter evaluates termination conditions each tick
- Meta-controller selects next goal when current goal completes

**Integration Point**:
```yaml
# cognitive_topology.yaml
goal_definitions:
  - id: "survive_energy"
    termination:
      all:
        - { bar: "energy", op: ">=", val: 0.8 }
        - { bar: "health", op: ">=", val: 0.7 }
```

**Success Criteria**:
- [ ] Goals complete when termination conditions satisfied
- [ ] Goal logic readable without code (policy reviewers understand behavior)
- [ ] Curriculum can adjust goals (SURVIVAL: energy >= 0.5 → energy >= 0.8)

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase11-declarative-goals.md`

**Estimated Effort**: 2 weeks (5 days implementation, 3 days testing, 2 days docs)

**Migration Risk**: Low - additive feature on Phase 9 hierarchical policy

---

## Phase 12: Advanced Compliance (2 weeks)

**Status**: Blocked by Phase 10

**Goal**: Extend EthicsFilter with situational bans, contextual norms, and compliance DSL for nuanced safety rules.

**Deliverables**:
1. Compliance DSL: structured conditions for penalize_actions (if: all/any over bar comparisons)
2. Situational bans: "steal forbidden unless target is abandoned_property"
3. Contextual norms: "call_ambulance when health high = abuse penalty"
4. cognitive_topology.yaml schema expansion: compliance with conditional penalties

**Architecture**:
- Compliance DSL parallel to goal termination DSL
- EthicsFilter evaluates conditions before veto/penalty
- Remains declarative (no arbitrary code)

**Integration Point**:
```yaml
# cognitive_topology.yaml
compliance:
  forbid_actions: ["attack", "steal"]
  penalize_actions:
    - action: "call_ambulance"
      if:
        all:
          - { bar: "health", op: ">=", val: 0.7 }
          - { bar: "mood", op: ">=", val: 0.8 }
      penalty: -5.0
      note: "stop faking emergencies"
```

**Success Criteria**:
- [ ] Situational bans work (steal blocked unless abandoned_property)
- [ ] Contextual norms work (ambulance abuse penalized)
- [ ] Compliance logic readable without code

**Documentation**: `docs/plans/2025-01-13-brain-as-code-phase12-advanced-compliance.md`

**Estimated Effort**: 2 weeks (5 days implementation, 3 days testing, 2 days docs)

**Migration Risk**: Low - extends Phase 4 EthicsFilter

---

## Success Criteria (Full Vision)

### Technical Success
- [ ] Launch from `configs/<run_name>/` creates `runs/<run_name>__<timestamp>/config_snapshot/` with 5 YAMLs
- [ ] GraphAgent builds from frozen snapshot (through factory.py) and ticks
- [ ] Checkpoints include weights, optimizers, RNG, config_snapshot, cognitive_hash
- [ ] Resume uses checkpoint-only, reproduces cognitive_hash (unless deliberately changed)
- [ ] Telemetry logs run_id, cognitive_hash, veto_reason, panic_reason for attribution
- [ ] UI shows Run Context Panel (panic, veto, goal, hash) live

### Pedagogical Success
- [ ] Beginner answers "Why didn't it steal?" by reading UI veto_reason + cognitive_topology.yaml forbid_actions
- [ ] Intermediate student swaps GRU → LSTM in agent_architecture.yaml, observes memory consequences
- [ ] Researcher edits execution_graph.yaml to bypass world_model_service, observes impulsivity
- [ ] Any emergent behavior clip traced to exact mind (hash), world rules, ethics rules, panic thresholds

### Governance Success
- [ ] Prove to auditor: at tick T in run R, agent had forbid_actions and EthicsFilter ran
- [ ] Replay agent exactly as it existed at tick T using only checkpoint directory

---

## Dependencies & Ordering

**Critical Path**:
1. Phase 4 (Safety Wrappers) - enables Phases 5-6
2. Phase 5 (Run Bundles) - enables Phase 6-10
3. Phase 6 (Telemetry) - enables teaching/governance (can parallelize with 7-9)
4. Phase 7 (Modular Perception) - enables Phase 8
5. Phase 8 (Predictive Models) - enables Phase 9
6. Phase 9 (Hierarchical Policy) - enables Phase 10-11
7. Phase 10 (Graph Execution) - enables full BAC vision
8. Phase 11-12 (Goals + Compliance) - polish, can be deferred

**Parallelization Opportunities**:
- Phases 6 (Telemetry) can run in parallel with 7-9 (modular brain)
- Phases 11-12 (Goals + Compliance) independent, can be tackled in any order

**Long Poles** (highest risk/effort):
- Phase 10 (Graph Execution) - 6 weeks, major rewrite
- Phase 9 (Hierarchical Policy) - 4 weeks, replaces Q-network
- Phase 8 (Predictive Models) - 4 weeks, new training objectives

---

## Risk Mitigation

### Migration Risk (Phases 7, 9, 10)
- **Strategy**: Maintain legacy VectorizedPopulation alongside GraphPopulation during migration
- **Validation**: Run both systems on same checkpoints, compare metrics
- **Rollback**: Keep legacy system until GraphPopulation reaches performance parity

### Training Regression (Phases 8, 9)
- **Strategy**: Pretrain new modules on logged data before end-to-end training
- **Validation**: Ablation studies (world_model on/off, hierarchical vs flat policy)
- **Rollback**: Disable new modules if performance degrades

### Scope Creep (All Phases)
- **Strategy**: Each phase has clear "DONE" criteria, resist feature additions mid-phase
- **Validation**: Code review against original phase spec
- **Rollback**: Cut features that don't meet MVP bar, defer to future phases

---

## Resource Estimates

**Total Effort**: 27 weeks (6-7 months) of focused implementation
- Phase 4: 2 weeks
- Phase 5: 2 weeks
- Phase 6: 2 weeks
- Phase 7: 3 weeks
- Phase 8: 4 weeks
- Phase 9: 4 weeks
- Phase 10: 6 weeks
- Phase 11: 2 weeks
- Phase 12: 2 weeks

**Timeline Assumptions**:
- Single developer, full-time
- No major blockers or scope changes
- Parallelization not included (could save 2-3 weeks)

**Realistic Timeline**: 12-18 months with:
- Context switching (other projects)
- Discovery work (Phase 10 unknowns)
- Performance tuning (Phase 9 training)
- Documentation/teaching materials

---

## Next Steps

**Immediate (Q1 2025)**:
1. Complete Phase 3 code review and merge
2. Create detailed implementation plan for Phase 4 (Safety Wrappers)
3. Set up dedicated worktree for Phase 4 development

**Short-term (Q2 2025)**:
- Execute Phases 4-6 (Safety + Provenance + Telemetry)
- Milestone: Governance-ready system with veto/panic tracking

**Medium-term (Q3-Q4 2025)**:
- Execute Phases 7-9 (Modular Brain + Hierarchical Policy)
- Milestone: Planning-capable agents with interpretable goals

**Long-term (Q1-Q2 2026)**:
- Execute Phase 10 (Graph Execution Engine)
- Milestone: Full BAC vision, Software Defined Agents

**Polish (Q3 2026)**:
- Execute Phases 11-12 (Declarative Goals + Advanced Compliance)
- Milestone: Production-ready, curriculum-complete

---

## Open Questions

1. **Pretraining Data**: Do we have sufficient logged data for world_model/social_model pretraining? (Phase 8)
2. **Performance Parity**: What metrics prove GraphAgent matches legacy Q-network performance? (Phase 10)
3. **Curriculum Compatibility**: Can curriculum configs migrate to 5-YAML bundles without breaking? (Phase 5)
4. **Social Model Scope**: Do we need full theory-of-mind or just occupancy prediction? (Phase 8)
5. **Hierarchical Training**: HIRO? Options framework? What's the meta-controller training objective? (Phase 9)

These questions should be resolved during detailed implementation planning for each phase.

---

## References

- `docs/architecture/BRAIN_AS_CODE.md` - Full vision document
- `docs/tasks/TASK-005-BRAIN-AS-CODE.md` - Phases 1-3 specification
- `docs/plans/2025-01-13-brain-as-code-phase1-feedforward.md` - Phase 1 implementation
- `docs/plans/2025-01-13-brain-as-code-phase2-recurrent.md` - Phase 2 implementation
- `docs/plans/2025-01-13-brain-as-code-phase3-advanced.md` - Phase 3 implementation

---

**Document Version**: 1.0
**Date**: 2025-01-13
**Status**: Strategic Roadmap (not for immediate execution)
**Owner**: Principal Technical Advisor (AI)
