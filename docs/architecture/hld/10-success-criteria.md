## 10. Success criteria

We judge success on three axes: technical, teaching, and governance. All three matter. If we don't hit all three, the story breaks.

### 10.1 Technical success

- [ ] We can launch a run from `configs/<run_name>/` and automatically create `runs/<run_name>__<timestamp>/` with a frozen `config_snapshot/` that contains:
  - `config.yaml`
  - `universe_as_code.yaml`
  - `cognitive_topology.yaml` (Layer 1)
  - `agent_architecture.yaml` (Layer 2)
  - `execution_graph.yaml` (Layer 3)

- [ ] `agent/factory.py` can reconstruct a functioning agent brain (GraphAgent) purely from that frozen `config_snapshot/`, without reading anything from live mutable config.

- [ ] `GraphAgent.think()` can tick once using only that snapshot: perception → hierarchical policy → panic_controller → EthicsFilter → `final_action`.

- [ ] Each checkpoint written under `runs/.../checkpoints/step_<N>/` includes:
  - model weights for every module (perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc)
  - optimiser states
  - RNG state
  - a nested copy of `config_snapshot/`
  - `cognitive_hash.txt` for that checkpoint

- [ ] Resuming from a checkpoint:
  - reloads only from `runs/.../checkpoints/step_<N>/`
  - writes a new run folder `runs/<run_name>__<launch_ts>_resume_<resume_ts>/`
  - reproduces the same cognitive hash if the snapshot is unmodified

- [ ] Telemetry logs one structured row per tick into `runs/.../telemetry/`, with:
  - `run_id`
  - tick index
  - full cognitive hash
  - current_goal
  - panic state
  - candidate_action
  - panic_adjusted_action (+ panic_reason)
  - final_action
  - ethics_veto_applied (+ veto_reason)
  - planning_depth
  - social_model.enabled
  - short belief/world/social summaries

- [ ] The runtime UI ("Run Context Panel") surfaces, live:
  - run_id
  - short_cognitive_hash (pretty form of the full hash)
  - tick / planned_run_length
  - current_goal
  - panic_state
  - planning_depth (world_model.rollout_depth)
  - social_model.enabled
  - panic_override_last_tick (+ panic_reason)
  - ethics_veto_last_tick (+ veto_reason)
  - agent_claimed_reason (if introspection.publish_goal_reason is on)

If we satisfy all of these criteria, we move from "a neural net that produces outputs" to a reproducible mind in a governed world.

---

### 10.2 Pedagogical success

The point of Townlet v2.5 is not just to make a smarter agent. It's to make a teachable agent. We hit pedagogical success when the system is something you can put in front of a class, and they can reason about it like a living system, not a superstition.

- [ ] A beginner can answer "Why didn't it steal the food?" using only:
  - the live Run Context Panel (which shows `ethics_veto_last_tick` and `veto_reason`)
  - the run's `cognitive_topology.yaml` (which shows `compliance.forbid_actions: ["steal", ...]`)

In other words: you do not need to read source code to answer an ethics/safety question. You can answer it from YAML + UI.

- [ ] An intermediate student can:
  - edit `agent_architecture.yaml` (for example, swap GRU → LSTM in the perception module, or change hidden_dim)
  - launch a new run
  - observe how memory/behaviour changes
  - explain the change in terms of memory capacity, not "the AI got weird"

So they can perform controlled ablations by editing config, not by rewriting thousands of lines of Torch.

- [ ] A researcher can:
  - edit `execution_graph.yaml` to, for example, temporarily bypass `world_model_service` input into the policy
  - rerun
  - show that the agent becomes more impulsive / short-horizon
  - prove that change via diff in `execution_graph.yaml` plus new `cognitive_hash.txt`

Meaning: "remove foresight, observe impulsivity" is now a 1-line wiring experiment, not a 2-week surgery.

- [ ] For any interesting emergent behaviour clip, we can pull the run folder and point to:
  - which mind (full cognitive hash)
  - which world rules (`universe_as_code.yaml`)
  - which panic thresholds
  - which compliance rules (`forbid_actions`, penalties)
  - what goal the agent believed it was pursuing at that tick (`current_goal`)
  - what reason the agent claimed (`agent_claimed_reason`)

This is critical for classroom demonstrations. Instructors can scrub to tick 842 and explain that the agent believed it was in SURVIVAL mode, panic was active, and EthicsFilter blocked `steal`.

---

### 10.3 Governance success

Governance stakeholders view the system through enforceability rather than aesthetics. Their central question is whether the artefact can withstand formal review.

- [ ] We can prove to an auditor that, at tick T in run R:
  - `cognitive_topology.yaml` at that tick had `forbid_actions: ["attack", "steal"]`
  - `execution_graph.yaml` at that tick still routed all candidate actions through `EthicsFilter`
  - telemetry for tick T shows `ethics_veto_applied: true` and `veto_reason: "steal forbidden"`

This allows us to state: the agent attempted to steal at tick T, the action was blocked, and both the configuration and telemetry demonstrate why.

- [ ] We can replay that same mind, at that same point in time, using only the checkpoint directory from that run. We don't need any mutable source code or live config. That replayed agent produces the same cognitive hash and the same cognitive wiring.

That is chain-of-custody for cognition.

- *Operational note:* To deliver that proof, pull the tick record from `runs/<run_id>/telemetry/` (each row is produced by `VectorizedPopulation.build_telemetry_snapshot` in `src/townlet/population/vectorized.py`) and pair it with the matching checkpoint hash in `runs/<run_id>/checkpoints/step_<N>/cognitive_hash.txt`. The snapshot structure comes straight from `AgentTelemetrySnapshot` (`src/townlet/population/runtime_registry.py`), so auditors know exactly which JSON fields must be present.

- [ ] We can demonstrate lineage rules:
  - If you resume without changing the snapshot, it's the same mind (same hash).
  - If you edit anything that changes cognition (panic thresholds, greed, social_model.enabled, EthicsFilter rules, rollout_depth, etc), the hash changes and we give it a new run_id. We don't pretend it's "the same agent, just adjusted a bit".

That's governance-grade identity, not research convenience.

---
