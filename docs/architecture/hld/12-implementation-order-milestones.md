## 12. Implementation order (milestones)

Section 11 outlined the conceptual order of operations. Section 12 translates that ordering into concrete delivery milestones for engineering, curriculum, safety, and audit teams. These steps form the boot sequence.

### 12.1 Milestone: Snapshots and run folders

Definition of done:

- `configs/<run_name>/` exists with all 5 YAMLs.
- Launching a run generates `runs/<run_name>__<timestamp>/`.
- `runs/<run_name>__<timestamp>/config_snapshot/` is a byte-for-byte copy of those YAMLs.
- `checkpoints/`, `telemetry/`, `logs/` directories are created.
- Runtime never re-reads mutable config after snapshot.

Why it matters:

- Hard provenance from the first tick.
- We can point to "this is the world and brain we actually ran", not "what we think is close".

### 12.2 Milestone: Minimal GraphAgent pipeline

Definition of done:

- `factory.py` can build all declared modules from the snapshot.
- `graph_executor.py` can compile `execution_graph.yaml` into a callable loop.
- `graph_agent.py` exposes `think(raw_observation, prev_recurrent_state) -> { final_action, new_recurrent_state }`.
- We can tick once end-to-end with stub panic_controller and stub EthicsFilter.

Why it matters:

- After this milestone, "the brain is data" is not a slogan, it's running code.

### 12.3 Milestone: cognitive_hash

Definition of done:

- We can generate `cognitive_hash.txt` for a run.
- The hash covers:

  - all 5 YAMLs from snapshot
  - compiled execution graph wiring
  - instantiated module architectures / dims / optimiser LRs
- Telemetry and checkpoints now both include that hash.

Why it matters:

- We now have mind identity you can take to audit.
- You can't quietly mutate cognition without changing the hash.

### 12.4 Milestone: Checkpoint writer and resume

Definition of done:

- We can dump checkpoints at `step_<N>/` with:

  - weights.pt
  - optimizers.pt
  - rng_state.json
  - config_snapshot/
  - cognitive_hash.txt
- We can resume into a brand new run folder using only a checkpoint subfolder.
- If we don't change the snapshot on resume, the resumed run reports the same cognitive hash.
- If we do change the snapshot before resume (panic thresholds, forbid_actions, etc), the resumed run reports a new hash and a new run_id.

Why it matters:

- Chain-of-custody for cognition.
- Controlled forks are now explicit, not sneaky.

### 12.5 Milestone: Telemetry and UI

Definition of done:

- Telemetry per tick logs:

  - run_id
  - tick_index
  - full_cognitive_hash
  - current_goal
  - agent_claimed_reason (if enabled)
  - panic_state
  - candidate_action
  - panic_adjusted_action (+ reason)
  - final_action
  - ethics_veto_applied (+ reason)
  - planning_depth
  - social_model.enabled
  - short summaries of internal beliefs/expectations
- The Run Context Panel renders live:

  - run_id
  - short_cognitive_hash
  - tick / planned_run_length
  - current_goal
  - panic_state
  - planning_depth
  - social_model.enabled
  - panic_override_last_tick (+ panic_reason)
  - ethics_veto_last_tick (+ veto_reason)
  - agent_claimed_reason (if introspection.publish_goal_reason)

Why it matters:

- Teaching becomes possible.
- Governance reviews become visual instead of adversarial.

### 12.6 Milestone: Panic and Ethics go live

Definition of done:

- `panic_controller` actually overrides `candidate_action` when bars cross panic_thresholds.
- `EthicsFilter` actually vetoes forbidden actions and substitutes a safe fallback.
- Both write structured reasons (`panic_reason`, `veto_reason`) into telemetry and show in UI.
- Both steps are present and ordered in `execution_graph.yaml`: policy → panic_controller → EthicsFilter.
- EthicsFilter is final authority.

Why it matters:

- Survival urgency and ethical constraint are now explicit, inspectable modules in the think loop, not implicit reward-shaping heuristics.
- You can show "panic tried X, ethics said no" as an auditable trace, with hash.
