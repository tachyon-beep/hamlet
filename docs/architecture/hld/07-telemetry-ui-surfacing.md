## 7. Telemetry and UI Surfacing

---

The goal is not to build "an AI that seems smart"; it is to build an AI whose cognition can be observed and cited in formal settings.

Townlet v2.5 therefore ships with first-class introspection. We log what the mind attempted, what intervened, and why—live, per tick, and tied to identity. This is the core of the glass-box capability.

We expose two layers of visibility:

1. a live panel in the UI for humans watching the sim in real time, and
2. structured telemetry on disk for replay, teaching, and audit.

These two layers must always agree. Any divergence is a defect.

### 7.1 Run Context Panel (Live Inspector HUD)

At runtime, clicking an agent opens a compact panel describing what the mind is doing at that moment. The panel is populated from the same data that we log to disk.

This panel MUST include at least:

- run_id
  Example: `L99_AusterityNightshift__2025-11-03-12-14-22`
  This tells you which frozen bundle of world + brain you're looking at.

- short_cognitive_hash
  A short form (e.g. first eight characters) of the agent's full cognitive hash.
  This identifies which exact mind occupies that body. If two bodies share the same short hash, we are observing two instances of the same brain specification under different conditions.

- tick
  Current tick index and planned_run_length from config.yaml.
  Lets you say "this happened at tick 842 out of 10,000", which matters when you're doing curriculum or staged hardship.

- current_goal
  The high-level strategic goal the meta-controller (hierarchical_policy.meta_controller) reports, e.g. `SURVIVAL`, `THRIVING`, `SOCIAL`.
  This reflects engine truth rather than interpretation.

- panic_state
  Boolean or enum. Are we currently in emergency override because we tripped `panic_thresholds` in cognitive_topology.yaml (Layer 1)?
  This is: "is the Panic Controller allowed to overrule normal planning right now?"

- panic_override_last_tick
  If the panic_controller overrode the policy during the previous tick:

  - which action it forced (e.g. `call_ambulance`), and
  - the reason (e.g. `energy_critical`).
    This conveys when emergency logic executed, rather than merely reporting that the agent moved.

- ethics_veto_last_tick
  Did EthicsFilter block the action last tick?
  If yes, we show `veto_reason` ("forbid_actions: ['steal']").
  This is how we tell instructors "it tried to steal, and we stopped it", not just "it didn't steal".

- planning_depth
  Pulled from cognitive_topology.yaml → world_model.rollout_depth.
  Literally: "how many ticks ahead this mind is allowed to imagine right now." That's an interpretable knob for 'impulsiveness'.

- social_model.enabled
  True/false.
  Are we currently reasoning about other agents as intentional actors, or are we running with social modelling disabled? This is huge for ablation labs ("this is what happens when you turn off Theory of Mind").

- agent_claimed_reason (if introspection.publish_goal_reason is true)
  This is what the agent thinks it's doing in words, e.g.
  "I'm going to work so I can pay rent."
  We very explicitly label this as self-report, not guaranteed causal truth.

Why this UI panel matters:

It lets you stand next to a student, point to the HUD, and narrate:
"See? It's currently in SURVIVAL, panic_state is true because health is below 25 percent, so panic_controller overrode the normal plan and told it to call an ambulance. Ethics allowed that because calling an ambulance is legal even if money is low. Also look: it tried to steal last tick, EthicsFilter vetoed that and recorded the reason. This is not chaos. This is a traceable mind reacting under policy."

That's the teaching win. That's also the regulatory win.

### 7.2 Telemetry (Per-Tick Trace to Disk)

In parallel with the live panel, we write structured telemetry into:

runs/<run_id>/telemetry/

One row per tick (or batched if we're throttling IO). This creates a replayable audit trail of the agent's cognition over time. It is the forensic record.

Each telemetry row MUST include at minimum:

- run_id
  Which run bundle we're in.

- tick_index
  Which tick this record corresponds to.

- full_cognitive_hash
  The full (not shortened) cognitive hash of the mind.
  This proves which mind produced this row.

- current_goal
  Engine truth from the meta-controller. For example: `SURVIVAL`.

- agent_claimed_reason
  If introspection.publish_goal_reason is enabled in Layer 1.
  This is "what the agent says it's doing" in natural language. Purely for humans. Not trusted as causal truth, but extremely useful for pedagogy ("listen to how it's rationalising").

- panic_state
  Whether panic_controller is active this tick.

- candidate_action
  The first action proposed by hierarchical_policy before any overrides.

- panic_adjusted_action
  What panic_controller wanted to do after checking panic_thresholds, plus:

  - panic_override_applied (bool)
  - panic_reason ("health_critical", "energy_critical", etc)

- final_action
  What actually went out to the environment after EthicsFilter.

- ethics_veto_applied
  Whether EthicsFilter overruled the panic-adjusted action.

- veto_reason
  If veto_applied is true, why (e.g. `"steal" is forbidden by compliance.forbid_actions`).

- belief_uncertainty_summary
  Short numeric/text summary of how confident the perception module is about critical bars.
  Example: `"energy_estimate_confidence": 0.42`.
  This exposes cases where an agent ignored a fridge because it did not believe it was starving.

- world_model_expectation_summary
  Short summary of what the world_model predicted would happen if it followed the chosen plan.
  Example: predicted immediate reward, predicted survival risk, etc.

- social_model_inference_summary
  Short summary of what the agent believes others are about to do.
  Example: `"Agent_2_intent": "use_fridge"` with a confidence score.

We also optionally include:

- planning_depth (current rollout horizon from Layer 1)
- social_model.enabled (boolean at this tick)

Why telemetry matters:

1. Debugging survival failures
   You can go back to tick 1842 and answer:

   - Did it not realise it was starving? (Perception failure)
   - Did it think the fridge was dangerous or pointless? (World Model failure)
   - Did it think someone else needed the fridge more? (Social Model prediction)
   - Did panic fail to trigger? (panic_thresholds mis-set)
   - Did ethics block theft of food? (EthicsFilter doing its job)

2. Teaching
   In class you can say "Here is an actual starvation death. Let's walk the trace and identify which part of the mind failed." That's a lab, not a lecture.

3. Governance
   If an agent does something spicy, you don't get "the AI panicked". You get:

   - At tick 783 it tried STEAL, EthicsFilter vetoed, veto_reason recorded.
   - At tick 784 it called ambulance, panic_reason=health_critical.
   - At tick 785 final_action=call_ambulance, legal, logged.
     All stamped with cognitive_hash. It's admissible, in plain English.

---
