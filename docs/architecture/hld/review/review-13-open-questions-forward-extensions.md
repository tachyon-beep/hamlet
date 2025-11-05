## 13. Open questions and forward extensions

Audience: systems, safety, curriculum, governance, future-you

These are the things we have intentionally not fully solved in Townlet v2.5. They are not bugs. They are decision surfaces. Changing any of these should change the cognitive hash, and in most cases should trigger a formal review.

We track them here to prevent unsanctioned parameter changes in production.

### 13.1 Panic vs ethics escalation

Right now, the execution_graph has a strict override chain:

1. hierarchical_policy proposes `candidate_action`
2. panic_controller may replace it with `panic_adjusted_action` if survival thresholds are breached
3. EthicsFilter receives that and produces `final_action`, with authority to veto

This encodes a hard rule: EthicsFilter beats panic_controller. Panic cannot legalise an illegal act.

Question:
Should panic_controller ever be allowed to bypass EthicsFilter for "survival at all costs" scenarios?

Example scenario:

- Agent has 3 health, 0 money, panic triggers.
- The only survivable move in the world model's projection is "steal_food".
- `steal_food` is in `forbid_actions`.
- EthicsFilter currently vetoes that, which might result in the agent dying.

We currently answer: No. Panic cannot grant an exemption from EthicsFilter. Ethics is final.

We should document that as governance policy. If later someone argues "in extreme medical danger, rules can be broken", that is a policy-level change, not an engineering tweak.

If such an override is ever allowed:

- That change must be expressed declaratively in `cognitive_topology.yaml` (Layer 1), not slipped into code.
- The execution_graph must explicitly reorder or weaken EthicsFilter.
- The hash must change.
- The run_id must change.
- Telemetry must say: `panic_overrode_ethics=true`.

### 13.2 Curriculum pressure vs identity drift

We allow "curriculum" during a run. Curriculum can:

- spawn new instances of existing affordances (for example, add a second 'Bed' affordance of the same type),
- adjust prices and wages over time,
- introduce new NPCs (rivals, dependants),
- scale difficulty by resource scarcity.

We do not allow silent mutation of physics.

Specifically: changing what an affordance actually does (effects_per_tick, costs, teleport destinations, etc) or changing the base world bar dynamics (like "health now decays twice as fast") is not curriculum. That is a world rules patch.

Why that matters:

- Module B (world_model) is learning the world's transition function. If you change the actual physics/affordance semantics mid-run, you have effectively dropped the agent into a new universe without telling audit.
- Telemetry before/after that point is no longer comparable, and your per-tick traces stop being legally useful.

Proposal:

- If you change the definition of any affordance (energy gain, money gain, capacity, interruptibility, teleport behaviour, etc),
  or you change world-level bar rules (base depletion rates, terminal conditions),
  or you introduce a new special `effect_type` handler,
  that is a world fork and must trigger a new run_id and snapshot.

In other words: adding "another Hospital" is allowed mid-run. Changing what "Hospital" does is not allowed mid-run.

Quick reference for policy and ops reviews:

| Change class | Treat as | Required action | Notes |
| --- | --- | --- | --- |
| Spawn additional instances of an existing affordance, tweak operating hours, or adjust wages/prices within declared ranges | Curriculum pressure | Log the change in the run notes/telemetry annotation stream and continue the run | World physics unchanged; run_id stays the same |
| Introduce a brand-new affordance that reuses an existing effect whitelist entry (for example, another `teleport_to:hospital`) | Curriculum pressure (if semantics identical) | Snapshot the updated `universe_as_code.yaml` next run; annotate telemetry | Only safe if the affordance's per-tick effects match an existing template exactly |
| Change per-tick effects, costs, interruptibility, or special handler semantics of an existing affordance | World fork | Cut a new config pack, snapshot, and launch a fresh run_id | Alters physics; must update hash and provenance |
| Add a new `effect_type` handler or modify bar depletion equations | World fork | Governance sign-off, new run folder + snapshot | Expands engine capability; affects all agents' world models |
| Edit cognitive knobs (panic thresholds, greed, social privileges, compliance rules) | Brain fork | Update BAC YAML, snapshot, and start a new run_id | Included here to remind curriculum owners the mind changes too |

### 13.3 Social modelling and privacy constraints

Layer 1 (`cognitive_topology.yaml`) can disable `social_model.enabled`, yielding a socially neutral baseline agent that does not model the intentions of others—useful for ablation studies.

Townlet v2.5 still lacks fine-grained social visibility controls. The social model (Module C) can, in principle, infer goals and predict upcoming actions for every visible agent using public cues and interaction traces.

This presents privacy and ethics challenges in multi-agent simulations.

Open questions:

- Are all agents equally legible, or should certain roles receive additional protection?
- Should some agents be represented only as occupied affordances rather than as intentional minds?
- Should specific inference channels (for example, predicting the next action of a child NPC) be explicitly disallowed?

A likely next step is to introduce privilege scoping in Layer 1, for example:

```yaml
social_model:
  enabled: true
  privilege:
    infer_goals_for:
      - "family"
      - "co_workers"
    forbid_intent_prediction_for:
      - "civilians"
      - "children"
    share_private_channel_with:
      - "family"
```

That implies:

- The social_model_service in the execution_graph must censor its outputs based on these rules before handing them to hierarchical_policy.
- Telemetry must log which privilege policy was in force so later we can prove "the agent could not, by design, predict intent for protected class X".

Layer mapping reminder:

- Layer 1 (`cognitive_topology.yaml`) carries the privilege matrix shown above and the in-group definitions.
- Layer 2 (`agent_architecture.yaml`) is where we would add the actual encoder/decoder blocks for the family channel or disable them entirely.
- Layer 3 (`execution_graph.yaml`) wires the channel outputs into the policy and determines whether EthicsFilter sees the same signal.

Keeping the responsibilities split like this is how we guarantee the hash changes whenever somebody tries to sneak extra bandwidth into the family line.

This becomes essential once you simulate family structures, dependants, or vulnerable cohorts. It's also the place where "are we training stalker behaviour?" gets answered.

### 13.4 Introspection honesty vs engine truth

Layer 1 currently has:

```yaml
introspection:
  visible_in_ui: "research"    # beginner | intermediate | research
  publish_goal_reason: true
```

If `publish_goal_reason` is true, the agent will narrate why it thinks it's doing what it's doing. That narration is exposed in UI as `agent_claimed_reason`, and also logged in telemetry.

The risk is that the claimed reason may not match the actual driver.

Example:

- Engine truth reports `current_goal = SURVIVAL` (panic mode, acquire calories immediately).
- The agent states "I'm working on long-term wealth accumulation."

This could indicate self-deception or deliberate confabulation—both relevant for research and governance.

Mitigation:

- Telemetry must log both:
  - `current_goal` (engine truth from the meta-controller)
  - `agent_claimed_reason` (self-report, if enabled)
- The Run Context Panel must display the two fields separately.

This enables instructors to demonstrate rationalisation and allows auditors to show that the system knew it was in SURVIVAL mode regardless of the agent's narrative.

Future extension: consider adding a "self-awareness / alignment" probe that measures the divergence between engine truth and self-claim as an explicit safety metric.

### 13.5 Compliance richness and context-sensitive ethics

Compliance in Layer 1 today has two tools:

- `forbid_actions`: hard veto, cannot be done ever
- `penalize_actions`: soft discourage, apply shaping cost or mark for review

This works for "don't steal" and "shove is rude". It does not cover case-by-case rules.

Examples we can't yet represent cleanly:

- "Steal is forbidden unless the target is flagged as `abandoned_property`."
- "Calling ambulance while health is fine and mood is fine is considered misuse of emergency services; apply a social penalty instead of banning it outright."
- "Pushing a stranger is forbidden, but pushing a family member out of danger is allowed."

We will need structured compliance rules, not just hard lists.

Two viable approaches:

Option A: enrich `penalize_actions` with conditional logic expressed as a tiny declarative DSL (similar to goal termination):

```yaml
penalize_actions:
  - action: "call_ambulance"
    if:
      all:
        - { bar: "health", op: ">=", val: 0.7 }
        - { bar: "mood", op: ">=", val: 0.8 }
    penalty: -5.0
    note: "unjustified emergency use"
```

Option B: introduce a dedicated `compliance_rules:` block in `cognitive_topology.yaml` with a micro-DSL that can express:

- situational bans (forbid if condition holds),
- contextual penalties,
- social exemptions (family vs non-family),
- mandatory de-escalation steps.

Whichever path we pick:

- It must still be declarative YAML.
- It must still be inspected, copied into snapshots, and hashed.
- EthicsFilter must evaluate it inside the execution_graph, not in opaque Python.
- Telemetry must log which rule fired, by name, each time it blocks or penalises an action.

Without declarative rules, ethical constraints will inevitably drift back into bespoke Python conditionals, undermining auditability.

### 13.6 World-side special effects

Universe as Code (UAC) is intentionally declarative. Affordances like "bed", "job", "hospital", "phone_ambulance" are just config objects with:

- capacity / exclusivity
- interruptibility
- per-tick bar deltas
- costs
- optional special handlers like `teleport`

We already limit special handlers to a small whitelist (for example `teleport`, later possibly `kill`, `spawn_family_member`, etc), each implemented once in the engine.

Two policy issues are still open:

1. Teleport semantics
   Teleport confers map control and safety bypass. Allowing an agent to call an ambulance from anywhere and teleport to hospital can be an intentional survival path, but it must not arise accidentally. Any new `effect_type` added to the whitelist should therefore be treated as a governance event, require an updated hash, and undergo review.

2. Claim semantics for capacity / reservation
   At present, use of an affordance (for example, a bed) is modelled as an ephemeral reservation with deterministic tie-breaking. YAML does not encode ownership, which avoids hidden state that the world model cannot learn. Introducing persistent ownership—for example, "this is my bed now"—would constitute a change to world rules and must trigger a new run. Otherwise decisions would depend on invisible state not reflected in `universe_as_code.yaml`.

---

### 13.7 Hash boundaries and "who signs off on identity"

Right now, `cognitive_hash.txt` covers:

- all five YAMLs in the snapshot,
- the compiled execution_graph wiring,
- the instantiated module architectures and optimiser LRs.

Deliberate exclusions:

- live weights
- optimiser momentum state
- RNG state
- current recurrent memory state

This is on purpose. The cognitive hash is meant to identify the mind design, not the momentary thoughts. Two questions remain unresolved:

1. Should panic thresholds, greed, social_model.enabled, etc be allowed to change mid-run via curriculum?
   Answer today: no. Changing cognitive knobs changes the brain. That must fork the run. That implies new hash, new run_id.

2. Should world-level curriculum adjustments (prices, wages, availability schedules) be allowed mid-run without forking?
   Answer today: yes, as long as we haven't changed affordance semantics or bar physics. But we should log curriculum events in telemetry so we can correlate behaviour shifts with world stressors.

Responsibility for "signing off on identity" sits with governance, not engineering. The compliance policy team must approve any change that alters the cognitive-hash rules. This becomes a formal control point.

---

### 13.8 Alignment between stated goal and actual goal

We are now explicitly modelling two goal channels:

- `current_goal` (engine truth from the meta-controller / hierarchical_policy)
- `agent_claimed_reason` (self-narrative string/label)

We should measure the divergence between these channels regularly. A persistent gap indicates that:

- the agent is strategically pursuing one objective,
- while narrating another (to humans or to itself).

Such divergence may reflect deception, confusion, or deliberate messaging—all areas that warrant scrutiny.

Future extension:

- define a "truthfulness index" per run (the fraction of ticks where `agent_claimed_reason` semantically matches `current_goal`);
- log the index in telemetry summaries;
- surface it in replay tooling so instructors can identify strongly self-justifying agents before classroom use.

---

### 13.9 Family channel, coordination, and collusion

The Personality & Family extension plans to give related agents:

- a heritable personality vector (greed, neuroticism, curiosity, etc),
- a shared `family_id`,
- and a private low-bandwidth signalling channel (SetCommChannel(int64)).

That channel is visible only to in-group members as an additional input, and those agents may learn to ground semantics in it (for example, "123" might come to mean "I've found money").

Open questions:

- Is in-group signalling governed by the `social_model.enabled` switch, or should it be managed separately?
- Could the private channel be used to coordinate behaviour that violates global norms in ways that EthicsFilter cannot observe—for example, two related agents colluding to starve out a third?
- Do we need to audit those communications for governance purposes, and if so how do we do that without eliminating the research value of emergent communication?

At minimum:

- The presence of any family or in-group channel must be declared in `cognitive_topology.yaml` and must feed into the hash.
- Telemetry must log that channel activity occurred, even if the content is not decoded.
- If policy logic treats "family" differently in EthicsFilter (for example, allowing a shove to move a family member out of danger while forbidding shoving strangers), that policy must be expressed declaratively and must change the hash.

---

### 13.10 Red lines we are choosing to keep

We record a set of non-negotiable invariants:

1. EthicsFilter must always execute after panic_controller within the execution graph. Moving or removing it produces a different class of agent and requires a new hash.

2. Hotpatching EthicsFilter or panic_controller at runtime is prohibited. Adjustments such as "raise the panic threshold for more urgency" constitute a new mind and require a fork.

3. Universe as Code cannot be altered mid-run in ways that change affordance semantics or bar physics. Curriculum may introduce pressure, but it may not rewrite reality without issuing a new run identifier.

4. Every reasoning step that can affect `final_action` must appear in `execution_graph.yaml`. Hidden side channels constitute a design violation.

These are governance-grade invariants. Violating them eliminates auditability and reproducibility, returning the system to a black-box state.
