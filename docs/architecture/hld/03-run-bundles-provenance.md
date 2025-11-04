## 3. Run Bundles and Provenance

---

Townlet v2.5 doesn't "run an agent". It mints an artefact with identity, provenance and chain of custody. That's the difference between "cool AI demo" and "system we can take in front of governance without sweating through our shirt".

### 3.1 The Run Bundle

Before a run starts, you prepare a bundle under `configs/<run_name>/`:

```text
configs/
  L99_AusterityNightshift/
    config.yaml                # runtime envelope: tick rate, duration, curriculum, seed
    universe_as_code.yaml      # the world (bars, affordances, prices, cues)
    cognitive_topology.yaml    # BAC Layer 1 (behaviour contract and safety knobs)
    agent_architecture.yaml    # BAC Layer 2 (module blueprints and interfaces)
    execution_graph.yaml       # BAC Layer 3 (think loop + panic/ethics chain)
```

- `universe_as_code.yaml` is the world spec. It defines bars like energy/health/money, affordances like Bed / Job / Hospital / PhoneAmbulance, their per-tick effects and costs, capacity limits, interrupt rules, and any special whitelisted effect (for example `teleport_to:hospital`). It also defines public cues other agents can see ("looks_tired", "bleeding", "panicking").

- The three BAC layers are the mind spec.

- `config.yaml` says how long we run, at what tick rate, with how many agents, and what curriculum (for example: "start alone, introduce a food-scarcity rival after 10k ticks").

This bundle is what we claim we are about to run.

### 3.2 Launching a Run

When we actually launch, we don't execute the live bundle. We snapshot it.

The launcher creates:

```text
runs/
  L99_AusterityNightshift__2025-11-03-12-14-22/
    config_snapshot/
      config.yaml
      universe_as_code.yaml
      cognitive_topology.yaml
      agent_architecture.yaml
      execution_graph.yaml
    checkpoints/
    telemetry/
    logs/
```

Critical details:

- `config_snapshot/` is a byte-for-byte copy of the five YAMLs at launch time. After launch, the live simulator reads only from this snapshot, never from the mutable configuration directory. This prevents untracked hotpatches to ethics during a run.

- We instantiate the agent from that snapshot via the factory. During that process we compile the execution graph (resolving all `@modules.*`, wiring actual module refs, fixing order) and record the resulting ordered cognition loop.

- We compute `full_cognitive_hash.txt` from:

  - the exact text of the five snapshot YAMLs,
  - the compiled execution graph (post-resolution, real step order),
  - the instantiated module architectures (types, hidden dims, optimiser hyperparameters).

That hash is this mind's identity. It's basically "brain fingerprint plus declared world".

- We start ticking. Every tick we log telemetry with:

  - run_id,
  - tick_index,
  - full_cognitive_hash,
  - current_goal (engine truth),
  - agent_claimed_reason (what it says it's doing, if introspection on),
  - panic_state and any panic override,
  - candidate_action,
  - final_action,
  - ethics_veto_applied and veto_reason,
  - planning_depth,
  - social_model.enabled,
  - brief prediction summaries from world_model and social_model.

That is now evidence. If someone later asks "why didn't the agent eat even though it was starving", we don't guess. We read the log.

### 3.3 Checkpoints and Resume

During the run we periodically checkpoint to:

runs/
L99_AusterityNightshift__2025-11-03-12-14-22/
checkpoints/
step_000500/
weights.pt
optimizers.pt
rng_state.json
config_snapshot/
config.yaml
universe_as_code.yaml
cognitive_topology.yaml
agent_architecture.yaml
execution_graph.yaml
full_cognitive_hash.txt

Each checkpoint is effectively "a frozen moment of mind plus world plus RNG". That gives us:

- Honest resume
  To resume, we load from the checkpoint's `config_snapshot/`, not from `configs/`. We write out a new run folder like
  `L99_AusterityNightshift__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/`
  and we recompute the cognitive hash.
  If the snapshot is unchanged, the hash matches and we can legitimately say "this is a continuation of the same mind".
  If we touch anything cognitive or world-rules (panic thresholds, forbid_actions, ambulance cost, bed healing rate, module architecture), the hash changes. That is now a fork, not a continuation. You cannot stealth-edit survival rules and claim it's still the same agent.

- Forensics
  We can go back to tick 842 and reconstruct:

  - what body state it believed it was in,
  - what goal it claimed,
  - whether panic took over,
  - whether EthicsFilter stopped something illegal,
  - and what world rules and costs it was operating under.

- Curriculum / science
  We can diff two runs and say "the only change was that we turned off the Social Model and raised panic aggressiveness; here's how behaviour shifted". It's not anecdote, it's a config diff plus a new hash.

### 3.4 Why Provenance Is Non-Negotiable

Without this provenance model, Townlet would revert to a generic agent-in-a-box demonstration, forcing governance to rely on trust rather than evidence.

With this provenance model:

- We can prove at audit time which ethics rules were live.
- We can prove panic never bypassed ethics unless someone explicitly allowed that in Layer 3 (and if they did, the hash changed).
- We can replay any behaviour clip and show both "what happened" and "which mind, under which declared rules, proposed, attempted, and was vetoed".

This capability enables deployment beyond laboratory settings.

So: Townlet v2.5 == Townlet 1.x post-refactor (the old "Hamlet" era formalised). It's the same agent-in-world system, formally expressed. Universe as Code defines the world. Brain as Code defines the mind. Runs freeze both, hash both, and log both. That is the story everywhere, full stop.

---
