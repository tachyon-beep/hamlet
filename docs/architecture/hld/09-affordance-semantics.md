## 9. Affordance Semantics in universe_as_code.yaml

---

Universe as Code is the other half of this story. Brain as Code (Layers 1–3) defines the mind. Universe as Code defines the body and the town.

Townlet avoids hardcoded rules such as "beds make you rested" embedded throughout the Python code. The world is declared as affordances with effects on bars. Beds, jobs, phones, ambulances, hospitals, fridges, and pubs are entries in the world configuration.

### 9.1 Affordances Are Declarative

Each actionable thing in the world (Bed, Job, Fridge, Hospital, Phone_Ambulance, etc) is defined in `universe_as_code.yaml` like so:

```yaml
- id: "bed_basic"
  quality: 1.0              # scales how effective the rest is
  capacity: 1               # how many agents can use it this tick
  exclusive: true           # if true, only one occupant at a time
  interaction_type: "multi_tick"
  interruptible: true       # can be abandoned mid-sleep
  distance_limit: 0         # must be on the tile
  costs:
    - { bar: "money", change: -0.05 }     # pay rent to crash here
  effects_per_tick:
    - { bar: "energy", change: +0.25, scale_by: "quality" }

  on_interrupt:
    refund_fraction: 0.0    # optional semantics for partial usage
    note: "no refund if you bail early"
```

…and a more "special" affordance like an ambulance call:

```yaml
- id: "phone_ambulance"
  interaction_type: "instant"
  distance_limit: 1
  costs:
    - { bar: "money", change: -3.00 }     # normalised cost (e.g. $300)
  effects:
    - { effect_type: "teleport",
        destination_tag: "nearest_hospital",
        precondition: { bar: "health", op: "<=", val: 0.2 } }
```

There are a few important things to notice:

- Everything is in terms of bars and per-tick deltas.
  Bed raises energy every tick, costs a bit of money, maybe hurts mood if it's gross, etc.

- capacity + exclusive let us model contention.
  Two agents can't both occupy a single-occupancy bed with capacity:1, exclusive:true. The engine will arbitrate who "wins" this tick in a deterministic way.

- interaction_type captures temporal shape.
  `multi_tick` means "stay here over multiple ticks and accumulate effects_per_tick".
  `instant` means "one-shot action now" (like calling ambulance).

- Special abilities (teleport etc) are referenced by name, not implemented ad hoc in YAML.
  The YAML is only allowed to invoke a small whitelist of engine-side effect handlers (teleport, etc). That keeps the world spec expressive but bounded. You don't get "nuke_city:true".

### 9.2 Engine Semantics (How the Runtime Interprets Affordances)

To keep the world deterministic, replayable, and trainable-for-World-Model, the engine follows strict rules:

1. Reservation
   When an agent tries to use an affordance, the engine does a local "reservation" check:

   - Is capacity available?
   - Are preconditions met (health low enough, money high enough, distance within limit)?
   - If yes, it assigns a reservation token to that agent for that tick.

   This reservation is not global mutable lore. It's per-tick, ephemeral.
   We don't create long-lived "ownership" state in random engine globals because that explodes complexity and makes the World Model's job harder.

2. Contention resolution
   If multiple agents want the same affordance and capacity is exceeded, break ties deterministically. For example: sort by distance, then by agent_id.
   Determinism matters because we want to replay the run exactly and train the World Model on consistent consequences.

3. Effects application
   Once reservations are resolved, all costs and effects_per_tick for all active affordances are collected, summed (per agent), and atomically applied to bars (energy, health, money, etc).
   Then we clamp bars to [0.0, 1.0] or whatever the world defines.

   Key point: we don't partially apply effects from some affordances and then let those partial updates influence others in the same tick. We apply atomically at the end of the tick. This gives clean training data.

4. Interrupts
   If `interruptible: true` and the agent walks off or is forced to bail (panic_controller might decide "leave bed now and call ambulance"), we stop applying future per-tick effects.
   `on_interrupt` can define whether you get any partial benefit or refund. That's still declarative.

5. Special effects whitelist
   YAML is allowed to reference a small set of named effect_type operations (like teleport), and the engine implements those centrally.
   That way, "teleport to nearest_hospital" is a normal, auditable world affordance, not a custom 'if agent.health < X then hack position'.

   This whitelist is versioned. If you add a new special effect, you're extending world semantics globally and that should change the hash once it's applied to a snapshot.

### 9.3 Why Universe as Code Matters for BAC

Universe as Code (UAC) and Brain as Code (BAC) are two halves of the same sentence:

- UAC: the world, bodies, bars, affordances, economy, social cues, ambulance rules, etc, are all declared in YAML.
  They are diffable. They are teachable. They are inspectable by non-coders.

- BAC: the mind, panic thresholds, ethics vetoes, planning depth, social reasoning, module architectures, and actual cognition loop are also declared in YAML.
  They are diffable. They are teachable. They are inspectable by non-coders.

When you run a simulation, Townlet snapshots both halves into a run folder, stamps them with a cognitive hash, and then logs decisions per tick against that identity.

So instead of "the AI did something weird overnight and now it's different", we can say:

- "At tick 842, Mind 4f9a7c21, in World Nightshift_v3 with ambulance_cost $300 and bed_quality 1.0, entered panic because health < 0.25.
  Panic escalated the action to call_ambulance.
  EthicsFilter allowed it.
  Money was deducted.
  Agent teleported to the nearest hospital affordance.
  See veto_reason for evidence that it also tried to STEAL food two ticks earlier and that was blocked."

That is the moment where governance stops being hypothetical and becomes screenshot material.

And that's the point of Townlet: it's not a toy black box any more. It's an accountable simulated society with auditable minds.
