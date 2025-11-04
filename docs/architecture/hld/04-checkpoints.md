## 4. Checkpoints

---

A checkpoint is not "saved weights lol". It's a frozen moment of a specific mind, in a specific world, under specific rules, at a specific instant in time.

Townlet treats every checkpoint as evidence. A checkpoint must include everything required to:

- pick up training honestly,
- replay behaviour honestly,
- and prove, later, which exact cognitive configuration produced which exact action.

When we write a checkpoint for a run, we create something like:

```text
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
```

Let's unpack what those pieces actually mean.

### 4.1 weights.pt

This is the live neural state of the brain at that tick:

- perception module weights
- world_model weights
- social_model weights
- hierarchical_policy weights
- panic_controller weights (if it's learned)
- EthicsFilter weights (if it's learned / parameterised)
- anything else registered in the agent module registry

In v1 these components all lived in one giant black-box DQN. In Townlet, they are the submodules declared in Layer 2 (`agent_architecture.yaml`) and wired by Layer 3 (`execution_graph.yaml`). We save them together because, for audit, "the brain" encompasses the entire SDA module set, not only the action head.

### 4.2 optimizers.pt

We log both parameters and optimiser state (for example, Adam moments) for each trainable module.

Why? Because "resume training" must mean "continue the same mind's learning process", not "respawn something with the same weights but different momentum and call it continuous". If you've ever done RL you know that quietly dropping optimiser state can absolutely change learning behaviour. We are not pretending that's irrelevant. We store it.

### 4.3 rng_state.json

Randomness is part of causality.

We store the RNG states that matter:

- environment RNG,
- agent RNG (PyTorch generators etc),
- anything else that would affect rollout sampling, tie-breaks in affordance contention, exploration noise, etc.

This allows us to re-run tick 501 and observe the same stochastic outcomes. When someone asks, "would it always have chosen STEAL here?" we can answer, "under this exact random sequence, here is what occurred," and reproduce the evidence without speculation.

### 4.4 config_snapshot/

This is critical.

Inside every checkpoint, we embed a fresh copy of the exact `config_snapshot/` that the run is using at that moment. That snapshot contains:

- `config.yaml` (runtime envelope: tick rate, max ticks, curriculum step, etc)
- `universe_as_code.yaml` (the world: meters, affordances, costs, social cues, ambulance behaviour, bed quality, etc)
- `cognitive_topology.yaml` (Layer 1, the behaviour contract: panic thresholds, ethics rules, greed, etc)
- `agent_architecture.yaml` (Layer 2, the blueprint: module shapes, learning rates, pretraining origins, interface dims)
- `execution_graph.yaml` (Layer 3, the think loop: who runs first, who can override whom, and in what order ethics and panic fire)

This is not a pointer. It's an embedded copy at that checkpoint tick.

Why embed it every time? Because curriculum might change some parts of the world over time (for example: add new competition, raise prices, close the hospital at night). If that's allowed under policy, those changes will appear in `universe_as_code.yaml` at tick 10,000 that didn't exist at tick 500. Checkpoint 500 needs to show what the world rules were then, not now.

Also: "panic thresholds" and "forbid_actions" in cognitive_topology.yaml are part of that snapshot. So when someone asks "did you allow it to steal at tick 842", we don't argue philosophy. We open the checkpoint around that time and read the file.

### 4.5 full_cognitive_hash.txt

This is the mind's ID badge.

The hash is deterministic over:

1. The exact text bytes of the 5 YAMLs in the snapshot (config.yaml, universe_as_code.yaml, cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml).
2. The compiled execution graph after resolution. Not the pretty YAML, but the actual ordered list of steps the agent is running after we bind them to modules. So if someone sneaks in "panic after ethics" instead of "panic before ethics", the hash changes.
3. The constructed module architectures. Types, hidden sizes, optimiser settings, interface dims. Not just "GRU exists", but "GRU with hidden_dim=512 paired with Adam lr=1e-4".

That means:

- If you fiddle the EthicsFilter to quietly allow STEAL under panic, hash changes.
- If you widen the GRU and try to pretend it's the same mind, hash changes.
- If you reduce ambulance cost in the world, hash changes (because universe_as_code.yaml changed).

We're basically tattooing "this exact mind in this exact world with this exact cognition loop" into the checkpoint.

### 4.6 Why Checkpoints Are Legally Interesting (Not Just Technically Interesting)

Because they kill plausible deniability.

If someone claims:

- "oh, it only stole because it was desperate"
  or
- "ethics must have bugged out at 2am"
  or
- "we didn't change anything important, we just tuned panic a little"

you can respond with:

- "here's the checkpoint; panic thresholds are documented; ethics still forbids STEAL; hash says it's the same mind before and after 2am; so no, it wasn't allowed to steal, it attempted to anyway and EthicsFilter vetoed it".

In other words, checkpoints turn anecdotes about behaviour into evidence trails.

---
