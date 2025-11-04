## 6. Runtime Engine Components

---

Under Townlet v2.5, the old pattern "one giant RL class owns everything" is gone. We replaced it with three core pieces: a factory, a graph agent, and an execution engine.

This is where we guarantee that what we run is what we declared, and what we declared is what we logged, and what we logged is what we can replay.

### 6.1 agent/factory.py

The brain constructor

The factory is the only code pathway allowed to build a live agent.

Inputs:

- the frozen `config_snapshot/` from the run (or from the checkpoint, on resume)

  - cognitive_topology.yaml (Layer 1: behaviour contract / ethics / panic)
  - agent_architecture.yaml (Layer 2: neural blueprints)
  - execution_graph.yaml (Layer 3: think loop spec)
  - universe_as_code.yaml (for observation/action space, affordance definitions, bar layout)
  - config.yaml (runtime envelope like tick rate, curriculum stage, etc)

What factory.py does:

1. Instantiates each cognitive module exactly as described in Layer 2
   For example: it builds the Perception GRU with hidden_dim=512 and Adam lr=1e-4 if that's what's in agent_architecture.yaml. Not "something roughly similar", not "the new default we just pushed to main". Exactly that.

2. Verifies interface contracts
   For example: if `perception_encoder` says it outputs a 128-dim belief vector and `hierarchical_policy` says it expects 128-dim belief input, factory checks that. If they don't match, that's a config error, not "we'll just reshape and hope".

   This matters because interface mismatches are how "quiet hacks" happen in research code. We are refusing to silently broadcast tensors.

3. Injects Layer 1 knobs into runtime modules

   - panic thresholds go into panic_controller
   - ethics rules (forbid_actions, penalize_actions) go into EthicsFilter
   - personality sliders (greed, curiosity, etc) get wired into the hierarchical policy's meta-controller
   - social_model.enabled toggles the Social Model service binding

   This is how we guarantee that what Layer 1 promised ("this agent will never steal", "this agent panics under 15 percent energy") is actually enforced in the live brain.

4. Creates a GraphAgent instance with:

   - a module registry (an nn.ModuleDict or equivalent keyed by name),
   - an executor (the compiled think loop from Layer 3),
   - recurrent / hidden state buffers as per Layer 2.

5. Finalises the cognitive hash
   The moment we have actual modules with actual dims, and the compiled execution graph order, we can compute the full_cognitive_hash. That value is then written to disk for provenance and gets attached to telemetry.

So, in short: factory.py is "build the declared mind; prove it's the declared mind; assign it an identity". After this point, there's no ambiguity about what we're running.

### 6.2 agent/graph_agent.py

The living brain

GraphAgent replaces the old giant RL class. It's the runtime object we actually step every tick.

GraphAgent owns:

- all submodules (perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc) in an internal registry,
- the recurrent / memory state,
- a GraphExecutor that knows how to walk the cognition loop in the right order every tick,
- a simple public API like:

```python
think(raw_observation, prev_recurrent_state)
  -> { final_action, new_recurrent_state }
```

The essential contract with the rest of the simulator is simple: given the latest observation and memory, produce the next action and updated memory. Internally the brain can implement sophisticated planning, simulation, social modelling, panic handling, and ethical vetoes without embedding that logic throughout the environment.

Also important: GraphAgent is always instantiated from the run's frozen snapshot. It never reads "live" configs during execution. This is how we stop "I hotpatched the EthicsFilter in memory for the live demo" type nonsense.

### 6.3 agent/graph_executor.py

The cognition runner (the microkernel of thought)

GraphExecutor is what actually runs the execution_graph.yaml.

At init time:

1. It takes the execution_graph.yaml from the snapshot.

2. It resolves all the symbolic bindings like `"@modules.world_model"` or `"@config.L1.panic_thresholds"` into concrete references.

3. It compiles that into an ordered list of callable steps:

   - run perception
   - unpack belief and recurrent state
   - run hierarchical policy (which itself calls world_model and social_model services)
   - get candidate_action
   - run panic_controller
   - run EthicsFilter
   - output final_action and new_recurrent_state

4. It validates data dependencies. If `panic_controller` expects `candidate_action` and it's not produced by any previous step in the graph, we fail fast. No silent placeholder tensors.

At runtime (each tick):

- GraphExecutor creates a scratchpad (data cache).
- Executes each step in the compiled order, passing along named outputs.
- Emits whatever the graph declared as outputs, typically:

  - `final_action`
  - `new_recurrent_state`
  - plus any debug/telemetry hooks (panic_reason, veto_reason, etc)

Why this matters:

- The execution order is not "whatever the code path happened to be today".
- The execution order is part of the declared cognitive identity and is hashed.
- If someone wants to insert a new veto stage, or let panic bypass ethics, they must edit Layer 3, recompile, and accept a new cognitive hash. The change is governed as well as engineered.

### 6.4 EthicsFilter

The seatbelt

EthicsFilter is a first-class module, not an afterthought.

Inputs per tick:

- the candidate action after panic_controller (which might already be escalated to survival mode),
- the compliance policy from Layer 1 (forbid_actions and penalize_actions),
- optionally current state summary for contextual norms in future extensions.

Outputs per tick:

- final_action (possibly substituted with a safe fallback),
- veto_reason (so telemetry can say "attempted STEAL, blocked by EthicsFilter"),
- ethics_veto_applied flag for the UI.

Important constraints:

- EthicsFilter is last. Panic can override normal planning for survival, but it cannot authorise illegal behaviour. Ethics wins.
- EthicsFilter logs every veto, every tick. Consequently we know not only that it behaved safely, but also when it attempted an unsafe action and was stopped. That is the artefact regulators expect to see.

Later extensions (which we've flagged in open questions) may allow more nuanced compliance rules like "soft penalties if you abuse ambulance when healthy" or "contextual exceptions in extreme survival", but in v2.5 we keep the invariant: panic does not bypass ethics, ethics is final, ethics is logged.

### 6.5 Why These Engine Pieces Exist at All

We split factory / graph_agent / graph_executor for two reasons.

1. Reproducibility and audit

   - factory.py binds "what we said" to "what we built" and gives it an ID.
   - graph_agent.py keeps the running brain honest to that snapshot.
   - graph_executor.py makes the reasoning loop explicit, stable, and hashable.

   This is how we can sit in front of audit and say "here is the mind that ran".

2. Experimental velocity without governance chaos
   Researchers can do surgical edits:

   - change world rules but keep the same brain,
   - change panic thresholds but keep the same world,
   - reorder panic/ethics in the execution graph,
   - swap GRU for LSTM in perception,
   - kill the Social Model and watch social blindness emerge.

  Every one of those changes produces a clean diff in YAML, a new run folder, and a new cognitive hash. The platform therefore supports experimentation while keeping governance fully informed.
