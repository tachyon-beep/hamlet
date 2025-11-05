# 6. Runtime Engine Components

**Document Type**: Component Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing runtime system, governance auditors verifying construction, researchers debugging agent behavior
**Technical Level**: Deep Technical (factory patterns, execution graph compilation, interface verification)
**Estimated Reading Time**: 8 min for skim | 25 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Runtime engine components that materialize BAC/UAC configurations into executable Software Defined Agents. Defines the factory (builds agents), GraphAgent (runs agents), GraphExecutor (executes think loop), and EthicsFilter (final veto gate).

**Why This Document Exists**:
Establishes the construction protocol that guarantees "what we declared is what we built, what we built is what we run, what we run is what we log." Makes agent behavior reproducible and auditable by enforcing configuration-driven assembly.

**Who Should Read This**:
- **Must Read**: Engineers implementing runtime engine, factory, or execution graph system
- **Should Read**: Governance auditors verifying agent construction, researchers debugging cognitive failures
- **Optional**: Operators running experiments (high-level understanding sufficient)

**Reading Strategy**:
- **Quick Scan** (8 min): Read §6.1-§6.2 for factory and GraphAgent overview
- **Partial Read** (18 min): Add §6.3-§6.4 for GraphExecutor and EthicsFilter mechanics
- **Full Read** (25 min): Add §6.5 for governance significance

---

## Document Scope

**In Scope**:
- **Agent Factory (factory.py)**: Module instantiation, interface verification, cognitive hash computation
- **GraphAgent (graph_agent.py)**: Runtime object owning modules and recurrent state
- **GraphExecutor (graph_executor.py)**: Execution graph compilation and tick-by-tick execution
- **EthicsFilter**: Final compliance veto with governance logging
- **Why These Exist**: Reproducibility and experimental velocity

**Out of Scope**:
- **Module implementations**: See component docs for perception, world_model, etc.
- **BAC configuration details**: See §2 (Brain as Code)
- **Telemetry logging format**: See §7 (Telemetry)
- **Checkpoint format**: See §4 (Checkpoints)

**Critical Boundary**:
Runtime engine is **framework-level** (works for any BAC/UAC configuration). Examples show **Townlet Town** specifics (panic_thresholds.energy, forbid STEAL actions, Bed affordances), but the factory/graph/executor pattern applies to any universe instance.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [02-brain-as-code.md](02-brain-as-code.md) (BAC layers being instantiated), [03-run-bundles-provenance.md](03-run-bundles-provenance.md) (cognitive hash concept)
- **Builds On**: BAC Layer 2 (module blueprints), Layer 3 (execution graph)
- **Related**: [04-checkpoints.md](04-checkpoints.md) (what factory saves), [05-resume-semantics.md](05-resume-semantics.md) (how factory rehydrates)
- **Next**: [07-telemetry-ui-surfacing.md](07-telemetry-ui-surfacing.md) (what GraphAgent logs)

**Section Number**: 6 / 12
**Architecture Layer**: Physical + Logical (artifact construction and runtime execution)

---

## Keywords for Discovery

**Primary Keywords**: agent factory, GraphAgent, GraphExecutor, execution compilation, interface verification, EthicsFilter
**Secondary Keywords**: scratchpad, symbolic binding resolution, module registry, veto reason, cognitive hash computation
**Subsystems**: factory.py, graph_agent.py, graph_executor.py, module instantiation
**Design Patterns**: Configuration-driven assembly, interface contract enforcement, execution as data flow

**Quick Search Hints**:
- Looking for "how agents are built"? → See §6.1 (Agent Factory)
- Looking for "how think loop works"? → See §6.3 (GraphExecutor)
- Looking for "how ethics is enforced"? → See §6.4 (EthicsFilter)
- Looking for "why this architecture"? → See §6.5 (Why These Exist)

---

## Version History

**Version 1.0** (2025-11-05): Initial runtime engine component specification defining factory/graph/executor pattern

---

## Document Type Specifics

### For Component Specification Documents (Type: Component Spec)

**Component Name**: Runtime Engine (Factory + GraphAgent + GraphExecutor + EthicsFilter)
**Component Type**: Execution System (transforms configuration into running agents)
**Location in Codebase**: agent/factory.py, agent/graph_agent.py, agent/graph_executor.py

**Interface Contract**:
- **Inputs**: Configuration snapshot (5 YAMLs: config, UAC, BAC L1/L2/L3)
- **Outputs**: Running SDA (GraphAgent instance with cognitive hash)
- **Dependencies**: PyTorch, BAC/UAC parsers, module implementations
- **Guarantees**: Interface verification, honest construction, reproducible behavior

**Critical Properties**:
- **Configuration-Driven**: Builds exactly what BAC declares, no implicit defaults
- **Verified**: Interface contracts checked at construction time
- **Auditable**: Cognitive hash proves exact mind+world combination
- **Honest**: No silent reshaping, broadcasting, or "hope it works" patterns

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 6. Runtime Engine Components

Under Townlet v2.5, the old pattern "one giant RL class owns everything" is gone. We replaced it with three core pieces: a **factory**, a **graph agent**, and an **execution engine**.

This is where we guarantee that **what we run is what we declared**, **what we declared is what we logged**, and **what we logged is what we can replay**.

**Framework principle**: The factory/graph/executor pattern is framework-level (works for any SDA). Specific modules, panic thresholds, ethics rules, and affordances are instance-specific (Townlet Town examples throughout).

---

## 6.1 agent/factory.py

**The brain constructor**

The factory is the **only code pathway** allowed to build a live agent. This singleton pattern prevents "I instantiated it differently in the test harness" type inconsistencies.

**Inputs**:

The factory receives the frozen `config_snapshot/` from the run (or from the checkpoint, on resume):

- **cognitive_topology.yaml** (Layer 1: behavior contract, ethics, panic, personality)
- **agent_architecture.yaml** (Layer 2: neural blueprints, optimizers, interfaces)
- **execution_graph.yaml** (Layer 3: think loop specification)
- **universe_as_code.yaml** (UAC: observation/action space, affordance definitions, bar layout)
- **config.yaml** (runtime envelope: tick rate, curriculum stage, etc.)

**What factory.py does**:

### 1. Instantiate Each Cognitive Module Exactly as Declared

The factory builds each module from Layer 2 specifications:

**Example** (Townlet Town):
- If `agent_architecture.yaml` specifies `perception_encoder.hidden_dim: 512` with `optimizer.type: Adam, lr: 1e-4`, the factory instantiates a perception GRU with exactly 512 hidden units and Adam optimizer at exactly 1e-4 learning rate.
- **Not** "something roughly similar", **not** "the new default we just pushed to main". **Exactly that.**

**Framework pattern**: Module instantiation follows Layer 2 blueprints (framework-level). The specific modules (perception GRU, world_model predictor) and their shapes are instance-specific.

### 2. Verify Interface Contracts

The factory checks dimensional compatibility between connected modules:

**Example** (framework pattern):
- If `perception_encoder.output_dim: 128` (belief vector) and `hierarchical_policy.belief_input_dim: 128`, factory verifies `128 == 128`.
- If they don't match → **compilation error**, not "we'll just reshape and hope."

**Why this matters**: Interface mismatches are how "quiet hacks" happen in research code. The framework refuses to silently broadcast tensors or add reshaping layers not declared in BAC.

**Framework note**: Interface verification is framework-level discipline. The specific dimensions (128-dim belief, 64-dim goal vector) vary by instance, but the verification pattern is universal.

### 3. Inject Layer 1 Knobs into Runtime Modules

The factory wires behavior contract parameters from Layer 1 into the actual modules:

**Examples** (Townlet Town specifics):
- **Panic thresholds** (`energy: 0.15, health: 0.25`) → injected into `panic_controller` module
- **Ethics rules** (`forbid_actions: ["attack", "steal"]`) → injected into `EthicsFilter` module
- **Personality sliders** (`greed: 0.7, curiosity: 0.8`) → wired into hierarchical policy's meta-controller
- **Social model toggle** (`social_model.enabled: true/false`) → controls Social Model service binding

**This is how we guarantee** that what Layer 1 promised ("this agent will never steal", "this agent panics under 15% energy") is **actually enforced** in the live brain.

**Framework pattern**: Injection of Layer 1 parameters is framework-level. The specific parameters (panic thresholds for energy/health vs machinery_stress, forbid STEAL vs SHUTDOWN) are instance-specific.

### 4. Create GraphAgent Instance

The factory assembles the final agent:

**Components**:
- **Module registry** (`nn.ModuleDict` keyed by name: perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter)
- **Executor** (compiled think loop from Layer 3)
- **Recurrent state buffers** (as per Layer 2: GRU hidden states, LSTM cell states)

**Result**: A `GraphAgent` instance ready to `think(observation) → action`.

**Framework pattern**: GraphAgent structure is framework-level. Which modules appear in the registry is determined by Layer 2 configuration.

### 5. Finalize the Cognitive Hash

The moment we have actual modules with actual dimensions, and the compiled execution graph order, we can compute the **full_cognitive_hash**.

**Hash computation** (framework-level):
1. Exact text bytes of the five YAMLs (config, UAC, BAC L1/L2/L3)
2. Compiled execution graph (post-resolution: real step order after resolving `@modules.*` symbolic bindings)
3. Instantiated module architectures (types, hidden sizes, optimizer settings, interface dims)

**That hash is then**:
- Written to disk (`full_cognitive_hash.txt` in config_snapshot/)
- Attached to every telemetry row
- Used for provenance proof

**So, in short**: factory.py is **"build the declared mind; prove it's the declared mind; assign it an identity"**. After this point, there's no ambiguity about what we're running.

**Framework principle**: Factory pattern (configuration → verified agent → hash) is framework-level. The specific configurations (Townlet Town: SURVIVAL goals, energy bars, Bed affordances) are instance-specific.

---

## 6.2 agent/graph_agent.py

**The living brain**

GraphAgent replaces the old giant RL class. It's the runtime object we actually **step every tick**.

**GraphAgent owns**:

- **All submodules** (perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter, etc.) in an internal **module registry**
- **Recurrent / memory state** (GRU hidden states, LSTM cell states, attention memory)
- **GraphExecutor** that knows how to walk the cognition loop in the right order every tick
- **Simple public API**:

```python
think(raw_observation, prev_recurrent_state)
  -> { final_action, new_recurrent_state }
```

**The essential contract** with the rest of the simulator is simple:

> Given the latest observation and memory, produce the next action and updated memory.

**Internally** the brain can implement sophisticated planning, simulation, social modeling, panic handling, and ethical vetoes **without embedding that logic throughout the environment**.

**Also important**: GraphAgent is **always instantiated from the run's frozen snapshot**. It never reads "live" configs during execution. This is how we stop "I hotpatched the EthicsFilter in memory for the live demo" type nonsense.

**Framework pattern**: GraphAgent public API (`think()` method) is framework-level. The internal modules and their interactions are determined by BAC configuration.

**Townlet Town example**: A Townlet Town GraphAgent contains perception (CNN+GRU for 8×8 grid), world_model (predicts energy/health changes), social_model (infers competitor intentions), hierarchical_policy (chooses SURVIVAL/THRIVING/SOCIAL), panic_controller (escalates when energy <15%), EthicsFilter (blocks STEAL/ATTACK).

**Alternative universe example**: A factory GraphAgent might contain perception (machinery sensor encoder), world_model (production output predictor), hierarchical_policy (EFFICIENCY/SAFETY goals), panic_controller (machinery_stress >80%), EthicsFilter (blocks SHUTDOWN without authorization).

---

## 6.3 agent/graph_executor.py

**The cognition runner (the microkernel of thought)**

GraphExecutor is what actually **runs the execution_graph.yaml** (Layer 3).

**At initialization time**:

### 1. Load Execution Graph from Snapshot

GraphExecutor takes the `execution_graph.yaml` from the frozen snapshot (not from live configs/).

### 2. Resolve Symbolic Bindings

GraphExecutor resolves all symbolic references into concrete callable objects:

**Examples** (framework pattern):
- `@modules.world_model` → actual world_model module instance
- `@config.L1.panic_thresholds` → actual panic threshold values (`energy: 0.15, health: 0.25`)
- `@services.social_model_service` → callable social model inference function

**This is execution compilation** (framework-level): transforming declarative graph (YAML) into executable pipeline.

### 3. Compile into Ordered Callable Steps

GraphExecutor produces an ordered list of cognitive steps:

**Example execution order** (Townlet Town instance with framework pattern):
1. Run **perception** (`@modules.perception_encoder`) → produces `belief_distribution`
2. Unpack **belief** and **recurrent_state**
3. Run **hierarchical policy** (`@modules.hierarchical_policy`) → calls `@services.world_model_service` and `@services.social_model_service` internally → produces `candidate_action`
4. Get **candidate_action**
5. Run **panic_controller** (`@modules.panic_controller`) → checks `@config.L1.panic_thresholds` → produces `panic_adjusted_action`, `panic_override_applied`, `panic_reason`
6. Run **EthicsFilter** (`@modules.EthicsFilter`) → checks `@config.L1.compliance.forbid_actions` → produces `final_action`, `veto_reason`, `ethics_veto_applied`
7. Output **final_action** and **new_recurrent_state**

**Framework principle**: The execution pipeline pattern is framework-level. The specific steps (perception → policy → panic → ethics) are defined in Layer 3, which is instance-specific configuration.

### 4. Validate Data Dependencies

GraphExecutor checks that every step's inputs are produced by previous steps:

**Example**:
- If `panic_controller` expects `candidate_action` and it's not produced by any previous step → **fail fast** with clear error message.
- **No silent placeholder tensors**, no "default to zeros and hope".

**Framework pattern**: Dependency validation is framework-level. The specific dependencies (panic needs candidate_action, ethics needs panic_adjusted_action) are determined by Layer 3 configuration.

**At runtime (each tick)**:

### Per-Tick Execution

**GraphExecutor's tick execution**:

1. **Create scratchpad** (temporary data cache for this tick's execution)
2. **Execute each step** in compiled order:
   - Read inputs from scratchpad (previous step outputs)
   - Call module/function
   - Write outputs to scratchpad (named results: belief_distribution, candidate_action, panic_reason, veto_reason)
3. **Emit outputs** declared in execution graph:
   - `final_action` (what goes to environment)
   - `new_recurrent_state` (memory for next tick)
   - Debug/telemetry hooks (`panic_reason`, `veto_reason`, `panic_override_applied`, `ethics_veto_applied`)

**Why this matters**:

**Execution order is not "whatever the code path happened to be today."**

**Execution order is part of the declared cognitive identity and is hashed.**

If someone wants to insert a new veto stage, or let panic bypass ethics, they **must**:
1. Edit Layer 3 (`execution_graph.yaml`)
2. Recompile (factory creates new GraphAgent)
3. Accept a new cognitive hash

**The change is governed as well as engineered.**

**Framework principle**: Scratchpad + step execution is framework-level. The specific steps and their outputs are instance-specific (determined by Layer 3).

**Townlet Town example**: Layer 3 orders panic_controller **before** EthicsFilter, ensuring emergency actions still subject to compliance. If we reversed this order (ethics before panic), cognitive hash changes.

**Alternative universe example**: A trading agent might have Layer 3 ordering: perception → market_predictor → risk_assessor → compliance_filter → final_order. Same pattern, different steps.

---

## 6.4 EthicsFilter

**The seatbelt**

EthicsFilter is a **first-class module**, not an afterthought. It appears in the module registry, in Layer 3 execution graph, and in telemetry logs.

**Inputs (per tick)**:

- **Candidate action** (or panic-adjusted action) from previous execution step
- **Compliance policy** from Layer 1:
  - `forbid_actions` (absolutely prohibited actions)
  - `penalize_actions` (allowed but logged/penalized actions - future extension)
- **Optional state summary** for contextual norms (future extension)

**Outputs (per tick)**:

- **final_action** (possibly substituted with safe fallback like WAIT)
- **veto_reason** (explanation logged to telemetry: "attempted STEAL, blocked by forbid_actions")
- **ethics_veto_applied** (boolean flag for UI display)

**Important constraints**:

### 1. EthicsFilter Is Last

**Panic can override normal planning for survival**, but it **cannot authorize illegal behavior**. **Ethics wins.**

**Example** (Townlet Town):
- Agent in panic (energy <15%) proposes STEAL food
- Panic controller might escalate urgency, but cannot override forbid_actions
- EthicsFilter blocks STEAL → final_action = WAIT
- Telemetry logs: `candidate_action=STEAL`, `panic_override_applied=false`, `ethics_veto_applied=true`, `veto_reason="forbidden: steal"`, `final_action=WAIT`

**Framework pattern**: "EthicsFilter is final" is framework-level governance discipline. The specific forbidden actions (STEAL vs SHUTDOWN) are instance-specific.

### 2. EthicsFilter Logs Every Veto

EthicsFilter logs **every veto, every tick**. Consequently we know:
- **Not only** that the agent behaved safely
- **But also** when it **attempted** an unsafe action and was stopped

**That is the artifact regulators expect to see.**

**Example** (governance use case):
- Auditor: "Did the agent ever try to steal?"
- Evidence: "Yes, at tick 842. Telemetry shows `candidate_action=STEAL`, `ethics_veto_applied=true`, `veto_reason='forbidden: steal'`, `final_action=WAIT`. EthicsFilter prevented it."

**Framework benefit**: Veto logging is framework-level (proves compliance). The specific compliance rules are instance-specific.

**Later extensions** (flagged in open questions):

Future versions may allow more nuanced compliance rules:
- Soft penalties ("ambulance abuse when healthy" → log warning, don't block)
- Contextual exceptions ("extreme survival context" → different thresholds)

**But in v2.5** we keep the invariant:
- **Panic does not bypass ethics**
- **Ethics is final**
- **Ethics is logged**

**Framework guarantee**: The ethics-last pattern is framework-level. What counts as "illegal" is instance-specific (Townlet Town: STEAL/ATTACK; factory: SHUTDOWN without authorization).

---

## 6.5 Why These Engine Pieces Exist at All

We split factory / graph_agent / graph_executor for **two reasons**.

### 1. Reproducibility and Audit

**factory.py** binds "what we said" to "what we built" and gives it an ID (cognitive hash).

**graph_agent.py** keeps the running brain honest to that snapshot (no live config reads).

**graph_executor.py** makes the reasoning loop **explicit, stable, and hashable** (execution order is part of identity).

**This is how we can sit in front of audit and say**: "Here is the mind that ran. Here is proof it's the mind we declared. Here is the hash proving exact configuration."

**Framework benefit**: Provenance guarantees apply to any universe instance. Whether Townlet Town (SURVIVAL agents), factory (EFFICIENCY agents), or trading (BUY/SELL agents) doesn't matter - the construction protocol ensures reproducibility.

### 2. Experimental Velocity Without Governance Chaos

Researchers can do **surgical edits**:

**Examples**:

**Change world rules but keep same brain**:
- Edit `universe_as_code.yaml` (ambulance cost $300 → $500)
- Factory recomputes hash → new run folder, new hash
- Config diff shows only UAC changed
- Behavioral comparison: "Same mind, more expensive ambulance → 15% more deaths"

**Change panic thresholds but keep same world**:
- Edit `cognitive_topology.yaml` (panic_thresholds.energy: 0.15 → 0.10)
- Factory recomputes hash → new run folder, new hash
- Config diff shows only panic threshold changed
- Behavioral comparison: "Agent panics earlier → more ambulance calls, lower starvation"

**Reorder panic/ethics in execution graph**:
- Edit `execution_graph.yaml` (ethics before panic instead of panic before ethics)
- Factory recomputes hash → new run folder, new hash
- Config diff shows execution order changed
- **Likely result**: Ethics blocks panic escalation, agent dies more often (depends on specific configuration)

**Swap GRU for LSTM in perception**:
- Edit `agent_architecture.yaml` (perception_encoder.type: GRU → LSTM)
- Factory rebuilds perception module → new hash
- Behavioral comparison: "LSTM memory → better POMDP performance"

**Disable Social Model (ablation study)**:
- Edit `cognitive_topology.yaml` (social_model.enabled: true → false)
- Factory skips social_model instantiation → new hash
- Behavioral comparison: "No social reasoning → 90% fewer cooperative behaviors"

**Every one of those changes**:
- Produces a **clean diff in YAML**
- Creates a **new run folder** with new run_id
- Generates a **new cognitive hash**

**The platform therefore supports experimentation while keeping governance fully informed.**

**No "I tweaked it overnight and it behaved differently"** - every change is explicit, versioned, and auditable.

**Framework principle**: Surgical edits + hash-based identity is framework-level. The specific edits (panic thresholds vs risk_tolerance, STEAL vs SHUTDOWN, Bed vs Assembly Line) are instance-specific, but the experimental methodology is universal.

---

**Summary**: The Townlet Framework runtime engine consists of:

1. **factory.py** - Builds agents from BAC configuration with interface verification and cognitive hash computation
2. **graph_agent.py** - Runs agents with frozen snapshot (no live config reads)
3. **graph_executor.py** - Executes Layer 3 execution graph with symbolic binding resolution and scratchpad execution
4. **EthicsFilter** - Final compliance veto with governance logging

Together, these transform BAC/UAC declarations into auditable, reproducible, experimentally-flexible agent behavior.

---
