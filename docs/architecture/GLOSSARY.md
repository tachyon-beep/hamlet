# Townlet Architecture Glossary

**Version**: 1.0
**Last Updated**: 2025-11-05
**Purpose**: Define standard terminology distinguishing framework-level concepts from experiment-specific implementations

---

## Framework vs Instance Boundary

**Critical Distinction**: The Townlet system has two conceptual layers:

1. **Framework Layer**: Reusable architecture for building Software Defined Agents in configurable universes
2. **Instance Layer**: Specific experimental configurations (e.g., Townlet Town survival simulation)

---

## Framework-Level Terms (Architecture)

### Core Architecture Concepts

**Townlet Framework**
- The complete UAC/BAC architecture system enabling Software Defined Agents
- Includes: Brain as Code, Universe as Code, runtime engine, provenance system
- **Scope**: Reusable across any universe/domain (towns, factories, trading floors, etc.)
- **Example**: "The Townlet Framework can model agents in any configurable environment"

**Brain as Code (BAC)**
- Declarative configuration defining agent cognition as three YAML layers
- Includes: cognitive_topology.yaml (L1), agent_architecture.yaml (L2), execution_graph.yaml (L3)
- **Scope**: Framework-level architecture pattern
- **Example**: "BAC makes the agent's mind auditable configuration"

**Universe as Code (UAC)**
- Declarative configuration defining world rules, physics, affordances, and economy
- Includes: universe_as_code.yaml with bars, affordances, cascades, costs, effects
- **Scope**: Framework-level architecture pattern
- **Example**: "UAC defines the environment without hardcoding rules in Python"

**Software Defined Agent (SDA)**
- An agent whose cognition and behavior are fully specified by declarative configuration
- Built by combining BAC (mind config) + UAC (world config) + runtime engine
- **Scope**: Framework-level architectural product
- **Example**: "SDAs are to agents what SDN is to networks - configuration over code"

**Cognitive Hash**
- Unique cryptographic fingerprint of a brain+world configuration snapshot
- Computed from BAC layers + UAC snapshot + runtime envelope
- **Scope**: Framework-level provenance mechanism
- **Purpose**: Enables exact reproduction and accountability
- **Example**: "Cognitive hash `9af3c2e1` identifies the exact mind+world combination that ran"

**Universe Instance**
- A specific experimental configuration using the Townlet Framework
- Includes: specific affordances, bars, agent goals, curriculum design
- **Scope**: Instance-level (Townlet Town is one universe instance)
- **Example**: "Townlet Town is our reference universe instance; future instances could model factories or markets"

**Reference Implementation**
- The canonical example universe instance used to demonstrate the framework
- Currently: Townlet Town (survival simulation)
- **Scope**: Instance-level (demonstrates framework capabilities)
- **Example**: "Townlet Town serves as the reference implementation showcasing framework features"

### Runtime Engine (Framework)

**Runtime Engine**
- The execution system that interprets BAC/UAC configurations and runs agents
- Components: factory.py, graph_agent.py, graph_executor.py
- **Scope**: Framework-level implementation
- **Example**: "The runtime engine materializes the declared mind and world into executable code"

**Agent Factory (factory.py)**
- Framework component that instantiates agents from BAC configuration
- Verifies interface contracts, injects Layer 1 parameters, computes cognitive hash
- **Scope**: Framework-level component
- **Example**: "The agent factory builds the declared mind exactly as specified in BAC"

**GraphAgent (graph_agent.py)**
- Framework component representing a running agent instance
- Owns cognitive modules, recurrent state, execution graph
- **Scope**: Framework-level component
- **Example**: "GraphAgent executes the think() loop each tick"

**GraphExecutor (graph_executor.py)**
- Framework component that executes the Layer 3 execution graph
- Compiles symbolic bindings, validates dependencies, runs cognition pipeline
- **Scope**: Framework-level component
- **Example**: "GraphExecutor ensures panic→ethics→action ordering is enforced"

**Scratchpad (Execution)**
- Temporary data cache used by GraphExecutor during tick execution
- Stores named outputs from each execution graph step
- **Scope**: Framework-level runtime mechanism
- **Purpose**: Pass data between cognitive steps without global state
- **Example**: "GraphExecutor creates scratchpad, runs perception → stores belief → policy reads belief"

**Execution Compilation**
- Process of resolving symbolic bindings into callable execution steps
- Occurs at agent instantiation (factory.py via GraphExecutor)
- **Scope**: Framework-level initialization process
- **Purpose**: Transform declarative graph (Layer 3 YAML) into executable pipeline
- **Example**: "Execution compilation resolves @modules.panic_controller into actual callable function"

### Cognitive Modules (Framework Patterns)

**Perception Encoder**
- Framework module pattern: transforms observations into belief state
- **Scope**: Framework-level architectural pattern (implementation varies by instance)
- **Example**: "Every SDA has a perception encoder; Townlet Town uses CNN+GRU"

**World Model**
- Framework module pattern: predicts state transitions for candidate actions
- **Scope**: Framework-level architectural pattern
- **Example**: "World Model enables agents to simulate futures before acting"

**Social Model**
- Framework module pattern: estimates other agents' goals and behaviors
- **Scope**: Framework-level architectural pattern
- **Example**: "Social Model uses public cues to infer neighbor intentions"

**Hierarchical Policy**
- Framework module pattern: selects strategic goals and concrete actions
- **Scope**: Framework-level architectural pattern
- **Example**: "Hierarchical Policy chooses high-level goals (SURVIVAL vs THRIVING) then specific actions"

**Panic Controller**
- Framework module pattern: overrides normal planning during survival crises
- **Scope**: Framework-level safety mechanism
- **Example**: "Panic Controller escalates actions when bars fall below thresholds"

**EthicsFilter**
- Framework module pattern: enforces compliance policy as final veto gate
- **Scope**: Framework-level governance mechanism
- **Example**: "EthicsFilter blocks forbidden actions even during panic"

### Decision Chain (Framework)

**Candidate Action**
- Initial action proposed by hierarchical policy before panic/ethics interventions
- **Scope**: Framework-level decision pipeline step
- **Purpose**: Captures "what the agent wants to do" before safety mechanisms
- **Example**: "Candidate action was STEAL before panic_controller and EthicsFilter intervened"

**Panic Override / Panic Adjusted Action**
- Action modified by panic_controller when survival thresholds are breached
- Includes panic_override_applied flag and panic_reason
- **Scope**: Framework-level safety intervention
- **Purpose**: Emergency escalation for survival (but still subject to ethics)
- **Example**: "Panic override escalated candidate_action from WAIT to CALL_AMBULANCE (panic_reason: health_critical)"

**Final Action**
- Actual action sent to environment after all cognitive steps (policy → panic → ethics)
- **Scope**: Framework-level decision pipeline output
- **Purpose**: The ground truth of what agent actually did
- **Example**: "Final action was WAIT after EthicsFilter blocked panic-adjusted STEAL"

**Veto Reason**
- Explanation logged when EthicsFilter blocks an action
- Examples: "forbidden: steal", "penalized: ambulance_abuse"
- **Scope**: Framework-level governance logging
- **Purpose**: Evidence trail for why actions were blocked
- **Example**: "veto_reason='forbidden: steal' logged when EthicsFilter blocked STEAL attempt"

**Ethics Veto**
- Event where EthicsFilter blocks candidate/panic-adjusted action
- Logged with veto_reason and ethics_veto_applied flag
- **Scope**: Framework-level governance event
- **Purpose**: Proves compliance policy was enforced
- **Example**: "Ethics veto applied at tick 842: attempted STEAL, blocked, reason='forbidden'"

### Provenance System (Framework)

**Run Snapshot**
- Complete frozen configuration for a specific execution
- Includes: BAC layers, UAC config, runtime envelope, cognitive hash
- **Scope**: Framework-level provenance artifact
- **Example**: "Every run produces a snapshot in `run_001/config_snapshot/`"

**Configuration Snapshot**
- Point-in-time freeze of all YAML configurations at launch time
- Byte-for-byte copy of the five YAMLs (config, UAC, BAC L1/L2/L3)
- **Scope**: Framework-level provenance mechanism
- **Purpose**: Prevents untracked changes during execution
- **Example**: "Runtime reads from config_snapshot/, never from mutable configs/ directory"

**Provenance Hash**
- See "Cognitive Hash" (synonym)

**Run Folder**
- Directory containing snapshot, checkpoints, telemetry, logs for a single execution
- Structure: `runs/<run_name>__<timestamp>/`
- Contains: config_snapshot/, checkpoints/, telemetry/, logs/
- **Scope**: Framework-level artifact structure
- **Example**: "`runs/L99_AusterityNightshift__2025-11-03-12-14-22/` contains complete provenance"

**Run Bundle**
- Pre-launch configuration directory prepared before execution
- Located in `configs/<run_name>/`
- Contains: config.yaml, universe_as_code.yaml, BAC layers (L1/L2/L3)
- **Scope**: Framework-level configuration preparation
- **Purpose**: Declaration of what will be run (snapshotted at launch)
- **Example**: "`configs/L99_AusterityNightshift/` is the run bundle declaring the experiment"

**Run ID**
- Unique identifier for a specific execution
- Format: `<run_name>__<timestamp>` (e.g., L99_AusterityNightshift__2025-11-03-12-14-22)
- **Scope**: Framework-level execution tracking
- **Purpose**: Links telemetry to specific configuration snapshot
- **Example**: "Every telemetry log entry includes run_id to identify which execution it came from"

**Tick Index**
- Step number within an execution (0, 1, 2, ...)
- **Scope**: Framework-level time measurement
- **Purpose**: Precise temporal reference for telemetry and replay
- **Example**: "At tick_index 842, EthicsFilter blocked STEAL action"

**Resume**
- Continuing execution from a previous checkpoint
- Loads from checkpoint's config_snapshot/, not from configs/
- Creates new run folder with `_resume_<timestamp>` suffix
- **Scope**: Framework-level checkpoint operation
- **Purpose**: Honest continuation (recomputes hash to verify same mind)
- **Example**: "`L99__2025-11-03-12-14-22_resume_2025-11-03-13-40-09/` is a continuation"

**Fork (Configuration)**
- Changed configuration creating new cognitive identity
- Occurs when resuming with modified BAC/UAC (hash changes)
- **Scope**: Framework-level provenance integrity
- **Contrast**: Resume (same config, same hash) vs Fork (changed config, new hash)
- **Example**: "Lowering panic threshold during resume creates a fork, not a continuation"

**Forensics (Behavioral)**
- Reconstructing past agent behavior from telemetry logs
- Uses: telemetry + config_snapshot + cognitive_hash
- **Scope**: Framework-level audit capability
- **Purpose**: Answer "why did agent X do Y at tick Z" with evidence
- **Example**: "Forensics at tick 842 shows panic override with veto_reason logged"

**Config Diff**
- Comparison of configuration changes between runs
- Shows exactly what changed (YAML diff) and how hash changed
- **Scope**: Framework-level experimental control
- **Purpose**: Scientific comparison ("only Social Model changed; here's behavioral shift")
- **Example**: "Config diff shows only difference was panic_thresholds.energy: 0.15 → 0.10"

**Runtime Envelope**
- Execution parameters: tick rate, duration, num_agents, curriculum schedule, seed
- Defined in config.yaml (part of run bundle)
- **Scope**: Framework-level execution configuration
- **Purpose**: Controls how long and how fast the simulation runs
- **Example**: "Runtime envelope specifies 10k ticks at 20 ticks/sec with 4 agents"

**Telemetry**
- Per-tick logs of agent state, decisions, overrides, vetoes
- Includes: run_id, tick_index, cognitive_hash, goal, panic_state, ethics_veto, etc.
- **Scope**: Framework-level observability
- **Purpose**: Evidence trail for governance and debugging
- **Example**: "Telemetry shows panic_state=true, candidate_action=STEAL, final_action=WAIT, veto_reason='forbidden'"

**Telemetry Row**
- Single per-tick record in telemetry logs
- Contains: run_id, tick_index, cognitive_hash, decision chain, predictions
- **Scope**: Framework-level audit artifact
- **Purpose**: Forensic record for behavioral reconstruction
- **Example**: "Telemetry row for tick 842 shows complete decision pipeline from candidate to final action"

**Short Cognitive Hash**
- Abbreviated form of cognitive hash for UI display
- Typically first 8 characters (e.g., "9af3c2e1" from full hash)
- **Scope**: Framework-level UI convenience
- **Purpose**: Human-readable identity in live displays
- **Example**: "Run Context Panel shows short_hash: 9af3c2e1 for quick agent identification"

**Run Context Panel**
- Live UI inspector showing agent cognition in real-time
- Displays: run_id, short_hash, tick, goal, panic_state, ethics_veto, planning_depth
- **Scope**: Framework-level UI component
- **Purpose**: Glass-box observation of cognitive processes during execution
- **Example**: "Run Context Panel lets instructors point to panic_override and narrate why it happened"

**Glass-Box Capability**
- Framework's ability to expose internal cognitive processes for observation and citation
- Opposite of black-box: visible cognition (candidate action, panic override, ethics veto)
- **Scope**: Framework-level design philosophy
- **Purpose**: Enables teaching, audit, and formal governance
- **Example**: "Glass-box capability means we can cite exact tick when EthicsFilter blocked STEAL"

**Forensic Record**
- Complete audit trail enabling behavioral reconstruction from logs
- Uses: telemetry rows + config_snapshot + cognitive_hash
- **Scope**: Framework-level audit capability
- **Purpose**: Answer "why did agent do X at tick Y" with evidence, not speculation
- **Example**: "Forensic record proves panic_state=false, perception failure caused starvation, not ethics block"

**Belief Uncertainty**
- Confidence/uncertainty measure in perception module estimates
- Logged in telemetry as belief_uncertainty_summary
- **Scope**: Framework-level introspection feature
- **Purpose**: Diagnose perception failures ("didn't see starvation" vs "saw it, chose poorly")
- **Example**: "Belief uncertainty high (0.42 confidence) explains why agent ignored fridge despite low energy"

**World Model Expectation**
- Predictions from world model about future outcomes
- Logged in telemetry as world_model_expectation_summary
- **Scope**: Framework-level introspection feature
- **Purpose**: Diagnose planning failures ("predicted wrong outcome" vs "predicted correctly, chose poorly")
- **Example**: "World model predicted ambulance cost = $300, agent chose it despite low money (panic override)"

**Social Model Inference**
- Predictions about other agents' intentions and actions
- Logged in telemetry as social_model_inference_summary
- **Scope**: Framework-level introspection feature
- **Purpose**: Diagnose social reasoning ("thought competitor would steal fridge")
- **Example**: "Social model inferred Agent_2_intent='use_fridge' with 0.72 confidence, agent yielded"

### Goals and Termination (Framework)

**Goal Definition**
- Declarative structure defining strategic objective with termination condition
- Defined in configuration as goal_definitions with id, termination rules
- **Scope**: Framework-level pattern (specific goals are instance-level)
- **Purpose**: Makes goals explicit data structures, not implicit reward shaping
- **Example**: "SURVIVAL goal terminates when energy ≥ 0.8 AND health ≥ 0.7"

**Termination Condition / Termination Rule**
- Declarative boolean expression determining when goal is satisfied
- Uses DSL with `all`/`any` blocks and bar comparisons
- **Scope**: Framework-level pattern
- **Purpose**: Makes "done with this goal" auditable YAML, not hidden code
- **Example**: "termination: { all: [{ bar: 'energy', op: '>=', val: 0.8 }] }"

**Goal Termination DSL**
- Domain-Specific Language for expressing goal satisfaction conditions
- Syntax: `all`/`any` blocks with bar comparisons, time_elapsed_ticks
- **Scope**: Framework-level configuration language
- **Purpose**: Safe, bounded expressions without arbitrary Python
- **Example**: "{ any: [{ bar: 'money', op: '>=', val: 1.0 }, { time_elapsed_ticks: '>=', val: 500 }] }"

**Engine Truth vs Self-Report**
- Framework distinction: current_goal (factual) vs agent_claimed_reason (narrative)
- Engine truth = meta-controller's actual goal selection
- Self-report = agent's introspective explanation (may differ from truth)
- **Scope**: Framework-level observability contrast
- **Purpose**: Expose gaps between actual cognition and agent's understanding
- **Example**: "Engine truth: SURVIVAL; Self-report: 'working for rent' (world model error)"

**Termination Interpreter**
- Runtime component evaluating goal termination rules against current state
- Executes Goal Termination DSL expressions each tick (or every N ticks)
- **Scope**: Framework-level execution component
- **Purpose**: Fires goal completion, triggers meta-controller goal switch
- **Example**: "Termination interpreter evaluates 'energy >= 0.8 AND health >= 0.7' → true → goal satisfied"

### Affordance Mechanics (Framework)

**Affordance (UAC Details)**
- See primary definition in Instance-Level Terms
- **UAC Configuration**: id, quality, capacity, exclusive, interaction_type, costs, effects_per_tick
- **Framework pattern**: Declarative objects with bar effects (framework-level), specific affordances (instance-level)
- **Example**: "Bed affordance: capacity=1, exclusive=true, effects_per_tick=[{bar: energy, change: +0.25}]"

**Quality (Affordance Parameter)**
- Scaling factor for affordance effectiveness (0.0-1.0 or higher)
- Multiplies effect magnitudes (quality=1.0 = full effect, quality=0.5 = half effect)
- **Scope**: Framework-level UAC parameter
- **Purpose**: Models variation in affordance utility (nice bed vs cheap bed)
- **Example**: "bed_basic.quality: 1.0 (full rest); bed_gross.quality: 0.5 (half energy recovery)"

**Capacity (Affordance Parameter)**
- Maximum number of agents that can use affordance simultaneously per tick
- **Scope**: Framework-level UAC parameter
- **Purpose**: Models resource contention
- **Example**: "Bed.capacity: 1 (one sleeper), Restaurant.capacity: 4 (four diners)"

**Exclusive (Affordance Parameter)**
- Boolean: if true, only one agent can occupy affordance (capacity still applies)
- **Scope**: Framework-level UAC parameter
- **Purpose**: Models single-occupancy resources
- **Example**: "Bed.exclusive: true (can't share bed); Park.exclusive: false (multiple visitors OK)"

**Interaction Type (Affordance Parameter)**
- Temporal shape: "multi_tick" (sustained over time) or "instant" (one-shot)
- **Scope**: Framework-level UAC parameter
- **Purpose**: Distinguishes continuous (sleep in bed) from atomic (phone ambulance) actions
- **Example**: "Bed.interaction_type: multi_tick (sleep over time); Phone.interaction_type: instant (one call)"

**Interruptible (Affordance Parameter)**
- Boolean: if true, agent can abandon affordance mid-interaction
- **Scope**: Framework-level UAC parameter
- **Purpose**: Models whether actions can be stopped early (panic might interrupt sleep)
- **Example**: "Bed.interruptible: true (can wake up early); Job.interruptible: false (must finish shift)"

**Distance Limit (Affordance Parameter)**
- Maximum distance from affordance to interact (0 = must be on tile, 1 = adjacent)
- **Scope**: Framework-level UAC parameter
- **Purpose**: Spatial constraint for affordance usage
- **Example**: "Bed.distance_limit: 0 (on tile); Phone.distance_limit: 1 (nearby)"

**Effects Per Tick (Affordance Parameter)**
- Array of bar changes applied each tick during multi_tick interaction
- Format: `{ bar: "energy", change: +0.25, scale_by: "quality" }`
- **Scope**: Framework-level UAC parameter
- **Purpose**: Defines sustained benefits/costs over time
- **Example**: "Bed.effects_per_tick: [{ bar: 'energy', change: +0.25 }] (restore energy while sleeping)"

**Costs (Affordance Parameter)**
- Array of bar changes deducted when initiating affordance
- Format: `{ bar: "money", change: -0.05 }`
- **Scope**: Framework-level UAC parameter
- **Purpose**: Upfront payment or resource consumption
- **Example**: "Bed.costs: [{ bar: 'money', change: -0.05 }] (pay rent to sleep)"

**On Interrupt (Affordance Parameter)**
- Configuration for partial refunds/benefits when interaction interrupted early
- Includes refund_fraction, note
- **Scope**: Framework-level UAC parameter
- **Purpose**: Declarative semantics for early exit
- **Example**: "Bed.on_interrupt: { refund_fraction: 0.0, note: 'no refund if you bail early' }"

**Reservation (Engine Mechanism)**
- Per-tick ephemeral assignment of affordance to agent
- Checks: capacity available, preconditions met, distance within limit
- **Scope**: Framework-level runtime mechanism
- **Purpose**: Allocates resources without long-lived mutable state
- **Example**: "Agent requests Bed, engine creates reservation token for this tick"

**Contention Resolution**
- Deterministic tie-breaking when multiple agents want same affordance
- Typical order: sort by distance, then agent_id
- **Scope**: Framework-level runtime mechanism
- **Purpose**: Reproducible outcomes for World Model training
- **Example**: "Two agents want Bed (capacity=1), agent_001 closer → wins reservation"

**Precondition (Affordance Constraint)**
- Boolean condition that must be true for affordance to activate
- Format: `{ bar: "health", op: "<=", val: 0.2 }`
- **Scope**: Framework-level UAC constraint
- **Purpose**: Conditional effects (ambulance only if health critical)
- **Example**: "Phone_Ambulance.precondition: { bar: 'health', op: '<=', val: 0.2 } (only if dying)"

**Atomic Effects Application**
- All affordance costs/effects collected per agent, then applied simultaneously
- Prevents partial updates influencing same-tick decisions
- **Scope**: Framework-level runtime discipline
- **Purpose**: Clean training data for World Model
- **Example**: "Collect all effects → sum per agent → apply once → clamp bars to [0.0, 1.0]"

**Special Effects Whitelist**
- Limited set of named effect types allowed in UAC (teleport, etc.)
- Prevents arbitrary operations ("nuke_city: true" not allowed)
- **Scope**: Framework-level security constraint
- **Purpose**: Keeps world spec expressive but bounded
- **Example**: "UAC allows 'teleport', 'heal', 'damage' - centrally implemented, versioned"

### Success Criteria & Implementation (Framework)

**Success Criteria**
- Three-axis evaluation framework: technical, pedagogical, governance
- All three must be satisfied for system acceptance
- **Scope**: Framework-level quality gates
- **Purpose**: Ensures system meets reproducibility, teachability, and auditability standards
- **Example**: "Success criteria demand cognitive hash provenance (technical), YAML-only ablations (pedagogical), and chain-of-custody (governance)"

**Technical Success**
- Framework capability criteria: snapshots, checkpoints, resume, telemetry, UI
- Checkboxes: frozen config_snapshot, GraphAgent.think() from snapshot, checkpoint with hash, resume with new run folder
- **Scope**: Framework-level acceptance criteria
- **Purpose**: Prove system can deliver reproducible minds in governed worlds
- **Example**: "Technical success requires GraphAgent built purely from config_snapshot, no live config reads"

**Pedagogical Success**
- Teachability criteria: YAML-only reasoning, controlled ablations, clip forensics
- Students answer questions using config + UI, not source code
- **Scope**: Framework-level educational quality gate
- **Purpose**: Transform RL from superstition to inspectable system
- **Example**: "Pedagogical success: beginner explains ethics veto using cognitive_topology.yaml + Run Context Panel, no code needed"

**Governance Success**
- Auditability criteria: tick-level proof, checkpoint replay, lineage rules
- Auditors can prove what happened at any tick with config snapshot + telemetry
- **Scope**: Framework-level compliance standard
- **Purpose**: Formal review capability for regulated deployments
- **Example**: "Governance success: prove agent attempted STEAL at tick T, action blocked, evidence chain complete"

**Chain-of-Custody (Cognition)**
- Unbroken provenance trail for agent cognition from launch through checkpoints to resume
- Uses: frozen config_snapshot + cognitive_hash + checkpoint directory
- **Scope**: Framework-level governance mechanism
- **Purpose**: Prove exact mind at exact tick without trusting live config
- **Example**: "Chain-of-custody established: checkpoint includes snapshot copy + hash, resume loads only from checkpoint"

**Lineage Rules**
- Identity continuity protocol: same snapshot → same hash, edited snapshot → new hash + new run_id
- Prevents silent cognitive mutations pretending to be "same agent, adjusted"
- **Scope**: Framework-level identity discipline
- **Purpose**: Honest fork detection and mind identity enforcement
- **Example**: "Lineage rules: edit panic_thresholds → new hash + new run_id, we don't lie about continuity"

**Controlled Ablation**
- Experimental modification via config edit, not code surgery
- Edit agent_architecture.yaml or execution_graph.yaml, launch new run, observe behavioral change
- **Scope**: Framework-level experimental capability
- **Purpose**: Scientific comparison without coding expertise
- **Example**: "Controlled ablation: swap GRU→LSTM in agent_architecture.yaml, observe memory capacity change"

**Teachable Agent**
- Agent whose behavior can be reasoned about by students using config + UI, not superstition
- Answers "why did it X?" with YAML inspection and telemetry citation
- **Scope**: Framework-level pedagogical property
- **Purpose**: Transform RL education from intuition-building to evidence-based reasoning
- **Example**: "Teachable agent: 'Why starving?' → point to high greed in cognitive_topology + GET_MONEY goal in telemetry"

**Governance-Grade Identity**
- Formal identity system meeting audit requirements: cognitive hash + lineage rules + chain-of-custody
- Not research convenience ("probably the same agent"), actual identity proof
- **Scope**: Framework-level compliance standard
- **Purpose**: Withstand formal review for regulated AI deployments
- **Example**: "Governance-grade identity: hash changes on any cognitive edit, provable via checkpoint snapshots"

**Snapshot Discipline**
- Framework requirement: freeze config at launch, runtime never re-reads mutable config
- Creates config_snapshot/ directory, all subsequent operations use snapshot only
- **Scope**: Framework-level provenance protocol
- **Purpose**: Lock down world+mind from first tick, prevent silent mutations
- **Example**: "Snapshot discipline: launcher copies 5 YAMLs to config_snapshot/, runtime reads only snapshot"

**Build Sequence**
- Recommended implementation ordering to establish provenance foundation first
- Six-step sequence: snapshots → GraphAgent → cognitive hash → checkpoints → telemetry/UI → panic/ethics
- **Scope**: Framework-level implementation guidance
- **Purpose**: Prevent retrofitting provenance (build it from day one)
- **Example**: "Build sequence: establish snapshot discipline before GraphAgent, hash before checkpoints"

**Boot Sequence**
- System initialization order at run launch
- Steps: copy config to snapshot → compute cognitive hash → instantiate GraphAgent → start telemetry
- **Scope**: Framework-level startup protocol
- **Purpose**: Ensure provenance established before first tick
- **Example**: "Boot sequence: config frozen → hash computed → GraphAgent built from snapshot → telemetry starts"

**Minimal GraphAgent Pipeline**
- First working milestone: GraphAgent.think() ticks once from config_snapshot
- Stub modules OK (world_model, social_model can pass through)
- **Scope**: Framework-level implementation milestone
- **Purpose**: Prove brain-from-YAML works before building complex modules
- **Example**: "Minimal pipeline: factory.py builds modules → GraphExecutor compiles graph → GraphAgent.think() runs perception→policy→action"

**Milestone**
- Concrete delivery checkpoint with Definition of Done
- Each milestone establishes one aspect of provenance or observability
- **Scope**: Framework-level project management
- **Purpose**: Incremental delivery with testable acceptance criteria
- **Example**: "Milestone 12.1: Snapshots and run folders - done when config_snapshot/ byte-for-byte copy exists"

**Definition of Done**
- Acceptance criteria for milestone completion
- Checkboxes proving capability works end-to-end
- **Scope**: Framework-level quality gate
- **Purpose**: No partial implementations, milestones must be demonstrable
- **Example**: "Definition of done for checkpoints: weights.pt + snapshot + cognitive_hash.txt all present, resume works"

---

## Instance-Level Terms (Townlet Town Experiment)

### Townlet Town Universe

**Townlet Town**
- The reference universe instance: agents learning to survive in a simulated town
- **Scope**: Specific experimental configuration of the Townlet Framework
- **Domain**: Low-fidelity life simulation ("trick students into learning RL by making them think they're playing The Sims")
- **Example**: "Townlet Town is one universe instance; we could create 'Townlet Factory' or 'Townlet Market' using the same framework"

**Town Environment**
- The specific spatial configuration for Townlet Town
- Currently: 8×8 grid with 14 affordances
- **Scope**: Townlet Town instance-specific
- **Example**: "The town environment is an 8×8 grid (instance-specific), but the framework supports any spatial structure"

### Affordances (Instance-Specific Vocabulary)

**Affordance**
- Framework concept: An interactable object defined in UAC
- Instance vocabulary: Townlet Town defines 14 specific affordances
- **Framework-level**: The pattern of "affordances with costs/effects"
- **Instance-level**: Bed, Hospital, Job, Fridge, etc.
- **Example**: "Affordances are a framework concept; Bed and Hospital are Townlet Town instance vocabulary"

**Townlet Town Affordances** (Instance Vocabulary):
- Bed, Hospital, HomeMeal, Job, Gym, Shower, Bar, Restaurant, Park, Phone, Mall, SocialEvent, Couch, Fridge
- **Scope**: Townlet Town instance-specific
- **Example**: "Bed is a Townlet Town affordance; other instances might have 'Assembly Line' or 'Trading Desk'"

### Bars/Meters (Instance-Specific Vocabulary)

**Bar (Framework) / Meter (Framework)**
- Framework concept: A continuous state variable (0.0 to 1.0) tracked per agent
- **Scope**: Framework-level pattern (UAC defines bars)
- **Example**: "Bars are a framework concept for tracking agent state"

**Townlet Town Bars** (Instance Vocabulary):
- Energy, Health, Satiation, Money, Mood, Social, Fitness, Hygiene
- **Scope**: Townlet Town instance-specific
- **Example**: "Energy and Health are Townlet Town bars; a factory instance might track 'Fatigue' and 'Skill'"

### Goals (Instance-Specific Vocabulary)

**Goal (Framework)**
- Framework concept: High-level strategic objective for Hierarchical Policy
- **Scope**: Framework-level pattern (Layer 1 defines allowed_goals)

**Townlet Town Goals** (Instance Vocabulary):
- SURVIVAL (meet critical needs)
- THRIVING (optimize quality of life)
- SOCIAL (prioritize relationships)
- **Scope**: Townlet Town instance-specific
- **Example**: "SURVIVAL/THRIVING/SOCIAL are Townlet Town goals; a trading instance might have BUY/SELL/HOLD"

### Curriculum Levels (Instance-Specific)

**Curriculum Level**
- Townlet Town training progression from simple to complex scenarios
- **Levels**: L0 (temporal credit), L0.5 (dual resource), L1 (full observability), L2 (POMDP), L3 (temporal mechanics)
- **Scope**: Townlet Town instance-specific pedagogical design
- **Example**: "Curriculum levels are specific to Townlet Town's educational mission"

---

## General Terms (Used Throughout)

### Training and Learning

**Agent**
- A learning entity that observes, decides, and acts in an environment
- **Scope**: General RL term (used at both framework and instance levels)

**Episode**
- A single run from initialization to termination (death or max steps)
- **Scope**: General RL term

**Checkpoint**
- Frozen moment of specific mind + world + RNG state at specific tick
- Contains: weights.pt, optimizers.pt, rng_state.json, config_snapshot/, cognitive hash
- **Scope**: Framework-level artifact (not just "saved weights")
- **Purpose**: Evidence for honest resume, replay, audit
- **Example**: "Checkpoint at step_000500 proves which mind+world configuration existed at tick 500"

**Checkpoint Artifact (weights.pt)**
- Neural network weights for all cognitive modules at checkpoint tick
- Includes: perception, world_model, social_model, hierarchical_policy, panic_controller, EthicsFilter
- **Scope**: Framework-level artifact component
- **Purpose**: Captures learned state of SDA modules
- **Example**: "weights.pt contains all module parameters from Layer 2 module registry"

**Checkpoint Artifact (optimizers.pt)**
- Optimizer state (Adam moments, etc.) for all trainable modules
- **Scope**: Framework-level artifact component
- **Purpose**: Honest resume of learning process (not just weights)
- **Why**: Quietly dropping optimizer state changes learning behavior
- **Example**: "optimizers.pt preserves Adam momentum for continuous training"

**Checkpoint Artifact (rng_state.json)**
- Random number generator states at checkpoint tick
- Includes: environment RNG, agent RNG (PyTorch generators), exploration noise
- **Scope**: Framework-level artifact component
- **Purpose**: Reproducible stochastic outcomes for replay
- **Example**: "rng_state.json lets us replay tick 501 with same random sequence"

**Module Registry**
- Framework component tracking all cognitive modules in an SDA
- Populated from Layer 2 (agent_architecture.yaml)
- **Scope**: Framework-level runtime structure
- **Purpose**: Enumerates all modules to save/load in checkpoints
- **Example**: "Module registry contains perception, world_model, social_model, policy, panic, ethics"

**Optimizer State**
- Internal state of optimization algorithms (Adam moments, RMSprop cache, etc.)
- Saved in optimizers.pt
- **Scope**: Framework-level training artifact
- **Purpose**: Required for honest training continuation
- **Example**: "Adam optimizer state includes first/second moment estimates per parameter"

**Evidence Trail**
- Sequence of checkpoints + telemetry proving behavioral causality
- Uses: config_snapshot + cognitive_hash + telemetry logs
- **Scope**: Framework-level audit capability
- **Purpose**: Transforms anecdotes into proof
- **Example**: "Evidence trail shows panic_threshold=0.15, STEAL forbidden, veto logged at tick 842"

**Continuity of Mind**
- Proof that resumed run uses exact same cognitive configuration
- Verified by matching cognitive hash after resume
- **Scope**: Framework-level resume guarantee
- **Purpose**: Governance assurance that agent didn't change during pause
- **Example**: "Same hash = same mind; different hash = fork (new agent)"

**Ablation Study**
- Controlled experiment removing/disabling specific component to measure impact
- Framework enforces explicit fork (new hash) when ablating
- **Scope**: Framework-level experimental methodology
- **Example**: "Disable social_model, creates fork with new hash, compare behavior to baseline"

**Rehydration**
- Loading agent from checkpoint's config_snapshot (not live configs/)
- Restores exact mind+world+RNG state from frozen snapshot
- **Scope**: Framework-level resume operation
- **Purpose**: Honest continuation (prevents drift from live config changes)
- **Example**: "Rehydrate from checkpoint's snapshot, not from configs/ directory"

**Governance Primitive**
- Core operation with governance significance (not just technical convenience)
- Resume is governance primitive: proves continuity or explicitly creates fork
- **Scope**: Framework-level design philosophy
- **Purpose**: Makes governance-relevant operations auditable and explicit
- **Example**: "Resume as governance primitive means fork vs continuation is provable, not claimed"

**Drift (Configuration)**
- Untracked gradual changes to agent configuration over time
- Framework prevents drift: changes create new hash (detectable fork)
- **Scope**: Framework-level integrity protection
- **Contrast**: Old systems allow untracked drift; framework makes it explicit
- **Example**: "No 'it drifted over time' - drift is now a recorded fork with new hash"

**Q-Network**
- Neural network estimating action-value function Q(s, a)
- **Scope**: General RL term (specific architectures are instance-level)

**POMDP**
- Partially Observable Markov Decision Process
- **Scope**: General RL term (L2 curriculum level uses POMDP)

### Configuration Terms

**YAML Configuration**
- Human-readable declarative file format for BAC/UAC
- **Scope**: Framework-level technical choice

**Layer 1 (L1) / Cognitive Topology**
- BAC layer defining behavior contract, ethics, panic, personality
- **Scope**: Framework-level BAC architecture

**Layer 2 (L2) / Agent Architecture**
- BAC layer defining module implementations (networks, optimizers)
- **Scope**: Framework-level BAC architecture

**Layer 3 (L3) / Execution Graph**
- BAC layer defining think-loop ordering (DAG of cognitive steps)
- **Scope**: Framework-level BAC architecture

**Personality**
- Framework pattern: Configurable behavioral biases (greed, curiosity, neuroticism, agreeableness)
- Defined in Layer 1 as numeric sliders (0.0-1.0)
- **Scope**: Framework-level pattern (specific personality traits vary by universe)
- **Example**: "Townlet Town defines greed/curiosity/neuroticism/agreeableness; trading agents might use risk_tolerance/patience"

**Introspection**
- Framework capability: Agent's ability to explain its reasoning in UI
- Configured in Layer 1 (publish_goal_reason, visible_in_ui)
- **Scope**: Framework-level transparency feature
- **Example**: "Introspection lets agents narrate 'I'm going to work because we need money'"

**Interface Contract**
- Framework concept: Explicit specification of data dimensions between modules
- Defined in Layer 2 (belief_distribution_dim, goal_vector_dim, etc.)
- **Scope**: Framework-level architectural discipline
- **Purpose**: Prevents silent mismatches, enables module swapping
- **Example**: "Layer 2 enforces that perception output dim matches policy input dim"

**Belief Distribution**
- Framework concept: Agent's internal representation of world state
- Output of perception encoder, input to policy
- **Scope**: Framework-level architectural concept
- **Example**: "Perception produces belief_distribution (128-dim vector) representing agent's understanding"

**Recurrent State**
- Framework concept: Memory carried between ticks for sequential decision-making
- Hidden state for recurrent modules (GRU, LSTM)
- **Scope**: Framework-level technical concept
- **Example**: "Recurrent state lets agent remember observations from previous ticks"

**DAG (Directed Acyclic Graph)**
- Framework concept: The execution ordering in Layer 3
- Defines cognitive steps and their dependencies without cycles
- **Scope**: Framework-level graph execution pattern
- **Example**: "Layer 3 execution graph is a DAG: perception → policy → panic → ethics → action"

**Meta-Controller**
- Framework pattern: High-level decision maker in hierarchical policy
- Selects strategic goals, delegates action selection to controller
- **Scope**: Framework-level policy architecture
- **Example**: "Meta-controller chooses SURVIVAL vs THRIVING goal every 50 ticks"

**Pretraining**
- Framework capability: Training cognitive modules before full agent runs
- Defined in Layer 2 (objective, dataset)
- **Scope**: Framework-level training strategy
- **Example**: "World model pretraining on UAC logs teaches dynamics before agent deployment"

**Symbolic Binding**
- Framework pattern: References to runtime components using @ syntax
- Examples: @modules.perception, @config.L1.panic_thresholds, @services.world_model
- **Scope**: Framework-level configuration mechanism
- **Purpose**: Declarative wiring without hardcoded references
- **Example**: "@modules.EthicsFilter binds to the ethics module at runtime"

**Service Binding**
- Framework pattern: Making cognitive modules available as callable services
- Example: @services.world_model_service, @services.social_model_service
- **Scope**: Framework-level dependency injection
- **Example**: "Hierarchical policy calls @services.world_model_service to simulate futures"

### System Components

**Vectorized Environment**
- GPU-native batched environment for parallel agent training
- **Scope**: Framework-level implementation pattern

**Replay Buffer**
- Experience storage for off-policy learning
- **Scope**: General RL term

**Exploration Strategy**
- Method for balancing exploration vs exploitation
- **Scope**: General RL term (specific strategies like RND are instance-level)

---

## Terminology Guidelines

### When Writing Documentation

**Framework-level discussions** should use:
- "Townlet Framework", "BAC", "UAC", "Software Defined Agent"
- "Cognitive modules" (generic), "affordances" (pattern), "bars" (pattern)
- "Universe instance", "reference implementation"

**Instance-level discussions** should use:
- "Townlet Town", "reference implementation"
- Specific affordances: "Bed", "Hospital", etc.
- Specific bars: "energy", "health", etc.
- Specific goals: "SURVIVAL", "THRIVING", "SOCIAL"

**Clear boundary examples**:
- ✅ "The Townlet Framework supports any universe; Townlet Town is our reference implementation focusing on survival"
- ✅ "BAC defines agent cognition (framework); Townlet Town configures SURVIVAL/THRIVING/SOCIAL goals (instance)"
- ✅ "Affordances are a UAC pattern (framework); Bed and Hospital are Townlet Town affordances (instance)"
- ❌ "Townlet has beds and hospitals" (conflates framework with instance)

---

## Acronym Reference

| Acronym | Full Term | Level | Definition |
|---------|-----------|-------|------------|
| **BAC** | Brain as Code | Framework | Agent cognition as declarative YAML configuration |
| **UAC** | Universe as Code | Framework | World rules as declarative YAML configuration |
| **SDA** | Software Defined Agent | Framework | Agent built from BAC+UAC+runtime |
| **L1** | Layer 1 / Cognitive Topology | Framework | BAC layer: behavior contract, ethics, panic |
| **L2** | Layer 2 / Agent Architecture | Framework | BAC layer: module implementations |
| **L3** | Layer 3 / Execution Graph | Framework | BAC layer: think-loop ordering |
| **POMDP** | Partially Observable MDP | General | Environment with hidden state |
| **RND** | Random Network Distillation | General | Intrinsic curiosity mechanism |
| **L0, L1, L2, L3** | Curriculum Levels | Instance | Townlet Town training progression |

**Note**: "L1/L2/L3" is ambiguous (BAC layers vs curriculum levels). Use full terms when context is unclear:
- "Layer 1" or "cognitive topology" for BAC
- "Level 1" or "L1 full observability" for curriculum

---

## Version History

**1.0** (2025-11-05): Initial glossary establishing framework vs instance terminology
- Sections 1-5 coverage: 44 terms (core architecture, BAC, UAC, provenance, checkpoints, resume)

**1.1** (2025-11-05): Added runtime engine and telemetry terms
- Sections 6-7 coverage: 15 new terms (execution compilation, decision chain, telemetry, observability)
- New subsections: Decision Chain (Framework), expanded Telemetry with glass-box capability terms
- Total terms: 59

**1.2** (2025-11-05): Added goals, termination, and affordance mechanics terms
- Sections 8-9 coverage: 19 new terms (goal definitions, termination DSL, affordance parameters, engine mechanisms)
- New subsections: Goals and Termination (Framework), Affordance Mechanics (Framework)
- Total terms: 78

**1.3** (2025-11-05): Added success criteria and implementation ordering terms
- Sections 10-12 coverage: 16 new terms (success criteria axes, chain-of-custody, lineage rules, controlled ablation, snapshot discipline, build sequence, milestones)
- New subsection: Success Criteria & Implementation (Framework)
- Total terms: 94

---

**Usage**: Reference this glossary when writing architecture docs, code comments, or training materials to maintain consistent terminology.
