# 2. Brain as Code (BAC)

**Document Type**: Component Spec
**Status**: Updated Draft
**Version**: 1.0 - Townlet Framework (BAC/UAC Architecture)
**Last Updated**: 2025-11-05
**Owner**: Principal Technical Advisor (AI)
**Parent Document**: TOWNLET_HLD.md

**Audience**: Engineers implementing cognitive modules, researchers configuring agents, governance teams reviewing behavior contracts
**Technical Level**: Deep Technical (YAML specifications, interface contracts, execution graphs)
**Estimated Reading Time**: 10 min for skim | 25 min for full read

---

## AI-Friendly Summary (Skim This First!)

**What This Document Describes**:
Brain as Code (BAC) - the three-layer YAML specification that defines agent cognition as
inspectable, diffable, enforceable configuration. Layer 1 (cognitive topology) defines what
the agent can do, Layer 2 (agent architecture) defines how modules are built, Layer 3
(execution graph) defines the think-loop ordering.

**Why This Document Exists**:
Provides the complete specification for declaring Software Defined Agent (SDA) minds as
configuration. Enables governance review, experimental reproducibility, and module swapping
without code changes.

**Who Should Read This**:
- **Must Read**: Engineers building cognitive modules, governance teams approving agent behavior
- **Should Read**: Researchers configuring experiments, safety auditors reviewing ethics enforcement
- **Optional**: Operators deploying agents (focus on Layer 1 only)

**Reading Strategy**:
- **Quick Scan** (10 min): Read §2.1 (Layer 1) for behavior contract and safety envelope
- **Partial Read** (15 min): Add §2.2 (Layer 2) for module architecture and interface contracts
- **Full Read** (25 min): Add §2.3 (Layer 3) for execution graph and governance ordering

---

## Document Scope

**In Scope**:
- **Layer 1 (Cognitive Topology)**: Behavior contract, ethics, panic, personality, allowed capabilities
- **Layer 2 (Agent Architecture)**: Module implementations, network types, interface contracts, pretraining
- **Layer 3 (Execution Graph)**: Think-loop DAG, symbolic bindings, governance ordering (panic → ethics)
- **Framework patterns**: How to configure any SDA, not just Townlet Town agents

**Out of Scope**:
- **Runtime engine implementation**: See §6 (Runtime Engine Components)
- **Universe as Code**: See §8 (UAC specification)
- **Training procedures**: See training documentation
- **Curriculum design**: See Townlet Town experiment docs

**Critical Boundary**:
BAC is **framework-level** architecture. Examples show **Townlet Town** configurations
(SURVIVAL/THRIVING/SOCIAL goals, energy/health bars, attack/steal actions) but the three-layer
pattern applies to any universe instance. See [GLOSSARY.md](../GLOSSARY.md) for terminology.

---

## Position in Architecture

**Related Documents** (Read These First/Next):
- **Prerequisites**: [01-executive-summary.md](01-executive-summary.md) (architectural overview)
- **Builds On**: N/A (foundational BAC specification)
- **Related**: [06-runtime-engine-components.md](06-runtime-engine-components.md) (how BAC is materialized), [GLOSSARY.md](../GLOSSARY.md) (terminology)
- **Next**: [03-run-bundles-provenance.md](03-run-bundles-provenance.md) (how BAC is snapshotted)

**Section Number**: 2 / 12
**Architecture Layer**: Logical (specification of cognitive architecture)

---

## Keywords for Discovery

**Primary Keywords**: Brain as Code (BAC), Layer 1, Layer 2, Layer 3, cognitive topology, agent architecture, execution graph
**Secondary Keywords**: interface contracts, symbolic binding, DAG, personality, introspection, pretraining, meta-controller
**Subsystems**: perception, world model, social model, hierarchical policy, panic controller, EthicsFilter
**Design Patterns**: Three-layer configuration, interface contracts, DAG execution, symbolic bindings

**Quick Search Hints**:
- Looking for "what can the agent do"? → See §2.1 (Layer 1)
- Looking for "module implementations"? → See §2.2 (Layer 2)
- Looking for "think-loop ordering"? → See §2.3 (Layer 3)
- Looking for "ethics enforcement"? → See §2.3 (EthicsFilter in execution graph)
- Looking for "framework vs instance"? → See "Document Scope" and examples with **(Townlet Town)** annotations

---

## Version History

**Version 1.0** (2025-11-05): Initial BAC specification establishing three-layer cognitive configuration architecture

---

## Document Type Specifics

### For Component Specification Documents (Type: Component Spec)

**Component Name**: Brain as Code (BAC)
**Component Type**: Configuration Architecture (three YAML layers)
**Location in Codebase**: Will be created in `config_snapshot/` per run

**Interface Contract**:
- **Inputs**: Three YAML files (cognitive_topology.yaml, agent_architecture.yaml, execution_graph.yaml)
- **Outputs**: Software Defined Agent (SDA) with defined behavior, modules, and think-loop
- **Dependencies**: Runtime engine (factory.py, graph_agent.py, graph_executor.py)
- **Guarantees**: Reproducible agent cognition, auditable behavior contract, provenance via cognitive hash

**Critical Properties**:
- **Declarative**: All cognition specified in YAML, not hidden in code
- **Layered**: Three levels of abstraction (what/how/order)
- **Auditable**: Governance can review behavior without reading implementation code
- **Reproducible**: Same YAML → same cognitive hash → same agent

---

**END OF FRONT MATTER**
**BEGIN ARCHITECTURE CONTENT**

---

## 2. Brain as Code (BAC): The Mind

BAC defines agent cognition as something we can inspect, diff, and enforce.

The BAC stack is three YAML layers. Together, they specify a Software Defined Agent (SDA).

Change the YAMLs, you change the mind. Snapshot the YAMLs, you freeze the mind. Hash the snapshot, you can prove which mind took which action.

**Framework note**: BAC is the architecture pattern. The examples below show **Townlet Town** configurations (survival goals, energy bars, attack/steal actions) but other universe instances would define different vocabulary using the same three-layer structure.

---

## 2.1 Layer 1: cognitive_topology.yaml

**Audience**: Governance, instructors, simulation designers
**Nickname**: The character sheet

Layer 1 defines the behavior contract and safety envelope for a specific agent instance in a specific run.

**It answers**:

- Is social reasoning enabled?
- How far ahead is the agent allowed to plan?
- When does it panic and override normal plans?
- What is the agent allowed to do, and what is absolutely forbidden?
- How greedy, anxious, curious, agreeable is it, as dials not as fairy dust?
- Is it allowed to narrate its motives in the UI?

**Example configuration**:

```yaml
perception:
  enabled: true
  uncertainty_awareness: true    # Agent can admit "I'm not sure"

world_model:
  enabled: true
  rollout_depth: 6               # Allowed planning horizon (ticks ahead)
  num_candidates: 4              # Futures evaluated per tick

social_model:
  enabled: true                  # false = does not model other minds
  use_family_channel: true       # allow private in-group signaling (Townlet Town: multi-agent)

hierarchical_policy:
  meta_controller_period: 50     # How often to reconsider high-level goal
  allowed_goals:                 # (Townlet Town instance vocabulary)
    - SURVIVAL
    - THRIVING
    - SOCIAL

personality:                     # Framework pattern (instance-specific sliders)
  greed: 0.7                     # money drive
  agreeableness: 0.3             # harmony vs confrontation
  curiosity: 0.8                 # exploration drive
  neuroticism: 0.6               # risk aversion / anxiety

panic_thresholds:                # Framework pattern (instance-specific bars)
  energy: 0.15                   # if energy < 15 percent => emergency mode
  health: 0.25                   # (Townlet Town bars: energy, health, satiation)
  satiation: 0.10

compliance:                      # Framework pattern (instance-specific actions)
  forbid_actions:                # Never allowed, even during panic
    - "attack"                   # (Townlet Town action vocabulary)
    - "steal"
  penalize_actions:              # Discouraged but not forbidden
    - { action: "shove", penalty: -5.0 }

introspection:                   # Framework capability
  publish_goal_reason: true      # Should the agent explain itself in UI?
  visible_in_ui: "research"      # beginner | intermediate | research
```

**How Layer 1 connects to runtime**:

- **panic_thresholds** → Tells `panic_controller` when to override normal planning for survival
- **forbid_actions** → Tells `EthicsFilter` what is never allowed, even if the agent is dying
- **personality** → Feeds into hierarchical policy's goal choice (greed: 0.7 means money-seeking wins internal debates)
- **publish_goal_reason** → Controls whether UI surfaces "I'm going to work because we need money"

**Governance significance**: Layer 1 is what policy teams sign off on. It's the file you show when someone asks "what kind of mind did you put in this world?"

**Cognitive hash dependency**: If you change Layer 1 between runs (e.g., allow STEAL, or lower panic threshold, or turn social modeling off), that's not the same agent anymore. The framework must produce a new cognitive hash.

**Framework vs Instance**: Patterns like `panic_thresholds`, `compliance`, `personality` are framework-level. The specific bars (energy/health), actions (attack/steal), goals (SURVIVAL/THRIVING/SOCIAL), and personality traits (greed/curiosity) are Townlet Town vocabulary. A factory instance might define `machinery_stress` bars, `shutdown` actions, and `efficiency/safety` goals.

---

## 2.2 Layer 2: agent_architecture.yaml

**Audience**: Engineers, grad students, researchers
**Nickname**: The blueprint

Layer 2 defines the internal build sheet for cognitive faculties. If Layer 1 says "there is a World Model and it's allowed to plan 6 ticks ahead", Layer 2 says "the World Model is a 2-layer MLP with 256 units, these heads, trained on this dataset, with Adam at this learning rate".

**This file specifies**:

- Network types (CNN, GRU, MLP, etc.)
- Hidden sizes, head dimensions
- Interface contracts between modules (enforced dimensional compatibility)
- Optimizer types and learning rates
- Pretraining objectives and datasets

**Purpose**: Enforces discipline so you can swap modules and reproduce experiments without mystery glue.

**Example configuration**:

```yaml
interfaces:                          # Framework pattern: explicit interface contracts
  belief_distribution_dim: 128      # Perception output
  imagined_future_dim: 256          # World Model summary
  social_prediction_dim: 128        # Social Model summary
  goal_vector_dim: 16               # Meta-controller goal embedding
  action_space_dim: 6               # (Townlet Town: up,down,left,right,interact,wait)

modules:
  perception_encoder:
    spatial_frontend:
      type: "CNN"
      channels: [16, 32, 32]
      kernel_sizes: [3, 3, 3]
    vector_frontend:
      type: "MLP"
      layers: [64]
      input_features: "auto"
    core:
      type: "GRU"
      hidden_dim: 512
      num_layers: 2
    heads:
      belief_dim: 128               # must match interfaces.belief_distribution_dim
    optimizer: { type: "Adam", lr: 0.0001 }
    pretraining:                    # Framework capability
      objective: "reconstruction+next_step"
      dataset: "observation_rollout_buffer"

  world_model:
    core_network:
      type: "MLP"
      layers: [256, 256]
      activation: "ReLU"
    heads:
      next_state_belief: { dim: 128 }
      next_reward:       { dim: 1 }
      next_done:         { dim: 1 }
      next_value:        { dim: 1 }
    optimizer: { type: "Adam", lr: 0.00005 }
    pretraining:
      objective: "dynamics+value"
      dataset: "uac_ground_truth_logs"  # UAC generates ground truth dynamics

  social_model:
    core_network:
      type: "GRU"
      hidden_dim: 128
    inputs:
      use_public_cues: true         # Framework pattern: UAC public cues
      use_family_channel: true      # (Townlet Town: family relationships)
      history_window: 12
    heads:
      goal_distribution: { dim: 16 }  # maps to goal_vector_dim
      next_action_dist:  { dim: 6 }   # maps to action_space_dim
    optimizer: { type: "Adam", lr: 0.0001 }
    pretraining:
      objective: "ctde_intent_prediction"
      dataset: "uac_ground_truth_logs"

  hierarchical_policy:
    meta_controller:                # Framework pattern: strategic goal selection
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        goal_output: { dim: 16 }    # goal_vector_dim
    controller:                     # Framework pattern: action selection
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        action_output: { dim: 6 }   # action_space_dim
    optimizer: { type: "Adam", lr: 0.0003 }
    pretraining:
      objective: "behavioural_cloning"
      dataset: "v1_agent_trajectories"  # (Townlet Town: previous agent runs)
```

**Why Layer 2 matters**:

**Reproducibility**: The mind is rebuildable in any controlled environment, not dependent on an individual developer's workstation.

**Experimental control**: Module swaps become controlled experiments ("same cognitive_topology, different world_model internals").

**Governance transparency**: If someone quietly changed the optimizer or widened the GRU and then claimed "no behavioral change expected", governance can call nonsense on that. Layer 2 makes such changes visible.

**Interface contracts**: The `interfaces` section enforces dimensional compatibility. If perception outputs 128-dim belief but policy expects 256-dim input, the factory catches this **before runtime**. No silent broadcasting, no mystery glue.

**Framework vs Instance**: Network architectures (CNN, GRU, MLP) and interface contracts are framework-level. Specific action_space_dim (6 for Townlet Town's movement+interact), pretraining datasets (v1_agent_trajectories), and module choices (use_family_channel for Townlet Town families) are instance-specific.

---

## 2.3 Layer 3: execution_graph.yaml

**Audience**: Safety teams, auditors, engineers debugging cognition at 2am
**Nickname**: The think loop

Layer 3 defines the actual cognition pipeline the framework runs every tick. This is the part that most RL projects pretend is "obvious" and never write down. **We write it down.**

**Structure**: A DAG (Directed Acyclic Graph) of named steps with symbolic bindings (@references).

**Example configuration**:

```yaml
inputs:
  - "@graph.raw_observation"
  - "@graph.prev_recurrent_state"

steps:
  perception_packet:
    node: "@modules.perception_encoder"     # Symbolic binding (Layer 2 module)
    inputs:
      - "@graph.raw_observation"
      - "@graph.prev_recurrent_state"

  belief_distribution:
    node: "@utils.unpack"
    input: "@steps.perception_packet"
    key: "belief"

  new_recurrent_state:
    node: "@utils.unpack"
    input: "@steps.perception_packet"
    key: "state"

  policy_packet:
    node: "@modules.hierarchical_policy"
    inputs:
      - "@steps.belief_distribution"
      - "@services.world_model_service"     # Service binding (callable module)
      - "@services.social_model_service"

  candidate_action:
    node: "@utils.unpack"
    input: "@steps.policy_packet"
    key: "action"

  panic_adjustment:
    node: "@modules.panic_controller"
    inputs:
      - "@steps.candidate_action"
      - "@config.L1.panic_thresholds"      # Symbolic binding (Layer 1 config)
    outputs:
      - "panic_action"
      - "panic_reason"                     # Logged for telemetry

  final_action:
    node: "@modules.EthicsFilter"
    inputs:
      - "@steps.panic_adjustment.panic_action"
      - "@config.L1.compliance.forbid_actions"
    outputs:
      - "action"
      - "veto_reason"                      # Logged for governance audit

outputs:
  final_action: "@steps.final_action.action"
  new_recurrent_state: "@steps.new_recurrent_state"
```

**In plain language**:

1. **Perception** digests what the agent currently sees + its memory from last tick, producing:
   - Belief about the world and itself (`belief_distribution`)
   - Updated recurrent state (memory for next tick)

2. **Hierarchical Policy** decides: "Given my current strategic goal, given what I think the world is, given what I think will happen next if I try X (via `@services.world_model_service`), and given what I think other agents are about to do (via `@services.social_model_service`), here's what I want to do now."

3. **Panic Controller** looks at bars (energy, health - Townlet Town bars) versus panic thresholds from Layer 1. If in crisis, it **can override** the policy's `candidate_action` with an emergency survival action ("call_ambulance", "go_to_bed_now"). That override is logged with `panic_reason`.

4. **EthicsFilter** takes that (possibly panic-adjusted) action and enforces Layer 1 compliance. If the action is forbidden (e.g., "steal"), EthicsFilter **vetoes it**, substitutes something allowed, and logs `veto_reason`. **EthicsFilter is final**. Panic cannot authorize illegal behavior. This ordering is governance policy, not just code order.

5. **Graph outputs**:
   - `final_action`: The one that actually gets sent into the world
   - `new_recurrent_state`: What the agent will remember next tick

**Why Layer 3 matters**:

**Explicit causal chain**: We can prove "panic, then ethics, then action" with configuration, not "trust us".

**Governance as code**: It defines who is actually in charge of the body at each step. If someone tries to sneak in "panic can bypass ethics if health < 5 percent", that changes the execution graph, therefore changes the cognitive hash, therefore is detectable in provenance logs.

**Debuggability**: When an agent does something unexpected, engineers can trace through the DAG to see which step produced the decision and why.

**Framework pattern**: The DAG structure, symbolic bindings (@modules, @config, @services), and panic→ethics ordering are framework-level. The specific modules (panic_controller, EthicsFilter) and their inputs (panic_thresholds, forbid_actions) are configured per universe instance.

**Put simply**: Layer 3 is the mind's wiring diagram, in writing, with order-of-operations as governance, not folklore.

---

**Summary**: BAC specifies Software Defined Agents through three layers:
- **Layer 1**: What can the agent do (behavior contract)
- **Layer 2**: How are modules built (architecture blueprint)
- **Layer 3**: In what order does cognition run (think-loop DAG)

Together, they make agent minds inspectable, diffable, and enforceable.

---
