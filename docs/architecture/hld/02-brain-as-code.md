## 2. Brain as Code (BAC): the mind

---

Brain as Code is how we define Townlet's mind as something we can inspect, diff, and enforce.

The BAC stack is three YAML layers. Together, they are the Software Defined Agent.

Change the YAMLs, you change the mind. Snapshot the YAMLs, you freeze the mind. Hash the snapshot, you can prove which mind took which action.

### 2.1 Layer 1: cognitive_topology.yaml

Audience: governance, instructors, simulation designers
Nickname: the character sheet

Layer 1 is the behaviour contract and safety envelope for a specific agent instance in a specific run.

It answers questions like:

- Is social reasoning enabled?
- How far ahead is it allowed to plan?
- When does it panic and override normal plans?
- What is it allowed to do, and what is absolutely forbidden?
- How greedy, anxious, curious, agreeable is it, as dials not as fairy dust?
- Is it allowed to narrate its motives in the UI?

Example fields you'll actually see:

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
  use_family_channel: true       # allow private in-group signalling

hierarchical_policy:
  meta_controller_period: 50     # How often to reconsider high-level goal
  allowed_goals:
    - SURVIVAL
    - THRIVING
    - SOCIAL

personality:
  greed: 0.7                     # money drive
  agreeableness: 0.3             # harmony vs confrontation
  curiosity: 0.8                 # exploration drive
  neuroticism: 0.6               # risk aversion / anxiety

panic_thresholds:
  energy: 0.15                   # if energy < 15 percent => emergency mode
  health: 0.25
  satiation: 0.10

compliance:
  forbid_actions:
    - "attack"
    - "steal"
  penalize_actions:
    - { action: "shove", penalty: -5.0 }

introspection:
  publish_goal_reason: true      # Should the agent explain itself in UI?
  visible_in_ui: "research"      # beginner | intermediate | research
```

How Layer 1 is used in runtime:

- `panic_thresholds` tells panic_controller when it is permitted to override the normal plan and focus solely on survival.
- forbid_actions tells EthicsFilter what is never allowed, even if the agent is dying.
- personality feeds into the hierarchical policy's goal choice, so "greed: 0.7" really does mean "money-seeking wins arguments inside its head."
- publish_goal_reason controls whether we surface "I'm going to work because we need money" to the human observer.

Layer 1 is what policy signs off on. It's the file you show when someone asks "what kind of mind did you put in this world?"

If you change Layer 1 between runs (for example, allow STEAL, or lower panic threshold, or turn social modelling off), that's not the same agent any more. That must produce a new cognitive hash.

### 2.2 Layer 2: agent_architecture.yaml

Audience: engineers, grad students, researchers
Nickname: the blueprint

Layer 2 is the internal build sheet for those faculties. If Layer 1 says "there is a World Model and it's allowed to plan 6 ticks ahead", Layer 2 says "the World Model is a 2-layer MLP with 256 units, these heads, trained on this dataset, with Adam at this learning rate".

This file covers:

- network types (CNN, GRU, MLP, etc),
- hidden sizes, head dimensions,
- interface contracts between modules,
- optimiser types and learning rates,
- pretraining objectives and datasets.

It enforces discipline so you can swap modules and reproduce experiments without mystery glue.

For example:

```yaml
interfaces:
  belief_distribution_dim: 128      # Perception output
  imagined_future_dim: 256          # World Model summary
  social_prediction_dim: 128        # Social Model summary
  goal_vector_dim: 16               # Meta-controller goal embedding
  action_space_dim: 6               # {up,down,left,right,interact,wait}

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
    pretraining:
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
      dataset: "uac_ground_truth_logs"

  social_model:
    core_network:
      type: "GRU"
      hidden_dim: 128
    inputs:
      use_public_cues: true
      use_family_channel: true
      history_window: 12
    heads:
      goal_distribution: { dim: 16 }  # maps to goal_vector_dim
      next_action_dist:  { dim: 6 }   # maps to action_space_dim
    optimizer: { type: "Adam", lr: 0.0001 }
    pretraining:
      objective: "ctde_intent_prediction"
      dataset: "uac_ground_truth_logs"

  hierarchical_policy:
    meta_controller:
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        goal_output: { dim: 16 }  # goal_vector_dim
    controller:
      network: { type: "MLP", layers: [256, 128], activation: "ReLU" }
      heads:
        action_output: { dim: 6 } # action_space_dim
    optimizer: { type: "Adam", lr: 0.0003 }
    pretraining:
      objective: "behavioural_cloning"
      dataset: "v1_agent_trajectories"
```

Why Layer 2 matters:

- It makes the mind rebuildable in any controlled environment, not dependent on an individual developer's workstation.
- It makes module-swaps a controlled experiment ("same cognitive_topology, different world_model internals").
- It makes it obvious if someone quietly changed the optimiser or widened the GRU and then claimed "no behavioural change expected". Governance can now call nonsense on that.

### 2.3 Layer 3: execution_graph.yaml

Audience: safety, audit, you at 2am
Nickname: the think loop

Layer 3 is the actual cognition pipeline Townlet runs every tick. This is the part that most RL projects pretend is "obvious" and never write down. We write it down.

It's a DAG of named steps. It says:

```yaml
inputs:
  - "@graph.raw_observation"
  - "@graph.prev_recurrent_state"

steps:
  perception_packet:
    node: "@modules.perception_encoder"
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
      - "@services.world_model_service"
      - "@services.social_model_service"

  candidate_action:
    node: "@utils.unpack"
    input: "@steps.policy_packet"
    key: "action"

  panic_adjustment:
    node: "@modules.panic_controller"
    inputs:
      - "@steps.candidate_action"
      - "@config.L1.panic_thresholds"
    outputs:
      - "panic_action"
      - "panic_reason"

  final_action:
    node: "@modules.EthicsFilter"
    inputs:
      - "@steps.panic_adjustment.panic_action"
      - "@config.L1.compliance.forbid_actions"
    outputs:
      - "action"
      - "veto_reason"

outputs:
  final_action: "@steps.final_action.action"
  new_recurrent_state: "@steps.new_recurrent_state"
```

In English:

1. Perception digests what the agent can currently see + its memory of last tick, and produces:

   - its belief about the world and itself (belief_distribution),
   - an updated recurrent state.

2. Hierarchical Policy says "given my current strategic goal, given what I think the world is, given what I think will happen next if I try X, and given what I think other agents are about to do, here's what I want to do now".

3. panic_controller looks at bars like energy and health vs the panic thresholds from Layer 1. If we're in crisis, it can override the policy's candidate_action with an emergency survival action ("call_ambulance", "go_to_bed_now"). That override is logged with panic_reason.

4. EthicsFilter takes that (possibly panic-adjusted) action and enforces Layer 1 compliance. If the action is forbidden (eg "steal"), EthicsFilter vetoes it, substitutes something allowed, and logs veto_reason. EthicsFilter is final. Panic cannot authorise illegal behaviour. This ordering is governance policy, not just code order.

5. The graph outputs:

   - final_action (the one that actually gets sent into the world),
   - new_recurrent_state (what the agent will remember next tick).

Why Layer 3 matters:

- It makes the causal chain explicit. We can prove "panic, then ethics, then action", not "trust us".
- It defines who is actually in charge of the body at each step.
- It's part of the cognitive hash. If someone tries to sneak in "panic can bypass ethics if health < 5 percent", that changes the execution graph, therefore changes the hash, therefore is detectable.

Put simply: Layer 3 is the mind's wiring diagram, in writing, with order-of-operations as governance, not folklore.

---
