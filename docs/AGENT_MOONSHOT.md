# **High Level Design: Hamlet v2.0 "Smart Collection" Architecture**

**Document Version:** 1.0
**Date:** November 1, 2025

### 1. 🚀 Executive Summary

This document details the v2.0 architecture for the Hamlet agent, a fundamental refactor from the current v1.0 monolithic system. This new architecture, codenamed the **"Smart Collection,"** migrates the agent from a "model-free" (reactive) paradigm to a "model-based" (predictive) framework.

This v2.0 architecture is explicitly designed to be a **research platform** capable of solving the "moonshot" roadmap goals (Phases 4-8), including Non-Stationarity, Partial Observability (POMDP), Multi-Agent Theory of Mind (ToM), and Hierarchical Reinforcement Learning (HRL).

It achieves this by decomposing the agent's "brain" into four specialized, interacting modules:

1. **Module A: Perception Encoder** (Solves POMDP)
2. **Module B: World Model** (Solves Non-Stationarity & Physics)
3. **Module C: Social Model** (Solves Multi-Agent ToM)
4. **Module D: Hierarchical Policy** (Solves HRL & Decision-Making)

These modules communicate via a "universal language" of well-defined data structures (chiefly the `BeliefDistribution`). This design's primary goal is to not only achieve the "moonshot" roadmap but to do so in a **pedagogically transparent** and **debuggable** way, transforming the agent from a "black box" into an interpretable teaching tool.

### 2. 🧠 Core Philosophy & Goals

The "significant tension" between our current v1.0 agent and our v2.0 roadmap goals is the tension between two different types of learning:

* **v1.0 ("The Flashcard Memorizer"):** A brilliant reactive, model-free agent that *memorizes* the value ($Q(s, a)$) of actions in states it has seen. It excels at stable, defined problems.
* **v2.0 ("The Grammar Engine"):** A predictive, model-based agent that learns the *underlying rules ("grammar")* of its world. By learning the "physics" (World Model) and "psychology" (Social Model), it can generate novel, intelligent behavior for situations it has *never* seen.

**Our "Pedagogical First" principle dictates that this v2.0 system must be interpretable.** The core deliverable is not just an agent that *works*, but an agent whose *thought process* can be inspected, debugged, and used for teaching.

### 3. v1.0 Baseline Architecture (For Context)

* **System:** Monolithic, model-free Deep Q-Network (DQN).
* **Policy:** A single network (`RecurrentSpatialQNetwork`) that maps observations directly to Q-values.
* **Strengths:** Excellent for Phase 3.5 (stable, single-agent, full/partial observability).
* **Limitations:** Fundamentally unequipped to handle non-stationarity (Phase 5), complex agent-agent interactions (Phase 4), or hierarchical planning (Phase 7) in a robust or interpretable way.

-----

### 4. v2.0 "Smart Collection" Detailed Architecture

The v2.0 agent is a system of four "expert" modules that collaborate to produce a single, intelligent action.

#### Module A: Perception Encoder (The "Eyes")

* **Roadmap Goal:** Phase 6 (POMDP Extension)
* **Purpose:** To solve partial observability. It answers the question: "What is the *true* state of the world, given my limited 5x5 view?"
* **Architecture:** A recurrent network is essential.
    1. `CNN` processes the 5x5 grid view.
    2. `MLP` processes the vector inputs (self-meters, time).
    3. The concatenated features + `prev_action` are fed into an **LSTM/GRU**.
    4. The output of the LSTM/GRU is passed to two heads to generate the parameters for the `BeliefDistribution` (e.g., `mean` and `log_std` for a Gaussian).
* **Output:** `BeliefDistribution` (see Sec. 5) - the "universal language."

#### Module B: World Model (The "Physics Engine")

* **Roadmap Goal:** Phase 5 (Non-Stationarity & Continuous Learning)
* **Purpose:** To learn the "physics" of the environment. It answers the question: "If I'm in this state and I take this action, what will happen to my meters, position, and reward?"
* **Architecture:** A predictive, feed-forward model.
    1. **Input:** `BeliefDistribution`, `ActionSequence`.
    2. **Core:** An autoregressive "step" model that predicts `t+1` from `t`.
    3. **Heads:** Predicts `next_state`, `next_reward`, `next_done`.
    4. **Refinement:** Includes a **`Value Head`**. This allows the model to be pre-trained to predict not just the next state, but the *value* of that state, aligning it with RL goals from day one.
* **Prerequisite:** This module *must* be trained on a data-driven environment. The hardcoded logic in `vectorized_env.py` (Actions \#1, \#12) must be refactored to be configuration-driven (e.g., YAML) so the World Model can learn these rules.

#### Module C: Social Model (The "Psychology Engine")

* **Roadmap Goal:** Phase 4 (Multi-Agent Competition & Theory of Mind)
* **Purpose:** To learn the "psychology" of other agents. It answers the question: "What does Agent 2 *intend* to do?"
* **Core Challenge:** Solves the "non-predictable complexity" (non-stationarity) of other agents.
* **Our "Detective" Solution:**
    1. **Environment "Cheat" (Cue-Based):** The environment is refactored to generate **`public_cues`** correlated with an agent's hidden state (e.g., `is_slumped`, `is_wearing_work_outfit`).
    2. **Training "Cheat" (CTDE):** The module is trained via **Centralized Training for Decentralized Execution**.
* **Architecture:**
  * **Input:** The `social_cues` component of the `BeliefDistribution`.
  * **Core:** An LSTM/GRU that processes a *history* of an opponent's `(public_cues, action)` pairs.
  * **Output:** `SocialPrediction` (see Sec. 5), a prediction of the opponent's *hidden* `Goal` or `intent`.

#### Module D: Hierarchical Policy (The "CEO")

* **Roadmap Goal:** Phase 7 (Hierarchical RL)

* **Purpose:** To be the agent's "will" or decision-maker. It explicitly separates long-term strategy from short-term tactics.

* **Architecture:** This is explicitly two separate sub-modules.

* **D1: Meta-Controller (The "Strategist")**

  * **Timescale:** Slow (e.g., runs every 50 steps or when a goal is complete).
  * **Inputs:** `BeliefDistribution` (self), `ImaginedFutures` (physics), `SocialPrediction` (others).
  * **Task:** Asks, "What is our *strategic goal* right now?"
  * **Output:** An explicit, interpretable `Goal` struct.

* **D2: Controller (The "Operator")**

  * **Timescale:** Fast (runs every game tick).
  * **Inputs:** `BeliefDistribution`, `ImaginedFutures`, `SocialPrediction`, AND the current `Goal`.
  * **Task:** Asks, "Given our *goal*, what is the single best *primitive action* to take *right now*?"
  * **Output:** A `PrimitiveAction` (e.g., `up`, `down`, `interact`).

### 5. 📦 Key Data Structures (The "Universal Language")

These are the non-negotiable API contracts between modules, defined as Python-style `dataclass` structs.

```python
# The "Universal Language" - Output by Module A
@dataclass
class BeliefDistribution:
    """The full probabilistic belief of the agent about the hidden world state."""
    # Parameters for a Gaussian distribution (e.g., [mean, log_std])
    # This represents the agent's belief over all hidden variables.
    distribution_params: torch.Tensor 
    distribution_type: str = "gaussian"
    
    # Raw data for other modules
    known_state: torch.Tensor      # Ground truth self-data (meters, pos, money)
    social_cues: torch.Tensor      # Raw public cues of other agents
    recurrent_state: Tuple         # The (h, c) of the LSTM for the next tick

# Output by Module B
@dataclass
class ImaginedFuture:
    """A single "dreamt" trajectory from the World Model."""
    action_sequence: list[int]
    predicted_states: list[torch.Tensor]
    predicted_rewards: list[float]
    predicted_dones: list[bool]
    predicted_values: list[float]  # Predicted value of each state
    confidence: float              # How much the WM trusts this "dream"
    uncertainty_reason: Optional[str] = None # e.g., "novel_state", "high_variance"

# Output by Module C
@dataclass
class SocialPrediction:
    """The agent's belief about *another* agent's hidden mind."""
    agent_id: int
    predicted_goal_dist: torch.Tensor # Belief over the opponent's goals
    predicted_next_action_dist: torch.Tensor
    confidence: float              # 0.0-1.0, standardized confidence
    uncertainty_reason: Optional[str] = None # e.g., "conflicting_cues"

# Output by Module D1 (Meta-Controller)
@dataclass
class Goal:
    """An explicit, interpretable, long-term strategic goal."""
    goal_type: str                  # e.g., "SURVIVAL", "THRIVING", "SOCIAL"
    priority: float                 # How important is this goal?
    target_meters: Dict[str, float] # e.g., {"energy": 80, "money": 100}
    termination_condition: Callable # Lambda func to check if goal is met
```

### 6. 🔁 System Data Flow (A Single `agent.think()` Step)

This is the end-to-end operational flow for a single game tick.

1. **SENSE:** `raw_observation_t` and `prev_recurrent_state` enter the system.
2. **PERCEIVE (Module A):** The `PerceptionEncoder` processes this data.
3. **BROADCAST:** Module A outputs the `BeliefDistribution_t`. This is the **single source of truth** for all other modules.
4. **PREDICT (Parallel):**
      * **Module B (World Model)** is called by the Policy (Module D) to "imagine" the outcomes of several candidate action sequences, generating a list of `ImaginedFutures`.
      * **Module C (Social Model)** processes the `BeliefDistribution_t.social_cues` to generate `SocialPrediction_t` for all observed agents.
5. **DECIDE (Module D):**
      * **D1 (Meta-Controller):** Checks if the `current_Goal` is complete (via `termination_condition`). If yes, it uses all inputs (`Belief`, `Futures`, `Social`) to select a new `Goal`.
      * **D2 (Controller):** Uses all inputs and the `current_Goal` to select the single, optimal `PrimitiveAction_t`.
6. **ACT:** The `PrimitiveAction_t` is sent to the environment. The `BeliefDistribution_t.recurrent_state` is saved for the next tick.

### 7. 🛠️ Training & Implementation Strategy

This architecture is complex. We will **not** build it in one "big bang." We will use a two-pronged strategy of Modular Pre-training and Incremental Migration.

#### 7.1 Part 1: Modular Pre-Training (The "Curriculum")

Before any end-to-end RL, each module is pre-trained on a simpler, supervised task to make it "smart" from the start.

| Module | Training Goal | Pre-Training Task | Loss Function |
| :--- | :--- | :--- | :--- |
| **A: Perception** | Learn to see & compress | Autoencoding & Next-Frame Prediction | `Reconstruction Loss + Prediction Loss` |
| **B: World Model** | Learn "physics" & "value" | Supervised Consequence Prediction | `MSE(pred_state, real_state) + MSE(pred_reward, real_reward) + MSE(pred_value, real_value)` |
| **C: Social Model** | Learn "psychology" | CTDE "Telepathic" Cheat | `MSE(pred_opponent_goal, real_opponent_goal)` |
| **D: Policy** | Learn "common sense" | Behavioral Cloning (from v1.0 agent) | `CrossEntropyLoss(policy_action, expert_action)` |

#### 7.2 Part 2: Incremental Migration (The "Scaffolding")

This is the engineering migration path to de-risk the refactor.

* **Phase 1: Foundation (Prerequisites)**

  * Implement the explicit data structures (Sec. 5).
  * Refactor the environment for `public_cues` (for Module C).
  * Refactor environment "physics" (Actions \#1, \#12) to be data-driven (for Module B).

* **Phase 2: v1.5 (Perception Test)**

  * **Architecture:** `Module A (Perception)` + v1.0 `Q-Network`.
  * **Logic:** The Q-Network's input is replaced. Instead of `raw_observation`, it now takes the `BeliefDistribution.distribution_params`.
  * **Goal:** Validate that `Module A` can solve the POMDP task and is a viable replacement.

* **Phase 3: v1.7 (Planning Test)**

  * **Architecture:** `v1.5` + `Module B (World Model)`.
  * **Logic:** Use the `ImaginedFutures` from the pre-trained World Model to enhance the Q-learning update (e.g., Dyna-style multi-step rollouts).
  * **Goal:** Validate that `Module B` can accelerate learning and adapt to non-stationarity.

* **Phase 4: v2.0 (Full System Integration)**

  * **Architecture:** All pre-trained modules are integrated.
  * **Logic:** The v1.0 `Q-Network` is fully *removed* and *replaced* by the pre-trained `Module D (Hierarchical Policy)`.
  * **Goal:** The full "Smart Collection" is now active and ready for final, end-to-end fine-tuning with RL.

### 8. 🎓 Pedagogical Payoff & UI Strategy

This architecture's primary benefit—beyond performance—is **interpretability**. This is how we achieve our "Pedagogical First" goal.

#### 8.1 The Debugging Workflow (The "Payoff")

This is the "product" of the v2.0 architecture: a fully transparent, debuggable "mind."

**Example Debugging Scenario:** "The agent is starving but is standing at the 'job' affordance. Why?"

```python
def debug_agent_failure(trace):
    
    # 1. Is it 'blind'?
    if trace.belief_state.uncertainty['my_meters'] > 0.9:
        print("DIAGNOSIS: PERCEPTION (A) FAILURE.")
        print("Agent is 'confused' and doesn't know its own meters are low.")

    # 2. Is it 'delusional' about physics?
    elif trace.imagined_futures['fridge_path'].reward < 0:
        print("DIAGNOSIS: WORLD MODEL (B) FAILURE.")
        print("Agent 'believes' the fridge has a negative reward. Its 'physics' is wrong.")
        
    # 3. Is it 'scared' or 'polite'?
    elif trace.social_prediction['agent_2'].predicted_goal == 'use_fridge':
        print("DIAGNOSIS: SOCIAL MODEL (C) INTERFERENCE.")
        print("Agent 'thinks' Agent 2 is going to the fridge, so it 'yielded'.")

    # 4. Is it just 'obsessed'?
    elif trace.policy_goal.goal_type == 'THRIVING' and trace.policy_goal.priority > 0.9:
        print("DIAGNOSIS: POLICY (D) FAILURE.")
        print("Agent 'knows' it's starving, but its 'Meta-Controller' has decided 'money' is more important.")
```

#### 8.2 Pedagogical Lab Curriculum

This architecture enables a new curriculum based on debugging the agent's "mind."

* **Lab 1: Perception Failures:** "Agent keeps bumping into walls. Examine the `BeliefDistribution`'s uncertainty. Why is it 'lost'?"
* **Lab 2: World Model Limitations:** "Agent tries to use the 'job' affordance at 3 AM. Analyze its `ImaginedFuture` for this action. What did it *think* would happen?"
* **Lab 3: Social Misunderstandings:** "Agent avoids an empty bed. Analyze the `SocialPrediction`. Which *public cue* from Agent 2 did it misinterpret?"
* **Lab 4: Goal Conflicts:** "Agent starves while having high 'money'. Debug the `Meta-Controller`'s `Goal` selection logic. Is its 'priority' wrong?"

#### 8.3 Progressive Complexity UI

To mitigate cognitive overload for students, the demo UI will be configurable to reveal modules progressively.

```python
# UI config to progressively enable module visualizations
AGENT_COMPLEXITY_LEVELS = {
    'beginner': ['policy'], # Show "Agent chose to 'go to job'"
    'intermediate': ['perception', 'policy'], # Show "Agent *believes* its energy is 50, so it chose 'go to bed'"
    'advanced': ['perception', 'world_model', 'policy'], # Show "Agent *imagined* the job would give +$20..."
    'research': ['perception', 'world_model', 'social_model', 'policy'] # Show full trace
}
```

### 9. ⚠️ Risks & Mitigation

| Risk | Mitigation |
| :--- | :--- |
| **Training Complexity** | **Modular Pre-training.** We solve 4 simple problems, not 1 complex one. |
| **Computational Overhead** | **Selective Imagination.** Module D only asks Module B to "imagine" a few high-priority candidate paths, not all of them. |
| **Belief Divergence** | **Observation Correction.** The `BeliefDistribution` is corrected by new `raw_observations` on every tick, preventing it from "dreaming" forever. |
| **Cognitive Overload (for Student)** | **Progressive UI (Sec 8.3).** The UI will match the student's learning, from "beginner" to "research" mode. |

### 10. 📊 Success Metrics

#### Technical Success

* [ ] **v1.5:** Agent matches or exceeds v1.0 performance in POMDP settings.
* [ ] **v1.7:** World Model predictions (for `known_state` & `reward`) achieve \>80% accuracy on a test set.
* [ ] **v2.0:** All modules are integrated and end-to-end training is stable and converges.
* [ ] **Demo:** The real-time visualization successfully renders all module outputs (`Belief`, `Goal`, etc.) without crashing.

#### Pedagogical Success

* [ ] **Lab Completion:** Students can successfully use the debugging UI to diagnose the 4 "Lab" scenarios (Sec 8.2).
* [ ] **Conceptual Understanding:** Students can verbally explain the "flashcard" vs. "grammar" analogy and identify which module is responsible for which behavior.
* [**Student-led Extension:** Students begin to extend the modules (e.g., custom `Goal` selectors in Module D) for their own projects.

### 11. 🔬 Future Research Projects Enabled

This architecture is not an endpoint; it's a platform. It unlocks numerous research avenues:

* "Extending the World Model to handle *novel* affordances without re-training."
* "Improving the Social Model's inference capabilities with more complex 'cues'."
* "Investigating emergent `Goal` selection strategies in the Meta-Controller."
* "Adding an explicit communication channel for Phase 8, using the `SocialPrediction` as a foundation."
