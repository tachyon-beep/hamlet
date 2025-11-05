# 4. Curriculum Design

**Document Type**: Overview + Component Spec
**Status**: Draft (L0-L3 implemented, L4-L8 planned)
**Version**: 2.5
**Last Updated**: 2025-11-05
**Parent Document**: TOWNLET_HLD.md

---

## SECTION 4: CURRICULUM DESIGN

### 4.1 Curriculum Philosophy

The Townlet Town curriculum (L0–L8) progressively removes assumptions, taking agents from god-mode tutorial to fully realistic multi-agent coordination.

**Note**: This curriculum is an instance-specific implementation for the "Townlet Town" environment. The framework itself is domain-agnostic and could support alternative universes (factories, villages, trading bots, etc.) with different curriculum progressions.

**Core principle**: Each level removes exactly one simplifying assumption, forcing agents to develop one new capability.

**Pedagogical scaffold**: Early curriculum levels (L0-L3) provide "training wheels" — full observability, perfect knowledge — to teach fundamental relationships before adding complexity.

**Realism progression**: Later curriculum levels (L4-L8) remove scaffolding piece by piece until agents operate under human-realistic constraints.

**Implementation status**:
- ✅ **L0-L3**: Implemented and validated (Townlet v2.5)
- 🚧 **L4**: Partial observability (POMDP) - in development
- 📋 **L5-L8**: Planned (temporal mechanics, social reasoning, emergent communication)

#### The Assumption Removal Ladder

| Level | Removed Assumption | New Capability Required | Architecture |
|-------|-------------------|------------------------|--------------|
| **L0** | None (tutorial) | "Affordances fix bars" | SimpleQNetwork |
| **L1** | None | "Money enables survival" | SimpleQNetwork |
| **L2** | None | "Multi-resource loops" | SimpleQNetwork |
| **L3** | None | "Optimize quality of life" | SimpleQNetwork |
| **L4** | Perfect geography | Exploration + memory | LSTM required |
| **L5** | Temporal omniscience | Temporal planning | LSTM + time input |
| **L6** | Omniscience about others | Social reasoning via cues | Module A-D |
| **L7** | — | Rich social prediction | Module A-D |
| **L8** | Pre-shared language | Emergent communication | Module A-D + channel |

**Design insight**: You can't learn social reasoning (Curriculum Level 6) until you can navigate (Level 4), and you can't navigate without knowing what affordances do (Levels 0-3). The order is pedagogically necessary, not arbitrary.

---

### 4.2 Level-by-Level Breakdown

**Implementation note**: Levels 0-3 are fully implemented and validated. Levels 4-8 are documented here as design specifications for future implementation.

#### L0: The "Two Unsolvable Problems" [✅ Implemented]

**World configuration**:

```yaml
grid_size: [5, 5]
affordances: ["Bed"]  # only Bed exists
observability: "full"  # agent sees everything
initial_money: 0.50  # $50
max_age_ticks: 200   # short episodes for rapid learning
```

**What the agent sees**:

```python
observation = {
    'bars': [1.0, 1.0, 1.0, 0.5, 0.7, 1.0, 1.0, 0.5],  # all 8 bars visible
    'position': [2, 2],
    'all_affordance_locations': {'Bed': [1, 1]},  # perfect map
    'grid_size': [5, 5]
}
```

**The learning experience**:

1. Energy drops (base_depletion = 0.005 per tick)
2. `r = energy × health` drops
3. RND (curiosity) drives agent to explore
4. Agent discovers Bed at (1,1)
5. Tries INTERACT → energy restored → `r` jumps up
6. Massive positive TD error
7. Agent learns: "Bed → energy ↑"

8. Agent hammers this new policy: Bed, Bed, Bed
9. After 10 uses: money = $0 (Bed costs $5 per use)
10. Can't afford Bed anymore
11. Energy drops, agent dies
12. Terminal: early death penalty, very low score

13. Simultaneously: satiation dropping
14. Satiation < 0.2 → cascade activates
15. Health decaying faster
16. No way to stop it (Fridge doesn't exist yet)
17. Agent dies of starvation even if it has energy

**The "Aha!" moment**: "I've solved energy, but now I face two new problems: I'll either die of starvation (no Fridge) or die when I run out of money (can't afford Bed)."

**Key learning**:

- `V(low_energy)` is terrible
- `V(low_money)` is terrible
- `V(low_satiation)` is terrible
- But only one solution (Bed) exists
- This is unsolvable by design

**Why it's L0**: Teaches the bare minimum — "affordances affect bars, bars affect reward."

**Human observer exception**: Full observability is scaffolding. A human learning a new town would be given a map. This is pedagogically honest.

---

#### L1: The "Economic Problem" [✅ Implemented]

**World configuration**:

```yaml
grid_size: [5, 5]
affordances: ["Bed", "Job"]  # ← Job added
observability: "full"
initial_money: 0.50
max_age_ticks: 200
```

**What's new**: Job affordance provides income

```yaml
- id: "job"
  interaction_type: "multi_tick"
  required_ticks: 8
  effects_per_tick:
    - { meter: "money", amount: 0.10 }  # earn $10/tick
    - { meter: "energy", amount: -0.05 }  # costs energy
  operating_hours: [9, 18]
```

**The learning experience**:

1. Agent inherits L0 knowledge: "Bed fixes energy"
2. Agent runs out of money (same as L0)
3. Now in `low_money` state (previously hopeless)
4. RND drives exploration again
5. Agent discovers Job at (6,6)
6. Tries INTERACT → money increases!
7. Agent learns second sub-policy: "Job → money ↑"

8. Agent creates stable loop: Job → earn money → Bed → restore energy → Job
9. This is economically stable (can pay for Bed indefinitely)
10. Agent still dies of starvation (satiation still dropping)
11. But survives much longer than L0

**The "Aha!" moment**: "The Job solves the money problem! I can now afford the Bed forever. But I still can't solve the starvation problem."

**Key learning**:

- Money is instrumentally valuable (enables affordance access)
- Work-rest cycles are necessary
- But still can't reach retirement (satiation cascade kills them)

**Why it's L1**: Isolates economic reasoning. Agent masters money/energy loop before tackling food.

**Expected behavior by episode 100**: Reliable work-rest cycles, but 100% death by starvation.

---

#### L2: The "First Stable Loop" [✅ Implemented]

**World configuration**:

```yaml
grid_size: [5, 5]
affordances: ["Bed", "Job", "Fridge"]  # ← Fridge added
observability: "full"
initial_money: 0.50
max_age_ticks: 500  # longer episodes now survivable
```

**What's new**: Fridge prevents starvation

```yaml
- id: "fridge"
  interaction_type: "instant"
  costs:
    - { meter: "money", amount: 0.04 }  # $4
  effects:
    - { meter: "satiation", amount: 0.40 }  # +40% satiation
```

**The learning experience**:

1. Agent has L1 knowledge: reliable Job ↔ Bed loop
2. Satiation drops as always
3. Cascade activates, health starts decaying
4. RND drives agent to new affordance: Fridge
5. Tries INTERACT → satiation jumps up
6. Cascade stops → health stabilizes
7. Agent learns third sub-policy: "Fridge → satiation ↑ → health stable"

8. Agent combines all three sub-policies:
   - Low energy → Bed
   - Low money → Job
   - Low satiation → Fridge

9. This three-way loop is infinitely stable
10. Agent can now reach retirement (age = 1.0)
11. Terminal reward: moderate score (survived, but minimal wellbeing)

**The "Aha!" moment**: "The Fridge solves the final problem! I can now survive indefinitely."

**Key learning**:

- Food prevents health decay (second-order relationship)
- Must balance three resources: energy, money, satiation
- Stable survival loop: Job → Fridge → Bed → repeat

**Why it's L2**: This is the minimum viable life. A "competent but miserable" agent can survive here indefinitely.

**Expected behavior by episode 200**: 70%+ retirement rate, but low terminal scores (wellbeing suffers).

**This is the baseline**: Any agent that can't master L2 is not viable for advanced levels.

---

#### L3: The "Full Simulation (Small Grid)" [✅ Implemented]

**World configuration**:

```yaml
grid_size: [5, 5]  # still small, fully visible
affordances: ["Bed", "Job", "Fridge", "Shower", "Bar", "Recreation",
              "Doctor", "Hospital", "Gym", "Labor", "Park", ...]  # all 15
observability: "full"
initial_money: 0.50
max_age_ticks: 1000  # full-length episodes
```

**What's new**: All tertiary affordances (mood, social, hygiene, fitness)

**The learning experience**:

1. Agent has L2 mastery: Job-Fridge-Bed loop
2. But retirement scores are still low
3. Why? Tertiary cascades are active:
   - Low hygiene → mood drops → energy drops faster
   - Low social → mood drops → energy drops faster
   - Low fitness → health drops faster (via modulation)

4. Agent notices: "I'm going to Bed more often than I used to"
5. World model learning: "Something is making energy decay faster"
6. Exploration: tries Shower → hygiene ↑
7. Observation: energy decay rate returns to normal
8. Learning: "Hygiene maintains energy efficiency"

9. Similar discovery for Bar (social), Gym (fitness), Recreation (mood)
10. Agent learns optimization: "These affordances don't directly give reward, but they prevent cascades, which makes survival cheaper"

**The "Aha!" moment**: "Going to the Bar isn't pointless fun — it's instrumentally useful because it prevents mood decay, which prevents energy decay, which means I need to work less."

**Key learning**:

- Tertiary meters matter (hygiene, social, fitness, mood)
- They affect survival through cascades (third-order relationships)
- There's an optimal allocation of time/money across all affordances
- Quality of life and survival efficiency are linked

**Why it's L3**: Teaches optimization and multi-objective reasoning before adding navigation/social complexity.

**Expected behavior by episode 500**:

- 80%+ retirement rate
- Higher terminal scores (wellbeing component improves)
- Agents discover "balanced life" strategies

**Graduation criteria**: Agent can reliably reach retirement with terminal score > 0.65.

---

#### L4: The "Fog of War" (Partial Observability) [🚧 In Development]

**World configuration**:

```yaml
grid_size: [8, 8]  # ← larger grid
visibility_radius: 2  # ← 5×5 window (agent can see 2 tiles in each direction)
affordances: [all 15]
observability: "partial"  # ← KEY CHANGE
initial_money: 0.50
max_age_ticks: 1200
```

**What the agent sees NOW**:

```python
observation = {
    'bars': [energy, health, ...],  # still see own state
    'position': [x, y],  # still know where they are
    'visible_grid': np.array([5, 5, N_AFFORDANCES]),  # ← only 5×5 window
    'time_of_day': hour,
    # ❌ NO all_affordance_locations (removed!)
}
```

**The learning experience**:

1. Agent tries L3 policy: "Go to Bed when energy low"
2. But: can't see Bed (it's outside the 5×5 window)
3. Policy fails: wanders randomly
4. Energy drops to 0 → dies
5. Terminal: massive negative reward (early death)

6. RND drives exploration (random curiosity-driven movement)
7. Eventually stumbles upon Bed at (2,1)
8. LSTM records: "At tick 50, I saw Bed at (2,1)"
9. Memory persists in hidden state

10. Later (tick 150): energy is low again
11. Agent is at position (7,6), can't see Bed
12. But LSTM's hidden state contains "Bed is at (2,1)"
13. Policy uses memory: "Navigate toward remembered location (2,1)"
14. Arrives at Bed, survives

**The "Aha!" moment**: "My policy is useless without memory. I must explore to build a mental map, then navigate from memory."

**Key learning**:

- Exploration is essential (RND-driven curiosity)
- LSTM is required (spatial memory)
- Mental map must be maintained across hundreds of ticks
- Navigation from memory is a new skill

**Architecture requirement**: SimpleQNetwork can no longer solve this. Must use RecurrentSpatialQNetwork (LSTM core).

**Why it's L4**: First removal of scaffolding. Agents must now discover the world themselves.

**Expected behavior**:

- Episodes 0-500: Terrible performance (learning to explore)
- Episodes 500-2000: Improving (building reliable mental maps)
- Episodes 2000+: Near-L3 performance (memory compensates for limited vision)

**Graduation criteria**: Agent can navigate to any affordance from any starting position, 80%+ success rate.

**Human observer validation**: "Can I see the whole town from any location?" NO (5×5 vision is realistic).

---

#### L5: The "9-to-5" (Temporal Constraints) [📋 Planned]

**World configuration**:

```yaml
grid_size: [8, 8]
visibility_radius: 2
affordances: [all 15, with operating_hours enforced]  # ← KEY CHANGE
observability: "partial"
temporal_mechanics: true  # ← enable time-of-day
initial_money: 0.50
max_age_ticks: 1200
```

**What's new**: Affordances have operating hours, and they're enforced

```yaml
- id: "job"
  operating_hours: [9, 18]  # 9am-6pm only

- id: "bar"
  operating_hours: [18, 28]  # 6pm-4am (wraps around)
```

**What the agent sees NOW**:

```python
observation = {
    'bars': [...],
    'position': [x, y],
    'visible_grid': [...],
    'time_of_day': 14,  # ← NEW: current hour [0-23]
}
```

**The learning experience**:

1. Agent has L4 mastery: can navigate from memory
2. At 3am: money is low, energy is moderate
3. Agent navigates to Job (from memory)
4. Tries INTERACT
5. Action is masked! (Job is closed at 3am)
6. Agent tries again... and again... stuck in loop
7. Wasting energy on failed interactions
8. Eventually runs out of energy and dies

9. Learning: "Job only works during certain hours"
10. Must correlate time_of_day with action success
11. Policy learns: "Don't go to Job at 3am, wait until 9am"

12. New strategy emerges: arrive at Job at 7am, WAIT until it opens at 9am
13. WAIT costs minimal energy (0.001 vs 0.005 for moving)
14. Efficient scheduling learned

**The "Aha!" moment**: "The WAIT action is strategic. Arriving early and waiting is better than arriving exactly at open (which requires perfect timing) or arriving late (which wastes work hours)."

**Key learning**:

- Time-of-day matters for action success
- Scheduling is as important as action selection
- WAIT is not useless, it's an efficiency tool
- Policy becomes: f(state, memory, time)

**Architecture requirement**: Still RecurrentSpatialQNetwork, but now time_of_day must be input.

**Why it's L5**: Teaches temporal planning without adding social complexity.

**Expected behavior**:

- Episodes 0-500: Many deaths due to bad timing
- Episodes 500-1500: Learning to check time before acting
- Episodes 1500+: Efficient scheduling (arrive early, wait, maximize work hours)

**Graduation criteria**: Agent can reliably schedule around operating hours, 75%+ retirement rate.

**Human observer validation**: "Can I know what time it is?" YES (clocks exist).

---

#### L6: The "Social Game" (Multi-Agent, Cues Only) [📋 Planned]

**World configuration**:

```yaml
grid_size: [8, 8]
visibility_radius: 2
affordances: [all 15, with capacity: 1 for contested resources]  # ← KEY CHANGE
num_agents: 2+  # ← multiple agents
observability: "partial + social"  # ← cues visible
temporal_mechanics: true
max_age_ticks: 1200
```

**What's new**: Other agents exist, you can see limited info about them

**What the agent sees NOW**:

```python
observation = {
    'bars': [...],
    'position': [x, y],
    'visible_grid': [...],
    'time_of_day': 14,

    # ← NEW: social information
    'other_agents_in_window': {
        'positions': [[4, 5], [6, 3]],  # where they are
        'public_cues': [
            ['looks_tired', 'at_job'],  # agent 2's cues
            ['looks_poor', 'looks_sad']  # agent 3's cues
        ],
        'recent_actions': [[2, 2, 4], [1, 1, 1]]  # action history
    }
}
```

**What the agent does NOT see**:

```python
# ❌ NO direct access to other agents' bars
# ❌ NO direct access to other agents' goals
# ❌ NO direct access to other agents' intentions
```

**Cues configuration** (from cues.yaml):

```yaml
cues:
  - id: "looks_tired"
    trigger: { bar: "energy", operator: "<", threshold: 0.3 }

  - id: "looks_poor"
    trigger: { bar: "money", operator: "<", threshold: 0.2 }

  - id: "at_job"
    trigger: { current_affordance: "Job" }
```

**The learning experience**:

**Scenario: The Race to the Job**

1. Agent A (you): L5 mastery, knows to go to Job at 9am
2. Agent B (rival): also learned to go to Job at 9am
3. Both agents arrive at Job at 8:59am
4. Both agents try INTERACT at 9am
5. Contention resolution: Agent B is 1 tile closer → Agent B wins
6. Agent A: INTERACT fails (Job is occupied, capacity=1)
7. Agent A: tries again... fails again... stuck
8. Agent A: can't earn money this shift
9. Agent A: can't afford food later
10. Agent A: dies of starvation
11. Terminal: massive penalty (early death)

**Retry with social reasoning:**

1. Agent A observes (at 8am):
   - Agent B position: (5, 5)
   - Agent B cues: ['looks_poor', 'moving_toward_industrial_zone']

2. Module C: Social Model prediction:

   ```python
   # Trained via CTDE to predict: cues → likely goal
   predicted_goal_dist = social_model(['looks_poor', 'at_industrial_zone'])
   # Output: {"go_to_job": 0.85, "go_to_labor": 0.10, "go_to_bed": 0.05}
   ```

3. Module D: Hierarchical Policy reasoning:

   ```python
   # Meta-controller:
   my_goal = "EARN_MONEY"

   # Evaluate options:
   option_1 = "go_to_job"
   world_model_prediction = "Agent B will also go to Job (85% confidence)"
   world_model_prediction += "I will lose contention (Agent B is closer)"
   expected_reward = -100  # early death if I can't earn money

   option_2 = "go_to_labor"
   world_model_prediction = "Labor is uncontested"
   expected_reward = +15  # earn less money, but > $0

   # Choose option 2
   final_action = "navigate_to_labor"
   ```

4. Agent A goes to Labor instead
5. Earns $48 (less than Job's $80, but > dying)
6. Survives

**The "Aha!" moment**: "I must reason about what other agents will do, and choose non-contested resources when I'll lose the race."

**Key learning**:

- Other agents are strategic competitors
- Public cues → predicted intentions (Module C)
- Predicted intentions → strategic response (Module D)
- Policy becomes: f(state, memory, time, belief_about_others)

**Architecture requirement**: Full Module A-D stack

- **Module A: Perception** - processes cues into belief states
- **Module B: World Model** - predicts outcomes of candidate actions
- **Module C: Social Model** - infers other agents' states/goals from cues (first mention)
- **Module D: Hierarchical Policy** - meta-controller picks goals, controller picks actions

**Why it's L6**: First genuine social reasoning. Theory of Mind emerges from cue interpretation.

**Expected behavior**:

- Episodes 0-1000: Frequent deaths due to contention (no social reasoning)
- Episodes 1000-3000: Learning cue correlations (CTDE pretraining helps)
- Episodes 3000+: Strategic resource selection (avoid contested affordances)

**Graduation criteria**: In contested scenarios, agent chooses alternate resources 60%+ of the time.

**Human observer validation**:

- ✅ "Can I see someone looks poor?" YES (visible clothing, demeanor)
- ❌ "Can I know their exact money value?" NO (internal state)

---

#### L7: Rich Social Reasoning (More Cues) [📋 Planned]

**World configuration**: Same as L6, but richer cue set

```yaml
cues:  # expanded from L6
  - "looks_tired"
  - "looks_energetic"
  - "looks_sad"
  - "looks_happy"
  - "looks_poor"
  - "looks_wealthy"
  - "looks_sick"
  - "looks_healthy"
  - "at_job"
  - "at_hospital"
  - "at_bar"
  - "carrying_items"  # behavioral cues
  - "rushing"
  - "resting"
```

**What's new**: Agents can infer more detailed state from richer cue vocabulary

**The learning experience**:

1. Agent A sees Agent B: ['looks_sick', 'at_hospital', 'looks_poor']
2. Module C: Social Model prediction:

   ```python
   predicted_state = {
       'health': ~0.25,  # "looks_sick"
       'money': ~0.15,   # "looks_poor"
       'current_goal': 'SURVIVAL'  # "at_hospital" while sick
   }
   ```

3. Strategic reasoning:
   - "Agent B is in survival crisis"
   - "They will prioritize hospital over work"
   - "Job will likely be available"
   - Decision: go to Job confidently

**Key learning**: More cues → better predictions → better strategy

**Why it's L7**: Incremental improvement over L6, not a new capability. Validates that cue richness scales.

**Expected behavior**: Higher retirement rates than L6 (better predictions → better resource allocation).

---

#### L8: Emergent Communication (Family Channel) [📋 Planned]

**World configuration**:

```yaml
grid_size: [8, 8]
visibility_radius: 2
affordances: [all 15, capacity: 1]
num_agents: 2+
observability: "partial + social"
temporal_mechanics: true
family_formation: true  # ← NEW: agents can form families
family_comm_channel: true  # ← NEW: in-group signaling
max_age_ticks: 1200
```

**What's new**: Families can communicate via abstract signals

**What the agent sees NOW**:

```python
observation = {
    'bars': [...],
    'position': [...],
    'visible_grid': [...],
    'time_of_day': 14,
    'other_agents_in_window': {...},  # same as L6/L7

    # ← NEW: family communication
    'family_comm_channel': [123, 0, 0],  # int64 signals from family members
    'family_id': 'family_42',

    # ❌ NO semantic dictionary provided
    # ❌ NO pre-shared meaning of signals
}
```

**Available actions NOW**:

```python
actions = [
    UP, DOWN, LEFT, RIGHT,
    INTERACT,
    WAIT,
    SET_COMM_CHANNEL(value)  # ← NEW: broadcast int64 signal to family
]
```

**The learning experience**:

**Scenario: The Coordination Problem**

1. Parent (Agent A) is at Job, money is full
2. Child (Agent C) is at home, money is low
3. No way to tell child "don't waste energy walking here, Job is taken"
4. Child walks to Job (wastes energy)
5. Job is occupied (Parent is using it)
6. Child can't work
7. Child doesn't earn money
8. Child dies of starvation later

**Learning emergent communication (over 1000s of episodes):**

1. Parent's Module D learns (via random exploration):
   - "When my money is full, try action: SET_COMM_CHANNEL(123)"
   - This action has high correlation with child surviving
   - (Because it happened to correlate with successful episodes during exploration)

2. Child's Module C: Social Model learns (via CTDE pretraining, then online refinement):
   - Input: family_comm_channel = [123, 0]
   - Ground truth (during training): Parent's money = high
   - Supervised learning: "signal 123 → parent money high"

3. Child's Module D: Hierarchical Policy learns:
   - "If signal=123, don't select goal: go_to_job"
   - "Instead, select goal: go_to_recreation (free time)"
   - Child's survival rate increases when respecting signal

4. Over many episodes:
   - Parent reliably sends 123 when Job is taken
   - Child reliably interprets 123 as "Job unavailable"
   - Coordination success rate increases
   - Family terminal reward > solo agent terminal reward

**The "Aha!" moment**: "The signal '123' has negotiated meaning. It wasn't given by designers, it emerged from correlation learning."

**Key learning**:

- Emergent communication via grounded correlation
- Signals start arbitrary (123 could have been 456)
- Meaning emerges from: signal → correlated state → coordinated action → better outcomes
- This is proto-language

**Architecture requirement**:

- Module A: process family_comm_channel as input
- Module C: predict state from signals (CTDE training)
- Module D: controller action space includes SET_COMM_CHANNEL

**Why it's L8**: This is the research frontier. Emergent communication without pre-shared semantics is an unsolved problem in multi-agent RL.

**Expected behavior**:

- Episodes 0-5000: Random signals, no coordination
- Episodes 5000-15000: Some signals start correlating with outcomes
- Episodes 15000+: Stable protocols emerge (e.g., 123="job taken", 456="danger", 789="food available")

**Graduation criteria**: Family coordination gain > 20% (family agents outperform solo agents with statistical significance).

**Evaluation metrics**:

- Signal diversity (number of unique signals used)
- Signal stability (P(same signal → same outcome))
- Semantic alignment (mutual information I(signal; state))
- Family coordination gain (terminal reward family vs solo)

**Human observer validation**:

- ✅ "Can I hear my family member's signal?" YES (acoustic/visual signal)
- ❌ "Do I know what '123' means without learning?" NO (must learn via correlation)

**Open research questions**:

- Do protocols transfer when families churn?
- Do different families develop different "dialects"?
- Can we decode learned semantics via causal interventions?

---

### 4.3 Observability Progression Table

| Level | Grid | Observability | Scaffolding | Human Equivalent |
|-------|------|--------------|-------------|------------------|
| **L0-3** | 5×5 | Full | All affordances visible | "Here's a map of town" |
| **L4-5** | 8×8 | Partial (5×5 window) | None | "Explore and remember" |
| **L6-7** | 8×8 | Partial + social cues | None | "Read body language" |
| **L8** | 8×8 | Partial + cues + channel | None | "Learn language" |

---

### 4.4 Architecture Requirements Table

| Level | Minimum Architecture | Why |
|-------|---------------------|-----|
| **L0-3** | SimpleQNetwork (feedforward) | Full observability, reactive policies sufficient |
| **L4-5** | RecurrentSpatialQNetwork (LSTM) | Partial obs requires memory for navigation |
| **L6-7** | Full Module A-D stack | Social reasoning requires cue interpretation |
| **L8** | Module A-D + comm channel | Emergent communication requires signal grounding |

**Module breakdown** (for Levels 6-8):

- **Module A: Perception** - builds belief state from partial observations
- **Module B: World Model** - predicts outcomes of candidate actions
- **Module C: Social Model** - infers other agents' states/goals from cues
- **Module D: Hierarchical Policy** - meta-controller (pick goal) + controller (pick action)

---
