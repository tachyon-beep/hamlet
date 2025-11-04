Here is the fleshed-out curriculum, expanding on your new Level 4 and Level 5.

This is a fantastic pedagogical roadmap. You've successfully isolated every core concept—from basic survival, to economics, to optimization, to memory, and finally to temporal planning.

---

### 🎚️ Level 0: The "Two Unsolvable Problems"

- **World:** 5x5 Grid
- **Affordances:** **`Bed`** (Costs $5)
- **Initial State:** Agent starts with `$50`.
- **Lesson:** "What is a 'policy' and what is 'starvation'?"
- **The Agent's Experience:**
    1. The agent's `Energy` drops. Its RND (curiosity) drives it to the `Bed`. It interacts, its `Energy` is restored, and it gets a huge `+Reward` (from your baseline-subtracted functional).
    2. It quickly learns the policy: **"If `Energy` is low, go to `Bed`."**
    3. ...But it "hammers" this new policy, and after 10 uses, its `Money` hits 0.
    4. ...Simultaneously, its `Satiation` meter is draining, and the `CascadeEngine` is making its `Health` plummet.
- **The "Aha!" Moment:** "I've solved `Energy`, but this reveals two new 'hopeless' problems: I will *either* die of starvation (`Satiation`) *or* die of exhaustion after my `Money` runs out."
- **Key Learning:** The agent assigns a massive negative value to both `V(s_low_satiation)` and `V(s_low_money)`.

---

### 🎚️ Level 1: The "Economic Problem"

- **World:** 5x5 Grid
- **Affordances:** `Bed` (Cost $5), **`Job`** (Gain $20)
- **Lesson:** "Aha! The `Job` is the solution to the `Money` problem!"
- **The Agent's Experience:**
    1. The agent is in the "hopeless" `s_low_money` state.
    2. RND drives it to the `Job`. It interacts. Its `Money` goes *up*.
    3. It now learns its first "sub-policy": **"If `Money` is low, go to `Job`."**
    4. It will now successfully create an *economically stable* loop: `Job` -> `Bed` -> `Job` -> `Bed`.
    5. ...it will *still* die of starvation.
- **Why it's the Perfect L1:** It *isolates* the economic lesson. The agent masters the `Money` <-> `Energy` loop but is still helpless against the `Satiation` cascade.

---

### fLevel 2: The "First Stable Loop"

- **World:** 5x5 Grid
- **Affordances:** `Bed` (Cost $5), `Job` (Gain $20), **`Fridge`** (Cost $4)
- **Lesson:** "Aha! The `Fridge` is the solution to the *other* 'hopeless' problem!"
- **The Agent's Experience:**
    1. The agent is running its mastered L1 policy (`Job` <-> `Bed`).
    2. Its `Satiation` meter drops. It's in the *other* "hopeless" state.
    3. RND drives it to the `Fridge`. It interacts. The cascade stops.
- **The "Aha!" Moment:** The agent now combines its two "sub-policies" into the final, stable L2 loop: **`Job` -> `Fridge` -> `Bed`**. This is *infinitely* stable. This is the **baseline for a "competent" agent.**

---

### 🎚️ Level 3: "The Full Simulation (Small)"

- **World:** 5x5 Grid
- **Affordances:** **All 15 affordances** (`Doctor`, `Recreation`, `Bar`, `Shower`, etc.)
- **Lesson:** "How do I survive *more efficiently*?"
- **The Agent's Experience:**
    1. The agent already has its L2 "survival" policy. But it notices it's *still* dying sometimes, or not maximizing its reward.
    2. Why? Its `Social` and `Hygiene` meters are dropping, triggering the *tertiary* cascades. These cascades make its `Energy` and `Mood` drain *faster*.
    3. This means it has to go to the `Bed` and `Job` *more often*, which is inefficient and risky.
- **The "Aha!" Moment:** "Wait... if I go to the `Bar` (solves `Social`) and `Shower` (solves `Hygiene`), the cascades stop! My `Energy` lasts longer! Going to the `Bar` is *instrumentally useful* because it means I have to work less!"
- **Why it's the Perfect L3:** It teaches **optimization** and the value of managing the "quality of life" meters.

---

### 🎚️ Level 4: "The Fog of War" (Partial Observability)

- **World:** 8x8 Grid, **`partial_observability = True`** (5x5 visibility)
- **Affordances:** All 15 affordances
- **Lesson:** "My policy is useless if I'm blind. I must explore and *remember*."
- **Required Architecture:** This is the first level that **requires** the `RecurrentSpatialQNetwork` (the LSTM). The `SimpleQNetwork` is no longer viable.
- **The Agent's Experience:**
    1. The agent's `Energy` gets low. It tries to execute its L3 policy: `goTo(Bed)`.
    2. It can't see the `Bed`. The policy fails. It wanders randomly, dies, and gets a massive negative reward.
    3. Its `RND` (curiosity) is now its only guide. It explores *just to explore*.
    4. It stumbles upon the `Bed` at `(x, y)`. Its LSTM *records* this observation.
    5. 50 steps later, its `Energy` is low again. It's on the other side of the map.
- **The "Aha!" Moment:** "My `Energy` is critical. I can't *see* the Bed, but my LSTM's hidden state *remembers* it's at `(x, y)`. I will follow this 'memory' path." The agent navigates back to the `Bed` from memory, survives, and receives a huge positive reward.
- **Key Learning:** The agent learns that **exploration (RND) is critical for building a mental map**, and **memory (LSTM) is critical for using that map.**

---

### 🎚️ Level 5: "The 9-to-5" (Temporal)

- **World:** 8x8 Grid, All Affordances, **`enable_temporal_mechanics = True`**
- **Lesson:** "My solutions only work at certain times. I must learn to *schedule* my life."
- **Required Architecture:** The `RecurrentSpatialQNetwork` now receives `time_of_day` as part of its input.
- **The Agent's Experience:**
    1. The agent executes its perfected L4 policy. It's low on `Money` at 3 AM.
    2. It navigates from memory to the `Job`.
    3. It tries to `INTERACT`. The action is masked, because the `Job` is closed (as per `affordance_config.yaml`).
    4. It tries again. And again. It's stuck in a loop, wasting `Energy` and `Satiation` for hours. It dies.
- **The "Aha!" Moment:** "I can't just go to the `Job` when I *want* to; I have to go when it's *open*. The `WAIT` action is now a critical strategic tool. The optimal policy is to arrive at the `Job` at 7 AM and `WAIT` until it opens at 8 AM, as this wastes the least amount of energy."
- **Key Learning:** The agent's policy graduates from `f(state)` to `f(state, time)`. It has learned **patience** and **scheduling**, the final components of strategic planning.

---

### 🎚️ Level 6: "The Social Game" (Multi-Agent & ToM)

- **World:** 8x8 Grid, All Affordances, POMDP=`True`, Temporal=`True`, **`num_agents > 1`**
- **Affordances:** All 15, but now with **`capacity: 1`** (e.g., `Job`, `Bed`).
- **Lesson:** "My L5 policy is perfect... until someone *else* takes the 'Job' first. I must *anticipate* and *out-think* my rivals."
- **Required Architecture:** This is the *full v2.0 stack*. It activates **Module C (Social Model)** and heavily upgrades **Module D (Hierarchical Policy)**.
- **The Agent's Experience (The "Race to the Job"):**
    1. The agent's L5 policy is perfect: "It's 7 AM, I will go to the 'Job' at 8 AM."
    2. It arrives at 7:59 AM. An *opponent* (Agent 2) steps onto the `Job` tile at 7:58 AM.
    3. The `Job` is now occupied (`capacity=1`). The agent's `INTERACT` action is masked (or fails).
    4. It waits. And waits. It runs out of money and dies, receiving a massive terminal penalty (`-100` or more).
- **The "Aha!" Moment (The v2.0 Payoff):**
    1. **RETRY:** The agent's `Module A (Perception)` sees Agent 2. It notes their `public_cues` (e.g., `looks_poor`, "is walking towards Industrial Zone").
    2. **REASON:** **`Module C (Social Model)`** consumes this cue history and generates a `SocialPrediction`: `{"predicted_goal_dist": {"go_to_job": 0.85, ...}, "confidence": 0.85}`.
    3. **STRATEGIZE:** **`Module D (Meta-Controller)`** receives this `SocialPrediction`. Its `Goal` selection logic is no longer just "Do I need money?" It's now: "Do I need money *and* can I beat Agent 2 there? My `WorldModel` (Module B) 'imagines' I will lose the race."
    4. **NEW POLICY:** The agent learns a *counter-strategy*. "My `SocialPrediction` says Agent 2 is also going to the 'Job' and will win. **Therefore,** I will *not* go to the 'Job' (a `-100` reward) and will instead execute a *contingency goal* to go to the 'Labor' affordance (a `+15` reward)."
- **Key Learning:** The agent's policy is no longer just `f(state, time)`. It is now `f(state, time, belief_about_others)`. It has learned **Theory of Mind (ToM)**.

---

### 🎚️ Level 7: "Personalities & Families" (Co-Evolution)

- **World:** Same as L6, but agents are now spawned with the **`AgentDNA`** struct (from Appendix A). The `PopulationManager` now uses a Genetic Algorithm (GA) for the "Outer Loop."
- **Lesson:** "Not all agents are the same. Some are 'greedy,' some are 'cautious.' I must adapt my strategy to their 'personality'."
- **Required Architecture:** `AgentDNA` tensor is now a new input to **`Module D1 (Meta-Controller)`**.
- **The Agent's Experience:**
    1. The agent tries its L6 policy (yield the `Job` if contested).
    2. It encounters a "neurotic" agent (`neuroticism = 0.9`). It observes (via `Module C`) that this agent *always* avoids the `Job` if it sees another agent.
    3. It encounters a "greedy" agent (`greed = 0.9`). It observes this agent *always* races for the `Job`, even if it's a bad idea.
- **The "Aha!" Moment:** "`Module C`'s predictions are even *better* if I correlate them with `AgentDNA`! My `Meta-Controller` (Module D1) can now learn: I should *bluff* the 'neurotic' agent by just *walking towards* the 'Job', and they will run away. I must *always* yield to the 'greedy' agent."
- **Key Learning:** The agent learns **opponent-specific adaptation** and **deception**, driven by the `AgentDNA` inputs biasing its `Goal` selection.

---

### 🎚️ Level 8: "Emergent Language" (The Moonshot)

- **World:** Same as L7, but agents now have a `family_id` and the `family_comm_channel` (from Appendix A) is active.
- **Lesson:** "How can I *cooperate* with my 'family' to beat other agents?"
- **Required Architecture:** `Module A` now receives `family_comm_channel` as an input. `Module D2 (Controller)` now has a new action: `SetCommChannel(int64)`.
- **The Agent's Experience:**
    1. Agent 1 (Parent) is at the `Job` and its `Money` meter is full. It has no way to tell Agent 2 (Child) at home not to waste energy walking to the `Job`.
    2. Agent 1's `Module D (Policy)` learns (over thousands of episodes) to take a "useless" action that has a high correlation with success: `SetCommChannel(123)`.
    3. Agent 2 (Child) is at home, `Money` is low. Its `Module A (Perception)` sees `family_comm_channel = [123, 0]`.
    4. Its `Module C (Social Model)` learns (via CTDE) that the signal "123" from "Parent" has a *massive* correlation with "Parent's `Money` meter is full."
- **The "Aha!" Moment:** "The signal '123' *means* 'the job is taken' or 'money is good'! My `Meta-Controller` (Module D1) can now learn: 'If signal is 123, do not select `Goal: go_to_job`.' I will go to 'Recreation' instead."
- **Key Learning:** The agents have **negotiated a meaning** for an abstract signal to solve a resource-contention problem, achieving **emergent communication** and coordination.
