---
title: "Reward Architecture"
document_type: "Design Rationale"
status: "Draft"
version: "2.5"
---

## SECTION 3: THE REWARD ARCHITECTURE

### 3.1 Per-Tick Reward: Framework Capability (Reference: `r = energy × health`)

**IMPORTANT**: The Townlet Framework supports arbitrary user-configurable reward functions via UNIVERSE_AS_CODE. The specific formula discussed here is the **reference implementation** used in the "Townlet Town" instance, not a hardcoded framework requirement.

The reference per-tick reward is deliberately minimal:

```python
r_t = bars[energy] * bars[health]
```

Where both values are normalized [0.0, 1.0], producing reward range [0.0, 1.0].

#### Why These Two Bars Specifically?

**Energy and health are the pivotal survival meters** — the only two that cause immediate death when they hit zero.

From `bars.yaml`:

```yaml
terminal_conditions:
  - meter: "energy"
    operator: "<="
    value: 0.0
    description: "Death by exhaustion"

  - meter: "health"
    operator: "<="
    value: 0.0
    description: "Death by health failure"
```

**Rationale**: The reward reflects "am I in immediate danger of dying?" Not "am I happy" or "am I rich" — those matter for long-term quality of life (captured in terminal reward), but moment-to-moment, survival dominates.

**Human equivalent**: A starving person doesn't care about their mood or social connections *right now* — they care about not dying. Once survival is assured, other concerns emerge. This is Maslow's hierarchy in a reward function.

#### Why Multiplicative, Not Additive?

**Additive reward** `r = energy + health` allows compensating:

```python
# Scenario A: balanced
energy = 0.5, health = 0.5
r = 0.5 + 0.5 = 1.0  ✓

# Scenario B: unbalanced
energy = 0.1, health = 0.9
r = 0.1 + 0.9 = 1.0  ✓ (same reward!)
```

But scenario B is dangerous — you're one more energy tick from death. The reward shouldn't suggest these are equivalent.

**Multiplicative reward** `r = energy × health` requires both:

```python
# Scenario A: balanced
energy = 0.5, health = 0.5
r = 0.5 × 0.5 = 0.25

# Scenario B: unbalanced
energy = 0.1, health = 0.9
r = 0.1 × 0.9 = 0.09  ✗ (much worse!)
```

The agent learns: "Having one bar very low is catastrophic, even if the other is high."

**Alternative considered: minimum**

```python
r = min(energy, health)  # "weakest link"
```

This also captures the danger of imbalance, but it's discontinuous (derivatives are undefined at energy=health), which can cause training instability. Multiplication gives smooth gradients.

#### What This Doesn't Tell Agents

The reward provides no guidance about:

**❌ Which affordances exist**

- No reward bump for discovering "Bed"
- No reward bump for first time using "Job"

**❌ Which affordances to use**

- No `+0.1` for sleeping
- No `+0.1` for eating
- No `+0.05` for working

**❌ When to use affordances**

- No temporal shaping ("good job working during business hours!")
- No economic shaping ("good job saving money!")

**❌ How bars are coupled**

- No hint that satiation affects health via cascades
- No hint that hygiene affects mood affects energy
- Must discover cascade dynamics through experience

**❌ Long-term strategy**

- No guidance about balancing work/rest
- No guidance about retirement planning
- No guidance about money management

**This is intentional.** The agent must learn world dynamics through exploration and credit assignment, not by following reward breadcrumbs.

#### What Agents Must Discover

With only `r = energy × health`, agents must learn through trial, error, and long-horizon reasoning:

**Direct relationships** (observable within 1-5 ticks):

```
Action: go to Bed
  → energy increases (direct effect from affordances.yaml)
  → r_t increases (immediate feedback)
  → Learning: "Bed is good when energy is low"

Action: go to Doctor
  → health increases (direct effect)
  → r_t increases (immediate feedback)
  → Learning: "Doctor is good when health is low"
```

**Indirect relationships via cascades** (observable over 10-50 ticks):

```
Action: ignore Fridge
  → satiation drops (base_depletion from bars.yaml)
  → satiation < 0.2 threshold (cascades.yaml)
  → cascade: low_satiation_hits_health
  → health drops faster than base rate
  → r_t drops gradually
  → Learning: "Not eating causes slow health decay"

Action: go to Fridge
  → satiation increases (direct effect)
  → satiation > 0.2 threshold
  → cascade prevented
  → health decay returns to normal
  → r_t stabilizes
  → Learning: "Eating prevents health decay"
```

**Economic loops** (observable over 50-200 ticks):

```
Action: go to Job repeatedly
  → money increases (direct effect)
  → energy decreases (cost per tick)
  → mood decreases (cost per tick)
  → r_t drops during work
  → BUT: money enables Fridge, Bed, etc
  → Using money on recovery → health/energy maintained
  → r_t recovers and stays high long-term
  → Learning: "Work is bad short-term but enables survival long-term"
```

**Temporal patterns** (observable over 100+ ticks):

```
Observation: Job is masked at night
  → Trying to work at 3am fails
  → Energy wasted walking to Job
  → r_t drops
  → Learning: "Job has operating hours, plan around them"

Strategy: arrive at Job at 7am, WAIT until 8am
  → Minimal energy waste
  → Can work full shift when it opens
  → More efficient money accumulation
  → Better r_t over episode
  → Learning: "Temporal planning matters"
```

**Social competition** (L6+, observable over entire episodes):

```
Observation: rival agent took the Job
  → I can't work this shift
  → Money doesn't increase
  → Can't afford food later
  → Health drops due to starvation
  → r_t plummets
  → Learning: "Other agents compete for resources"

Strategy: predict when rival will go to Job
  → Arrive earlier, or choose alternate income (Labor)
  → Money still flows
  → Survival maintained
  → Better r_t over episode
  → Learning: "Social reasoning enables resource access"
```

#### Why This Is Hard (And Why That's Good)

The credit assignment problem is severe:

**Example: The delayed starvation death**

```
Tick 50: Agent chooses to skip Fridge (saves $4)
Tick 51-99: Satiation drops slowly
Tick 100: Satiation < 0.2, cascade activates
Tick 101-200: Health decays at 2× normal rate
Tick 210: Health = 0, agent dies
Terminal reward: 0.1 × (low score) due to early death penalty

Credit assignment challenge:
  The decision at tick 50 killed the agent at tick 210.
  That's 160 ticks of delayed consequence.
  No intermediate rewards pointed to the mistake.
  Agent must learn: "skip meal → death 160 ticks later"
```

**This requires**:

- Value function learning (predict long-term outcomes)
- World model learning (understand cascade dynamics)
- Temporal reasoning (connect distant cause and effect)

**Comparison to dense shaping**:

```python
# Dense shaping (easy mode)
if action == "eat":
    r += 0.1  # immediate feedback, no credit assignment needed

# Sparse (hard mode)
# No feedback at all until health drops 100 ticks later
```

**Why hard mode is better for research**:

- Tests genuine learning, not reward following
- Produces policies that understand causality
- Transfers better when world physics change
- Exhibits real emergent behavior (discovery moments)

---

### 3.2 Terminal Reward: Framework Capability (Reference: Retirement Scoring)

**IMPORTANT**: Terminal reward formulas are user-configurable via UNIVERSE_AS_CODE. The formula described here is the **reference implementation** for the "Townlet Town" instance.

When an episode ends, the agent receives a terminal bonus based on how well they lived.

#### The Formula

```python
def compute_terminal_reward(bars, money, ticks_survived, max_age, outcome):
    """
    Terminal reward depends on:
    - Did you complete your full lifespan? (retirement vs early death)
    - Quality of life at the end (wellbeing)
    - Financial security (wealth)
    """

    # Component 1: Lifespan completion
    lifespan_score = ticks_survived / max_age

    # Component 2: Wellbeing at terminal state
    wellbeing_bars = [
        bars['health'],
        bars['energy'],
        bars['mood'],
        bars['social'],
        bars['hygiene']
    ]
    wellbeing_score = mean(wellbeing_bars)

    # Component 3: Wealth (diminishing returns)
    wealth_score = sqrt(money)  # or linear, log - configurable

    # Weighted combination
    raw_score = (
        config.retirement.weights.lifespan * lifespan_score +
        config.retirement.weights.wellbeing * wellbeing_score +
        config.retirement.weights.wealth * wealth_score
    )

    # Apply outcome multiplier
    if outcome == "retirement":  # age >= 1.0
        return raw_score
    elif outcome == "death":  # energy=0 or health=0
        return config.retirement.early_death_multiplier * raw_score
```

**Default configuration**:

```yaml
retirement:
  max_age_ticks: 1200

  weights:
    lifespan: 0.5   # 50% - did you complete your life?
    wellbeing: 0.3  # 30% - were you healthy/happy at the end?
    wealth: 0.2     # 20% - did you have financial security?

  wealth_curve: "sqrt"  # diminishing returns on money
  early_death_multiplier: 0.1  # 90% penalty for dying early

  wellbeing_bars:
    - health
    - energy
    - mood
    - social
    - hygiene
```

#### Component Breakdown

**Lifespan (50% weight by default)**

```python
lifespan_score = ticks_survived / max_age
```

**Interpretation**:

- Die at tick 600 / 1200 = 0.5 lifespan score
- Retire at tick 1200 / 1200 = 1.0 lifespan score

**Why this matters**:

- Surviving longer is inherently valuable
- Distinguishes "scraped by to retirement" from "died young"
- Even if you retire broke, completing your life counts for something

**Wellbeing (30% weight by default)**

```python
wellbeing_score = mean([health, energy, mood, social, hygiene])
```

**Interpretation**:

- Retiring with all bars at 0.8+ = 0.8 wellbeing score (comfortable)
- Retiring with mixed bars (0.9, 0.3, 0.5, 0.4, 0.6) = 0.54 wellbeing score (struggled)
- Retiring at death's door (0.1, 0.1, 0.1, 0.0, 0.2) = 0.10 wellbeing score (miserable)

**Why this matters**:

- Prevents "die rich but filthy and alone" strategies
- Rewards quality of life, not just survival
- Makes mood/social/hygiene instrumentally valuable (they affect terminal score)

**Note**: Fitness is excluded because it's already instrumentally valuable (modulates health decay via cascades). Including it would double-count its importance.

**Wealth (20% weight by default)**

```python
wealth_score = sqrt(money)  # money normalized [0, 1] where 1.0 = $100
```

**Interpretation**:

```python
money = 0.00 → sqrt(0.00) = 0.00  # broke
money = 0.25 → sqrt(0.25) = 0.50  # $25 saved
money = 0.64 → sqrt(0.64) = 0.80  # $64 saved
money = 1.00 → sqrt(1.00) = 1.00  # $100 saved (cap)
```

**Why sqrt (diminishing returns)**:

Linear wealth would incentivize endless grinding:

```python
# Linear (bad)
money = 0.90 → score = 0.90
money = 1.00 → score = 1.00  # +10% money = +10% score
# Incentive: work until last possible moment
```

Sqrt wealth has diminishing returns:

```python
# Sqrt (good)
money = 0.00 → score = 0.00
money = 0.25 → score = 0.50  # first $25 is very valuable
money = 0.64 → score = 0.80  # next $39 is moderately valuable
money = 1.00 → score = 1.00  # last $36 is marginally valuable
```

**Effect on behavior**:

- Saving some money matters a lot (0 → $25)
- Hoarding beyond "comfortable nest egg" has less marginal benefit
- Prevents late-life grinding (working when you should be enjoying retirement)

**Alternative curves** (configurable):

```yaml
wealth_curve: "linear"  # every dollar matters equally
wealth_curve: "log"     # even more aggressive diminishing returns
wealth_curve: "sqrt"    # balanced (default)
```

#### Retirement vs Early Death (The 10× Multiplier)

The outcome of an episode drastically changes the terminal reward:

**Retirement** (age >= 1.0):

```python
terminal_reward = raw_score  # full value
```

**Early Death** (energy=0 or health=0 before max_age):

```python
terminal_reward = 0.1 * raw_score  # 90% penalty
```

**Example**:

```python
# Agent A: Dies at tick 600
ticks_survived = 600
lifespan_score = 600 / 1200 = 0.5
wellbeing_score = 0.6
wealth_score = sqrt(0.36) = 0.6

raw_score = 0.5*0.5 + 0.3*0.6 + 0.2*0.6 = 0.25 + 0.18 + 0.12 = 0.55
terminal_reward = 0.1 * 0.55 = 0.055  # early death penalty

# Agent B: Retires at tick 1200
ticks_survived = 1200
lifespan_score = 1200 / 1200 = 1.0
wellbeing_score = 0.6
wealth_score = sqrt(0.36) = 0.6

raw_score = 0.5*1.0 + 0.3*0.6 + 0.2*0.6 = 0.50 + 0.18 + 0.12 = 0.80
terminal_reward = 0.80  # no penalty

# Agent B earned 14.5× more terminal reward despite similar wellbeing/wealth
```

**Why this extreme multiplier**:

- Making it to retirement should be the primary goal
- Early death is a catastrophic failure, regardless of how much money you saved
- Teaches agents to prioritize long-term survival over short-term gains

**Human equivalent**: "You worked yourself to death at 40 with $100k in the bank" is not a success story, even though you were wealthy. "You retired at 65 with $10k, healthy and happy" is better.

---

### 3.3 The Age Bar: Natural Horizon Without Cheating

Most RL environments use arbitrary episode length:

```python
# Typical approach (external to world)
if steps >= max_steps:
    done = True  # "time's up!"
```

**Problems**:

- Not in-world (agent can't observe it approaching)
- Can't be explained ("why did my life suddenly end?")
- Can't be planned for (agent doesn't know episode length)

**Townlet Framework's approach**: Age is a visible meter that advances naturally.

#### Age Bar Configuration

From `bars.yaml`:

```yaml
- name: "age"
  index: 8  # the 9th bar (adding to the standard 8)
  tier: "lifecycle"
  range: [0.0, 1.0]
  initial: 0.0
  base_depletion: 0.0  # doesn't passively decay

  terminal_condition:
    operator: ">="
    value: 1.0
    outcome: "retirement"

  description: "Life progression toward retirement"
```

#### How Age Advances

In the environment step function:

```python
def step(self, action):
    # Execute action, apply affordances, cascades, etc
    # ...

    # Advance age
    age_increment = 1.0 / self.config.retirement.max_age_ticks
    self.bars['age'] += age_increment

    # Check terminal conditions
    if self.bars['age'] >= 1.0:
        done = True
        outcome = "retirement"
    elif self.bars['energy'] <= 0 or self.bars['health'] <= 0:
        done = True
        outcome = "death"
```

**Example with max_age_ticks=1200**:

```python
age_increment = 1.0 / 1200 = 0.000833...

tick 0:    age = 0.000
tick 600:  age = 0.500  # halfway through life
tick 1199: age = 0.999
tick 1200: age = 1.000  # retirement!
```

#### Why This Works

**✅ Observable by the agent**:

```python
observation['bars'][8] = age  # agent can see they're aging
```

**✅ Allows planning**:

- Agent at age=0.3 can predict: "I have ~840 ticks left"
- Can adjust strategy: "accumulate savings early, reduce work late"

**✅ In-world narrative**:

- Not "episode ended arbitrarily"
- But "you lived your full natural lifespan"

**✅ Curriculum adjustable**:

```yaml
# Level 0-1: Short episodes for rapid iteration
retirement:
  max_age_ticks: 200

# Level 2-3: Medium episodes for economic learning
retirement:
  max_age_ticks: 500

# Level 4+: Full episodes for long-horizon planning
retirement:
  max_age_ticks: 1200
```

**Human observer test**:

- ✅ Can a human know they're aging? YES (visible passage of time)
- ✅ Can a human plan for retirement? YES (they know it's coming)
- ❌ Does time suddenly stop for no reason? NO (age is in-world physics)

---

### 3.4 Value Systems as Configuration

The terminal reward formula encodes a value system: what constitutes a "good life"?

By making weights configurable, the Townlet Framework lets researchers explore different moral frameworks.

#### Capitalist World (Wealth Priority)

```yaml
retirement:
  weights:
    lifespan: 0.3   # completing life matters less
    wellbeing: 0.1  # quality of life matters less
    wealth: 0.6     # money is 60% of score

  wealth_curve: "linear"  # every dollar matters equally
```

**Expected agent behavior**:

- Work constantly, even when tired/sad
- Minimize spending on recreation, social, comfort
- Tolerate low mood, poor hygiene, loneliness
- Accumulate maximum money for terminal payout
- "Die rich but miserable" becomes rational strategy

**Research questions**:

- Does the agent learn to exploit itself?
- What's the equilibrium wealth at retirement?
- Does personality (greed slider) matter more in this world?

#### Balanced World (Default)

```yaml
retirement:
  weights:
    lifespan: 0.5   # survival to retirement is primary
    wellbeing: 0.3  # quality of life matters
    wealth: 0.2     # financial security matters

  wealth_curve: "sqrt"  # diminishing returns
```

**Expected agent behavior**:

- Sustainable work-rest cycles
- Balanced spending on necessities + some comfort
- Moderate savings, not obsessive hoarding
- "Comfortable middle-class retirement" strategy

**Research questions**:

- What's the optimal work/leisure balance?
- How much money is "enough"?
- Does the agent discover preventative healthcare?

#### Hedonist World (Experience Priority)

```yaml
retirement:
  weights:
    lifespan: 0.2   # making it to old age is less important
    wellbeing: 0.7  # quality of life is 70% of score
    wealth: 0.1     # money barely matters

  wealth_curve: "sqrt"
```

**Expected agent behavior**:

- Minimal work, maximum leisure
- High spending on Bar, Recreation, social activities
- Live paycheck-to-paycheck
- Risk of death due to insufficient emergency funds
- "Live fast, die happy" strategy

**Research questions**:

- Does the agent still survive to retirement?
- What's the minimum viable income?
- Does social/mood optimization emerge?

#### Comparing Worlds

Researchers can run the same agent architecture in different value systems:

```bash
# Same brain, different value systems
townlet train --config configs/capitalist/ --seed 42
townlet train --config configs/balanced/ --seed 42
townlet train --config configs/hedonist/ --seed 42

# Compare outcomes
townlet analyze --runs runs/capitalist_* runs/balanced_* runs/hedonist_*
```

**Metrics to compare**:

- Mean terminal reward (which world scores highest?)
- Retirement rate (which world has more early deaths?)
- Wellbeing at retirement (which world produces happier agents?)
- Wealth distribution (which world has more inequality?)
- Time allocation (work vs leisure hours)

**Expected finding**: Agents learn different personalities/strategies under different scoring systems, even with identical initial conditions.

---

### 3.5 Learning Dynamics: Credit Assignment Over Multiple Time Scales

With sparse per-tick reward and delayed terminal bonus, learning happens at multiple time scales.

#### First-Order Relationships (1-5 ticks)

These are directly observable cause-and-effect pairs:

```
Action: Bed
  Tick 0: energy = 0.3, health = 0.8, r = 0.24
  Tick 1: energy = 0.55 (direct effect +0.25)
  Tick 1: r = 0.55 × 0.8 = 0.44

  Learning signal: r increased immediately
  Lesson: "Bed → energy ↑ → reward ↑"
  Credit assignment: trivial (1 tick delay)
```

**Agents learn these quickly** because the feedback loop is tight. By episode 10-20, agents reliably use Bed when energy is low.

#### Second-Order Relationships (10-50 ticks)

These involve cascades — indirect effects that take time to manifest:

```
Action: Skip Fridge (save $4)
  Tick 0: satiation = 0.25, health = 0.8
  Tick 1-9: satiation drops to 0.19 (base depletion)
  Tick 10: satiation < 0.2 threshold
  Tick 10: cascade activates: low_satiation_hits_health
  Tick 11-30: health drops 0.010 per tick (cascade strength)
  Tick 30: health = 0.60 (lost 0.20 health)
  Tick 30: r drops from 0.60 to 0.45 (energy also dropped)

  Learning signal: r decreased gradually, 30 ticks after decision
  Lesson: "Not eating → health decay → reward decay"
  Credit assignment: moderate difficulty (30 tick delay)
```

**Agents learn these by mid-training** (episodes 50-200) once they've experienced enough starvation events to correlate "skip Fridge" → "feel bad later".

**World model helps**: If the agent trains a world model (Module B), it can predict "if I skip Fridge, my satiation will drop below 0.2, which I've observed triggers health decay."

#### Third-Order Relationships (50-200 ticks)

These involve multi-step economic loops:

```
Strategy: Work → Save Money → Use Money Later

  Tick 0-80: Go to Job repeatedly
    - Direct effect: money ↑ (0.0 → 0.8)
    - Direct cost: energy ↓, mood ↓
    - Immediate r: drops (working is tiring)

  Tick 80: Stop working, have $80 saved

  Tick 81-120: Use savings on recovery
    - Fridge (cost $4) → satiation maintained
    - Bed (free) → energy recovered
    - Shower (cost $2) → hygiene maintained
    - Recreation (cost $5) → mood recovered

  Tick 120-200: Maintain high bars without working
    - Energy high (from rest)
    - Health high (from food + prevented cascades)
    - Mood high (from recreation)
    - r stays high for 80 ticks

  Terminal: High lifespan, high wellbeing, moderate wealth
  Terminal reward: 0.75

  Learning signal: short-term pain (working) enables long-term gain (comfortable life)
  Lesson: "Money is instrumentally valuable for survival"
  Credit assignment: hard (80-200 tick delayed benefit)
```

**Agents learn these late in training** (episodes 500+) after experiencing many full episodes and learning the value of savings.

**This is why terminal reward matters**: Without the delayed terminal bonus, agents would just optimize immediate `r = energy × health`, never learning that work (which lowers immediate reward) enables better terminal outcomes.

#### Fourth-Order: Temporal Planning (100-500 ticks)

These involve scheduling around world constraints:

```
Observation: Job has operating_hours [9, 18]

  Naive strategy: Try to work whenever money is low
    - Wake up at 3am, energy low
    - Walk to Job
    - Try INTERACT
    - Action is masked (Job closed)
    - Energy wasted on commute
    - Can't work, money doesn't increase
    - Result: death by starvation

  Learned strategy: Plan work around operating hours
    - Wake up at 7am
    - Arrive at Job at 7:45am
    - WAIT until 8am (minimal energy cost)
    - Job opens at 9am
    - INTERACT for full 8-hour shift
    - Earn $80 efficiently
    - Result: survival with savings

  Learning signal: terminal reward is much higher with temporal planning
  Lesson: "Schedule matters as much as action choice"
  Credit assignment: very hard (must connect schedule across entire episode)
```

**Agents learn this gradually** (episodes 1000+) by experiencing failed work attempts and eventually correlating "time of day" with "action success."

**This is why the LSTM is required at Level 4+**: Temporal planning requires memory of when things worked before.

#### Fifth-Order: Social Coordination (Full Episode Scope in Level 6+)

These involve predicting and responding to other agents:

```
Scenario: Two agents need the same Job (capacity=1)

  Naive strategy: Both agents try to work simultaneously
    - Agent A: goes to Job
    - Agent B: goes to Job (same tick)
    - Contention resolution: Agent A wins (closer)
    - Agent B: wasted energy commuting, earns $0
    - Agent B: can't buy food
    - Agent B: dies of starvation

  Learned strategy: Agent B predicts Agent A's behavior
    - Module C (Social Model): sees Agent A "looks_poor" + "moving toward Job"
    - Predicts: Agent A will work this shift
    - Module D (Policy): selects alternate goal: go to Labor instead
    - Agent B: earns $48 at Labor (less than Job, but > $0)
    - Agent B: survives

  Terminal: Agent B gets lower score than Agent A, but > early death penalty

  Learning signal: social reasoning prevents catastrophic failure
  Lesson: "Predict competitors, choose non-contested resources"
  Credit assignment: extremely hard (requires correlating cues → intent → resource contention → terminal outcome)
```

**Agents learn this very late** (episodes 5000+, often requires CTDE pretraining for Module C).

**This is why Level 6+ is the research frontier**: Multi-agent credit assignment over full episode lengths with partial observability is an unsolved hard problem in RL.

---

### 3.6 Why This Reward Architecture Enables Real Research

The combination of sparse per-tick + rich terminal produces three research-valuable properties:

#### 1. Agents Learn World Dynamics, Not Reward Hacking

**Dense shaping version** (what we're avoiding):

```python
r = 0.0
if action == "eat": r += 0.1
if action == "sleep": r += 0.1
# etc...
```

**What the agent learns**: "Maximize action counts"
**What the agent doesn't learn**: "Food prevents starvation cascades"

**Townlet Framework version** (reference implementation):

```python
r = energy × health  # just reflect internal state
```

**What the agent learns**: "Cascades are real, I must prevent them"
**Emergent understanding**: Agent discovers the causal graph (Fridge → satiation → health)

#### 2. Policies Transfer Across World Variants

**Dense shaping**: If you change world physics (make food more expensive), you must retune all the shaping coefficients or the policy breaks.

**Sparse reward**: If you change world physics, the reward function stays the same (`r = energy × health`). The agent must re-learn strategies, but the objective is still "survive well."

**Example**:

```yaml
# Baseline world
affordances:
  - id: "fridge"
    costs: [{ meter: "money", amount: 0.04 }]  # $4

# Austerity world (just change one line)
affordances:
  - id: "fridge"
    costs: [{ meter: "money", amount: 0.08 }]  # $8 (doubled)
```

With sparse rewards, you can directly compare:

- How much more does the agent work?
- Does it skip comfort affordances?
- Does the starvation rate increase?

The reward function didn't change, so behavioral changes are due to world physics, not reward tuning artifacts.

#### 3. Terminal Bonus Enables Value Alignment Research

The configurable terminal reward is essentially a **learned value alignment** mechanism:

```yaml
# What does this society value?
retirement:
  weights:
    lifespan: 0.5
    wellbeing: 0.3
    wealth: 0.2
```

Agents internalize these values through RL. Change the weights, and agents learn different priorities.

**This is actual alignment research** — not "stop the AI from saying bad words" but "can the AI learn to value what we configure it to value?"

**Research question**: Do agents trained under different value systems generalize their values to new situations? Or do they just memorize "maximize terminal score" without understanding the underlying values?

---

**End of Section 3**

---
