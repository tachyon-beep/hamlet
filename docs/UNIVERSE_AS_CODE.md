# Townlet v2.0: Universe as Code

Document Date: 3 November 2025
Status: Approved for Implementation
Owner: Principal Technical Advisor (AI)

## 1. What this system is

Townlet v2.0 is a configurable simulated world. We call it Universe as Code.

Instead of hardcoding game rules, survival logic, opening hours, reward shaping, and so on directly in Python, we describe the universe in data files. The engine then loads those files and enforces them.

In practice that means:

* How fast you get hungry
* What a shower does to your hygiene and mood
* What happens to you if you’re broke at 2am
* Whether you're considered "dead" or "retired"
* How good your life was in the end
* All of that is declared in YAML.

This setup does two things.

First, it lets us control the simulation with dials instead of rewrites. We can create "easy mode society", "late stage capitalism mode", "cyberpunk gig economy mode" or "Scandi social democracy mode" just by swapping config packs. No code changes. No retuning hand-written if statements. Designers and students can edit knobs like hunger strength or wages and immediately see how that changes agent behaviour.

Second, it gives learning systems something intelligible to learn. The agent doesn't just memorise "press button X on tile Y". The agent is sitting inside an explicit, inspectable ruleset. That ruleset becomes the ground truth for:

* predicting consequences (World Model / Module B)
* reading other agents (Social Model / Module C)
* planning a life instead of just surviving a tick

In other words: this isn't just a game; it's a specification of a society that an agent can reason about.

### Where this lives in code

Universe as Code is implemented through four cooperating subsystems:

* `cascade_config` and `cascade_engine` — bar definitions, depletion, cascades, terminal conditions, ageing.
* `affordance_config` and `affordance_engine` — what actions exist in the world, what they cost, and what they do.
* `vectorized_env.VectorizedTownletEnv` — the actual runtime world: positions on the grid, time of day, masking which actions are legal, tracking multi-step interactions like sleeping or working a shift.
* `reward_model` (end-of-life scoring) — how we decide if that run of your life was "good".

Together, that config output is the single source of truth for:

* the agent’s internal survival meters (energy, health, hunger, money, mood, etc)
* passive decay and cross-meter pressure ("if you're filthy, your mood rots faster")
* affordances like Bed, Job, Hospital, Bar, etc: opening hours, per-tick wages, side effects
* the clock (things that close at 6pm, bars that run until 4am)
* ageing and retirement (you don't just die; you can also age out with dignity)
* how we score the life you managed to carve out

Everything in those runtime systems is normalised to [0.0, 1.0] so it's all vector math. Money is treated the same way: 1.0 means $100 by convention in the baseline world. That avoids mixed units and keeps policy learning sane.

There are two big design principles baked into this:

1. Physics of the world are data.
   The engine enforces rules, but the YAML defines the rules.

2. Values are data.
   We don't secretly slip "capitalist is good" or "self-care is virtuous" into the code. We spell out how we score a life at the end. You can change it.

That separation turns moral philosophy into a literal config file, which is exactly the point.

Words of Estimative Probability on this section matching system intent and direction: high (~90 percent).

## 2. Survival model: the meters that define a life

At the core of the world is a set of tracked quantities we call "meters" or "bars". These are continuous values from 0.0 to 1.0. They represent needs, resources, and physical condition.

The engine treats these as authoritative state. The learning agent sees them (or a partial view of them, depending on observability level), and acts to keep itself alive, functional, and ideally well off.

### Canonical meters

Right now the runtime keeps eight core meters in a fixed order. The order matters because it's baked into tensors and models.

```
Index  Name        Meaning
0      energy      Can you still move and act?
1      hygiene     How clean / disease-free are you?
2      satiation   How fed are you?
3      money       How much spendable cash you have? (1.0 ≈ $100)
4      mood        Are you mentally okay?
5      social      Are you connected to other people?
6      health      Are you medically stable?
7      fitness     Are you physically resilient?
```

Those indices are wired everywhere (policy nets, replay buffers, cascade maths, affordance effects). Changing them casually will break everything. So we treat them as stable ABI.

Some important notes about how these behave:

* Energy and health are pivotal.
  If either hits zero, you are done. You can no longer continue the run. This is classic "you collapsed" / "medical failure".

* Money is a gate, not a "feelings bar".
  You don't die when you're broke, but most recovery options cost money. Affordability is enforced at interaction time (you can't buy dinner if you've got nothing) and through affordance configs. Poverty isn't instant death, it's slow death.

* Hygiene, satiation, mood, social, fitness are all indirect killers.
  None of them instantly kill you by dropping to zero, but they lean on the pivotal bars through cascades. For example: if you never eat, your health and energy drain harder. If you never shower, your mood and fitness degrade, which then feeds back into health. It's a pressure network not a single fail switch.

* Meters are clamped after every update.
  Nothing goes below 0.0 or above 1.0. That means "perfect health" is literally 1.0, and we don't allow "extra health banked for later". Same for money: by default we cap at 1.0 = $100, though you can redefine that scale in config packs if you want a wealth-based sim instead of a subsistence sim.

This pressure network is where the interesting behaviour comes from. Not eating doesn't just make you "hungry", it accelerates death indirectly by smashing the things that actually kill you (energy, health). Being lonely doesn't just make you "sad", it quietly compromises mood, which bleeds into energy, which gets you killed when you can't get out of bed to buy food. It's all coupled.

### Lifecycle: you don't just die, you can also finish

On top of those eight meters we also track a separate scalar we call `lifecycle`.

Think of `lifecycle` as "how far through your life you are". It starts near 0.0 and drifts upward over time toward 1.0. When it reaches 1.0, you don't "die". You retire. The run ends, but it ends in retirement, not collapse.

This matters for two reasons.

1. Episodes no longer end just because a hard-coded step limit says "class dismissed".
   The run ends because the simulated life has run its course.

2. We can tell the difference between "you made it to retirement in decent shape" and "you bled out behind the kebab shop at 2:17am with $3.20 and an infection".

Those are not morally equivalent outcomes. We want the agent to know that.

Mechanically:

* lifecycle increases a little every tick as a baseline "time passes".
* bad conditions can make it climb faster. Starving, sick, and miserable people burn through their life faster.
* if lifecycle hits its cap (1.0) before you hit zero energy or zero health, you're considered retired.
* retirement is treated as a distinct terminal outcome from "death".

The engine uses that distinction. At the end of a run, it computes a one-off "final life score". That score is driven by config, not compiled into the code. Typical scoring in the baseline world will look like:

* you get credit for how much money you finished with
* you get penalised if you finished in horrible physical condition
* you get penalised if you finished in a terrible mental state

That becomes part of the total reward signal the agent learns from. Which means we're literally teaching the policy: "being alive at the end, with money in the bank and decent health, is success".

If you just die, you still get a score, but you get a big penalty multiplier (for example, 10 percent of what that same life would've been worth if you hadn't died early).

So survival isn't the only goal. Dignified survival is the goal.

### Why this design works

From a training point of view, this setup does a few beautiful things.

* It gives you two different end states with different semantics.
  "Dead" and "Retired" are not just flavours of done. They have different payouts, which means the agent has a reason to plan for retirement instead of just "delay death".

* It turns morality (what we reward at the end) into config.
  You can spin up a world where money matters most, or a world where mood and social bonds are weighted higher, or a world where healthcare at end-of-life is king. The RL agent will internalise whatever world view you give it. It's alignment by YAML.

* It moves away from short-horizon hackery.
  Instead of giving shaped "good job!" rewards for individual actions like SLEEP or EAT, we let most of the shaping come from "did you actually build a survivable, financially stable, mentally tolerable life that reached retirement". That encourages long-horizon policy learning, which is what we actually care about.

## 3. bars.yaml – what we measure, how fast it falls apart, and what ends a life

bars.yaml is where we declare what "being alive" actually means in this world.

Each meter (energy, hunger, money, etc) is defined here with:

* its index in the tensor
* its default starting value
* how fast it passively erodes every tick
* how important it is in the survival hierarchy

The engine does not guess any of this. It loads bars.yaml, validates it with Pydantic, and then uses it as ground truth in the CascadeEngine and the environment. If you lie here, the world lies everywhere.

This file also declares terminal conditions: the rules that say "this life is now over". Death-by-exhaustion and death-by-health are defined the same way as any other rule. There’s no hardcoded secret kill switch in the engine.

Finally, this is also where lifecycle lives conceptually. Lifecycle is allowed to end a run even if you're physically fine. We treat "you finished your life and retired" as a terminal condition like death, except it’s dignified and high-scoring. That part is computed at the environment / reward layer, but the idea is: bars.yaml is the canonical roster of survival-relevant meters, and lifecycle sits alongside them in design terms as “time to retirement”. We treat that as first-class in how we score an episode.

A standard bars.yaml in the baseline world looks like this:

```yaml
version: "1.0"
description: "Townlet v2 meters"
bars:
  - name: "energy"
    index: 0
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.005
    description: "Ability to act and move"
    key_insight: "Dies if zero"

  - name: "hygiene"
    index: 1
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.003
    description: "Cleanliness supports health and mood"

  - name: "satiation"
    index: 2
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.004
    description: "Food level; core driver of survival"
    cascade_pattern: "strong to energy and health when low"

  - name: "money"
    index: 3
    tier: "resource"
    range: [0.0, 1.0]
    initial: 0.5             # 0.5 = $50
    base_depletion: 0.0
    description: "Budget for affordances"

  - name: "mood"
    index: 4
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.7
    base_depletion: 0.001
    description: "Affects energy dynamics when low"

  - name: "social"
    index: 5
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.006
    description: "Supports mood over time"

  - name: "health"
    index: 6
    tier: "pivotal"
    range: [0.0, 1.0]
    initial: 1.0
    base_depletion: 0.0
    description: "General condition; death if zero (managed via fitness modulation)"

  - name: "fitness"
    index: 7
    tier: "secondary"
    range: [0.0, 1.0]
    initial: 0.5
    base_depletion: 0.0
    description: "Improves health dynamics when high"

terminal_conditions:
  - meter: "energy"
    operator: "<="
    value: 0.0
    description: "Death by exhaustion"

  - meter: "health"
    operator: "<="
    value: 0.0
    description: "Death by health failure"

notes:
  - "All values normalised. Money: 1.0 means $100."
```

Let’s unpack the moving parts like a designer, not like a compiler.

### Meters

* `tier` is a storytelling hint and a balancing hint. Pivotal means "if this goes, game over". Secondary means "feeds into pivotal via cascades". Resource means "lets you buy your way out of trouble".
  The engine doesn’t enforce tier names directly, but other parts of the system assume you’re using this vocabulary sanely.

* `initial` is what you spawn with at reset. This is important for curriculum too. If you want to generate harsher starting lives, you can start the agent broke, sick, lonely, whatever. That’s legal.

* `base_depletion` is passive decay per step before any cascades. This is the "cost of being alive". For example, hygiene decays every tick just because existing is gross. Energy decays just because being awake is tiring.

  These base rates are then modulated in cascades.yaml, which we'll get to.

  Important: we do not bake “being at work makes you tired” here. That’s an affordance effect, in affordances.yaml. bars.yaml is only the background radiation of life.

* Ranges are locked to [0.0, 1.0]. That is enforced by validation. You cannot have a bar that goes to 2.5. You want "debt"? That goes somewhere else (like a negative money affordance outcome plus a new rule in cascades that punishes negative money by smashing mood). We keep the meter space clean and normalised because the policy networks ingest these floats directly.

### Terminal conditions

This is how we decide "this run ended by force". Each condition is a tuple:

* which meter to check
* what comparison to apply
* threshold
* human-readable reason

The CascadeEngine turns these conditions into booleans every tick and tells the environment who's done. This is also what feeds the "dead agent is masked off, cannot act further" code in VectorizedTownletEnv.

You can add new terminal conditions. You could absolutely write one that says:

* if `money <= 0.0` and `mood <= 0.1` for long enough, you’re considered "institutionalised" and the run ends. That would simulate social collapse without physical death.

You can also imagine a condition like:

* if lifecycle >= 1.0 you are "retired".
  That’s a graceful terminal, not a failure state. That goes into the same mechanism and gives us two distinct finishes for scoring.

This is why we call it Universe as Code. "What counts as the end of your life?" is a setting.

---

## 4. cascades.yaml – how the world punishes neglect

bars.yaml explains what exists. cascades.yaml explains how it all starts eating you alive if you ignore it.

There are two big systems in cascades.yaml:

1. Modulations

   * Ongoing multipliers that say "if X is high/low, then Y decays slower/faster".
   * Example: high fitness protects health. Low fitness lets health rot faster.
   * This is smooth and continuous, not thresholded.

2. Threshold cascades

   * If a source meter drops below some threshold, it starts actively damaging some other meter.
   * The lower you go below threshold, the harder the hit.
   * This is where hunger ruins health, bad hygiene wrecks mood, isolation wrecks mood, and so on.

This is where we encode real human collapse patterns without having to handwrite 200 lines of if/elif. Hunger drains both energy and health? Good. That’s survival pressure. No social contact eventually bleeds into mood, which bleeds into energy, which stops you earning money, which stops you buying food, which tanks satiation, which kills you. This is how we get spirals.

Now, crucial detail: execution_order in this file is not documentation fluff. The engine actually reads that list and applies those cascade groups in that order every tick. If you move stages around, you change the physics.

Here’s the current baseline cascades.yaml:

```yaml
version: "1.0"
description: "Townlet cascade physics"
math_type: "gradient_penalty"

modulations:
  - name: "fitness_modulates_health_decay"
    description: "Good fitness reduces health decay; poor fitness increases it"
    source: "fitness"
    target: "health"
    type: "depletion_multiplier"
    base_multiplier: 0.5       # at fitness = 1.0
    range: 2.5                 # 0.5 + 2.5 = 3.0 at fitness = 0.0
    baseline_depletion: 0.002  # align with bars.health.base_depletion
    note: "Replicates legacy behaviour"

cascades:
  # secondary to pivotal via primary_to_pivotal category label
  - name: "low_satiation_hits_energy"
    description: "Hungry drains energy quickly"
    category: "primary_to_pivotal"
    source: "satiation"
    source_index: 2
    target: "energy"
    target_index: 0
    threshold: 0.2
    strength: 0.015

  - name: "low_satiation_hits_health"
    description: "Hungry also undermines health"
    category: "primary_to_pivotal"
    source: "satiation"
    source_index: 2
    target: "health"
    target_index: 6
    threshold: 0.2
    strength: 0.010

  - name: "low_mood_hits_energy"
    description: "Depressed, low energy"
    category: "primary_to_pivotal"
    source: "mood"
    source_index: 4
    target: "energy"
    target_index: 0
    threshold: 0.2
    strength: 0.010

  - name: "low_fitness_hits_health"
    description: "Unfit, poor health dynamics"
    category: "primary_to_pivotal"
    source: "fitness"
    source_index: 7
    target: "health"
    target_index: 6
    threshold: 0.2
    strength: 0.010

  # tertiary to secondary via secondary_to_primary category label
  - name: "low_hygiene_hits_secondary"
    description: "Poor hygiene worsens satiation, mood, fitness dynamics"
    category: "secondary_to_primary"
    source: "hygiene"
    source_index: 1
    target: "satiation"
    target_index: 2
    threshold: 0.2
    strength: 0.006

  - name: "low_hygiene_hits_mood"
    description: "Hygiene affects mood"
    category: "secondary_to_primary"
    source: "hygiene"
    source_index: 1
    target: "mood"
    target_index: 4
    threshold: 0.2
    strength: 0.006

  - name: "low_hygiene_hits_fitness"
    description: "Hygiene affects fitness"
    category: "secondary_to_primary"
    source: "hygiene"
    source_index: 1
    target: "fitness"
    target_index: 7
    threshold: 0.2
    strength: 0.006

  - name: "low_social_hits_mood"
    description: "Isolation reduces mood"
    category: "secondary_to_primary"
    source: "social"
    source_index: 5
    target: "mood"
    target_index: 4
    threshold: 0.2
    strength: 0.010

  # weak tertiary to pivotal route if desired
  - name: "low_hygiene_hits_health_weak"
    description: "Hygiene to health weak path"
    category: "secondary_to_pivotal_weak"
    source: "hygiene"
    source_index: 1
    target: "health"
    target_index: 6
    threshold: 0.1
    strength: 0.003

execution_order:
  - "modulations"
  - "primary_to_pivotal"
  - "secondary_to_primary"
  - "secondary_to_pivotal_weak"

notes:
  - "Engine applies each category in order per step"
  - "Penalty = strength * ((threshold - source)/threshold) for source below threshold"
```

Let’s decode.

### Modulations

Take the first block: `fitness_modulates_health_decay`.

The English version is:

* If you're fit, your health doesn't drip away as fast.
* If you're unfit, every tick chews through your health harder.

Numerically:

* health has some baseline passive decay (baseline_depletion).
* we multiply that decay by a fitness-based multiplier.
* when fitness is high (near 1.0), the multiplier is low (0.5x).
* when fitness is wrecked (near 0.0), the multiplier is high (0.5 + 2.5 = 3.0x).
* so going to the gym is literally adding days to your future.

This is not thresholded. It's continuous. You don't suddenly "start dying" at some cut-off. You just always decay faster if you're unfit.

That’s not just biologically plausible, it's RL-friendly. The gradient is smooth. The model can learn "small fitness gain = slightly better survival".

### Threshold cascades

Now look at `low_satiation_hits_energy`.

The English version:

* If you're hungry (satiation < 0.2), you get tired dramatically faster than normal.
* The hungrier you are below that line, the worse the hit.

This is implemented as:
penalty = strength * ((threshold - source) / threshold)

Step through it:

* If satiation = 0.2, you're fine. No penalty.
* If satiation = 0.1, you're 50 percent below threshold, so you eat 0.5 * strength that tick.
* If satiation = 0.0, you eat full strength that tick.

That penalty is subtracted directly from the target meter (here: energy). This is literally how hunger kills you. Not by flipping one big "dead" flag, but by smashing your pivotal bars until they empty.

We repeat this in a few directions:

* Hunger → Energy and Hunger → Health
* Low Mood → Energy (depression fatigue)
* Low Hygiene → Mood / Satiation / Fitness ("you feel gross, you eat worse, you stop training, you spiral")
* Low Social → Mood ("lonely and flat")

This is how we get slow death loops that feel like real poverty. One thing goes wrong, then three things go wrong, then you're too wrecked to fix any of them.

### Execution order

`execution_order` is applied literally by CascadeEngine every tick in that sequence.

In plain terms:

1. We apply modulations. That mostly changes how fast certain core bars are draining.
2. We apply "primary_to_pivotal". This is the high-severity stuff like hunger eating energy and health.
3. We apply "secondary_to_primary". Low hygiene, low social, etc, starts eroding mood, fitness, satiation.
4. We apply "secondary_to_pivotal_weak". This is the low-grade direct rot from tertiary straight to lethal bars.

Reordering those changes the feel of the world. You can make hygiene hit health immediately and brutally by moving "secondary_to_pivotal_weak" earlier and bumping the strength numbers. You have that control.

### Difficulty as curriculum, not code

One ridiculously powerful trick here is "teaching packs":

* cascades_weak.yaml
  halve all `strength` values
  result: basically a comfy society. You can slack on hygiene for a while and not die.

* cascades_strong.yaml
  boost all `strength` values by 50 percent
  result: a hostile world. You miss dinner, you're functionally in medical trouble by morning.

Because all this is in YAML and validated, we can do structured curriculum: start the agent in weak, then promote to baseline, then to strong, and measure if it's actually learning survival strategies rather than just memorising coordinates.

And yes: lifecycle plugs into this story. If your cascades are harsh, many lives won't make it to retirement. So the final score doesn't just track "did you technically survive", it tracks "what kind of civilisation did you survive in, and how beat up were you when you cashed out".

---

## 5. affordances.yaml – what you can actually do in the world

If bars.yaml says what you are, and cascades.yaml says how you fall apart, affordances.yaml says what you can do about it.

In plain language: every meaningful action in the world (sleep, eat, work a shift, go to therapy, call an ambulance) is defined here as an affordance. The simulation doesn't special-case "work" or "sleep" or "call 000". They’re all entries in this list.

Each affordance is basically a contract:

* how you interact with it (instant vs multi-tick etc)
* what it costs you to use
* what it gives you back
* when it's open
* how long you need to stay with it before it pays off

The world loads this YAML at runtime and that becomes the action surface. So if you change affordances.yaml, you're not tweaking balance. You're literally changing what kind of civilisation this is.

### Interaction types (how time flows when you do it)

We currently support four interaction types:

* `instant`
  One-tick action. You press INTERACT, it fires, you're done (e.g. Shower, Eat). Good for impulses and emergencies.

* `multi_tick`
  You have to commit for a number of ticks to finish the activity (e.g. Sleeping, Working a shift, Gym). If you walk off the tile early you lose progress and you don't get the completion bonus.

* `continuous`
  Conceptually "stays on while you remain here", tick after tick. Think "holding down a lever." We include this mode in the schema because it's useful design space, even if most of the current baseline things are expressed as multi_tick.

* `dual`
  Hybrid forms that act like instant on the first tick and then sustain like multi_tick across more ticks. This gives us stuff like "check in at hospital (instant triage), then stay under observation (multi_tick)". The engine already understands this shape, even if not every world uses it yet.

Under the hood, the environment tracks per-agent `interaction_progress` for multi_tick and dual actions, and clears it if you leave.

If you're thinking "that's how you'd code a job shift, sleep, or physio rehab" – yes. That's exactly why it exists.

### Key fields in each affordance entry

* `id` and `name`
  These link to map placement. The environment uses these names to spawn tiles and to align observation channels. If you rename "Bed" to "NapCasket" in YAML but don't update the env layout, RIP.

* `interaction_type`
  Tells the env how to advance progress and when to pay bonuses.

* `required_ticks`
  How long you need to commit before the thing counts as "done". Only valid for multi_tick and dual. Omit it for instant.

* `operating_hours`
  Opening window in hours. `[9, 18]` means 09:00–18:00 inclusive of start, exclusive of end. `[18, 28]` means 18:00–04:00 next day. Time-of-day logic lives in the env and checks this every tick.

* `costs` / `costs_per_tick`
  What you lose. This can be money, energy, hygiene, whatever. cost is upfront (instant). costs_per_tick is ongoing (multi_tick / dual). All amounts are normalised: 0.10 money means $10 in the default scaling.

* `effects` / `effects_per_tick`
  What you gain (or lose, if it's negative). Same idea: effects is instant, effects_per_tick is per tick during ongoing interactions.

* `completion_bonus`
  What you get when you fully complete the interaction (i.e. finish required_ticks without bailing). This lets us do things like "sleep four ticks, and on the last tick you get an extra bump to energy" or "finish a full shift and get paid".

The environment enforces "you can only INTERACT if you're standing on the tile and it's open right now". It does not enforce affordability before letting you try. This is intentional.

If you hit INTERACT on a thing you can't afford, the tick is basically wasted and you feel bad. Lesson learned: being broke is not just an integer problem. It's time loss, and time loss is survival pressure. This is training signal.

### The baseline affordance set

This is the reference economy / lifestyle for the default world:

```yaml
version: "1.0"
description: "Townlet affordances"
status: "PRODUCTION"

affordances:
  - id: "bed"
    name: "Bed"
    category: "energy_restoration"
    interaction_type: "multi_tick"
    required_ticks: 4
    costs: []                       # instant costs (not used here)
    costs_per_tick: []              # free to sleep
    effects: []                     # not used for multi_tick
    effects_per_tick:
      - { meter: "energy", amount: 0.25, type: "linear" }
    completion_bonus:
      - { meter: "energy", amount: 0.25 }
    operating_hours: [0, 24]
    teaching_note: "Cheap but slow recovery"

  - id: "luxury_bed"
    name: "LuxuryBed"
    category: "energy_restoration"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs_per_tick:
      - { meter: "money", amount: 0.05 }    # 5 dollars per tick
    effects_per_tick:
      - { meter: "energy", amount: 0.30, type: "linear" }
    completion_bonus:
      - { meter: "hygiene", amount: 0.10 }
    operating_hours: [0, 24]
    design_intent: "Faster but costs money"

  - id: "shower"
    name: "Shower"
    category: "hygiene_recovery"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.02 }    # 2 dollars
    effects:
      - { meter: "hygiene", amount: 0.40 }
    operating_hours: [0, 24]

  - id: "home_meal"
    name: "HomeMeal"
    category: "food"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.05 }    # 5 dollars
    effects:
      - { meter: "satiation", amount: 0.40 }
    operating_hours: [6, 24]

  - id: "fast_food"
    name: "FastFood"
    category: "food"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.08 }
    effects:
      - { meter: "satiation", amount: 0.50 }
      - { meter: "hygiene", amount: -0.05 }
    operating_hours: [10, 28]   # 10 to 4

  - id: "job"
    name: "Job"
    category: "income"
    interaction_type: "multi_tick"
    required_ticks: 8
    costs_per_tick:
      - { meter: "energy", amount: 0.05 }   # see env movement costs too
    effects_per_tick:
      - { meter: "money", amount: 0.10 }    # 10 dollars per tick
      - { meter: "mood", amount: -0.02 }    # consumed focus reduces mood
    completion_bonus:
      - { meter: "money", amount: 0.40 }
    operating_hours: [9, 18]

  - id: "labor"
    name: "Labor"
    category: "income"
    interaction_type: "multi_tick"
    required_ticks: 4
    costs_per_tick:
      - { meter: "energy", amount: 0.08 }
      - { meter: "hygiene", amount: 0.03 }
    effects_per_tick:
      - { meter: "money", amount: 0.12 }
      - { meter: "fitness", amount: 0.04 }
    completion_bonus: []
    operating_hours: [7, 17]

  - id: "gym"
    name: "Gym"
    category: "fitness_builder"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs_per_tick:
      - { meter: "energy", amount: 0.06 }
      - { meter: "money", amount: 0.05 }
    effects_per_tick:
      - { meter: "fitness", amount: 0.12 }
      - { meter: "mood", amount: 0.04 }
    completion_bonus: []
    operating_hours: [6, 22]

  - id: "bar"
    name: "Bar"
    category: "social_mood"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.20 }
    effects:
      - { meter: "social", amount: 0.30 }
      - { meter: "mood", amount: 0.15 }
    operating_hours: [18, 28]

  - id: "park"
    name: "Park"
    category: "mood_social_free"
    interaction_type: "instant"
    costs: []
    effects:
      - { meter: "mood", amount: 0.10 }
      - { meter: "social", amount: 0.05 }
    operating_hours: [6, 20]

  - id: "recreation"
    name: "Recreation"
    category: "mood_tier1"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs_per_tick:
      - { meter: "money", amount: 0.05 }
    effects_per_tick:
      - { meter: "mood", amount: 0.20 }
    completion_bonus: []
    operating_hours: [10, 22]

  - id: "therapist"
    name: "Therapist"
    category: "mood_tier2"
    interaction_type: "multi_tick"
    required_ticks: 3
    costs_per_tick:
      - { meter: "money", amount: 0.15 }
    effects_per_tick:
      - { meter: "mood", amount: 0.25 }
    completion_bonus: []
    operating_hours: [9, 17]

  - id: "doctor"
    name: "Doctor"
    category: "health_tier1"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs_per_tick:
      - { meter: "money", amount: 0.15 }
    effects_per_tick:
      - { meter: "health", amount: 0.25 }
    completion_bonus: []
    operating_hours: [8, 18]

  - id: "hospital"
    name: "Hospital"
    category: "health_tier2"
    interaction_type: "multi_tick"
    required_ticks: 2
    costs_per_tick:
      - { meter: "money", amount: 0.30 }
    effects_per_tick:
      - { meter: "health", amount: 0.40 }
      - { meter: "energy", amount: 0.10 }
    completion_bonus: []
    operating_hours: [0, 24]

  # Ambulance by phone within the current engine model
  - id: "call_ambulance"
    name: "HomePhoneAmbulance"
    category: "health_emergency"
    interaction_type: "instant"
    costs:
      - { meter: "money", amount: 0.50 }     # 50 dollars
    effects:
      - { meter: "health", amount: 0.30 }    # immediate stabilisation
      - { meter: "energy", amount: -0.05 }   # shock and fatigue
    operating_hours: [0, 24]
    teaching_note: "Approximates ambulance response without movement"
```

You can read a lot of social policy in there, by the way.

For example:

* Being poor makes it harder to rest well (LuxuryBed costs money but restores energy faster and helps hygiene).
* Working pays money but erodes mood, and it costs energy, so there's a grind loop.
* Mood is not free. You can buy mood at the bar (social + mood, but expensive), or you can generate mood slowly and cheaply through Recreation or Park.
* Health care is time-gated and price-gated. Emergency is available always, but it's ruinously expensive and leaves you tired.

That’s intentional. The world is already expressing class, labour, burnout, social life, and health access through config.

### Emergency care and mobility

The ambulance affordance is a design hack around a current engine limit: the engine doesn't yet directly support "this interaction teleports you somewhere else". So instead we fake ambulance like this:

* It’s `instant`.
* It's always open.
* It's very expensive.
* It spikes your health up so you don't die this tick.
* It drains some energy to reflect shock.

That buys you survival at the cost of future affordability. It is absolutely meant to teach: "if you let your health crash, you are now in debt to stay alive". This gives the agent a reason to plan for preventative care instead of constantly gambling and hitting panic buttons.

WEP that this mapping is aligned with how we've actually got `call_ambulance` wired right now: high (~85 percent).

### Optional extension: physical relocation

Eventually (and frankly this is inevitable), the ambulance should physically move you to the Hospital tile.

Minimal engine change to support that:

1. Add a richer effect type to the model:

```python
class AffordanceEffect(BaseModel):
    meter: str | None = None
    amount: float | None = None
    type: Literal["linear", "relocate"] | None = None
    destination: str | None = None
```

2. In `AffordanceEngine.apply_instant_interaction`, detect `type == "relocate"`. Instead of trying to change a meter, you emit a side-channel update, e.g.:

```python
info["relocations"] = [(agent_idx, destination_name)]
```

3. In `VectorizedTownletEnv._handle_interactions_legacy`, consume that `relocations` list after applying effects, and literally set the agent’s position to the coordinates for that destination affordance.

With that in place, you can write an ambulance affordance like:

```yaml
- id: "call_ambulance"
  name: "HomePhoneAmbulance"
  category: "health_emergency"
  interaction_type: "instant"
  costs:
    - { meter: "money", amount: 0.50 }
  effects:
    - { type: "relocate", destination: "Hospital" }
  operating_hours: [0, 24]
```

Now it's not just "patch your health", it's "you lose autonomy and wake up in hospital". Which is honestly what we want for realism and for training. It teaches: there's a world state transition after crisis.

Words of Estimative Probability that the relocation hook as described can be slotted into the current engine with low blast radius: moderate (~70 percent). It assumes we're comfortable piping side-channel info back into `VectorizedTownletEnv`, which is consistent with how we already pass other per-step metadata.

---

## 6. Time of day and action masks

Time is part of the physics, not just flavour.

* Every affordance has `operating_hours`. That's enforced tick by tick.

  * `[open, close]` with 0 ≤ open < close ≤ 24 means it's open from that hour until just before the close hour.
  * If close > 24 (up to 28), that means "this wraps past midnight". For example `[18, 28]` means open at 18:00 and stays open through midnight until 04:00.

* The environment keeps a `time_of_day` per world. If temporal mechanics are enabled in config, it increments:
  `time_of_day = (time_of_day + 1) % 24` every tick.

* You only get INTERACT unmasked if:

  * you're standing on the tile,
  * that tile's affordance is open at the current hour.

Money is not part of that masking. You can still try to buy something you can’t afford. The idea there is: "poverty wastes your turns" is an experience we want the agent to discover, not something we silently protect it from.

### Action space

The universal action set is:
`[UP, DOWN, LEFT, RIGHT, INTERACT, WAIT]`

The environment charges basic movement / idle costs automatically:

* moving:

  * energy ~0.005
  * hygiene ~0.003
  * satiation ~0.004
    (all configurable via `environment.energy_move_depletion` etc in the run config)

* waiting:

  * lower energy cost, default ~0.001
  * no hygiene hit, no satiation hit
    (tunable via `environment.energy_wait_depletion`)

* interacting:

  * a separate `energy_interact_depletion` per press
    default 0.0, but can be increased if you want "using things is tiring"

Because those are environment-level knobs, you can do whole-world difficulty variants like:

* "high commute tax": walking is brutally expensive
* "burnout society": interaction always drains energy
* "welfare state": waiting is cheap and not so punishing

This is one of the main levers for making a world where a long workday is viable vs a world where just getting to work is dangerous.

WEP that this matches the current VectorizedTownletEnv semantics is high (~90 percent).

---

## 7. Multi-tick semantics

Some things take time, and commitment is part of the mechanic.

For any affordance with `interaction_type: "multi_tick"` or `"dual"`:

* The environment tracks `interaction_progress` on a per-agent basis.
* Each tick you stay on that same affordance tile and keep hitting INTERACT (or auto-continue, depending on how we wire control), `interaction_progress` increments.
* If you walk off the tile, switch to another affordance, or the place closes mid-way, progress resets. No completion bonus. Tough luck.

At the tick where you hit `required_ticks`:

* You apply that last tick’s `effects_per_tick`.
* Then you apply the `completion_bonus` one-off.
* Then progress resets.

Why this matters:

* Sleep only pays off if you stay down.
* Work only pays off if you stay the whole shift.
* Gym only pays off if you finish the block.
* The agent has to learn "sprint to bed, stay there, defend that choice against distractions" instead of just spamming micro-sleeps.

That’s not just fun for gameplay. It's also curriculum-friendly. It teaches planning and follow-through.

WEP: high (~95 percent). This is exactly how the code handles multi-tick.

---

## 8. Positions and observation alignment

The world is not abstract. It's spatial.

`VectorizedTownletEnv` seeds a default layout by placing each named affordance at a specific grid coordinate. Observations encode both:

* your current position, and
* what’s at each position in the map.

That means names in affordances.yaml must line up with what the env expects to place on the map. If you rename "Gym" to "FitnessCentre" in the YAML but the env layout still calls it "Gym", your agent will be standing on something that doesn't exist in config terms. The affordance won't fire. You’ll cry, and you will deserve it.

Here’s the reference layout:

```text
Bed:        [1,1]
LuxuryBed:  [2,1]
Shower:     [2,2]
HomeMeal:   [1,3]
FastFood:   [5,6]
Job:        [6,6]
Labor:      [7,6]
Gym:        [7,3]
Bar:        [7,0]
Park:       [0,4]
Recreation: [0,7]
Therapist:  [1,7]
Doctor:     [5,1]
Hospital:   [6,1]
```

The environment also serialises and restores these positions in checkpoints. That way, training runs and eval runs agree on what "go to the Hospital" means in world coordinates, and the agent isn't punished by randomised geography unless we explicitly ask for that.

This positional stability is important for two reasons:

1. The agent can actually learn spatial habits like "bed is north-east of start".
2. We can tune economic geography.
   Example: Put "Job" far from "Hospital", and you’ve just modelled a world where work is physically incompatible with health care access for low-energy agents. That is not theory. That is game law once you check it in.

---

## 9. Validation rules (what the loaders will yell at you about)

Universe as Code is meant to be data-driven, but it is not meant to be "anything goes". The loaders enforce hard structure so we don't get weird half-worlds.

Here’s what’s enforced when we load configs:

### bars.yaml

* There must be exactly eight bars.
* They must use indices 0 through 7, with no gaps and no duplicates.
* The names must match those indices one-to-one.
* Each bar’s range must be `[0.0, 1.0]`.
  We never allow a bar to run "hot" above 1.0 or dip below 0.0. The engines clamp.

If you invent a ninth bar like "spiritual_alignment" and try to sneak it in, the loader will simply refuse to build the world. That’s deliberate. Bars are part of the world ABI and downstream models get very cranky if you silently reshuffle them.

### cascades.yaml

* Every cascade block needs a unique `name`. No duplicates.
* `category` is required, but not restricted. It's how we group cascades for execution order.
  For example:

  * `primary_to_pivotal`
  * `secondary_to_primary`
  * `secondary_to_pivotal_weak`
    These category labels show up again in `execution_order`, which is basically "run this group of cascades in this order every tick".
* Modulation entries must provide the specific depletion multiplier fields (`base_multiplier`, `range`, etc) in the right shape. If you leave them out or rename them, we can't compute health decay properly and the loader will complain.

This is what keeps health decay, hunger crash, hygiene rot and so on all mathematically coherent instead of random if/else logic.

### affordances.yaml

* `interaction_type` has to be one of the supported literals:

  * `instant`
  * `multi_tick`
  * `continuous`
  * `dual`

  If you invent `"sustained_brooding"` as an interaction_type and forget to update the engine, the loader will block you. Again: ABI stability.

* `required_ticks`:

  * Must exist for `multi_tick` and `dual`.
  * Must not exist for `instant`.
    If you put `required_ticks` on an instant affordance, the loader treats that as illegal because there is no progress counter to advance.

* `operating_hours` must be a two-integer list and must make temporal sense:

  * First value (open) must be 0 to 23.
  * Second value (close) must be 1 to 28.
  * Close can go past 24 to indicate wrap-around (late night into early morning).
    This contract is what lets the time-of-day mask work without bespoke per-affordance code.

This validation layer gives us confidence that if a world loads, it's at least internally legal. It also means you can ship variant packs ("austerity", "boom", custom dystopia) without having to ship new Python with them.

WEP on this section being true to the current Pydantic validations: high (~90 percent).

## 10. Equivalence, tests, and teaching packs

We are replacing a bunch of hardcoded behaviour with config. That’s lovely in theory, but there's a risk: you change a YAML number and quietly change the survival game without noticing.

Two things keep us sane:

### 1. Behavioural equivalence tests

We already have legacy logic in code for:

* base depletion
* cascade order
* hospital behaviour
* etc

We keep that code around in tests, then run the new config-driven engine with the same inputs and assert that outputs match within tight tolerance. That gives us regression alarms if a refactor accidentally changes physics.

Basically: "the new universe still feels like the old universe unless you meant to change it".

### 2. Teaching packs

On top of baseline world packs, we explicitly carry curated variants that model different economic or social conditions. These are small diffs, not forks.

Recommended packs:

* `cascades_weak.yaml`
  Halves all the cascade `strength` values. Hunger, hygiene collapse, loneliness and similar still hurt, but they're gentle. This is our "easy mode / tutorial biology".

* `cascades.yaml`
  Baseline. This is the intended default physical/physiological model for the sim.

* `cascades_strong.yaml`
  Roughly +50 percent to cascade `strength`. Hunger punishes energy and health much harder. Hygiene crashes mood brutally. This is "you are fragile, plan ahead".

With just those three knobs, you can already teach:

* resilience,
* fragility,
* and how fast neglect kills you.

Affordance packs sit alongside cascade packs and give us socioeconomic regimes. Some obvious examples:

* "austerity"
  Sleep and food get cheaper (Beds are basically free and HomeMeal is low money cost), but labour income drops and health care costs more. You live, barely, but you stay broke and sick.

* "boom"
  Wages go up. Energy recovery gets pricier (LuxuryBed price per tick spikes; Shower costs more). You're rich enough to paper over your basic needs, but you're bleeding money just to exist clean and rested.

* "nightlife"
  Bars and Recreation stay open deep into wrap-around hours `[18, 28]`. Gym might open later. Doctor hours might even shrink. Health access becomes clock-gated while mood/social access is always available. The world is telling you: it is easier to drink than to heal.

Because these are just config packs, designers can explore political economy by editing YAML, not Python.

WEP this describes how we actually intend people to fork worlds: very high (~95 percent).

## 11. Emergency care path (Doctor, Hospital, Ambulance) as an intentional gameplay loop

The medical pathway in Universe as Code is not an afterthought. It's one of the core survival loops. The idea is: health is pivotal, time is scarce, and treatment has structure and cost.

Right now, an agent in trouble has three main options:

1. Go to the Doctor (during office hours)

   * Limited hours.
   * Cheaper per tick.
   * Slower.
   * Represents primary care / clinic / GP level service.

2. Go to the Hospital

   * Always open.
   * Faster restoration.
   * More expensive.
   * Represents acute care. You burn money to not die.

3. Use `HomePhoneAmbulance`

   * Available anywhere in the map, any time of day.
   * One-tick `instant` action. No long commute.
   * Extremely expensive.
   * Immediately stabilises health and (in the forward-looking relocation version) dumps you at Hospital, removing your control of the situation.

That's not just flavour. That is policy, encoded in YAML.

What that teaches the agent:

* Preventative care is cheaper, but time-gated.
* Emergency care will save you now, but it will wreck you financially, and possibly leave you stranded somewhere else with new constraints.
* There is an economy to health. You can absolutely be too poor to stay alive comfortably. That will be reflected in your long-run survival reward.

It also gives the model a reason to learn the clock. If health is low at 17:45, sprinting to the Doctor before they close at 18:00 is a completely rational move.

WEP alignment to current ambulance design and the proposed "relocate to Hospital" hook: ~80 percent (mild extrapolation on relocation behaviour, but consistent with section 5).

## 12. How the tick actually runs under the hood

This section is mostly here so engineers don't get surprised when they're stepping through the code.

On each tick for each living agent, high level order of operations is:

1. We process the agent's chosen action:

   * movement (`UP/DOWN/LEFT/RIGHT`)
   * INTERACT
   * WAIT

   Movement and WAIT costs come from the environment (`VectorizedTownletEnv._execute_actions`), not from YAML:

   * Moving drains energy, hygiene, satiation at fixed per-step amounts (tunable via env config).
   * Waiting drains a little energy and basically nothing else.
   * Interact applies an `energy_interact_depletion` cost per use (default often 0.0, but configurable).

2. If the action was INTERACT and the location/operating_hours allow it:

   * The AffordanceEngine applies any instant costs/effects.
   * For multi_tick / dual, it also advances `interaction_progress`, applies per-tick effects, and checks for completion to apply `completion_bonus`.

3. After actions and affordance effects, we run base depletions from bars.yaml through the CascadeEngine. This is your passive decay each tick. This can be globally scaled by curriculum if we want to make "survival training worlds" harsher or softer without touching YAML.

4. We apply cascades in the order defined in `cascades.yaml` under `execution_order`. The canonical sequence is:

   * `depletions` (base drain)
   * `primary_to_pivotal`
   * `secondary_to_primary`
   * `secondary_to_pivotal_weak`

   These map to code like:

   * `apply_primary_to_pivotal_effects`
   * `apply_secondary_to_primary_effects`
   * etc

   The point is: hunger hits energy/health before hygiene hits mood, and so on. Order matters. This is how the system models spirals.

5. We clamp all bars to [0.0, 1.0].

6. We evaluate terminal conditions:

   * If any pivotal bar hits zero (e.g. `energy <= 0`, `health <= 0`), that agent is declared "dead" (or "retired" if you're using a softer narrative layer).
   * Dead/retired agents are then masked: they no longer get valid actions. From that point, they're effectively frozen in time for that episode, and the RL loop can treat that as end-of-life reward accounting.

So from an RL perspective, "I stayed alive another tick" is literally "the engine didn't mask me out". Longevity reward is clean to compute.

WEP alignment with live code: high (~90 percent). The only light abstraction here is that I grouped steps conceptually.

## 13. Config packs and world folders

A world, in practice, is just a folder of YAML.

At minimum, that folder needs:

```text
configs/test/
  bars.yaml
  cascades.yaml
  affordances.yaml
```

You point the environment at this directory and it boots that reality.

* `bars.yaml` says which meters exist, how they deplete, and how you can literally die.
* `cascades.yaml` says how those meters punch each other in the face over time.
* `affordances.yaml` says what you’re allowed to do about it.

If you want to layer in social signalling, scripted events, NPC cues, etc, you add new files alongside those. For example, we expect to introduce `cues.yaml`, which will describe social and environmental signals the agent can perceive without directly interacting. That will feed perception/pretraining modules, not just control.

The big idea here is: content designers, economists, narrative designers and AI researchers are now editing the same source of truth. The point of "Universe as Code" is that the world isn't hand-waved in prose, it's executable.

## 14. Walkthrough: one crisis tick with ambulance

Let's run a single tick to show how all of this fits together.

Scenario:

* Time: 02:00 (2 am)
* Agent is at home
* health = 0.18
* energy = 0.40
* money = 0.60
* Doctor is closed
* Hospital is open but it's a walk away
* Agent is in trouble

Tick timeline:

1. The agent chooses INTERACT on `HomePhoneAmbulance`.

   * This affordance is `interaction_type: "instant"`.
   * It is open 24/7.
   * The environment checks: are we standing on the tile that offers `HomePhoneAmbulance`, and is the current hour within `[0, 24]`?
     Yes. So the action is valid.

2. AffordanceEngine fires the instant interaction:

   * Deducts the cost:

     * money goes from 0.60 to 0.10 (cost 0.50, read as 50 dollars).
   * Applies effects:

     * health jumps from 0.18 to 0.48.
     * energy drops from 0.40 to 0.35 (shock/fatigue penalty).

   At this point the agent is still alive. The immediate emergency is contained.

   If we’ve implemented the relocation extension:

   * The interaction also emits a relocation event like `("agent0", "Hospital")`.
   * The environment consumes that event and teleports the agent to the Hospital coordinates before cascade resolution.
   * Narrative read: "you blacked out in the lounge and woke up in hospital".

3. We now run the normal tick housekeeping:

   * Base depletions tick all bars according to `base_depletion`.
   * Cascades fire in the configured `execution_order`.
     For example: if satiation is very low, it will hammer energy and health again, so if you're starving on top of being injured, you're still in danger even after ambulance.
   * All meters are clamped [0.0, 1.0].

4. We check terminal conditions:

   * If health or energy fell to 0.0 or below after cascades, you're done.
   * Otherwise you continue into the next tick at 02:01, either at home (no relocation version) or in Hospital (relocation version).

Game design read:

* You stayed alive, but you torched your money.
* You might have been moved somewhere where you can now get better care, but you also lost positional freedom.
* You are deep in a liability spiral you will have to dig out of by working, which will cost mood and energy.

RL read:

* The agent sees ambulance as a last-ditch survival affordance that is extremely costly.
* The agent gets a strong signal that avoiding this state is better than repeatedly invoking it.
* Long-term policy should learn to manage health proactively, keep money in reserve, and respect clinic hours.

That is exactly the behaviour we want to see emerge.

WEP that this walkthrough matches actual engine step order and affordance semantics: high (~90 percent).
