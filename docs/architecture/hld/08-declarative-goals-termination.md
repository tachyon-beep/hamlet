## 8. Declarative Goals and Termination Conditions

---

Townlet agents pursue explicit high-level goals—SURVIVAL, THRIVING, SOCIAL—and can report which goal is active at any moment.

We do two things:

1. We make goals explicit data structures, not vague "the RL policy probably cares about reward shaping".
2. We make "I'm done with this goal" a declarative rule in YAML, not a secret lambda hidden in code.

### 8.1 Goal Definitions Live in Config, Not in Python

We define goals in a small, safe DSL inside the run snapshot. For example:

```yaml
goal_definitions:
  - id: "SURVIVAL"
    termination:
      all:
        - { bar: "energy", op: ">=", val: 0.8 }
        - { bar: "health", op: ">=", val: 0.7 }

  - id: "GET_MONEY"
    termination:
      any:
        - { bar: "money", op: ">=", val: 1.0 }       # money 1.0 = $100
        - { time_elapsed_ticks: ">=", val: 500 }
```

Conventions:

- All bars (energy, health, mood, etc) are normalised 0.0–1.0 based on universe_as_code.yaml.
  So 0.8 means "80 percent of full", not "magic number 80".
- Money can also be normalised. e.g. `money: 1.0` means $100 if the world spec defines $100 ↔ 1.0.
- `termination` can use `all` or `any` blocks.
- Leaves are simple comparisons on bars or runtime counters (`time_elapsed_ticks`, etc). No arbitrary Python. No hidden side effects.

At runtime:

- The meta-controller (in hierarchical_policy) picks a goal struct (SURVIVAL, GET_MONEY, etc).
- Each tick (or every N ticks) it evaluates that goal's termination rule using a tiny interpreter.
- If the termination rule fires, that goal is considered satisfied, and the meta-controller may select a new one.

### 8.2 Why This Matters

- For governance/audit
  We can answer the question "Why was it still pursuing GET_MONEY while its health was collapsing?" by pointing to the YAML.
  Maybe GET_MONEY didn't terminate until health ≥ 0.7. That's a design decision, not 'the AI went rogue'.

- For curriculum
  Early in training you might define SURVIVAL as "energy ≥ 0.5 is fine". Later curriculum tightens that to 0.8. That becomes a diff in YAML, not a code poke.
  Students can directly compare behaviour when SURVIVAL is lenient versus strict.

- For teaching
  Instructors can ask: "The agent is starving but still working. Does the SURVIVAL goal terminate too late, or is the meta-controller failing to switch because greed is set too high in `cognitive_topology.yaml`?"
  That's not abstract RL theory, that's direct inspection.

### 8.3 Honesty in Introspection

Now that goals are formal objects and termination is a declarative rule, we can show two different "explanations" side by side:

- current_goal (engine truth): `SURVIVAL`
- agent_claimed_reason (self-report / introspection): `"I'm going to work to save up for rent"`

Sometimes those match. Sometimes they don't.

That gap is important:

- If they match, nice, we can narrate behaviour in plain language to non-technical stakeholders.
- If they do not match, the discrepancy becomes a teaching moment: "The agent claims it is working for rent, but engine truth shows it remains in SURVIVAL mode and mis-evaluated what would keep it alive. That is a world-model error."

We log both in telemetry on purpose.

---
