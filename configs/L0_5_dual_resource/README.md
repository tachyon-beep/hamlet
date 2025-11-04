# L0.5: Dual Resource Management

## Overview

**L0.5** is an intermediate experiment between **L0 (single resource)** and **L1 (full environment)** designed to test whether **interoception-based rewards** enable multi-resource balancing.

## Environment

- **Grid**: 6×6 (36 cells)
- **Affordances**: Bed (energy) + Hospital (health)
- **Full observability**: Agent sees entire grid
- **Instant interactions**: No temporal mechanics

## Economics

| Affordance | Cost | Effect |
|------------|------|--------|
| **Bed** | $3 | +50% energy, +2% health |
| **Hospital** | $3 | +40% health, +2% energy |

- **Starting money**: $50 (vs L0's $20)
- **No income**: Agent will eventually go broke and die
- **Goal**: Learn optimal resource management before bankruptcy

## Depletion Rates

- **Energy**: 0.5% per move, 0.4% per wait, 0.2% per interact
- **Health**: 0.3% per move, 0.2% per wait, 0.1% per interact

Both meters deplete independently → agent must manage both!

## Interoception Hypothesis

With `reward = health × energy`, the agent should learn natural prioritization:

### High Energy (80%), Low Health (20%)

```
Current reward: 0.8 × 0.2 = 0.16  ❌ "Feeling sick!"

Option A: Use Bed
After: 1.0 × 0.2 = 0.20 (gain = +0.04)
ROI: LOW

Option B: Use Hospital
After: 0.8 × 0.6 = 0.48 (gain = +0.32)
ROI: HIGH ✅ → Agent should choose Hospital
```

### Low Energy (20%), High Health (80%)

```
Current reward: 0.2 × 0.8 = 0.16  ❌ "Exhausted!"

Option A: Use Bed
After: 0.7 × 0.8 = 0.56 (gain = +0.40)
ROI: HIGH ✅ → Agent should choose Bed

Option B: Use Hospital
After: 0.2 × 1.0 = 0.20 (gain = +0.04)
ROI: LOW
```

### Both Low (20% each)

```
Current reward: 0.2 × 0.2 = 0.04  🔴 "Critical!"

Either affordance gives huge ROI:
- Bed: 0.7 × 0.2 = 0.14 (gain = +0.10)
- Hospital: 0.2 × 0.6 = 0.12 (gain = +0.08)

Multiplicative penalty creates urgency for action!
```

## Expected Behavior

### Early Episodes (Random Exploration)

- Agent discovers Bed → "This helps!"
- Agent discovers Hospital → "This helps too!"
- Wasteful usage: uses both even when not needed

### Mid Episodes (Learning Prioritization)

- Agent starts using Bed when energy is low
- Agent starts using Hospital when health is low
- Still some wasteful usage

### Late Episodes (Optimal Strategy)

- Agent waits until energy ~20-30% before using Bed
- Agent waits until health ~20-30% before using Hospital
- Bounces between resources as needed
- Survives ~40-50 actions before bankruptcy

## Running L0.5

```bash
cd /home/john/hamlet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

python scripts/run_demo.py --config configs/L0_5_dual_resource/
```

## Success Metrics

1. **Resource prioritization**: Energy at Bed usage decreases over episodes
2. **Resource prioritization**: Health at Hospital usage decreases over episodes
3. **Balanced usage**: Agent uses both affordances (not fixating on one)
4. **Convergence**: Stable survival time after ~200-300 episodes
5. **Efficiency**: Fewer wasted interactions over time

## Comparison to L0

| Metric | L0 (Bed only) | L0.5 (Bed + Hospital) |
|--------|---------------|------------------------|
| Grid size | 5×5 | 6×6 |
| Affordances | 1 | 2 |
| Resources managed | Energy | Energy + Health |
| Exploration complexity | Low | Medium |
| Expected convergence | ~100 episodes | ~200-300 episodes |
| Starting money | $20 | $50 |
| Cost per interaction | $5 | $3 |

## Teaching Value

**L0.5 demonstrates that interoception rewards scale to multi-resource problems WITHOUT hand-crafted priorities.**

The multiplicative reward structure (`health × energy`) naturally creates:

- ✅ Resource-specific gradients (low energy → high ROI for Bed)
- ✅ Urgency signals (both low → huge penalty)
- ✅ Optimal timing (wait until resource is critical before spending money)

This is learned behavior, not programmed logic!
