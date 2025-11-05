---
document_type: Overview / Success Criteria
status: Draft
purpose: Draft README.md content for repository landing page
note: This serves as draft README.md content - Performance metrics TBD pending implementation
dependencies:
  - Section 11 (Implementation Roadmap)
  - Section 1-10 (Technical specifications)
last_updated: 2025-11-05
---

## SECTION 12: THE PITCH

**This is your README.md** — the first thing researchers see when they visit your repo.

---

# Townlet

> **Configuration-driven multi-agent RL for long-horizon planning, social reasoning, and emergent coordination**

Townlet is a research platform where you design worlds and minds as YAML files, not Python code. Agents learn survival strategies from sparse rewards, reason about competitors via observable cues, and coordinate through emergent communication — all without dense reward shaping or privileged information access.

**The core insight**: Most RL simulators hardcode their physics and give agents omniscience. Townlet makes worlds auditable (YAML configs), observations realistic (human observer principle), and results reproducible (cryptographic provenance).

---

## Quick Example

**Create a world** (configure physics as YAML):

```yaml
# universe_my_world.yaml
bars:
  - name: energy
    initial: 1.0
    base_depletion: 0.005  # drops 0.5% per tick

cascades:
  - source: satiation
    target: health
    threshold: 0.2  # starving
    strength: 0.010  # -1% health per tick

affordances:
  - id: bed
    effects:
      - { meter: energy, amount: 0.25 }
    costs:
      - { meter: money, amount: 0.05 }
```

**Train an agent** (no dense shaping, just survival):

```bash
townlet train --config configs/my_world/ --level L4
```

**Result**: Agent learns to work → earn money → buy food → avoid starvation, discovering multi-step strategies from sparse reward (`r = energy × health`).

**No hand-crafted rewards like**:

```python
# ❌ Dense shaping (what we DON'T do)
if action == "eat": reward += 0.1
if action == "sleep": reward += 0.1
```

**Just natural consequences**:

```python
# ✅ Sparse rewards (what we DO)
reward = energy * health  # feel good when healthy and energized
terminal_bonus = quality_of_life_at_retirement  # long-horizon planning
```

---

## Why This Exists

Most multi-agent RL simulators:

- **Hardcode physics** in Python → impossible to compare across experiments
- **Give agents omniscience** → unrealistic policies that break on deployment
- **Use dense reward shaping** → agents follow breadcrumbs, don't discover strategies
- **Black-box checkpoints** → can't audit "which brain did what"

Townlet enforces:

1. **Configuration over code** — worlds are YAML files (auditable, version-controlled)
2. **Human observer principle** — agents only see what humans could see (no telepathy)
3. **Sparse rewards** — agents discover strategies via exploration, not shaping
4. **Provenance by design** — every run is cryptographically signed (brain identity + tamper protection)

**Result**: Experiments that are reproducible, auditable, and scientifically rigorous.

---

## Key Features

### 🎯 **Sparse Rewards + Long Horizons**

- Per-tick: `r = energy × health` (no action-specific shaping)
- Terminal: Retirement bonus based on lifespan, wellbeing, wealth
- Agents learn 50+ tick credit assignment (Job → money → food → survival)

### 🧠 **8-Level Proposed Curriculum (L0-L8)**

**Current implementation**: L0-L3 (survival basics with full/partial observability)
**Planned**: L4-L8 (multi-zone navigation, multi-agent competition, emergent communication)

- **L0-3**: Learn survival basics (full observability) [IMPLEMENTED]
- **L4-5**: Navigate under uncertainty (partial observability, LSTM required) [PLANNED]
- **L6-7**: Compete with others (social reasoning via observable cues) [PLANNED]
- **L8**: Coordinate via emergent communication (signals without pre-shared semantics) [PLANNED]

See [Section 11: Implementation Roadmap](docs/implementation_roadmap.md) for development timeline.

### 👥 **Multi-Agent Social Reasoning** (Planned)

- Agents observe competitors via **public cues** (body language, location)
- No telepathy: can't see others' internal state directly
- Module C learns: `['looks_tired', 'at_job'] → predicted_state`
- Strategic resource allocation (avoid contested affordances)

### 🧬 **Population Genetics** (Planned)

- Families form, breed, and pass on learned strategies
- Child initialization: DNA crossover + weight inheritance
- Three modes: meritocratic (baseline), dynasty (inheritance), arranged (diversity)
- Research questions: Does selection work? Do protocols transfer across generations?

### 🔒 **Governance-Ready** (Planned)

- Deterministic EthicsFilter (provably enforces rules)
- Cognitive hashing (unique ID per brain configuration)
- Signed checkpoints (HMAC tamper detection)
- Glass-box telemetry (candidate action → panic → ethics → final action)

### 📊 **Configuration-Driven Science**

- Hypothesis: "Does scarcity affect behavior?"
- Config A: `fridge.cost: 0.04`
- Config B: `fridge.cost: 0.08`
- Compare results with exact config diffs
- Publish with provenance (cognitive hash proves brain identity)

---

## What You Can Build

### Research Applications

**Multi-Agent Coordination** (Future)

- Do agents learn Theory of Mind from cues?
- Can emergent communication protocols evolve?
- How does wealth inequality emerge in dynasties?

**Long-Horizon RL** (Current)

- Test credit assignment over 100+ steps
- Validate curriculum learning benefits
- Compare sparse vs dense reward shaping

**Evolutionary Dynamics** (Future)

- Does natural selection work in RL populations?
- Lamarckian vs Darwinian evolution
- Optimal mutation rates

### Educational Use Cases

**RL Pedagogy**

- Students modify YAML configs (no coding required)
- Observe behavioral changes from parameter tweaks
- Learn systems thinking (cascades create feedback loops)

**Example Assignment**:
> "Double food cost. Explain why agent behavior changed. Submit config diff + analysis."

### Policy Simulations (Future)

**Testable Interventions**

- Model welfare systems, UBI, tax policies
- Measure behavioral outcomes (work incentives, poverty rates)
- Present results with provenance (auditable)

---

## Beyond Towns: The General Platform (Future Vision)

Townlet's core abstraction (**bars + cascades + affordances**) works for any domain where:

- State evolves over time
- Variables are coupled
- Actions have effects

**Example applications** (not yet implemented, but enabled by architecture):

**Economic Simulation**

- Bars: GDP, inflation, unemployment, debt
- Affordances: raise_interest_rates, government_spending
- Research question: Can RL learn monetary policy?

**Ecosystem Management**

- Bars: predator_pop, herbivore_pop, vegetation
- Affordances: hunt, graze, migrate
- Research question: Do Lotka-Volterra cycles emerge?

**Supply Chain Optimization**

- Bars: inventory, customer_satisfaction, cash
- Affordances: order_inventory, expedited_shipping
- Research question: Can RL discover (s,S) policies?

See [Section 9: The Bigger Vision](docs/bigger_vision.md) for details.

---

## Quick Start

### Installation

```bash
git clone https://github.com/tachyon-beep/townlet.git
cd townlet
pip install -e .
```

### Run Your First Experiment (5 minutes)

```bash
# Train agent on baseline survival world (Level 0)
townlet train --config configs/baseline/ --level L0

# Agent learns: "Bed fixes energy" by episode 20
# Watch training: tensorboard --logdir runs/
```

### Customize a World (10 minutes)

```bash
# Copy baseline config
cp -r configs/baseline/ configs/my_world/

# Edit physics
vim configs/my_world/affordances.yaml
# Change: fridge.cost from 0.04 to 0.08 (make food expensive)

# Train on modified world
townlet train --config configs/my_world/ --level L2

# Compare results
townlet compare --baseline runs/baseline_L2_* --treatment runs/my_world_L2_*
# See: agents work more, hoard less, higher starvation rate
```

### Next Steps

- **Tutorial**: [30-minute walkthrough](docs/TUTORIAL.md)
- **Curriculum Guide**: [Understanding L0-L8](docs/curriculum.md)
- **Configuration Reference**: [Complete YAML spec](docs/configuration.md)
- **Research Examples**: [Population genetics experiments](docs/experiments.md)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIVERSE AS CODE                          │
│  bars.yaml     cascades.yaml    affordances.yaml  cues.yaml  │
│  (what exists) (how coupled)    (what actions)   (visible)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   VECTORIZED ENVIRONMENT                     │
│  • Reads world physics from YAML                             │
│  • Enforces human observer principle                         │
│  • Computes sparse rewards (r = energy × health)             │
│  • Emits cryptographically signed telemetry                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      BRAIN AS CODE                           │
│  Layer 1: Cognitive Topology (behavior rules)                │
│  Layer 2: Agent Architecture (neural blueprints)             │
│  Layer 3: Execution Graph (reasoning pipeline)               │
│                                                              │
│  Modules: Perception → World Model → Social Model → Policy  │
│  Controllers: Panic (survival override) → Ethics (veto)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       PROVENANCE                             │
│  • Cognitive hash (brain fingerprint)                        │
│  • Snapshot immutability (config frozen at launch)           │
│  • Signed checkpoints (HMAC tamper detection)                │
│  • Glass-box telemetry (full reasoning trace)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Curriculum Levels at a Glance

| Level | Challenge | Skills Learned | Architecture | Status |
|-------|-----------|---------------|--------------|--------|
| **L0-1** | Survival basics | Affordance semantics, economic loops | SimpleQNetwork | ✅ Implemented |
| **L2-3** | Multi-resource balance | Cascade management, optimization | SimpleQNetwork / LSTM | ✅ Implemented |
| **L4-5** | Fog of war | Exploration, spatial memory, temporal planning | LSTM required | 🚧 Planned |
| **L6-7** | Competition | Social reasoning, strategic resource allocation | Module A-D | 🚧 Planned |
| **L8** | Coordination | Emergent communication, family protocols | Module A-D + channel | 🚧 Planned |

**Pedagogical principle**: Each level removes one assumption

- L0-3: Full observability (scaffolding) [IMPLEMENTED]
- L4-5: Partial observability (must explore and remember) [PLANNED]
- L6-7: Social observability (cues only, no telepathy) [PLANNED]
- L8: Communication (signals without semantics) [PLANNED]

---

## Example: Emergent Communication (L8) [PLANNED]

**Setup**: Parents and child can broadcast integer signals [0, 999]

**No semantic bootstrapping**:

```python
# ❌ We DON'T provide meanings
signal_meanings = {
    123: "job_taken",
    456: "danger",
}

# ✅ We DO provide raw channel
family_comm_channel = [0.123, 0.456, 0.0]  # what family members broadcast
# Agents must learn correlations via CTDE
```

**Learning process**:

1. Parent at Job broadcasts signal `123` (exploration)
2. Child observes: parent at Job correlates with signal `123`
3. Module C learns via supervised learning: `signal=123 → parent_at_job`
4. Policy learns: "When I see `123`, don't go to Job (it's occupied)"
5. Coordination emerges: family avoids resource conflicts

**Research result**: Do stable protocols emerge by episode 20k?

---

## Documentation

### Core Docs

- **[Quick Start](docs/QUICKSTART.md)** — 10-minute tutorial
- **[Tutorial](docs/TUTORIAL.md)** — 30-minute walkthrough
- **[Configuration Reference](docs/configuration.md)** — Complete YAML spec
- **[Curriculum Guide](docs/curriculum.md)** — L0-L8 progression
- **[API Reference](docs/API_REFERENCE.md)** — Python API

### Design Docs

- **[Master Review](docs/master_review.md)** — Complete technical specification (this document)
- **[Design Principles](docs/design_principles.md)** — Human observer, sparse rewards, config-driven
- **[Reward Architecture](docs/reward_architecture.md)** — Why `r = energy × health` works
- **[Observation Space](docs/observation_space.md)** — Tensor specifications by level
- **[The Cues System](docs/cues_system.md)** — Social observability implementation

### Research Guides

- **[Population Genetics](docs/population_genetics.md)** — Family dynamics, inheritance modes
- **[Multi-Agent Mechanics](docs/multi_agent_mechanics.md)** — Contention resolution, social reasoning
- **[Emergent Communication](docs/emergent_communication.md)** — L8 protocols, analysis methods
- **[The Bigger Vision](docs/bigger_vision.md)** — Beyond towns (economy, ecosystems, etc.)

### Governance Docs

- **[Provenance System](docs/provenance.md)** — Cognitive hashing, checkpoint signing
- **[Ethics Architecture](docs/ethics.md)** — Deterministic compliance enforcement
- **[Audit Guide](docs/audit.md)** — How to verify "which brain did what"

---

## Performance

**Performance metrics TBD** - See [Section 11: Implementation Roadmap](docs/implementation_roadmap.md) for development timeline and planned benchmarks.

Current implementation (L0-L3) demonstrates:
- Successful temporal credit assignment learning
- Multi-resource management strategies
- Partial observability navigation with LSTM memory

Planned benchmarks for L4-L8:
- Multi-zone navigation efficiency
- Multi-agent coordination emergence
- Communication protocol stability

**Scalability**: Vectorized environments support 100+ parallel agents (current implementation)

---

## Validation & Testing

```bash
# Run all tests
pytest tests/

# Validate config
townlet validate --config configs/my_world/

# Check provenance (planned feature)
townlet verify --checkpoint runs/my_run/checkpoints/step_1000/
# Output: ✓ Signature valid, brain hash: 4f9a7c21ab...

# Compare experiments
townlet compare --baseline baseline_run --treatment treatment_run
# Output: Config diff, performance metrics, statistical significance
```

**Test coverage**: 80%+ (target)

---

## Research Using Townlet

**Papers** (to be added as they're published):

- *Coming soon* — Submit yours!

**Experiments** (proposed):

- Meritocratic selection in multi-agent RL populations
- Emergent communication protocols in families
- Wealth concentration in dynasty mode
- Curriculum learning for long-horizon tasks

**Open questions**:

- Can we decode emergent protocols via causal intervention?
- Do different dynasties evolve distinct strategies?
- What's the relationship between personality DNA and communication style?
- Can RL discover optimal monetary policy in economic simulations?

---

## Community & Contributing

**Discussions**: [GitHub Discussions](https://github.com/tachyon-beep/townlet/discussions)

**Issues**: [Bug reports & feature requests](https://github.com/tachyon-beep/townlet/issues)

**Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

- We welcome configs (new worlds!)
- We welcome experiments (research results!)
- We welcome docs (tutorials, examples!)
- We welcome code (features, bug fixes!)

**Code of Conduct**: Be respectful, collaborative, and curious

---

## Roadmap

### ✅ v0.1 (Current)

- L0-L3 curriculum (survival basics, full/partial observability)
- Single-agent training
- Configuration system (UNIVERSE_AS_CODE)
- Basic documentation

### 🚧 v0.2 (Next 3-6 months)

- L4-L5 implementation (multi-zone navigation)
- Schema validation tooling (UNIVERSE_AS_CODE contracts)
- Improved checkpoint system
- Video tutorials

### 🔮 v1.0 (6-12 months)

- L6-L8 implementation (multi-agent, communication)
- Population genetics system
- Provenance system (cognitive hashing, signed checkpoints)
- Web-based visualization dashboard

### 🔮 v2.0 (12-24 months)

- Alternative applications (economy, ecosystems)
- Continuous action spaces
- Relational networks (beyond grids)
- Benchmark suite (standardized tasks)

---

## Citation

If you use Townlet in your research, please cite:

```bibtex
@software{townlet2025,
  title = {Townlet: Configuration-Driven Multi-Agent RL for Long-Horizon Planning and Social Reasoning},
  author = {[TBD]},
  year = {2025},
  url = {https://github.com/tachyon-beep/townlet},
  version = {0.1.0}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

Built with:

- PyTorch (deep learning)
- Gymnasium (RL interface)
- YAML (configuration)
- pytest (testing)

Inspired by:

- Neural MMO (multi-agent survival)
- Melting Pot (social dilemmas)
- AlphaStar (emergent coordination)
- Open-Endedness literature (evolutionary RL)

Special thanks to the RL research community for feedback and ideas.

---

## Contact

- **GitHub**: [TBD]
- **Email**: [TBD]
- **Issues**: [Report bugs or request features](https://github.com/tachyon-beep/townlet/issues)

---

**Ready to build worlds?**

```bash
pip install townlet
townlet create --name my_first_world
townlet train --config configs/my_first_world/ --level L0
```

**Welcome to the Townlet Framework.** 🏘️

---
