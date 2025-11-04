# HAMLET Documentation

This directory contains comprehensive documentation for the HAMLET Deep Reinforcement Learning environment, organized by purpose and audience.

## Overview

HAMLET follows a **research-driven development methodology** with strong emphasis on:

- **Reproducibility** - UNIVERSE_AS_CODE and BRAIN_AS_CODE principles
- **Pedagogy** - Capturing "interesting failures" as teaching moments
- **Design exploration** - Research before implementation for complex decisions
- **Operational excellence** - Comprehensive guides for training and visualization

**Total Documentation**: 85+ markdown files across 9 categories

---

## Directory Structure

```
docs/
├── README.md                      # This file
├── TASK-*.md                      # Formal task specifications (root level)
├── architecture/                  # System design & strategic docs
│   └── archive/                   # Historical architecture
├── manual/                        # User guides & how-to docs
├── methods/                       # Development workflows
├── plans/                         # Implementation plans (dated)
│   └── archive/                   # Completed plans
├── research/                      # Design exploration & analysis
│   └── archive/                   # Completed research
├── reviews/                       # Code review summaries
├── tasks/                         # Quick task definitions
└── teachable_moments/             # Pedagogical insights
```

---

## Quick Navigation

### By Role

**👨‍💻 For Developers**

- Start: [`architecture/TOWNLET_HLD.md`](architecture/TOWNLET_HLD.md) - System design
- Roadmap: [`architecture/ROADMAP.md`](architecture/ROADMAP.md) - Development phases
- Philosophy: [`architecture/UNIVERSE_AS_CODE.md`](architecture/UNIVERSE_AS_CODE.md) - Configuration principles

**🔬 For Researchers**

- Current work: [`research/`](research/) - Active design exploration
- Open problems: [`research/RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md`](research/RESEARCH-TASK-000-UNSOLVED-PROBLEMS-CONSOLIDATED.md)
- Methodology: [`methods/RESEARCH-PLAN-REVIEW-LOOP.md`](methods/RESEARCH-PLAN-REVIEW-LOOP.md)

**👤 For Users/Operators**

- Training: [`manual/TRAINING_SYSTEM.md`](manual/TRAINING_SYSTEM.md) - How to train agents
- Visualization: [`manual/UNIFIED_SERVER_USAGE.md`](manual/UNIFIED_SERVER_USAGE.md) - Live inference
- Recording: [`manual/REPLAY_USAGE.md`](manual/REPLAY_USAGE.md) - Episode replay

**🎓 For Instructors/Students**

- Philosophy: [`teachable_moments/trick_students_pedagogy.md`](teachable_moments/trick_students_pedagogy.md)
- Examples: [`teachable_moments/`](teachable_moments/) - Reward hacking, emergent behavior
- Curriculum: [`architecture/TRAINING_LEVELS.md`](architecture/TRAINING_LEVELS.md) - L0→L3 progression

---

## Documentation Categories

### 📋 Task Specifications (Root Level)

**Location**: `docs/TASK-*.md`

Formal specification documents following UNIVERSE_AS_CODE philosophy:

- **TASK-000**: Configurable spatial substrates
- **TASK-001**: Variable-size meter system
- **TASK-002**: UAC contracts & schema validation
  - **TASK-002A**: Core data transfer objects
  - **TASK-002B**: System capabilities
- **TASK-003**: Action space configuration
- **TASK-004**: Universe compiler implementation
- **TASK-005**: BRAIN_AS_CODE (agent architecture as config)
- **TASK-006**: Substrate-agnostic visualization

**Purpose**: Define formal requirements for making the universe fully configurable

**Audience**: Developers implementing UNIVERSE_AS_CODE features

---

### 🏗️ Architecture (`architecture/`)

**7 active documents + 5 archived**

Strategic system design and high-level documentation:

**Key Documents**:

- [`TOWNLET_HLD.md`](architecture/TOWNLET_HLD.md) - High-level design (v2.5, current system)
- [`ROADMAP.md`](architecture/ROADMAP.md) - Development roadmap (Phase 3 complete)
- [`TRAINING_LEVELS.md`](architecture/TRAINING_LEVELS.md) - Progressive complexity (L0→L3)
- [`UNIVERSE_AS_CODE.md`](architecture/UNIVERSE_AS_CODE.md) - Configuration philosophy
- [`BRAIN_AS_CODE.md`](architecture/BRAIN_AS_CODE.md) - Agent mind as config (future)

**Purpose**: Strategic architecture, design principles, system philosophy

**When to use**: Understanding overall system design, making architectural decisions

---

### 📖 Manual (`manual/`)

**7 operational guides**

User-facing documentation for running experiments and visualizing results:

- [`TRAINING_SYSTEM.md`](manual/TRAINING_SYSTEM.md) - Training infrastructure overview
- [`CONFIGURING_A_RUN.md`](manual/CONFIGURING_A_RUN.md) - How to configure training
- [`UNIFIED_SERVER_USAGE.md`](manual/UNIFIED_SERVER_USAGE.md) - Live inference server
- [`REPLAY_USAGE.md`](manual/REPLAY_USAGE.md) - Episode replay system
- [`VIDEO_EXPORT_USAGE.md`](manual/VIDEO_EXPORT_USAGE.md) - Export training videos
- [`RECORDING_SYSTEM_SUMMARY.md`](manual/RECORDING_SYSTEM_SUMMARY.md) - Recording overview
- [`TENSORBOARD_INTEGRATION.md`](manual/TENSORBOARD_INTEGRATION.md) - Metrics visualization

**Purpose**: Operational "how-to" guides for users

**Audience**: Anyone running experiments, training agents, or visualizing behavior

---

### 🔬 Research (`research/`)

**9 active documents + 5 archived**

Deep design exploration before implementation:

**Active Research**:

- Spatial substrates design space
- Code generation vs soft lookups
- Distance semantics for non-euclidean spaces
- Observation encoding strategies
- Substrate-agnostic visualization
- Universe compiler design
- Unsolved problems consolidated

**Purpose**: Explore design space, identify tradeoffs, document open problems

**Philosophy**: Research deeply when multiple approaches exist with unclear tradeoffs

**Methodology**: See [`methods/RESEARCH-PLAN-REVIEW-LOOP.md`](methods/RESEARCH-PLAN-REVIEW-LOOP.md)

**When to use**: Before implementing complex features with multiple viable approaches

---

### 📅 Plans (`plans/`)

**5 active plans + 11 archived**

Dated implementation plans capturing design decisions:

**Active Plans**:

- `2025-11-04-spatial-substrates.md` - Spatial substrate exploration
- `2025-11-04-uac-action-space.md` - Action space configuration
- `plan-task-001-variable-size-meters-tdd-ready.md` - TDD-ready meter plan

**Archived Plans** (Completed implementations):

- Temporal mechanics design/implementation (Level 3)
- Multi-day demo design
- Phase 0-3 implementation plans
- Episode recording system

**Purpose**: Capture implementation plans at specific points in time

**Archive policy**: Plans move to `archive/` once implemented or superseded

---

### 🔍 Reviews (`reviews/`)

**2 review documents**

Post-implementation code review summaries:

- `REVIEW-TASK-001-SUMMARY.md` - Task 001 review
- `REVIEW-TASK-001-TDD-PLAN.md` - Task 001 TDD plan review

**Purpose**: Validate implementation against goals, capture learnings

**When to create**: After major task completion for retrospective analysis

---

### 🛠️ Methods (`methods/`)

**2 methodology documents**

Development workflows and processes:

- [`RESEARCH-PLAN-REVIEW-LOOP.md`](methods/RESEARCH-PLAN-REVIEW-LOOP.md) - When to research vs TDD
- [`TEST_REFACTORING_METHODOLOGY.md`](methods/TEST_REFACTORING_METHODOLOGY.md) - Test refactoring approach

**Purpose**: Define development methodologies

**Key insight**: Research → Plan → Review for complex architectural decisions, TDD for clear requirements

---

### 🎓 Teachable Moments (`teachable_moments/`)

**13 pedagogical documents**

Capturing "interesting failures" and emergent behaviors as teaching materials:

**Key Documents**:

- [`README.md`](teachable_moments/README.md) - Index of all insights
- [`trick_students_pedagogy.md`](teachable_moments/trick_students_pedagogy.md) - Core pedagogy
- [`three_stages_of_learning.md`](teachable_moments/three_stages_of_learning.md) - Learning framework
- [`reward_hacking_interact_spam.md`](teachable_moments/reward_hacking_interact_spam.md) - Classic example
- [`interoception_reward_design.md`](teachable_moments/interoception_reward_design.md) - Reward design

**Philosophy**: "Trick students into learning graduate-level RL by making them think they're just playing The Sims"

**Purpose**: Preserve emergent discoveries as teaching moments rather than bugs to fix

**Value**: Creates pedagogical curriculum from actual system behavior

---

### ✅ Tasks (`tasks/`)

**1 quick task**

Lightweight task tracking for smaller work items:

- `QUICK-001-AFFORDANCE-DB-INTEGRATION.md`

**Purpose**: Track smaller implementation tasks that don't need full TASK-NNN specification

---

## Core Principles

### UNIVERSE_AS_CODE

**Principle**: "Everything configurable. Schema enforced mercilessly."

All game mechanics (meters, affordances, cascades, actions, cues) defined in YAML configuration files, not hardcoded in Python.

**Benefits**:

- Domain-agnostic (could model villages, factories, trading bots)
- Experimentable (test new mechanics without code changes)
- Pedagogical (students learn by editing configs)

**Implementation**: See `TASK-002` (contracts), `TASK-004` (compiler)

### BRAIN_AS_CODE (Future)

**Principle**: Agent architecture also becomes configuration

Network architecture, policy, exploration strategy, training schedule all defined in YAML.

**Benefits**:

- Experiment with architectures without code changes
- A/B test exploration strategies
- Reproduce exact agent configurations
- Domain-specific architectures from configs

**Status**: Specification in progress (see `TASK-005`)

### No-Defaults Principle

**Principle**: All behavioral parameters must be explicitly specified in config files

**Rationale**: Hidden defaults create non-reproducible configs, operators don't know what values are used, changing code defaults silently breaks old configs

**Exemptions**: Metadata, visualization hints, computed values

**Enforcement**: Pydantic DTOs require all fields, missing field → compilation error

### Research-Driven Development

**When to research**: Multiple viable approaches with unclear tradeoffs

**Process**: Research → Plan → Review (see `methods/RESEARCH-PLAN-REVIEW-LOOP.md`)

**When to TDD**: Clear requirements, single obvious approach

**Value**: Avoid premature optimization, explore design space thoroughly

---

## Naming Conventions

### Task Specifications

- **Format**: `TASK-NNN-DESCRIPTIVE-NAME.md`
- **Example**: `TASK-002-UAC-CONTRACTS.md`
- **Location**: Root `docs/` directory
- **Purpose**: Formal requirements

### Research Documents

- **Format**: `RESEARCH-TOPIC-NAME.md`
- **Example**: `RESEARCH-SPATIAL-SUBSTRATES.md`
- **Location**: `research/` directory
- **Purpose**: Design exploration

### Implementation Plans

- **Format**: `YYYY-MM-DD-descriptive-name.md`
- **Example**: `2025-11-04-spatial-substrates.md`
- **Location**: `plans/` directory
- **Purpose**: Dated implementation roadmap

### Review Documents

- **Format**: `REVIEW-TASK-NNN-TYPE.md`
- **Example**: `REVIEW-TASK-001-SUMMARY.md`
- **Location**: `reviews/` directory
- **Purpose**: Post-implementation analysis

---

## Archive Policy

Documents move to `archive/` subdirectories when:

1. **Plans**: Implementation complete or plan superseded
2. **Research**: Problem solved or research concluded
3. **Architecture**: Design superseded by newer version

**Rationale**: Preserve history while keeping active directories focused

**Current archives**: 23 files (27% of total documentation)

---

## Contributing Documentation

### Adding Research

1. Create `research/RESEARCH-YOUR-TOPIC.md`
2. Explore design space, identify tradeoffs
3. Document open problems
4. Link from relevant TASK or plan
5. Move to `archive/` when concluded

### Adding Plans

1. Create `plans/YYYY-MM-DD-your-feature.md`
2. Include: Goals, approach, milestones, acceptance criteria
3. Link to relevant research and tasks
4. Move to `archive/` when implemented

### Adding Teachable Moments

1. Create `teachable_moments/your_discovery.md`
2. Document: What happened, why it's interesting, pedagogical value
3. Add entry to `teachable_moments/README.md`
4. Link from `teachable_moments/DISCOVERED_INSIGHTS.md`

### Adding User Guides

1. Create `manual/YOUR-GUIDE.md`
2. Focus on: How to use feature, examples, troubleshooting
3. Update main `README.md` to link to guide
4. Test all commands in guide work

---

## Cross-References

- **Main README**: [`../README.md`](../README.md) - Project overview
- **Claude Guide**: [`../CLAUDE.md`](../CLAUDE.md) - AI assistant guidance
- **Config Examples**: [`../configs/`](../configs/) - YAML configurations

---

## Documentation Statistics

- **Total Files**: 85+ markdown documents
- **Active Work**: 62 files (73%)
- **Archived History**: 23 files (27%)
- **Categories**: 9 (architecture, manual, methods, plans, research, reviews, tasks, teachable_moments, root)
- **Maturity Level**: High (comprehensive coverage across planning, implementation, operation, pedagogy)

---

## Getting Started

**New to HAMLET?**

1. Read [`../README.md`](../README.md) - Quick start
2. Read [`architecture/TOWNLET_HLD.md`](architecture/TOWNLET_HLD.md) - System overview
3. Read [`manual/TRAINING_SYSTEM.md`](manual/TRAINING_SYSTEM.md) - Run your first experiment

**Adding a feature?**

1. Check [`research/`](research/) for related research
2. Check [`TASK-*.md`](.) for formal specifications
3. Create implementation plan in [`plans/`](plans/)
4. Follow methodology in [`methods/RESEARCH-PLAN-REVIEW-LOOP.md`](methods/RESEARCH-PLAN-REVIEW-LOOP.md)

**Teaching with HAMLET?**

1. Start with [`teachable_moments/trick_students_pedagogy.md`](teachable_moments/trick_students_pedagogy.md)
2. Review [`teachable_moments/DISCOVERED_INSIGHTS.md`](teachable_moments/DISCOVERED_INSIGHTS.md)
3. Use [`architecture/TRAINING_LEVELS.md`](architecture/TRAINING_LEVELS.md) for curriculum

---

*This documentation structure reflects a mature research-driven project with emphasis on reproducibility, pedagogy, and design exploration.*
