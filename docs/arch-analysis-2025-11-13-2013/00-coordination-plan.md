# Architecture Analysis Coordination Plan

**Project:** HAMLET Deep Reinforcement Learning Environment
**Analysis Date:** 2025-11-13
**Workspace:** `docs/arch-analysis-2025-11-13-2013/`
**Analyst:** Claude (axiom-system-archaeologist workflow)

---

## Scope

### Target System
**HAMLET (Townlet)** - A pedagogical Deep Reinforcement Learning environment where agents learn to survive by managing multiple competing needs (energy, hygiene, satiation, money, health, fitness, mood, social).

### Analysis Objectives
1. Document the complete Townlet architecture (GPU-native vectorized training system)
2. Identify all major subsystems and their interactions
3. Create C4 architecture diagrams (Context, Container, Component levels)
4. Synthesize findings into a comprehensive architecture report
5. Highlight architectural patterns, concerns, and design rationale

### In-Scope
- **Active codebase:** `src/townlet/` (current production system)
- **Configuration system:** `configs/` directory with curriculum levels
- **Frontend visualization:** `frontend/src/components/`
- **Testing infrastructure:** `tests/test_townlet/`
- **Documentation:** `docs/` (as reference for understanding intent)

### Out-of-Scope
- **Legacy code:** `src/hamlet/` (marked obsolete per CLAUDE.md)
- **Historical implementations:** Focus on current Townlet system only
- **Deep algorithm analysis:** Document architecture, not mathematical derivations
- **Performance profiling:** Focus on structural design, not runtime metrics

---

## Strategy

### Analysis Approach
**Parallel Subagent Analysis** - The codebase has ~10 clearly independent subsystems that can be analyzed concurrently:
- Environment (vectorized execution)
- Agent (neural network architectures)
- Population (batched training)
- Curriculum (adversarial & static strategies)
- Exploration (RND, ICM, adaptive)
- Training (replay buffer, state management)
- Universe Compiler (UAC seven-stage pipeline)
- VFS (Variable & Feature System)
- Drive As Code (DAC reward engine)
- Frontend (visualization system)

**Rationale:** With ≥10 subsystems and clear module boundaries, parallel analysis maximizes efficiency while maintaining quality through validation gates.

### Technology Stack Context
- **Language:** Python 3.x with type hints
- **Deep Learning:** PyTorch (GPU-native tensors)
- **RL Algorithm:** DQN / Double DQN with experience replay
- **Configuration:** YAML-based declarative system
- **Frontend:** Vue 3 + WebSocket (real-time visualization)
- **Testing:** pytest with fixtures

### Key Architectural Themes to Investigate
1. **GPU-Native Design:** How vectorization enables parallel agent training
2. **Declarative Configuration:** UAC compiler pipeline, VFS, DAC reward system
3. **Pedagogical Mission:** "Trick students into learning grad-level RL"
4. **Curriculum Progression:** L0 → L3 complexity escalation
5. **No-Defaults Principle:** All behavioral params must be explicit
6. **Pre-Release Freedom:** Zero backwards compatibility constraints

---

## Execution Plan

### Phase 1: Workspace Setup ✅
- [x] Create workspace directory structure
- [x] Write coordination plan (this document)

### Phase 2: Holistic Assessment
**Deliverable:** `01-initial-assessment.md`

**Tasks:**
1. Map directory structure (`src/townlet/`, `configs/`, `frontend/`, `tests/`, `docs/`)
2. Identify entry points (`scripts/run_demo.py`, `townlet.demo.live_inference`, frontend dev server)
3. Document technology stack (PyTorch, Vue 3, YAML, pytest)
4. List candidate subsystems (10-12 expected)
5. Note architectural peculiarities (GPU-native, declarative configs, "interesting failures")

**Method:** Top-down reconnaissance without deep diving into implementation

### Phase 3: Detailed Subsystem Analysis
**Deliverable:** `02-subsystem-catalog.md`

**Orchestration Strategy:** **Parallel Analysis** (10 concurrent subagents)

**Subsystem Assignments:**
1. **Environment Subsystem** - `src/townlet/environment/vectorized_env.py`, `dac_engine.py`
2. **Agent Subsystem** - `src/townlet/agent/networks.py`
3. **Population Subsystem** - `src/townlet/population/vectorized.py`
4. **Curriculum Subsystem** - `src/townlet/curriculum/{adversarial,static}.py`
5. **Exploration Subsystem** - `src/townlet/exploration/` (RND, ICM, adaptive)
6. **Training Subsystem** - `src/townlet/training/{state,replay_buffer}.py`
7. **Universe Compiler (UAC)** - `src/townlet/universe/compiler.py`
8. **VFS (Variables)** - `src/townlet/vfs/{schema,registry,observation_builder}.py`
9. **DAC (Rewards)** - DACEngine integration (drive_as_code.yaml → runtime execution)
10. **Frontend Visualization** - `frontend/src/components/{Grid,AspatialView}.vue`

**Contract:** Each analysis produces EXACTLY 8 sections per subsystem:
1. Subsystem Name (H2)
2. Location
3. Responsibility (single sentence)
4. Key Components (bulleted)
5. Dependencies (Inbound/Outbound format)
6. Patterns Observed
7. Concerns (or "None observed")
8. Confidence level with reasoning
9. Separator `---`

**Validation Gate:** After all subsystem analyses complete, validate catalog for contract compliance before proceeding.

### Phase 4: Architecture Diagrams
**Deliverable:** `03-diagrams.md`

**Diagram Levels:**
1. **Context (C4 Level 1):** HAMLET system with external actors (researchers, students, operators)
2. **Container (C4 Level 2):** All 10 subsystems with dependency arrows
3. **Component (C4 Level 3):** Internal structure of 2-3 representative subsystems
   - Candidates: UAC (complex pipeline), Environment (GPU vectorization), DAC (YAML → runtime)

**Abstraction Strategy:** If Container diagram exceeds 8 subsystems, group by architectural layer:
- **Execution Layer:** Environment, Agent, Population
- **Intelligence Layer:** Exploration, Training, Curriculum
- **Configuration Layer:** UAC, VFS, DAC
- **Presentation Layer:** Frontend

**Validation Gate:** Validate diagrams for C4 compliance, cross-reference with catalog dependencies.

### Phase 5: Final Report Synthesis
**Deliverable:** `04-final-report.md`

**Report Structure:**
- Front matter (metadata)
- Table of contents (20+ entries expected)
- Executive summary (2-3 paragraphs for non-technical stakeholders)
- System overview (pedagogical mission, technical foundation)
- Architecture diagrams (embedded from Phase 4)
- Subsystem details (synthesized from catalog)
- Key findings (architectural patterns, concerns, recommendations)
- Appendices (methodology, assumptions, confidence assessment)

**Synthesis Focus:**
- Pattern identification across subsystems (declarative configs, GPU-native design)
- Concern extraction and prioritization
- Cross-referencing (40+ internal links expected)
- Multi-audience entry points

**Validation Gate:** Final validation for completeness, consistency, professional quality.

---

## Quality Gates

### Validation Points
1. **After Catalog:** Contract compliance (8 sections per entry, no extras)
2. **After Diagrams:** C4 compliance + catalog dependency cross-check
3. **After Report:** Comprehensive consistency validation across all artifacts

### Success Criteria
- All subsystems documented with HIGH or MEDIUM confidence
- Diagrams accurately reflect catalog dependencies
- Report synthesizes insights (not just concatenates sources)
- Zero CRITICAL validation violations
- All artifacts written to workspace directory

### Failure Handling
- **If validation fails:** Fix issues and re-validate (max 2 retries per phase)
- **If ambiguity blocks analysis:** Document assumption with MEDIUM/LOW confidence
- **If time pressure emerges:** Reduce scope with explicit documentation, don't skip validation

---

## Risk Assessment

### Known Challenges
1. **Large codebase:** ~10K+ lines across townlet/ - parallel analysis mitigates this
2. **Multiple domains:** DRL algorithms + GPU programming + web frontend - require broad expertise
3. **Rapid evolution:** Pre-release system with frequent breaking changes - focus on current state
4. **Declarative complexity:** UAC/VFS/DAC compile YAML → runtime behavior - trace compilation paths

### Mitigation Strategies
- **Parallel analysis:** Reduces wall-clock time, maintains thoroughness
- **Validation gates:** Catch inconsistencies early before downstream work
- **Confidence levels:** Explicitly mark uncertain areas rather than guess
- **CLAUDE.md reference:** Understand project philosophy to avoid misinterpreting "interesting failures"

---

## Success Metrics

### Deliverables
- [x] `00-coordination-plan.md` (this document)
- [ ] `01-initial-assessment.md`
- [ ] `02-subsystem-catalog.md` (10 entries expected)
- [ ] `03-diagrams.md` (1 Context + 1 Container + 2-3 Component diagrams)
- [ ] `04-final-report.md` (comprehensive synthesis)
- [ ] `temp/validation-*.md` (validation reports for each phase)

### Quality Indicators
- **Coverage:** All active subsystems documented (src/townlet/, configs/, frontend/)
- **Accuracy:** Dependencies match actual import statements and data flow
- **Insight:** Patterns and concerns identified beyond surface-level description
- **Usability:** Report accessible to multiple audiences (execs, architects, engineers)
- **Reproducibility:** Methodology documented, assumptions explicit

---

## Timeline Estimate

**Phase 2 (Assessment):** ~10 minutes (reconnaissance)
**Phase 3 (Catalog):** ~40 minutes (10 subsystems × 4 min parallel)
**Phase 3 (Validation):** ~5 minutes
**Phase 4 (Diagrams):** ~20 minutes
**Phase 4 (Validation):** ~5 minutes
**Phase 5 (Report):** ~20 minutes
**Phase 5 (Validation):** ~5 minutes

**Total Estimated Duration:** ~105 minutes (~1.75 hours)

**Note:** This is wall-clock time with parallel analysis. Sequential analysis would take ~3 hours.

---

## Coordination Log

This section tracks execution decisions and deviations from plan:

**2025-11-13 20:13:** Workspace created, coordination plan written. Proceeding to Phase 2 (holistic assessment).

---

_Next step: Create `01-initial-assessment.md` with holistic codebase reconnaissance._
