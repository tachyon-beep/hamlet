# Townlet Architecture Analysis - Coordination Plan

**Analysis Date:** 2025-11-12
**Archaeologist:** Claude Code (System Archaeologist Skill)
**Target:** HAMLET/Townlet Deep RL Training System

---

## Scope Definition

### In Scope
- **Primary target:** `src/townlet/` (entire production codebase)
- **Configuration system:** All YAML schemas, UAC compiler pipeline
- **Reference config:** `configs/L1_full_observability/` (baseline full-observability curriculum level)
- **Documentation:** Subsystem catalogs, C4 diagrams, architecture reports
- **Focus:** Internal data flows, subsystem boundaries, component interfaces

### Explicitly Out of Scope
- Test suite (`tests/`)
- Legacy code (all `src/hamlet/` removed from codebase)
- Frontend integration (`frontend/`)
- External system integration (WebSocket server, TensorBoard)

---

## Complexity Assessment

**Initial Survey Results:**

```
Primary subsystems identified: 13
├── agent/          - Neural network architectures (SimpleQNetwork, RecurrentSpatialQNetwork)
├── compiler/       - UAC CLI interface
├── config/         - Configuration DTOs (bars, affordances, training, etc.)
├── curriculum/     - Curriculum strategies (adversarial, static)
├── demo/           - Live inference and demo runner
├── environment/    - Vectorized environment (GPU-native)
├── exploration/    - Exploration strategies (RND, epsilon-greedy, adaptive intrinsic)
├── population/     - Population management (vectorized training)
├── recording/      - Episode recording and replay
├── substrate/      - Spatial substrates (Grid2D, Grid3D, GridND, Continuous, Aspatial)
├── training/       - Training loop, replay buffer, checkpointing
├── universe/       - Universe compiler implementation (7-stage pipeline)
└── vfs/            - Variable & Feature System (schema, registry, observation builder)
```

**Estimated LOC:** ~15,000+ (production code only)
**Coupling:** Moderate - subsystems have clear boundaries but coordinated data flows
**Complexity:** High - GPU-native vectorized training, multi-stage compiler, configurable substrates

---

## Orchestration Strategy: PARALLEL

### Decision Rationale
- ✅ **13 independent subsystems** - Each has distinct responsibilities
- ✅ **Large codebase** - 15K+ LOC across 13 domains
- ✅ **Loosely coupled** - Clean subsystem boundaries (agent, training, environment, etc.)
- ✅ **Time efficiency** - Parallel analysis reduces wall-clock time from ~4 hours → ~1 hour
- ✅ **Stakeholder delivery** - Comprehensive technical deep-dive for research stakeholders

### Parallel Execution Plan
1. **Phase 1:** Holistic assessment (sequential, ~30 min)
   - Directory structure mapping
   - Technology stack identification
   - Entry point discovery
   - Initial subsystem boundary identification

2. **Phase 2:** Subsystem deep-dive (parallel, ~45 min)
   - Spawn 13 parallel subagents (one per subsystem)
   - Each produces subsystem catalog entry
   - Validation gate after completion

3. **Phase 3:** Diagram generation (sequential, ~30 min)
   - C4 Context diagram (system boundary)
   - C4 Container diagram (major subsystems)
   - C4 Component diagrams (internal structure of key subsystems)
   - Data flow diagrams (config → UAC → runtime pipeline)
   - Validation gate after completion

4. **Phase 4:** Synthesis (sequential, ~30 min)
   - Comprehensive architecture report
   - Cross-subsystem analysis
   - Design pattern documentation
   - Final validation gate

**Total estimated time:** 2.5-3 hours

---

## Quality Gates

All deliverables must pass validation before proceeding:

1. **Discovery Findings** → Contract compliance check
2. **Subsystem Catalog** → Bidirectional dependency validation, contract compliance
3. **C4 Diagrams** → Consistency with catalog, completeness check
4. **Final Report** → Comprehensive stakeholder readiness validation

**Validation approach:** Dedicated validation subagent (preferred for this scale)

---

## Execution Log

| Timestamp | Action | Status |
|-----------|--------|--------|
| 2025-11-12 Initial | Created workspace `archeologist_archive/` | ✅ Complete |
| 2025-11-12 Initial | Wrote coordination plan | ✅ Complete |
| [Next] | Perform holistic assessment → `01-discovery-findings.md` | Pending |
| [Next] | Spawn 13 parallel subagents for subsystem analysis | Pending |
| [Next] | Validate subsystem catalog | Pending |
| [Next] | Generate C4 diagrams | Pending |
| [Next] | Validate diagrams | Pending |
| [Next] | Synthesize final report | Pending |
| [Next] | Final validation | Pending |

---

## Stakeholder Deliverables

**Primary audience:** Technical and research stakeholders
**Expected outputs:**

1. **01-discovery-findings.md** - Holistic system overview
2. **02-subsystem-catalog.md** - Deep-dive component inventory with interfaces
3. **03-diagrams.md** - C4 architecture diagrams (Context, Container, Component)
4. **04-final-report.md** - Comprehensive architecture documentation
5. **temp/** - Working artifacts (subagent task specs, validation reports)

---

## Notes

- No time constraints - full systematic analysis
- User has full freedom for artifact selection
- Focus on internal architecture, not external integration
- Reference config pack: L1_full_observability (8×8 grid, 14 affordances, full observability baseline)
