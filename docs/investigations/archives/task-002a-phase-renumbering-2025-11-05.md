# TASK-002A: Phase Renumbering - 2025-11-05

## Critical Issue: Phase Ordering Dependency

**Problem**: Original numbering had Phase 3 (Environment Integration) requiring substrate.yaml files that Phase 6 (Config Migration) creates - a dependency inversion that would break all config packs.

**Solution**: Renumbered phases to execute in correct dependency order.

---

## Phase Renumbering Map

| New # | New Name | Old # | Old Name | File Name | Change |
|-------|----------|-------|----------|-----------|---------|
| **Phase 1** | Substrate Abstractions | Phase 1 | *(same)* | `phase-1to2-foundation.md` | ✅ No change |
| **Phase 2** | Config Schema | Phase 2 | *(same)* | `phase-1to2-foundation.md` | ✅ No change |
| **Phase 3** | Config Migration | **Phase 6** | Config Migration | `phase3-config-migration.md` | 🔄 **MOVED UP** |
| **Phase 4** | Environment Integration | **Phase 3** | Environment Integration | `phase4-environment-integration.md` | 🔄 **MOVED DOWN** |
| **Phase 5** | Position Management | **Phase 4** | Position Management | `phase5-position-management.md` | 🔄 Renumbered |
| **Phase 6** | Observation Builder | **Phase 5** | Observation Builder | `phase6-observation-builder.md` | 🔄 Renumbered |
| **Phase 7** | Frontend Visualization | Phase 7 | *(same)* | `phase7-frontend-visualization.md` | ✅ No change |
| **Phase 8** | Testing & Verification | Phase 8 | *(same)* | `phase8-testing-verification.md` | ✅ No change |

---

## Dependency Chain (Correct Order)

```
Phase 1-2: Substrate Foundation
  ↓ (creates substrate classes and schema)

Phase 3: Config Migration
  ↓ (creates substrate.yaml for all 7 config packs)

Phase 4: Environment Integration
  ↓ (enforces substrate.yaml requirement - now files exist!)

Phase 5: Position Management
  ↓ (uses substrate.initialize_positions(), apply_movement())

Phase 6: Observation Builder
  ↓ (uses substrate.encode_observation())

Phase 7: Frontend Visualization
  ↓ (renders substrate-aware UI)

Phase 8: Testing & Verification
  ↓ (validates entire substrate abstraction)
```

**Key Insight**: Phase 3 (Config Migration) MUST complete before Phase 4 (Environment Integration) or training will fail with "substrate.yaml not found" errors.

---

## File Changes

### Renamed Files

```bash
# Before
docs/plans/task-002a-phase-1to3-configurable-spatial-substrates.md
docs/plans/task-002a-phase4-position-management.md
docs/plans/task-002a-phase5-observation-builder.md
docs/plans/task-002a-phase6-config-migration.md
docs/plans/task-002a-phase7-frontend-visualization.md
docs/plans/task-002a-phase8-testing-verification.md

# After
docs/plans/task-002a-phase-1to2-foundation.md           # Split from 1-3
docs/plans/task-002a-phase3-config-migration.md         # Was phase6
docs/plans/task-002a-phase4-environment-integration.md  # Was phase3 (split from 1-3)
docs/plans/task-002a-phase5-position-management.md      # Was phase4
docs/plans/task-002a-phase6-observation-builder.md      # Was phase5
docs/plans/task-002a-phase7-frontend-visualization.md   # Unchanged
docs/plans/task-002a-phase8-testing-verification.md     # Unchanged
```

### File Operations Performed

```bash
# 1. Rename Phase 6 → Phase 3 (config migration)
mv task-002a-phase6-config-migration.md task-002a-phase3-config-migration.md

# 2. Rename Phase 5 → Phase 6 (observation builder)
mv task-002a-phase5-observation-builder.md task-002a-phase6-observation-builder.md

# 3. Rename Phase 4 → Phase 5 (position management)
mv task-002a-phase4-position-management.md task-002a-phase5-position-management.md

# 4. Split Phase 1-3 file into foundation (1-2) and environment integration (4)
head -n 1309 task-002a-phase-1to3-configurable-spatial-substrates.md > task-002a-phase-1to2-foundation.md
tail -n +1310 task-002a-phase-1to3-configurable-spatial-substrates.md > task-002a-phase4-environment-integration.md
rm task-002a-phase-1to3-configurable-spatial-substrates.md
```

---

## Impact on Effort Estimates

**Total effort unchanged**: 73.5 hours

| Phase | Effort | Critical Path |
|-------|--------|---------------|
| Phase 1-2 | 10h | Foundation |
| **Phase 3** | **4h** | **BLOCKS Phase 4** |
| Phase 4 | 1.5h | Uses Phase 3 files |
| Phase 5 | 26h | Critical path |
| Phase 6 | 20h | Critical path |
| Phase 7 | 5h | Polish |
| Phase 8 | 7h | Verification |

**Critical path**: Phases 5-6 (Position + Observation) = 46 hours

---

## What This Fixes

### ❌ Before (Broken)

```bash
# Implement Phase 3 (Environment Integration)
python -m townlet.demo.runner --config configs/L1_full_observability

# ERROR: FileNotFoundError: substrate.yaml is required but not found
# Training breaks immediately because Phase 6 hasn't created the files yet!
```

### ✅ After (Correct)

```bash
# Implement Phase 3 (Config Migration first)
# Creates configs/L1_full_observability/substrate.yaml

# Then implement Phase 4 (Environment Integration)
python -m townlet.demo.runner --config configs/L1_full_observability

# SUCCESS: Loads substrate.yaml, training starts correctly
```

---

## Cross-References That Need Updating

The following plan documents may reference "Phase N" and need updates:

1. ✅ `task-002a-phase-1to2-foundation.md` - No cross-refs, phases 1-2 standalone
2. ⚠️ `task-002a-phase3-config-migration.md` - May reference "Phase 3" (now Phase 4) in dependencies
3. ⚠️ `task-002a-phase4-environment-integration.md` - References to Phase 3 now wrong
4. ⚠️ `task-002a-phase5-position-management.md` - Depends on "Phase 3", now Phase 4
5. ⚠️ `task-002a-phase6-observation-builder.md` - Depends on "Phase 4", now Phase 5
6. ⚠️ `task-002a-phase7-frontend-visualization.md` - May reference Phase 5, now Phase 6
7. ⚠️ `task-002a-phase8-testing-verification.md` - Tests Phases 0-7, numbering changed

**Action Required**: Update "Dependencies" sections in each plan to reference NEW phase numbers.

---

## Research Documents (No Changes Needed)

Research documents reference concepts, not phase numbers, so no updates needed:

- ✅ `research-task-002a-phase4-position-management.md` - Concept-based
- ✅ `research-task-002a-phase5-observation-builder.md` - Concept-based
- ✅ `research-task-002a-phase6-config-migration.md` - Concept-based
- ✅ `research-task-002a-phase7-frontend.md` - Concept-based
- ✅ `research-task-002a-phase8-testing.md` - Concept-based

**Note**: Research file names still use old numbers (phase4, phase5, etc.) but this is acceptable since they describe the *content* (position management, observation builder, etc.), not the *execution order*.

---

## Review Documents

Review documents reference old phase numbers in their analysis. Treat these as historical:

- `review-task-002a-plan-round1.md` - References old numbering (historical)
- `review-task-002a-phase4-position-management.md` - "Phase 4" was position management (now Phase 5)
- `review-task-002a-phase4-round2-breaking-changes.md` - Same
- `review-task-002a-phases0-3-breaking-changes.md` - "Phase 3" was environment integration (now Phase 4)
- `review-task-002a-phase5-observation-builder.md` - "Phase 5" was observation (now Phase 6)
- `review-task-002a-phase6-config-migration.md` - "Phase 6" was config migration (now Phase 3)
- `review-task-002a-phases7-8.md` - Phases 7-8 unchanged

**Action**: Add note to each review: "NOTE: This review uses original phase numbering. See task-002a-phase-renumbering-2025-11-05.md for current numbering."

---

## Implementation Checklist

Before starting implementation:

- [x] Phase files renamed to correct order
- [x] Phase 1-3 file split into foundation (1-2) and environment (4)
- [x] Phase numbering map documented
- [ ] Update dependency sections in plan files (reference new numbers)
- [ ] Add historical note to review documents
- [ ] Update TASK-002A.md with corrected phase order
- [ ] Update final planning summary with new numbering

---

## Quick Reference: "Which Phase Am I Looking At?"

If you see a file or reference with old numbering:

| You See | It's Actually | Why Different |
|---------|---------------|---------------|
| "Phase 3" in old docs | **Phase 4** (Environment Integration) | Moved down |
| "Phase 4" in old docs | **Phase 5** (Position Management) | Bumped up |
| "Phase 5" in old docs | **Phase 6** (Observation Builder) | Bumped up |
| "Phase 6" in old docs | **Phase 3** (Config Migration) | Moved up |
| "Phase 7-8" in old docs | **Phase 7-8** (unchanged) | No change |

---

## Rationale for Reordering

**Why Phase 3 (Config Migration) must come before Phase 4 (Environment Integration)**:

1. **Phase 4 enforces requirement**: `if not substrate_config_path.exists(): raise FileNotFoundError(...)`
2. **Phase 3 creates the files**: Creates `substrate.yaml` for all 7 config packs
3. **Dependency**: Phase 4 requires artifacts from Phase 3

**Alternative considered**: Make substrate.yaml optional in Phase 4, then enforce in later phase.
**Rejected because**: Breaking changes authorized - no need for gradual migration, cleaner to enforce immediately.

---

## Timeline Impact

**Before reordering**:
- Week 3: Implement Phase 3 (Environment Integration) → **BREAKS ALL CONFIGS**
- Week 6: Implement Phase 6 (Config Migration) → **Too late, already broken**

**After reordering**:
- Week 3: Implement Phase 3 (Config Migration) → **Creates substrate.yaml files**
- Week 4: Implement Phase 4 (Environment Integration) → **Works perfectly, files exist**

**Result**: Prevents 3 weeks of broken config packs during implementation.

---

## Status

✅ **Phase renumbering complete**

All plan files correctly numbered and in proper execution order. Ready for implementation.

**Next steps**:
1. Update cross-references in plan files (dependency sections)
2. Begin Phase 1-2 implementation (Substrate Foundation, 10h)
3. Proceed sequentially through Phases 3-8 (63.5h)

---

**Date**: 2025-11-05
**Author**: Claude (Sonnet 4.5) via RESEARCH-PLAN-REVIEW-LOOP methodology
**Status**: Complete
