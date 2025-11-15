# Archive: Design Iteration Documents

This folder contains historical documents from the ENH-28 design process.

## Contents

### Design Iterations

- **target-config-design.md** (v1) - Initial four-layer architecture
  - Date: 2025-11-15
  - Status: Superseded by v2
  - Issues: Made grid_encoding/local_window mutually exclusive, perception couldn't vary across curriculum, temporal mechanics all-or-nothing

- **target-config-design-v2.1-patch.md** - Code review round 2 fixes
  - Date: 2025-11-15
  - Status: Merged into v2.1 final design
  - Addressed: Vision range normalization, meter range_type, substrate params, cascade validation, day_length enforcement, observation_encoding clarification

- **design-v2-changes-summary.md** - v1 → v2 transition document
  - Date: 2025-11-15
  - Documents: Support vs Active pattern introduction, critical issues from v1 code review, fixes applied

### Working Documents (Brainstorming Artifacts)

- **config-semantic-categories.md** - First principles categorization
  - Date: 2025-11-15
  - Purpose: Organized all config settings by WHAT they control semantically
  - Led to: WHAT vs HOW split (vocabulary breaks checkpoints, parameters don't)

- **config-settings-audit.md** - Breaking vs non-breaking analysis
  - Date: 2025-11-15
  - Purpose: Master list of ~80 config settings categorized by checkpoint portability
  - Led to: Clarity on which settings are experiment-level vs curriculum-level

## Design Evolution Timeline

1. **BUG-43 Implementation** (2025-11-15)
   - Implemented curriculum masking for transfer learning
   - All curriculum levels now have identical obs_dim (L1=L2=121)

2. **Initial Brainstorming** (2025-11-15)
   - Created semantic categories and settings audit
   - Discovered WHAT vs HOW split principle

3. **Design v1** (2025-11-15)
   - Created four-layer architecture
   - Code review identified 7 critical/major issues

4. **Design v2** (2025-11-15)
   - Introduced Support vs Active pattern
   - Fixed all critical issues from v1
   - Code review identified 7 more issues

5. **Design v2.1 Patch** (2025-11-15)
   - Normalized vision_range, clarified observation_encoding
   - All critical/major issues resolved
   - Code review approved for implementation (100/100 confidence)

6. **Design v2.1 Final** (2025-11-15)
   - Merged v2 + v2.1 patch + code review clarifications
   - Created complete reference config (reference-config-v2.1-complete.yaml)
   - Ready for implementation

## Key Insights Captured

1. **Support vs Active Pattern**: Experiment declares which fields CAN exist, curriculum declares which ARE active vs masked
2. **WHAT vs HOW Split**: Vocabulary (what exists) breaks checkpoints, parameters (how it behaves) don't
3. **No-Defaults Principle**: All settings mandatory, use `null` to explicitly disable
4. **Normalized Vision Range**: 0.0-1.0 float eliminates validation complexity
5. **Observation Encoding Modes**: All produce identical obs_dim (value ranges differ, not dimensions)

## Active Documents

See parent directory for current active documents:
- `README.md` - Bundle overview
- `BUG-43-partial-observability-global-view-masking-and-obs-dim.md` - Closed issue
- `ENH-28-experiment-level-configuration-hierarchy.md` - Enhancement tracker
- `target-config-design-v2.md` - Current design specification (v2 + v2.1 integrated)
- `reference-config-v2.1-complete.yaml` - Complete implementation reference
