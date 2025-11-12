# Task: Analyze Agent Networks Subsystem

## Context
- **Workspace:** `archeologist_archive/`
- **Read:** `01-discovery-findings.md` (holistic assessment)
- **Write to:** `02-subsystem-catalog.md` (append your section)

## Scope
Analyze the **Agent Networks** subsystem located in `src/townlet/agent/`

## Expected Output
Follow the EXACT contract specified in the `analyzing-unknown-codebases` skill.

## Validation Criteria
- [ ] All contract sections complete
- [ ] Confidence level marked with reasoning
- [ ] Dependencies bidirectional
- [ ] Output appended to `02-subsystem-catalog.md`
- [ ] NO extra sections added

## Analysis Focus
- Document SimpleQNetwork architecture (MLP for full observability)
- Document RecurrentSpatialQNetwork architecture (LSTM for POMDP)
- Map network selection logic (when to use which architecture)
- Identify encoder components (vision, position, meter, affordance, temporal)
- Analyze parameter counts and architectural choices
