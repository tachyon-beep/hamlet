# Task: Analyze Demo System Subsystem

## Context
- **Workspace:** `archeologist_archive/`
- **Read:** `01-discovery-findings.md` (holistic assessment)
- **Write to:** `02-subsystem-catalog.md` (append your section)

## Scope
Analyze the **Demo System** subsystem located in `src/townlet/demo/`

## Expected Output
Follow the EXACT contract specified in the `analyzing-unknown-codebases` skill.

## Validation Criteria
- [ ] All contract sections complete
- [ ] Confidence level marked with reasoning
- [ ] Dependencies bidirectional
- [ ] Output appended to `02-subsystem-catalog.md`
- [ ] NO extra sections added

## Analysis Focus
- Document DemoRunner context manager (training orchestration)
- Identify live inference server (WebSocket protocol)
- Map unified server implementation
- Document database integration (SQLite for metrics)
- Analyze checkpoint loading and inference loops
