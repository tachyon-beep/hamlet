# Task: Analyze Universe Compiler Subsystem

## Context
- **Workspace:** `archeologist_archive/`
- **Read:** `01-discovery-findings.md` (holistic assessment)
- **Write to:** `02-subsystem-catalog.md` (append your section)

## Scope
Analyze the **Universe Compiler** subsystem located in `src/townlet/universe/`

## Expected Output
Follow the EXACT contract specified in the `analyzing-unknown-codebases` skill.

## Validation Criteria
- [ ] All contract sections complete
- [ ] Confidence level marked with reasoning
- [ ] Dependencies bidirectional
- [ ] Output appended to `02-subsystem-catalog.md`
- [ ] NO extra sections added

## Analysis Focus
- Document the 7-stage compilation pipeline (Stage 0-6)
- Identify compiler phases (parse, resolve, validate, optimize, emit)
- Map DTO outputs (UniverseMetadata, ObservationSpec, etc.)
- Document caching strategy (MessagePack artifacts)
- Analyze error handling (CompilationError, CompilationErrorCollector)
