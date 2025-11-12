# Task: Analyze Recording Subsystem

## Context
- **Workspace:** `archeologist_archive/`
- **Read:** `01-discovery-findings.md` (holistic assessment)
- **Write to:** `02-subsystem-catalog.md` (append your section)

## Scope
Analyze the **Recording** subsystem located in `src/townlet/recording/`

## Expected Output
Follow the EXACT contract specified in the `analyzing-unknown-codebases` skill.

## Validation Criteria
- [ ] All contract sections complete
- [ ] Confidence level marked with reasoning
- [ ] Dependencies bidirectional
- [ ] Output appended to `02-subsystem-catalog.md`
- [ ] NO extra sections added

## Analysis Focus
- Document episode recorder (state capture, serialization)
- Identify replay mechanism (playback from recorded episodes)
- Map video export pipeline (frame rendering, ffmpeg integration)
- Document compression strategy (LZ4, MessagePack)
- Analyze recording criteria (when to record episodes)
