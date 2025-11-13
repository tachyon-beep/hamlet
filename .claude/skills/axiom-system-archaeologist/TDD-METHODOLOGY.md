# TDD Methodology for Skill Development

The document outlines a Test-Driven Development approach adapted for AI agent skill documentation. The core methodology follows four phases:

**RED Phase:** Agents operate without the skill across 5-7 pressure scenarios (time constraints, authority pressure, complexity overwhelm). Developers document failures and capture verbatim rationalizations agents use to justify shortcuts.

**GREEN Phase:** Skills are written to address only observed failures from baseline testing. As the document states: "Every 'must' or 'mandatory' maps to observed baseline failure" and developers should use the agents' own language when countering rationalizations.

**GREEN Testing:** The skill is deployed against 2-3 critical scenarios from the RED phase. Success requires that previously skipped process steps are now followed and baseline rationalizations no longer appear.

**REFACTOR Phase:** Gaps identified during GREEN testing are addressed through iteration, particularly around new creative bypasses agents discover and ambiguous guidance requiring clarification.

The axiom-system-archaeologist skillpack required approximately five hours total development time using this methodology, producing a 280-line skill validated across multiple pressure conditions.

The document emphasizes that skills must be grounded in observed agent behavior rather than hypothetical concerns, with mandatory steps clearly marked using strong language and comprehensive rationalization tables documenting specific failure patterns discovered during baseline testing.
