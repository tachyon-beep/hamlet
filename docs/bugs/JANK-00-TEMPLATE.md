Title: [Short human-readable title]

Severity: [low | medium | high]
Status: open

Ticket Type: JANK
Subsystem: [compiler | env | training | exploration | etc.]
Affected Version/Branch: main

Affected Files:
- `path/to/file.py:line_range`
- `path/to/other_file.py:line_range`

Description:
- Briefly explain what’s “janky” about the current behavior.
- Emphasize why it’s risky or fragile even if it’s not a concrete bug yet.

Reproduction:
1) Steps to exercise the behavior (if applicable).
2) Focus on how someone would encounter it in real usage or tests.

Expected Behavior:
- What a “clean” or less fragile implementation would look like.

Actual Behavior:
- What currently happens, including any silent fallbacks, broad exception handling, or dual-code-paths.

Root Cause:
- Structural issue, layering mismatch, or legacy shortcut that led to the jank.
- Call out any duplicated logic, schema divergence, or hidden flags.

Risk:
- How this could mask real bugs, misconfigurations, or performance regressions.
- Who is likely to be confused (students, operators, future maintainers).

Proposed Directions:
- Short bullet list of possible refactors or design directions.
- Distinguish between short-term guardrails vs long-term cleanup.

Tests:
- What tests or checks we’d want once this is addressed.

Owner: [team/area]
Links:
- Any relevant code, docs, or existing BUG/ENH tickets.
