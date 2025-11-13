# System Archaeologist - Codebase Architecture Analysis

## Complete Document Content

This document describes a systematic workflow for analyzing existing codebases to generate architecture documentation. The core methodology emphasizes mandatory process steps, quality validation gates, and pressure-resistant workflows.

### Key Mandatory Elements

**Workspace Creation:** All analysis begins with creating a dated workspace directory (`docs/arch-analysis-YYYY-MM-DD-HHMM/`) to organize artifacts and enable subagent coordination.

**Coordination Plan:** As stated in the document, "Undocumented work is unreviewable and non-reproducible." Every analysis requires a written coordination plan documenting scope, strategy, and execution decisions.

**Holistic Assessment First:** Before detailed analysis, systematically map directory structure, entry points, technology stack, and subsystems to prevent getting lost in implementation details.

**Subagent Orchestration:** Choose between sequential (for smaller projects with tight dependencies) or parallel analysis (for independent subsystems ≥5 total) based on explicit reasoning documented in the coordination log.

**Validation Gates:** After each major document, mandatory validation occurs—either through a separate validation subagent or systematic self-validation against documented contracts.

### Critical Pressure-Handling Principle

The document explicitly rejects time constraints as justification for skipping process steps. When facing deadlines, the appropriate response is to provide scoped alternatives with explicitly documented limitations rather than rush complete analysis.

### Anti-Patterns to Avoid

The document lists six specific anti-patterns including skipping workspace creation, omitting coordination logging, working solo despite scale opportunities, bypassing validation gates, and refusing work entirely under pressure.
