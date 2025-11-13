# Analyzing Unknown Codebases

This document provides a systematic methodology for analyzing unfamiliar code while maintaining strict adherence to output specifications.

## Core Purpose

The guide emphasizes analyzing code to create subsystem catalog entries that comply with exact formatting requirements. The central principle: "Your analysis quality doesn't matter if you violate the output contract."

## Key Requirements

**Output Format (Mandatory):**
The catalog entry must include these sections in exact order:
- Subsystem Name (H2 heading)
- Location
- Responsibility (single sentence)
- Key Components (bulleted list)
- Dependencies (Inbound/Outbound)
- Patterns Observed
- Concerns (or "None observed")
- Confidence level with reasoning
- Separator line

"Extra sections break downstream tools. The coordinator expects EXACT format for parsing and validation."

## Analysis Methodology

The five-step approach recommended:

1. Read task specification first
2. Follow layered exploration (metadata → structure → routers → sampling → quantitative)
3. Mark confidence explicitly (High/Medium/Low with justification)
4. Distinguish between complete, placeholder, and planned components
5. Write output ensuring contract compliance

## Critical Compliance Rules

**Prohibited:**
- Adding extra sections
- Changing section names or reordering
- Writing to separate files
- Skipping sections

**Required:**
- Copy template structure exactly
- Append to specified file
- Include all sections
- Support claims with evidence

The document stresses that the contract represents specification, not a minimum standard, and improvements violating it constitute failures regardless of intent.
