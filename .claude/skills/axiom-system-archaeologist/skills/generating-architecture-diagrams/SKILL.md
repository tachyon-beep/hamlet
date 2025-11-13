# Architecture Diagram Generation Guide

This document provides instructions for creating C4 architecture diagrams from subsystem catalogs at three abstraction levels: Context (system boundary), Container (major subsystems), and Component (internal structure of selected subsystems).

## Key Principles

The core philosophy emphasizes readability over completeness: "Readable diagrams communicate architecture. Overwhelming diagrams obscure it." Rather than documenting every detail, diagrams should help readers understand system structure.

## C4 Levels Overview

**Context Diagram** shows the system as a single entity with external actors and systems. **Container Diagram** visualizes major subsystems and dependencies, using grouping strategies for complex systems exceeding eight subsystems. **Component Diagrams** provide internal architecture for two to three representative subsystems selected for architectural diversity and critical path importance.

## Abstraction Techniques

When managing complexity, the guide recommends three strategies: natural grouping by existing categories or layers, metadata enrichment (skill counts, line counts) without visual clutter, and strategic sampling of roughly twenty percent of components showing architectural variety.

## Required Documentation

Each diagram must include a title, code block, narrative description, and legend. The output file must conclude with an "Assumptions and Limitations" section explaining inferences, gaps, and confidence levels.

## Selection Criteria for Components

Choose subsystems demonstrating different architectural patterns, representing scale appropriately, occupying critical paths, and avoiding redundant examples. Always document the rationale for these selections.
