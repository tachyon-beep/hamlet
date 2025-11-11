# HAMLET/Townlet Architecture Diagrams

This directory contains comprehensive architectural documentation for the HAMLET/Townlet system, generated through autonomous codebase analysis.

## 📁 Files

### Main Documentation
- **`ARCHITECTURE_REPORT.md`** - Complete architectural analysis with:
  - Executive summary
  - 6 major subsystem definitions with responsibilities
  - 3 embedded Mermaid diagrams
  - Data flow descriptions
  - Design patterns and principles
  - External integrations summary
  - Development recommendations

### Mermaid Diagram Files

1. **`c1_system_context.mmd`** - System Context Diagram (C1 Level)
   - Shows HAMLET/Townlet as a single system
   - External actors (researchers, operators)
   - External integrations (filesystem, database, WebSocket, TensorBoard)
   - Frontend visualization connections

2. **`c2_component_diagram.mmd`** - Component Diagram (Internal Architecture)
   - Zooms into `src/townlet/` subsystems
   - Cold Path: Configuration → Compilation
   - Hot Path: Training execution
   - Persistence & Visualization layers
   - Inter-subsystem relationships

3. **`c3_dependency_graph.mmd`** - Module Dependency Graph
   - Critical module-level dependencies
   - Entry points (CLI commands)
   - Configuration layer
   - Compilation pipeline
   - Runtime execution flow
   - Persistence & telemetry

## 🖼️ Viewing Diagrams

### Option 1: VS Code (Recommended)
Install the [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) extension:

```bash
code --install-extension bierner.markdown-mermaid
```

Then open `ARCHITECTURE_REPORT.md` and click the preview button.

### Option 2: Mermaid CLI
```bash
npm install -g @mermaid-js/mermaid-cli

# Generate PNG
mmdc -i c1_system_context.mmd -o c1_system_context.png

# Generate SVG
mmdc -i c1_system_context.mmd -o c1_system_context.svg -b transparent
```

### Option 3: Mermaid Live Editor
Copy the contents of any `.mmd` file and paste into https://mermaid.live

### Option 4: GitHub Rendering
GitHub automatically renders Mermaid diagrams in markdown files. View `ARCHITECTURE_REPORT.md` directly on GitHub.

## 📊 Diagram Summary

| Diagram | Purpose | Best For |
|---------|---------|----------|
| **C1 System Context** | External view of the system | Understanding system boundaries, external dependencies, user interactions |
| **C2 Component Diagram** | Internal subsystem architecture | Understanding code organization, subsystem responsibilities, data flow |
| **C3 Dependency Graph** | Module-level dependencies | Understanding compilation flow, import relationships, critical paths |

## 🔍 Quick Navigation

**Want to understand...**
- **Overall system purpose?** → Read Executive Summary in `ARCHITECTURE_REPORT.md`
- **What are the major subsystems?** → See "Identified Subsystems" section
- **How does training work?** → See "Data Flow: End-to-End Training Pipeline"
- **External integrations?** → See "External Integrations" table
- **Code metrics?** → See "Key Metrics & Statistics"
- **Design principles?** → See "Architectural Patterns & Design Principles"

## 🛠️ Analysis Methodology

These diagrams were generated using:
1. **Autonomous subagent exploration** with three specialized agents:
   - Directory structure analyzer
   - Import pattern analyzer
   - External integration analyzer
2. **Complete src/ directory scan** (98 Python files, ~23K LOC)
3. **Module dependency mapping** via import statement analysis
4. **System boundary identification** via external library usage

All findings are derived from actual codebase inspection, not documentation inference.

## 📝 Maintenance

To regenerate these diagrams after significant architectural changes:

```bash
# Run the architecture analysis prompt again
# (See original prompt in ARCHITECTURE_REPORT.md)
```

**Last Generated**: 2025-11-12
**Codebase Version**: Commit `f9b5752` (branch: 004a-compiler-implementation)

## 🔗 Related Documentation

- `/docs/UNIVERSE-COMPILER.md` - Universe Compiler detailed design
- `/docs/architecture/COMPILER_ARCHITECTURE.md` - Compilation pipeline stages
- `/docs/config-schemas/` - Configuration file schemas
- `/docs/vfs-integration-guide.md` - Variable & Feature System guide
- `/CLAUDE.md` - Development commands and quick reference

---

**Note**: These diagrams use Mermaid.js syntax and are optimized for readability in markdown viewers that support Mermaid rendering.
