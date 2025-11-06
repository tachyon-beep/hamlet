# Repository Guidelines

## Project Structure & Module Organization

Core reinforcement-learning code lives in `src/townlet/` (`environment/`, `agent/`, `training/`). Config presets are under `configs/`, and runnable entry points (for example `scripts/run_demo.py`) sit in `scripts/`. Tests mirror the runtime layout: units in `tests/test_townlet/unit/`, integration and e2e flows beside them. The Vue observer UI is isolated in `frontend/`, and experiment artifacts belong in `runs/` but out of version control.

## Build, Test, and Development Commands

- `uv sync` — create or update the local virtual environment with dev extras.
- `uv run pytest` — execute the default suite (skipping `slow`) with coverage enabled.
- `uv run pytest -m "slow"` — opt into long-running or GPU-tagged scenarios before proposing major changes.
- `uv run ruff check` / `uv run black --check .` / `uv run mypy src` — mirror the CI lint pipeline.
- `npm install && npm run dev` from `frontend/` — serve the visualization dashboard when validating inference pipelines.

## Coding Style & Naming Conventions

Python code follows Black formatting with a 140-character line limit and 4-space indentation. Keep imports sorted (Ruff enforces this), prefer type-hinted functions, and avoid hidden defaults so the no-defaults guard stays green. Use `snake_case` for functions and modules, `PascalCase` for classes, and suffix async utilities with `_async`. Frontend code should keep the existing Vue single-file component structure and ESLint defaults.

## Testing Guidelines

Pytest auto-discovers files named `test_*.py`, classes `Test*`, and functions `test_*`. Maintain ≥70% coverage by extending `tests/test_townlet/` alongside each feature, and place heavyweight scenarios under `slow` or `gpu` markers so they stay opt-in locally. When reproducing bugs, add regression cases under `tests/test_townlet/regressions/` with fixtures in `fixtures/`. Use integration flows when the curriculum hand-off is involved.

## Commit & Pull Request Guidelines

Commit history uses Conventional Commit semantics (`feat(env): ...`, `fix(actions): ...`); continue that pattern with imperative subject lines under 72 characters. Every PR should outline functional impact, list test commands (`uv run pytest`, etc.), and link related issues. Provide screenshots when the UI changes, note migrations in `configs/`, and document behavioral shifts in `docs/` or the changelog as needed.

## Security & Configuration Tips

Load secrets via environment variables or `.env` files; never commit credentialized configs or generated databases (`test.db`) and large artifacts in `runs/`. Reuse the provided YAML configs instead of hard-coding paths, and validate new endpoints against `SECURITY.md` guidelines before exposing them in the API layer.
