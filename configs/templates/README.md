# Config Templates

This directory hosts reference templates for the DTO-backed configuration system. Each template mirrors the fields enforced by the corresponding Pydantic model.

## Available Templates

- `training.yaml` – Training hyperparameters (TrainingConfig)
- `environment.yaml` – Environment setup (EnvironmentConfig)
- `population.yaml` – Population parameters (PopulationConfig)
- `curriculum.yaml` – Curriculum thresholds (CurriculumConfig)
- `bars.yaml` – Meter definitions (BarConfig)
- `cascades.yaml` / `cascades_strength*.yaml` – Meter interactions (CascadeConfig)
- `affordances.yaml` – Basic affordance definitions (AffordanceConfig)
- `substrate*.yaml` – Spatial substrates (SubstrateConfig, already implemented via TASK-002A)
- `variables_reference.yaml` – VFS observation vocabulary (existing VFS schema)
- `action_labels_*.yaml` – Optional label presets

> **Note:** Substrate and action schemas already exist in `townlet.substrate.config` and `townlet.environment.action_config`. TASK-003 focuses on the remaining DTOs; reuse these templates as you migrate config packs.

## Usage

1. Copy the relevant template into your config pack (e.g., `cp configs/templates/training.yaml configs/L0_0_minimal/training.yaml`).
2. Fill in **all** required fields—DTOs enforce the "no defaults" principle.
3. Run the config validation script (to be delivered in TASK-003 Cycle 8) to confirm the pack passes schema checks.
4. Keep templates in sync with DTO changes; each DTO cycle should update its template annotated with required fields.
