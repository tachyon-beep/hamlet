"""Helper to create instant-mode affordances for testing immediate effects."""

from pathlib import Path

import yaml


def convert_to_instant_mode(affordances_yaml: Path) -> None:
    """Convert affordances.yaml to use instant mode for all affordances.

    Args:
        affordances_yaml: Path to affordances.yaml file to modify

    Modifies the file in-place:
    - Changes all interaction_type to 'instant'
    - Removes duration_ticks, capabilities, costs_per_tick
    - Scales per_tick effects by duration_ticks (to get total instant effect)
    - Moves all effects to effect_pipeline.on_start for instant execution
    """
    with open(affordances_yaml) as f:
        data = yaml.safe_load(f)

    for aff in data.get("affordances", []):
        # Force instant mode
        aff["interaction_type"] = "instant"

        # Save duration_ticks before removing (needed to scale per_tick effects)
        duration_ticks = aff.get("duration_ticks", 1)

        # Remove multi-tick fields
        aff.pop("duration_ticks", None)
        aff.pop("capabilities", None)
        aff.pop("costs_per_tick", None)

        # Convert effect_pipeline for instant mode
        # Keep effect_pipeline but move everything to on_start for instant execution
        if "effect_pipeline" in aff:
            pipeline = aff["effect_pipeline"]

            # Combine all effects into on_start for instant mode
            # Merge effects for the same meter to avoid multiple entries
            effect_totals = {}
            if isinstance(pipeline, dict):
                # Collect all effects
                all_effects = []
                all_effects.extend(pipeline.get("on_start", []))
                all_effects.extend(pipeline.get("on_completion", []))

                # Scale per_tick effects by duration_ticks
                for effect in pipeline.get("per_tick", []):
                    scaled_effect = effect.copy()
                    if "amount" in scaled_effect:
                        scaled_effect["amount"] = scaled_effect["amount"] * duration_ticks
                    all_effects.append(scaled_effect)

                # Merge effects by meter
                for effect in all_effects:
                    meter = effect.get("meter")
                    if meter:
                        if meter not in effect_totals:
                            effect_totals[meter] = {"meter": meter, "amount": 0.0}
                        effect_totals[meter]["amount"] += effect.get("amount", 0.0)

                instant_effects = list(effect_totals.values())

            # Replace pipeline with instant-only effects
            aff["effect_pipeline"] = {
                "on_start": instant_effects,
                "per_tick": [],
                "on_completion": [],
                "on_early_exit": [],
                "on_failure": [],
            }
        else:
            # Ensure effect_pipeline exists even if empty
            aff["effect_pipeline"] = {
                "on_start": [],
                "per_tick": [],
                "on_completion": [],
                "on_early_exit": [],
                "on_failure": [],
            }

    # Write back
    with open(affordances_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
