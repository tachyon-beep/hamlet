"""Placeholder config validation script for TASK-003 Cycle 8.

Once the DTOs land, this script will load each config pack via HamletConfig and
report validation errors. For now, it simply enumerates config directories.
"""

from pathlib import Path

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def list_config_packs(config_root: Path = DEFAULT_CONFIG_ROOT) -> list[Path]:
    return [p for p in config_root.iterdir() if p.is_dir() and not p.name.startswith("templates")]


def main() -> None:
    packs = list_config_packs()
    print("Config packs detected:")
    for pack in packs:
        print(f" - {pack.name}")
    print("\nTODO: Wire this up to HamletConfig.load once DTOs exist.")


if __name__ == "__main__":
    main()
