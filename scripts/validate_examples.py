#!/usr/bin/env python3
"""Validate every bundled demo and confirm complete 3x3 task coverage."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import load_and_validate  # noqa: E402


EXPECTED_MODES = {
    "t2v",
    "i2v",
    "v2v",
    "traj2v_t2v",
    "traj2v_i2v",
    "traj2v_v2v",
    "r2v_t2v",
    "r2v_i2v",
    "r2v_v2v",
}


def main() -> None:
    modes = set()
    configs = sorted((ROOT / "examples").glob("*.json"))
    for config_path in configs:
        config = load_and_validate(config_path)
        modes.add(config["mode"])
        print(f"OK  {config_path.name:<48} {config['mode']}")

    missing = EXPECTED_MODES - modes
    if missing:
        raise SystemExit(f"Missing demo modes: {', '.join(sorted(missing))}")
    print(f"Validated {len(configs)} configs; all 9 modes are covered.")


if __name__ == "__main__":
    main()
