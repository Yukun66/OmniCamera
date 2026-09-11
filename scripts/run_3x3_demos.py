#!/usr/bin/env python3
"""Validate or generate one representative example for every 3x3 task."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "text_motion_t2v.json",
    "text_motion_i2v_white_car.json",
    "text_motion_v2v_michael_jackson.json",
    "trajectory_t2v_canyon_lake.json",
    "trajectory_i2v_white_car.json",
    "trajectory_v2v_michael_jackson.json",
    "reference_video_t2v_ski_resort.json",
    "reference_video_i2v_white_car.json",
    "reference_video_v2v_matrix.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run full GPU inference. Without this flag, only validate configs.",
    )
    args = parser.parse_args()

    for name in CONFIGS:
        command = [
            sys.executable,
            str(ROOT / "infer.py"),
            "--config",
            str(ROOT / "examples" / name),
        ]
        if not args.generate:
            command.append("--dry-run")
        print(f"\n>>> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
