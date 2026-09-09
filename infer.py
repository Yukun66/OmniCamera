#!/usr/bin/env python3
"""Run OmniCamera inference from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
REQUIRED_KEYS = {"camera_type", "content_type", "prompt"}
MODE_TABLE = {
    ("Text motion", "Text"): "t2v",
    ("Text motion", "Image"): "i2v",
    ("Text motion", "Video"): "v2v",
    ("Trajectory", "Text"): "traj2v_t2v",
    ("Trajectory", "Image"): "traj2v_i2v",
    ("Trajectory", "Video"): "traj2v_v2v",
    ("Reference camera video", "Text"): "r2v_t2v",
    ("Reference camera video", "Image"): "r2v_i2v",
    ("Reference camera video", "Video"): "r2v_v2v",
}
CAMERA_CONDITION_TYPES = {camera for camera, _ in MODE_TABLE}
CONTENT_CONDITION_TYPES = {content for _, content in MODE_TABLE}


def _resolve_path(value, config_path):
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.exists() or path.parts[0] in {"assets", "outputs"}:
        return root_candidate
    return config_path.parent / path


def load_and_validate(config_path):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")

    camera_type = config["camera_type"]
    content_type = config["content_type"]
    if camera_type not in CAMERA_CONDITION_TYPES:
        raise ValueError(f"Unsupported camera_type: {camera_type}")
    if content_type not in CONTENT_CONDITION_TYPES:
        raise ValueError(f"Unsupported content_type: {content_type}")
    camera_motion = config.get("camera_motion", "Dolly In")
    if camera_type == "Text motion" and not camera_motion:
        raise ValueError("Text motion mode requires camera_motion")

    for key in (
        "trajectory_json",
        "reference_camera_video",
        "content_image",
        "content_video",
    ):
        config[key] = _resolve_path(config.get(key), config_path)
        if config[key] is not None and not config[key].is_file():
            raise FileNotFoundError(f"{key}: {config[key]}")

    mode = MODE_TABLE[(camera_type, content_type)]
    if camera_type == "Trajectory" and config["trajectory_json"] is None:
        raise ValueError("Trajectory mode requires trajectory_json")
    if (
        camera_type == "Reference camera video"
        and config["reference_camera_video"] is None
    ):
        raise ValueError(
            "Reference camera video mode requires reference_camera_video"
        )
    if content_type == "Image" and config["content_image"] is None:
        raise ValueError("Image content mode requires content_image")
    if content_type == "Video" and config["content_video"] is None:
        raise ValueError("Video content mode requires content_video")

    config["output"] = _resolve_path(
        config.get("output", f"outputs/{mode}.mp4"), config_path
    )
    config["camera_motion"] = camera_motion
    config["mode"] = mode
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to a demo JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without loading the model",
    )
    args = parser.parse_args()
    config = load_and_validate(args.config)
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    from app import generate_video

    content_image = None
    if config["content_image"] is not None:
        content_image = Image.open(config["content_image"]).convert("RGB")
    generated_path, summary = generate_video(
        config["camera_type"],
        config["content_type"],
        config["camera_motion"],
        str(config["trajectory_json"]) if config["trajectory_json"] else None,
        int(config.get("trajectory_camera_id", 1)),
        str(config["reference_camera_video"])
        if config["reference_camera_video"]
        else None,
        config["prompt"],
        content_image,
        str(config["content_video"]) if config["content_video"] else None,
        int(config.get("seed", 0)),
        int(config.get("steps", 50)),
        int(config.get("num_frames", 41)),
    )
    config["output"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_path, config["output"])
    config["output"].with_suffix(".txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(f"Saved: {config['output']}")


if __name__ == "__main__":
    main()
