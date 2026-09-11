# Demo configurations

Each JSON file is a complete local inference request. Paths are resolved from
the repository root, and generated videos are written to `outputs/`.

## Minimal 3x3 set

| Camera condition | Content condition | Configuration |
| --- | --- | --- |
| Motion text | Text | `text_motion_t2v.json` |
| Motion text | Image | `text_motion_i2v_white_car.json` |
| Motion text | Video | `text_motion_v2v_michael_jackson.json` |
| Camera trajectory | Text | `trajectory_t2v_canyon_lake.json` |
| Camera trajectory | Image | `trajectory_i2v_white_car.json` |
| Camera trajectory | Video | `trajectory_v2v_michael_jackson.json` |
| Reference camera video | Text | `reference_video_t2v_ski_resort.json` |
| Reference camera video | Image | `reference_video_i2v_white_car.json` |
| Reference camera video | Video | `reference_video_v2v_matrix.json` |

Run any example from the repository root:

```bash
python infer.py --config examples/trajectory_i2v_white_car.json
```

Use `--dry-run` to validate paths and parameters without loading the model.
Run `python scripts/validate_examples.py` to validate the complete collection
and confirm that all nine modes are represented.
