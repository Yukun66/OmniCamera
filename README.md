# OmniCamera

Official inference code for **OmniCamera: A Unified Framework for Multi-task
Video Generation with Arbitrary Camera Control**.

[Paper](https://arxiv.org/abs/2604.06010) ·
[Project page](https://yukun66.github.io/omnicamera-webdemo/) ·
[Model weights](https://huggingface.co/wykup316/OmniCamera)

OmniCamera controls camera motion and scene content independently. A single
full-finetuned checkpoint supports all nine combinations of three camera
conditions and three content conditions:

| Camera condition | Text content | Image content | Video content |
| --- | --- | --- | --- |
| Motion text | `t2v` | `i2v` | `v2v` |
| Camera trajectory | `traj2v_t2v` | `traj2v_i2v` | `traj2v_v2v` |
| Reference camera video | `r2v_t2v` | `r2v_i2v` | `r2v_v2v` |

> This repository currently releases the inference implementation and demo
> configurations. Training code will be added after its data- and
> infrastructure-specific parts are cleaned. The checkpoint is a partial
> full-finetuning checkpoint, not a LoRA.

## Installation

Python 3.10 or 3.11 and a CUDA GPU are recommended. Install a PyTorch build
matching your CUDA environment, then install the remaining dependencies:

```bash
git clone https://github.com/Yukun66/OmniCamera.git
cd OmniCamera
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The base model is downloaded from `Wan-AI/Wan2.2-TI2V-5B`, and the public
OmniCamera checkpoint is downloaded automatically from Hugging Face. No access
token is required for either repository.

The model repository and checkpoint filename may be overridden without editing
the code:

```bash
export OMNICAMERA_MODEL_REPO=wykup316/OmniCamera
export OMNICAMERA_CHECKPOINT=joint_params_step61000.pth
```

## Quick start

Validate a demo configuration without loading the model:

```bash
python infer.py --config examples/text_motion_t2v.json --dry-run
```

Run inference:

```bash
python infer.py --config examples/text_motion_t2v.json
```

The generated video and a text summary are written to `outputs/`. The included
configurations demonstrate:

| Configuration | Camera input | Content input |
| --- | --- | --- |
| `examples/text_motion_t2v.json` | motion text | text |
| `examples/trajectory_t2v.json` | 3D trajectory JSON | text |
| `examples/reference_video_t2v.json` | camera-motion video | text |
| `examples/trajectory_i2v.json` | 3D trajectory JSON | image |
| `examples/reference_video_v2v.json` | camera-motion video | video |
| `examples/text_motion_t2v_beach.json` | motion text (Truck Left) | beach text |
| `examples/trajectory_t2v_canyon_lake.json` | Forward-Up + Tilt Down trajectory | canyon-lake text |
| `examples/reference_video_t2v_ski_resort.json` | Truck Left camera video | ski-resort text |
| `examples/text_motion_v2v_michael_jackson.json` | motion text (Dolly In) | reference video |
| `examples/reference_video_v2v_michael_jackson.json` | Dolly Out camera video | reference video |
| `examples/text_motion_i2v_white_car.json` | motion text (Dolly In) | reference image |
| `examples/trajectory_v2v_michael_jackson.json` | Arc Right trajectory | reference video |
| `examples/reference_video_i2v_white_car.json` | Dolly Out camera video | reference image |
| `examples/reference_video_v2v_matrix.json` | Pan Right camera video | reference video |

Every configuration is ordinary JSON, so prompts, input paths, seeds, frame
counts and inference steps can be changed directly. The implementation supports
41 or 81 output frames at 16 FPS and uses the trained 1248 × 704 resolution.

## Optional local interface

The primary release is the command-line inference code and reproducible demo
configurations. An optional local Gradio interface can also access all 3 × 3
modes; it is not a hosted inference service:

```bash
python app.py
```

## Camera trajectory format

Trajectory inference consumes camera poses, not a rendered trajectory image.
The included JSON files contain `frame0`, `frame1`, ... entries and one or more
`camXX` 4 × 4 camera-to-world matrices per frame. OmniCamera follows its
training loader: it takes the first requested 41 or 81 source poses and samples
every fourth pose for the video latent sequence.

Three ready-to-run 41-frame trajectory examples are included:

- `assets/trajectories/truck_left.json`
- `assets/trajectories/forward_up_tilt_down.json`
- `assets/trajectories/arc_right.json`

## Reproducible demos and results

Each result below has a matching configuration and all required inputs in this
repository, so it can be viewed immediately or regenerated locally:

| Content | Camera condition | Configuration | Result |
| --- | --- | --- | --- |
| Dog running in a garden | motion text: Truck Left | [config](examples/text_motion_t2v.json) | [video](assets/results/dog_text_motion_truck_left.mp4) |
| Dog running in a garden | trajectory: Truck Left | [config](examples/trajectory_t2v.json) | [video](assets/results/dog_trajectory_truck_left.mp4) |
| Dog running in a garden | reference video: Truck Left | [config](examples/reference_video_t2v.json) | [video](assets/results/dog_reference_video_truck_left.mp4) |
| Empty beach and lifeguard tower | motion text: Truck Left | [config](examples/text_motion_t2v_beach.json) | [video](assets/results/beach_text_motion_truck_left.mp4) |
| Canyon lake | trajectory: Forward-Up + Tilt Down | [config](examples/trajectory_t2v_canyon_lake.json) | [video](assets/results/canyon_lake_trajectory_forward_up_tilt_down.mp4) |
| Empty ski resort | reference video: Truck Left | [config](examples/reference_video_t2v_ski_resort.json) | [video](assets/results/ski_resort_reference_video_truck_left.mp4) |
| Michael Jackson performance clip | motion text: Dolly In | [config](examples/text_motion_v2v_michael_jackson.json) | [video](assets/results/michael_jackson_text_motion_dolly_in.mp4) |
| Michael Jackson performance clip | reference video: Dolly Out | [config](examples/reference_video_v2v_michael_jackson.json) | [video](assets/results/michael_jackson_reference_video_dolly_out.mp4) |
| White car in a canyon | motion text: Dolly In | [config](examples/text_motion_i2v_white_car.json) | [video](assets/results/white_car_text_motion_dolly_in.mp4) |
| Michael Jackson performance clip | trajectory: Arc Right | [config](examples/trajectory_v2v_michael_jackson.json) | [video](assets/results/michael_jackson_trajectory_arc_right.mp4) |
| White car in a canyon | reference video: Dolly Out | [config](examples/reference_video_i2v_white_car.json) | [video](assets/results/white_car_reference_video_dolly_out.mp4) |
| Matrix-inspired action clip | reference video: Pan Right | [config](examples/reference_video_v2v_matrix.json) | [video](assets/results/matrix_reference_video_pan_right.mp4) |

These small files are for qualitative inspection and reproducibility. The
Matrix and Michael Jackson source clips are third-party research examples and
are not covered by the repository's Apache-2.0 license; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Additional results are
available on the project page.

## Acknowledgements

This implementation contains a modified subset of
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), distributed
under Apache-2.0. The example camera trajectories are adapted from
[ReCamMaster](https://github.com/KwaiVGI/ReCamMaster), distributed under MIT.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `licenses/`.

## Citation

```bibtex
@article{wang2026omnicamera,
  title   = {OmniCamera: A Unified Framework for Multi-task Video Generation with Arbitrary Camera Control},
  author  = {Wang, Yukun and Li, Ruihuang and Tao, Jiale and Yang, Shiyuan and Chen, Liyi and Yang, Zhantao and Handz and Guo, Yulan and Shao, Shuai and Lu, Qinglin},
  journal = {arXiv preprint arXiv:2604.06010},
  year    = {2026}
}
```

## License

The OmniCamera code is released under the Apache License 2.0. Third-party
components remain subject to their respective licenses.
