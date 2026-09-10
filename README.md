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

Every configuration is ordinary JSON, so prompts, input paths, seeds, frame
counts and inference steps can be changed directly. The implementation supports
41 or 81 output frames at 16 FPS and uses the trained 1248 × 704 resolution.

## Gradio demo

Launch the interactive interface to access all 3 × 3 modes:

```bash
python app.py
```

## Camera trajectory format

Trajectory inference consumes camera poses, not a rendered trajectory image.
The included JSON files contain `frame0`, `frame1`, ... entries and one or more
`camXX` 4 × 4 camera-to-world matrices per frame. OmniCamera follows its
training loader: it takes the first requested 41 or 81 source poses and samples
every fourth pose for the video latent sequence.

Two ready-to-run 41-frame trajectory examples are included:

- `assets/trajectories/truck_left.json`
- `assets/trajectories/forward_up_tilt_down.json`

## Example results

The following videos use the same text prompt, seed 0, 41 frames and 50 steps:

- [Motion-text Truck Left](assets/results/dog_text_motion_truck_left.mp4)
- [Trajectory Truck Left](assets/results/dog_trajectory_truck_left.mp4)
- [Reference-video Truck Left](assets/results/dog_reference_video_truck_left.mp4)

These small files are for qualitative inspection only. Additional results are
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
