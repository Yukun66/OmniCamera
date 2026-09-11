# OmniCamera

### ECCV 2026

Official inference code for **OmniCamera: A Unified Framework for Multi-task
Video Generation with Arbitrary Camera Control**.

[Paper](https://arxiv.org/abs/2604.06010) ·
[Project Page](https://yukun66.github.io/omnicamera-webdemo/) ·
[Model Weights](https://huggingface.co/wykup316/OmniCamera)

> **Release note:** Due to constraints related to an ongoing company project,
> this public release provides an early-stage training checkpoint and its
> corresponding inference code. It is not the final model version, so its
> results may differ from those shown in the paper and project page.

OmniCamera provides independent control over camera motion and video content.
One model supports all nine combinations of three camera conditions and three
content conditions.

| Camera condition | Text | Image | Video |
| --- | --- | --- | --- |
| Motion text | ✓ | ✓ | ✓ |
| Camera trajectory | ✓ | ✓ | ✓ |
| Reference camera video | ✓ | ✓ | ✓ |

## Installation

Python 3.10 or 3.11 and a CUDA GPU are recommended.

```bash
git clone https://github.com/Yukun66/OmniCamera.git
cd OmniCamera
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The base model and public OmniCamera checkpoint are downloaded automatically
from Hugging Face. No access token is required.

## Inference

Each JSON file in [`examples/`](examples/) defines one complete inference task.
Choose a configuration and run it with `infer.py`:

```bash
# Motion text + text content
python infer.py --config examples/text_motion_t2v.json

# Camera trajectory + image content
python infer.py --config examples/trajectory_i2v_white_car.json

# Reference camera video + video content
python infer.py --config examples/reference_video_v2v_matrix.json
```

On the first run, the base model and OmniCamera checkpoint are downloaded
automatically. The generated video is then saved in `outputs/`.

To create a custom task, copy the closest example and edit:

- `camera_type` and `camera_motion` for camera control;
- `prompt`, `content_image`, or `content_video` for scene content;
- `trajectory_json` or `reference_camera_video` when required;
- `steps`, `num_frames`, `seed`, and `output` for generation settings.

Ready-to-use inputs, camera references, trajectories, and generated results are
included in [`assets/`](assets/). The provided configurations cover the full
3 × 3 combination of camera and content conditions.

Validate an example without loading the model:

```bash
python infer.py --config examples/text_motion_t2v.json --dry-run
```

See the [project page](https://yukun66.github.io/omnicamera-webdemo/) for more
video results.

## Citation

```bibtex
@article{wang2026omnicamera,
  title   = {OmniCamera: A Unified Framework for Multi-task Video Generation with Arbitrary Camera Control},
  author  = {Wang, Yukun and Li, Ruihuang and Tao, Jiale and Yang, Shiyuan and Chen, Liyi and Yang, Zhantao and Handz and Guo, Yulan and Shao, Shuai and Lu, Qinglin},
  journal = {arXiv preprint arXiv:2604.06010},
  year    = {2026}
}
```

## Acknowledgements

This repository includes modified components from
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and example
camera trajectories adapted from
[ReCamMaster](https://github.com/KwaiVGI/ReCamMaster). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

## License

The OmniCamera code is released under the [Apache License 2.0](LICENSE).
